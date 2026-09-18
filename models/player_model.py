import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, Input

from config import (
    MAX_SEQUENCE_LENGTH, ROSTER_SIZE, NORM_STATS_PATH,
    SEED, TEST_FRAC, HOLDOUT_FRAC, HOLDOUT_MANIFEST_NAME,
    MODEL_DIM, NUM_LAYERS, NUM_HEADS, FF_DIM, ROSTER_SAB_LAYERS,
)
from data_loading import load_all_cleaned, load_training_corpus, resolve_partition
from models.norm_stats_io import load_norm_stats, save_norm_stats
from encoder.encoder import Encoder
from models.artifacts import ModelArtifacts, DEFAULT_ARTIFACTS_ROOT, warm_start_weights
from models.backbone import build_backbone
from models.event_time_model import (
    _norm_stats_path,
    OnCourtCandidateMask,
    EMBED_DIMS,
    ROSTER_DIM,
    CATEGORICAL_FIELDS,
    ROSTER_COLS,
    target_on_court,
)
from models.roster_set_encoder import (
    RosterEncoderParams,
    SequenceRosterEncoder,
    build_sequence_roster_encoder,
    encode_both_rosters,
)
from models.season_features import (
    SEASON_INPUT_KEYS,
    merge_season_features,
    append_season_batches,
    make_season_inputs,
    season_team_projections,
    attach_recency_weights,
    apply_recency,
)
from models.game_state_features import (
    GAME_STATE_INPUT_KEYS,
    merge_game_state_features,
    append_game_state_batches,
    make_game_state_inputs,
    game_state_projections,
)
from models.rotation_features import (
    NUM_ROSTER_SCALARS,
    ROSTER_STATE_KEYS,
    merge_rotation_features,
    append_rotation_batches,
    make_rotation_inputs,
    side_scalars,
)
from models.train_steps import build_trainer
from models.regime import (
    GAME_INDEX_KEY,
    REGIME_KEY,
    append_regime_batches,
    make_regime_input,
    regime_projection,
    regime_std,
)
from models.prior_features import (
    PRIOR_INPUT_KEYS,
    append_prior_batches,
    make_prior_inputs,
    merge_prior_features,
    prior_team_projections,
)
from reporting import ReportCollector, RunConfig
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT


class PlayerModel:
    """
    Player head: predict *who* performs an event, given the prior plays plus that
    event's already-decided event-type and time.

    A near 1:1 mirror of EventTimeModel (same transformer backbone, embeddings,
    roster set-encoder, reporting, artifact contract). Two differences:

      * Two extra inputs per timestep — ``next_event`` and ``next_delta_time`` — the
        next-step event id and standardized Δt for the event being placed. These are
        exactly the values EventTimeModel emits as its ``event_target`` /
        ``time_target``, so the Player model conditions on the Event/Time model's
        output.
      * One output head — ``player_output`` (softmax over the player vocab) — instead
        of the event + time heads.

    Indexing mirrors EventTimeModel's next-step shift: at position ``i`` the model
    sees prior rows ``0..i`` plus ``next_event[i] = event[i+1]`` /
    ``next_delta_time[i] = delta_time[i+1]`` and predicts ``player_target[i] =
    player[i+1]``. EventTimeModel.preprocess/_build_split is the source of truth for
    the shared encoding + Δt normalization.
    """

    # Stable key used for the on-disk artifact layout and the model registry.
    KEY = "player"

    def __init__(self, encoder: Encoder,
                 sequence_length=MAX_SEQUENCE_LENGTH,
                 model_dim=MODEL_DIM,
                 path="./data",
                 processed_dir="./data/processed"):

        self.sequence_length = sequence_length
        self.model_dim = model_dim
        self.encoder = encoder

        self.data_dir = Path(path)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir.resolve()}")
        self.processed_dir = Path(processed_dir)

        self.roster_encoder = None
        self.norm_stats = None

    # =====================
    # --- Data Loading  ---
    # =====================
    #
    # Identical loader to EventTimeModel: cleaned season files (game_id + roster cols),
    # game_id kept globally unique across files.

    def _load_all(self) -> pd.DataFrame:
        """Cleaned season files concatenated with globally-unique game_id (shared loader)."""
        return load_training_corpus(self.data_dir)

    # =====================
    # --- Preprocessing ---
    # =====================

    def preprocess(self, rebuild_vocabs=False, test_frac=TEST_FRAC,
                   holdout_frac=HOLDOUT_FRAC, seed=SEED,
                   game_partition=None, refit_norm_stats=True):
        """
        Build the model-ready tensor arrays and persist them + the time-norm stats.

        Same loader / encoding / Δt normalization / train-val-holdout game split (same
        ``seed`` + ``test_frac`` + ``holdout_frac`` as EventTimeModel, so the splits and
        norm_stats line up exactly). ``rebuild_vocabs`` defaults to False because the
        shared vocab language is already built + frozen by the Event/Time model; we just
        load + freeze it.

        Adds, vs EventTimeModel: the conditioning inputs ``next_event`` /
        ``next_delta_time`` and the ``player_target`` (all the next-step shift).

        ``game_partition`` / ``refit_norm_stats`` (curriculum use): see
        ``EventTimeModel.preprocess`` — an explicit chronological split and a switch to reuse
        the warmup-fit normalization stats instead of recomputing them from the slice.
        """
        df = self._load_all()

        # Shared vocab language: load (or rebuild) then FREEZE so token ids are stable.
        if rebuild_vocabs:
            # The alias map is written by the subset extract, which runs AFTER this object
            # was constructed -- so re-read it before any name is registered, or the floor
            # silently does nothing. See Encoder.prepare_for_rebuild.
            self.encoder.prepare_for_rebuild()
            for col, src in ROSTER_COLS.items():
                df[src].apply(self.encoder.encode_roster)
            for field in CATEGORICAL_FIELDS:
                df[field].apply(getattr(self.encoder, f"encode_{field}"))
            self.encoder.save_all()
        else:
            self.encoder.load_all()
        self.encoder.freeze_all()

        # Encode categoricals to ints + rosters to fixed-5 arrays (mirrors Event/Time).
        enc = {f: df[f].apply(getattr(self.encoder, f"encode_{f}")).to_numpy() for f in CATEGORICAL_FIELDS}
        rosters = {
            name: np.stack(df[src].apply(self.encoder.encode_roster).to_numpy())
            for name, src in ROSTER_COLS.items()
        }
        time = df["time"].to_numpy(dtype=np.float64)
        game_id = df["game_id"].to_numpy()

        # Per-game delta_time (no cross-game leakage).
        delta = (
            df.groupby("game_id")["time"].diff().fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
        )

        # Split games train/test/holdout BEFORE computing normalization stats (shared
        # logic + same seed/fracs as EventTimeModel, so the partitions line up exactly).
        train_games, test_games, holdout_games = resolve_partition(
            game_partition, game_id, seed, test_frac, holdout_frac,
        )

        train_mask = np.array([g in train_games for g in game_id])

        # Normalization stats from TRAIN ONLY (or loaded warmup stats, for staged warm-start).
        if refit_norm_stats:
            max_time = float(time[train_mask].max()) or 1.0
            train_delta = delta[train_mask]
            delta_mean = float(train_delta.mean())
            delta_std = float(train_delta.std()) or 1.0
            self.norm_stats = {"max_time": max_time, "delta_mean": delta_mean, "delta_std": delta_std}
        else:
            self.norm_stats = load_norm_stats(self.processed_dir, self.KEY)
            max_time = float(self.norm_stats["max_time"]) or 1.0
            delta_mean = float(self.norm_stats["delta_mean"])
            delta_std = float(self.norm_stats["delta_std"]) or 1.0

        time_abs = (time / max_time).astype(np.float32)
        delta_norm = ((delta - delta_mean) / delta_std).astype(np.float32)

        cols = {
            **{f: enc[f] for f in CATEGORICAL_FIELDS},
            "home_roster": rosters["home_roster"],
            "away_roster": rosters["away_roster"],
            "time_abs": time_abs,
            "delta_time": delta_norm,
        }
        merge_season_features(
            df, cols, rosters, self.encoder.encode_player("PAD"), train_mask, self.norm_stats,
            refit=refit_norm_stats,
        )
        merge_game_state_features(df, cols)  # running score / period-clock / team fouls
        merge_rotation_features(df, cols)  # per-player stint / minutes / fouls
        # Season-to-date per-player and per-team rates, joined from the causal sidecar
        # (player_priors.py). Fixed-constant normalization, so no norm_stats keys.
        merge_prior_features(df, cols, rosters, str(self.data_dir))
        train = self._build_split(cols, game_id, train_games)
        test = self._build_split(cols, game_id, test_games)
        holdout = self._build_split(cols, game_id, holdout_games)
        attach_recency_weights(
            [(train, train_games), (test, test_games), (holdout, holdout_games)], df, game_id)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.processed_dir / "player_train.npz", **train)
        np.savez_compressed(self.processed_dir / "player_test.npz", **test)
        np.savez_compressed(self.processed_dir / "player_holdout.npz", **holdout)
        # Identical (deterministic) holdout manifest as EventTimeModel — write it so a
        # Player-only preprocess still leaves the contract in place.
        (self.processed_dir / HOLDOUT_MANIFEST_NAME).write_text(
            json.dumps(sorted(int(g) for g in holdout_games), indent=2), encoding="utf-8"
        )
        if refit_norm_stats:
            save_norm_stats(self.processed_dir, self.KEY, self.norm_stats)

        print(f"Preprocessed {len(train_games)} train / {len(test_games)} test / "
              f"{len(holdout_games)} holdout games -> {self.processed_dir} "
              f"(SEQ={self.sequence_length})")
        return train, test

    def _build_split(self, cols, game_id, games) -> dict:
        """
        Pad/truncate each game to sequence_length and stack into batched arrays.

        Same layout as EventTimeModel._build_split, plus the player-head specifics:
          - inputs ``next_event`` / ``next_delta_time``: next-step shift of event /
            delta_time (the conditioning signal — what event happens here and when).
          - target ``player_target``: next-step shift of player (who does it).
          - loss_mask: 1 for real rows with a valid next-step target, else 0.
        """
        SEQ = self.sequence_length
        PAD_PLAYER = self.encoder.encode_player("PAD")
        PAD_EVENT = self.encoder.encode_event("PAD")
        PAD_TYPE = self.encoder.encode_type("PAD")
        PAD_RESULT = self.encoder.encode_result("PAD")
        PAD_SEASON = self.encoder.encode_season("PAD")
        PAD_SECONDARY = self.encoder.encode_secondary_player("PAD")
        pad_scalar = {
            "event": PAD_EVENT, "player": PAD_PLAYER, "type": PAD_TYPE,
            "result": PAD_RESULT, "season": PAD_SEASON, "secondary_player": PAD_SECONDARY,
        }

        keys_1d = CATEGORICAL_FIELDS
        keys_roster = ["home_roster", "away_roster"]
        keys_cont = ["time_abs", "delta_time"]

        batches = {k: [] for k in (*keys_1d, *keys_roster, *keys_cont, *SEASON_INPUT_KEYS,
                                   *GAME_STATE_INPUT_KEYS, *ROSTER_STATE_KEYS, *PRIOR_INPUT_KEYS, REGIME_KEY, GAME_INDEX_KEY,
                                   "next_event", "next_delta_time",
                                   "player_target", "pad_mask", "loss_mask")}

        game_ids_sorted = [g for g in np.unique(game_id) if g in games]
        for g in game_ids_sorted:
            idx = np.where(game_id == g)[0]
            idx = idx[:SEQ]  # truncate overlong games
            n = len(idx)

            def pad1d(arr, pad_val, dtype):
                buf = np.full((SEQ,), pad_val, dtype=dtype)
                buf[:n] = arr[idx]
                return buf

            for k in keys_1d:
                batches[k].append(pad1d(cols[k], pad_scalar[k], np.int32))

            roster_bufs = {}
            for k in keys_roster:
                buf = np.full((SEQ, ROSTER_SIZE), PAD_PLAYER, dtype=np.int32)
                buf[:n] = cols[k][idx]
                roster_bufs[k] = buf
                batches[k].append(buf)

            for k in keys_cont:
                buf = np.zeros((SEQ, 1), dtype=np.float32)
                buf[:n, 0] = cols[k][idx]
                batches[k].append(buf)

            append_season_batches(batches, cols, idx, n, SEQ)
            append_game_state_batches(batches, cols, idx, n, SEQ)
            append_rotation_batches(batches, cols, idx, n, SEQ)
            append_prior_batches(batches, cols, idx, n, SEQ)
            append_regime_batches(batches, len(batches[REGIME_KEY]), SEQ)

            # Conditioning inputs + target: next-step shift within this game.
            next_event = np.full((SEQ,), PAD_EVENT, dtype=np.int32)
            next_delta = np.zeros((SEQ, 1), dtype=np.float32)
            player_t = np.full((SEQ,), PAD_PLAYER, dtype=np.int32)
            if n > 1:
                next_event[: n - 1] = cols["event"][idx][1:]
                next_delta[: n - 1, 0] = cols["delta_time"][idx][1:]
                player_t[: n - 1] = cols["player"][idx][1:]
            batches["next_event"].append(next_event)
            batches["next_delta_time"].append(next_delta)
            batches["player_target"].append(player_t)

            pad_mask = np.zeros((SEQ,), dtype=np.float32)
            pad_mask[:n] = 1.0
            loss_mask = np.zeros((SEQ,), dtype=np.float32)
            loss_mask[: max(n - 1, 0)] = 1.0  # last real row has no next target
            # Target-in-candidates guard (see OnCourtCandidateMask): the head's softmax is
            # masked to the row's on-court ten, so a row whose target actor is outside it
            # (the pre-"end" sentinel row, rare lineup glitches) would train on a -1e9
            # target logit — pure noise. Zero those rows out of the loss instead.
            loss_mask *= target_on_court(
                player_t, roster_bufs["home_roster"], roster_bufs["away_roster"])
            batches["pad_mask"].append(pad_mask)
            batches["loss_mask"].append(loss_mask)

        return {k: np.stack(v) if v else np.empty((0,)) for k, v in batches.items()}

    # =====================
    # --- Model / Train ---
    # =====================

    def build_roster_encoder(self, dropout: float = 0.1) -> SequenceRosterEncoder:
        num_players = self.encoder.player_vocab.next_token
        params = RosterEncoderParams(
            roster_size=ROSTER_SIZE,
            num_players=num_players,
            roster_dim=ROSTER_DIM,
            num_sab_layers=ROSTER_SAB_LAYERS,
            num_heads=4,
            d_ff=256,
            dropout=dropout,
            num_scalars=NUM_ROSTER_SCALARS,
        )
        return build_sequence_roster_encoder(params, name="roster_vec")

    def model(self, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=0.2):
        """
        Build the causal Player transformer.

        Inputs (per timestep): the Event/Time inputs (event/player/type/result/season/
        secondary_player ids, home/away rosters, time_abs, delta_time, pad_mask) plus
        the conditioning ``next_event`` / ``next_delta_time`` for the event being
        placed.

        Output: ``player`` logits over the full player vocab (softmax via from_logits
        loss).
        """
        SEQ = self.sequence_length
        D = self.model_dim
        vocab = self.encoder.vocabs
        event_vocab_size = self.encoder.event_vocab.next_token
        player_vocab_size = self.encoder.player_vocab.next_token

        self.roster_encoder = self.build_roster_encoder(dropout=dropout)

        # ---- Inputs ----
        cat_inputs = {
            f: Input(shape=(SEQ,), dtype="int32", name=f) for f in CATEGORICAL_FIELDS
        }
        home_roster = Input(shape=(SEQ, ROSTER_SIZE), dtype="int32", name="home_roster")
        away_roster = Input(shape=(SEQ, ROSTER_SIZE), dtype="int32", name="away_roster")
        time_abs = Input(shape=(SEQ, 1), dtype="float32", name="time_abs")
        delta_time = Input(shape=(SEQ, 1), dtype="float32", name="delta_time")
        rest_home, rest_away, team_inputs = make_season_inputs(SEQ)
        rotation_inputs = make_rotation_inputs(SEQ)
        prior_inputs, team_prior_inputs = make_prior_inputs(SEQ)
        regime_input = make_regime_input(SEQ)
        game_state_inputs = make_game_state_inputs(SEQ)
        next_event = Input(shape=(SEQ,), dtype="int32", name="next_event")
        next_delta_time = Input(shape=(SEQ, 1), dtype="float32", name="next_delta_time")
        pad_mask = Input(shape=(SEQ,), dtype="float32", name="pad_mask")

        # ---- Per-field embeddings ----
        # player and secondary_player share one Embedding table (weight-tied).
        player_emb_layer = layers.Embedding(player_vocab_size, EMBED_DIMS["player"], name="emb_player")
        embs = []
        emb_season = None
        for f in CATEGORICAL_FIELDS:
            if f in ("player", "secondary_player"):
                embs.append(player_emb_layer(cat_inputs[f]))
            else:
                v = vocab[f]
                embs.append(
                    layers.Embedding(v.next_token, EMBED_DIMS[f], name=f"emb_{f}")(cat_inputs[f])
                )
                # Captured by NAME for the FiLM context, so a change to CATEGORICAL_FIELDS
                # order cannot silently hand it the wrong embedding.
                if f == "season":
                    emb_season = embs[-1]

        # Conditioning event embedding (separate table: "what happens here", not history).
        next_event_emb = layers.Embedding(
            event_vocab_size, EMBED_DIMS["event"], name="emb_next_event"
        )(next_event)

        # ---- Roster encoding across the sequence (shared home/away, with per-player rest) ----
        # W7: one call, because cross-roster attention needs both sides' slots live at the
        # same moment. encode_both_rosters dispatches on which encoder this build uses, so
        # the CROSS_ROSTER_ENABLED branch lives in one place rather than six.
        home_vec, away_vec = encode_both_rosters(
            self.roster_encoder,
            [home_roster, *side_scalars(rest_home, rotation_inputs, "home", prior_inputs)],
            [away_roster, *side_scalars(rest_away, rotation_inputs, "away", prior_inputs)])

        # ---- Continuous projections ----
        t_abs = layers.Dense(16, name="time_abs_proj")(time_abs)
        t_delta = layers.Dense(16, name="delta_time_proj")(delta_time)
        t_next_delta = layers.Dense(16, name="next_delta_time_proj")(next_delta_time)
        t_team = season_team_projections(team_inputs)  # games-played + team rest per side
        t_gs = game_state_projections(game_state_inputs)
        t_prior = prior_team_projections(team_prior_inputs)  # season-to-date team rates
        t_regime = regime_projection(regime_input)  # the per-game shared component  # score / period-clock / team fouls

        # ---- Fusion + the shared causal backbone (models/backbone.py) ----
        x = build_backbone(
            [*embs, next_event_emb, home_vec, away_vec, t_abs, t_delta, t_next_delta, *t_team,
             *t_gs, *t_prior, t_regime],
            pad_mask, seq_len=SEQ, d_model=D,
            num_layers=num_layers, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout,
            # W6: the game's identity -- season, both teams' season-to-date rates, the regime latent.
            film_context=[emb_season, *t_prior, t_regime],
        )

        # ---- Output head (float32 keeps logits stable under mixed_float16) ----
        player_logits = layers.Dense(player_vocab_size, dtype="float32", name="player_logits")(x)
        # Mask to the row's on-court ten so the softmax denominator (and gradient) spans only
        # the players inference actually samples from (train/inference legality parity) —
        # the head learns who acts among those on the floor, not league appearance volume.
        player_logits = OnCourtCandidateMask(name="player_output")(
            player_logits, home_roster, away_roster)

        inputs = {
            **cat_inputs,
            "home_roster": home_roster, "away_roster": away_roster,
            "time_abs": time_abs, "delta_time": delta_time,
            "rest_home": rest_home, "rest_away": rest_away, **team_inputs,
            **prior_inputs, **team_prior_inputs, REGIME_KEY: regime_input,
            **game_state_inputs,
            **rotation_inputs,
            "next_event": next_event, "next_delta_time": next_delta_time,
            "pad_mask": pad_mask,
        }
        return keras.Model(
            inputs=inputs,
            outputs={"player_output": player_logits},
            name="PlayerModel",
        )

    # =====================
    # --- Training      ---
    # =====================

    INPUT_KEYS = (
        "event", "player", "type", "result", "season", "secondary_player",
        "home_roster", "away_roster", "time_abs", "delta_time",
        *SEASON_INPUT_KEYS, *GAME_STATE_INPUT_KEYS, *ROSTER_STATE_KEYS, *PRIOR_INPUT_KEYS, REGIME_KEY,
        "next_event", "next_delta_time", "pad_mask",
    )

    def _load_processed(self, name: str) -> dict:
        path = self.processed_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run preprocess() before train()."
            )
        with np.load(path) as data:
            return {k: data[k] for k in data.files}

    def _make_dataset(self, split: dict, batch_size: int, shuffle: bool) -> tf.data.Dataset:
        """Yield (inputs, targets, sample_weights) with PAD steps zero-weighted."""
        inputs = {k: split[k] for k in self.INPUT_KEYS}
        # game_index rides along in the batch but is NOT a model input: RegimeModel.train_step pops
        # it, looks the game's latent row up and writes it into the `regime` plane before the
        # forward pass. Absent from a split written before 3.0, in which case the zeros plane
        # stands and the head trains exactly as it did.
        if GAME_INDEX_KEY in split:
            inputs[GAME_INDEX_KEY] = split[GAME_INDEX_KEY]
        targets = {"player_output": split["player_target"]}
        mask = apply_recency(split["loss_mask"], split)
        sample_weights = {"player_output": mask}

        ds = tf.data.Dataset.from_tensor_slices((inputs, targets, sample_weights))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(mask), 1024), reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def configure_gpu(mixed_precision: bool = True):
        """Place training on CUDA when a GPU is visible (mirrors EventTimeModel)."""
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        if gpus and mixed_precision:
            keras.mixed_precision.set_global_policy("mixed_float16")
        return gpus

    def train(self, epochs=50, batch_size=64, lr=3e-4,
              patience=10, artifacts_root=DEFAULT_ARTIFACTS_ROOT,
              mixed_precision=True, jit_compile=False,
              num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=0.2,
              warmup_epochs=1, lr_alpha=0.05,
              report=True, run_name=None, reports_root=DEFAULT_REPORTS_ROOT,
              init_weights_root=None):
        """
        Fit the Player model on the preprocessed train split, validating on test.
        Masked SparseCCE(from_logits) on the player head, with loss_mask as
        sample_weight so PAD/no-next steps contribute zero loss. Saves the trained
        model alongside the vocabs and norm stats, and emits the standardized
        training/testing report (HTML + queryable Parquet).
        """
        gpus = self.configure_gpu(mixed_precision=mixed_precision)
        print(f"Training on {'GPU x' + str(len(gpus)) if gpus else 'CPU (no visible GPU)'}")
        if not self.encoder.player_vocab.frozen:
            self.encoder.load_all()
            self.encoder.freeze_all()
        _p = _norm_stats_path(self.encoder)
        if self.norm_stats is None and _p.exists():
            self.norm_stats = json.loads(_p.read_text(encoding="utf-8"))

        train_split = self._load_processed("player_train.npz")
        test_split = self._load_processed("player_test.npz")

        train_ds = self._make_dataset(train_split, batch_size, shuffle=True)
        val_ds = self._make_dataset(test_split, batch_size, shuffle=False)

        model = self.model(num_layers=num_layers, num_heads=num_heads,
                           ff_dim=ff_dim, dropout=dropout)
        model.summary()
        warm_start_weights(model, self.KEY, init_weights_root)

        # W3: wrap the functional head in its per-game latent table. `inner` stays the thing that is
        # compiled, evaluated and SAVED -- build_regime_model returns `model` untouched when the
        # latent is off, so nothing below has to branch. See models/regime.py for why val_loss will
        # read worse than a run without it, and why that is the honest number.
        inner = model
        model = build_trainer(inner, n_games=int(train_split["pad_mask"].shape[0]),
                              scheduled_sampling=False)

        steps_per_epoch = int(np.ceil(train_split["pad_mask"].shape[0] / batch_size))
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * warmup_epochs
        decay_steps = max(1, total_steps - warmup_steps)
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            warmup_target=lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            alpha=lr_alpha,
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0,
        )
        model.compile(
            optimizer=optimizer,
            loss={"player_output": keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
            # weighted_metrics (NOT metrics): only weighted_metrics receive the sample_weight
            # mask, so acc reflects real event rows, not every padded position (see note in
            # conditional_type_model.train).
            weighted_metrics={"player_output": [keras.metrics.SparseCategoricalAccuracy(name="acc")]},
            jit_compile=jit_compile,
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True,
            ),
        ]

        collector = None
        if report:
            cfg = RunConfig(
                model_key=self.KEY, epochs_planned=epochs, batch_size=batch_size,
                lr=lr, time_loss_weight=0.0, patience=patience,  # no time head
                mixed_precision=mixed_precision, jit_compile=jit_compile,
                arch={
                    "model_dim": self.model_dim,
                    "sequence_length": self.sequence_length,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "ff_dim": ff_dim,
                    "dropout": dropout,
                    "embed_dims": EMBED_DIMS,
                    "roster_dim": ROSTER_DIM,
                    "lr_schedule": "warmup_cosine",
                    "warmup_epochs": warmup_epochs,
                    "lr_alpha": lr_alpha,
                },
            )
            collector = ReportCollector(cfg, run_name=run_name, reports_root=reports_root)
            collector.capture_data(
                train_games=int(train_split["pad_mask"].shape[0]),
                test_games=int(test_split["pad_mask"].shape[0]),
                sequence_length=self.sequence_length,
                vocab_sizes={n: v.next_token for n, v in self.encoder.vocabs.items()},
                norm_stats=self.norm_stats,
            )
            collector.capture_model(model)
            callbacks.append(collector.callback)

        status = "completed"
        history = None
        try:
            history = model.fit(
                train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks,
            )
            if len(history.epoch) < epochs:
                status = "early_stopped"
        except Exception:
            status = "failed"
            if collector is not None:
                collector.finalize(status=status)
            raise

        # The latent table is a TRAINING device: what survives it is the per-dimension spread the
        # rollout samples from. Written before save_artifacts so norm_stats.json carries it.
        if model is not inner:
            self.norm_stats["regime_std"] = regime_std(model.fitted_table())
            print(f"[regime] fitted latent sd per dim: "
                  f"{[round(v, 4) for v in self.norm_stats['regime_std']]}")

        # The INNER functional model, always -- the table is not part of the graph a rollout loads.
        self.save_artifacts(inner, root=artifacts_root)

        if collector is not None:
            test_metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
            arts = collector.finalize(
                status=status,
                final_test_metrics={k: float(v) for k, v in test_metrics.items()},
            )
            print(f"Report '{self.KEY}/{arts.run_id}' -> {arts.run_dir.resolve()}")
        return model, history

    # =====================
    # --- Persistence   ---
    # =====================

    def save_artifacts(self, model, root=DEFAULT_ARTIFACTS_ROOT) -> ModelArtifacts:
        """Persist the trained model under <root>/<KEY>/ (shared ModelArtifacts layout)."""
        arts = ModelArtifacts.for_key(self.KEY, root)
        arts.ensure_dir()
        model.save(arts.keras_path)
        model.save_weights(arts.weights_path)
        self.encoder.save_all()
        if self.norm_stats is not None:
            arts.norm_stats_path.write_text(
                json.dumps(self.norm_stats, indent=2), encoding="utf-8"
            )
        print(f"Saved '{self.KEY}': model + weights + vocabs + norm stats "
              f"-> {arts.model_dir.resolve()}")
        return arts

    @classmethod
    def from_artifacts(cls, root=DEFAULT_ARTIFACTS_ROOT, encoder=None, **kwargs):
        """
        Reload a trained model from <root>/<KEY>/ (rebuild graph from frozen vocabs,
        then restore weights). Returns (instance, model) ready for inference.
        """
        arts = ModelArtifacts.for_key(cls.KEY, root)
        if not arts.weights_path.exists():
            raise FileNotFoundError(
                f"No weights for '{cls.KEY}' at {arts.weights_path.resolve()}; "
                f"train the model first."
            )
        if encoder is None:
            encoder = Encoder()
        encoder.load_all()
        encoder.freeze_all()

        inst = cls(encoder, **kwargs)
        if arts.norm_stats_path.exists():
            inst.norm_stats = json.loads(arts.norm_stats_path.read_text(encoding="utf-8"))

        model = inst.model()
        model.load_weights(arts.weights_path)
        return inst, model
