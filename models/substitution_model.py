"""
Substitution head — predict the *incoming* player of a substitution.

Where ``PlayerModel`` predicts ``player`` (here the player going **out**, sampled at
inference from the active on-court roster), this model predicts ``secondary_player`` —
the player coming **in**, sampled at inference from the substituting team's bench:

    EventTimeModel -> PlayerModel (outgoing) -> [SubstitutionModel] (incoming)

It is a near-twin of ``ConditionalTypeModel``: same causal-transformer backbone,
embeddings, roster set-encoder, reporting, and artifact contract. It differs in three
fixed (non-spec) ways:

  * it learns only on ``substitution`` rows (event-masked loss);
  * it predicts ``secondary_player`` over the **player** vocab (the incoming player);
  * it conditions on the decided outgoing ``player`` (plus ``next_event`` /
    ``next_delta_time`` like Player), embedded with the shared player table.

**Opening lineups.** The starting five is generated the same way an in-game sub is:
the outgoing slot is the literal ``"start"`` token and the model picks each incoming
starter. These ``start -> starter`` rows are *not* in the shared cleaned data — they are
synthesized here, in this model's own preprocessing, so no other model is affected and no
re-clean is needed for them. For each game we insert, right after the ``start`` frame, five
``start -> starter`` substitutions per team with the on-court roster filling 0->5.

Indexing mirrors the rest of the chain's next-step shift: at position ``i`` the model sees
rows ``0..i`` plus the decided ``next_event`` / ``next_delta_time`` / ``next_player``
(outgoing) of the event being placed, and predicts ``next_secondary_player[i]`` (incoming).
Its own preprocess writes ``sub_{train,test,holdout}.npz``.
"""
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
)
from data_loading import load_all_cleaned, split_games
from encoder.encoder import Encoder
from models.artifacts import ModelArtifacts, DEFAULT_ARTIFACTS_ROOT
from models.event_time_model import (
    AddPositionalEmbedding,
    KeyPaddingMask,
    EMBED_DIMS,
    ROSTER_DIM,
    CATEGORICAL_FIELDS,
    ROSTER_COLS,
)
from models.roster_set_encoder import (
    RosterEncoderParams,
    SequenceRosterEncoder,
)
from models.season_features import (
    SEASON_INPUT_KEYS,
    DEFAULT_REST_DAYS,
    merge_season_features,
    append_season_batches,
    make_season_inputs,
    season_team_projections,
)
from reporting import ReportCollector, RunConfig
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT

# The event whose rows this head learns on, and the outgoing/incoming fields.
SUB_EVENT = "substitution"
OUTGOING_FIELD = "player"            # decided by PlayerModel (conditioning input)
INCOMING_FIELD = "secondary_player"  # predicted by this model (target, player vocab)
START_TOKEN = "start"                # the outgoing "player" for an opening sub

# Own family file names (independent of the conditional heads' cond_*.npz).
_PROCESSED = {"train": "sub_train.npz", "test": "sub_test.npz", "holdout": "sub_holdout.npz"}

# Base history inputs (the Event/Time inputs); conditioning inputs are appended below.
_BASE_INPUT_KEYS = (
    "event", "player", "type", "result", "season", "secondary_player",
    "home_roster", "away_roster", "time_abs", "delta_time",
    *SEASON_INPUT_KEYS, "pad_mask",
)


class SubstitutionModel:
    """Causal transformer head predicting a substitution's incoming player."""

    KEY = "substitution"

    def __init__(self, encoder: Encoder,
                 sequence_length=MAX_SEQUENCE_LENGTH,
                 model_dim=256,
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

    @property
    def output_name(self) -> str:
        return f"{INCOMING_FIELD}_output"  # "secondary_player_output"

    @property
    def INPUT_KEYS(self) -> tuple:
        # Base history + always-on conditioning + the decided outgoing player.
        return (*_BASE_INPUT_KEYS, "next_event", "next_delta_time", "next_player")

    # =====================
    # --- Data Loading  ---
    # =====================

    def _load_all(self) -> pd.DataFrame:
        """Cleaned season files concatenated with globally-unique game_id (shared loader)."""
        return load_all_cleaned(self.data_dir)

    # ==========================================
    # --- Opening-lineup synthesis (in-memory) ---
    # ==========================================

    def _opening_sub_rows(self, start_row, home_starters, away_starters,
                          home_rest, away_rest) -> list[dict]:
        """Five ``start -> starter`` subs per team, rosters filling 0->5 (home then away).

        ``home_rest`` / ``away_rest`` are the starters' season-context rest values, aligned
        to ``home_starters`` / ``away_starters`` so each built-up lineup carries the matching
        per-player rest (see season_features / the roster set-encoder).
        """
        rows: list[dict] = []
        for k in range(1, len(home_starters) + 1):
            rows.append(self._sub_row(start_row, home_starters[k - 1],
                                      home_roster=home_starters[:k], away_roster=[],
                                      rest_home=home_rest[:k], rest_away=[]))
        for k in range(1, len(away_starters) + 1):
            rows.append(self._sub_row(start_row, away_starters[k - 1],
                                      home_roster=home_starters, away_roster=away_starters[:k],
                                      rest_home=home_rest, rest_away=away_rest[:k]))
        return rows

    def _sub_row(self, start_row, incoming, home_roster, away_roster,
                 rest_home, rest_away) -> dict:
        """One synthetic opening sub: outgoing = ``"start"``, incoming = a starter, time 0.

        Inherits all columns from the game's ``start`` row (game_id, season, playoff, team
        scalars, …) and overrides only the substitution-specific fields plus the
        roster-parallel ``rest_home`` / ``rest_away`` so they stay aligned with the rebuilt
        rosters.
        """
        row = start_row.to_dict()
        row.update({
            "roster_home": list(home_roster),
            "roster_away": list(away_roster),
            "rest_home": list(rest_home),
            "rest_away": list(rest_away),
            "time": 0,
            "event": SUB_EVENT,
            OUTGOING_FIELD: START_TOKEN,  # outgoing = "start"
            "type": SUB_EVENT,
            "result": SUB_EVENT,
            INCOMING_FIELD: incoming,     # incoming = a starter
        })
        return row

    def _start_rest(self, start_row, col, count) -> list:
        """The starting five's rest values from the ``start`` row, length ``count``.

        Falls back to the default rest if the data isn't season-enriched (e.g. a unit test
        on bare cleaned rows), so opening-sub synthesis never depends on the column.
        """
        if col in start_row and pd.notna(start_row[col]):
            vals = self.encoder.str_to_list(start_row[col])[:count]
            if len(vals) >= count:
                return vals
            return vals + [DEFAULT_REST_DAYS] * (count - len(vals))
        return [DEFAULT_REST_DAYS] * count

    def _augment_with_opening_subs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Insert the synthetic opening subs after each game's ``start`` frame.

        The ``start`` frame's rosters are blanked so the lineup is built up from empty —
        mirroring how the simulator generates the opening five — and the synthesized subs
        fill the on-court roster 0->5 per team before normal play resumes.
        """
        parts: list[pd.DataFrame] = []
        for _, game in df.groupby("game_id", sort=False):
            game = game.reset_index(drop=True)
            start_idx = game.index[game["event"] == "start"]
            if len(start_idx) == 0:
                parts.append(game)
                continue
            s = int(start_idx[0])
            start_row = game.loc[s]
            home_starters = self.encoder.str_to_list(start_row["roster_home"])[:ROSTER_SIZE]
            away_starters = self.encoder.str_to_list(start_row["roster_away"])[:ROSTER_SIZE]
            # Season-context rest, aligned to the starting fives (default if not enriched).
            home_rest = self._start_rest(start_row, "rest_home", len(home_starters))
            away_rest = self._start_rest(start_row, "rest_away", len(away_starters))
            synth = pd.DataFrame(self._opening_sub_rows(
                start_row, home_starters, away_starters, home_rest, away_rest))

            game.at[s, "roster_home"] = []  # build the five up from empty
            game.at[s, "roster_away"] = []
            if "rest_home" in game.columns:  # keep rest aligned with the blanked roster
                game.at[s, "rest_home"] = []
                game.at[s, "rest_away"] = []
            parts.append(pd.concat(
                [game.iloc[: s + 1], synth, game.iloc[s + 1:]], ignore_index=True
            ))
        return pd.concat(parts, ignore_index=True)

    # =====================
    # --- Preprocessing ---
    # =====================

    def preprocess(self, rebuild_vocabs=False, test_frac=TEST_FRAC,
                   holdout_frac=HOLDOUT_FRAC, seed=SEED):
        """
        Build the substitution tensor arrays (with synthesized opening subs) + persist.

        Same loader / encoding / Δt normalization / train-val-holdout split (same ``seed`` +
        fracs as the rest of the chain, so partitions + norm_stats line up). ``rebuild_vocabs``
        defaults False — the shared vocab language is built + frozen by the Event/Time model;
        here we only load + freeze it. Writes ``sub_{train,test,holdout}.npz``.
        """
        df = self._load_all()
        df = self._augment_with_opening_subs(df)

        if rebuild_vocabs:
            for col, src in ROSTER_COLS.items():
                df[src].apply(self.encoder.encode_roster)
            for field in CATEGORICAL_FIELDS:
                df[field].apply(getattr(self.encoder, f"encode_{field}"))
            self.encoder.save_all()
        else:
            self.encoder.load_all()
        self.encoder.freeze_all()

        enc = {f: df[f].apply(getattr(self.encoder, f"encode_{f}")).to_numpy() for f in CATEGORICAL_FIELDS}
        rosters = {
            name: np.stack(df[src].apply(self.encoder.encode_roster).to_numpy())
            for name, src in ROSTER_COLS.items()
        }
        time = df["time"].to_numpy(dtype=np.float64)
        game_id = df["game_id"].to_numpy()

        # Per-game delta_time (no cross-game leakage); opening subs sit at time 0 (Δt 0).
        delta = (
            df.groupby("game_id")["time"].diff().fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
        )

        train_games, test_games, holdout_games = split_games(
            np.unique(game_id), seed=seed, test_frac=test_frac, holdout_frac=holdout_frac,
        )
        train_mask = np.array([g in train_games for g in game_id])

        max_time = float(time[train_mask].max()) or 1.0
        train_delta = delta[train_mask]
        delta_mean = float(train_delta.mean())
        delta_std = float(train_delta.std()) or 1.0
        self.norm_stats = {"max_time": max_time, "delta_mean": delta_mean, "delta_std": delta_std}

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
            df, cols, rosters, self.encoder.encode_player("PAD"), train_mask, self.norm_stats
        )
        train = self._build_split(cols, game_id, train_games)
        test = self._build_split(cols, game_id, test_games)
        holdout = self._build_split(cols, game_id, holdout_games)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.processed_dir / _PROCESSED["train"], **train)
        np.savez_compressed(self.processed_dir / _PROCESSED["test"], **test)
        np.savez_compressed(self.processed_dir / _PROCESSED["holdout"], **holdout)
        (self.processed_dir / HOLDOUT_MANIFEST_NAME).write_text(
            json.dumps(sorted(int(g) for g in holdout_games), indent=2), encoding="utf-8"
        )

        print(f"Preprocessed {len(train_games)} train / {len(test_games)} test / "
              f"{len(holdout_games)} holdout games -> {self.processed_dir} "
              f"(SEQ={self.sequence_length}, substitution arrays w/ opening subs)")
        return train, test

    def _build_split(self, cols, game_id, games) -> dict:
        """
        Pad/truncate each game to sequence_length and stack into batched arrays.

        Same base layout as the rest of the chain, plus the next-step (chain) arrays this
        head needs: ``next_event`` / ``next_delta_time`` / ``next_player`` (outgoing) as
        conditioning, and ``next_secondary_player`` (incoming) as the target. ``loss_mask``
        is 1 for real rows with a valid next step; the per-event (substitution) mask is
        applied at dataset time.
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
        next_pad = {
            "next_event": PAD_EVENT, "next_player": PAD_PLAYER,
            "next_secondary_player": PAD_SECONDARY,
        }
        next_src = {
            "next_event": "event", "next_player": "player",
            "next_secondary_player": "secondary_player",
        }

        keys_1d = CATEGORICAL_FIELDS
        keys_roster = ["home_roster", "away_roster"]
        keys_cont = ["time_abs", "delta_time"]
        keys_next_cat = ["next_event", "next_player", "next_secondary_player"]

        batches = {k: [] for k in (*keys_1d, *keys_roster, *keys_cont, *SEASON_INPUT_KEYS,
                                   *keys_next_cat, "next_delta_time",
                                   "pad_mask", "loss_mask")}

        game_ids_sorted = [g for g in np.unique(game_id) if g in games]
        for g in game_ids_sorted:
            idx = np.where(game_id == g)[0]
            idx = idx[:SEQ]  # truncate overlong games
            n = len(idx)

            for k in keys_1d:
                buf = np.full((SEQ,), pad_scalar[k], dtype=np.int32)
                buf[:n] = cols[k][idx]
                batches[k].append(buf)

            for k in keys_roster:
                buf = np.full((SEQ, ROSTER_SIZE), PAD_PLAYER, dtype=np.int32)
                buf[:n] = cols[k][idx]
                batches[k].append(buf)

            for k in keys_cont:
                buf = np.zeros((SEQ, 1), dtype=np.float32)
                buf[:n, 0] = cols[k][idx]
                batches[k].append(buf)

            append_season_batches(batches, cols, idx, n, SEQ)

            for k in keys_next_cat:
                buf = np.full((SEQ,), next_pad[k], dtype=np.int32)
                if n > 1:
                    buf[: n - 1] = cols[next_src[k]][idx][1:]
                batches[k].append(buf)
            next_delta = np.zeros((SEQ, 1), dtype=np.float32)
            if n > 1:
                next_delta[: n - 1, 0] = cols["delta_time"][idx][1:]
            batches["next_delta_time"].append(next_delta)

            pad_mask = np.zeros((SEQ,), dtype=np.float32)
            pad_mask[:n] = 1.0
            loss_mask = np.zeros((SEQ,), dtype=np.float32)
            loss_mask[: max(n - 1, 0)] = 1.0  # last real row has no next target
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
            num_sab_layers=2,
            num_heads=4,
            d_ff=256,
            dropout=dropout,
        )
        return SequenceRosterEncoder(params, name="roster_vec")

    def model(self, num_layers=4, num_heads=8, ff_dim=1024, dropout=0.2):
        """
        Build the causal substitution transformer.

        Inputs: the Event/Time history inputs, the always-on ``next_event`` /
        ``next_delta_time`` conditioning, plus ``next_player`` (the decided outgoing player).
        Output: ``secondary_player_output`` logits over the full player vocab — the incoming
        player (restricting to the team's bench is a sampling-time concern in the simulator).
        """
        SEQ = self.sequence_length
        D = self.model_dim
        vocab = self.encoder.vocabs
        event_vocab_size = self.encoder.event_vocab.next_token
        player_vocab_size = self.encoder.player_vocab.next_token
        target_vocab_size = player_vocab_size  # incoming player over the player vocab

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
        next_event = Input(shape=(SEQ,), dtype="int32", name="next_event")
        next_delta_time = Input(shape=(SEQ, 1), dtype="float32", name="next_delta_time")
        next_player = Input(shape=(SEQ,), dtype="int32", name="next_player")
        pad_mask = Input(shape=(SEQ,), dtype="float32", name="pad_mask")

        # ---- Per-field embeddings (player + secondary_player weight-tied) ----
        player_emb_layer = layers.Embedding(player_vocab_size, EMBED_DIMS["player"], name="emb_player")
        embs = []
        for f in CATEGORICAL_FIELDS:
            if f in ("player", "secondary_player"):
                embs.append(player_emb_layer(cat_inputs[f]))
            else:
                v = vocab[f]
                embs.append(
                    layers.Embedding(v.next_token, EMBED_DIMS[f], name=f"emb_{f}")(cat_inputs[f])
                )

        # ---- Conditioning embeddings ("what happens here", separate from history) ----
        cond_vecs = [
            layers.Embedding(event_vocab_size, EMBED_DIMS["event"], name="emb_next_event")(next_event),
            # Outgoing player tied to the shared player table (same entity space).
            player_emb_layer(next_player),
        ]

        # ---- Roster encoding across the sequence (shared home/away, with per-player rest) ----
        home_vec = self.roster_encoder([home_roster, rest_home])
        away_vec = self.roster_encoder([away_roster, rest_away])

        # ---- Continuous projections ----
        t_abs = layers.Dense(16, name="time_abs_proj")(time_abs)
        t_delta = layers.Dense(16, name="delta_time_proj")(delta_time)
        t_next_delta = layers.Dense(16, name="next_delta_time_proj")(next_delta_time)
        t_team = season_team_projections(team_inputs)  # games-played + team rest per side

        # ---- Fusion ----
        x = layers.Concatenate(axis=-1, name="fusion_concat")(
            [*embs, *cond_vecs, home_vec, away_vec, t_abs, t_delta, t_next_delta, *t_team]
        )
        x = layers.Dense(D, name="fusion_projection")(x)
        x = layers.LayerNormalization(epsilon=1e-6, name="fusion_ln")(x)

        # ---- Positional encoding (learned) ----
        x = AddPositionalEmbedding(SEQ, D, name="positional_embedding")(x)
        x = layers.Dropout(dropout, name="emb_dropout")(x)

        # ---- Attention mask: (B, 1, SEQ) boolean key-padding mask ----
        attn_mask = KeyPaddingMask(name="attn_pad_mask")(pad_mask)

        # ---- Causal transformer encoder ----
        for i in range(num_layers):
            h = layers.LayerNormalization(epsilon=1e-6, name=f"block{i}_ln1")(x)
            attn = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=D // num_heads, dropout=dropout,
                name=f"block{i}_mha",
            )(h, h, attention_mask=attn_mask, use_causal_mask=True)
            x = layers.Add(name=f"block{i}_res1")([x, attn])

            h = layers.LayerNormalization(epsilon=1e-6, name=f"block{i}_ln2")(x)
            f1 = layers.Dense(ff_dim, activation="gelu", name=f"block{i}_ff1")(h)
            f1 = layers.Dropout(dropout, name=f"block{i}_ffdrop")(f1)
            f2 = layers.Dense(D, name=f"block{i}_ff2")(f1)
            x = layers.Add(name=f"block{i}_res2")([x, f2])

        x = layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)

        # ---- Output head (float32 keeps logits stable under mixed_float16) ----
        logits = layers.Dense(target_vocab_size, dtype="float32", name=self.output_name)(x)

        inputs = {
            **cat_inputs,
            "home_roster": home_roster, "away_roster": away_roster,
            "time_abs": time_abs, "delta_time": delta_time,
            "rest_home": rest_home, "rest_away": rest_away, **team_inputs,
            "next_event": next_event, "next_delta_time": next_delta_time,
            "next_player": next_player,
            "pad_mask": pad_mask,
        }
        return keras.Model(
            inputs=inputs,
            outputs={self.output_name: logits},
            name="SubstitutionModel",
        )

    # =====================
    # --- Training      ---
    # =====================

    def _load_processed(self, name: str) -> dict:
        path = self.processed_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run preprocess() before train()."
            )
        with np.load(path) as data:
            return {k: data[k] for k in data.files}

    def _make_dataset(self, split: dict, batch_size: int, shuffle: bool) -> tf.data.Dataset:
        """Yield (inputs, targets, sample_weights), with the loss masked to substitution rows.

        ``loss_mask`` already zeroes PAD / no-next steps; we additionally zero every step
        whose decided event is not ``substitution`` so the head only learns on sub placements
        (including the synthesized opening subs).
        """
        inputs = {k: split[k] for k in self.INPUT_KEYS}
        targets = {self.output_name: split["next_secondary_player"]}

        event_id = self.encoder.encode_event(SUB_EVENT)
        event_mask = (split["next_event"] == event_id).astype(np.float32)
        mask = split["loss_mask"] * event_mask
        sample_weights = {self.output_name: mask}

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
              num_layers=4, num_heads=8, ff_dim=1024, dropout=0.2,
              warmup_epochs=1, lr_alpha=0.05,
              report=True, run_name=None, reports_root=DEFAULT_REPORTS_ROOT):
        """
        Fit the substitution head on the preprocessed train split, validating on test.
        Masked SparseCCE(from_logits) over the player vocab, with the substitution-restricted
        loss_mask as sample_weight. Saves the model alongside the vocabs + norm stats and
        emits the standardized report.
        """
        gpus = self.configure_gpu(mixed_precision=mixed_precision)
        print(f"Training '{self.KEY}' on {'GPU x' + str(len(gpus)) if gpus else 'CPU (no visible GPU)'}")
        if not self.encoder.player_vocab.frozen:
            self.encoder.load_all()
            self.encoder.freeze_all()
        if self.norm_stats is None and Path(NORM_STATS_PATH).exists():
            self.norm_stats = json.loads(Path(NORM_STATS_PATH).read_text(encoding="utf-8"))

        train_split = self._load_processed(_PROCESSED["train"])
        test_split = self._load_processed(_PROCESSED["test"])

        train_ds = self._make_dataset(train_split, batch_size, shuffle=True)
        val_ds = self._make_dataset(test_split, batch_size, shuffle=False)

        model = self.model(num_layers=num_layers, num_heads=num_heads,
                           ff_dim=ff_dim, dropout=dropout)
        model.summary()

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
            loss={self.output_name: keras.losses.SparseCategoricalCrossentropy(from_logits=True)},
            metrics={self.output_name: [keras.metrics.SparseCategoricalAccuracy(name="acc")]},
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
                lr=lr, time_loss_weight=0.0, patience=patience,  # single classification head
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
                    "conditional_event": SUB_EVENT,
                    "target_field": INCOMING_FIELD,
                    "condition_fields": [OUTGOING_FIELD],
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

        self.save_artifacts(model, root=artifacts_root)

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
        Reload the trained head from <root>/<KEY>/ (rebuild graph from frozen vocabs, then
        restore weights). Returns (instance, model) for inference.
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
