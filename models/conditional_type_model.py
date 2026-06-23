"""
Conditional type/result generation heads.

These are the play-detail stages of the autoregressive chain. Where ``EventTimeModel``
learns ``p(event, Δt | history)`` and ``PlayerModel`` learns ``p(player | history, event,
Δt)``, these models learn the *kind* and *outcome* of an event once who/what/when is
decided:

    EventTimeModel -> PlayerModel -> [type heads] -> [result head]

The prior (BallPredict) solution had one of these per event type; the cleaned data here
makes that explicit — ``type`` is a genuine choice for shot/assist/turnover/foul while
``result`` is deterministic from ``(event, type)`` for every event *except* ``shot``. So
the learned set is exactly: Shot Type, Shot Result, Assist Type, Turnover Type, Foul Type.

They are near-identical to ``PlayerModel`` (same causal transformer backbone, embeddings,
roster set-encoder, reporting, artifact contract). They differ only in three spec-driven
ways, captured by ``TypeGenSpec``:

  * which event the loss is masked to (a shot-type head only learns on shot rows);
  * which field it predicts (``type`` or ``result``);
  * which already-decided chain values it conditions on (always ``next_event`` /
    ``next_delta_time`` like Player, plus ``next_player`` for every head here, plus
    ``next_type`` for the result head — the type a shot is about to be).

So one parameterized class, ``ConditionalTypeModel``, is instantiated five times. The
concrete per-key classes in ``CONDITIONAL_MODEL_CLASSES`` bind a spec so the registry's
``key -> class with .KEY + .from_artifacts`` contract is unchanged.

Indexing mirrors EventTimeModel/PlayerModel's next-step shift: at position ``i`` the model
sees prior rows ``0..i`` plus the decided ``next_*`` values for the event being placed and
predicts ``next_<field>[i]``. One shared preprocess writes ``cond_{train,test,holdout}.npz``
holding the base tensors plus every ``next_*`` chain array, so all five heads train off the
same file (the per-event mask + target selection happen per-spec at dataset time).
"""
import json
from dataclasses import dataclass
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
    merge_season_features,
    append_season_batches,
    make_season_inputs,
    season_team_projections,
)
from reporting import ReportCollector, RunConfig
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT


@dataclass(frozen=True)
class TypeGenSpec:
    """Everything that distinguishes one conditional head from another.

    ``next_event`` / ``next_delta_time`` are always-present conditioning inputs (as in
    PlayerModel); ``condition_fields`` lists the *extra* decided chain values this head
    consumes — ``"player"`` (every head here) and/or ``"type"`` (the result head).
    """

    key: str                       # stable artifact KEY / registry key, e.g. "shot_type"
    event: str                     # event token whose rows this head learns on, e.g. "shot"
    target_field: str              # "type" or "result" — the field predicted
    condition_fields: tuple        # extra decided fields, e.g. ("player",) or ("player", "type")


# The learned set (the prior solution's, minus the deterministic-result heads, plus the
# rebound-type head). ``rebound_type`` carries NO ``condition_fields``: the off/def split is
# decided from history alone (``next_event`` / ``next_delta_time``) *before* the rebounder is
# picked, so it does not condition on ``next_player`` — the controller samples the type first,
# then the rebounder from the team that type implies.
TYPE_GEN_SPECS: dict[str, TypeGenSpec] = {
    "shot_type":     TypeGenSpec("shot_type",     "shot",     "type",   ("player",)),
    "shot_result":   TypeGenSpec("shot_result",   "shot",     "result", ("player", "type")),
    "assist_type":   TypeGenSpec("assist_type",   "assist",   "type",   ("player",)),
    "turnover_type": TypeGenSpec("turnover_type", "turnover", "type",   ("player",)),
    "foul_type":     TypeGenSpec("foul_type",     "foul",     "type",   ("player",)),
    "rebound_type":  TypeGenSpec("rebound_type",  "rebound",  "type",   ()),
}

# Shared family file names: one preprocess feeds every head (see module docstring).
_PROCESSED = {"train": "cond_train.npz", "test": "cond_test.npz", "holdout": "cond_holdout.npz"}

# Base history inputs (the Event/Time inputs); conditioning inputs are appended per-spec.
_BASE_INPUT_KEYS = (
    "event", "player", "type", "result", "season", "secondary_player",
    "home_roster", "away_roster", "time_abs", "delta_time",
    *SEASON_INPUT_KEYS, "pad_mask",
)


class ConditionalTypeModel:
    """One causal transformer head that predicts an event's ``type`` or ``result``.

    Parameterized by a ``TypeGenSpec``; the concrete per-key subclasses in
    ``CONDITIONAL_MODEL_CLASSES`` bind a spec and set ``KEY``/``SPEC`` so this plugs into
    the shared registry/bundle/persistence machinery unchanged.
    """

    # KEY / SPEC are set on the bound subclasses; the base is never registered directly.
    KEY: str = "conditional_type"
    SPEC: TypeGenSpec | None = None

    def __init__(self, spec: TypeGenSpec, encoder: Encoder,
                 sequence_length=MAX_SEQUENCE_LENGTH,
                 model_dim=256,
                 path="./data",
                 processed_dir="./data/processed"):
        self.spec = spec
        self.sequence_length = sequence_length
        self.model_dim = model_dim
        self.encoder = encoder

        self.data_dir = Path(path)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir.resolve()}")
        self.processed_dir = Path(processed_dir)

        self.roster_encoder = None
        self.norm_stats = None

    # The head's output name (no collision with the like-named input field, mirroring
    # EventTimeModel's event_output vs event input).
    @property
    def output_name(self) -> str:
        return f"{self.spec.target_field}_output"

    # Base history inputs + always-on conditioning + the spec's extra decided fields.
    @property
    def INPUT_KEYS(self) -> tuple:
        return (
            *_BASE_INPUT_KEYS,
            "next_event", "next_delta_time",
            *(f"next_{f}" for f in self.spec.condition_fields),
        )

    # =====================
    # --- Data Loading  ---
    # =====================

    def _load_all(self) -> pd.DataFrame:
        """Cleaned season files concatenated with globally-unique game_id (shared loader)."""
        return load_all_cleaned(self.data_dir)

    # =====================
    # --- Preprocessing ---
    # =====================

    def preprocess(self, rebuild_vocabs=False, test_frac=TEST_FRAC,
                   holdout_frac=HOLDOUT_FRAC, seed=SEED):
        """
        Build the shared conditional tensor arrays + persist them and the time-norm stats.

        Same loader / encoding / Δt normalization / train-val-holdout game split (same
        ``seed`` + fracs as Event/Time + Player, so the partitions and norm_stats line up
        exactly). Writes ONE family file set (``cond_{train,test,holdout}.npz``) holding the
        base tensors plus every ``next_*`` chain array, so all five heads train off it.
        ``rebuild_vocabs`` defaults False (the shared vocab language is built + frozen by
        the Event/Time model); we just load + freeze it.
        """
        df = self._load_all()

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

        # Per-game delta_time (no cross-game leakage).
        delta = (
            df.groupby("game_id")["time"].diff().fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
        )

        train_games, test_games, holdout_games = split_games(
            np.unique(game_id), seed=seed, test_frac=test_frac, holdout_frac=holdout_frac,
        )
        train_mask = np.array([g in train_games for g in game_id])

        # Normalization stats from TRAIN ONLY (recomputed identically to Event/Time).
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
              f"(SEQ={self.sequence_length}, shared conditional arrays)")
        return train, test

    def _build_split(self, cols, game_id, games) -> dict:
        """
        Pad/truncate each game to sequence_length and stack into batched arrays.

        Same base layout as Event/Time + Player, plus the full set of next-step (chain)
        arrays so any spec can read what it needs:
          - ``next_event`` / ``next_delta_time``: the decided event + standardized Δt of
            the event being placed (the always-on conditioning, == Event/Time targets).
          - ``next_player`` / ``next_type`` / ``next_result``: the decided player and the
            target type/result (a type head's target is ``next_type``; the result head also
            conditions on ``next_type`` and targets ``next_result``).
          - ``loss_mask``: 1 for real rows with a valid next step (per-event masking is
            applied at dataset time, not here, so the one file serves all heads).
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
        # PAD value for each next_* shift array (continuous Δt pads with 0).
        next_pad = {
            "next_event": PAD_EVENT, "next_player": PAD_PLAYER,
            "next_type": PAD_TYPE, "next_result": PAD_RESULT,
        }
        # Source column each next_* array shifts.
        next_src = {
            "next_event": "event", "next_player": "player",
            "next_type": "type", "next_result": "result",
        }

        keys_1d = CATEGORICAL_FIELDS
        keys_roster = ["home_roster", "away_roster"]
        keys_cont = ["time_abs", "delta_time"]
        keys_next_cat = ["next_event", "next_player", "next_type", "next_result"]

        batches = {k: [] for k in (*keys_1d, *keys_roster, *keys_cont, *SEASON_INPUT_KEYS,
                                   *keys_next_cat, "next_delta_time",
                                   "pad_mask", "loss_mask")}

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

            for k in keys_roster:
                buf = np.full((SEQ, ROSTER_SIZE), PAD_PLAYER, dtype=np.int32)
                buf[:n] = cols[k][idx]
                batches[k].append(buf)

            for k in keys_cont:
                buf = np.zeros((SEQ, 1), dtype=np.float32)
                buf[:n, 0] = cols[k][idx]
                batches[k].append(buf)

            append_season_batches(batches, cols, idx, n, SEQ)

            # Conditioning / target arrays: next-step shift within this game.
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
        Build the causal conditional transformer for this spec.

        Inputs: the Event/Time history inputs, the always-on ``next_event`` /
        ``next_delta_time`` conditioning, plus one ``next_<f>`` per ``condition_fields``.
        Output: ``<field>_output`` logits over the full type/result vocab (softmax via
        from_logits loss; restricting to an event's legal classes is a sampling-time
        concern handled later in the simulator).
        """
        SEQ = self.sequence_length
        D = self.model_dim
        spec = self.spec
        vocab = self.encoder.vocabs
        event_vocab_size = self.encoder.event_vocab.next_token
        player_vocab_size = self.encoder.player_vocab.next_token
        type_vocab_size = vocab["type"].next_token
        target_vocab_size = vocab[spec.target_field].next_token

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
        pad_mask = Input(shape=(SEQ,), dtype="float32", name="pad_mask")
        cond_inputs = {}
        if "player" in spec.condition_fields:
            cond_inputs["next_player"] = Input(shape=(SEQ,), dtype="int32", name="next_player")
        if "type" in spec.condition_fields:
            cond_inputs["next_type"] = Input(shape=(SEQ,), dtype="int32", name="next_type")

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
            layers.Embedding(event_vocab_size, EMBED_DIMS["event"], name="emb_next_event")(next_event)
        ]
        if "next_player" in cond_inputs:
            # Tie to the shared player table: the actor is the same entity space.
            cond_vecs.append(player_emb_layer(cond_inputs["next_player"]))
        if "next_type" in cond_inputs:
            cond_vecs.append(
                layers.Embedding(type_vocab_size, EMBED_DIMS["type"], name="emb_next_type")(cond_inputs["next_type"])
            )

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
            **cond_inputs,
            "pad_mask": pad_mask,
        }
        return keras.Model(
            inputs=inputs,
            outputs={self.output_name: logits},
            name=f"ConditionalTypeModel_{spec.key}",
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
        """Yield (inputs, targets, sample_weights), with the loss masked to this head's event.

        The shared file carries every event's rows; ``loss_mask`` already zeroes PAD /
        no-next steps. We additionally zero every step whose decided event is not this
        head's event, so e.g. the shot-type head only learns on shot placements.
        """
        inputs = {k: split[k] for k in self.INPUT_KEYS}
        targets = {self.output_name: split[f"next_{self.spec.target_field}"]}

        event_id = self.encoder.encode_event(self.spec.event)
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
        Fit this conditional head on the preprocessed train split, validating on test.
        Masked SparseCCE(from_logits) on the single head, with the event-restricted
        loss_mask as sample_weight so PAD / no-next / other-event steps contribute zero
        loss. Saves the model alongside the vocabs + norm stats and emits the standardized
        training/testing report.
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
                    "conditional_event": self.spec.event,
                    "target_field": self.spec.target_field,
                    "condition_fields": list(self.spec.condition_fields),
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
        Reload a trained head from <root>/<KEY>/ (rebuild graph from frozen vocabs + the
        bound ``SPEC``, then restore weights). Returns (instance, model) for inference.
        """
        if cls.SPEC is None:
            raise TypeError(
                "from_artifacts must be called on a bound conditional class "
                "(e.g. the registry entry for 'shot_type'), not ConditionalTypeModel itself."
            )
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


def _bind(spec: TypeGenSpec) -> type:
    """Create a concrete per-key subclass that binds ``spec`` (so the registry's
    ``key -> class with .KEY + .from_artifacts`` contract holds unchanged)."""

    class _Bound(ConditionalTypeModel):
        KEY = spec.key
        SPEC = spec

        def __init__(self, encoder, **kwargs):
            super().__init__(spec, encoder, **kwargs)

    _Bound.__name__ = f"ConditionalTypeModel_{spec.key}"
    _Bound.__qualname__ = _Bound.__name__
    return _Bound


# One bound class per spec; the registry consumes these by KEY.
CONDITIONAL_MODEL_CLASSES = [_bind(spec) for spec in TYPE_GEN_SPECS.values()]
