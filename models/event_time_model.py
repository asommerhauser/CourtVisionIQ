import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, Input

from config import MAX_SEQUENCE_LENGTH, ROSTER_SIZE, NORM_STATS_PATH
from encoder.encoder import Encoder
from models.artifacts import ModelArtifacts, DEFAULT_ARTIFACTS_ROOT
from models.roster_set_encoder import (
    RosterSetEncoder,
    RosterEncoderParams,
    SequenceRosterEncoder,
)
from reporting import ReportCollector, RunConfig
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT

# Per-field embedding dimensions (categoricals never enter the model as raw scalars).
EMBED_DIMS = {"event": 32, "player": 128, "type": 32, "result": 16, "season": 16}
ROSTER_DIM = 128


@keras.saving.register_keras_serializable(package="cviq")
class AddPositionalEmbedding(layers.Layer):
    """Add a learned position embedding over [0, seq_len) to a (B, SEQ, D) tensor."""

    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model

    def build(self, input_shape):
        # Own variable (created in build, not a nested Embedding) so it serializes
        # and reloads cleanly. Shape (SEQ, D) broadcasts over the batch axis.
        self.pos = self.add_weight(
            name="pos_table",
            shape=(self.seq_len, self.d_model),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos                         # (SEQ, D) broadcasts over (B, SEQ, D)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return cfg

@keras.saving.register_keras_serializable(package="cviq")
class KeyPaddingMask(layers.Layer):
    """Turn a (B, SEQ) float pad-mask into a (B, 1, SEQ) boolean key-padding mask.

    A registered layer (rather than a Lambda) so the full .keras model reloads
    under Keras 3 safe mode without custom code execution.
    """

    def call(self, m):
        return tf.cast(m, "bool")[:, tf.newaxis, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[1])


# Cleaned-data columns the model consumes.
# secondary_player shares the player embedding table (weight-tied) — see model().
CATEGORICAL_FIELDS = ["event", "player", "type", "result", "season", "secondary_player"]
ROSTER_COLS = {"home_roster": "roster_home", "away_roster": "roster_away"}


class EventTimeModel:
    """
    Core Event/Time Transformer.

    preprocess() turns cleaned, game-grouped CSV(s) into padded, normalized tensor
    arrays (one game = one fixed-length sequence) plus the shared vocab "language"
    and time-normalization stats. model() builds the causal transformer; train()
    fits it. Encoding is standardized so every downstream model speaks the same
    token language.
    """

    # Stable key used for the on-disk artifact layout and the model registry.
    KEY = "event_time"

    def __init__(self, encoder: Encoder,
                 sequence_length=MAX_SEQUENCE_LENGTH,
                 model_dim=256,
                 event_classes=7,
                 path="./data",
                 processed_dir="./data/processed"):

        self.sequence_length = sequence_length
        self.model_dim = model_dim
        self.event_classes = event_classes
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

    def _cleaned_csvs(self):
        """Cleaned season files only (have game_id + roster columns)."""
        out = []
        for p in sorted(self.data_dir.glob("*.csv")):
            try:
                cols = pd.read_csv(p, nrows=0).columns
            except Exception:
                continue
            if "game_id" in cols and "roster_home" in cols:
                out.append(p)
        return out

    def _load_all(self) -> pd.DataFrame:
        frames = []
        offset = 0
        for p in self._cleaned_csvs():
            df = pd.read_csv(p)
            # Keep game_id globally unique across files.
            df["game_id"] = df["game_id"].astype(int) + offset
            offset = int(df["game_id"].max()) + 1
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No cleaned CSVs found in {self.data_dir.resolve()}")
        return pd.concat(frames, ignore_index=True)

    # =====================
    # --- Preprocessing ---
    # =====================

    def preprocess(self, rebuild_vocabs=True, test_frac=0.2, seed=42):
        """
        Build the model-ready tensor arrays and persist them, the vocabs, and the
        time-normalization stats. Returns (train, test) dicts of numpy arrays.
        """
        df = self._load_all()

        # 1) Build (or load) the shared vocab language, then FREEZE it so token IDs
        #    are stable and unseen values map to UNK rather than mutating the space.
        if rebuild_vocabs:
            for col, src in ROSTER_COLS.items():
                df[src].apply(self.encoder.encode_roster)
            for field in CATEGORICAL_FIELDS:
                df[field].apply(getattr(self.encoder, f"encode_{field}"))
            self.encoder.save_all()
        else:
            self.encoder.load_all()
        self.encoder.freeze_all()

        # 2) Encode every categorical column to ints and rosters to fixed-5 arrays.
        enc = {f: df[f].apply(getattr(self.encoder, f"encode_{f}")).to_numpy() for f in CATEGORICAL_FIELDS}
        rosters = {
            name: np.stack(df[src].apply(self.encoder.encode_roster).to_numpy())
            for name, src in ROSTER_COLS.items()
        }  # each (N, 5)
        time = df["time"].to_numpy(dtype=np.float64)
        game_id = df["game_id"].to_numpy()

        # 3) Per-game delta_time (no cross-game leakage): first row of each game
        #    gets delta 0; clip guards against any backwards time stamps.
        delta = (
            df.groupby("game_id")["time"].diff().fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
        )

        # 4) Split games into train/test BEFORE computing normalization stats.
        unique_games = np.unique(game_id)
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_games)
        n_test = max(1, int(round(len(unique_games) * test_frac)))
        test_games = set(unique_games[:n_test].tolist())
        train_games = set(unique_games[n_test:].tolist())

        train_mask = np.array([g in train_games for g in game_id])

        # 5) Normalization stats from TRAIN ONLY. time_abs = time/max_time;
        #    delta standardized. Persist so inference uses identical transforms.
        max_time = float(time[train_mask].max()) or 1.0
        train_delta = delta[train_mask]
        delta_mean = float(train_delta.mean())
        delta_std = float(train_delta.std()) or 1.0
        self.norm_stats = {"max_time": max_time, "delta_mean": delta_mean, "delta_std": delta_std}

        time_abs = (time / max_time).astype(np.float32)
        delta_norm = ((delta - delta_mean) / delta_std).astype(np.float32)

        # 6) Assemble per-game padded sequences for each split.
        cols = {
            **{f: enc[f] for f in CATEGORICAL_FIELDS},
            "home_roster": rosters["home_roster"],
            "away_roster": rosters["away_roster"],
            "time_abs": time_abs,
            "delta_time": delta_norm,
        }
        train = self._build_split(cols, game_id, train_games)
        test = self._build_split(cols, game_id, test_games)

        # 7) Persist.
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.processed_dir / "train.npz", **train)
        np.savez_compressed(self.processed_dir / "test.npz", **test)
        Path(NORM_STATS_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(NORM_STATS_PATH).write_text(json.dumps(self.norm_stats, indent=2), encoding="utf-8")

        print(f"Preprocessed {len(train_games)} train / {len(test_games)} test games "
              f"-> {self.processed_dir} (SEQ={self.sequence_length})")
        return train, test

    def _build_split(self, cols, game_id, games) -> dict:
        """
        Pad/truncate each game to sequence_length and stack into batched arrays.

        Produces inputs, shifted targets, and two masks:
          - pad_mask:  1 for real timesteps (incl. the terminal 'end' row), else 0.
                       Used to mask attention over padded timesteps.
          - loss_mask: 1 for real timesteps that have a valid next-step target
                       (i.e. not the last real row), else 0. Used as sample_weight.
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

        batches = {k: [] for k in (*keys_1d, *keys_roster, *keys_cont,
                                   "event_target", "time_target", "pad_mask", "loss_mask")}

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

            # Targets: next-step shift within this game.
            event_t = np.full((SEQ,), PAD_EVENT, dtype=np.int32)
            time_t = np.zeros((SEQ, 1), dtype=np.float32)
            if n > 1:
                event_t[: n - 1] = cols["event"][idx][1:]
                time_t[: n - 1, 0] = cols["delta_time"][idx][1:]
            batches["event_target"].append(event_t)
            batches["time_target"].append(time_t)

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

    def build_roster_encoder(self) -> SequenceRosterEncoder:
        num_players = self.encoder.player_vocab.next_token
        params = RosterEncoderParams(
            roster_size=ROSTER_SIZE,
            num_players=num_players,
            roster_dim=ROSTER_DIM,
            num_sab_layers=2,
            num_heads=4,
            d_ff=256,
            dropout=0.1,
        )
        # Applies the shared set-encoder across the time axis via reshape (not
        # TimeDistributed, which unrolls SEQ in graph mode and exhausts memory).
        return SequenceRosterEncoder(params, name="roster_vec")

    def model(self, num_layers=4, num_heads=8, ff_dim=1024, dropout=0.1):
        """
        Build the causal Event/Time Transformer.

        Inputs (per timestep, right-padded to sequence_length):
          event/player/type/result/season ids, home_roster/away_roster (5 slots),
          time_abs, delta_time, and pad_mask (1 real / 0 pad) for attention masking.

        Outputs:
          event  -> logits over the full event vocab (softmax via from_logits loss)
          time   -> scalar next-step normalized delta_time (regression)
        """
        SEQ = self.sequence_length
        D = self.model_dim
        vocab = self.encoder.vocabs
        event_vocab_size = self.encoder.event_vocab.next_token

        # Shared roster set-encoder (weight-tied across home/away).
        self.roster_encoder = self.build_roster_encoder()

        # ---- Inputs ----
        cat_inputs = {
            f: Input(shape=(SEQ,), dtype="int32", name=f) for f in CATEGORICAL_FIELDS
        }
        home_roster = Input(shape=(SEQ, ROSTER_SIZE), dtype="int32", name="home_roster")
        away_roster = Input(shape=(SEQ, ROSTER_SIZE), dtype="int32", name="away_roster")
        time_abs = Input(shape=(SEQ, 1), dtype="float32", name="time_abs")
        delta_time = Input(shape=(SEQ, 1), dtype="float32", name="delta_time")
        pad_mask = Input(shape=(SEQ,), dtype="float32", name="pad_mask")

        # ---- Per-field embeddings ----
        # player and secondary_player share one Embedding table (weight-tied): same
        # entity, same representation, regardless of which role they play in the event.
        player_vocab_size = self.encoder.player_vocab.next_token
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

        # ---- Roster encoding across the sequence ----
        # One shared SequenceRosterEncoder applied to both rosters: weight-ties
        # home/away and encodes all timesteps in a single reshaped pass.
        home_vec = self.roster_encoder(home_roster)
        away_vec = self.roster_encoder(away_roster)

        # ---- Continuous projections ----
        t_abs = layers.Dense(16, name="time_abs_proj")(time_abs)
        t_delta = layers.Dense(16, name="delta_time_proj")(delta_time)

        # ---- Fusion ----
        x = layers.Concatenate(axis=-1, name="fusion_concat")(
            [*embs, home_vec, away_vec, t_abs, t_delta]
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

        # ---- Output heads (names must not collide with input field names) ----
        # dtype float32 keeps logits/regression stable under mixed_float16.
        event_logits = layers.Dense(event_vocab_size, dtype="float32", name="event_output")(x)
        time_delta = layers.Dense(1, activation="linear", dtype="float32", name="time_output")(x)

        inputs = {
            **cat_inputs,
            "home_roster": home_roster, "away_roster": away_roster,
            "time_abs": time_abs, "delta_time": delta_time, "pad_mask": pad_mask,
        }
        return keras.Model(
            inputs=inputs,
            outputs={"event_output": event_logits, "time_output": time_delta},
            name="EventTimeModel",
        )

    # =====================
    # --- Training      ---
    # =====================

    INPUT_KEYS = (
        "event", "player", "type", "result", "season", "secondary_player",
        "home_roster", "away_roster", "time_abs", "delta_time", "pad_mask",
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
        targets = {
            "event_output": split["event_target"],
            "time_output": split["time_target"],
        }
        # loss_mask zeroes the loss on PAD steps and on the final (no-next) step.
        mask = split["loss_mask"]
        sample_weights = {"event_output": mask, "time_output": mask}

        ds = tf.data.Dataset.from_tensor_slices((inputs, targets, sample_weights))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(mask), 1024), reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def configure_gpu(mixed_precision: bool = True):
        """
        Place training on CUDA when a GPU is visible. Enables per-GPU memory
        growth (so TF doesn't grab all VRAM up front) and, on GPU, the
        mixed_float16 policy (Tensor Cores -> faster training, less VRAM).

        Returns the list of visible GPUs (empty on a CPU-only build, e.g. native
        Windows TF >= 2.11, where the 4070 is invisible and training falls back
        to CPU). Output heads are forced to float32 so from_logits softmax stays
        numerically stable under mixed precision.
        """
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        if gpus and mixed_precision:
            keras.mixed_precision.set_global_policy("mixed_float16")
        return gpus

    def train(self, epochs=50, batch_size=64, lr=3e-4, time_loss_weight=0.25,
              patience=6, artifacts_root=DEFAULT_ARTIFACTS_ROOT,
              mixed_precision=True, jit_compile=False,
              report=True, run_name=None, reports_root=DEFAULT_REPORTS_ROOT):
        """
        Fit the Event/Time model on the preprocessed train split, validating on
        test. Multi-task masked loss: SparseCCE(from_logits) on the event head +
        MAE on the time head, with loss_mask as sample_weight so PAD/no-next
        steps contribute zero loss. Saves the trained model alongside the vocabs
        and norm stats (the full reusable artifact).

        When ``report`` is set (default), a standardized training/testing report
        is emitted to ``reports_root`` via the shared reporting layer: a
        self-contained HTML report with per-epoch loss/metric graphs plus a
        queryable Parquet data model (see the reporting package). ``run_name`` is
        an optional human label folded into the run id.

        Runs on CUDA automatically when a GPU is visible (see configure_gpu);
        otherwise falls back to CPU.
        """
        gpus = self.configure_gpu(mixed_precision=mixed_precision)
        print(f"Training on {'GPU x' + str(len(gpus)) if gpus else 'CPU (no visible GPU)'}")
        # Ensure the shared language is loaded + frozen (no-op if preprocess() just
        # ran in this session) so embedding sizes match the saved token IDs.
        if not self.encoder.player_vocab.frozen:
            self.encoder.load_all()
            self.encoder.freeze_all()
        if self.norm_stats is None and Path(NORM_STATS_PATH).exists():
            self.norm_stats = json.loads(Path(NORM_STATS_PATH).read_text(encoding="utf-8"))

        train_split = self._load_processed("train.npz")
        test_split = self._load_processed("test.npz")

        train_ds = self._make_dataset(train_split, batch_size, shuffle=True)
        val_ds = self._make_dataset(test_split, batch_size, shuffle=False)

        model = self.model()
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr, weight_decay=1e-4, clipnorm=1.0,
        )
        model.compile(
            optimizer=optimizer,
            loss={
                "event_output": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                "time_output": keras.losses.MeanAbsoluteError(),
            },
            loss_weights={"event_output": 1.0, "time_output": time_loss_weight},
            metrics={
                "event_output": [keras.metrics.SparseCategoricalAccuracy(name="acc")],
                "time_output": [keras.metrics.MeanAbsoluteError(name="mae")],
            },
            jit_compile=jit_compile,
        )

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True,
            ),
        ]

        # Standardized reporting: capture config/env/data/model, attach the
        # per-epoch callback, and emit the report in finally so even a crash
        # leaves a status="failed" record.
        collector = None
        if report:
            cfg = RunConfig(
                model_key=self.KEY, epochs_planned=epochs, batch_size=batch_size,
                lr=lr, time_loss_weight=time_loss_weight, patience=patience,
                mixed_precision=mixed_precision, jit_compile=jit_compile,
                arch={
                    "model_dim": self.model_dim,
                    "sequence_length": self.sequence_length,
                    "embed_dims": EMBED_DIMS,
                    "roster_dim": ROSTER_DIM,
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

        # Persist the full reusable artifact: model + weights + vocabs + norm stats.
        self.save_artifacts(model, root=artifacts_root)

        if collector is not None:
            # Final test pass on the (best-weights-restored) model for the report.
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
    #
    # Contract shared by every model wrapper (so ModelBundle can load them all the
    # same way): save_artifacts() writes the standard layout, from_artifacts()
    # rebuilds the graph and restores weights. See models/artifacts.py.

    def save_artifacts(self, model, root=DEFAULT_ARTIFACTS_ROOT) -> ModelArtifacts:
        """
        Persist the trained model under <root>/<KEY>/ using the shared layout:
        a full <KEY>.keras, a weights-only <KEY>.weights.h5, and norm_stats.json,
        plus the shared vocabs. Returns the resolved ModelArtifacts.
        """
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
        Reload a trained model from <root>/<KEY>/ (robust path: rebuild the graph
        from the frozen vocabs, then restore weights — no custom-object
        deserialization needed). Returns (instance, model) ready for inference.

        Pass `encoder` to point at a specific vocab dir; otherwise a default
        Encoder is created. Extra kwargs (e.g. path=, sequence_length=) flow to
        the EventTimeModel constructor.
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
