"""
Stint-length head — predict *how long an entering player will stay on the floor*.

Where ``SubstitutionModel`` predicts ``secondary_player`` (the player coming **in**), this
model predicts a single scalar: the **realized stint length** (game-seconds on the floor) of
that incoming player. It is the timing brain of the rotation overhaul — the
:class:`~simulation.controller.GameController` samples a length when a player enters,
schedules their exit at ``clock + length``, and at each dead ball pulls anyone past their
scheduled exit. That replaces leaving substitution *timing* to the event head.

    EventTimeModel -> PlayerModel (outgoing) -> SubstitutionModel (incoming) -> [StintLengthModel] (how long)

It is a near-twin of :class:`SubstitutionModel`: same causal-transformer backbone, embeddings,
roster set-encoder, reporting, and artifact contract, and the **same** synthesized opening
subs (so starter stints are learned too). It differs in three fixed ways:

  * it conditions on the **fully resolved** substitution — ``next_player`` (outgoing) *and*
    ``next_secondary_player`` (incoming) — because we predict the incoming player's stint;
  * its output is a single regression scalar ``stint_output`` (mirroring EventTimeModel's
    ``time_output`` head), trained with masked MAE on substitution rows;
  * the target is computed in preprocess by walking each game forward: an entering player's
    stint = (their next exit) - (their entry), with players still on court at the final
    whistle right-censored to the game's last event time.

Targets are stored as standardized ``log1p(seconds)``; the per-model ``norm_stats.json`` gains
``stint_log_mean`` / ``stint_log_std`` (extra keys alongside the shared time/rest stats). Own
preprocess writes ``stint_{train,test,holdout}.npz``.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, Input

from config import (
    NORM_STATS_PATH, ROSTER_SIZE, SEED, TEST_FRAC, HOLDOUT_FRAC, HOLDOUT_MANIFEST_NAME,
    NUM_LAYERS, NUM_HEADS, FF_DIM,
)
from data_loading import resolve_partition
from models.norm_stats_io import load_norm_stats, save_norm_stats
from models.artifacts import DEFAULT_ARTIFACTS_ROOT, warm_start_weights
from models.event_time_model import (
    AddPositionalEmbedding,
    KeyPaddingMask,
    EMBED_DIMS,
    ROSTER_DIM,
    CATEGORICAL_FIELDS,
    ROSTER_COLS,
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
from models.substitution_model import (
    SubstitutionModel, SUB_EVENT, OUTGOING_FIELD, INCOMING_FIELD, _BASE_INPUT_KEYS,
)
from reporting import ReportCollector, RunConfig
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT

# Own family file names (independent of the substitution head's sub_*.npz).
_PROCESSED = {"train": "stint_train.npz", "test": "stint_test.npz", "holdout": "stint_holdout.npz"}


@keras.saving.register_keras_serializable(package="cviq")
class StintSecondsMAE(keras.metrics.Mean):
    """Masked MAE in real game-seconds, inverting the standardized log1p(stint) target.

    The head trains on ``(log1p(seconds) - stint_log_mean) / stint_log_std``, so its native
    ``mae`` reads in standardized log space (a "0.5" there is ~0.5 std of log-stint, not half a
    second). This metric maps both prediction and target back to seconds via
    ``expm1(x * std + mean)`` before taking the absolute error, giving a number that reads in
    plain seconds. It reuses the same substitution-row ``sample_weight`` mask as the loss, so
    only real sub rows count.
    """

    def __init__(self, log_mean, log_std, name="mae_seconds", **kwargs):
        super().__init__(name=name, **kwargs)
        self.log_mean = float(log_mean)
        self.log_std = float(log_std)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        true_sec = tf.math.expm1(y_true * self.log_std + self.log_mean)
        pred_sec = tf.math.expm1(y_pred * self.log_std + self.log_mean)
        values = tf.reduce_mean(tf.abs(pred_sec - true_sec), axis=-1)  # (B, SEQ)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, values.dtype)
        return super().update_state(values, sample_weight=sample_weight)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(log_mean=self.log_mean, log_std=self.log_std)
        return cfg


class StintLengthModel(SubstitutionModel):
    """Causal transformer head regressing an entering player's realized stint length."""

    KEY = "stint_length"

    @property
    def output_name(self) -> str:
        return "stint_output"

    @property
    def INPUT_KEYS(self) -> tuple:
        # Base history + always-on conditioning + the fully-decided sub (outgoing + incoming).
        return (*_BASE_INPUT_KEYS, "next_event", "next_delta_time",
                "next_player", "next_secondary_player")

    # =====================
    # --- Stint targets ---
    # =====================

    def _stint_seconds_per_row(self, df: pd.DataFrame) -> np.ndarray:
        """Realized stint seconds for each substitution(-in) row; 0 on every other row.

        Walks each game in order tracking, per player, the time they last entered (a
        ``substitution`` row's ``secondary_player`` — including the synthesized opening subs at
        time 0). When that player next appears as the **outgoing** ``player`` of a later sub,
        their stint = ``exit_time - entry_time`` is written back onto the row where they
        entered (the row we condition on). Players still on the floor at the final whistle are
        right-censored to the game's last event time and treated as observed — a minor
        underestimate that is fine for driving a scheduler.
        """
        out = np.zeros(len(df), dtype=np.float64)
        for _, game in df.groupby("game_id", sort=False):
            idx = game.index.to_numpy()
            times = game["time"].to_numpy(dtype=np.float64)
            events = game["event"].to_numpy()
            players = game["player"].to_numpy()           # outgoing
            incomings = game["secondary_player"].to_numpy()  # incoming
            last_time = float(times[-1]) if len(times) else 0.0
            current: dict[str, tuple[int, float]] = {}  # player -> (df-index, entry_time)
            for k in range(len(idx)):
                if events[k] != SUB_EVENT:
                    continue
                outgoing = players[k]
                if outgoing in current:                  # close the outgoing player's stint
                    row_idx, t_in = current.pop(outgoing)
                    out[row_idx] = times[k] - t_in
                incoming = incomings[k]                  # open the incoming player's stint
                current[incoming] = (int(idx[k]), times[k])
            for _player, (row_idx, t_in) in current.items():  # still on court at the whistle
                out[row_idx] = last_time - t_in
        return np.clip(out, 0.0, None)

    # =====================
    # --- Preprocessing ---
    # =====================

    def preprocess(self, rebuild_vocabs=False, test_frac=TEST_FRAC,
                   holdout_frac=HOLDOUT_FRAC, seed=SEED,
                   game_partition=None, refit_norm_stats=True):
        """Build the stint-length tensor arrays (with synthesized opening subs) + persist.

        Same loader / encoding / Δt normalization / train-val-holdout split as the rest of the
        chain (same ``seed`` + fracs, so partitions + norm_stats line up). Adds the regression
        target: standardized ``log1p`` realized stint seconds per substitution row, with
        ``stint_log_mean`` / ``stint_log_std`` folded into ``norm_stats``. Writes
        ``stint_{train,test,holdout}.npz``.

        ``game_partition`` / ``refit_norm_stats`` (curriculum use): see
        ``EventTimeModel.preprocess``. With ``refit_norm_stats=False`` the log-stint scaling
        (``stint_log_mean`` / ``stint_log_std``) is loaded too, so the regression target stays in
        the same units the warm-started head was trained against.
        """
        df = self._load_all()
        df = self._augment_with_opening_subs(df).reset_index(drop=True)

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

        train_games, test_games, holdout_games = resolve_partition(
            game_partition, game_id, seed, test_frac, holdout_frac,
        )
        train_mask = np.array([g in train_games for g in game_id])

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

        # --- Regression target: standardized log1p(stint seconds) on substitution rows. ---
        # Scaling is recomputed from train (refit) or loaded (staged) so it matches the
        # warm-started head's units.
        is_sub = enc["event"] == self.encoder.encode_event(SUB_EVENT)
        log_stint = np.log1p(self._stint_seconds_per_row(df))
        if refit_norm_stats:
            train_sub = train_mask & is_sub
            stint_log_mean = float(log_stint[train_sub].mean()) if train_sub.any() else 0.0
            stint_log_std = (float(log_stint[train_sub].std()) if train_sub.any() else 1.0) or 1.0
        else:
            stint_log_mean = float(self.norm_stats["stint_log_mean"])
            stint_log_std = float(self.norm_stats["stint_log_std"]) or 1.0
        stint_target = ((log_stint - stint_log_mean) / stint_log_std).astype(np.float32)
        stint_target[~is_sub] = 0.0  # non-sub rows are masked out of the loss anyway

        cols = {
            **{f: enc[f] for f in CATEGORICAL_FIELDS},
            "home_roster": rosters["home_roster"],
            "away_roster": rosters["away_roster"],
            "time_abs": time_abs,
            "delta_time": delta_norm,
            "stint_target": stint_target,
        }
        merge_season_features(
            df, cols, rosters, self.encoder.encode_player("PAD"), train_mask, self.norm_stats,
            refit=refit_norm_stats,
        )
        self.norm_stats["stint_log_mean"] = stint_log_mean
        self.norm_stats["stint_log_std"] = stint_log_std

        train = self._build_split(cols, game_id, train_games)
        test = self._build_split(cols, game_id, test_games)
        holdout = self._build_split(cols, game_id, holdout_games)
        attach_recency_weights(
            [(train, train_games), (test, test_games), (holdout, holdout_games)], df, game_id)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.processed_dir / _PROCESSED["train"], **train)
        np.savez_compressed(self.processed_dir / _PROCESSED["test"], **test)
        np.savez_compressed(self.processed_dir / _PROCESSED["holdout"], **holdout)
        (self.processed_dir / HOLDOUT_MANIFEST_NAME).write_text(
            json.dumps(sorted(int(g) for g in holdout_games), indent=2), encoding="utf-8"
        )
        if refit_norm_stats:
            save_norm_stats(self.processed_dir, self.KEY, self.norm_stats)

        print(f"Preprocessed {len(train_games)} train / {len(test_games)} test / "
              f"{len(holdout_games)} holdout games -> {self.processed_dir} "
              f"(SEQ={self.sequence_length}, stint-length arrays w/ opening subs)")
        return train, test

    def _build_split(self, cols, game_id, games) -> dict:
        """Pad/truncate each game to sequence_length and stack into batched arrays.

        Same base + chain layout as SubstitutionModel, but ``next_secondary_player`` (incoming)
        is a **conditioning input** here and the target is the continuous, next-step
        ``next_stint_target`` (the incoming player's realized stint). ``loss_mask`` is 1 for
        real rows with a valid next step; the per-event (substitution) mask is applied at
        dataset time.
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
                                   *keys_next_cat, "next_delta_time", "next_stint_target",
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
            next_stint = np.zeros((SEQ, 1), dtype=np.float32)
            if n > 1:
                next_delta[: n - 1, 0] = cols["delta_time"][idx][1:]
                next_stint[: n - 1, 0] = cols["stint_target"][idx][1:]
            batches["next_delta_time"].append(next_delta)
            batches["next_stint_target"].append(next_stint)

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

    def model(self, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=0.2):
        """Build the causal stint-length transformer.

        Inputs: the Event/Time history inputs, the always-on ``next_event`` / ``next_delta_time``
        conditioning, plus ``next_player`` (outgoing) and ``next_secondary_player`` (incoming).
        Output: ``stint_output`` — a single linear regression scalar (standardized log-stint),
        mirroring EventTimeModel's time head.
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
        next_event = Input(shape=(SEQ,), dtype="int32", name="next_event")
        next_delta_time = Input(shape=(SEQ, 1), dtype="float32", name="next_delta_time")
        next_player = Input(shape=(SEQ,), dtype="int32", name="next_player")
        next_secondary_player = Input(shape=(SEQ,), dtype="int32", name="next_secondary_player")
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

        # ---- Conditioning embeddings (the fully decided sub: outgoing + incoming) ----
        cond_vecs = [
            layers.Embedding(event_vocab_size, EMBED_DIMS["event"], name="emb_next_event")(next_event),
            player_emb_layer(next_player),             # outgoing
            player_emb_layer(next_secondary_player),   # incoming (whose stint we predict)
        ]

        # ---- Roster encoding across the sequence (shared home/away, with per-player rest) ----
        home_vec = self.roster_encoder([home_roster, rest_home])
        away_vec = self.roster_encoder([away_roster, rest_away])

        # ---- Continuous projections ----
        t_abs = layers.Dense(16, name="time_abs_proj")(time_abs)
        t_delta = layers.Dense(16, name="delta_time_proj")(delta_time)
        t_next_delta = layers.Dense(16, name="next_delta_time_proj")(next_delta_time)
        t_team = season_team_projections(team_inputs)

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

        # ---- Output head: a single linear regression scalar (float32 under mixed_float16) ----
        stint = layers.Dense(1, activation="linear", dtype="float32", name=self.output_name)(x)

        inputs = {
            **cat_inputs,
            "home_roster": home_roster, "away_roster": away_roster,
            "time_abs": time_abs, "delta_time": delta_time,
            "rest_home": rest_home, "rest_away": rest_away, **team_inputs,
            "next_event": next_event, "next_delta_time": next_delta_time,
            "next_player": next_player, "next_secondary_player": next_secondary_player,
            "pad_mask": pad_mask,
        }
        return keras.Model(inputs=inputs, outputs={self.output_name: stint},
                           name="StintLengthModel")

    def _make_dataset(self, split: dict, batch_size: int, shuffle: bool) -> tf.data.Dataset:
        """Yield (inputs, targets, sample_weights), with the loss masked to substitution rows.

        ``loss_mask`` zeroes PAD / no-next steps; we additionally zero every step whose decided
        event is not ``substitution`` so the head only regresses stint length on sub placements
        (including the synthesized opening subs).
        """
        inputs = {k: split[k] for k in self.INPUT_KEYS}
        targets = {self.output_name: split["next_stint_target"]}

        event_id = self.encoder.encode_event(SUB_EVENT)
        event_mask = (split["next_event"] == event_id).astype(np.float32)
        mask = apply_recency(split["loss_mask"] * event_mask, split)
        sample_weights = {self.output_name: mask}

        ds = tf.data.Dataset.from_tensor_slices((inputs, targets, sample_weights))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(mask), 1024), reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def train(self, epochs=50, batch_size=64, lr=3e-4,
              patience=10, artifacts_root=DEFAULT_ARTIFACTS_ROOT,
              mixed_precision=True, jit_compile=False,
              num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=0.2,
              warmup_epochs=1, lr_alpha=0.05,
              report=True, run_name=None, reports_root=DEFAULT_REPORTS_ROOT,
              init_weights_root=None):
        """Fit the stint-length head on the preprocessed train split, validating on test.

        Masked MAE on the single regression head over substitution rows (the
        substitution-restricted loss_mask as sample_weight). Saves the model alongside the
        vocabs + norm stats and emits the standardized report.
        """
        gpus = self.configure_gpu(mixed_precision=mixed_precision)
        print(f"Training '{self.KEY}' on {'GPU x' + str(len(gpus)) if gpus else 'CPU (no visible GPU)'}")
        if not self.encoder.player_vocab.frozen:
            self.encoder.load_all()
            self.encoder.freeze_all()
        if self.norm_stats is None and Path(NORM_STATS_PATH).exists():
            self.norm_stats = json.loads(Path(NORM_STATS_PATH).read_text(encoding="utf-8"))
        # The seconds metric needs the log-stint scaling; load the per-model stats if the
        # currently-held norm_stats predate it (e.g. only the shared file was loaded above).
        if not self.norm_stats or "stint_log_mean" not in self.norm_stats:
            self.norm_stats = load_norm_stats(self.processed_dir, self.KEY)
        stint_log_mean = float(self.norm_stats["stint_log_mean"])
        stint_log_std = float(self.norm_stats["stint_log_std"]) or 1.0

        train_split = self._load_processed(_PROCESSED["train"])
        test_split = self._load_processed(_PROCESSED["test"])

        train_ds = self._make_dataset(train_split, batch_size, shuffle=True)
        val_ds = self._make_dataset(test_split, batch_size, shuffle=False)

        model = self.model(num_layers=num_layers, num_heads=num_heads,
                           ff_dim=ff_dim, dropout=dropout)
        model.summary()
        warm_start_weights(model, self.KEY, init_weights_root)

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
            loss={self.output_name: keras.losses.MeanAbsoluteError()},
            # weighted_metrics (NOT metrics): only weighted_metrics receive the sample_weight
            # mask, so mae reflects real rows, not every padded position. ``mae`` is in
            # standardized log-stint units; ``mae_seconds`` inverts the transform to plain seconds.
            weighted_metrics={self.output_name: [
                keras.metrics.MeanAbsoluteError(name="mae"),
                StintSecondsMAE(stint_log_mean, stint_log_std, name="mae_seconds"),
            ]},
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
                lr=lr, time_loss_weight=0.0, patience=patience,  # single regression head
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
                    "target_field": "stint_length",
                    "condition_fields": [OUTGOING_FIELD, INCOMING_FIELD],
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
