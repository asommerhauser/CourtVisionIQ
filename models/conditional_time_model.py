"""
Conditional time head — predict the inter-event Δt *conditioned on the event we are about to emit*.

The EventTimeModel time head predicts ``E[Δt | history]`` — the **marginal** gap, conditioned only
on the past, never on which event comes next (event and time are independent heads off one hidden
state). That is fine for aggregate pace (the marginal summed over a correct event head reproduces
total game time) but wrong for *per-play* timing: the clock advance does not follow the specific
play. This head closes that gap. It mirrors the existing conditional chain:

    EventTimeModel -> PlayerModel (actor) -> [ConditionalTimeModel] (Δt) -> type/result heads

At inference the controller samples the next event, then the actor, then asks **this** head how
long until that event — so Δt follows the play instead of an average. It is a near-twin of
:class:`SubstitutionModel`: same causal-transformer backbone, embeddings, roster set-encoder,
reporting, and artifact contract. It differs in fixed (non-spec) ways:

  * it conditions on ``next_event`` + ``next_player`` (the decided actor) — **not**
    ``next_delta_time`` (Δt is what it predicts);
  * its output is a single regression scalar ``time_output`` (standardized Δt, reusing the shared
    ``delta_mean`` / ``delta_std``), trained with masked **MSE** (mean-targeting, like the
    EventTimeModel time head — the rollout fills a fixed 48 minutes, so a median-targeting MAE
    would pack too many possessions);
  * it learns on the **raw** event stream (no synthesized opening subs) with the same loss mask as
    the EventTimeModel time head: PAD, the last (no-next) row, and any step whose next event is a
    substitution (subs are injected by the rotation scheduler, never timed by this head).

Own preprocess writes ``condtime_{train,test,holdout}.npz``.
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from keras import layers, Input

from config import NORM_STATS_PATH, NUM_LAYERS, NUM_HEADS, FF_DIM
from data_loading import resolve_partition
from models.norm_stats_io import load_norm_stats, save_norm_stats
from models.artifacts import DEFAULT_ARTIFACTS_ROOT, warm_start_weights
from models.event_time_model import (
    AddPositionalEmbedding,
    KeyPaddingMask,
    DeltaSecondsMAE,
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
from models.substitution_model import SubstitutionModel, SUB_EVENT, _BASE_INPUT_KEYS
from reporting import ReportCollector, RunConfig
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT

# Own family file names (independent of the substitution/stint heads).
_PROCESSED = {"train": "condtime_train.npz", "test": "condtime_test.npz",
              "holdout": "condtime_holdout.npz"}


class ConditionalTimeModel(SubstitutionModel):
    """Causal transformer head regressing the next inter-event Δt, conditioned on (event, actor)."""

    KEY = "event_time_cond"

    @property
    def output_name(self) -> str:
        return "time_output"

    @property
    def INPUT_KEYS(self) -> tuple:
        # Base history + the decided next event and its actor. NO next_delta_time (Δt is the target).
        return (*_BASE_INPUT_KEYS, "next_event", "next_player")

    # =====================
    # --- Preprocessing ---
    # =====================

    def preprocess(self, rebuild_vocabs=False, test_frac=None, holdout_frac=None, seed=None,
                   game_partition=None, refit_norm_stats=True):
        """Build the conditional-time tensor arrays (raw event stream, no opening subs) + persist.

        Same loader / encoding / Δt normalization / train-val-holdout split as the rest of the
        chain (same defaults pulled from config, so partitions + norm_stats line up). Unlike the
        substitution/stint heads, it does **not** synthesize opening subs — it learns timing on the
        real play-by-play, exactly like the EventTimeModel time head. ``game_partition`` /
        ``refit_norm_stats`` (curriculum / full-train use): see ``EventTimeModel.preprocess``.
        """
        from config import TEST_FRAC, HOLDOUT_FRAC, SEED
        test_frac = TEST_FRAC if test_frac is None else test_frac
        holdout_frac = HOLDOUT_FRAC if holdout_frac is None else holdout_frac
        seed = SEED if seed is None else seed

        df = self._load_all()  # raw cleaned stream (no opening-sub augmentation)

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

        train = self._build_split(cols, game_id, train_games)
        test = self._build_split(cols, game_id, test_games)
        holdout = self._build_split(cols, game_id, holdout_games)
        attach_recency_weights(
            [(train, train_games), (test, test_games), (holdout, holdout_games)], df, game_id)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.processed_dir / _PROCESSED["train"], **train)
        np.savez_compressed(self.processed_dir / _PROCESSED["test"], **test)
        np.savez_compressed(self.processed_dir / _PROCESSED["holdout"], **holdout)
        if refit_norm_stats:
            save_norm_stats(self.processed_dir, self.KEY, self.norm_stats)

        print(f"Preprocessed {len(train_games)} train / {len(test_games)} test / "
              f"{len(holdout_games)} holdout games -> {self.processed_dir} "
              f"(SEQ={self.sequence_length}, conditional-time arrays)")
        return train, test

    def _build_split(self, cols, game_id, games) -> dict:
        """Pad/truncate each game to sequence_length and stack into batched arrays.

        Base history + the next-step conditioning (``next_event`` / ``next_player``) and the
        continuous next-step ``next_time_target`` (the standardized Δt of the next row). ``loss_mask``
        is 1 for real rows with a valid next step; the per-step substitution mask is applied at
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
        next_pad = {"next_event": PAD_EVENT, "next_player": PAD_PLAYER}
        next_src = {"next_event": "event", "next_player": "player"}

        keys_1d = CATEGORICAL_FIELDS
        keys_roster = ["home_roster", "away_roster"]
        keys_cont = ["time_abs", "delta_time"]
        keys_next_cat = ["next_event", "next_player"]

        batches = {k: [] for k in (*keys_1d, *keys_roster, *keys_cont, *SEASON_INPUT_KEYS,
                                   *keys_next_cat, "next_time_target", "pad_mask", "loss_mask")}

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
                buf = np.full((SEQ, 5), PAD_PLAYER, dtype=np.int32)
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
            next_time = np.zeros((SEQ, 1), dtype=np.float32)
            if n > 1:
                next_time[: n - 1, 0] = cols["delta_time"][idx][1:]
            batches["next_time_target"].append(next_time)

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
        """Build the causal conditional-time transformer.

        Inputs: the Event/Time history inputs plus ``next_event`` and ``next_player`` (the decided
        actor). Output: ``time_output`` — a single linear regression scalar (standardized Δt),
        mirroring EventTimeModel's time head but conditioned on the event we are about to emit.
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
        home_roster = Input(shape=(SEQ, 5), dtype="int32", name="home_roster")
        away_roster = Input(shape=(SEQ, 5), dtype="int32", name="away_roster")
        time_abs = Input(shape=(SEQ, 1), dtype="float32", name="time_abs")
        delta_time = Input(shape=(SEQ, 1), dtype="float32", name="delta_time")
        rest_home, rest_away, team_inputs = make_season_inputs(SEQ)
        next_event = Input(shape=(SEQ,), dtype="int32", name="next_event")
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

        # ---- Conditioning embeddings (the decided next event + its actor) ----
        cond_vecs = [
            layers.Embedding(event_vocab_size, EMBED_DIMS["event"], name="emb_next_event")(next_event),
            player_emb_layer(next_player),  # the actor of the event we are timing
        ]

        # ---- Roster encoding across the sequence (shared home/away, with per-player rest) ----
        home_vec = self.roster_encoder([home_roster, rest_home])
        away_vec = self.roster_encoder([away_roster, rest_away])

        # ---- Continuous projections ----
        t_abs = layers.Dense(16, name="time_abs_proj")(time_abs)
        t_delta = layers.Dense(16, name="delta_time_proj")(delta_time)
        t_team = season_team_projections(team_inputs)

        # ---- Fusion ----
        x = layers.Concatenate(axis=-1, name="fusion_concat")(
            [*embs, *cond_vecs, home_vec, away_vec, t_abs, t_delta, *t_team]
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

        # ---- Output head: single linear regression scalar (float32 under mixed_float16) ----
        time_delta = layers.Dense(1, activation="linear", dtype="float32", name=self.output_name)(x)

        inputs = {
            **cat_inputs,
            "home_roster": home_roster, "away_roster": away_roster,
            "time_abs": time_abs, "delta_time": delta_time,
            "rest_home": rest_home, "rest_away": rest_away, **team_inputs,
            "next_event": next_event, "next_player": next_player,
            "pad_mask": pad_mask,
        }
        return keras.Model(inputs=inputs, outputs={self.output_name: time_delta},
                           name="ConditionalTimeModel")

    def _make_dataset(self, split: dict, batch_size: int, shuffle: bool) -> tf.data.Dataset:
        """Yield (inputs, targets, sample_weights); MSE on Δt, masked like the EventTimeModel time head.

        ``loss_mask`` zeroes PAD / no-next steps; we additionally zero every step whose next event is
        a substitution (subs are injected by the rotation scheduler at inference, never timed here),
        exactly mirroring ``EventTimeModel._make_dataset``.
        """
        inputs = {k: split[k] for k in self.INPUT_KEYS}
        targets = {self.output_name: split["next_time_target"]}

        sub_id = self.encoder.encode_event(SUB_EVENT)
        mask = (split["loss_mask"] * (split["next_event"] != sub_id)).astype(np.float32)
        mask = apply_recency(mask, split)
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
        """Fit the conditional-time head on the preprocessed train split, validating on test.

        Masked **MSE** on the single regression head over non-substitution next steps (mean-targeting,
        matching the EventTimeModel time head's rationale). Saves the model alongside the vocabs +
        norm stats and emits the standardized report.
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
        delta_std = float((self.norm_stats or {}).get("delta_std", 1.0) or 1.0)
        model.compile(
            optimizer=optimizer,
            loss={self.output_name: keras.losses.MeanSquaredError()},
            # weighted_metrics (NOT metrics): only weighted_metrics get the sample_weight mask,
            # so mae is over real conditioned rows, not every padded position.
            weighted_metrics={self.output_name: [
                keras.metrics.MeanAbsoluteError(name="mae"),
                DeltaSecondsMAE(delta_std, name="mae_sec"),
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
                lr=lr, time_loss_weight=1.0, patience=patience,  # single regression head
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
                    "target_field": "delta_time",
                    "condition_fields": ["next_event", "next_player"],
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
