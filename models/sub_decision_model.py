"""
The rotation trigger: how many substitutions follow this opportunity, per team.

Substitutions used to be produced by a timer with a dial on it -- ``_schedule_stint`` sampled a
stint length per entering player and ``_fatigue_bias`` nudged the outgoing pick. Player minutes
are the single largest remaining box-score error, and a timer is why. This head replaces the
timer with a decision: at every position where NBA Rule 3 permits a substitution, predict how
many each side makes before play resumes (``0 / 1 / 2 / 3+``).

**Asked only where a substitution is legal**, which is what makes the number learnable. Over
every row a substitution follows about 1% of the time, and a head trained that way learns that
substitutions are rare; over legal opportunities the rate is 20-27% depending on era, which is
the real answer to "does anyone come off here". ``rotation_features.can_substitute`` defines
those positions for training and ``GameController.can_sub`` for rollout, from one rule, so the
head is never asked a question it did not learn -- see that function for the clauses.

Two outputs on one backbone rather than one call per side: the question is symmetric, both
answers come from the same history, and a single forward pass at each opportunity is what keeps
the rollout cost to one extra head call rather than two.
"""
from __future__ import annotations

import json

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from config import (
    MAX_SEQUENCE_LENGTH, ROSTER_SIZE, BENCH_SIZE, NORM_STATS_PATH,
    SEED, TEST_FRAC, HOLDOUT_FRAC, HOLDOUT_MANIFEST_NAME,
    MODEL_DIM, NUM_LAYERS, NUM_HEADS, FF_DIM, ROSTER_SAB_LAYERS,
)
from data_loading import resolve_partition
from models.artifacts import DEFAULT_ARTIFACTS_ROOT, warm_start_weights
from models.backbone import build_backbone
from models.event_time_model import (
    CATEGORICAL_FIELDS, EMBED_DIMS, ROSTER_COLS, ROSTER_DIM,
)
from models.norm_stats_io import load_norm_stats, save_norm_stats
from models.season_features import (
    SEASON_INPUT_KEYS, attach_recency_weights, apply_recency,
    merge_season_features, append_season_batches, make_season_inputs,
    season_team_projections,
)
from models.game_state_features import (
    GAME_STATE_INPUT_KEYS, merge_game_state_features, append_game_state_batches,
    make_game_state_inputs, game_state_projections,
)
from models.rotation_features import (
    BENCH_KEYS, NUM_ROSTER_SCALARS, ROSTER_STATE_KEYS, SUB_COUNT_CLASSES, SUB_DECISION_KEYS,
    append_rotation_batches, append_sub_decision_batches, bench_scalars, make_bench_inputs,
    make_rotation_inputs, merge_rotation_features, merge_sub_decisions, side_scalars,
)
from models.substitution_model import SubstitutionModel, _BASE_INPUT_KEYS
from reporting import ReportCollector, RunConfig
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT

# Own family file names, independent of the substitution head's sub_*.npz.
_PROCESSED = {"train": "subdec_train.npz", "test": "subdec_test.npz",
              "holdout": "subdec_holdout.npz"}

# Output names, one per side. The simulator asks for both from a single forward pass.
HOME_OUTPUT = "sub_count_home_output"
AWAY_OUTPUT = "sub_count_away_output"


class SubDecisionModel(SubstitutionModel):
    """Causal transformer head predicting the substitution count per side at an opportunity."""

    KEY = "sub_decision"

    @property
    def output_name(self) -> str:
        # A default for the shared plumbing; the simulator names the side it wants explicitly.
        return HOME_OUTPUT

    @property
    def INPUT_KEYS(self) -> tuple:
        # History plus the bench bundle. Deliberately NO next-step conditioning: the question is
        # asked about the position itself, not about an event already decided, so there is
        # nothing to condition on that history does not already carry.
        return (*_BASE_INPUT_KEYS, *BENCH_KEYS)

    # =====================
    # --- Preprocessing ---
    # =====================

    def preprocess(self, rebuild_vocabs=False, test_frac=TEST_FRAC,
                   holdout_frac=HOLDOUT_FRAC, seed=SEED,
                   game_partition=None, refit_norm_stats=True):
        """Build the sub-decision arrays and persist them.

        Same loader, encoding, delta-t normalization and partitioning as the rest of the chain,
        so the splits and norm stats line up. Adds three columns from
        ``rotation_features.merge_sub_decisions``: the opportunity mask and the two per-side
        counts.

        **The opening lineup is not augmented in.** ``SubstitutionModel`` synthesises ten
        opening substitutions so the incoming-pick head learns starters; this head answers "how
        many come off at this stoppage", and a tip-off is not a stoppage. Feeding it those rows
        would teach it that games open with five substitutions a side.
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

        enc = {f: df[f].apply(getattr(self.encoder, f"encode_{f}")).to_numpy()
               for f in CATEGORICAL_FIELDS}
        rosters = {name: np.stack(df[src].apply(self.encoder.encode_roster).to_numpy())
                   for name, src in ROSTER_COLS.items()}
        time = df["time"].to_numpy(dtype=np.float64)
        game_id = df["game_id"].to_numpy()
        delta = (df.groupby("game_id")["time"].diff().fillna(0).clip(lower=0)
                 .to_numpy(dtype=np.float64))

        train_games, test_games, holdout_games = resolve_partition(
            game_partition, game_id, seed, test_frac, holdout_frac,
        )
        train_mask = np.array([g in train_games for g in game_id])

        if refit_norm_stats:
            max_time = float(time[train_mask].max()) or 1.0
            train_delta = delta[train_mask]
            self.norm_stats = {"max_time": max_time,
                               "delta_mean": float(train_delta.mean()),
                               "delta_std": float(train_delta.std()) or 1.0}
        else:
            self.norm_stats = load_norm_stats(self.processed_dir, self.KEY)
            max_time = float(self.norm_stats["max_time"]) or 1.0
        delta_mean = float(self.norm_stats["delta_mean"])
        delta_std = float(self.norm_stats["delta_std"]) or 1.0

        cols = {
            **{f: enc[f] for f in CATEGORICAL_FIELDS},
            "home_roster": rosters["home_roster"],
            "away_roster": rosters["away_roster"],
            "time_abs": (time / max_time).astype(np.float32),
            "delta_time": ((delta - delta_mean) / delta_std).astype(np.float32),
        }
        merge_season_features(
            df, cols, rosters, self.encoder.encode_player("PAD"), train_mask, self.norm_stats,
            refit=refit_norm_stats,
        )
        merge_game_state_features(df, cols)
        merge_rotation_features(
            df, cols,
            encode_bench=lambda names: self.encoder.encode_roster(names, BENCH_SIZE))
        merge_sub_decisions(df, cols)

        train = self._build_split(cols, game_id, train_games)
        test = self._build_split(cols, game_id, test_games)
        holdout = self._build_split(cols, game_id, holdout_games)
        attach_recency_weights(
            [(train, train_games), (test, test_games), (holdout, holdout_games)], df, game_id)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        for name, split in (("train", train), ("test", test), ("holdout", holdout)):
            np.savez_compressed(self.processed_dir / _PROCESSED[name], **split)
        (self.processed_dir / HOLDOUT_MANIFEST_NAME).write_text(
            json.dumps(sorted(int(g) for g in holdout_games), indent=2), encoding="utf-8")
        if refit_norm_stats:
            save_norm_stats(self.processed_dir, self.KEY, self.norm_stats)

        opportunities = float(train["can_sub"].sum())
        made = float((train["subs_home"] + train["subs_away"])[train["can_sub"] > 0].sum())
        print(f"Preprocessed {len(train['can_sub'])} games -> {self.processed_dir} "
              f"({opportunities:.0f} opportunities, {made / max(opportunities * 2, 1):.3f} "
              f"substitutions per team per opportunity)", flush=True)
        return train, test

    def _build_split(self, cols, game_id, games) -> dict:
        """Pad/stack each game, adding the opportunity mask and the two count targets."""
        SEQ = self.sequence_length
        PAD_PLAYER = self.encoder.encode_player("PAD")
        pad_scalar = {
            "event": self.encoder.encode_event("PAD"),
            "player": PAD_PLAYER,
            "type": self.encoder.encode_type("PAD"),
            "result": self.encoder.encode_result("PAD"),
            "season": self.encoder.encode_season("PAD"),
            "secondary_player": self.encoder.encode_secondary_player("PAD"),
        }

        batches = {k: [] for k in (*CATEGORICAL_FIELDS, "home_roster", "away_roster",
                                   "time_abs", "delta_time", *SEASON_INPUT_KEYS,
                                   *GAME_STATE_INPUT_KEYS, *ROSTER_STATE_KEYS, *BENCH_KEYS,
                                   *SUB_DECISION_KEYS, "pad_mask")}

        for g in [g for g in np.unique(game_id) if g in games]:
            idx = np.where(game_id == g)[0][:SEQ]
            n = len(idx)

            for k in CATEGORICAL_FIELDS:
                buf = np.full((SEQ,), pad_scalar[k], dtype=np.int32)
                buf[:n] = cols[k][idx]
                batches[k].append(buf)
            for k in ("home_roster", "away_roster"):
                buf = np.full((SEQ, ROSTER_SIZE), PAD_PLAYER, dtype=np.int32)
                buf[:n] = cols[k][idx]
                batches[k].append(buf)
            for k in ("time_abs", "delta_time"):
                buf = np.zeros((SEQ, 1), dtype=np.float32)
                buf[:n, 0] = cols[k][idx]
                batches[k].append(buf)

            append_season_batches(batches, cols, idx, n, SEQ)
            append_game_state_batches(batches, cols, idx, n, SEQ)
            append_rotation_batches(batches, cols, idx, n, SEQ, PAD_PLAYER)
            append_sub_decision_batches(batches, cols, idx, n, SEQ)

            pad = np.zeros((SEQ,), dtype=np.float32)
            pad[:n] = 1.0
            batches["pad_mask"].append(pad)

        # A split can be empty (test_frac=0 in the tiny fixtures), and np.stack([]) raises.
        # Same guard EventTimeModel._build_split uses.
        return {k: np.stack(v) if v else np.empty((0,)) for k, v in batches.items()}

    # _load_processed is inherited: it takes a FILE NAME, and this head passes its own
    # _PROCESSED entries, the way every head with its own npz family does.

    # =====================
    # --- Model graph   ---
    # =====================

    def model(self, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=0.2):
        """Two softmaxes over ``SUB_COUNT_CLASSES``, one per side, on a shared backbone."""
        SEQ = self.sequence_length
        D = MODEL_DIM
        vocab = self.encoder.vocabs
        self.roster_encoder = self.build_roster_encoder(dropout=dropout)
        self.bench_encoder = self.build_bench_encoder(dropout=dropout)

        cat_inputs = {f: layers.Input(shape=(SEQ,), dtype="int32", name=f)
                      for f in CATEGORICAL_FIELDS}
        home_roster = layers.Input(shape=(SEQ, ROSTER_SIZE), dtype="int32", name="home_roster")
        away_roster = layers.Input(shape=(SEQ, ROSTER_SIZE), dtype="int32", name="away_roster")
        time_abs = layers.Input(shape=(SEQ, 1), dtype="float32", name="time_abs")
        delta_time = layers.Input(shape=(SEQ, 1), dtype="float32", name="delta_time")
        rest_home, rest_away, team_inputs = make_season_inputs(SEQ)
        game_state_inputs = make_game_state_inputs(SEQ)
        rotation_inputs = make_rotation_inputs(SEQ)
        bench_inputs = make_bench_inputs(SEQ)
        pad_mask = layers.Input(shape=(SEQ,), dtype="float32", name="pad_mask")

        player_emb_layer = layers.Embedding(
            self.encoder.player_vocab.next_token, EMBED_DIMS["player"], name="emb_player")
        embs = []
        for f in CATEGORICAL_FIELDS:
            if f in ("player", "secondary_player"):
                embs.append(player_emb_layer(cat_inputs[f]))
            else:
                embs.append(layers.Embedding(vocab[f].next_token, EMBED_DIMS[f],
                                             name=f"emb_{f}")(cat_inputs[f]))

        home_vec = self.roster_encoder(
            [home_roster, *side_scalars(rest_home, rotation_inputs, "home")])
        away_vec = self.roster_encoder(
            [away_roster, *side_scalars(rest_away, rotation_inputs, "away")])
        bench_home_vec = self.bench_encoder(
            [bench_inputs["bench_home"], *bench_scalars(bench_inputs, "home")])
        bench_away_vec = self.bench_encoder(
            [bench_inputs["bench_away"], *bench_scalars(bench_inputs, "away")])

        t_abs = layers.Dense(16, name="time_abs_proj")(time_abs)
        t_delta = layers.Dense(16, name="delta_time_proj")(delta_time)
        t_team = season_team_projections(team_inputs)
        t_gs = game_state_projections(game_state_inputs)

        x = build_backbone(
            [*embs, home_vec, away_vec, bench_home_vec, bench_away_vec,
             t_abs, t_delta, *t_team, *t_gs],
            pad_mask, seq_len=SEQ, d_model=D,
            num_layers=num_layers, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout,
        )

        home_logits = layers.Dense(SUB_COUNT_CLASSES, dtype="float32", name=HOME_OUTPUT)(x)
        away_logits = layers.Dense(SUB_COUNT_CLASSES, dtype="float32", name=AWAY_OUTPUT)(x)

        inputs = {
            **cat_inputs,
            "home_roster": home_roster, "away_roster": away_roster,
            "time_abs": time_abs, "delta_time": delta_time,
            "rest_home": rest_home, "rest_away": rest_away, **team_inputs,
            **game_state_inputs, **rotation_inputs, **bench_inputs,
            "pad_mask": pad_mask,
        }
        return keras.Model(inputs=inputs,
                           outputs={HOME_OUTPUT: home_logits, AWAY_OUTPUT: away_logits},
                           name="SubDecisionModel")

    # =====================
    # --- Training      ---
    # =====================

    def _make_dataset(self, split: dict, batch_size: int, shuffle: bool) -> tf.data.Dataset:
        """Yield (inputs, targets, sample_weights), with the loss masked to real opportunities.

        ``can_sub`` is the mask and it is the whole point: weighting every row equally would
        train the head on ~400 positions a game of which ~99 are questions anyone ever asks, and
        the rate it learned would be the diluted one.
        """
        inputs = {k: split[k] for k in self.INPUT_KEYS}
        targets = {HOME_OUTPUT: split["subs_home"].astype(np.int32),
                   AWAY_OUTPUT: split["subs_away"].astype(np.int32)}
        mask = apply_recency(split["can_sub"], split)
        sample_weights = {HOME_OUTPUT: mask, AWAY_OUTPUT: mask}

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
        """Fit the sub-decision head, validating on the test split.

        Cross-entropy on both sides against the opportunity mask, with accuracy as the reported
        metric. Accuracy is a weak read here -- predicting zero everywhere scores ~82% -- so the
        number that matters is the post-train substitutions-per-game check in the rotation gate,
        not this.
        """
        self.configure_gpu(mixed_precision=mixed_precision)
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
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0, warmup_target=lr, warmup_steps=warmup_steps,
            decay_steps=max(1, total_steps - warmup_steps), alpha=lr_alpha,
        )
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0),
            loss={HOME_OUTPUT: loss, AWAY_OUTPUT: loss},
            # weighted_metrics (NOT metrics): only weighted_metrics receive the sample_weight,
            # so accuracy reflects real opportunities rather than every padded position.
            weighted_metrics={
                HOME_OUTPUT: [keras.metrics.SparseCategoricalAccuracy(name="acc")],
                AWAY_OUTPUT: [keras.metrics.SparseCategoricalAccuracy(name="acc")],
            },
            jit_compile=jit_compile,
        )

        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True)]

        collector = None
        if report:
            cfg = RunConfig(
                model_key=self.KEY, epochs_planned=epochs, batch_size=batch_size,
                lr=lr, time_loss_weight=0.0, patience=patience,
                mixed_precision=mixed_precision, jit_compile=jit_compile,
                arch={
                    "model_dim": self.model_dim,
                    "sequence_length": self.sequence_length,
                    "num_layers": num_layers, "num_heads": num_heads, "ff_dim": ff_dim,
                    "dropout": dropout, "embed_dims": EMBED_DIMS, "roster_dim": ROSTER_DIM,
                    "lr_schedule": "warmup_cosine", "warmup_epochs": warmup_epochs,
                    "lr_alpha": lr_alpha,
                    "sub_count_classes": SUB_COUNT_CLASSES,
                    "asked_at": "rule_3_section_v_opportunities",
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
            history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                                callbacks=callbacks)
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
                final_test_metrics={k: float(v) for k, v in test_metrics.items()})
            print(f"Report '{self.KEY}/{arts.run_id}' -> {arts.run_dir.resolve()}")
        return model, history
