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
from data_loading import load_all_cleaned, resolve_partition
from encoder.encoder import Encoder
from models.artifacts import ModelArtifacts, DEFAULT_ARTIFACTS_ROOT, warm_start_weights
from models.norm_stats_io import load_norm_stats, save_norm_stats
from models.roster_set_encoder import (
    RosterSetEncoder,
    RosterEncoderParams,
    SequenceRosterEncoder,
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
from reporting import ReportCollector, RunConfig
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT

# Per-field embedding dimensions (categoricals never enter the model as raw scalars).
# player bumped 128 -> 192 for train 2 (richer per-player representation, weight-tied with
# secondary_player); other fields are low-cardinality and stay put.
EMBED_DIMS = {"event": 32, "player": 192, "type": 32, "result": 16, "season": 16}
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


@keras.saving.register_keras_serializable(package="cviq")
class ApplyAvailabilityMask(layers.Layer):
    """Mask player logits to the per-game available set before the loss/softmax.

    A head over the player vocab (Player / Substitution) emits logits for *every* player,
    so its cross-entropy denominator spans the whole league and the gradient learns global
    appearance *volume*. Adding a large-negative bias to the logits of players who were not
    available that game restricts the softmax to the co-available set, so the head learns
    *relative* appearance frequency among players who could actually have been picked.

    A registered layer (not a Lambda) so the full .keras model reloads under Keras 3 safe
    mode without custom code execution, mirroring KeyPaddingMask / AddPositionalEmbedding.

    Inputs: ``logits`` (B, SEQ, V) and ``avail_mask`` (B, V) with 1.0 for available players
    and 0.0 otherwise. The mask is per-game (constant across the sequence) and broadcasts
    over the SEQ axis.
    """

    NEG = -1e9

    def call(self, logits, avail_mask):
        bias = (1.0 - tf.cast(avail_mask, logits.dtype)) * self.NEG  # (B, V)
        return logits + tf.expand_dims(bias, axis=1)                 # broadcast over SEQ

    def compute_output_shape(self, logits_shape, avail_mask_shape=None):
        return logits_shape


@keras.saving.register_keras_serializable(package="cviq")
class OnCourtCandidateMask(layers.Layer):
    """Mask player logits to the row's on-court candidate set (train/inference legality parity).

    The Player head picks among the players actually on the floor and the Substitution head
    picks off the bench, but both emit logits over the whole player vocab — historically the
    legality restriction was applied only at sampling time, so the training softmax normalized
    over a much larger set (the whole league, or the ~25-player game pool under
    ``ApplyAvailabilityMask``) and the gradient still carried an appearance-volume prior.
    This layer builds the candidate set **in-graph** from the row's ``home_roster`` /
    ``away_roster`` inputs (the cleaned data's on-court fives at that row), so the training
    denominator matches what inference actually samples from.

    Two modes:
      * ``exclude_on_court=False`` (Player/actor head): candidates = the on-court ten. The
        row's rosters are the lineup *during* that row's event, and rosters only change on
        substitution rows, so the current row's ten covers both the next actor (next row's
        lineup == this row's except across a sub) and the outgoing pick of a next-row sub
        (the outgoing player is on court *now*, and the data's sub rows carry post-sub
        rosters that already drop them).
      * ``exclude_on_court=True`` (Substitution/incoming head): candidates = ``avail_mask``
        minus the on-court ten — the legal bench at the decision, exactly the candidate set
        ``predict_incoming`` samples from. Covers the synthesized opening subs too (partial
        built-up lineup -> the remaining roster stays candidate).

    Off-candidate logits get a large-negative additive bias (finite, not -inf, so masked-out
    rows still produce finite losses under a zero sample_weight). PAD is never a candidate.
    Rows whose target falls outside the candidate set must be zeroed in the loss mask by the
    caller (see the target-in-candidates guards in the heads' ``_build_split``).

    Inputs: ``logits`` (B, SEQ, V), ``home_roster`` / ``away_roster`` (B, SEQ, ROSTER_SIZE)
    int ids, and — in exclude mode — ``avail_mask`` (B, V).
    """

    NEG = -1e9
    PAD_ID = 0  # player vocab reserves PAD=0 (encoder/vocab.py)

    def __init__(self, exclude_on_court: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.exclude_on_court = bool(exclude_on_court)

    def call(self, logits, home_roster, away_roster, avail_mask=None):
        V = tf.shape(logits)[-1]
        ids = tf.concat([home_roster, away_roster], axis=-1)  # (B, SEQ, 2*ROSTER_SIZE)
        # One-hot one roster slot at a time and max-accumulate: peak memory 2x(B,SEQ,V)
        # instead of the (B,SEQ,10,V) a single one_hot would materialize.
        oncourt = None
        for j in range(ids.shape[-1]):
            oh = tf.one_hot(ids[:, :, j], V, dtype=tf.float32)  # (B, SEQ, V)
            oncourt = oh if oncourt is None else tf.maximum(oncourt, oh)
        if self.exclude_on_court:
            if avail_mask is None:
                raise ValueError("exclude_on_court=True requires avail_mask")
            cand = tf.expand_dims(tf.cast(avail_mask, tf.float32), 1) * (1.0 - oncourt)
        else:
            cand = oncourt
        # PAD (id 0) is never a candidate (roster buffers pad with it).
        pad_col = tf.one_hot(tf.fill(tf.shape(ids)[:2], self.PAD_ID), V, dtype=tf.float32)
        cand = cand * (1.0 - pad_col)
        return logits + tf.cast((1.0 - cand) * self.NEG, logits.dtype)

    def compute_output_shape(self, logits_shape, *_, **__):
        return logits_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"exclude_on_court": self.exclude_on_court})
        return cfg


@keras.saving.register_keras_serializable(package="cviq")
class DeltaSecondsMAE(keras.metrics.MeanAbsoluteError):
    """Time-head MAE expressed in real seconds.

    The time head predicts standardized deltas ((delta - mean) / std), so the
    normalized MAE is unit-less. Absolute error scales linearly, so seconds error
    is just the normalized MAE multiplied by the training delta_std. Subclassing
    MeanAbsoluteError reuses its accumulation + sample_weight (loss mask) handling
    so this metric masks PAD/no-next steps exactly like the time loss.
    """

    def __init__(self, delta_std: float = 1.0, name: str = "mae_sec", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta_std = float(delta_std)

    def result(self):
        return super().result() * self.delta_std

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"delta_std": self.delta_std})
        return cfg


# Cleaned-data columns the model consumes.
# secondary_player shares the player embedding table (weight-tied) — see model().
CATEGORICAL_FIELDS = ["event", "player", "type", "result", "season", "secondary_player"]
ROSTER_COLS = {"home_roster": "roster_home", "away_roster": "roster_away"}


def game_available_mask(cols, idx, player_vocab_size: int, pad_player_id: int) -> np.ndarray:
    """Per-game availability over the player vocab for the player-picking heads.

    "Available" = every player id that appears in this game's on-court rosters or as an
    actor / secondary actor across rows ``idx`` (i.e. everyone who saw the floor) — the
    set the Player / Substitution heads should normalize their softmax over so they learn
    *relative* appearance frequency rather than league-wide volume (see
    ``ApplyAvailabilityMask``). PAD is always 0. ``player`` and ``secondary_player`` share
    the player vocab (weight-tied), so their ids are valid indices here.
    """
    mask = np.zeros((player_vocab_size,), dtype=np.float32)
    ids = np.unique(np.concatenate([
        cols["home_roster"][idx].reshape(-1),
        cols["away_roster"][idx].reshape(-1),
        cols["player"][idx].reshape(-1),
        cols["secondary_player"][idx].reshape(-1),
    ]))
    ids = ids[(ids >= 0) & (ids < player_vocab_size)]
    mask[ids] = 1.0
    mask[pad_player_id] = 0.0
    return mask


def target_on_court(target: np.ndarray, home_buf: np.ndarray, away_buf: np.ndarray) -> np.ndarray:
    """(SEQ,) 1.0 where ``target[t]`` appears in the row's on-court rosters, else 0.0.

    The numpy-side companion of ``OnCourtCandidateMask``: a head whose logits are masked to
    the row's candidate set must zero the loss on rows whose target is *outside* it (the
    pre-``end`` sentinel row, the rare cleaned-data row whose actor isn't in the recorded
    lineup) — otherwise the target sits on a large-negative logit and the row trains on
    noise. ``home_buf`` / ``away_buf`` are the padded (SEQ, ROSTER_SIZE) roster buffers.
    """
    oncourt = np.concatenate([home_buf, away_buf], axis=1)     # (SEQ, 2*ROSTER_SIZE)
    return (target[:, None] == oncourt).any(axis=1).astype(np.float32)


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
                 model_dim=MODEL_DIM,
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

    def _load_all(self) -> pd.DataFrame:
        """Cleaned season files concatenated with globally-unique game_id.

        Delegates to the shared loader so the box-score validation and every model see the
        same games under the same numbering.
        """
        return load_all_cleaned(self.data_dir)

    # =====================
    # --- Preprocessing ---
    # =====================

    def preprocess(self, rebuild_vocabs=True, test_frac=TEST_FRAC,
                   holdout_frac=HOLDOUT_FRAC, seed=SEED,
                   game_partition=None, refit_norm_stats=True):
        """
        Build the model-ready tensor arrays and persist them, the vocabs, and the
        time-normalization stats. Returns (train, test) dicts of numpy arrays.

        Games are partitioned three ways (see ``data_loading.split_games``): ``train``,
        ``test`` (the early-stopping validation split), and ``holdout`` — a fully reserved
        batch of real games the model never sees, for unbiased real-game testing. The
        holdout game ids are written to a manifest so the box-score validation can find
        exactly those games.

        ``game_partition`` (curriculum use): an explicit ``(train, test, holdout)`` tuple of
        game-id collections to use instead of the random ``split_games`` (e.g. the chronological
        next-N split from ``training.chronology.sequential_partition``). ``refit_norm_stats``
        (default True) recomputes + persists normalization stats from the train slice; set False
        in staged runs to load the warmup-fit stats so standardization stays fixed for warm-start.
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

        # 4) Split games into train/test/holdout BEFORE computing normalization stats.
        #    The holdout is carved off first and excluded from both train and test.
        train_games, test_games, holdout_games = resolve_partition(
            game_partition, game_id, seed, test_frac, holdout_frac,
        )

        train_mask = np.array([g in train_games for g in game_id])

        # 5) Normalization stats from TRAIN ONLY (or loaded, for staged warm-start). time_abs =
        #    time/max_time; delta standardized. Persist so inference uses identical transforms.
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

        # 6) Assemble per-game padded sequences for each split. Season-context features
        #    (per-player rest + team scalars) are normalized off the TRAIN split too and
        #    fold rest_mean/rest_std into norm_stats (persisted below for inference).
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

        # 7) Persist. Holdout tensors enable event-level eval; the manifest of holdout
        #    game ids is the contract the box-score validation reads to load real games.
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.processed_dir / "train.npz", **train)
        np.savez_compressed(self.processed_dir / "test.npz", **test)
        np.savez_compressed(self.processed_dir / "holdout.npz", **holdout)
        (self.processed_dir / HOLDOUT_MANIFEST_NAME).write_text(
            json.dumps(sorted(int(g) for g in holdout_games), indent=2), encoding="utf-8"
        )
        # Only (re)write the normalization stats when refitting; staged runs reuse warmup stats.
        if refit_norm_stats:
            save_norm_stats(self.processed_dir, self.KEY, self.norm_stats)
            Path(NORM_STATS_PATH).parent.mkdir(parents=True, exist_ok=True)
            Path(NORM_STATS_PATH).write_text(json.dumps(self.norm_stats, indent=2), encoding="utf-8")

        print(f"Preprocessed {len(train_games)} train / {len(test_games)} test / "
              f"{len(holdout_games)} holdout games -> {self.processed_dir} "
              f"(SEQ={self.sequence_length})")
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

        batches = {k: [] for k in (*keys_1d, *keys_roster, *keys_cont, *SEASON_INPUT_KEYS,
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

            append_season_batches(batches, cols, idx, n, SEQ)

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
        )
        # Applies the shared set-encoder across the time axis via reshape (not
        # TimeDistributed, which unrolls SEQ in graph mode and exhausts memory).
        return SequenceRosterEncoder(params, name="roster_vec")

    def model(self, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=0.2):
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
        # One shared SequenceRosterEncoder applied to both rosters (with per-player rest):
        # weight-ties home/away and encodes all timesteps in a single reshaped pass.
        home_vec = self.roster_encoder([home_roster, rest_home])
        away_vec = self.roster_encoder([away_roster, rest_away])

        # ---- Continuous projections ----
        t_abs = layers.Dense(16, name="time_abs_proj")(time_abs)
        t_delta = layers.Dense(16, name="delta_time_proj")(delta_time)
        t_team = season_team_projections(team_inputs)  # games-played + team rest per side

        # ---- Fusion ----
        x = layers.Concatenate(axis=-1, name="fusion_concat")(
            [*embs, home_vec, away_vec, t_abs, t_delta, *t_team]
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
            "time_abs": time_abs, "delta_time": delta_time,
            "rest_home": rest_home, "rest_away": rest_away, **team_inputs,
            "pad_mask": pad_mask,
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
        "home_roster", "away_roster", "time_abs", "delta_time",
        *SEASON_INPUT_KEYS, "pad_mask",
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
        # loss_mask zeroes the loss on PAD steps and on the final (no-next) step. We also zero
        # every step whose NEXT event is a substitution: substitutions are injected by the
        # rotation scheduler at inference, not sampled from the event stream, so the event/time
        # heads must never learn to emit them (no probability mass to renormalize away, and the
        # time head never predicts the gap to a sub). Sub rows stay in the sequence as context.
        sub_id = self.encoder.encode_event("substitution")
        mask = (split["loss_mask"] * (split["event_target"] != sub_id)).astype(np.float32)
        mask = apply_recency(mask, split)
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

    def train(self, epochs=50, batch_size=64, lr=3e-4, time_loss_weight=0.5,
              patience=10, artifacts_root=DEFAULT_ARTIFACTS_ROOT,
              mixed_precision=True, jit_compile=False,
              num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=0.2,
              warmup_epochs=1, lr_alpha=0.05,
              report=True, run_name=None, reports_root=DEFAULT_REPORTS_ROOT,
              init_weights_root=None):
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

        model = self.model(num_layers=num_layers, num_heads=num_heads,
                           ff_dim=ff_dim, dropout=dropout)
        # Print the per-layer table + Total/Trainable/Non-trainable params to the
        # console each run (the report also records these in its Model size table).
        model.summary()
        # Curriculum warm-start: continue the previous stage's weights when given.
        warm_start_weights(model, self.KEY, init_weights_root)

        # Warmup + cosine-decay LR schedule. The old ReduceLROnPlateau collapsed the
        # LR once loss flattened, stalling learning while the curve was still flat;
        # a warmup-then-cosine schedule keeps the LR meaningful across the whole run
        # and is the cleaner fix for an early plateau. (A schedule and plateau-based
        # reduction don't compose, so ReduceLROnPlateau is dropped.)
        steps_per_epoch = int(np.ceil(train_split["pad_mask"].shape[0] / batch_size))
        total_steps = steps_per_epoch * epochs
        warmup_steps = steps_per_epoch * warmup_epochs
        decay_steps = max(1, total_steps - warmup_steps)
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,   # start of linear warmup
            warmup_target=lr,            # peak LR reached after warmup
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,     # cosine decay phase after warmup
            alpha=lr_alpha,              # floor = lr_alpha * lr
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0,
        )
        # Time head is trained on standardized deltas; mae_sec reports the same error
        # in real seconds (normalized MAE * delta_std) so convergence is readable.
        #
        # Loss is MSE, not MAE, on purpose: inter-event gaps are right-skewed, and MAE targets the
        # conditional *median* (< mean for a right skew). The rollout fills a fixed 48 minutes by
        # advancing the clock by the predicted gap each step, so a median-targeting head
        # systematically under-shoots the gap and packs too many events (hence too many shots) into
        # the game. MSE targets the conditional *mean*, which makes the expected event count (and
        # thus pace / FGA) come out right. clipnorm=1.0 on the optimizer guards the rare long gap.
        delta_std = float((self.norm_stats or {}).get("delta_std", 1.0) or 1.0)
        model.compile(
            optimizer=optimizer,
            loss={
                "event_output": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                "time_output": keras.losses.MeanSquaredError(),
            },
            loss_weights={"event_output": 1.0, "time_output": time_loss_weight},
            # weighted_metrics (NOT metrics): only weighted_metrics receive the sample_weight
            # mask, so acc/mae are computed over real (non-PAD) rows instead of every padded
            # position. Plain `metrics=` would dilute them across the whole sequence.
            weighted_metrics={
                "event_output": [keras.metrics.SparseCategoricalAccuracy(name="acc")],
                "time_output": [
                    keras.metrics.MeanAbsoluteError(name="mae"),
                    DeltaSecondsMAE(delta_std, name="mae_sec"),
                ],
            },
            jit_compile=jit_compile,
        )

        callbacks = [
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
