"""
A per-game regime latent: one small vector, drawn once per rollout and held for the game.

The 2.0 evaluation's sharpest finding is that the two teams in a sim do not share a game. Across
the 50 sims of one game, corr(home pts, away pts) is 0.02; in real 2022-23 games it is 0.35. Pace sd
within a game is 2.5 against a real 4.8 across games. The consequence is arithmetic: with
independent sides, Var(H-A) and Var(H+A) both collapse to VarH + VarA, so the margin comes out at
16.5 against a real 13.7 while the total comes out at 16.7 against a real 19.7. Both halves of that
are one missing term. It is a *missing* shared component, not a mis-sized one, which is why no
shrinkage dial is the right fix and why MARGIN_CALIBRATION_SLOPE is meant to retire rather than be
tuned.

What is missing is a game-level quantity that both teams' events condition on. Running pace
(``models/game_state_features.running_pace``) is the observable half: the model can read how fast
the game has gone so far. This is the unobservable half -- whatever it is about a particular night
that makes both teams shoot well, or the whistle swallow, or the pace drag, before any of it has
happened.

**How it is trained.** One free embedding row per training game, L2-regularized toward zero, tiled
across the sequence and concatenated into the fusion. Gradient descent puts into that row whatever
about the game the other inputs cannot explain -- a per-game random effect, the same device a mixed
model uses. The regularizer is what stops it memorising the game outright: it can only pay for
structure that repeats across many events of the same game, which is exactly the shared component
the sim is missing.

**How it is used.** After training, the per-dimension standard deviation of the fitted table is
written to ``norm_stats["regime_std"]``. At rollout the simulator draws ``z ~ N(0, regime_std)``
**once per game** and holds it for every row -- so two sims of the same matchup get different nights,
and within one sim both teams share the same night. That is the whole mechanism.

**What validation loss will do, and why not to "fix" it.** A validation game's embedding row is
never trained, so it holds its initialization. ``val_loss`` therefore measures the model under an
uninformative latent -- which is exactly what a rollout gets, and so is the honest number. It will
read worse than a run without the latent. Leaking validation indices into the table to make that
number look better would be measuring the one thing the model can never have at inference.

**Gate** (docs/v3_direction.md §3 W3): sim corr(home, away) ~ 0.35, pace sd ~ 5.3, margin sd ~ 13.7,
with no shrinkage dial. Measured starting point, from the four v2 runs: corr 0.017, pace sd 2.47,
margin sd 16.46, dispersion 1.36x. The gate is read off ``eval_metrics.joint_metrics``; it is not a
box-score number and must not be judged as one.
"""
from __future__ import annotations

import numpy as np

from config import REGIME_DIM

# The model input the latent arrives on, and the training-only column that says which game a row is.
REGIME_KEY = "regime"
GAME_INDEX_KEY = "game_index"


def make_regime_input(seq_len: int):
    """The ``(SEQ, REGIME_DIM)`` Keras Input. Tiled across time because the fusion is per-row."""
    from tensorflow import keras

    return keras.Input(shape=(seq_len, REGIME_DIM), name=REGIME_KEY, dtype="float32")


def regime_projection(regime_input):
    """``Dense(16)``, matching every other continuous projection into the fusion."""
    from tensorflow.keras import layers

    return layers.Dense(16, name="regime_proj")(regime_input)


def append_regime_batches(batches: dict, game_pos: int, seq_len: int) -> None:
    """Zeros for the latent plane, and this game's index.

    The plane is zeros on disk: :class:`RegimeModel` overwrites it inside ``train_step`` from the
    embedding table, and at rollout the simulator writes its own draw. Storing zeros rather than a
    sample keeps the npz honest about what it contains -- there is no per-game value until a table
    has been fitted.
    """
    batches[REGIME_KEY].append(np.zeros((seq_len, REGIME_DIM), dtype=np.float32))
    batches[GAME_INDEX_KEY].append(np.array([game_pos], dtype=np.int32))


def regime_std(table: np.ndarray) -> list[float]:
    """Per-dimension sd of a fitted table, mean removed. What the rollout samples from.

    The mean is removed because a constant offset is not a regime -- the heads can absorb it into a
    bias, and sampling around it would just add noise with no shared structure.
    """
    table = np.asarray(table, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] < 2:
        return [0.0] * REGIME_DIM
    centered = table - table.mean(axis=0, keepdims=True)
    return [float(v) for v in centered.std(axis=0)]


def sample_regime(rng, std) -> np.ndarray:
    """One game's draw. ``std`` of all zeros returns zeros, which reproduces a model without one."""
    std = np.asarray(std, dtype=np.float32)
    if std.size != REGIME_DIM or not np.any(std > 0):
        return np.zeros((REGIME_DIM,), dtype=np.float32)
    return (rng.standard_normal(REGIME_DIM) * std).astype(np.float32)


def build_regime_model(inner, n_games: int):
    """Wrap a functional head in the training-time embedding table.

    Returns ``inner`` unchanged when the latent is switched off or there are no games to index, so
    a caller never has to branch.
    """
    from config import REGIME_ENABLED

    if not REGIME_ENABLED or n_games < 2:
        return inner
    # _regime_model_class(), not a bare RegimeModel: the class is built lazily so importing this
    # module stays TF-free, and PEP 562's module __getattr__ resolves attribute access from OUTSIDE
    # the module -- it is not consulted for a global lookup inside it.
    return _regime_model_class()(inner, n_games)


def _regime_model_class():
    """Defined lazily so importing this module does not import TensorFlow."""
    import tensorflow as tf
    from tensorflow import keras
    from config import REGIME_L2

    class RegimeModel(keras.Model):
        """A functional head plus one trainable latent row per training game.

        ``train_step`` pops the game index out of the batch, looks up its row, tiles it over the
        sequence and writes it into the ``regime`` input before the forward pass. The inner model is
        an ordinary functional model that simply takes ``regime`` as an input -- which is what makes
        the persistence path unchanged: ``save_artifacts`` saves ``inner``, and nothing about the
        table survives except the standard deviations written into norm_stats.

        ``test_step`` deliberately does NOT look the row up. A validation game has no trained row,
        and feeding it one would be measuring information a rollout can never have.
        """

        def __init__(self, inner, n_games, **kwargs):
            super().__init__(**kwargs)
            self.inner = inner
            self.n_games = int(n_games)
            self.table = keras.layers.Embedding(
                self.n_games, REGIME_DIM,
                embeddings_initializer=keras.initializers.RandomNormal(stddev=0.01),
                embeddings_regularizer=keras.regularizers.l2(REGIME_L2),
                name="regime_table")
            # Built eagerly so the table's variables exist before the first batch -- otherwise
            # train_step's first call creates them inside the GradientTape, and fitted_table()
            # cannot be read at all until a fit has happened.
            self.table.build((None,))

        def call(self, inputs, training=False):
            return self.inner(inputs, training=training)

        def _with_regime(self, x):
            """Replace the zeros plane with this batch's latent rows."""
            idx = tf.reshape(tf.cast(x[GAME_INDEX_KEY], tf.int32), (-1,))
            z = self.table(idx)                                   # (B, REGIME_DIM)
            seq_len = tf.shape(x[REGIME_KEY])[1]
            tiled = tf.tile(tf.expand_dims(z, 1), [1, seq_len, 1])
            out = {k: v for k, v in x.items() if k != GAME_INDEX_KEY}
            out[REGIME_KEY] = tf.cast(tiled, x[REGIME_KEY].dtype)
            return out

        def _track_loss(self, loss, x):
            """Feed the reported-loss metric, the way Keras' own train_step does.

            Not optional bookkeeping. Without it every epoch reports loss 0.0, and
            ``EarlyStopping(monitor="val_loss", restore_best_weights=True)`` -- which is this
            repo's only checkpoint selector -- sees a flat zero, never improves, and restores
            epoch 1's weights at the end of a full train. Gradients flow the whole time, so
            nothing looks wrong until the run is over.
            """
            tracker = getattr(self, "_loss_tracker", None)
            if tracker is not None:
                tracker.update_state(loss, sample_weight=tf.shape(tf.nest.flatten(x)[0])[0])

        def train_step(self, data):
            x, y, w = keras.utils.unpack_x_y_sample_weight(data)
            with tf.GradientTape() as tape:
                preds = self.inner(self._with_regime(x), training=True)
                loss = self.compute_loss(x=x, y=y, y_pred=preds, sample_weight=w, training=True)
                self._track_loss(loss, x)
                # Mixed precision is on for every GPU train (configure_gpu sets mixed_float16), so
                # the loss must be scaled before the tape or fp16 gradients underflow to zero.
                scaled = self.optimizer.scale_loss(loss) if hasattr(self.optimizer, "scale_loss")                     else loss
            trainable = self.inner.trainable_variables + self.table.trainable_variables
            self.optimizer.apply_gradients(zip(tape.gradient(scaled, trainable), trainable))
            return self.compute_metrics(x, y, preds, sample_weight=w)

        def test_step(self, data):
            x, y, w = keras.utils.unpack_x_y_sample_weight(data)
            x = {k: v for k, v in x.items() if k != GAME_INDEX_KEY}
            preds = self.inner(x, training=False)
            loss = self.compute_loss(x=x, y=y, y_pred=preds, sample_weight=w, training=False)
            self._track_loss(loss, x)
            return self.compute_metrics(x, y, preds, sample_weight=w)

        def fitted_table(self) -> np.ndarray:
            """The latent rows as they stand. Empty if the layer somehow never built."""
            if not self.table.built:
                return np.zeros((0, REGIME_DIM), dtype=np.float32)
            return np.asarray(self.table.embeddings.numpy())

    return RegimeModel


def __getattr__(name):
    if name == "RegimeModel":
        return _regime_model_class()
    raise AttributeError(name)


__all__ = ["REGIME_KEY", "GAME_INDEX_KEY", "make_regime_input", "regime_projection",
           "append_regime_batches", "regime_std", "sample_regime", "build_regime_model"]
