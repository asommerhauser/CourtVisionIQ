"""
The one training wrapper: the per-game regime latent and scheduled sampling, in a single train_step.

Both 3.0 training changes want to alter what the model sees before the forward pass -- W3's latent
writes a per-game vector into the ``regime`` plane, W4 rung 1 replaces some previous-event tokens
with the model's own samples. Two separate ``keras.Model`` wrappers could not compose: the outer
one's ``self.inner(x)`` would hand the inner wrapper a batch it does not accept, and the ordering
between them would be implicit. So there is exactly one wrapper, and both features are optional
inside it.

Everything here is a TRAINING device. ``save_artifacts`` always persists ``inner`` -- a plain
functional model that takes ``regime`` as an ordinary input -- so the artifact layout, the reload
path and the rollout are unchanged. What survives training is one number per latent dimension in
``norm_stats["regime_std"]``, and nothing at all from scheduled sampling.

The per-feature reasoning lives in ``models/regime.py`` and ``models/scheduled_sampling.py``; this
module is the mechanics.

Three things in ``train_step`` are easy to leave out and silent when they are missing:

1. **The loss tracker must be fed.** Keras' own ``train_step`` does it; a custom one that forgets
   reports loss 0.0 every epoch while gradients flow perfectly well. ``EarlyStopping(monitor=
   "val_loss", restore_best_weights=True)`` is this repo's only checkpoint selector, so against a
   constant zero it never improves and restores epoch 1's weights at the end of a full train.
2. **The loss must be scaled.** ``configure_gpu`` sets ``mixed_float16`` for every GPU train, and an
   unscaled loss underflows fp16 gradients to zero.
3. **Validation is always plain.** No latent lookup (a validation game has no trained row) and no
   mixing (val_loss must mean the same thing every epoch, because EarlyStopping reads it).
"""
from __future__ import annotations

import numpy as np

import config
from config import REGIME_DIM, REGIME_L2
from models.regime import GAME_INDEX_KEY, REGIME_KEY


def build_trainer(inner, *, n_games: int = 0, scheduled_sampling: bool = False):
    """Wrap ``inner`` for training, or return it unchanged when neither feature is on.

    ``n_games`` sizes the latent table; 0 or 1 disables it. ``scheduled_sampling`` is passed
    explicitly rather than read from config here, because it is enabled per head -- only
    ``event_time`` uses it in this retrain.
    """
    # config.X, not a module-level import: these are switched off in tests and on the command
    # line, and a value bound at import time would ignore both.
    want_regime = config.REGIME_ENABLED and n_games >= 2
    want_mixing = config.SCHEDULED_SAMPLING and scheduled_sampling
    if not (want_regime or want_mixing):
        return inner
    return _trainer_class()(inner, n_games=n_games if want_regime else 0,
                            scheduled_sampling=want_mixing)


def _trainer_class():
    """Built lazily so importing this module does not import TensorFlow."""
    import tensorflow as tf
    from tensorflow import keras

    from models.scheduled_sampling import MIXABLE_FIELDS

    class Trainer(keras.Model):
        """``inner`` plus, optionally, a latent table and self-sampled context."""

        def __init__(self, inner, *, n_games: int = 0, scheduled_sampling: bool = False, **kwargs):
            super().__init__(**kwargs)
            self.inner = inner
            self.n_games = int(n_games)
            self.scheduled_sampling = bool(scheduled_sampling)
            self.table = None
            if self.n_games >= 2:
                self.table = keras.layers.Embedding(
                    self.n_games, REGIME_DIM,
                    embeddings_initializer=keras.initializers.RandomNormal(stddev=0.01),
                    embeddings_regularizer=keras.regularizers.l2(REGIME_L2),
                    name="regime_table")
                # Built eagerly so its variables exist before the first batch; otherwise they are
                # created inside the GradientTape and fitted_table() cannot be read before a fit.
                self.table.build((None,))
            # A variable, not a Python float: train_step is traced once, and a plain attribute would
            # be frozen into the graph at its first value and never change again.
            self.mixing_p = tf.Variable(0.0, trainable=False, dtype=tf.float32,
                                        name="scheduled_sampling_p")

        # -------------------------------------------------------------- plumbing
        def call(self, inputs, training=False):
            return self.inner(self._strip(inputs), training=training)

        @staticmethod
        def _strip(x):
            """Drop the training-only columns the functional model does not declare."""
            return {k: v for k, v in x.items() if k != GAME_INDEX_KEY}

        def set_mixing_probability(self, p: float) -> None:
            self.mixing_p.assign(float(p))

        def fitted_table(self) -> np.ndarray:
            if self.table is None or not self.table.built:
                return np.zeros((0, REGIME_DIM), dtype=np.float32)
            return np.asarray(self.table.embeddings.numpy())

        def _track(self, loss, x):
            tracker = getattr(self, "_loss_tracker", None)
            if tracker is not None:
                tracker.update_state(loss, sample_weight=tf.shape(tf.nest.flatten(x)[0])[0])

        # -------------------------------------------------------------- the two features
        def _with_regime(self, x):
            """Replace the zeros plane with this batch's latent rows, tiled over the sequence."""
            if self.table is None or GAME_INDEX_KEY not in x:
                return self._strip(x)
            idx = tf.reshape(tf.cast(x[GAME_INDEX_KEY], tf.int32), (-1,))
            z = self.table(idx)                                        # (B, REGIME_DIM)
            seq_len = tf.shape(x[REGIME_KEY])[1]
            tiled = tf.tile(tf.expand_dims(z, 1), [1, seq_len, 1])
            out = self._strip(x)
            out[REGIME_KEY] = tf.cast(tiled, x[REGIME_KEY].dtype)
            return out

        def _mix(self, x, sample_weight):
            """Replace the previous event's token with the model's own sample, at query positions.

            The shift is the one teacher forcing already has: the target at t is the real row t+1,
            so the model's prediction AT t is what row t+1's token would have been, and writing it
            into position t+1's input is exactly "the previous event is my own". Position 0 is never
            touched -- a game's opening row is a given.
            """
            if "event" not in x:
                return x
            preds = self.inner(x, training=False)
            logits = preds.get("event_output") if isinstance(preds, dict) else preds
            if logits is None:
                return x
            # stop_gradient because this pass only chooses tokens. tf.random.categorical already
            # breaks the path (its output is integer indices), but saying so explicitly keeps the
            # sampling forward pass off the tape rather than merely unused by it.
            logits = tf.stop_gradient(logits)
            # A sample, not an argmax: the rollout samples, so the mistakes worth learning to
            # recover from are sampled ones rather than the model's most confident ones.
            flat = tf.reshape(logits, (-1, tf.shape(logits)[-1]))
            sampled = tf.reshape(tf.random.categorical(flat, 1, dtype=tf.int32),
                                 tf.shape(logits)[:2])

            weight = sample_weight.get("event_output") if isinstance(sample_weight, dict) \
                else sample_weight
            asked = (tf.cast(weight, tf.float32) > 0.0) if weight is not None \
                else tf.ones(tf.shape(sampled), tf.bool)
            draw = tf.random.uniform(tf.shape(sampled), dtype=tf.float32) < self.mixing_p
            replace = tf.pad(tf.logical_and(asked, draw)[:, :-1], [[0, 0], [1, 0]],
                             constant_values=False)
            shifted = tf.pad(sampled[:, :-1], [[0, 0], [1, 0]])

            out = dict(x)
            out["event"] = tf.cast(tf.where(replace, shifted, tf.cast(x["event"], shifted.dtype)),
                                   x["event"].dtype)
            return out

        # -------------------------------------------------------------- the steps
        def train_step(self, data):
            x, y, w = keras.utils.unpack_x_y_sample_weight(data)
            with tf.GradientTape() as tape:
                # The embedding lookup has to happen INSIDE the tape, or the table receives no
                # gradient and trains as all zeros -- silently, because the head itself still
                # learns perfectly well and only regime_std comes out empty at the end.
                #
                # Regime before mixing: the no-grad forward that draws the replacement tokens
                # should see the same latent the gradient pass will, or it samples from a different
                # model than the one being trained.
                fed = self._with_regime(x)
                if self.scheduled_sampling:
                    fed = tf.cond(self.mixing_p > 0.0,
                                  lambda: self._mix(fed, w), lambda: dict(fed))
                preds = self.inner(fed, training=True)
                loss = self.compute_loss(x=fed, y=y, y_pred=preds, sample_weight=w, training=True)
                self._track(loss, fed)
                scaled = self.optimizer.scale_loss(loss) \
                    if hasattr(self.optimizer, "scale_loss") else loss
            trainable = list(self.inner.trainable_variables)
            if self.table is not None:
                trainable += list(self.table.trainable_variables)
            self.optimizer.apply_gradients(zip(tape.gradient(scaled, trainable), trainable))
            return self.compute_metrics(fed, y, preds, sample_weight=w)

        def test_step(self, data):
            """Plain: no latent lookup, no mixing. See the module docstring, point 3."""
            x, y, w = keras.utils.unpack_x_y_sample_weight(data)
            x = self._strip(x)
            preds = self.inner(x, training=False)
            loss = self.compute_loss(x=x, y=y, y_pred=preds, sample_weight=w, training=False)
            self._track(loss, x)
            return self.compute_metrics(x, y, preds, sample_weight=w)

    return Trainer


__all__ = ["build_trainer"]
