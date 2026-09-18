"""
Cross-roster attention: each lineup's slots attend over the other lineup's.

The gap it closes (3.2 W7, from ``docs/v3_2_direction.md`` 5.1): the two rosters pass through one
weight-tied encoder **independently** and meet only at the fusion concat, so there is no structural way
to represent one lineup *against* another. A switch-heavy defence and a rim-protecting one are the same
input to the offence's representation.

**Why this is a wrapper rather than a bare ``MAB``.** ``layers/mab.py`` is written correctly for
cross-attention -- ``call(X, Y)`` already means "X attends to Y" -- but it has never been *used* that
way: ``SAB`` calls it with ``Y = X`` and ``PMA`` with learned seeds. Three consequences, all of which
bite when you drop it into a functional graph:

* It defines no ``compute_output_shape``, and a second positional tensor argument is not in the
  standard ``inputs`` slot, so Keras 3 has nothing to infer the output spec from.
* It creates its variables lazily on first call, which under ``load_weights`` means the weights have
  nowhere to land. ``RosterSetEncoder.build`` solves the same problem the same way, and for the same
  reason: a parent ``build()`` must create all child state.
* It is **post-norm** (``ln2(H + ff_out)``) while the backbone is pre-norm. That is fine here precisely
  because this sits inside the roster encoder, among the post-norm ``SAB`` layers it was designed
  alongside -- it is not spliced into the pre-norm residual stream.

Weights are shared between the two directions: one block computes home-attending-to-away and
away-attending-to-home. That is what makes it learn "how a lineup reads an opponent" rather than
memorising which side of the ledger a team sits on.
"""
from __future__ import annotations

import tensorflow as tf
import keras

from layers.mab import MAB


@keras.saving.register_keras_serializable(package="cviq")
class CrossRosterBlock(keras.layers.Layer):
    """One shared attention block: ``(x_self, x_other, other_valid) -> (B, N, D)``.

    ``other_valid`` is ``(B, N)`` boolean over the *other* roster's slots, so a PAD slot on the far side
    cannot be attended to.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.0,
                 name: str = "cross_roster", **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.dropout_rate = float(dropout)
        self.mab = MAB(d_model=self.d_model, num_heads=self.num_heads, d_ff=self.d_ff,
                       dropout=self.dropout_rate, name="mab")

    def build(self, input_shape):
        # Force the MAB subtree to create its variables now. Without this it builds lazily on the first
        # call and reads as "never built" at load time, so saved weights have nowhere to land -- the
        # same failure RosterSetEncoder.build exists to prevent.
        n = 2
        dummy = tf.zeros((1, n, self.d_model), dtype="float32")
        mask = tf.ones((1, 1, n), dtype="bool")
        self.mab(dummy, dummy, attention_mask=mask)
        super().build(input_shape)

    def call(self, x_self, x_other, other_valid=None, training: bool = False):
        mask = None
        if other_valid is not None:
            valid = tf.cast(other_valid, tf.bool)
            # If the far roster has no valid slot at all, attend to everything instead of to nothing.
            # A fully-masked query row makes softmax divide by zero and yields NaN, which would spread
            # through the pooled vector; the result is discarded at pooling anyway, so an unmasked row
            # is the harmless choice. Real rosters always have five, but an all-PAD roster is
            # constructible (``encode_roster([])``) and a NaN is not a failure anyone would trace back
            # here.
            any_valid = tf.reduce_any(valid, axis=-1, keepdims=True)    # (B, 1)
            valid = tf.logical_or(valid, tf.logical_not(any_valid))     # (B, N)
            mask = valid[:, tf.newaxis, :]                              # (B, 1, N) -> broadcast
        return self.mab(x_self, x_other, training=training, attention_mask=mask)

    def compute_output_shape(self, x_self_shape, x_other_shape=None, other_valid_shape=None):
        # Queries keep their own shape: (B, N_self, D).
        return tuple(x_self_shape)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads,
                    "d_ff": self.d_ff, "dropout": self.dropout_rate})
        return cfg


__all__ = ["CrossRosterBlock"]
