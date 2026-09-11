from __future__ import annotations
import tensorflow as tf
import keras


@keras.saving.register_keras_serializable(package="cviq")
class RowFF(keras.layers.Layer):
    """
    Row-wise FeedForward network (rFF) from the Set Transformer paper.

    What it does:
        Applies the SAME small MLP to each element (row) in a set.

    Input shape:
        (B, N, D)  -> batch B, set size N, feature dim D

    Output shape:
        (B, N, D)  -> same shape (D stays the same)

    Why it exists:
        After attention mixes information across elements, this MLP gives each
        element extra non-linear processing while keeping set structure.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "relu",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.activation = activation

        self.fc1 = keras.layers.Dense(d_ff, activation=activation, name="fc1")
        self.drop = keras.layers.Dropout(dropout, name="drop")
        self.fc2 = keras.layers.Dense(d_model, name="fc2")

    def call(self, x, training: bool = False):
        # Fold every leading axis into the batch dimension before the two Dense layers, and
        # restore the shape afterwards. rFF is row-wise, so this is the same function -- but
        # it is not the same graph, and the difference is measured in gigabytes.
        #
        # Keras 3's Dense is `ops.matmul(inputs, kernel)`. On a rank-3 input that lowers to a
        # *broadcasting* BatchMatMul, whose gradient materialises one kernel gradient PER
        # batch element -- (B, d_model, d_ff) -- before reducing it to the (d_model, d_ff)
        # the kernel actually is. Inside the roster encoder the batch axis is already B*SEQ
        # (SequenceRosterEncoder collapses time into it), so at batch 64 / SEQ 600 each of
        # those intermediates is 38400 x 128 x 256 x 2 bytes = 2.34 GiB in float16 -- eight
        # per roster application, twice a step for home and away. Flattened to rank 2 the op
        # is a plain MatMul and the kernel gradient is (d_model, d_ff), i.e. 128 KiB.
        #
        # Keras 2's Dense reshaped internally for rank > 2, which is why this only surfaced
        # now: nothing about the set transformer changed, the Dense underneath it did.
        lead = tf.shape(x)[:-1]
        flat = tf.reshape(x, (-1, x.shape[-1]))
        flat = self.fc1(flat)
        flat = self.drop(flat, training=training)
        flat = self.fc2(flat)
        return tf.reshape(flat, tf.concat([lead, [self.d_model]], axis=0))

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "d_model": self.d_model,
                "d_ff": self.d_ff,
                "dropout": self.dropout_rate,
                "activation": self.activation,
            }
        )
        return cfg
