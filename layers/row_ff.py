from __future__ import annotations
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
        x = self.fc1(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        return x

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
