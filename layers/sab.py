from __future__ import annotations
import keras
from .mab import MAB


@keras.saving.register_keras_serializable(package="cviq")
class SAB(keras.layers.Layer):
    """
    Set Attention Block (SAB).

    SAB is just self-attention over a set:
        SAB(X) = MAB(X, X)

    Input shape:
        (B, N, D)

    Output shape:
        (B, N, D)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        ln_eps: float = 1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.ln_eps = ln_eps

        self.mab = MAB(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            ln_eps=ln_eps,
            name="mab",
        )

    def call(self, X, training: bool = False, attention_mask=None):
        # Self-attention: X attends to X
        return self.mab(X, X, training=training, attention_mask=attention_mask)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "d_ff": self.d_ff,
                "dropout": self.dropout_rate,
                "ln_eps": self.ln_eps,
            }
        )
        return cfg