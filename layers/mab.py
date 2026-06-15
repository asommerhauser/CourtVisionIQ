# layers/mab.py
from __future__ import annotations

import keras

from .row_ff import RowFF


@keras.saving.register_keras_serializable(package="cviq")
class MAB(keras.layers.Layer):
    """
    Multihead Attention Block (MAB) from the Set Transformer paper.

    This is the core building block that everything else uses.

    What it does (high level):
        1) Attention: X attends to Y  (query=X, key/value=Y)
        2) Add + Normalize (residual)
        3) Row-wise FeedForward (MLP applied to each element)
        4) Add + Normalize (residual)

    Shapes:
        X: (B, Nx, D)
        Y: (B, Ny, D)
        Output: (B, Nx, D)

    Notes:
        - If you pass Y=X, this becomes self-attention (used in SAB).
        - attention_mask is optional. Keras expects broadcastable masks for MHA.
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

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.ln_eps = ln_eps

        # 1) Multi-head attention
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name="mha",
        )

        # 2) Row-wise MLP (applied to each element independently)
        self.rff = RowFF(d_model=d_model, d_ff=d_ff, dropout=dropout, name="rff")

        # Normalization + dropout around residual connections
        self.drop = keras.layers.Dropout(dropout, name="drop")
        self.ln1 = keras.layers.LayerNormalization(epsilon=ln_eps, name="ln1")
        self.ln2 = keras.layers.LayerNormalization(epsilon=ln_eps, name="ln2")

    def call(self, X, Y, training: bool = False, attention_mask=None):
        """
        X attends to Y.

        Args:
            X: (B, Nx, D)  queries
            Y: (B, Ny, D)  keys/values
            attention_mask: optional mask for attention

        Returns:
            (B, Nx, D)
        """

        # Attention
        # Each element in X mixes information from Y based on learned weights.
        attn_out = self.attn(
            query=X,
            value=Y,
            key=Y,
            attention_mask=attention_mask,
            training=training,
        )
        attn_out = self.drop(attn_out, training=training)

        # Residual + Norm
        H = self.ln1(X + attn_out)

        # Row-wise FeedForward
        ff_out = self.rff(H, training=training)
        ff_out = self.drop(ff_out, training=training)

        # Residual + Norm
        out = self.ln2(H + ff_out)
        return out

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