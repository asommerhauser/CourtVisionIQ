# layers/pma.py
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from .mab import MAB
from .row_ff import RowFF


class PMA(keras.layers.Layer):
    """
    Pooling by Multihead Attention (PMA).

    Purpose:
        Compress a set of N elements into K pooled "summary" vectors using
        learned seed vectors (like trainable query slots).

    Paper form:
        PMA(X) = MAB(S, rFF(X))
        - S is learnable seeds (K of them)
        - S attends over the set elements X

    Input shape:
        X: (B, N, D)

    Output shape:
        (B, K, D)    if return_pooled_vector=False
        (B, D)       if K=1 and return_pooled_vector=True
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        k_seeds: int = 1,
        dropout: float = 0.0,
        ln_eps: float = 1e-6,
        return_pooled_vector: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.k_seeds = k_seeds
        self.dropout_rate = dropout
        self.ln_eps = ln_eps
        self.return_pooled_vector = return_pooled_vector

        # Learnable seed vectors S: (K, D)
        self.seeds = self.add_weight(
            name="seed_vectors",
            shape=(k_seeds, d_model),
            initializer="glorot_uniform",
            trainable=True,
        )

        # rFF(X) before pooling (matches the paper)
        self.pre_ff = RowFF(d_model=d_model, d_ff=d_ff, dropout=dropout, name="pre_rff")

        # Seed attends to the set: MAB(S, rFF(X))
        self.mab = MAB(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            ln_eps=ln_eps,
            name="mab",
        )

    def call(self, X, training: bool = False, attention_mask=None):
        # X: (B, N, D)
        B = tf.shape(X)[0]

        # Tile seeds across the batch: (B, K, D)
        S = tf.tile(self.seeds[tf.newaxis, :, :], [B, 1, 1])

        # Pre-FF over set elements (row-wise)
        X2 = self.pre_ff(X, training=training)

        # Pool: seeds attend over the set
        pooled = self.mab(S, X2, training=training, attention_mask=attention_mask)  # (B, K, D)

        # Common convenience: if K=1 return (B, D)
        if self.return_pooled_vector and self.k_seeds == 1:
            return tf.squeeze(pooled, axis=1)

        return pooled

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "d_ff": self.d_ff,
                "k_seeds": self.k_seeds,
                "dropout": self.dropout_rate,
                "ln_eps": self.ln_eps,
                "return_pooled_vector": self.return_pooled_vector,
            }
        )
        return cfg