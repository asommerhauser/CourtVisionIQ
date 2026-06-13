from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from keras import layers

from layers.sab import SAB
from layers.pma import PMA


@dataclass(frozen=True)
class RosterEncoderParams:
    # Input format
    roster_size: int = 5

    # Vocab size (how many player IDs exist)
    num_players: int = 0

    # Embedding + internal set-transformer dimension
    roster_dim: int = 128

    # Set Transformer
    num_sab_layers: int = 2
    num_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1

    # PAD player id; roster slots equal to this are masked out of pooling.
    pad_token: int = 0


class RosterSetEncoder(keras.Model):
    """
    Encodes a roster (fixed-length list of player IDs) into a single vector.

    Input:  (B, roster_size) int32 player IDs, PAD-filled (pad_token) for empty slots
    Output: (B, roster_dim) float roster vector

    The encoder derives its own slot mask from `ids != pad_token` and threads it
    into every SAB (so PAD slots don't contaminate set self-attention) and into the
    PMA pooling seed (so the pooled vector ignores PAD players). This keeps the call
    signature single-input, so it can be wrapped directly in TimeDistributed to run
    across a (B, SEQ, roster_size) sequence.

    Permutation invariance comes from the Set Transformer architecture, not from any
    ordering of the input ids.
    """

    def __init__(self, params: RosterEncoderParams, name: str = "roster_encoder"):
        super().__init__(name=name)
        if params.num_players <= 0:
            raise ValueError("RosterEncoderParams.num_players must be set to > 0")

        self.params = params

        self.embed = layers.Embedding(
            input_dim=params.num_players,
            output_dim=params.roster_dim,
            name="player_embedding",
        )
        self.sabs = [
            SAB(
                d_model=params.roster_dim,
                num_heads=params.num_heads,
                d_ff=params.d_ff,
                dropout=params.dropout,
                name=f"sab_{i}",
            )
            for i in range(params.num_sab_layers)
        ]
        self.pma = PMA(
            d_model=params.roster_dim,
            num_heads=params.num_heads,
            d_ff=params.d_ff,
            k_seeds=1,
            dropout=params.dropout,
            return_pooled_vector=True,
            name="pma",
        )
        self.out_ln = layers.LayerNormalization(epsilon=1e-6, name="out_ln")

    def call(self, ids, training: bool = False):
        # ids: (B, roster_size) int32
        # Per-slot validity: True where a real player sits, False for PAD.
        slot_valid = tf.not_equal(ids, self.params.pad_token)          # (B, N) bool
        # Attention mask shaped (B, 1, N): queries (rows / seed) may attend only to
        # valid key slots. Broadcasts over the query axis and over heads.
        attn_mask = slot_valid[:, tf.newaxis, :]                       # (B, 1, N)

        x = self.embed(ids)                                            # (B, N, D)
        for sab in self.sabs:
            x = sab(x, training=training, attention_mask=attn_mask)    # (B, N, D)
        v = self.pma(x, training=training, attention_mask=attn_mask)   # (B, D)
        return self.out_ln(v)

    def compute_output_shape(self, input_shape):
        # (B, roster_size) -> (B, roster_dim); lets TimeDistributed build the SEQ axis.
        return (input_shape[0], self.params.roster_dim)
