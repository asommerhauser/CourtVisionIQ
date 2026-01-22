from __future__ import annotations

from dataclasses import dataclass
from keras import layers, models, Input

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

    # Optional (keep for future padding support)
    pad_token: int = 0


class RosterSetEncoder(models.Model):
    """
    Encodes a roster (set of player IDs) into a single vector.

    Input:  (B, 5) int32 player IDs
    Output: (B, roster_dim) float roster vector

    This is a standalone sub-model that you plug into larger models.
    """

    def __init__(self, params: RosterEncoderParams, name: str = "roster_encoder"):
        if params.num_players <= 0:
            raise ValueError("RosterEncoderParams.num_players must be set to > 0")

        self.params = params

        roster_ids = Input(
            shape=(params.roster_size,),
            dtype="int32",
            name="roster_ids"
        )

        # (B, 5) -> (B, 5, D)
        x = layers.Embedding(
            input_dim=params.num_players,
            output_dim=params.roster_dim,
            name="player_embedding",
        )(roster_ids)

        # (B, 5, D) -> (B, 5, D) (contextualize within the set)
        for i in range(params.num_sab_layers):
            x = SAB(
                d_model=params.roster_dim,
                num_heads=params.num_heads,
                d_ff=params.d_ff,
                dropout=params.dropout,
                name=f"sab_{i}",
            )(x)

        # (B, 5, D) -> (B, D) (pool set to a single vector)
        roster_vec = PMA(
            d_model=params.roster_dim,
            num_heads=params.num_heads,
            d_ff=params.d_ff,
            k_seeds=1,
            dropout=params.dropout,
            return_pooled_vector=True,
            name="pma",
        )(x)

        roster_vec = layers.LayerNormalization(epsilon=1e-6, name="out_ln")(roster_vec)

        super().__init__(inputs=roster_ids, outputs=roster_vec, name=name)