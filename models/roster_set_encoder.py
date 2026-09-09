from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
import keras
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

    # How many per-player scalars ride alongside each player id. One (days of rest) through
    # 2.0's rotation work, which adds seconds in the current stint, seconds played and personal
    # fouls. They are projected together by a single Dense over a (B, N, num_scalars) stack --
    # identical to summing a projection per scalar, but the kernel's shape then encodes the
    # count, so a graph rebuilt with the wrong number fails on shapes instead of loading quietly.
    num_scalars: int = 1

    # PAD player id; roster slots equal to this are masked out of pooling.
    pad_token: int = 0


@keras.saving.register_keras_serializable(package="cviq")
class RosterSetEncoder(keras.layers.Layer):
    """
    Encodes a roster (fixed-length list of player IDs) into a single vector.

    Input:  [ids, *scalars] where
              ids     : (B, roster_size) int32 player IDs, PAD-filled (pad_token) for empties
              scalars : num_scalars tensors of (B, roster_size) float, one per per-player
                        quantity -- days of rest, and from 2.0 also seconds in the current
                        stint, seconds played and personal fouls
    Output: (B, roster_dim) float roster vector

    The scalars are projected together and added to the player embedding before the set
    transformer, so each player's state rides along with his representation (and, since the
    same encoder feeds every head, influences player selection too). PAD slots are masked
    out of attention/pooling regardless of their scalar values.

    The encoder derives its own slot mask from `ids != pad_token` and threads it into every
    SAB (so PAD slots don't contaminate set self-attention) and into the PMA pooling seed
    (so the pooled vector ignores PAD players).

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
        # Projects the per-player scalars up to roster_dim so they can be added to the player
        # embedding (mirrors the Dense projections of the model's other scalars). One Dense over
        # the stacked scalars rather than one per scalar: the sum of per-scalar projections IS a
        # single projection of their concatenation, and this way there is one layer, one name,
        # and a kernel whose first dimension is num_scalars.
        self.scalar_proj = layers.Dense(params.roster_dim, name="scalar_proj")
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

    def build(self, input_shape):
        # Force the whole subtree (embedding + scalar_proj + SABs + PMA + out_ln) to create
        # its variables now, by running one dummy pass through the same calls as call().
        # Without this the children build lazily on first call and are "never built"
        # at load time, so saved weights have nowhere to land. (Keras requires a
        # parent build() to create ALL child state.)
        dummy = tf.zeros((1, self.params.roster_size), dtype="int32")
        dummy_scalars = tf.zeros((1, self.params.roster_size, self.params.num_scalars),
                                 dtype="float32")
        mask = tf.ones((1, 1, self.params.roster_size), dtype="bool")
        x = self.embed(dummy) + self.scalar_proj(dummy_scalars)
        for sab in self.sabs:
            x = sab(x, attention_mask=mask)
        v = self.pma(x, attention_mask=mask)
        self.out_ln(v)
        super().build(input_shape)

    def call(self, inputs, training: bool = False):
        # inputs: [ids (B, N) int32, then num_scalars per-player (B, N) float tensors]
        ids, scalars = inputs[0], inputs[1:]
        if len(scalars) != self.params.num_scalars:
            raise ValueError(
                f"{self.name} expects {self.params.num_scalars} per-player scalars, "
                f"got {len(scalars)}")
        # Per-slot validity: True where a real player sits, False for PAD.
        slot_valid = tf.not_equal(ids, self.params.pad_token)          # (B, N) bool
        # Attention mask shaped (B, 1, N): queries (rows / seed) may attend only to
        # valid key slots. Broadcasts over the query axis and over heads.
        attn_mask = slot_valid[:, tf.newaxis, :]                       # (B, 1, N)

        emb = self.embed(ids)                                          # (B, N, D)
        stacked = tf.stack([tf.cast(v, emb.dtype) for v in scalars], axis=-1)   # (B, N, S)
        x = emb + self.scalar_proj(stacked)                            # (B, N, D)
        for sab in self.sabs:
            x = sab(x, training=training, attention_mask=attn_mask)    # (B, N, D)
        v = self.pma(x, training=training, attention_mask=attn_mask)   # (B, D)
        return self.out_ln(v)

    def compute_output_shape(self, input_shape):
        # [ (B, N) ids, (B, N) x num_scalars ] -> (B, roster_dim).
        ids_shape = input_shape[0]
        return (ids_shape[0], self.params.roster_dim)

    def get_config(self):
        # Flatten the frozen RosterEncoderParams dataclass so Keras can serialize it.
        cfg = super().get_config()
        cfg.update(
            {
                "roster_size": self.params.roster_size,
                "num_players": self.params.num_players,
                "roster_dim": self.params.roster_dim,
                "num_sab_layers": self.params.num_sab_layers,
                "num_heads": self.params.num_heads,
                "d_ff": self.params.d_ff,
                "dropout": self.params.dropout,
                "num_scalars": self.params.num_scalars,
                "pad_token": self.params.pad_token,
            }
        )
        return cfg

    @classmethod
    def from_config(cls, config):
        # Pull the flattened params back into a RosterEncoderParams; keep `name`.
        name = config.get("name", "roster_encoder")
        params = _config_to_params(config)
        return cls(params, name=name)


def _params_to_config(params: RosterEncoderParams) -> dict:
    return {
        "roster_size": params.roster_size,
        "num_players": params.num_players,
        "roster_dim": params.roster_dim,
        "num_sab_layers": params.num_sab_layers,
        "num_heads": params.num_heads,
        "d_ff": params.d_ff,
        "dropout": params.dropout,
        "num_scalars": params.num_scalars,
        "pad_token": params.pad_token,
    }


def _config_to_params(config: dict) -> RosterEncoderParams:
    return RosterEncoderParams(
        roster_size=config["roster_size"],
        num_players=config["num_players"],
        roster_dim=config["roster_dim"],
        num_sab_layers=config["num_sab_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        # .get, not [], for exactly one reason: models saved before 2.0's rotation work carry no
        # such key and had one scalar. Every other key is required, as before.
        num_scalars=config.get("num_scalars", 1),
        pad_token=config["pad_token"],
    )


@keras.saving.register_keras_serializable(package="cviq")
class SequenceRosterEncoder(keras.layers.Layer):
    """
    Apply a (shared) RosterSetEncoder across a time axis.

    Input:  [rosters (B, SEQ, roster_size) int32 player IDs,
             then num_scalars tensors of (B, SEQ, roster_size) float per-player scalars]
    Output: (B, SEQ, roster_dim)  float

    Implemented with an explicit reshape -> encode -> reshape instead of
    `TimeDistributed`. In graph mode (model.fit) TimeDistributed unrolls the SEQ
    axis, which for SEQ=600 explodes the training graph and exhausts host RAM.
    Collapsing (B, SEQ, N) -> (B*SEQ, N), encoding once, and reshaping back keeps
    the graph a single application and is memory-flat. The same instance is applied
    to both rosters, so home/away stay weight-tied.
    """

    def __init__(self, params: RosterEncoderParams, name: str = "roster_vec", **kwargs):
        super().__init__(name=name, **kwargs)
        self.params = params
        self.encoder = RosterSetEncoder(params)

    def build(self, input_shape):
        # Build the inner encoder for [ (·, N) ids, (·, N) per scalar ] so its weights exist
        # before any weight load. The list length has to track num_scalars, not the two it was
        # fixed at while rest was the only scalar.
        n = self.params.roster_size
        self.encoder.build([(None, n)] * (1 + self.params.num_scalars))
        super().build(input_shape)

    def call(self, inputs, training: bool = False):
        rosters, scalars = inputs[0], inputs[1:]                # each (B, SEQ, N)
        n = self.params.roster_size
        s = tf.shape(rosters)                                   # (B, SEQ, N)
        flat = [tf.reshape(rosters, (-1, n))]                   # (B*SEQ, N)
        flat += [tf.reshape(v, (-1, n)) for v in scalars]       # (B*SEQ, N) each
        v = self.encoder(flat, training=training)               # (B*SEQ, D)
        return tf.reshape(v, (s[0], s[1], self.params.roster_dim))  # (B, SEQ, D)

    def compute_output_shape(self, input_shape):
        rosters_shape = input_shape[0]
        return (rosters_shape[0], rosters_shape[1], self.params.roster_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(_params_to_config(self.params))
        return cfg

    @classmethod
    def from_config(cls, config):
        name = config.get("name", "roster_vec")
        return cls(_config_to_params(config), name=name)
