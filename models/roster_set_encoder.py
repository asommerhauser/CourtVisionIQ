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

    # PAD player id; roster slots equal to this are masked out of pooling.
    pad_token: int = 0


@keras.saving.register_keras_serializable(package="cviq")
class RosterSetEncoder(keras.layers.Layer):
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

    def build(self, input_shape):
        # Force the whole subtree (embedding + SABs + PMA + out_ln) to create its
        # variables now, by running one dummy pass through the same calls as call().
        # Without this the children build lazily on first call and are "never built"
        # at load time, so saved weights have nowhere to land. (Keras requires a
        # parent build() to create ALL child state.)
        dummy = tf.zeros((1, self.params.roster_size), dtype="int32")
        mask = tf.ones((1, 1, self.params.roster_size), dtype="bool")
        x = self.embed(dummy)
        for sab in self.sabs:
            x = sab(x, attention_mask=mask)
        v = self.pma(x, attention_mask=mask)
        self.out_ln(v)
        super().build(input_shape)

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
        pad_token=config["pad_token"],
    )


@keras.saving.register_keras_serializable(package="cviq")
class SequenceRosterEncoder(keras.layers.Layer):
    """
    Apply a (shared) RosterSetEncoder across a time axis.

    Input:  (B, SEQ, roster_size) int32 player IDs
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
        # Build the inner encoder for a (·, roster_size) batch so its weights exist
        # before any weight load.
        self.encoder.build((None, self.params.roster_size))
        super().build(input_shape)

    def call(self, rosters, training: bool = False):
        s = tf.shape(rosters)                                   # (B, SEQ, N)
        flat = tf.reshape(rosters, (-1, self.params.roster_size))   # (B*SEQ, N)
        v = self.encoder(flat, training=training)              # (B*SEQ, D)
        return tf.reshape(v, (s[0], s[1], self.params.roster_dim))  # (B, SEQ, D)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.params.roster_dim)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(_params_to_config(self.params))
        return cfg

    @classmethod
    def from_config(cls, config):
        name = config.get("name", "roster_vec")
        return cls(_config_to_params(config), name=name)
