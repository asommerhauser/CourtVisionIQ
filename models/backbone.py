"""The causal transformer backbone every head shares.

Six heads -- event/time, player, conditional-time, conditional-type, substitution and
stint-length -- differ only in what they feed the fusion and what they hang off the top. The
stack in between was a **byte-identical** copy in all six files: the fusion concat, its
projection to ``model_dim``, the learned positional embedding, the key-padding mask, N
pre-norm transformer blocks, and a final layer norm. Six copies meant six places to keep in
step, and §9's local-attention masking would have had to land in each of them separately.

This module owns that stack. Layer names are carried over verbatim, because weight reload
matches **by name**: ``from_artifacts`` rebuilds the graph from config and restores into it,
and the single-file ``<key>.keras`` reload does the same. A rename here is silent at build
time and surfaces later as a failed -- or partially restored -- reload.
``scripts/dump_layer_names.py`` exists to check exactly that.

The two custom layers live here too, since they are the stack's own pieces. They keep their
``@keras.saving.register_keras_serializable(package="cviq")`` registration, whose key is
``cviq>ClassName`` and does not include the module path, so models saved before this move
still reload.
"""
from __future__ import annotations

import tensorflow as tf
import keras
from keras import layers

from config import NUM_LAYERS, NUM_HEADS, FF_DIM


@keras.saving.register_keras_serializable(package="cviq")
class AddPositionalEmbedding(layers.Layer):
    """Add a learned position embedding over [0, seq_len) to a (B, SEQ, D) tensor."""

    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model

    def build(self, input_shape):
        # Own variable (created in build, not a nested Embedding) so it serializes
        # and reloads cleanly. Shape (SEQ, D) broadcasts over the batch axis.
        self.pos = self.add_weight(
            name="pos_table",
            shape=(self.seq_len, self.d_model),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos                         # (SEQ, D) broadcasts over (B, SEQ, D)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return cfg


@keras.saving.register_keras_serializable(package="cviq")
class KeyPaddingMask(layers.Layer):
    """Turn a (B, SEQ) float pad-mask into a (B, 1, SEQ) boolean key-padding mask.

    A registered layer (rather than a Lambda) so the full .keras model reloads
    under Keras 3 safe mode without custom code execution.
    """

    def call(self, m):
        return tf.cast(m, "bool")[:, tf.newaxis, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[1])


def build_backbone(parts, pad_mask, *, seq_len, d_model,
                   num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=0.2):
    """Fuse ``parts``, add position, and run the causal transformer stack.

    ``parts`` is the head's own list of per-timestep tensors to concatenate -- token
    embeddings, the conditioning vectors it carries, the roster vectors, the continuous
    projections. It is the *only* thing that differs between the six heads; everything from
    the concat down is identical, which is why it lives here.

    ``pad_mask`` is the (B, SEQ) float mask (1 real / 0 pad). Attention is causal
    (``use_causal_mask=True``) on top of the key-padding mask.

    Returns the (B, SEQ, ``d_model``) encoded sequence, layer-normalized, ready for the
    head's own output layers.
    """
    x = layers.Concatenate(axis=-1, name="fusion_concat")(parts)
    x = layers.Dense(d_model, name="fusion_projection")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="fusion_ln")(x)

    # ---- Positional encoding (learned) ----
    x = AddPositionalEmbedding(seq_len, d_model, name="positional_embedding")(x)
    x = layers.Dropout(dropout, name="emb_dropout")(x)

    # ---- Attention mask: (B, 1, SEQ) boolean key-padding mask ----
    attn_mask = KeyPaddingMask(name="attn_pad_mask")(pad_mask)

    # ---- Causal transformer encoder ----
    for i in range(num_layers):
        h = layers.LayerNormalization(epsilon=1e-6, name=f"block{i}_ln1")(x)
        attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout,
            name=f"block{i}_mha",
        )(h, h, attention_mask=attn_mask, use_causal_mask=True)
        x = layers.Add(name=f"block{i}_res1")([x, attn])

        h = layers.LayerNormalization(epsilon=1e-6, name=f"block{i}_ln2")(x)
        f1 = layers.Dense(ff_dim, activation="gelu", name=f"block{i}_ff1")(h)
        f1 = layers.Dropout(dropout, name=f"block{i}_ffdrop")(f1)
        f2 = layers.Dense(d_model, name=f"block{i}_ff2")(f1)
        x = layers.Add(name=f"block{i}_res2")([x, f2])

    return layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)
