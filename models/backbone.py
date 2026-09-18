"""The causal transformer backbone every head shares.

Six heads -- event/time, player, conditional-time, conditional-type, substitution and
stint-length -- differ only in what they feed the fusion and what they hang off the top. The
stack in between was a **byte-identical** copy in all six files: the fusion concat, its
projection to ``model_dim``, the learned positional embedding, the key-padding mask, N
pre-norm transformer blocks, and a final layer norm. Six copies meant six places to keep in
step, and §9's local-attention masking would have had to land in each of them separately.

This module owns that stack -- which is also what lets §9's local attention land in one place
instead of six. Layer names are carried over verbatim, because weight reload matches **by name**: ``from_artifacts`` rebuilds the graph from config and restores into it,
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

import config
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


@keras.saving.register_keras_serializable(package="cviq")
class BandedAttentionMask(layers.Layer):
    """Per-head attention mask: the first ``local_heads`` heads see only a trailing window.

    Attention builds each row from a weighted average over every earlier row, and nothing pushes
    any head toward the last few -- but basketball is overwhelmingly local. Restricting a minority
    of heads to a short window gives the block a recency bias without taking global context away
    from the rest, and it needs no new weights and no custom kernel: it is the same masking
    mechanism as the padding mask, one axis wider.

    Turns a (B, SEQ) float pad-mask into a (B, H, SEQ, SEQ) boolean mask. Local heads get
    ``band AND pad``, global heads get ``pad`` alone. Only the band's *lower* edge is applied --
    ``MultiHeadAttention(use_causal_mask=True)`` already forbids attending forward, so the upper
    edge would be redundant.

    Costs one (B, H, SEQ, SEQ) bool tensor -- about 176 MB at SEQ=600, H=8, batch 64. It is built
    once outside the block loop and shared by every block, the same lifetime the plain padding
    mask already has.
    """

    def __init__(self, num_heads: int, local_heads: int, window: int, **kwargs):
        super().__init__(**kwargs)
        if not 0 < local_heads <= num_heads:
            raise ValueError(
                f"local_heads must be in (0, {num_heads}], got {local_heads}. "
                "Zero local heads is the plain KeyPaddingMask path, not this layer.")
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        self.num_heads = num_heads
        self.local_heads = local_heads
        self.window = window

    def call(self, m):
        pad = tf.cast(m, "bool")                             # (B, SEQ)
        seq = tf.shape(pad)[1]
        i = tf.range(seq)[:, tf.newaxis]
        j = tf.range(seq)[tf.newaxis, :]
        band = (i - j) < self.window                         # (SEQ, SEQ), lower edge only
        is_local = tf.range(self.num_heads) < self.local_heads
        per_head = tf.where(is_local[:, tf.newaxis, tf.newaxis],
                            band[tf.newaxis], tf.ones_like(band)[tf.newaxis])
        return per_head[tf.newaxis] & pad[:, tf.newaxis, tf.newaxis, :]

    def compute_output_shape(self, input_shape):
        seq = input_shape[1]
        return (input_shape[0], self.num_heads, seq, seq)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_heads": self.num_heads, "local_heads": self.local_heads,
                    "window": self.window})
        return cfg


def _attention_mask(pad_mask, num_heads):
    """The mask every block attends under -- banded per-head, or the plain padding mask.

    Both carry the layer name ``attn_pad_mask``, so flipping the switch perturbs no layer
    naming and the reload contract holds either way; neither layer has weights, and the full
    ``.keras`` reload records the class in its config.

    ``config.LOCAL_ATTENTION_HEADS`` is read here, at build time, rather than imported at module
    load, so a test can set it and rebuild.
    """
    local = int(getattr(config, "LOCAL_ATTENTION_HEADS", 0) or 0)
    if local <= 0:
        return KeyPaddingMask(name="attn_pad_mask")(pad_mask)
    return BandedAttentionMask(
        num_heads, local, int(config.LOCAL_ATTENTION_WINDOW), name="attn_pad_mask",
    )(pad_mask)


def _film_context(parts, *, name="film"):
    """One game-context vector for FiLM, or ``None`` when the feature is off.

    ``parts`` are per-timestep tensors -- the season embedding, the team priors, the regime latent --
    already ``(B, SEQ, ...)``, so the modulation is per row with no broadcasting to arrange. They are
    summarised through a bottleneck so the per-block projections stay small; see ``config.FILM_DIM``.
    """
    if not parts or not getattr(config, "FILM_ENABLED", False):
        return None
    ctx = parts[0] if len(parts) == 1 else layers.Concatenate(axis=-1, name=f"{name}_concat")(parts)
    dim = int(getattr(config, "FILM_DIM", 64))
    return layers.Dense(dim, activation="gelu", name=f"{name}_ctx")(ctx)


def _film(h, ctx, block: int, slot: int, d_model: int):
    """``h * (1 + gamma) + beta``, with ``gamma`` and ``beta`` predicted from the game context.

    Zero-initialised on purpose, kernel AND bias: at initialisation gamma and beta are exactly zero, so
    this is the identity and a FiLM graph starts numerically identical to one built without it. The
    alternative -- predicting the scale directly -- perturbs the residual stream before training begins
    and makes the A/B a comparison of two different initialisations.

    Applied to the NORMALISED branch, never to ``x`` itself. Modulating ``x`` would scale the residual
    identity path, which is the thing that makes a deep stack trainable.
    """
    if ctx is None:
        return h
    zeros = dict(kernel_initializer="zeros", bias_initializer="zeros")
    gamma = layers.Dense(d_model, name=f"block{block}_film{slot}_scale", **zeros)(ctx)
    beta = layers.Dense(d_model, name=f"block{block}_film{slot}_shift", **zeros)(ctx)
    scaled = layers.Multiply(name=f"block{block}_film{slot}_mul")([h, gamma])
    return layers.Add(name=f"block{block}_film{slot}")([h, scaled, beta])


def build_backbone(parts, pad_mask, *, seq_len, d_model,
                   num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=0.2,
                   film_context=None):
    """Fuse ``parts``, add position, and run the causal transformer stack.

    ``parts`` is the head's own list of per-timestep tensors to concatenate -- token
    embeddings, the conditioning vectors it carries, the roster vectors, the continuous
    projections. It is the *only* thing that differs between the six heads; everything from
    the concat down is identical, which is why it lives here.

    ``pad_mask`` is the (B, SEQ) float mask (1 real / 0 pad). Attention is causal
    (``use_causal_mask=True``) on top of the key-padding mask, and when
    ``config.LOCAL_ATTENTION_HEADS`` is non-zero a band restricts that many heads per block to a
    trailing window -- see :class:`BandedAttentionMask`.

    Returns the (B, SEQ, ``d_model``) encoded sequence, layer-normalized, ready for the
    head's own output layers.
    """
    x = layers.Concatenate(axis=-1, name="fusion_concat")(parts)
    x = layers.Dense(d_model, name="fusion_projection")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name="fusion_ln")(x)

    # ---- Positional encoding (learned) ----
    x = AddPositionalEmbedding(seq_len, d_model, name="positional_embedding")(x)
    x = layers.Dropout(dropout, name="emb_dropout")(x)

    # ---- Attention mask: key padding, plus the per-head band when local heads are on ----
    attn_mask = _attention_mask(pad_mask, num_heads)

    # ---- Context modulation (3.2 W6): one vector, a scale and shift per block ----
    ctx = _film_context(film_context)

    # ---- Causal transformer encoder ----
    for i in range(num_layers):
        h = layers.LayerNormalization(epsilon=1e-6, name=f"block{i}_ln1")(x)
        h = _film(h, ctx, i, 1, d_model)
        attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout,
            name=f"block{i}_mha",
        )(h, h, attention_mask=attn_mask, use_causal_mask=True)
        x = layers.Add(name=f"block{i}_res1")([x, attn])

        h = layers.LayerNormalization(epsilon=1e-6, name=f"block{i}_ln2")(x)
        h = _film(h, ctx, i, 2, d_model)
        f1 = layers.Dense(ff_dim, activation="gelu", name=f"block{i}_ff1")(h)
        f1 = layers.Dropout(dropout, name=f"block{i}_ffdrop")(f1)
        f2 = layers.Dense(d_model, name=f"block{i}_ff2")(f1)
        x = layers.Add(name=f"block{i}_res2")([x, f2])

    return layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)
