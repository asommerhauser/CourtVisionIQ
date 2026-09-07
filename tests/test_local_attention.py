"""Banded local attention: two heads per block see only the last few rows.

Attention averages each row over every earlier row and nothing pushes any head toward the last
few, so §9 restricts a minority of heads to a trailing window. The mechanism is the padding mask
one axis wider -- a (B, H, SEQ, SEQ) boolean instead of (B, 1, SEQ) -- so these tests check the
mask the layer actually emits, plus the two invariants that make the setting safe to ship: with
the switch off the graph is exactly what it was before, and the setting is recorded as
architecture so a mismatched reload cannot pass silently.

``config.LOCAL_ATTENTION_HEADS`` is read at build time, so each test sets it and rebuilds.
"""
from __future__ import annotations

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import Input

import config
from models import manifest
from models.backbone import BandedAttentionMask, KeyPaddingMask, build_backbone

SEQ = 12
D = 16
FF = 24
HEADS = 4
WINDOW = 4


@pytest.fixture
def local_attention(monkeypatch):
    """Set the switch for one test; the module reads config at build time."""
    def _set(heads, window=WINDOW):
        monkeypatch.setattr(config, "LOCAL_ATTENTION_HEADS", heads)
        monkeypatch.setattr(config, "LOCAL_ATTENTION_WINDOW", window)
    return _set


def _backbone_model(num_heads=HEADS):
    keras.utils.set_random_seed(0)   # deterministic init: the probe below compares two builds
    a = Input(shape=(SEQ, 5), dtype="float32", name="a")
    pad = Input(shape=(SEQ,), dtype="float32", name="pad_mask")
    out = build_backbone([a], pad, seq_len=SEQ, d_model=D,
                         num_layers=1, num_heads=num_heads, ff_dim=FF, dropout=0.0)
    return keras.Model(inputs=[a, pad], outputs=out, name="LocalBackbone")


def _mask(local_heads, window=WINDOW, pad=None, num_heads=HEADS):
    """The (B, H, SEQ, SEQ) mask the layer emits for one all-real (or given) pad row."""
    if pad is None:
        pad = np.ones((1, SEQ), np.float32)
    layer = BandedAttentionMask(num_heads, local_heads, window)
    return np.asarray(layer(tf.constant(pad)))


# --------------------------------------------------------------------------- #
# --- The mask itself                                                       -- #
# --------------------------------------------------------------------------- #

def test_local_heads_see_only_the_window():
    m = _mask(local_heads=2)[0]                       # (H, SEQ, SEQ)
    i, j = np.indices((SEQ, SEQ))
    expected = (i - j) < WINDOW                       # lower edge only; causality is MHA's job
    for h in range(2):
        assert (m[h] == expected).all(), f"head {h} is not banded to the window"


def test_global_heads_see_every_real_row():
    m = _mask(local_heads=2)[0]
    for h in range(2, HEADS):
        assert m[h].all(), f"head {h} should be unrestricted"


def test_the_split_is_the_first_n_heads():
    for local in (1, 3):
        m = _mask(local_heads=local)[0]
        banded = [h for h in range(HEADS) if not m[h].all()]
        assert banded == list(range(local))


def test_padding_is_masked_out_for_every_head():
    pad = np.ones((1, SEQ), np.float32)
    pad[0, SEQ - 3:] = 0.0
    m = _mask(local_heads=2, pad=pad)[0]
    assert not m[:, :, SEQ - 3:].any(), "a padded key is visible to some head"
    # ...and real keys inside the window survive the AND.
    assert m[0, SEQ - 4, SEQ - 4]


def test_window_of_one_is_the_diagonal():
    m = _mask(local_heads=1, window=1)[0]
    assert (m[0] == np.eye(SEQ, dtype=bool)).all()


def test_a_local_head_count_outside_the_head_count_is_rejected():
    with pytest.raises(ValueError, match="local_heads"):
        BandedAttentionMask(HEADS, HEADS + 1, WINDOW)
    with pytest.raises(ValueError, match="local_heads"):
        BandedAttentionMask(HEADS, 0, WINDOW)      # zero is the KeyPaddingMask path, not this one
    with pytest.raises(ValueError, match="window"):
        BandedAttentionMask(HEADS, 1, 0)


def test_the_layer_round_trips_through_its_config():
    layer = BandedAttentionMask(HEADS, 2, WINDOW, name="attn_pad_mask")
    clone = BandedAttentionMask.from_config(layer.get_config())
    assert (clone.num_heads, clone.local_heads, clone.window) == (HEADS, 2, WINDOW)


# --------------------------------------------------------------------------- #
# --- The switch                                                            -- #
# --------------------------------------------------------------------------- #

def test_the_switch_off_rebuilds_the_original_graph(local_attention):
    """LOCAL_ATTENTION_HEADS = 0 must be a genuine no-op, or the A/B is not a comparison."""
    local_attention(0)
    model = _backbone_model()
    layer = model.get_layer("attn_pad_mask")
    assert isinstance(layer, KeyPaddingMask)
    assert tuple(layer.output.shape) == (None, 1, SEQ)


def test_the_switch_on_bands_the_mask(local_attention):
    local_attention(2)
    model = _backbone_model()
    layer = model.get_layer("attn_pad_mask")
    assert isinstance(layer, BandedAttentionMask)
    assert (layer.local_heads, layer.window) == (2, WINDOW)
    assert tuple(layer.output.shape) == (None, HEADS, SEQ, SEQ)


def test_the_mask_layer_keeps_its_name_either_way(local_attention):
    """The name is the reload contract; flipping the switch must not perturb it."""
    for heads in (0, 2):
        local_attention(heads)
        names = [l.name for l in _backbone_model().layers]
        assert names.count("attn_pad_mask") == 1


def test_local_heads_change_what_the_model_computes(local_attention):
    """A row outside the window must stop reaching a later row once the band is on.

    The probe is within one graph, so it does not depend on the two builds sharing weights:
    perturb row 0, then read row SEQ-1, which sits outside a WINDOW-row window. One attention
    layer, and everything else in the block is per-position, so with every head banded row 0
    cannot reach row SEQ-1 at all and the output is unchanged; with every head global it must
    move.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1, SEQ, 5)).astype(np.float32)
    a2 = a.copy()
    a2[0, 0] = 50.0
    pad = np.ones((1, SEQ), np.float32)

    def _last_row_delta(local_heads):
        local_attention(local_heads)
        model = _backbone_model()
        base = np.asarray(model({"a": a, "pad_mask": pad}, training=False))
        moved = np.asarray(model({"a": a2, "pad_mask": pad}, training=False))
        return np.abs(base[0, -1] - moved[0, -1]).max()

    assert _last_row_delta(HEADS) < 1e-5, "a banded head still reached outside its window"
    assert _last_row_delta(0) > 1e-5, "the global baseline did not react at all"


# --------------------------------------------------------------------------- #
# --- The reload guard                                                      -- #
# --------------------------------------------------------------------------- #

def test_the_setting_is_recorded_as_architecture():
    """It changes no weight shapes, so only the manifest can catch a mismatched reload."""
    assert "LOCAL_ATTENTION_HEADS" in manifest.ARCH_KEYS
    assert "LOCAL_ATTENTION_WINDOW" in manifest.ARCH_KEYS
    snap = manifest.arch_snapshot()
    assert snap["LOCAL_ATTENTION_HEADS"] == config.LOCAL_ATTENTION_HEADS
    assert snap["LOCAL_ATTENTION_WINDOW"] == config.LOCAL_ATTENTION_WINDOW


def test_the_setting_is_not_a_rollout_dial():
    """Nothing at sim time reads it, and it is not A/B-able without a retrain."""
    assert "LOCAL_ATTENTION_HEADS" not in config._TUNING_KEYS
    assert "LOCAL_ATTENTION_WINDOW" not in config._TUNING_KEYS
