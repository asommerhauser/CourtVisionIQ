"""
Cross-roster attention (3.2 W7).

The load-bearing property is :func:`test_the_home_vector_depends_on_the_opposing_lineup` -- with a
companion asserting the *opposite* for the un-crossed encoder, because a test that only checks the new
path cannot tell "cross-attention works" from "something else made the numbers move".

Worth noting what did not exist before this file: there is no ``tests/test_mab.py``, ``test_sab.py`` or
``test_pma.py``. ``MAB`` had never been called in true cross mode -- ``SAB`` passes ``Y = X`` and ``PMA``
passes learned seeds -- so the ``X`` attends to ``Y`` path it was written for was entirely unexercised.
"""
import numpy as np
import pytest
import keras
from keras import layers

import config
from layers.cross_roster import CrossRosterBlock
from models.roster_set_encoder import (
    CrossSequenceRosterEncoder,
    RosterEncoderParams,
    SequenceRosterEncoder,
    build_sequence_roster_encoder,
    encode_both_rosters,
)

SEQ, N, S, D = 4, 5, 3, 16


def _params(**kw):
    base = dict(roster_size=N, num_players=20, roster_dim=D, num_sab_layers=1,
                num_heads=2, d_ff=32, dropout=0.0, num_scalars=S)
    base.update(kw)
    return RosterEncoderParams(**base)


def _graph(cross: bool):
    home = layers.Input(shape=(SEQ, N), dtype="int32", name="home")
    away = layers.Input(shape=(SEQ, N), dtype="int32", name="away")
    hs = [layers.Input(shape=(SEQ, N), name=f"hs{i}") for i in range(S)]
    as_ = [layers.Input(shape=(SEQ, N), name=f"as{i}") for i in range(S)]
    params = _params()
    enc = CrossSequenceRosterEncoder(params) if cross else SequenceRosterEncoder(params)
    home_vec, away_vec = encode_both_rosters(enc, [home, *hs], [away, *as_])
    return keras.Model([home, away, *hs, *as_], [home_vec, away_vec], name="Pair")


def _inputs(rng, home=None, away=None, n=1):
    home = rng.integers(1, 20, size=(n, SEQ, N)).astype("int32") if home is None else home
    away = rng.integers(1, 20, size=(n, SEQ, N)).astype("int32") if away is None else away
    scalars = [np.zeros((n, SEQ, N), dtype="float32") for _ in range(2 * S)]
    return [home, away, *scalars]


# --------------------------------------------------------------------------- the block itself

def test_the_block_returns_its_query_shape():
    """``MAB`` has no ``compute_output_shape``, which is one of three reasons it needs a wrapper."""
    block = CrossRosterBlock(d_model=D, num_heads=2, d_ff=32)
    assert tuple(block.compute_output_shape((None, N, D))) == (None, N, D)


def test_the_block_builds_its_variables_up_front():
    """Lazily-built children read as "never built" at load time, so saved weights have nowhere to land.

    ``RosterSetEncoder.build`` solves the same problem the same way and for the same reason.
    """
    block = CrossRosterBlock(d_model=D, num_heads=2, d_ff=32)
    assert not block.weights
    block.build((None, N, D))
    assert block.weights, "build() must create the MAB subtree's variables"


def test_a_fully_padded_far_roster_does_not_produce_nan():
    """A fully-masked query row makes softmax divide by zero.

    The result is discarded at pooling anyway, so falling back to an unmasked row is harmless -- and a
    NaN spreading through a pooled vector is not a failure anyone would trace back to here. Real rosters
    always have five, but an all-PAD roster is constructible.
    """
    model = _graph(cross=True)
    rng = np.random.default_rng(0)
    pad = np.zeros((1, SEQ, N), dtype="int32")
    home_vec, away_vec = model.predict(_inputs(rng, away=pad), verbose=0)
    assert np.isfinite(home_vec).all() and np.isfinite(away_vec).all()


# --------------------------------------------------------------------------- what it buys

def test_the_home_vector_depends_on_the_opposing_lineup():
    """**The property W7 exists for.**

    Before this, the two rosters passed through one weight-tied encoder independently and met only at
    the fusion concat, so nothing in the graph could represent one lineup *against* another.
    """
    model = _graph(cross=True)
    rng = np.random.default_rng(1)
    home = rng.integers(1, 20, size=(1, SEQ, N)).astype("int32")
    away_a = rng.integers(1, 20, size=(1, SEQ, N)).astype("int32")
    away_b = rng.integers(1, 20, size=(1, SEQ, N)).astype("int32")

    with_a, _ = model.predict(_inputs(rng, home=home, away=away_a), verbose=0)
    with_b, _ = model.predict(_inputs(rng, home=home, away=away_b), verbose=0)
    assert not np.allclose(with_a, with_b, atol=1e-6)


def test_without_cross_attention_it_does_not():
    """The companion that makes the test above mean something."""
    model = _graph(cross=False)
    rng = np.random.default_rng(1)
    home = rng.integers(1, 20, size=(1, SEQ, N)).astype("int32")
    away_a = rng.integers(1, 20, size=(1, SEQ, N)).astype("int32")
    away_b = rng.integers(1, 20, size=(1, SEQ, N)).astype("int32")

    with_a, _ = model.predict(_inputs(rng, home=home, away=away_a), verbose=0)
    with_b, _ = model.predict(_inputs(rng, home=home, away=away_b), verbose=0)
    np.testing.assert_allclose(with_a, with_b, atol=1e-6)


def test_the_two_directions_share_one_block():
    """Shared weights are what make it learn how a lineup reads an opponent.

    Per-direction weights would let it memorise which side of the ledger a team sits on, which is the
    thing the home/away framing already over-encodes.
    """
    params = _params()
    enc = CrossSequenceRosterEncoder(params)
    enc.build([(None, SEQ, N)] * (2 * (1 + S)))
    blocks = [l for l in enc._flatten_layers() if isinstance(l, CrossRosterBlock)]
    assert len(blocks) == 1


def test_both_sides_keep_their_shape_and_stay_finite():
    model = _graph(cross=True)
    rng = np.random.default_rng(2)
    home_vec, away_vec = model.predict(_inputs(rng, n=2), verbose=0)
    assert home_vec.shape == away_vec.shape == (2, SEQ, D)
    assert np.isfinite(home_vec).all() and np.isfinite(away_vec).all()


def test_the_wrong_number_of_tensors_is_refused():
    """Two sides' lists arrive concatenated, so a miscount is silent unless it is checked."""
    enc = CrossSequenceRosterEncoder(_params())
    with pytest.raises(ValueError, match="expects 8 tensors"):
        enc([np.zeros((1, SEQ, N), dtype="int32")] * 7)


# --------------------------------------------------------------------------- the seam it replaced

def test_splitting_the_encoder_did_not_change_what_it_computes():
    """``call`` is now ``encode_slots`` then ``pool_slots``, and must be exactly what it was."""
    from models.roster_set_encoder import RosterSetEncoder
    enc = RosterSetEncoder(_params())
    ids = np.array([[1, 2, 3, 4, 0]], dtype="int32")
    scalars = [np.full((1, N), 0.5, dtype="float32") for _ in range(S)]
    direct = enc([ids, *scalars])
    x, mask, valid = enc.encode_slots([ids, *scalars])
    staged = enc.pool_slots(x, mask)
    np.testing.assert_allclose(np.asarray(direct), np.asarray(staged), atol=1e-6)
    assert tuple(valid.shape) == (1, N) and bool(valid.numpy()[0][-1]) is False


def test_the_factory_follows_the_flag(monkeypatch):
    """One place decides which encoder a build uses, so the heads cannot disagree."""
    monkeypatch.setattr(config, "CROSS_ROSTER_ENABLED", True)
    assert isinstance(build_sequence_roster_encoder(_params()), CrossSequenceRosterEncoder)
    monkeypatch.setattr(config, "CROSS_ROSTER_ENABLED", False)
    assert isinstance(build_sequence_roster_encoder(_params()), SequenceRosterEncoder)


def test_the_helper_dispatches_on_the_encoder_it_is_given():
    """``encode_both_rosters`` is the one branch; six copies of it is how lists drift out of step."""
    rng = np.random.default_rng(3)
    for cross in (False, True):
        model = _graph(cross)
        home_vec, away_vec = model.predict(_inputs(rng), verbose=0)
        assert home_vec.shape == (1, SEQ, D)


def test_cross_roster_is_recorded_in_the_arch_keys():
    """Off produces a graph with different layers, which ``load_weights`` matches by name and skips."""
    from models.manifest import ARCH_KEYS
    assert "CROSS_ROSTER_ENABLED" in ARCH_KEYS


def test_the_layer_round_trips_through_its_config():
    """``RosterEncoderParams`` is a frozen dataclass flattened for serialization; a new layer that does
    not round-trip breaks weight reload rather than failing here."""
    enc = CrossSequenceRosterEncoder(_params())
    clone = CrossSequenceRosterEncoder.from_config(enc.get_config())
    assert clone.params == enc.params
    assert clone.name == enc.name

    block = CrossRosterBlock(d_model=D, num_heads=2, d_ff=32, dropout=0.1)
    cfg = block.get_config()
    twin = CrossRosterBlock.from_config(cfg)
    assert (twin.d_model, twin.num_heads, twin.d_ff, twin.dropout_rate) == (D, 2, 32, 0.1)
