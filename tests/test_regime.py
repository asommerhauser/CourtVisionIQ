"""
The per-game regime latent: the sampling contract, and the training wrapper.

Why it exists is measured, not assumed: across the sims of one game the simulator's two teams have
corr(home pts, away pts) = 0.02 against a real 0.35, which makes its margin sd 16.5 against a real
13.7 AND its total sd 16.7 against a real 19.7 -- one missing shared term seen from both sides. The
latent is that term.

The tests that matter here are the ones about *what survives training*. The embedding table is a
training device; the only thing a rollout ever sees is the per-dimension spread written into
norm_stats, and a model trained without a latent has to keep behaving exactly as it did.
"""
from __future__ import annotations

import numpy as np
import pytest

from config import REGIME_DIM
from models.regime import (
    GAME_INDEX_KEY,
    REGIME_KEY,
    append_regime_batches,
    regime_std,
    sample_regime,
)


# --------------------------------------------------------------------------- sampling

def test_the_spread_the_rollout_samples_from_is_the_tables_own_spread():
    rng = np.random.default_rng(0)
    table = rng.normal(scale=[0.5, 0.2, 0.1, 0.05], size=(4000, REGIME_DIM))
    std = regime_std(table)
    assert std == pytest.approx([0.5, 0.2, 0.1, 0.05], abs=0.03)


def test_a_constant_offset_is_not_a_regime():
    """A shift every game shares is something the heads absorb into a bias. Sampling around it
    would add noise with no shared structure, so the mean comes out before the sd is taken."""
    table = np.tile(np.array([9.0, -4.0, 1.0, 0.0]), (100, 1))
    assert regime_std(table) == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_a_table_with_one_row_carries_no_spread_rather_than_a_nan():
    assert regime_std(np.zeros((1, REGIME_DIM))) == [0.0] * REGIME_DIM


def test_draws_match_the_stated_spread():
    rng = np.random.default_rng(1)
    std = [0.4, 0.2, 0.1, 0.05]
    draws = np.array([sample_regime(rng, std) for _ in range(5000)])
    assert draws.std(axis=0) == pytest.approx(std, abs=0.02)
    assert draws.mean(axis=0) == pytest.approx([0.0] * REGIME_DIM, abs=0.02)


def test_a_model_trained_without_a_latent_draws_exactly_zero():
    """The compatibility contract. A pre-3.0 checkpoint has no regime_std in its norm_stats, and
    its rollout must be bit-identical to what it was -- a zero plane contributes nothing through
    regime_proj beyond its bias, which that model's weights already account for."""
    rng = np.random.default_rng(2)
    assert sample_regime(rng, ()).tolist() == [0.0] * REGIME_DIM
    assert sample_regime(rng, [0.0] * REGIME_DIM).tolist() == [0.0] * REGIME_DIM


def test_a_malformed_spread_is_treated_as_no_latent_rather_than_crashing_a_rollout():
    """norm_stats is JSON written by a train; a wrong-length list must not take a 36-hour eval down."""
    rng = np.random.default_rng(3)
    assert sample_regime(rng, [0.4, 0.2]).tolist() == [0.0] * REGIME_DIM


# --------------------------------------------------------------------------- the npz columns

def test_the_stored_plane_is_zeros_and_the_index_is_the_games_position():
    """Zeros on disk is deliberate: there is no per-game value until a table has been fitted, and
    train_step overwrites the plane anyway. Storing a sample would make the npz claim otherwise."""
    batches = {REGIME_KEY: [], GAME_INDEX_KEY: []}
    for pos in range(3):
        append_regime_batches(batches, pos, seq_len=7)
    assert len(batches[REGIME_KEY]) == 3
    assert batches[REGIME_KEY][0].shape == (7, REGIME_DIM)
    assert not batches[REGIME_KEY][0].any()
    assert [int(v[0]) for v in batches[GAME_INDEX_KEY]] == [0, 1, 2]


# --------------------------------------------------------------------------- the training wrapper

def test_the_wrapper_is_a_no_op_when_the_latent_is_off(monkeypatch):
    import config
    from models.regime import build_regime_model

    monkeypatch.setattr(config, "REGIME_ENABLED", False)
    sentinel = object()
    assert build_regime_model(sentinel, 100) is sentinel


def test_the_wrapper_is_a_no_op_with_nothing_to_index():
    from models.regime import build_regime_model

    sentinel = object()
    assert build_regime_model(sentinel, 1) is sentinel


def test_the_table_learns_the_per_game_component_the_other_inputs_cannot_explain():
    """The mechanism, end to end on eight synthetic games.

    Each game's target is a constant offset; the only non-latent input is noise. If the latent is
    wired up, the table absorbs the offsets and training loss collapses. It also pins the two things
    that would otherwise fail silently: the reported loss must not be a flat zero (without the loss
    tracker, EarlyStopping -- this repo's only checkpoint selector -- would see no improvement and
    restore epoch 1), and validation, which does NOT look the row up, must stay high because that is
    what a rollout actually gets.
    """
    import tensorflow as tf
    from tensorflow import keras

    from models.regime import build_regime_model

    seq, n = 4, 16
    regime_in = keras.Input(shape=(seq, REGIME_DIM), name=REGIME_KEY)
    other = keras.Input(shape=(seq, 1), name="other")
    out = keras.layers.Dense(1, name="y")(keras.layers.Concatenate()([regime_in, other]))
    inner = keras.Model({REGIME_KEY: regime_in, "other": other}, {"y": out})

    model = build_regime_model(inner, n)
    assert model is not inner
    model.compile(optimizer=keras.optimizers.Adam(0.05), loss={"y": "mse"})

    rng = np.random.default_rng(0)
    offsets = rng.normal(scale=3.0, size=n)
    inputs = {REGIME_KEY: np.zeros((n, seq, REGIME_DIM), np.float32),
              "other": rng.normal(size=(n, seq, 1)).astype(np.float32),
              GAME_INDEX_KEY: np.arange(n, dtype=np.int32).reshape(n, 1)}
    targets = {"y": np.repeat(offsets[:, None, None], seq, axis=1).astype(np.float32)}
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(4)

    history = model.fit(ds, epochs=80, verbose=0)
    first, last = history.history["loss"][0], history.history["loss"][-1]
    assert first > 1.0, "a flat zero here means the loss tracker is not being fed"
    assert last < first / 10.0, "the latent should absorb a purely per-game offset"
    assert np.any(regime_std(model.fitted_table()))

    # Validation does not look the row up, so it stays at roughly the variance of the offsets.
    assert model.evaluate(ds, verbose=0) > last * 10
    # And the thing that gets saved is the plain functional model, not the wrapper.
    assert model.inner is inner
