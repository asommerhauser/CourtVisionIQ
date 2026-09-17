"""
W4 rung 1: scheduled sampling, and the one training wrapper both 3.0 features share.

Every head trains on next-step likelihood with the real history in context, and is scored on what
600 self-fed steps produce. Those are different tasks, and the model never sees a context it
generated. Rung 1 closes part of that gap by replacing some previous-event tokens with the model's
own samples.

The tests below are mostly about what must NOT change. Mixing is deliberately confined to the
categorical event column at positions the controller actually queries; the derived game-state and
lineup-state columns stay real, because they come from pure-Python folds that cannot be recomputed
in a graph and re-implementing them would be the second implementation this project forbids. If
those columns ever start moving, the A/B stops measuring what it claims to.
"""
from __future__ import annotations

import numpy as np
import pytest

import config
from models.regime import GAME_INDEX_KEY, REGIME_KEY
from models.scheduled_sampling import MIXABLE_FIELDS, mixing_probability


# --------------------------------------------------------------------------- the schedule

def test_the_probability_is_zero_through_the_warmup():
    """A model that has not fit the next-step distribution has nothing worth sampling from."""
    for epoch in range(config.SCHEDULED_SAMPLING_WARMUP_EPOCHS):
        assert mixing_probability(epoch) == 0.0


def test_it_ramps_linearly_and_then_holds_at_the_cap():
    warm = config.SCHEDULED_SAMPLING_WARMUP_EPOCHS
    ramp = config.SCHEDULED_SAMPLING_RAMP_EPOCHS
    cap = config.SCHEDULED_SAMPLING_MAX_P
    assert mixing_probability(warm) == pytest.approx(0.0)
    assert mixing_probability(warm + ramp // 2) == pytest.approx(cap * (ramp // 2) / ramp)
    assert mixing_probability(warm + ramp) == pytest.approx(cap)
    assert mixing_probability(warm + ramp + 50) == pytest.approx(cap)


def test_the_knob_switches_it_off_entirely(monkeypatch):
    monkeypatch.setattr(config, "SCHEDULED_SAMPLING", False)
    assert mixing_probability(100) == 0.0


def test_only_categorical_columns_are_ever_mixable():
    """The scope limit, as a list. Nothing derived may join it: score, clock, fouls, stint seconds
    and minutes come from GameStateScan and LineupScan, which have no in-graph equivalent."""
    from models.game_state_features import GAME_STATE_KEYS
    from models.rotation_features import ROSTER_STATE_KEYS

    assert not set(MIXABLE_FIELDS) & set(GAME_STATE_KEYS)
    assert not set(MIXABLE_FIELDS) & set(ROSTER_STATE_KEYS)
    assert REGIME_KEY not in MIXABLE_FIELDS


# --------------------------------------------------------------------------- the wrapper

def _toy(seq=6, vocab=5):
    """A head shaped like the real one: a categorical event column and the regime plane."""
    from tensorflow import keras

    event = keras.Input(shape=(seq,), name="event", dtype="int32")
    regime = keras.Input(shape=(seq, config.REGIME_DIM), name=REGIME_KEY)
    merged = keras.layers.Concatenate()([keras.layers.Embedding(vocab, 4)(event), regime])
    out = keras.layers.Dense(vocab, name="event_output")(merged)
    return keras.Model({"event": event, REGIME_KEY: regime}, {"event_output": out})


def test_the_wrapper_is_skipped_when_neither_feature_is_on(monkeypatch):
    from models.train_steps import build_trainer

    monkeypatch.setattr(config, "REGIME_ENABLED", False)
    monkeypatch.setattr(config, "SCHEDULED_SAMPLING", False)
    sentinel = object()
    assert build_trainer(sentinel, n_games=100, scheduled_sampling=True) is sentinel


def test_a_head_that_does_not_ask_for_mixing_does_not_get_it():
    """Only event_time is wired for rung 1. player / substitution / sub_decision must never be:
    their inputs ARE lineup state, which is exactly what this leaves stale."""
    from models.train_steps import build_trainer

    model = build_trainer(_toy(), n_games=8, scheduled_sampling=False)
    assert model.scheduled_sampling is False


def test_both_features_run_together_and_the_saved_graph_stays_functional():
    """The reason there is one wrapper and not two: they have to share a train_step."""
    import tensorflow as tf
    from tensorflow import keras

    from models.regime import regime_std
    from models.scheduled_sampling import ScheduledSamplingSchedule
    from models.train_steps import build_trainer

    seq, n, vocab = 6, 16, 5
    inner = _toy(seq, vocab)
    model = build_trainer(inner, n_games=n, scheduled_sampling=True)
    assert model.scheduled_sampling and model.table is not None
    model.compile(optimizer=keras.optimizers.Adam(0.02),
                  loss={"event_output": keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True)})

    rng = np.random.default_rng(0)
    inputs = {"event": rng.integers(0, vocab, (n, seq)).astype("int32"),
              REGIME_KEY: np.zeros((n, seq, config.REGIME_DIM), np.float32),
              GAME_INDEX_KEY: np.arange(n, dtype="int32").reshape(n, 1)}
    targets = {"event_output": rng.integers(0, vocab, (n, seq)).astype("int32")}
    weights = {"event_output": np.ones((n, seq), np.float32)}
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets, weights)).batch(4)

    epochs = config.SCHEDULED_SAMPLING_WARMUP_EPOCHS + config.SCHEDULED_SAMPLING_RAMP_EPOCHS + 1
    history = model.fit(ds, validation_data=ds, epochs=epochs, verbose=0,
                        callbacks=[ScheduledSamplingSchedule(model)])

    # The loss tracker is fed. A flat zero would make EarlyStopping restore epoch 1 (see
    # models/train_steps.py) -- which is silent, because the model trains fine either way.
    assert history.history["loss"][0] > 0.0
    assert history.history["val_loss"][0] > 0.0
    # p is logged per epoch, which is what lands it in epochs.parquet.
    logged = history.history["scheduled_sampling_p"]
    assert logged[0] == 0.0 and logged[-1] == pytest.approx(config.SCHEDULED_SAMPLING_MAX_P)
    # and it actually reached the model, not just the log.
    assert float(model.mixing_p.numpy()) == pytest.approx(config.SCHEDULED_SAMPLING_MAX_P)
    # The latent still learned alongside it -- the lookup is inside the tape.
    assert np.any(regime_std(model.fitted_table()))
    # And the artifact is the plain functional model.
    assert model.inner is inner


def test_the_latent_table_receives_gradients():
    """The bug this exists for: an embedding lookup outside the GradientTape trains as all zeros,
    while the head itself learns perfectly well. Nothing looks wrong until regime_std comes out
    empty at the end of a full train."""
    import tensorflow as tf
    from tensorflow import keras

    from models.regime import regime_std
    from models.train_steps import build_trainer

    seq, n = 4, 16
    regime = keras.Input(shape=(seq, config.REGIME_DIM), name=REGIME_KEY)
    other = keras.Input(shape=(seq, 1), name="other")
    out = keras.layers.Dense(1, name="y")(keras.layers.Concatenate()([regime, other]))
    inner = keras.Model({REGIME_KEY: regime, "other": other}, {"y": out})

    model = build_trainer(inner, n_games=n)
    model.compile(optimizer=keras.optimizers.Adam(0.05), loss={"y": "mse"})
    rng = np.random.default_rng(0)
    offsets = rng.normal(scale=3.0, size=n)
    inputs = {REGIME_KEY: np.zeros((n, seq, config.REGIME_DIM), np.float32),
              "other": rng.normal(size=(n, seq, 1)).astype(np.float32),
              GAME_INDEX_KEY: np.arange(n, dtype=np.int32).reshape(n, 1)}
    targets = {"y": np.repeat(offsets[:, None, None], seq, axis=1).astype(np.float32)}
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(4)

    history = model.fit(ds, epochs=80, verbose=0)
    assert np.any(regime_std(model.fitted_table())), "the table got no gradient"
    assert history.history["loss"][-1] < history.history["loss"][0] / 10.0
