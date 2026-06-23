"""
Sampling-primitive tests for the rollout: temperature on the softmax and the per-candidate
bias / temperature in ``_masked_sample``. These exercise the smoothing knob that keeps a star
from taking ~75% of a team's possessions and the fatigue nudge on the outgoing-sub pick — both
pure numpy, no trained models or TF graph.
"""
from __future__ import annotations

import numpy as np

from simulation.game_simulator import GameSimulator, _softmax


def _bare_sim(seed: int = 0) -> GameSimulator:
    """A GameSimulator shell with just an RNG — enough to call ``_masked_sample`` directly."""
    sim = object.__new__(GameSimulator)
    sim.rng = np.random.default_rng(seed)
    return sim


# Candidate -> index into the logits vector (stands in for the player vocab encode).
CANDS = ["A", "B", "C", "D", "E"]
ENCODE = {c: i for i, c in enumerate(CANDS)}.__getitem__
SKEWED = np.array([4.0, 1.0, 1.0, 1.0, 1.0])   # A is the heavy favorite


def test_softmax_temperature_flattens():
    sharp = _softmax(SKEWED, temperature=1.0)
    flat = _softmax(SKEWED, temperature=5.0)
    assert sharp[0] > flat[0]                       # high temp pulls mass off the favorite
    assert flat[0] < 0.5                            # …and well below the raw ~0.7
    assert np.isclose(flat.sum(), 1.0)


def test_softmax_temperature_sharpens():
    sharp = _softmax(SKEWED, temperature=0.5)
    raw = _softmax(SKEWED, temperature=1.0)
    assert sharp[0] > raw[0]                        # low temp concentrates mass on the favorite
    assert np.isclose(sharp.sum(), 1.0)


def test_masked_sample_temperature_raises_favorite_share():
    def share(temperature):
        sim = _bare_sim(seed=1)
        picks = [sim._masked_sample(SKEWED, CANDS, ENCODE, temperature=temperature)
                 for _ in range(4000)]
        return picks.count("A") / len(picks)

    assert share(0.5) > share(1.0)                  # sharper distribution → more A picks


def test_softmax_temperature_one_is_noop():
    base = SKEWED - np.max(SKEWED)
    expected = np.exp(base) / np.exp(base).sum()
    assert np.allclose(_softmax(SKEWED), expected)
    assert np.allclose(_softmax(SKEWED, temperature=1.0), expected)


def test_masked_sample_temperature_lowers_favorite_share():
    def share(temperature):
        sim = _bare_sim(seed=1)
        picks = [sim._masked_sample(SKEWED, CANDS, ENCODE, temperature=temperature)
                 for _ in range(4000)]
        return picks.count("A") / len(picks)

    assert share(5.0) < share(1.0)                  # flatter distribution → fewer A picks


def test_masked_sample_bias_can_flip_greedy_pick():
    sim = _bare_sim()
    # A is the argmax by logit, but a large bias on E overrides it under greedy.
    assert sim._masked_sample(SKEWED, CANDS, ENCODE, greedy=True) == "A"
    biased = sim._masked_sample(SKEWED, CANDS, ENCODE, greedy=True, bias={"E": 10.0})
    assert biased == "E"


def test_masked_sample_greedy_ignores_temperature():
    sim = _bare_sim()
    assert sim._masked_sample(SKEWED, CANDS, ENCODE, greedy=True, temperature=0.1) == "A"
    assert sim._masked_sample(SKEWED, CANDS, ENCODE, greedy=True, temperature=9.0) == "A"


def test_masked_sample_bias_raises_outgoing_share():
    """A fatigue-style bias makes a low-logit player the likely outgoing pick."""
    def share(bias):
        sim = _bare_sim(seed=2)
        picks = [sim._masked_sample(SKEWED, CANDS, ENCODE, bias=bias) for _ in range(4000)]
        return picks.count("E") / len(picks)

    assert share({"E": 6.0}) > share(None)
