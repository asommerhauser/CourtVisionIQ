"""Compiled-inference seam (_compiled_forward): tf.function path matches eager, with safe fallback.

Validates the (opt-in) GPU-utilization speedup's *correctness* on CPU (no trained weights): when
enabled, the cached tf.function returns the same numbers as eager; a graph-incompatible head
degrades to eager permanently; and by default (CVIQ_TF_INFER unset) inference stays eager.
"""
from __future__ import annotations

import numpy as np
import pytest

import simulation.game_simulator as gs
from simulation.game_simulator import _EAGER, _compiled_forward


@pytest.fixture
def _enabled(monkeypatch):
    """Force the compiled path on (it is opt-in / off by default)."""
    monkeypatch.setattr(gs, "_TF_INFER_ENABLED", True)


def _tiny_model():
    import keras
    from keras import layers
    inp = {"x": keras.Input(shape=(3,), name="x")}
    out = {"y": layers.Dense(2, name="d")(inp["x"])}
    return keras.Model(inp, out)


def test_compiled_matches_eager_and_caches(_enabled):
    model = _tiny_model()
    x = {"x": np.random.RandomState(0).randn(4, 3).astype("float32")}
    eager = {k: np.asarray(v) for k, v in model(x, training=False).items()}

    cache: dict = {}
    out = {k: np.asarray(v) for k, v in _compiled_forward(cache, model, "m", x).items()}
    assert np.allclose(eager["y"], out["y"], atol=1e-5)
    # The signature compiled (not pinned to eager) and is reused on a second call.
    assert cache[("m", ("x",))] is not _EAGER
    out2 = {k: np.asarray(v) for k, v in _compiled_forward(cache, model, "m", x).items()}
    assert np.allclose(out["y"], out2["y"], atol=1e-6)


def test_graph_incompatible_head_falls_back_to_eager(_enabled):
    # A "model" that works on numpy but raises when handed tf tensors (as tf.function does) forces
    # the fallback; the eager result must still come back and the signature must pin to eager.
    def numpy_only(x, training=False):
        if not isinstance(x["x"], np.ndarray):
            raise TypeError("graph-incompatible")
        return {"y": x["x"] * 2.0}

    cache: dict = {}
    x = {"x": np.ones((2, 1), dtype="float32")}
    out = _compiled_forward(cache, numpy_only, "m", x)
    assert np.allclose(np.asarray(out["y"]), 2.0)
    assert cache[("m", ("x",))] is _EAGER  # pinned; subsequent calls skip tracing
    assert np.allclose(np.asarray(_compiled_forward(cache, numpy_only, "m", x)["y"]), 2.0)


def test_default_is_eager():
    # Opt-in: with CVIQ_TF_INFER unset, _compiled_forward runs eagerly and caches nothing.
    assert gs._TF_INFER_ENABLED is False
    calls = {"n": 0}

    def model(x, training=False):
        calls["n"] += 1
        return {"y": x["x"]}

    cache: dict = {}
    _compiled_forward(cache, model, "m", {"x": np.ones((1, 1), dtype="float32")})
    assert calls["n"] == 1 and cache == {}  # ran eagerly, nothing compiled/cached
