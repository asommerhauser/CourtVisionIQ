"""
Rollout dials: runtime coercion, and the invariant that keeps runtime overrides working.

The dials in ``config._TUNING_KEYS`` are module globals that the ``cviq`` shell rebinds between
runs. That only works while every consumer reads ``config.<DIAL>`` at call time. Two ways to
silently break it, both of which look harmless in review and neither of which any behavioural
test would catch:

    from config import DELTA_TIME_SCALE          # binds at import
    def f(*, temperature=TYPE_TEMPERATURE): ...  # binds at def time

``test_no_module_scope_dial_imports`` / ``test_no_dial_valued_default_args`` parse the rollout
modules and fail on either.
"""
import ast
import json
from pathlib import Path

import pytest

import config

ROOT = Path(__file__).resolve().parents[1]
# The modules that read dials during a rollout; these are what the shell's `set` must reach.
DIAL_CONSUMERS = ("simulation/controller.py", "simulation/game_simulator.py")


# --------------------------------------------------------------------------- #
# --- Coercion                                                              -- #
# --------------------------------------------------------------------------- #

def test_set_dial_coerces_string_to_float():
    assert config.set_dial("DELTA_TIME_SCALE", "0.99") == 0.99
    assert isinstance(config.DELTA_TIME_SCALE, float)


def test_set_dial_coerces_string_to_int():
    """FOUL_OUT_LIMIT is the one int dial; '5' must not become the float 5.0."""
    config.set_dial("FOUL_OUT_LIMIT", "5")
    assert config.FOUL_OUT_LIMIT == 5
    assert isinstance(config.FOUL_OUT_LIMIT, int)


def test_set_dial_parses_json_for_dict_dials():
    config.set_dial("TYPE_BIAS", '{"foul_type": {"shooting": 0.5}}')
    assert config.TYPE_BIAS == {"foul_type": {"shooting": 0.5}}


def test_set_dial_rejects_unknown_name():
    """A typo must raise rather than mint a new module global that nothing reads."""
    with pytest.raises(KeyError, match="unknown dial"):
        config.set_dial("DELTA_TIME_SCLAE", 0.99)


def test_set_dial_rejects_non_dict_for_dict_dial():
    with pytest.raises((TypeError, json.JSONDecodeError)):
        config.set_dial("TYPE_BIAS", 0.5)


def test_get_dials_round_trips_through_apply_dials():
    original = config.get_dials()
    config.apply_dials({"DELTA_TIME_SCALE": 1.11, "TYPE_BIAS": {"foul_type": {"shooting": 9.0}}})
    config.apply_dials(original)
    assert config.get_dials() == original


def test_get_dials_returns_deep_copies():
    """A caller mutating a nested dict from get_dials() must not reach into module state."""
    snap = config.get_dials()
    snap["TYPE_BIAS"]["foul_type"]["shooting"] = 99.0
    assert config.TYPE_BIAS["foul_type"]["shooting"] != 99.0


# --------------------------------------------------------------------------- #
# --- The reporting seam                                                    -- #
# --------------------------------------------------------------------------- #

def test_tuning_snapshot_follows_an_override():
    """The snapshot is what lands in report.json; if it lags, every eval report lies."""
    config.set_dial("DELTA_TIME_SCALE", 0.99)
    assert config.tuning_snapshot()["DELTA_TIME_SCALE"] == 0.99


def test_tuning_snapshot_json_encodes_dict_dials():
    """Dict dials are JSON-encoded so each sits in a single Parquet column."""
    config.set_dial("TYPE_BIAS", {"foul_type": {"shooting": 0.5}})
    assert json.loads(config.tuning_snapshot()["TYPE_BIAS"]) == {"foul_type": {"shooting": 0.5}}


def test_dials_context_manager_restores():
    before = config.DELTA_TIME_SCALE
    with config.dials(DELTA_TIME_SCALE=1.5):
        assert config.DELTA_TIME_SCALE == 1.5
    assert config.DELTA_TIME_SCALE == before


def test_dials_context_manager_restores_on_exception():
    before = config.DELTA_TIME_SCALE
    with pytest.raises(RuntimeError):
        with config.dials(DELTA_TIME_SCALE=1.5):
            raise RuntimeError("boom")
    assert config.DELTA_TIME_SCALE == before


# --------------------------------------------------------------------------- #
# --- The invariant guard                                                   -- #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("relpath", DIAL_CONSUMERS)
def test_no_module_scope_dial_imports(relpath):
    """`from config import <DIAL>` freezes the value at import; the shell could never move it."""
    tree = ast.parse((ROOT / relpath).read_text(encoding="utf-8"))
    bound = [alias.name for node in ast.walk(tree)
             if isinstance(node, ast.ImportFrom) and node.module == "config"
             for alias in node.names
             if alias.name in config._TUNING_KEYS]
    assert not bound, (
        f"{relpath} binds dial(s) {bound} at import time. Use `config.<DIAL>` at call site "
        f"instead, or the shell's `set` will not reach them."
    )


@pytest.mark.parametrize("relpath", DIAL_CONSUMERS)
def test_no_dial_valued_default_args(relpath):
    """`def f(*, temperature=TYPE_TEMPERATURE)` evaluates once at def time — same freeze, subtler."""
    tree = ast.parse((ROOT / relpath).read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
        for d in defaults:
            if isinstance(d, ast.Name) and d.id in config._TUNING_KEYS:
                offenders.append(f"{node.name}(...={d.id})")
    assert not offenders, (
        f"{relpath} uses dial(s) as default args: {offenders}. Default expressions evaluate at "
        f"def time; use a `None` sentinel resolved from `config.<DIAL>` in the body."
    )
