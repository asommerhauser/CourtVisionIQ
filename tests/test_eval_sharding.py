"""
Eval sharding: --shard I/N splits the holdout across N processes.

The rollout's worker threads are GIL-bound, so a single process cannot use a many-core box no
matter how wide its batch is; separate processes on disjoint slices are what turn cores into
throughput. These tests pin the two properties that make that safe -- the slices are disjoint and
cover the holdout exactly once, and a sharded call does not write the shared report files -- plus
the argument guards and the TF-free import path the process pool depends on.

No TensorFlow, no trained model: FullRun.eval is driven with evaluate_stage monkeypatched.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import evaluate
from training.full_run import FullRun

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# --- parse_shard                                                          --- #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    ("1/1", (1, 1)), ("2/5", (2, 5)), ("  3/4 ", (3, 4)), ("10/10", (10, 10)), ("7 / 9", (7, 9)),
])
def test_parse_shard_accepts_valid_forms(text, expected):
    assert evaluate.parse_shard(text) == expected


@pytest.mark.parametrize("text", [
    "", "1", "/3", "3/", "0/4", "5/4", "-1/4", "1/0", "a/b", "1/2/3", "1.5/4",
])
def test_parse_shard_rejects_malformed(text):
    with pytest.raises(ValueError):
        evaluate.parse_shard(text)


# --------------------------------------------------------------------------- #
# --- The slicing contract                                                 --- #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("total,n", [(100, 4), (100, 3), (100, 7), (10, 10), (10, 1), (5, 8)])
def test_slices_are_disjoint_and_cover_the_holdout_exactly_once(total, n):
    """The whole safety argument: no game is simulated twice, none is dropped.

    Two shards landing on one game would interleave writes into the same folder's eight files;
    a dropped game would silently shrink the report's denominator.
    """
    holdout = list(range(1000, 1000 + total))
    slices = [holdout[i - 1::n] for i in range(1, n + 1)]

    flat = [g for s in slices for g in s]
    assert sorted(flat) == holdout, "union must be the holdout, with no duplicates"
    assert len(flat) == total
    assert max(len(s) for s in slices) - min(len(s) for s in slices) <= 1, "slices stay balanced"


def _state(tmp_path, holdout) -> str:
    p = tmp_path / "state.json"
    p.write_text(json.dumps({
        "version": "v1.0", "status": "trained", "data_dir": str(tmp_path / "data"),
        "processed_dir": str(tmp_path / "proc"), "artifacts_root": str(tmp_path / "artifacts"),
        "reports_root": str(tmp_path / "reports"), "batch_size": 32, "epochs": 1,
        "holdout_game_ids": holdout, "trained_models": [],
    }), encoding="utf-8")
    return str(p)


@pytest.fixture
def spy_eval(tmp_path, monkeypatch):
    """Run FullRun.eval against a stubbed evaluate_stage; return the captured kwargs."""
    import simulation.stage_eval as stage_eval

    seen: dict = {}

    def fake_evaluate_stage(stage_name, **kw):
        seen.update(kw)
        seen["stage_name"] = stage_name
        return {"done": len(kw["holdout_ids"]), "total": len(kw["holdout_ids"]),
                "run_dir": str(tmp_path / "run")}

    monkeypatch.setattr(stage_eval, "evaluate_stage", fake_evaluate_stage)
    monkeypatch.setattr("reporting.eval_report.resolve_results_run_dir",
                        lambda *a, **kw: tmp_path / "run")
    return seen


def test_shard_passes_only_its_slice_and_skips_the_report(tmp_path, spy_eval):
    holdout = list(range(1000, 1012))
    run = FullRun(state_path=_state(tmp_path, holdout))

    run.eval(name="trial1", shard=(2, 4))

    assert spy_eval["holdout_ids"] == holdout[1::4]
    assert spy_eval["write_report"] is False, "shards must not write the shared report files"


def test_unsharded_eval_writes_the_report_and_records_the_run(tmp_path, spy_eval):
    holdout = list(range(1000, 1012))
    state_path = _state(tmp_path, holdout)
    run = FullRun(state_path=state_path)

    run.eval(name="trial1")

    assert spy_eval["holdout_ids"] == holdout
    assert spy_eval["write_report"] is True
    assert json.loads(Path(state_path).read_text())["last_eval_name"] == "run"


def test_seed_reaches_the_rollout(tmp_path, spy_eval):
    """--seed has to survive the whole chain or a pooled run silently stops being reproducible."""
    run = FullRun(state_path=_state(tmp_path, list(range(1000, 1006))))
    run.eval(name="trial1", seed=11)
    assert spy_eval["seed0"] == 11


def test_shard_leaves_the_run_state_untouched(tmp_path, spy_eval):
    """Concurrent shards would race the state file, so a sharded call must not write it."""
    holdout = list(range(1000, 1012))
    state_path = _state(tmp_path, holdout)
    before = Path(state_path).read_text()

    FullRun(state_path=state_path).eval(name="trial1", shard=(1, 3))

    assert Path(state_path).read_text() == before
    assert "last_eval_name" not in json.loads(before)


def test_every_shard_together_covers_the_holdout(tmp_path, monkeypatch):
    """End to end through FullRun.eval: four shards, and every holdout game is claimed once."""
    import simulation.stage_eval as stage_eval

    holdout = list(range(2000, 2037))
    claimed: list[int] = []

    def fake_evaluate_stage(stage_name, **kw):
        claimed.extend(kw["holdout_ids"])
        return {"done": 0, "total": len(kw["holdout_ids"]), "run_dir": str(tmp_path / "run")}

    monkeypatch.setattr(stage_eval, "evaluate_stage", fake_evaluate_stage)
    monkeypatch.setattr("reporting.eval_report.resolve_results_run_dir",
                        lambda *a, **kw: tmp_path / "run")

    state_path = _state(tmp_path, holdout)
    for i in range(1, 5):
        FullRun(state_path=state_path).eval(name="trial1", shard=(i, 4))

    assert sorted(claimed) == holdout


# --------------------------------------------------------------------------- #
# --- CLI guards                                                           --- #
# --------------------------------------------------------------------------- #

def _cli(*args):
    """Run evaluate.py in a subprocess; return (rc, stderr). Never reaches the TF import."""
    return subprocess.run([sys.executable, "evaluate.py", *args], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=120)


def test_shard_without_run_is_rejected():
    r = _cli("--shard", "1/4")
    assert r.returncode == 2
    assert "--shard requires --run" in r.stderr


def test_shard_with_report_only_is_rejected():
    r = _cli("--shard", "1/4", "--run", "trial1", "--report-only")
    assert r.returncode == 2
    assert "cannot be combined with --report-only" in r.stderr


def test_malformed_shard_is_rejected_by_the_cli():
    r = _cli("--shard", "9/4", "--run", "trial1")
    assert r.returncode == 2
    assert "1 <= I <= N" in r.stderr


def test_missing_dial_file_is_rejected():
    r = _cli("--dials", "does-not-exist.json", "--run", "trial1")
    assert r.returncode == 2
    assert "no dial file at" in r.stderr


def test_importing_evaluate_does_not_pull_tensorflow():
    """The process-pool supervisor imports this module and must never load TF.

    A TF import per supervisor costs seconds and, worse, would create a CUDA context in a process
    whose only job is to spawn children -- VRAM taken from the workers for nothing.
    """
    code = ("import sys, runpy; "
            "runpy.run_path('evaluate.py', run_name='not_main'); "
            "print('tensorflow' in sys.modules or 'keras' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False", "evaluate.py must stay TF-free at import"


# --------------------------------------------------------------------------- #
# --- Dial package: the one thing that silently desyncs shards             --- #
# --------------------------------------------------------------------------- #

def test_dial_package_round_trips(tmp_path):
    """The parent writes its live dials; each child applies them. Both halves must agree.

    Without this, a child re-reads config.py from disk and runs DIFFERENT physics than the shell
    that launched it -- one report, two tunings, and nothing raises.
    """
    import config

    path = tmp_path / "dials.json"
    # Values deliberately off the committed defaults, so "restored" and "applied" are separable.
    moved = {"DELTA_TIME_SCALE": config.DELTA_TIME_SCALE + 0.11,
             "EVENT_TEMPERATURE": config.EVENT_TEMPERATURE + 0.23}
    with config.dials(**moved):
        written = config.write_dial_file(path)
        assert written["DELTA_TIME_SCALE"] == moved["DELTA_TIME_SCALE"]

    assert config.DELTA_TIME_SCALE != moved["DELTA_TIME_SCALE"], "context manager should restore"
    applied = config.apply_dial_file(path)
    assert applied["DELTA_TIME_SCALE"] == moved["DELTA_TIME_SCALE"]
    assert config.DELTA_TIME_SCALE == moved["DELTA_TIME_SCALE"]
    assert config.EVENT_TEMPERATURE == moved["EVENT_TEMPERATURE"]
    assert set(applied) == set(config._TUNING_KEYS), "the package must pin EVERY dial, not a diff"


def test_apply_dial_file_rejects_junk(tmp_path):
    import config

    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        config.apply_dial_file(bad)

    bad.write_text('{"NOT_A_DIAL": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown dial"):
        config.apply_dial_file(bad)

    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        config.apply_dial_file(bad)
