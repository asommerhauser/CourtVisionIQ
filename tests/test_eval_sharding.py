"""
Tests for sharded eval-all (parallel-process holdout split).

The rollout is CPU/GIL-bound, so eval-all can be run as N parallel processes, each taking
``holdout[shard::num_shards]``. These tests pin the slicing contract without touching the GPU:
the union of shards is the whole holdout, shards are disjoint, and a sharded run suppresses the
intermediate report (only the final unsharded pass writes the aggregate).
"""
from __future__ import annotations

import training.full_run as full_run
from training.full_run import FullRun


def _run_with_state(tmp_path, holdout):
    run = FullRun(state_path=str(tmp_path / "state.json"))
    run.state = {
        "status": "trained", "holdout_game_ids": list(holdout),
        "data_dir": "./data", "processed_dir": "./data/processed",
        "artifacts_root": "./artifacts_full2", "reports_root": "./reports",
        "eval_batch": 10,
    }
    return run


def _capture_evaluate_stage(monkeypatch):
    """Replace the heavy evaluate_stage with a spy that records its kwargs."""
    calls = {}

    def fake_evaluate_stage(stage_name, **kwargs):
        calls.update(kwargs)
        return {"done": len(kwargs["holdout_ids"]), "total": len(kwargs["holdout_ids"]),
                "run_dir": "x"}

    monkeypatch.setattr(full_run, "evaluate_stage", fake_evaluate_stage, raising=False)
    # eval_all imports evaluate_stage locally from simulation.stage_eval — patch there too.
    import simulation.stage_eval as se
    monkeypatch.setattr(se, "evaluate_stage", fake_evaluate_stage, raising=True)
    return calls


def test_shards_partition_the_holdout(tmp_path, monkeypatch):
    holdout = list(range(100, 130))  # 30 games
    num_shards = 4
    seen = []
    for shard in range(num_shards):
        calls = _capture_evaluate_stage(monkeypatch)
        _run_with_state(tmp_path, holdout).eval_all(shard=shard, num_shards=num_shards)
        seen.append(calls["holdout_ids"])

    flat = [g for s in seen for g in s]
    assert sorted(flat) == holdout          # complete cover, no game lost
    assert len(flat) == len(set(flat))      # disjoint, no game simulated twice
    # Balanced within one game across shards (30 / 4 -> sizes 8,8,7,7).
    assert max(len(s) for s in seen) - min(len(s) for s in seen) <= 1


def test_sharded_run_suppresses_intermediate_report(tmp_path, monkeypatch):
    calls = _capture_evaluate_stage(monkeypatch)
    _run_with_state(tmp_path, list(range(20))).eval_all(shard=0, num_shards=3)
    assert calls["report_every"] is None     # partial shard must not write the aggregate


def test_unsharded_run_is_unchanged(tmp_path, monkeypatch):
    calls = _capture_evaluate_stage(monkeypatch)
    run = _run_with_state(tmp_path, list(range(20)))
    run.eval_all()  # defaults: shard 0 / num_shards 1
    assert calls["holdout_ids"] == list(range(20))   # whole holdout
    assert calls["report_every"] == 10               # eval_batch -> intermediate reports on


def test_bad_shard_index_rejected(tmp_path, monkeypatch):
    _capture_evaluate_stage(monkeypatch)
    run = _run_with_state(tmp_path, list(range(20)))
    try:
        run.eval_all(shard=3, num_shards=3)          # valid shards are 0,1,2
        assert False, "expected SystemExit for out-of-range shard"
    except SystemExit:
        pass
