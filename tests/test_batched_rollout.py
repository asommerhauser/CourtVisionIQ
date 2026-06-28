"""
Batched-rollout coordinator tests — pure scheduling, no TF / trained models.

Drives :class:`simulation.batched_rollout._BatchCoordinator` with a deterministic dummy ``infer_fn``
(a stand-in for the batched model call) and plain worker threads (stand-ins for GameControllers), so
the group-by-head / stack / scatter / barrier logic is exercised on CPU. Watchdog timeouts turn a
scheduling deadlock into a fast failure instead of a hang.
"""
from __future__ import annotations

import threading

import numpy as np

from simulation.batched_rollout import _BatchCoordinator, GameJob  # noqa: F401 (import smoke)


def _run(n_workers, routine, infer_fn, timeout=10.0):
    """Run ``routine`` on ``n_workers`` threads against one coordinator; return the shared results."""
    coord = _BatchCoordinator(infer_fn, n_workers=n_workers)
    results: dict = {}

    def worker(wid):
        try:
            routine(wid, coord, results)
        finally:
            coord.worker_done(0)

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    cthread = threading.Thread(target=coord.run)
    for t in workers:
        t.start()
    cthread.start()
    for t in workers:
        t.join(timeout)
    cthread.join(timeout)
    assert not cthread.is_alive(), "coordinator deadlocked"
    assert all(not t.is_alive() for t in workers), "a worker deadlocked"
    return results


def test_batches_all_workers_each_round_and_returns_correct_slices():
    seen_batch_sizes = []

    def infer(model_key, stacked):
        seen_batch_sizes.append(stacked["x"].shape[0])
        return {"y": stacked["x"] * 2}

    def routine(wid, coord, results):
        for r in range(3):
            out = coord.request(wid, "m", {"x": np.array([[float(wid * 10 + r)]])})
            results[(wid, r)] = float(out["y"][0, 0])

    results = _run(4, routine, infer)

    # Every worker gets its OWN doubled value back (correct scatter, no cross-talk).
    assert results[(2, 1)] == (2 * 10 + 1) * 2
    assert results[(0, 0)] == 0.0
    # The barrier pooled all four workers every round (3 rounds of 4).
    assert seen_batch_sizes == [4, 4, 4]


def test_handles_desync_and_early_finish_without_deadlock():
    seen = []

    def infer(model_key, stacked):
        seen.append(stacked["x"].shape[0])
        return {"y": stacked["x"] * 2}

    def routine(wid, coord, results):
        # Worker i makes i+1 requests, so workers finish at different rounds.
        for _ in range(wid + 1):
            out = coord.request(wid, "m", {"x": np.array([[float(wid)]])})
            results.setdefault(wid, []).append(float(out["y"][0, 0]))

    results = _run(4, routine, infer)

    assert results[0] == [0.0]                 # 1 request
    assert results[3] == [6.0, 6.0, 6.0, 6.0]  # 4 requests, value 3*2
    # Batch size shrinks as workers drop out: 4 then 3 then 2 then 1.
    assert seen == [4, 3, 2, 1]


def test_groups_requests_by_head_within_a_round():
    seen = []

    def infer(model_key, stacked):
        seen.append((model_key, stacked["x"].shape[0]))
        return {"y": stacked["x"] + 1}

    def routine(wid, coord, results):
        head = "a" if wid % 2 == 0 else "b"   # even workers -> head a, odd -> head b
        out = coord.request(wid, head, {"x": np.array([[float(wid)]])})
        results[wid] = float(out["y"][0, 0])

    results = _run(4, routine, infer)

    assert results[1] == 2.0 and results[2] == 3.0
    # One barrier round, two head groups of two -> exactly two batched calls.
    assert ("a", 2) in seen and ("b", 2) in seen
    assert len(seen) == 2
