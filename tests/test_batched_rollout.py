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
            coord.worker_done()

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


# ===================================================================== #
# --- Slot pool: backfill instead of strict cohorts                    --
# ===================================================================== #

class _StubMaster:
    """Enough of a GameSimulator for run_jobs_batched: an _infer and the attrs _WorkerSim copies."""

    def __init__(self):
        self.model = object()
        self.instance = None
        self.heads = {}
        self.condtime_norm_stats = {}
        self.batch_widths: list[int] = []

    def _infer(self, model_key, stacked):
        self.batch_widths.append(stacked["x"].shape[0])
        return {"y": stacked["x"]}


def _patch_pool(monkeypatch, lengths, fail=()):
    """Replace _WorkerSim/GameController so a "game" is just N coordinator requests.

    ``lengths[i]`` is how many requests job i makes — the knob that creates the ragged tails a
    strict-cohort loop cannot backfill. ``fail`` is a set of job seeds whose rollout raises after
    making its requests, standing in for a sim that dies mid-run.
    """
    import simulation.batched_rollout as br

    class FakeWorkerSim:
        def __init__(self, master, coordinator, worker_id):
            self.coord, self.wid = coordinator, worker_id

    class FakeController:
        def __init__(self, sim, seed=0, greedy=False):
            self.sim, self.seed = sim, seed

        def start(self, *a, **kw):
            pass

        def run(self):
            for _ in range(lengths[self.seed]):
                self.sim.coord.request(self.sim.wid, "m", {"x": np.array([[float(self.seed)]])})
            if self.seed in fail:
                raise RuntimeError(f"job {self.seed} blew up")
            return [{"event": "e"}] * lengths[self.seed]

    monkeypatch.setattr(br, "_WorkerSim", FakeWorkerSim)
    monkeypatch.setattr(br, "GameController", FakeController)


def test_slots_backfill_and_every_job_runs(monkeypatch):
    """More jobs than slots: results stay aligned to `jobs`, and no job is skipped or doubled."""
    import simulation.batched_rollout as br

    lengths = [3, 12, 4, 1, 9, 2, 7, 5, 6, 11]      # deliberately ragged
    _patch_pool(monkeypatch, lengths)
    master = _StubMaster()
    jobs = [br.GameJob(home_roster=[], away_roster=[], seed=i) for i in range(len(lengths))]

    histories = br.run_jobs_batched(master, jobs, batch_size=3, show_progress=False)

    assert [len(h) for h in histories] == lengths, "results must stay aligned to job order"


def test_backfill_keeps_the_batch_full_past_the_first_cohort(monkeypatch):
    """The point of the pool: a finished short game is replaced, not left as a hole in the batch.

    With 3 slots and one very long job, the strict-cohort loop would drain to width 1 three times
    (once per cohort of 3). The pool holds width 3 until the job list is genuinely exhausted.
    """
    import simulation.batched_rollout as br

    lengths = [1, 1, 1, 1, 1, 1, 1, 1, 40]
    _patch_pool(monkeypatch, lengths)
    master = _StubMaster()
    jobs = [br.GameJob(home_roster=[], away_roster=[], seed=i) for i in range(len(lengths))]

    br.run_jobs_batched(master, jobs, batch_size=3, show_progress=False)

    widths = master.batch_widths
    assert sum(widths) == sum(lengths), "every request must be served exactly once"
    # The long job's tail is unavoidably width 1, but the short jobs must not have been.
    assert max(widths) == 3
    assert widths.count(1) <= 40 + 2, "short jobs should have been pooled, not run one at a time"


def test_pool_handles_more_slots_than_jobs(monkeypatch):
    import simulation.batched_rollout as br

    lengths = [2, 3]
    _patch_pool(monkeypatch, lengths)
    master = _StubMaster()
    jobs = [br.GameJob(home_roster=[], away_roster=[], seed=i) for i in range(2)]

    histories = br.run_jobs_batched(master, jobs, batch_size=32, show_progress=False)
    assert [len(h) for h in histories] == lengths


def test_pool_with_no_jobs_is_a_noop():
    import simulation.batched_rollout as br

    assert br.run_jobs_batched(_StubMaster(), [], batch_size=8, show_progress=False) == []


# ===================================================================== #
# --- Streaming sink + per-job failure isolation                       --
# ===================================================================== #

def test_on_complete_fires_once_per_job_with_its_own_history(monkeypatch):
    """The streaming seam: every finished sim is handed to the sink exactly once, in job terms."""
    import simulation.batched_rollout as br

    lengths = [3, 1, 4, 1, 5]
    _patch_pool(monkeypatch, lengths)
    jobs = [br.GameJob(home_roster=[], away_roster=[], seed=i) for i in range(len(lengths))]

    seen: list[tuple[int, int]] = []
    lock = threading.Lock()

    def sink(job_idx, history):
        with lock:
            seen.append((job_idx, len(history)))

    histories = br.run_jobs_batched(_StubMaster(), jobs, batch_size=2, show_progress=False,
                                    on_complete=sink)

    assert sorted(seen) == sorted(enumerate(lengths)), "each job streamed exactly once"
    assert [len(h) for h in histories] == lengths, "keep_histories=True still returns them"


def test_keep_histories_false_streams_without_retaining(monkeypatch):
    """The memory bound: the sink sees every history, the returned list holds none of them."""
    import simulation.batched_rollout as br

    lengths = [2, 3, 4]
    _patch_pool(monkeypatch, lengths)
    jobs = [br.GameJob(home_roster=[], away_roster=[], seed=i) for i in range(len(lengths))]

    streamed: list[int] = []
    lock = threading.Lock()

    def sink(job_idx, history):
        with lock:
            streamed.append(len(history))

    histories = br.run_jobs_batched(_StubMaster(), jobs, batch_size=3, show_progress=False,
                                    on_complete=sink, keep_histories=False)

    assert sorted(streamed) == sorted(lengths), "the sink still saw every sim"
    assert histories == [None] * len(lengths), "nothing retained"


def test_a_failing_job_does_not_kill_its_slot(monkeypatch):
    """The regression this guards: a raising sim used to kill its thread for the rest of the run.

    With 2 slots and 3 of 8 jobs raising, a dead-slot pool would drop to width 1 and — because
    ``worker_done`` never fires for a thread that died mid-loop — could strand the barrier. Every
    surviving job must still finish, and the failures must land as ``None``, not as an exception.
    """
    import simulation.batched_rollout as br

    lengths = [1] * 8
    fail = {1, 4, 5}
    _patch_pool(monkeypatch, lengths, fail=fail)
    jobs = [br.GameJob(home_roster=[], away_roster=[], seed=i) for i in range(len(lengths))]

    progress = br._Progress(total=len(jobs), enabled=False)
    histories = br.run_jobs_batched(_StubMaster(), jobs, batch_size=2, show_progress=False,
                                    progress=progress)

    for i in range(len(lengths)):
        if i in fail:
            assert histories[i] is None, f"job {i} raised, so it has no history"
        else:
            assert histories[i] is not None, f"job {i} never ran - its slot died"
    # complete_one still fires for a failed job, so the barrier accounting stays exact.
    assert progress.completed == len(jobs)


def test_a_failing_sink_does_not_lose_the_history(monkeypatch):
    """A disk error while streaming must not discard a sim that actually ran."""
    import simulation.batched_rollout as br

    lengths = [2, 2]
    _patch_pool(monkeypatch, lengths)
    jobs = [br.GameJob(home_roster=[], away_roster=[], seed=i) for i in range(2)]

    def sink(job_idx, history):
        raise OSError("disk full")

    histories = br.run_jobs_batched(_StubMaster(), jobs, batch_size=2, show_progress=False,
                                    on_complete=sink)

    assert [len(h) for h in histories] == lengths, "history is recorded before the sink runs"
