"""
batched_rollout.py — run many independent game-sims at once, batching their GPU forward passes.

A single game rollout makes ~5 model forward passes per event (event/time, actor, conditional-Δt,
type, result) over ~500 events — all at **batch size 1**, where the GPU is overhead-bound launching
tiny kernels rather than compute-bound. Independent game-sims are embarrassingly parallel, so the win
is to pool their forward passes: run B games concurrently and, at each decision point, do **one
batched forward pass per head** across the games that need it.

Design — a thread-based inference coordinator, chosen so the rollout's rules stay untouched:

  * Each game runs its **unmodified** :class:`~simulation.controller.GameController` on a worker
    thread, driving a per-game :class:`_WorkerSim` that shares the loaded weights but has its own
    history / rosters / rng.
  * Every model call funnels through ``GameSimulator._infer`` (the one seam). ``_WorkerSim`` overrides
    it to **enqueue the request and block**; it does not touch the GPU.
  * One :class:`_BatchCoordinator` thread waits until all live workers have a request pending, groups
    them by head, runs each head **once** on the stacked batch, and hands every worker its slice back.
    Only the coordinator calls TensorFlow, so there is no TF threading issue; the workers do pure
    Python rule logic.

Because the coordinator is pure scheduling, a game's result depends only on its own seed and the
(shared, fixed) weights — batching changes throughput, not behavior. The coordinator takes a plain
``infer_fn`` rather than a simulator, so the scheduler is unit-testable on CPU with a dummy model.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from simulation.controller import GameController
from simulation.game_simulator import GameSimulator, HOME


# ===================================================================== #
# --- Job spec                                                         --
# ===================================================================== #

@dataclass
class GameJob:
    """One game-sim to run: the matchup + setup the controller needs, plus its seed."""
    home_roster: list[str]
    away_roster: list[str]
    season: str = "2003"
    possession: str = HOME
    home_starters: list[str] | None = None
    away_starters: list[str] | None = None
    season_context: dict | None = None
    seed: int = 0


# ===================================================================== #
# --- Worker simulator (shares weights, batches its inference)         --
# ===================================================================== #

class _WorkerSim(GameSimulator):
    """A per-game simulator that shares the master's loaded heads but routes inference to the pool.

    State (history / rosters / rng / season context) is its own; the heavy objects (the keras heads,
    encoder, norm stats) are shared references from the master, so B workers cost one set of weights.
    """

    def __init__(self, master: GameSimulator, coordinator: "_BatchCoordinator", worker_id: int):
        super().__init__(master.model, master.instance)
        self.heads = master.heads
        self.stint_norm_stats = master.stint_norm_stats
        self.condtime_norm_stats = master.condtime_norm_stats
        self._coordinator = coordinator
        self._worker_id = worker_id

    def _infer(self, model_key: str, inputs: dict) -> dict:
        # Block until the coordinator runs this game's forward pass as part of a batch.
        return self._coordinator.request(self._worker_id, model_key, inputs)


# ===================================================================== #
# --- Batch coordinator                                                --
# ===================================================================== #

def _stack(inputs_list: list[dict]) -> dict:
    """Concatenate a list of batch-1 input dicts along the batch axis -> one (G, …) batch."""
    keys = inputs_list[0].keys()
    return {k: np.concatenate([inp[k] for inp in inputs_list], axis=0) for k in keys}


class _BatchCoordinator:
    """Pools worker forward passes into one batched call per head, per round.

    ``infer_fn(model_key, stacked_inputs) -> {output_name: array}`` runs a head on a stacked batch
    (the master :meth:`GameSimulator._infer`, or a dummy in tests). The coordinator owns no game
    state; it only schedules.
    """

    def __init__(self, infer_fn: Callable[[str, dict], dict], n_workers: int,
                 progress: "_Progress | None" = None):
        self.infer_fn = infer_fn
        self.live = n_workers
        self.cond = threading.Condition()
        self.pending: dict[int, tuple[str, dict]] = {}   # worker_id -> (model_key, inputs)
        self.results: dict[int, dict] = {}               # worker_id -> output dict
        self.progress = progress

    # --- worker-facing API (called from worker threads) ---
    def request(self, worker_id: int, model_key: str, inputs: dict) -> dict:
        """Submit one forward pass and block until the batched result for this worker is ready."""
        with self.cond:
            self.pending[worker_id] = (model_key, inputs)
            self.cond.notify_all()
            while worker_id not in self.results:
                self.cond.wait()
            return self.results.pop(worker_id)

    def worker_done(self, n_events: int = 0) -> None:
        """A worker finished its game; drop it from the live set so the barrier can re-evaluate."""
        with self.cond:
            self.live -= 1
            self.cond.notify_all()
        if self.progress is not None:
            self.progress.complete_one(n_events)

    # --- coordinator loop (run on the main/driver thread) ---
    def run(self) -> None:
        """Batch-and-dispatch until every worker has finished."""
        while True:
            with self.cond:
                # Wait until all still-running workers are blocked on a request (or all are done).
                while self.live > 0 and len(self.pending) < self.live:
                    self.cond.wait()
                if self.live == 0 and not self.pending:
                    break
                batch = self.pending
                self.pending = {}
            # Heavy work outside the lock — every batched worker is parked in request().
            outs = self._run_batch(batch)
            with self.cond:
                self.results.update(outs)
                self.cond.notify_all()
            if self.progress is not None:
                self.progress.tick(passes=len(self._last_groups), rows=len(batch))

    def _run_batch(self, batch: dict[int, tuple[str, dict]]) -> dict[int, dict]:
        """Group requests by head, run each head once on the stacked batch, split results back."""
        groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for wid, (model_key, inputs) in batch.items():
            groups[model_key].append((wid, inputs))
        self._last_groups = groups

        results: dict[int, dict] = {}
        for model_key, items in groups.items():
            wids = [w for w, _ in items]
            stacked = _stack([inp for _, inp in items])
            out = self.infer_fn(model_key, stacked)            # one forward pass for the whole group
            for i, wid in enumerate(wids):
                results[wid] = {k: v[i:i + 1] for k, v in out.items()}
        return results


# ===================================================================== #
# --- Progress / throughput display                                    --
# ===================================================================== #

@dataclass
class _Progress:
    """Live cmd readout of rollout throughput (game-sims/min, events/s, forward-passes/s, ETA)."""
    total: int
    enabled: bool = True
    completed: int = 0
    events: int = 0
    passes: int = 0
    rows: int = 0
    _start: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_render: float = 0.0

    def complete_one(self, n_events: int) -> None:
        with self._lock:
            self.completed += 1
            self.events += int(n_events)
        self._maybe_render(force=True)

    def tick(self, *, passes: int, rows: int) -> None:
        with self._lock:
            self.passes += int(passes)
            self.rows += int(rows)
        self._maybe_render()

    def _maybe_render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self._last_render < 0.25:   # throttle redraws
            return
        self._last_render = now
        elapsed = max(now - self._start, 1e-6)
        with self._lock:
            done, total, ev, ps, rw = self.completed, self.total, self.events, self.passes, self.rows
        pct = 100.0 * done / total if total else 100.0
        rate_min = done / (elapsed / 60.0)
        eta = (total - done) / rate_min * 60.0 if rate_min > 0 else float("inf")
        occ = (rw / ps) if ps else 0.0
        bar_n = 24
        filled = int(bar_n * done / total) if total else bar_n
        bar = "#" * filled + "-" * (bar_n - filled)
        line = (f"\r[{bar}] {pct:5.1f}%  {done}/{total} sims  "
                f"{rate_min:5.1f}/min  ev/s {ev / elapsed:6.0f}  fwd/s {ps / elapsed:6.0f}  "
                f"batch {occ:4.1f}  ETA {self._fmt(eta)}   ")
        sys.stdout.write(line)
        sys.stdout.flush()

    @staticmethod
    def _fmt(seconds: float) -> str:
        if seconds == float("inf") or seconds != seconds:
            return "—"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

    def finish(self) -> None:
        if not self.enabled:
            return
        elapsed = max(time.monotonic() - self._start, 1e-6)
        rate_hr = self.completed / (elapsed / 3600.0)
        per_1000 = (1000.0 / rate_hr) if rate_hr > 0 else float("inf")
        sys.stdout.write("\n")
        print(f"[batched-rollout] {self.completed} game-sims in {self._fmt(elapsed)} "
              f"({rate_hr:.0f}/hr, avg batch {self.rows / self.passes if self.passes else 0:.1f}). "
              f"1,000 game-sims ≈ {self._fmt(per_1000)} of GPU wall-clock.")


# ===================================================================== #
# --- Public driver                                                    --
# ===================================================================== #

def run_jobs_batched(master: GameSimulator, jobs: list[GameJob], *, batch_size: int,
                     greedy: bool = False, show_progress: bool = True,
                     progress: _Progress | None = None) -> list[list[dict]]:
    """Run ``jobs`` in cohorts of ``batch_size`` concurrent games; return histories in job order.

    Each cohort spins up one worker thread per game (each a real ``GameController`` over a
    ``_WorkerSim``) plus the coordinator on this thread; all share the master's weights. The result
    list is aligned to ``jobs``.
    """
    histories: list[list[dict]] = [None] * len(jobs)  # type: ignore[list-item]
    if progress is None:
        progress = _Progress(total=len(jobs), enabled=show_progress)

    for start in range(0, len(jobs), batch_size):
        cohort = list(range(start, min(start + batch_size, len(jobs))))
        coord = _BatchCoordinator(master._infer, n_workers=len(cohort), progress=progress)

        def _worker(slot: int, job_idx: int) -> None:
            job = jobs[job_idx]
            wsim = _WorkerSim(master, coord, worker_id=slot)
            ctrl = GameController(wsim, seed=job.seed, greedy=greedy)
            ctrl.start(job.home_roster, job.away_roster, possession=job.possession,
                       season=str(job.season), home_starters=job.home_starters,
                       away_starters=job.away_starters, season_context=job.season_context)
            history = None
            try:
                history = ctrl.run()
                histories[job_idx] = history
            finally:
                coord.worker_done(len(history) if history else 0)

        threads = [threading.Thread(target=_worker, args=(slot, job_idx), daemon=True)
                   for slot, job_idx in enumerate(cohort)]
        for t in threads:
            t.start()
        coord.run()
        for t in threads:
            t.join()

    if show_progress:
        progress.finish()
    return histories


__all__ = ["GameJob", "run_jobs_batched", "_BatchCoordinator", "_WorkerSim", "_Progress"]
