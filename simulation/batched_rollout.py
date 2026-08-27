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
import traceback
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
        # Two conditions over ONE lock (so there is no lost-wakeup window between them). A submit
        # only ever needs to wake the coordinator, but a single shared Condition made it wake all
        # B-1 sleeping workers too — B notifies x B sleepers x ~2500 rounds is millions of
        # pointless GIL handoffs per run, on the exact thread pool the rollout is bottlenecked on.
        self.lock = threading.Lock()
        self.submit_cond = threading.Condition(self.lock)   # only the coordinator waits here
        self.result_cond = threading.Condition(self.lock)   # workers wait here
        self.pending: dict[int, tuple[str, dict]] = {}   # worker_id -> (model_key, inputs)
        self.results: dict[int, dict] = {}               # worker_id -> output dict
        self.progress = progress

    # --- worker-facing API (called from worker threads) ---
    def request(self, worker_id: int, model_key: str, inputs: dict) -> dict:
        """Submit one forward pass and block until the batched result for this worker is ready."""
        with self.result_cond:                  # same lock as submit_cond
            self.pending[worker_id] = (model_key, inputs)
            self.submit_cond.notify()           # wake ONLY the coordinator
            while worker_id not in self.results:
                self.result_cond.wait()
            return self.results.pop(worker_id)

    def worker_done(self) -> None:
        """This slot will submit no more requests; drop it so the barrier can re-evaluate.

        Called once per *slot*, at slot exit — not once per game. A slot runs many games
        back-to-back (see :func:`run_jobs_batched`), and ``live`` must mean "slots that can still
        submit", which is what the barrier predicate ``len(pending) < live`` tests.
        """
        with self.lock:
            self.live -= 1
            self.submit_cond.notify()

    # --- coordinator loop (run on the main/driver thread) ---
    def run(self) -> None:
        """Batch-and-dispatch until every worker has finished."""
        while True:
            with self.submit_cond:
                # Wait until all still-running workers are blocked on a request (or all are done).
                while self.live > 0 and len(self.pending) < self.live:
                    self.submit_cond.wait()
                if self.live == 0 and not self.pending:
                    break
                batch = self.pending
                self.pending = {}
            # Heavy work outside the lock — every batched worker is parked in request().
            outs = self._run_batch(batch)
            with self.result_cond:
                self.results.update(outs)
                self.result_cond.notify_all()   # every parked worker does have a result now
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
        # A shard's stdout is a log file, not a terminal (see eval_pool._launch). Redrawing a
        # carriage-return bar 4x/sec into a file leaves megabytes of overwritten frames that only
        # render after stripping the CRs; a newline-terminated line every 30s makes `tail -f` the
        # liveness check for a multi-hour shard.
        tty = sys.stdout.isatty()
        if not force and now - self._last_render < (0.25 if tty else 30.0):
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
        line = (f"[{bar}] {pct:5.1f}%  {done}/{total} sims  "
                f"{rate_min:5.1f}/min  ev/s {ev / elapsed:6.0f}  fwd/s {ps / elapsed:6.0f}  "
                f"batch {occ:4.1f}  ETA {self._fmt(eta)}   ")
        sys.stdout.write(f"\r{line}" if tty else f"{line}\n")
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
        if sys.stdout.isatty():
            sys.stdout.write("\n")
        print(f"[batched-rollout] {self.completed} game-sims in {self._fmt(elapsed)} "
              f"({rate_hr:.0f}/hr, avg batch {self.rows / self.passes if self.passes else 0:.1f}). "
              f"1,000 game-sims ≈ {self._fmt(per_1000)} of GPU wall-clock.")


# ===================================================================== #
# --- Public driver                                                    --
# ===================================================================== #

def run_jobs_batched(master: GameSimulator, jobs: list[GameJob], *, batch_size: int,
                     greedy: bool = False, show_progress: bool = True,
                     progress: _Progress | None = None,
                     on_complete: Callable[[int, list[dict]], None] | None = None,
                     keep_histories: bool = True) -> list[list[dict] | None]:
    """Run ``jobs`` on a pool of ``batch_size`` concurrent slots; return histories in job order.

    ``batch_size`` is the number of games in flight at once (the VRAM knob), **not** a cohort
    size. Each slot is a thread that pulls the next unclaimed job, plays it with a real
    ``GameController`` over a ``_WorkerSim``, and immediately pulls another — so a slot that draws
    a short game backfills instead of idling. The previous strict-cohort loop drained and restarted
    the whole pool every ``batch_size`` jobs, and since real games run ~400-1200 events, the last
    fifth of every cohort ran at an effective batch of 1-5. Slots share the master's weights and
    the one coordinator; the result list is aligned to ``jobs``.

    Scheduling only: ``histories`` is index-addressed and ``GameJob.seed`` does not depend on
    position, so which games happen to be concurrent cannot change any game's result.

    ``on_complete(job_idx, history)`` (when given) fires **on the slot thread**, the moment that
    job's rollout returns — the seam a caller uses to persist a finished sim immediately instead of
    waiting for the whole pool to drain. Paired with ``keep_histories=False`` it also bounds memory:
    the driver drops its reference, so a 100-sim game holds one history at a time per slot rather
    than all 100 until the end. Both defaults reproduce the original behaviour exactly.

    A job that raises is logged and skipped; its slot keeps pulling work. That matters more than it
    sounds: a slot that dies never comes back, so one bad sim used to narrow the pool for the rest
    of a multi-hour run.
    """
    histories: list[list[dict]] = [None] * len(jobs)  # type: ignore[list-item]
    if progress is None:
        progress = _Progress(total=len(jobs), enabled=show_progress)
    if not jobs:
        if show_progress:
            progress.finish()
        return histories

    n_slots = max(1, min(batch_size, len(jobs)))
    coord = _BatchCoordinator(master._infer, n_workers=n_slots, progress=progress)

    claim_lock = threading.Lock()
    cursor = 0

    def _claim() -> int | None:
        """Hand out the next unclaimed job index, or None once the list is drained."""
        nonlocal cursor
        with claim_lock:
            if cursor >= len(jobs):
                return None
            cursor += 1
            return cursor - 1

    def _slot(slot: int) -> None:
        try:
            while (job_idx := _claim()) is not None:
                job = jobs[job_idx]
                history = None
                try:
                    wsim = _WorkerSim(master, coord, worker_id=slot)
                    ctrl = GameController(wsim, seed=job.seed, greedy=greedy)
                    ctrl.start(job.home_roster, job.away_roster, possession=job.possession,
                               season=str(job.season), home_starters=job.home_starters,
                               away_starters=job.away_starters, season_context=job.season_context)
                    history = ctrl.run()
                    # Recorded BEFORE the sink runs: a failing sink must not lose a good sim.
                    if keep_histories:
                        histories[job_idx] = history
                    if on_complete is not None:
                        on_complete(job_idx, history)
                except Exception as e:      # noqa: BLE001 - one bad sim must not kill the slot
                    if sys.stdout.isatty():
                        sys.stdout.write("\n")
                    print(f"[batched-rollout] job {job_idx} (seed {job.seed}) failed: {e!r}")
                    traceback.print_exc()
                finally:
                    n_events = len(history) if history else 0
                    history = None          # release this slot's reference before the next claim
                    progress.complete_one(n_events)
        finally:
            # Once per slot, at slot exit — the barrier counts slots, not games.
            coord.worker_done()

    threads = [threading.Thread(target=_slot, args=(slot,), daemon=True)
               for slot in range(n_slots)]
    for t in threads:
        t.start()
    coord.run()
    for t in threads:
        t.join()

    if show_progress:
        progress.finish()
    return histories


__all__ = ["GameJob", "run_jobs_batched", "_BatchCoordinator", "_WorkerSim", "_Progress"]
