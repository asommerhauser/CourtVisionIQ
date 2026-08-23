"""
eval_pool.py -- supervise N eval processes over disjoint holdout slices, then merge one report.

Batching already fills the GPU *within* a process (``simulation/batched_rollout.py``), but the work
around each forward pass -- building inputs, sampling, running the rule engine -- is Python, so one
process is capped by the GIL at roughly one core. On a 16- or 32-core pod that leaves most of the
machine idle. The fix is processes, not more threads.

``evaluate.py --shard I/N`` is the primitive: it simulates ``holdout[I-1::N]`` and skips the
aggregate report. This module is only a **supervisor** over that primitive -- it sizes the pool,
launches the children, renders one merged progress line, and merges the report at the end. Keeping
the primitive separate means the same code path also serves a hand-driven split across two
machines, and there is no second implementation to keep in sync.

Design notes worth keeping:

  * **Top-level module, not under ``simulation/``.** ``simulation/__init__.py`` imports
    ``GameSimulator``, which pulls TensorFlow. A supervisor whose only job is to spawn children
    must never load TF: it would cost seconds per launch and, worse, take a CUDA context's worth of
    VRAM away from the workers. Everything imported here is TF-free, and
    ``tests/test_eval_pool.py`` asserts it.
  * **subprocess, not multiprocessing.** Each child is a fresh ``python evaluate.py --shard i/N``.
    That guarantees no child inherits a CUDA context -- stronger than
    ``set_start_method("spawn")``, and impossible to undo by an import somewhere -- and it follows
    the pattern ``shell/actions.launch_train`` already uses for detached training runs.
  * **Static slices, not a work queue.** ``holdout[i-1::N]`` is disjoint by construction and needs
    no claim protocol; two workers racing one game folder would interleave writes into eight files.
    Resume is free: a game that already has a ``record.json`` is skipped
    (``simulation/stage_eval.py``). Stride rather than block, because the holdout is chronological
    and block slicing would hand one shard a run of overtime games.
  * **Waves instead of dynamic balancing.** When every child has exited, any game still missing a
    ``record.json`` (a crashed child, an unlucky slice) goes into a second, smaller pool over
    exactly the remainder. That recovers most of what a work queue would, with none of the
    protocol -- and every child of a wave is reaped before the next wave computes its remainder, so
    two waves can never target the same game folder.
  * **Dials travel with the pool.** A child re-reads ``config.py`` from disk, so a parent that
    tuned dials at runtime must hand them over explicitly or the run silently mixes two tunings
    into one report. :func:`shard_commands` requires the dial file for that reason, and
    :func:`assert_one_tuning` checks after the fact.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import config

# A shard doing fewer than this many games spends more time loading 11 model heads than simulating.
MIN_GAMES_PER_SHARD = 2

REPO_ROOT = Path(__file__).resolve().parent


# ===================================================================== #
# --- Sizing                                                           --
# ===================================================================== #

def free_vram_gb() -> float | None:
    """Free VRAM on device 0 in GiB, or None if it cannot be determined.

    Deliberately context-free -- both probes read the driver without initializing CUDA, so asking
    the question does not itself cost the memory we are trying to measure.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
            return info.free / (1024 ** 3)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0]) / 1024.0
    except Exception:
        pass
    return None


def usable_cores() -> int:
    """Cores this process may actually run on.

    ``sched_getaffinity`` rather than ``cpu_count`` because a CPU-limited container on a rented pod
    reports the *host's* core count from ``cpu_count()`` -- size a pool off that and you
    oversubscribe several times over and finish slower than one process would.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except OSError:
            pass
    return os.cpu_count() or 1


def autosize_procs(*, holdout_games: int, requested=None,
                   reserved_vram_gb: float = 0.0) -> tuple[int, str]:
    """How many eval processes to run, plus a one-line explanation of how that number was reached.

    ``requested`` wins when it is an explicit integer (clamped to something runnable, with the
    clamp shown). ``None`` or ``"auto"`` takes the minimum of what the cores, the free VRAM, and
    the amount of work each allow. ``reserved_vram_gb`` is how a caller says "I am already holding
    a model" -- the resident shell does, and its children have to fit in what is left.

    The explanation string is not decoration: an auto-sized pool that picks 1 on a 32-core box
    should say *which* cap bound it, or the user has no way to tell a VRAM probe failure from a
    genuinely small holdout. Always returns >= 1.
    """
    cap = max(1, holdout_games)
    if requested not in (None, "auto", ""):
        n = int(requested)
        if n < 1:
            raise ValueError(f"--procs must be >= 1 (or 'auto'), got {requested!r}")
        clamped = min(n, cap)
        why = f"procs {clamped}  (requested {n}"
        why += f", clamped to the {cap} unfinished games)" if clamped != n else ", explicit)"
        return clamped, why

    cores = usable_cores()
    cpu_cap = max(1, cores // max(1, config.EVAL_PROC_CORES))
    work_cap = max(1, holdout_games // MIN_GAMES_PER_SHARD)

    free = free_vram_gb()
    caps = [cpu_cap, work_cap, config.EVAL_PROC_MAX]
    if free is None:
        vram_why = "vram unknown"
    else:
        vram_cap = max(1, int((free - reserved_vram_gb) // config.EVAL_PROC_VRAM_GB))
        caps.append(vram_cap)
        vram_why = (f"vram ({free:.1f}-{reserved_vram_gb:.1f})/"
                    f"{config.EVAL_PROC_VRAM_GB:g}={vram_cap}")

    n = max(1, min(caps))
    why = (f"procs {n}  (cpu {cores}/{config.EVAL_PROC_CORES}={cpu_cap}, {vram_why}, "
           f"work {holdout_games}/{MIN_GAMES_PER_SHARD}={work_cap}, "
           f"cap {config.EVAL_PROC_MAX})")
    return n, why


# ===================================================================== #
# --- Shards                                                           --
# ===================================================================== #

@dataclass
class Shard:
    """One child process: the command that runs it and where its output lands."""
    index: int
    total: int
    cmd: list[str]
    log: Path
    proc: subprocess.Popen | None = None
    rc: int | None = None
    _fh: object = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return f"{self.index}/{self.total}"

    @property
    def failed(self) -> bool:
        return self.rc is not None and self.rc != 0


def shard_commands(*, model: str | None, run: str, n: int, dials_path, state_path,
                   run_dir, sims=None, concurrency=None, seed=0, python: str | None = None,
                   tag: str = "") -> list[Shard]:
    """Build the N ``evaluate.py --shard i/N`` commands and their log paths.

    ``dials_path`` is required, not optional: it is the only thing stopping a child from re-reading
    ``config.py`` and running different physics than the parent that launched it. ``report_every``
    is deliberately never forwarded -- an intermediate flush from a child would write exactly the
    shared report files a shard is not allowed to touch.
    """
    python = python or sys.executable
    logs = Path(run_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    shards = []
    for i in range(1, n + 1):
        cmd = [python, "evaluate.py", "--run", str(run), "--shard", f"{i}/{n}",
               "--dials", str(dials_path), "--state", str(state_path)]
        if model:
            cmd[2:2] = ["--model", str(model)]
        if sims:
            cmd += ["--monte-carlo", str(sims)]
        if concurrency:
            cmd += ["--concurrency", str(concurrency)]
        if seed:
            cmd += ["--seed", str(seed)]
        shards.append(Shard(index=i, total=n, cmd=cmd,
                            log=logs / f"shard{tag}-{i}of{n}.log"))
    return shards


# ===================================================================== #
# --- Supervision                                                      --
# ===================================================================== #

def finished_games(run_dir) -> int:
    """Games with a cached record -- the same marker the resume path uses, so it cannot disagree."""
    return len(list((Path(run_dir) / "games").glob("*/record.json")))


def _bar(done: int, total: int, width: int = 24) -> str:
    filled = int(width * done / total) if total else width
    return "#" * filled + "-" * (width - filled)


def _fmt(seconds: float) -> str:
    if seconds in (float("inf"), float("-inf")) or seconds != seconds:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _launch(shard: Shard, *, cwd) -> None:
    env = dict(os.environ)
    # Without this the second child dies the moment it touches the GPU: TF grabs the whole card by
    # default, so children 2..N would find nothing left.
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    shard._fh = open(shard.log, "w", encoding="utf-8", errors="replace")
    # Both streams into the log file, stdin closed: the supervisor never holds a pipe to drain, so
    # it cannot deadlock behind a child that out-writes the buffer.
    shard.proc = subprocess.Popen(shard.cmd, cwd=str(cwd), stdout=shard._fh,
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)


def _reap(shard: Shard) -> None:
    """Poll one shard and record its exit code the first time it has one."""
    if shard.proc is None or shard.rc is not None:
        return
    rc = shard.proc.poll()
    if rc is None:
        return
    shard.rc = rc
    if shard._fh is not None:
        shard._fh.close()
        shard._fh = None


def run_pool(shards: list[Shard], *, run_dir, total_games: int, cwd=REPO_ROOT,
             echo=print, poll: float = 0.5, render: bool = True) -> list[Shard]:
    """Launch every shard, render one merged progress line, and wait for them all to exit.

    Progress is read from the run dir rather than from the children: the count of per-game
    ``record.json`` files is authoritative, survives a child restart, and needs no cooperation from
    the child processes at all.

    A failing child does not stop its siblings -- its games simply stay unfinished, which the
    resume path and the remainder wave already handle. Ctrl-C stops the pool and leaves every
    finished game cached.
    """
    run_dir = Path(run_dir)
    start = time.monotonic()
    started = finished_games(run_dir)
    try:
        for s in shards:
            _launch(s, cwd=cwd)
    except Exception:
        # A failed launch part-way through would otherwise orphan the children already running,
        # each holding a CUDA context on a card nobody is watching.
        _terminate(shards)
        for s in shards:
            if s._fh is not None:
                s._fh.close()
                s._fh = None
        raise

    try:
        while any(s.rc is None for s in shards):
            for s in shards:
                _reap(s)
            if render:
                _render(shards, run_dir, total_games, start, started)
            if any(s.rc is None for s in shards):
                time.sleep(poll)
    except KeyboardInterrupt:
        echo("\n[pool] interrupted - stopping shards. Finished games stay cached; re-run to resume.")
        _terminate(shards)
        raise
    finally:
        for s in shards:
            if s._fh is not None:
                s._fh.close()
                s._fh = None

    if render:
        _render(shards, run_dir, total_games, start, started, final=True)
        made = max(finished_games(run_dir) - started, 0)
        echo(f"[pool] {made} games in {_fmt(time.monotonic() - start)} "
             f"across {len(shards)} processes.")
    return shards


def _render(shards, run_dir, total_games, start, started, *, final=False) -> None:
    done = finished_games(run_dir)
    elapsed = max(time.monotonic() - start, 1e-6)
    rate = max(done - started, 0) / elapsed
    eta = (total_games - done) / rate if rate > 0 else float("inf")
    live = sum(1 for s in shards if s.rc is None)
    pct = 100.0 * done / total_games if total_games else 100.0
    sys.stdout.write(f"\r[{_bar(done, total_games)}] {pct:5.1f}%  {done}/{total_games} games  "
                     f"{live}/{len(shards)} procs  {rate * 3600:5.1f} games/hr  "
                     f"ETA {_fmt(eta)}   ")
    if final:
        sys.stdout.write("\n")
    sys.stdout.flush()


def _terminate(shards, grace: float = 20.0) -> None:
    for s in shards:
        if s.proc is not None and s.rc is None:
            s.proc.terminate()
    deadline = time.monotonic() + grace
    for s in shards:
        if s.proc is None or s.rc is not None:
            continue
        try:
            s.rc = s.proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            s.proc.kill()
            s.rc = s.proc.wait()


def report_failures(shards, *, echo=print, tail_lines: int = 40) -> list[Shard]:
    """Print the tail of each failed shard's log plus its exact re-run command."""
    failed = [s for s in shards if s.failed]
    for s in failed:
        echo(f"\n[pool] shard {s.label} exited {s.rc}. Last {tail_lines} lines of {s.log}:")
        try:
            tail = s.log.read_text(encoding="utf-8", errors="replace").splitlines()[-tail_lines:]
            for line in tail:
                echo("    " + line)
        except OSError as e:
            echo(f"    (could not read log: {e})")
        echo("  re-run just this shard:\n    " + " ".join(s.cmd))
    return failed


def run_waves(*, build, run_dir, total_games: int, cwd=REPO_ROOT, echo=print,
              max_waves: int = 3, render: bool = True) -> list[Shard]:
    """Run the pool, then re-pool over whatever is still unfinished, up to ``max_waves``.

    ``build(wave, remaining)`` returns the shard commands for one wave (or an empty list to stop).
    A wave that finishes zero new games ends the loop: without that guard a single
    permanently-failing game would respawn a pool forever.
    """
    run_dir = Path(run_dir)
    all_shards: list[Shard] = []
    for wave in range(1, max_waves + 1):
        remaining = total_games - finished_games(run_dir)
        if remaining <= 0:
            break
        shards = build(wave, remaining)
        if not shards:
            break
        if wave > 1:
            echo(f"\n[pool] wave {wave}: {remaining} games still unfinished, re-sharding across "
                 f"{len(shards)} processes.")
        before = finished_games(run_dir)
        # Every child of this wave is reaped inside run_pool before the next wave computes its
        # remainder, so two waves can never target the same game folder.
        run_pool(shards, run_dir=run_dir, total_games=total_games, cwd=cwd, echo=echo,
                 render=render)
        all_shards.extend(shards)
        report_failures(shards, echo=echo)
        if finished_games(run_dir) <= before:
            echo("[pool] a wave finished zero new games - stopping rather than respawning.")
            break
    return all_shards


# ===================================================================== #
# --- Post-run integrity                                               --
# ===================================================================== #

def assert_one_tuning(run_dir, *, echo=print) -> bool:
    """Warn if the finished games were not all simulated under the same inference dials.

    The failure this catches is the quiet one: a child re-reads ``config.py`` and runs different
    physics than the parent, producing one report over two tunings with nothing raising.
    ``record.json`` carries ``tuning`` captured per game *at sim time*, so the evidence is there.
    """
    seen: dict[str, set] = {}
    for rec in Path(run_dir).glob("games/*/record.json"):
        try:
            tuning = json.loads(rec.read_text(encoding="utf-8")).get("tuning") or {}
        except (OSError, json.JSONDecodeError):
            continue
        for k, v in tuning.items():
            seen.setdefault(k, set()).add(json.dumps(v, sort_keys=True))
    mixed = {k: v for k, v in seen.items() if len(v) > 1}
    if mixed:
        echo(f"[pool] WARNING: this run's games were simulated under DIFFERENT dials: "
             f"{', '.join(sorted(mixed))}. The aggregate mixes tunings and is not comparable.")
        return False
    return True


# ===================================================================== #
# --- Entry point (evaluate.py --procs N)                              --
# ===================================================================== #

def run_procs(args) -> None:
    """Supervise a pooled eval for ``evaluate.py --procs N``. Never imports TensorFlow.

    Resolves the run dir ONCE and passes the concrete name to every child, so the auto-``eval-NNN``
    race that ``--shard`` has to guard against cannot happen here. The merge runs as one more
    subprocess for the same reason the supervisor exists: importing ``evaluate_stage`` pulls
    ``GameSimulator``, and this process should stay TF-free start to finish.
    """
    from models.artifacts import model_name                 # TF-free
    from reporting.eval_report import resolve_results_run_dir   # pandas, not TF

    state_path = Path(args.state)
    if not state_path.is_file():
        raise SystemExit(f"No full-run state at {state_path}; --procs needs it for the holdout.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    holdout = state.get("holdout_game_ids") or []
    if not holdout:
        raise SystemExit(f"{state_path} has no holdout_game_ids.")

    name_ = model_name(args.model or state.get("version") or config.DEFAULT_MODEL)
    run_dir = resolve_results_run_dir(name_, name=args.run, holdout_total=len(holdout))
    run_name = run_dir.name

    # The dial package: written once, handed to every child. Without it each child re-reads
    # config.py and a runtime-tuned parent silently gets a report over two different tunings.
    dials_path = run_dir / "dials.json"
    config.write_dial_file(dials_path)

    total = len(holdout)
    done = finished_games(run_dir)
    if done >= total:
        print(f"[pool] all {total} games already finished in {run_dir}; merging the report.")
    else:
        def build(wave, remaining):
            n, why = autosize_procs(holdout_games=remaining, requested=args.procs)
            if wave == 1:
                print(f"[pool] {name_} -> {run_dir}")
                print(f"[pool] {total} holdout games ({done} already done), dials -> {dials_path}")
                print(f"[pool] {why}")
            if n <= 1 and wave == 1:
                return []      # caller falls back to the in-process path
            return shard_commands(model=name_, run=run_name, n=n, dials_path=dials_path,
                                  state_path=state_path, run_dir=run_dir,
                                  sims=args.monte_carlo, concurrency=args.concurrency,
                                  seed=args.seed, tag="" if wave == 1 else f"-w{wave}")

        shards = run_waves(build=build, run_dir=run_dir, total_games=total)
        if not shards:
            print("[pool] pool of 1 -- running in this process instead.")
            from training.full_run import FullRun            # the TF import, only on this path
            FullRun(state_path=str(state_path)).eval(
                version=args.model, name=run_name, n_sims=args.monte_carlo,
                concurrency=args.concurrency, seed=args.seed)
            return

    assert_one_tuning(run_dir)
    merge_report(model=name_, run=run_name, state_path=state_path)


def merge_report(*, model: str, run: str, state_path, python: str | None = None,
                 cwd=REPO_ROOT) -> int:
    """Rebuild the aggregate over every finished game, in a child so the supervisor stays TF-free."""
    cmd = [python or sys.executable, "evaluate.py", "--model", model, "--run", run,
           "--report-only", "--state", str(state_path)]
    print(f"[pool] merging the report: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(cwd)).returncode


__all__ = ["Shard", "autosize_procs", "free_vram_gb", "usable_cores", "shard_commands",
           "run_pool", "run_waves", "report_failures", "finished_games", "assert_one_tuning",
           "run_procs", "merge_report", "MIN_GAMES_PER_SHARD"]
