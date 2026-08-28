"""
harvest.py -- move finished games' play-by-play off a live run dir, so a long eval cannot fill the disk.

A finished game folder is ~14 MB, of which 13.9 MB is ``playbyplay/`` (one CSV per sim) and 28 KB is
``record.json``. Nothing in the sim path or the report path ever reads a play-by-play back:

  * ``record.json`` is the sole completion marker -- ``eval_pool.finished_games`` counts it and
    ``simulation/stage_eval.evaluate_stage`` reloads it instead of re-simulating the game.
  * ``record.json`` is the sole report input -- ``eval_pool.build_run_report`` globs
    ``games/*/record.json`` and reads nothing else.

So a finished game's pbp can be archived off the volume and deleted, and the run still skips that
game on resume and still merges a correct final report. That is what this does, on a timer, next to
a running pool -- turning a disk that fills up every ~30 games into one that stays flat.

Design notes worth keeping:

  * **Top-level module, stdlib only, TF-free.** Same rule as ``eval_pool.py`` and for the same
    reason: a helper that imported TensorFlow would take a CUDA context's worth of VRAM away from
    the very workers it is babysitting. ``tests/test_harvest.py`` asserts it.
  * **Harvestable means ``record.json`` exists.** That test is race-free by construction:
    ``record.json`` is written last and atomically (``simulation/stage_eval._atomic_write``), after
    every one of that game's pbp files is already on disk. A folder without it is owned by a live
    shard and is never touched while looping.
  * **Verify, then delete. Never the reverse.** The archive goes to a temp name, is reopened, and
    every member is checked against the source's name and size before a single file is removed. A
    corrupt or truncated archive costs a retry, not the data.
  * **Idempotent.** ``manifest.jsonl`` records what has been archived; a game already in it with its
    pbp gone is skipped, so the daemon can be killed and restarted at any point.
  * **The prune modes are the dangerous ones**, because they delete without archiving -- for games
    already safe on another machine, and for part-written folders a re-run would clear anyway
    (``_PbpSink.prepare``). Both refuse to run while an ``evaluate.py`` is alive, since a live
    shard's in-flight folder is exactly the one that has no ``record.json`` yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_NAME = "manifest.jsonl"
LOG_NAME = "harvest.log"
PBP_DIR = "playbyplay"
RECORD = "record.json"

_STOP = False


# ===================================================================== #
# --- Small shared helpers                                             --
# ===================================================================== #

def human(n: float) -> str:
    """Bytes as something a person can read off a terminal at 3am."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Log:
    """stdout plus an append-only file, so a nohup'd daemon leaves a record either way."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        if self.path is not None:
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass        # a full disk must not stop the thing whose job is to empty it


def game_dirs(run_dir: Path) -> list[Path]:
    games = run_dir / "games"
    if not games.is_dir():
        raise SystemExit(f"No games/ under {run_dir} -- is that a results run dir?")
    return sorted(p for p in games.iterdir() if p.is_dir())


def is_finished(game: Path) -> bool:
    """The run's own completion test, and the only one that is safe against a live writer."""
    return (game / RECORD).is_file()


def file_sizes(root: Path) -> dict[str, int]:
    """Every regular file under ``root``, keyed by path relative to ``root.parent``.

    Relative to the *parent* because that is the arcname a tar rooted at the game folder uses, so
    the verification pass can compare the two dicts directly.
    """
    out: dict[str, int] = {}
    base = root.parent
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            try:
                if p.is_symlink() or not p.is_file():
                    continue
                out[p.relative_to(base).as_posix()] = p.stat().st_size
            except OSError:
                continue
    return out


def pbp_stats(game: Path) -> tuple[int, int]:
    """(file count, bytes) of this game's ``playbyplay/``; (0, 0) when it is gone or empty."""
    pbp = game / PBP_DIR
    if not pbp.is_dir():
        return 0, 0
    sizes = file_sizes(pbp)
    return len(sizes), sum(sizes.values())


def game_id(game: Path) -> str:
    m = re.match(r"game(\d+)_", game.name)
    return m.group(1) if m else game.name


def free_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def live_evals(pattern: str = "evaluate.py") -> list[tuple[int, str]] | None:
    """Running ``evaluate.py`` processes, or None where /proc cannot answer (non-Linux).

    Stdlib only on purpose -- this runs on a pod with a full disk, and shelling out to ``ps`` is one
    more thing that can fail when the volume is wedged.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    found: list[tuple[int, str]] = []
    me = os.getpid()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        cmd = raw.decode("utf-8", "replace")
        if pattern in cmd:
            found.append((int(entry.name), cmd.replace("\0", " ").strip()))
    return found


def require_no_live_pool(log: Log, *, force: bool) -> None:
    """Refuse to prune while a shard runs -- its in-flight folder is exactly the one with no record."""
    procs = live_evals()
    if procs is None:
        log("WARNING: no /proc on this platform; cannot confirm the pool is stopped.")
        return
    if not procs:
        return
    for pid, cmd in procs:
        log(f"  live: pid {pid}  {cmd[:120]}")
    if force:
        log(f"WARNING: {len(procs)} evaluate.py process(es) alive; --force given, continuing anyway.")
        return
    raise SystemExit(
        f"Refusing to prune: {len(procs)} evaluate.py process(es) are still running. A live shard "
        f"owns exactly the folders that have no {RECORD} yet, and its streamed sims would be "
        f"deleted mid-game. Stop the pool first (pkill -f evaluate.py), or pass --force.")


# ===================================================================== #
# --- Manifest                                                         --
# ===================================================================== #

def read_manifest(out_dir: Path) -> dict[str, dict]:
    """Everything archived so far, keyed by game folder name. Tolerates a torn last line."""
    path = out_dir / MANIFEST_NAME
    entries: dict[str, dict] = {}
    if not path.is_file():
        return entries
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue        # a kill mid-append costs one line, not the manifest
            if rec.get("game"):
                entries[rec["game"]] = rec
    except OSError:
        pass
    return entries


def append_manifest(out_dir: Path, entry: dict) -> None:
    with (out_dir / MANIFEST_NAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ===================================================================== #
# --- Archive one game                                                 --
# ===================================================================== #

def archive_game(game: Path, out_dir: Path, *, log: Log) -> dict | None:
    """Tar a game folder, verify every member against the source, then drop its ``playbyplay/``.

    Returns the manifest entry, or None if anything went wrong -- in which case the pbp is still on
    disk, untouched. The order here is the whole safety property: write, verify, replace, delete.
    """
    final = out_dir / f"{game.name}.tar.gz"
    tmp = out_dir / f"{game.name}.tar.gz.tmp"
    expected = file_sizes(game)
    if not expected:
        return None

    try:
        with tarfile.open(tmp, "w:gz") as tf:
            tf.add(game, arcname=game.name, recursive=True)
    except (OSError, tarfile.TarError) as e:
        log(f"  {game.name}: archive FAILED ({e!r}); play-by-play left in place.")
        tmp.unlink(missing_ok=True)
        return None

    # Reopen and compare. A tar that lost a member, or wrote a short one, must never be the reason
    # the only other copy gets deleted.
    try:
        with tarfile.open(tmp, "r:gz") as tf:
            got = {m.name: m.size for m in tf.getmembers() if m.isfile()}
    except (OSError, tarfile.TarError) as e:
        log(f"  {game.name}: archive unreadable after write ({e!r}); play-by-play left in place.")
        tmp.unlink(missing_ok=True)
        return None

    if got != expected:
        missing = sorted(set(expected) - set(got))
        wrong = sorted(k for k in set(expected) & set(got) if expected[k] != got[k])
        log(f"  {game.name}: archive does NOT match source "
            f"({len(missing)} missing, {len(wrong)} wrong size); play-by-play left in place.")
        tmp.unlink(missing_ok=True)
        return None

    digest = sha256_of(tmp)
    size = tmp.stat().st_size
    os.replace(tmp, final)

    n_pbp, pbp_bytes = pbp_stats(game)
    try:
        shutil.rmtree(game / PBP_DIR)
    except OSError as e:
        log(f"  {game.name}: archived to {final.name} but could not delete pbp ({e!r}).")
        pbp_bytes = 0

    entry = {"game": game.name, "game_id": game_id(game), "tarball": final.name,
             "files": len(expected), "source_bytes": sum(expected.values()),
             "archive_bytes": size, "pbp_files": n_pbp, "pbp_bytes": pbp_bytes,
             "sha256": digest, "archived_at": _now()}
    append_manifest(out_dir, entry)
    log(f"  {game.name}: {len(expected)} files -> {final.name} ({human(size)}), "
        f"freed {human(pbp_bytes)}")
    return entry


# ===================================================================== #
# --- Modes                                                            --
# ===================================================================== #

def cmd_status(run_dir: Path, *, log: Log) -> None:
    """Everything an operator needs in one paste: disk, the finished/unfinished split, the game set."""
    games = game_dirs(run_dir)
    fin = [g for g in games if is_finished(g)]
    unfin = [g for g in games if not is_finished(g)]
    usage = shutil.disk_usage(run_dir)
    fin_files = fin_bytes = unfin_files = unfin_bytes = 0
    for g in fin:
        n, b = pbp_stats(g)
        fin_files += n
        fin_bytes += b
    for g in unfin:
        n, b = pbp_stats(g)
        unfin_files += n
        unfin_bytes += b
    run_bytes = sum(file_sizes(run_dir).values())

    log(f"run dir      {run_dir}")
    log(f"volume       {human(usage.used)} used / {human(usage.total)} total, "
        f"{human(usage.free)} free")
    log(f"run dir size {human(run_bytes)}")
    log(f"game folders {len(games)}  ({len(fin)} finished, {len(unfin)} unfinished)")
    log(f"finished pbp {fin_files} files, {human(fin_bytes)}   <- reclaimable: archive or prune")
    log(f"unfinish pbp {unfin_files} files, {human(unfin_bytes)}   <- waste: a re-run clears it anyway")
    partial = sum(1 for g in games if (g / "record.partial.json").is_file() and not is_finished(g))
    if partial:
        log(f"partials     {partial} folders hold a salvaged record.partial.json")
    procs = live_evals()
    if procs is None:
        log("live evals   (no /proc on this platform)")
    else:
        log(f"live evals   {len(procs)}")
        for pid, cmd in procs:
            log(f"             pid {pid}  {cmd[:110]}")
    log("finished game ids: " + " ".join(game_id(g) for g in fin))
    log("finished folders:")
    for g in fin:
        n, b = pbp_stats(g)
        log(f"  {g.name}  pbp={n} {human(b)}")


def cmd_archive(run_dir: Path, out_dir: Path, *, log: Log, dry_run: bool, keep: int,
                min_free_gb: float) -> tuple[int, int]:
    """One pass: archive every finished game that still has a play-by-play. Returns (games, bytes)."""
    games = game_dirs(run_dir)
    done = read_manifest(out_dir) if out_dir.is_dir() else {}
    candidates = [g for g in games if is_finished(g) and pbp_stats(g)[0] > 0]
    if keep > 0:
        # The newest N stay hot -- a knob for someone who wants the last few games inspectable on
        # the pod. The default is 0: harvest eagerly, keep the disk flat.
        by_age = sorted(candidates, key=lambda g: (g / RECORD).stat().st_mtime, reverse=True)
        hot = {p.name for p in by_age[:keep]}
        candidates = [g for g in candidates if g.name not in hot]

    n_games = n_bytes = 0
    for g in candidates:
        if _STOP:
            break
        if g.name in done and not (g / PBP_DIR).is_dir():
            continue
        n, b = pbp_stats(g)
        if dry_run:
            log(f"  would archive {g.name}: {n} pbp files, {human(b)}")
            n_games += 1
            n_bytes += b
            continue
        if free_bytes(out_dir) < min_free_gb * (1024 ** 3):
            log(f"  {g.name}: only {human(free_bytes(out_dir))} free on the archive volume "
                f"(--min-free-gb {min_free_gb:g}); skipping the rest of this pass.")
            break
        entry = archive_game(g, out_dir, log=log)
        if entry is not None:
            n_games += 1
            n_bytes += entry["pbp_bytes"]
    return n_games, n_bytes


def _drop_pbp(game: Path, *, log: Log, dry_run: bool) -> tuple[int, int]:
    n, b = pbp_stats(game)
    if n == 0:
        return 0, 0
    if dry_run:
        log(f"  would delete {game.name}/{PBP_DIR}: {n} files, {human(b)}")
        return 1, b
    try:
        shutil.rmtree(game / PBP_DIR)
    except OSError as e:
        log(f"  {game.name}: could not delete pbp ({e!r})")
        return 0, 0
    log(f"  deleted {game.name}/{PBP_DIR}: {n} files, {human(b)}")
    return 1, b


def cmd_prune_finished(run_dir: Path, names_file: Path, *, log: Log, dry_run: bool,
                       force: bool) -> tuple[int, int]:
    """Delete (without archiving) the pbp of the games named in ``names_file``.

    For games whose folder is already complete on another machine, where tarring them would only
    spend the space we are trying to reclaim. Every name must resolve to a folder that has a
    ``record.json`` *here* -- one that does not means the two machines disagree about what finished,
    and the whole operation is refused rather than half-applied.
    """
    require_no_live_pool(log, force=force)
    if not names_file.is_file():
        raise SystemExit(f"No such --already-home file: {names_file}")
    wanted = [ln.strip() for ln in names_file.read_text(encoding="utf-8").splitlines()]
    wanted = [w for w in wanted if w and not w.startswith("#")]
    if not wanted:
        raise SystemExit(f"{names_file} lists no games.")

    by_name = {g.name: g for g in game_dirs(run_dir)}
    by_id = {game_id(g): g for g in by_name.values()}
    resolved: list[Path] = []
    problems: list[str] = []
    for w in wanted:
        g = by_name.get(w) or by_id.get(w) or by_id.get(w.removeprefix("game").split("_")[0])
        if g is None:
            problems.append(f"  {w}: no such game folder on this machine")
        elif not is_finished(g):
            problems.append(f"  {w}: has no {RECORD} here -- it is NOT finished on this machine")
        else:
            resolved.append(g)
    if problems:
        log(f"{len(problems)} of {len(wanted)} listed games do not match this run:")
        for p in problems:
            log(p)
        raise SystemExit("Refusing to prune: the finished sets disagree. Reconcile first.")

    n_games = n_bytes = 0
    for g in resolved:
        n, b = _drop_pbp(g, log=log, dry_run=dry_run)
        n_games += n
        n_bytes += b
    return n_games, n_bytes


def cmd_prune_unfinished(run_dir: Path, *, log: Log, dry_run: bool,
                         force: bool) -> tuple[int, int]:
    """Clear pbp from every folder with no ``record.json``.

    Pure waste: ``_PbpSink.prepare`` deletes stale ``sim_*.csv`` before re-running a game anyway, so
    these files are guaranteed to be thrown away the moment the pool comes back to them. Any
    ``record.partial.json`` is left alone -- the re-run unlinks it itself, and it costs 28 KB.
    """
    require_no_live_pool(log, force=force)
    n_games = n_bytes = 0
    for g in game_dirs(run_dir):
        if is_finished(g):
            continue
        n, b = _drop_pbp(g, log=log, dry_run=dry_run)
        n_games += n
        n_bytes += b
    return n_games, n_bytes


# ===================================================================== #
# --- Entry point                                                      --
# ===================================================================== #

def _on_signal(signum, frame):      # pragma: no cover - exercised by hand, not by pytest
    """Stop after the current game, so a kill never lands between a verify and its delete."""
    global _STOP
    _STOP = True
    print(f"\n[harvest] signal {signum}: stopping after the current game.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Archive finished games' play-by-play off a results run dir so a long eval "
                    "cannot fill the volume. Safe to run beside a live pool.")
    ap.add_argument("--run", required=True, metavar="RUNDIR",
                    help="Results run dir, e.g. results/v1.0/full3-s100")
    ap.add_argument("--out", default=None, metavar="DIR",
                    help="Where tarballs, manifest.jsonl and harvest.log go. Required to archive.")
    ap.add_argument("--status", action="store_true",
                    help="Print the inventory (disk, finished/unfinished split, game ids) and exit.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what the chosen mode would archive or delete; touch nothing.")
    ap.add_argument("--loop", type=float, default=None, metavar="SECONDS",
                    help="Daemon mode: archive every SECONDS until killed (60 is the usual).")
    ap.add_argument("--keep", type=int, default=0, metavar="N",
                    help="Leave the pbp of the N most recently finished games on disk (default 0).")
    ap.add_argument("--min-free-gb", type=float, default=0.5, metavar="GB",
                    help="Skip archiving when the archive volume has less than this free.")
    ap.add_argument("--prune-finished", action="store_true",
                    help="One-shot: delete (without archiving) the pbp of the games in "
                         "--already-home. Requires the pool to be stopped.")
    ap.add_argument("--already-home", default=None, metavar="FILE",
                    help="File of game folder names (or ids), one per line, already safe elsewhere.")
    ap.add_argument("--prune-unfinished", action="store_true",
                    help="One-shot: clear pbp from every folder with no record.json. Requires the "
                         "pool to be stopped.")
    ap.add_argument("--force", action="store_true",
                    help="Let the prune modes run even with an evaluate.py alive. Don't.")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise SystemExit(f"No such run dir: {run_dir}")
    out_dir = Path(args.out) if args.out else None
    if sum(bool(x) for x in (args.status, args.prune_finished, args.prune_unfinished)) > 1:
        ap.error("--status, --prune-finished and --prune-unfinished are separate one-shot modes.")
    if args.prune_finished and not args.already_home:
        ap.error("--prune-finished needs --already-home FILE (the games already safe elsewhere).")
    if args.keep < 0:
        ap.error("--keep must be >= 0")

    quiet_file = out_dir is None or args.dry_run or args.status
    log = Log(None if quiet_file else out_dir / LOG_NAME)

    if args.status:
        cmd_status(run_dir, log=log)
        return 0

    if args.prune_finished:
        n, b = cmd_prune_finished(run_dir, Path(args.already_home), log=log,
                                  dry_run=args.dry_run, force=args.force)
        log(f"prune-finished: {n} games, {'would free' if args.dry_run else 'freed'} {human(b)}")
        return 0

    if args.prune_unfinished:
        n, b = cmd_prune_unfinished(run_dir, log=log, dry_run=args.dry_run, force=args.force)
        log(f"prune-unfinished: {n} games, {'would free' if args.dry_run else 'freed'} {human(b)}")
        return 0

    if out_dir is None:
        ap.error("--out DIR is required to archive (only --status and the prune modes work without).")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError, AttributeError):
            pass

    if args.loop is None:
        n, b = cmd_archive(run_dir, out_dir, log=log, dry_run=args.dry_run, keep=args.keep,
                           min_free_gb=args.min_free_gb)
        log(f"archive: {n} games, {'would free' if args.dry_run else 'freed'} {human(b)}")
        return 0

    log(f"harvest daemon: {run_dir} -> {out_dir}, every {args.loop:g}s "
        f"(keep {args.keep}, min free {args.min_free_gb:g} GB). PID {os.getpid()}.")
    idle = total_games = total_bytes = 0
    while not _STOP:
        try:
            n, b = cmd_archive(run_dir, out_dir, log=log, dry_run=args.dry_run, keep=args.keep,
                               min_free_gb=args.min_free_gb)
        except Exception as e:      # noqa: BLE001 - a bad pass must not end a multi-day daemon
            log(f"pass FAILED ({e!r}); continuing.")
            n = b = 0
        if n:
            total_games += n
            total_bytes += b
            idle = 0
            log(f"pass: {n} games, freed {human(b)}  (total {total_games} games, "
                f"{human(total_bytes)}; {human(free_bytes(out_dir))} free)")
        else:
            idle += 1
            if idle % 30 == 0:      # a heartbeat, not a log line every minute
                log(f"idle: nothing to harvest ({total_games} games so far, "
                    f"{human(free_bytes(out_dir))} free)")
        # Sleep in slices so a SIGTERM lands within a second rather than at the end of the interval.
        deadline = time.monotonic() + args.loop
        while not _STOP and time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    log(f"harvest daemon stopped: {total_games} games, {human(total_bytes)} freed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
