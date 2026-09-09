"""
The three processes -- LOAD, RUN, TRAIN -- as plain functions.

Kept free of any REPL concern so each is unit-testable without a TTY; ``shell/repl.py`` is only
argument parsing and printing on top of these.
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import config
from models.artifacts import ModelArtifacts, list_models, model_root, set_active_model
from models.manifest import ARCH_KEYS
from shell.heavy import ensure_tf

# The heads GameController requires to play a game, from config so the shell, the controller
# and the tests cannot drift. Checked at LOAD so a missing head fails immediately instead of
# mid-rollout, hours into an eval.
REQUIRED_HEADS = config.REQUIRED_HEADS

# ARCH_KEYS is imported from models.manifest, not restated here. It used to be a second copy,
# and the copies drifted: workstream 10a added LOCAL_ATTENTION_HEADS and LOCAL_ATTENTION_WINDOW
# to the manifest's list, per correction K, but not to this one -- so the silent local/global
# reload that correction exists to prevent was being recorded in every manifest and checked in
# none. The point of the key is the check, so there is one list.


class ShellError(Exception):
    """A user-facing failure: printed as one line, never a traceback."""


# --------------------------------------------------------------------------- #
# --- LOAD                                                                  -- #
# --------------------------------------------------------------------------- #

def load_model(session, name, *, dial_file=None, force=False, echo=print) -> None:
    """Swap the resident model to ``name``, unloading whatever is currently loaded.

    Validation runs cheapest-first and fails loudly, because every alternative to a clear message
    here is a confusing failure much later: a missing head becomes a mid-eval RuntimeError, and an
    arch or vocab mismatch becomes a Keras shape traceback inside ``load_weights``.
    """
    try:
        root = Path(model_root(name))
    except ValueError as e:
        raise ShellError(str(e)) from None

    if not root.is_dir():
        raise ShellError(f"no model named {name!r} at {root}. Available: "
                         f"{', '.join(list_models()) or '(none)'}")

    manifest = read_manifest(root)
    if not manifest:
        echo(f"  warning: {root} has no manifest.json (pre-manifest model); skipping arch and "
             f"vocab checks. Run 'adopt {name}' to record one.")
    else:
        _check_arch(manifest, name, force=force, echo=echo)

    missing = [k for k in REQUIRED_HEADS if not ModelArtifacts.for_key(k, root).exists()]
    if missing and not force:
        raise ShellError(f"{name} is missing head(s) required to play a game: "
                         f"{', '.join(missing)}. Use --force to load anyway (run will fail).")

    ensure_tf()
    from encoder.encoder import Encoder
    from simulation.game_simulator import GameSimulator

    # A per-model vocab snapshot pins the token ids these weights were trained against. Without
    # one we fall back to the shared encoder/vocabs/, which the most recent train may have moved.
    vocab_dir = root / "vocabs"
    encoder = Encoder(vocab_dir=vocab_dir) if vocab_dir.is_dir() else None
    if encoder is None and manifest.get("vocabs"):
        _check_vocabs(manifest, name, force=force, echo=echo)

    prev = session.unload()
    if prev:
        echo(f"  unloaded {prev}")

    try:
        sim = GameSimulator.load(artifacts_root=str(root), encoder=encoder)
    except Exception as e:
        raise ShellError(f"failed to load {name}: {type(e).__name__}: {e}") from None

    session.sim = sim
    session.model = name
    session.artifacts_root = str(root)
    session.manifest = manifest
    session.heads = ("event_time",) + tuple(sorted(sim.heads))
    session.loaded_at = _dt.datetime.now().strftime("%H:%M:%S")
    set_active_model(name)

    echo(f"  loaded {len(session.heads)} heads from {root}"
         + (f" (vocabs: {vocab_dir})" if encoder is not None else ""))

    try:
        session.holdout_ids, session.holdout_source = session.resolve_holdout()
        echo(f"  holdout: {len(session.holdout_ids)} games  (from {session.holdout_source})")
    except FileNotFoundError as e:
        session.holdout_ids, session.holdout_source = [], ""
        echo(f"  warning: {e}")

    if dial_file:
        n = apply_dial_file(dial_file)
        echo(f"  dials: applied {n} override(s) from {dial_file}")
    elif session.changed_dials:
        names = ", ".join(sorted(session.changed_dials))
        echo(f"  dials: {len(session.changed_dials)} override(s) carried over ({names})")

    if manifest.get("recommended_dials"):
        echo("  note: this model records recommended dials; 'dials --recommended' to see them.")


def _check_arch(manifest, name, *, force, echo) -> None:
    arch = manifest.get("arch") or {}
    diffs = [f"{k}: trained at {arch[k]}, config.py now says {getattr(config, k)}"
             for k in ARCH_KEYS if k in arch and arch[k] != getattr(config, k, None)]
    if not diffs:
        return
    msg = (f"architecture mismatch for {name} -- loading would fail on weight shapes:\n    "
           + "\n    ".join(diffs)
           + "\n  The graph is rebuilt from config.py, so these must match the trained weights.")
    if force:
        echo(f"  warning (forced): {msg}")
    else:
        raise ShellError(msg)


def _check_vocabs(manifest, name, *, force, echo) -> None:
    from encoder.encoder import Encoder
    from models.manifest import vocab_fingerprint
    live = vocab_fingerprint(Encoder())
    drift = [f"{k}: weights expect {v.get('size')} tokens, encoder/vocabs/ now has "
             f"{live.get(k, {}).get('size')}"
             for k, v in (manifest.get("vocabs") or {}).items()
             if k in live and live[k].get("size") != v.get("size")]
    if not drift:
        return
    msg = (f"vocab drift for {name} -- its embedding tables are sized for the old vocab:\n    "
           + "\n    ".join(drift)
           + f"\n  A later train rewrote the shared encoder/vocabs/. Restore "
             f"artifacts/{name}/vocabs/, or load a model matching the current vocab.")
    if force:
        echo(f"  warning (forced): {msg}")
    else:
        raise ShellError(msg)


def read_manifest(root) -> dict:
    """The model's manifest, or an empty dict when absent so pre-manifest models still load."""
    p = Path(root) / "manifest.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def apply_dial_file(path) -> int:
    """Apply a dial package (a JSON object of DIAL -> value). Returns how many were set.

    The parsing lives in ``config`` so the eval CLI can hand the same package to child processes
    without importing the shell (which pulls TensorFlow); this is the shell-error wrapper.
    """
    try:
        return len(config.apply_dial_file(path))
    except ValueError as e:
        raise ShellError(str(e)) from None


# --------------------------------------------------------------------------- #
# --- RUN                                                                   -- #
# --------------------------------------------------------------------------- #

def run_eval(session, name, *, games=None, sims=None, concurrency=None, seed=0, subset=None,
             report_only=False, report_every=None, echo=print) -> dict:
    """Evaluate the resident model, writing to ``results/<model>/<run name>/``.

    Uses the loaded simulator and the cached cleaned frame, so a second run costs a rollout rather
    than a model rebuild plus a full re-read of every season CSV.
    """
    if not session.loaded:
        raise ShellError("no model loaded. Run 'load <name>' first; 'models' lists what is here.")
    if not session.holdout_ids:
        raise ShellError(f"no holdout games resolved for {session.model} "
                         f"(source tried: {session.holdout_source or 'none'}).")

    ensure_tf()
    from config import EVAL_GAMES_PER_BATCH, ROLLOUT_BATCH_SIZE, STAGE_SIMS
    from reporting.eval_report import pin_run_holdout, resolve_results_run_dir, subset_holdout
    from simulation.stage_eval import evaluate_stage

    run_dir = resolve_results_run_dir(
        session.model, name=name,
        holdout_total=len(subset_holdout(session.holdout_ids, subset)))
    holdout_ids = pin_run_holdout(run_dir, session.holdout_ids, subset=subset)
    df = cleaned_frame(session, echo=echo)

    n_sims = sims or STAGE_SIMS
    batch_size = concurrency or ROLLOUT_BATCH_SIZE
    echo(f"  {session.model} -> {run_dir}")
    subset_note = (f" (subset of {len(session.holdout_ids)}, pinned)"
                   if len(holdout_ids) != len(session.holdout_ids) else "")
    echo(f"  {len(holdout_ids)} holdout games{subset_note}, {n_sims} sims each, "
         f"concurrency {batch_size}" + (f", max {games} new" if games else ""))
    if session.changed_dials:
        echo("  dials: " + ", ".join(f"{k} {a}->{b}"
                                     for k, (a, b) in sorted(session.changed_dials.items())))

    report = evaluate_stage(
        session.model, sim=session.sim, df=df, run_label=name or run_dir.name,
        holdout_ids=holdout_ids, n_sims=n_sims,
        max_new=0 if report_only else games, report_every=report_every,
        data_dir=session.data_dir, processed_dir=session.processed_dir,
        artifacts_root=session.artifacts_root, results_run_dir=run_dir,
        seed0=seed, batch_size=batch_size, games_per_batch=EVAL_GAMES_PER_BATCH,
    )
    session.last_run_dir = Path(report["run_dir"])
    echo(f"  {report['done']}/{report['total']} games -> {report['run_dir']}")
    return report


def run_eval_pooled(session, name, *, procs, sims=None, concurrency=None, seed=0, subset=None,
                    echo=print):
    """Evaluate across N child processes, then merge one report in this process.

    The parent supervises only -- it does not take a slice. It has to stay responsive to render the
    merged progress line and to handle Ctrl-C, and a parent blocked inside a rollout can do
    neither; cleanly interrupting both its own rollout and N children is a lot of machinery to buy
    one shard's throughput. The resident model does sit idle holding its VRAM, which is why that is
    subtracted from the children's budget below (and said out loud, since it costs a child).

    The run dir is resolved ONCE here and its concrete name is passed to every child, so the
    auto-``eval-NNN`` race that bare ``--shard`` has to guard against cannot happen from the shell.
    The merge runs in-process: TensorFlow is already loaded and the cached frame skips a full
    re-read of every season CSV.
    """
    from eval_pool import (assert_one_tuning, autosize_procs, finished_games, run_waves,
                           shard_commands)

    if not session.loaded:
        raise ShellError("no model loaded. Run 'load <name>' first; 'models' lists what is here.")
    if not session.holdout_ids:
        raise ShellError(f"no holdout games resolved for {session.model} "
                         f"(source tried: {session.holdout_source or 'none'}).")

    from reporting.eval_report import pin_run_holdout, resolve_results_run_dir, subset_holdout

    # Each child is a fresh `evaluate.py`, which reads the holdout + data paths from the run
    # state. The shell deliberately does NOT need that file (Session.resolve_holdout falls back to
    # the holdout manifest and to previous runs), so a pooled run can be asked for on a machine
    # that cannot serve it. Say so here rather than letting N children each die with a stack trace.
    state_path = Path(config.FULL_RUN_STATE_PATH)
    if not state_path.is_file():
        raise ShellError(
            f"--procs needs {state_path} (each child process is a fresh 'evaluate.py', which reads "
            f"the holdout and data paths from it). This shell resolved its holdout from "
            f"{session.holdout_source or 'another source'} instead. "
            f"Run without --procs to evaluate in this process, or create the state on "
            f"this machine first.")

    # The equality check is against the FULL holdout on both sides: the children re-derive theirs
    # from the state file, and a subset is applied identically on top (they read the same pinned
    # holdout.json this call is about to write).
    run_dir = resolve_results_run_dir(
        session.model, name=name,
        holdout_total=len(subset_holdout(session.holdout_ids, subset)))

    state_holdout = json.loads(state_path.read_text(encoding="utf-8")).get(
        "holdout_game_ids") or []
    if sorted(int(g) for g in state_holdout) != sorted(session.holdout_ids):
        raise ShellError(
            f"the holdout in {state_path} ({len(state_holdout)} games) is not the one this shell "
            f"resolved ({len(session.holdout_ids)} games, from "
            f"{session.holdout_source or 'unknown'}). The children would evaluate a different set "
            f"of games than this report claims to cover. Run without --procs.")

    total = len(pin_run_holdout(run_dir, session.holdout_ids, subset=subset))
    if total != len(session.holdout_ids):
        echo(f"  holdout subset: {total} of {len(session.holdout_ids)} games, pinned in "
             f"{run_dir.name}/holdout.json")

    # Every child re-reads config.py from disk, so a shell that tuned dials at runtime MUST hand
    # them over or the run silently mixes two tunings into one report.
    dials_path = run_dir / "dials.json"
    config.write_dial_file(dials_path)

    echo(f"  {session.model} -> {run_dir}")
    if session.changed_dials:
        echo("  dials: " + ", ".join(f"{k} {a}->{b}"
                                     for k, (a, b) in sorted(session.changed_dials.items())))
    echo(f"  dial package -> {dials_path}")
    echo("  resident model holds its VRAM while the children run; 'unload' frees host RAM but "
         "VRAM stays in TF's allocator pool, so it may not buy you a child.")

    reserved = config.EVAL_PROC_VRAM_GB if session.loaded else 0.0

    def build(wave, remaining):
        n, why = autosize_procs(holdout_games=remaining, requested=procs,
                                reserved_vram_gb=reserved)
        if wave == 1:
            echo(f"  {total} holdout games ({finished_games(run_dir)} already done), {why}")
        if n <= 1 and wave == 1:
            return []
        return shard_commands(model=session.model, run=run_dir.name, n=n,
                              dials_path=dials_path, state_path=state_path, run_dir=run_dir,
                              sims=sims, concurrency=concurrency, seed=seed, subset=subset,
                              tag="" if wave == 1 else f"-w{wave}")

    shards = run_waves(build=build, run_dir=run_dir, total_games=total, echo=echo)
    if not shards and finished_games(run_dir) < total:
        echo("  pool of 1 -- running in this process instead.")
        return run_eval(session, name, sims=sims, concurrency=concurrency, seed=seed,
                        subset=subset, echo=echo)

    assert_one_tuning(run_dir, echo=echo)

    # Merge in-process: TF is loaded and the cached frame skips re-reading every season CSV.
    echo("  merging the report over every finished game ...")
    report = run_eval(session, run_dir.name, sims=sims, concurrency=concurrency, seed=seed,
                      subset=subset, report_only=True, echo=echo)
    if report["done"] < report["total"]:
        echo(f"  NOTE: {report['done']}/{report['total']} games -- the report covers finished "
             f"games only. Re-run to continue.")
    return report


def cleaned_frame(session, *, echo=print):
    """The parsed cleaned play-by-play, cached on the session across runs."""
    if session.cleaned_df is not None and session.cleaned_df_key == session.data_dir:
        return session.cleaned_df
    from data_loading import load_all_cleaned
    echo(f"  reading cleaned data from {session.data_dir} (cached for later runs) ...")
    session.cleaned_df = load_all_cleaned(session.data_dir, parse_rosters=True)
    session.cleaned_df_key = session.data_dir
    return session.cleaned_df


# --------------------------------------------------------------------------- #
# --- TRAIN                                                                 -- #
# --------------------------------------------------------------------------- #

RUNS_DIR = Path("./training/runs")


def train_command(name, *, batch_size, epochs, clean=False, rebuild_vocabs=False) -> list:
    """The ``train.py`` command line for a full train of ``name``."""
    cmd = [sys.executable, "train.py", "--full", "--name", name, "--epochs", str(epochs)]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if clean:
        cmd.append("--clean")
    if rebuild_vocabs:
        cmd.append("--rebuild-vocabs")
    cmd += ["--state", str(RUNS_DIR / f"{name}.json")]
    return cmd


def launch_train(session, name, *, batch_size, epochs, clean, rebuild_vocabs, go,
                 echo=print) -> None:
    """Print the train command; with ``go``, launch it detached.

    Training never runs in this process. ``models.pipeline`` calls
    ``keras.backend.clear_session()`` between heads, which would destroy the resident inference
    model -- the exact thing the shell exists to keep alive. A full train is also a multi-day GPU
    job, and would contend with inference for the same VRAM.
    """
    try:
        root = model_root(name)
    except ValueError as e:
        raise ShellError(str(e)) from None
    if Path(root).is_dir():
        raise ShellError(f"{name} already exists at {root}. A retrain takes a NEW name so its "
                         f"weights and results never mix with the old ones.")

    cmd = train_command(name, batch_size=batch_size, epochs=epochs, clean=clean,
                        rebuild_vocabs=rebuild_vocabs)
    printable = " ".join(cmd[1:])
    if not go:
        echo(f"  would run:  python {printable}")
        echo("  Training is a multi-day GPU job and does not run inside the shell.")
        echo("  Launch it yourself on the GPU box, or re-issue with --go to start it detached.")
        return

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log = RUNS_DIR / f"{name}.log"
    echo(f"  launching detached: python {printable}")
    echo(f"  log: {log}")
    with open(log, "ab") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
    echo(f"  pid {proc.pid}. 'train --status {name}' for progress, "
         f"'train --follow {name}' to tail the log.")


def train_status(name=None) -> list:
    """Per-train state read straight off disk -- no TensorFlow import."""
    if not RUNS_DIR.is_dir():
        return []
    files = [RUNS_DIR / f"{name}.json"] if name else sorted(RUNS_DIR.glob("*.json"))
    out = []
    for f in files:
        if not f.is_file():
            continue
        try:
            st = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        st["_name"] = f.stem
        st["_log"] = str(RUNS_DIR / f"{f.stem}.log")
        out.append(st)
    return out
