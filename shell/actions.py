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
from shell.heavy import ensure_tf

# The heads GameController requires to play a game. Checked at LOAD so a missing head fails
# immediately instead of mid-rollout, hours into an eval.
REQUIRED_HEADS = ("player", "substitution", "shot_type", "shot_result",
                  "assist_type", "turnover_type", "foul_type", "rebound_type")

# Capacity dims baked into the weights: from_artifacts rebuilds the graph from these, so a
# mismatch surfaces as an opaque Keras shape error deep inside load_weights.
ARCH_KEYS = ("MODEL_DIM", "NUM_LAYERS", "NUM_HEADS", "FF_DIM", "ROSTER_SAB_LAYERS",
             "MAX_SEQUENCE_LENGTH", "ROSTER_SIZE")


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
    """Apply a dial package (a JSON object of DIAL -> value). Returns how many were set."""
    p = Path(path)
    if not p.is_file():
        raise ShellError(f"no dial file at {p}")
    try:
        values = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ShellError(f"{p} is not valid JSON: {e}") from None
    if not isinstance(values, dict):
        raise ShellError(f"{p} must contain a JSON object of DIAL -> value")
    try:
        return len(config.apply_dials(values))
    except (KeyError, TypeError, ValueError) as e:
        raise ShellError(f"{p}: {e}") from None


# --------------------------------------------------------------------------- #
# --- RUN                                                                   -- #
# --------------------------------------------------------------------------- #

def run_eval(session, name, *, games=None, sims=None, concurrency=None, seed=0,
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
    from reporting.eval_report import resolve_results_run_dir
    from simulation.stage_eval import evaluate_stage

    run_dir = resolve_results_run_dir(session.model, name=name,
                                      holdout_total=len(session.holdout_ids))
    df = cleaned_frame(session, echo=echo)

    n_sims = sims or STAGE_SIMS
    batch_size = concurrency or ROLLOUT_BATCH_SIZE
    echo(f"  {session.model} -> {run_dir}")
    echo(f"  {len(session.holdout_ids)} holdout games, {n_sims} sims each, "
         f"concurrency {batch_size}" + (f", max {games} new" if games else ""))
    if session.changed_dials:
        echo("  dials: " + ", ".join(f"{k} {a}->{b}"
                                     for k, (a, b) in sorted(session.changed_dials.items())))

    report = evaluate_stage(
        session.model, sim=session.sim, df=df, run_label=name or run_dir.name,
        holdout_ids=session.holdout_ids, n_sims=n_sims,
        max_new=0 if report_only else games, report_every=report_every,
        data_dir=session.data_dir, processed_dir=session.processed_dir,
        artifacts_root=session.artifacts_root, results_run_dir=run_dir,
        seed0=seed, batch_size=batch_size, games_per_batch=EVAL_GAMES_PER_BATCH,
    )
    session.last_run_dir = Path(report["run_dir"])
    echo(f"  {report['done']}/{report['total']} games -> {report['run_dir']}")
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
