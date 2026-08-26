"""
The ``cviq>`` prompt.

``cmd.Cmd`` handles dispatch, readline history and tab completion; each command owns an
``argparse`` parser for its flags. Those parsers raise instead of calling ``sys.exit``, because a
mistyped flag must not kill a process holding a loaded model.

Everything here is presentation. The actual work lives in :mod:`shell.actions`, so it can be
tested by calling functions rather than by driving a terminal.
"""
from __future__ import annotations

import argparse
import cmd
import json
import shlex
from pathlib import Path

import config
from models.artifacts import list_models, model_root
from shell.actions import (ShellError, apply_dial_file, launch_train, load_model, run_eval,
                           run_eval_pooled, train_status)
from shell.session import Session

BANNER = r"""
  cviq -- CourtVisionIQ

  load <name>          bring a model's weights into memory (unloads the current one)
  run [<name>]         evaluate the loaded model  ->  results/<model>/<name>/
  set <DIAL> <value>   change a rollout dial; the next run uses it
  train <name>         print (or --go launch) a training run
  status               what is loaded, which dials moved, where the last run went
  help [<command>]     details.  quit / Ctrl-D to exit
"""


def _split(arg):
    r"""Tokenize a command's arguments, Windows-safely.

    ``posix=True`` treats backslash as an escape, which silently eats the separators in
    ``C:\Users\...`` -- so paths typed at the prompt arrive mangled. ``posix=False`` keeps them,
    but leaves surrounding quotes attached, so those are stripped by hand.
    """
    out = []
    for tok in shlex.split(arg, posix=False):
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
            tok = tok[1:-1]
        out.append(tok)
    return out


class ArgError(Exception):
    """Raised instead of SystemExit when a command's flags do not parse."""


class _Parser(argparse.ArgumentParser):
    """An ArgumentParser that never exits the process."""

    def error(self, message):
        raise ArgError(message)

    def exit(self, status=0, message=None):
        raise ArgError(message or "")


def _parser(prog, desc=None) -> _Parser:
    return _Parser(prog=prog, description=desc, add_help=False)


P_LOAD = _parser("load")
P_LOAD.add_argument("name", nargs="?")
P_LOAD.add_argument("--dials", metavar="FILE", help="apply a dial package after loading")
P_LOAD.add_argument("--force", action="store_true",
                    help="load despite an arch/vocab mismatch or a missing head")

P_RUN = _parser("run")
P_RUN.add_argument("name", nargs="?", help="run name; omit for an auto eval-NNN")
P_RUN.add_argument("--games", type=int, help="cap NEW games simulated this call")
P_RUN.add_argument("--holdout", type=int, metavar="N",
                   help="evaluate an N-game subset of the holdout (every total//N-th game); "
                        "pinned to the run dir, unlike --games it changes the run's denominator")
P_RUN.add_argument("--sims", "--monte-carlo", dest="sims", type=int,
                   help="Monte-Carlo sims per game (default STAGE_SIMS)")
P_RUN.add_argument("--concurrency", type=int,
                   help="rollout cohort width -- the VRAM knob, independent of --sims")
P_RUN.add_argument("--seed", type=int, default=0)
P_RUN.add_argument("--report-only", action="store_true",
                   help="rebuild the report from finished games; simulate nothing")
P_RUN.add_argument("--report-every", type=int, metavar="N",
                   help="write an intermediate report every N finished games")
P_RUN.add_argument("--procs", metavar="N",
                   help="split the holdout across N eval processes ('auto' sizes from cores + "
                        "free VRAM); omit or 1 for the in-process path")

P_DIALS = _parser("dials")
P_DIALS.add_argument("--changed", action="store_true", help="only dials that moved")
P_DIALS.add_argument("--save", metavar="FILE", help="write current values as a dial package")
P_DIALS.add_argument("--recommended", action="store_true",
                     help="show the dials recorded by the loaded model's train")

P_TRAIN = _parser("train")
P_TRAIN.add_argument("name", nargs="?")
P_TRAIN.add_argument("--batch-size", type=int)
P_TRAIN.add_argument("--epochs", type=int, default=50)
P_TRAIN.add_argument("--clean", action="store_true")
P_TRAIN.add_argument("--rebuild-vocabs", action="store_true")
P_TRAIN.add_argument("--go", action="store_true", help="actually launch it, detached")
P_TRAIN.add_argument("--status", action="store_true")
P_TRAIN.add_argument("--follow", action="store_true")
P_TRAIN.add_argument("--list", action="store_true")


class CviqShell(cmd.Cmd):
    """The REPL. One :class:`~shell.session.Session` for the life of the process."""

    prompt = "cviq> "
    intro = BANNER

    def __init__(self, session=None, **kw):
        super().__init__(**kw)
        self.session = session or Session()

    # ------------------------------------------------------------ dispatch
    def onecmd(self, line):
        """Run one command, containing every failure so the shell survives it.

        A loaded model is expensive to rebuild, so nothing short of an explicit quit should end
        the process: not a bad flag, not a traceback in an action, not Ctrl-C during a run.
        ``FullRun._require`` raises SystemExit, hence catching that too.
        """
        try:
            return super().onecmd(line)
        except (ArgError, ShellError) as e:
            print(f"  {e}")
        except KeyboardInterrupt:
            print("\n  interrupted. The model is still loaded; finished games are cached.")
        except SystemExit as e:
            if e.code not in (0, None):
                print(f"  {e}")
        except Exception as e:  # noqa: BLE001 - deliberately broad; see docstring
            print(f"  {type(e).__name__}: {e}")
        return False

    def emptyline(self):
        return False

    def default(self, line):
        print(f"  unknown command: {line.split()[0]!r}. 'help' lists commands.")

    # ------------------------------------------------------------ LOAD
    def do_load(self, arg):
        """load <name> [--dials FILE] [--force]

        Bring a model's weights into memory, unloading the current one first. With no name,
        loads config.DEFAULT_MODEL. Dial overrides are deliberately NOT reset by a load, so the
        same tuning can be compared across two models.
        """
        a = P_LOAD.parse_args(_split(arg))
        name = a.name or config.DEFAULT_MODEL
        load_model(self.session, name, dial_file=a.dials, force=a.force)

    def complete_load(self, text, *_):
        return [n for n in list_models() if n.startswith(text)]

    def do_unload(self, arg):
        """unload -- release the resident model.

        Frees host RAM. VRAM stays in TensorFlow's allocator pool, which the next load reuses
        rather than allocating on top of, so repeated swaps plateau instead of climbing.
        """
        was = self.session.unload()
        print(f"  unloaded {was}" if was else "  nothing loaded")

    def do_models(self, arg):
        """models -- every loadable model under ./artifacts."""
        names = list_models()
        if not names:
            print("  no models found under ./artifacts")
            return
        active = self.session.model
        for n in names:
            root = Path(model_root(n))
            heads = sum(1 for d in root.iterdir() if d.is_dir() and d.name != "vocabs")
            marks = []
            if (root / "manifest.json").is_file():
                marks.append("manifest")
            if (root / "vocabs").is_dir():
                marks.append("vocabs")
            flag = "*" if n == active else " "
            print(f"  {flag} {n:24} {heads:2} heads  {'  '.join(marks) or '(no manifest)'}")
        if active:
            print("  (* = loaded)")

    # ------------------------------------------------------------ RUN
    def do_run(self, arg):
        """run [<name>] [--games N] [--sims N] [--concurrency N] [--procs N] [--seed N]
               [--report-only]

        Evaluate the loaded model into results/<model>/<name>/. A named run is stable: running it
        again resumes it rather than starting over, since finished games are cached per game.

        --sims and --concurrency are independent on purpose: sims is how many Monte-Carlo
        rollouts each game gets, concurrency is how many run at once and is what bounds VRAM.

        --procs is the third, separate knob: concurrency fills the GPU inside ONE process, but
        that process is GIL-bound to about one core, so --procs is what uses the rest of the box.
        The run dir is resolved once here and handed to every child, so an auto-named run is safe.

        --holdout N narrows the run to N of the holdout's games (every total//N-th, so the sample
        spans the season) and pins them to the run dir. Trading games for sims -- 'run s100g20
        --holdout 20 --sims 100' -- costs about what the full holdout at 21 sims costs, but with
        ~2.2x less Monte-Carlo error on each game.
        """
        a = P_RUN.parse_args(_split(arg))
        if a.procs and str(a.procs) != "1":
            if a.report_only:
                raise ArgError("--report-only simulates nothing; drop --procs.")
            if a.games:
                raise ArgError("--games is the single-process interrupt knob and does not combine "
                               "with --procs. Drop one.")
            run_eval_pooled(self.session, a.name, procs=a.procs, sims=a.sims,
                            concurrency=a.concurrency, seed=a.seed, subset=a.holdout)
            return
        run_eval(self.session, a.name, games=a.games, sims=a.sims, concurrency=a.concurrency,
                 seed=a.seed, report_only=a.report_only, report_every=a.report_every,
                 subset=a.holdout)

    def do_runs(self, arg):
        """runs [<model>] -- evaluation runs on disk under ./results."""
        model = arg.strip() or self.session.model
        root = Path("./results") / model if model else Path("./results")
        if not root.is_dir():
            print(f"  no runs at {root}")
            return
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            n = len(list((d / "games").glob("*/record.json"))) if (d / "games").is_dir() else 0
            has = "report.html" if (d / "report.html").is_file() else "-"
            print(f"  {d.name:28} {n:4} games  {has}")

    # ------------------------------------------------------------ DIALS
    def do_set(self, arg):
        """set <DIAL> <value> -- change a rollout dial for the next run.

        Values coerce to the dial's current type; the three dict dials (SHOT_RESULT_BIAS,
        EVENT_BIAS, TYPE_BIAS) take JSON. No reload is needed -- the rollout reads dials at call
        time, and each run records the values it used in its report.
        """
        parts = arg.strip().split(None, 1)
        if len(parts) < 2:
            raise ArgError("usage: set <DIAL> <value>   ('dials' lists them)")
        # Value taken as the raw remainder, not tokenized: the dict dials are JSON and any
        # splitting/unquoting would corrupt them.
        name, raw = parts[0], parts[1].strip()
        try:
            value = config.set_dial(name, raw)
        except KeyError:
            close = [k for k in config._TUNING_KEYS if k.startswith(name[:4].upper())]
            raise ShellError(f"unknown dial {name!r}."
                             + (f" Did you mean: {', '.join(close)}?" if close else
                                " 'dials' lists them.")) from None
        except (TypeError, ValueError) as e:
            raise ShellError(f"{name}: {e}") from None
        print(f"  {name} = {value}")

    def complete_set(self, text, *_):
        return [k for k in config._TUNING_KEYS if k.startswith(text.upper())]

    def do_unset(self, arg):
        """unset <DIAL> -- restore one dial to its startup value."""
        name = arg.strip()
        if name not in self.session.baseline:
            raise ShellError(f"unknown dial {name!r}. 'dials' lists them.")
        config.set_dial(name, self.session.baseline[name])
        print(f"  {name} = {getattr(config, name)}  (restored)")

    complete_unset = complete_set

    def do_reset(self, arg):
        """reset -- restore every dial to its startup value."""
        n = self.session.reset_dials()
        print(f"  restored {n} dial(s)" if n else "  no dials had changed")

    def do_dials(self, arg):
        """dials [--changed] [--save FILE] [--recommended] -- show or export the rollout dials."""
        a = P_DIALS.parse_args(_split(arg))
        if a.recommended:
            rec = self.session.manifest.get("recommended_dials")
            if not rec:
                print("  the loaded model records no recommended dials")
                return
            for k, v in rec.items():
                cur = getattr(config, k, None)
                print(f"  {k:26} {v!r}" + ("" if cur == v else f"   (now {cur!r})"))
            return
        if a.save:
            Path(a.save).write_text(json.dumps(config.get_dials(), indent=2), encoding="utf-8")
            print(f"  wrote {len(config._TUNING_KEYS)} dials -> {a.save}")
            return
        changed = self.session.changed_dials
        if a.changed:
            if not changed:
                print("  no dials changed from startup")
            for k, (before, after) in sorted(changed.items()):
                print(f"  {k:26} {before!r} -> {after!r}")
            return
        for k in config._TUNING_KEYS:
            mark = " *" if k in changed else "  "
            print(f" {mark} {k:26} {getattr(config, k)!r}")
        if changed:
            print(f"  (* = changed from startup; {len(changed)} of {len(config._TUNING_KEYS)})")

    def do_dialfile(self, arg):
        """dialfile <FILE> -- apply a dial package written by 'dials --save'."""
        if not arg.strip():
            raise ArgError("usage: dialfile <FILE>")
        print(f"  applied {apply_dial_file(arg.strip())} dial(s) from {arg.strip()}")

    # ------------------------------------------------------------ TRAIN
    def do_train(self, arg):
        """train <name> [--batch-size N] [--epochs N] [--go] | --status|--follow|--list [<name>]

        Prints the command by default and does not run it: a full train is a multi-day GPU job,
        and training in this process would clear the Keras session out from under the loaded
        model. --go launches it detached instead.
        """
        a = P_TRAIN.parse_args(_split(arg))
        if a.list or (a.status and not a.name):
            rows = train_status()
            if not rows:
                print("  no training runs under ./training/runs")
            for st in rows:
                print(f"  {st['_name']:24} {st.get('status', '?'):10} "
                      f"{len(st.get('trained_models', []))} heads trained")
            return
        if not a.name:
            raise ArgError("usage: train <name> [--go]   (or --list)")
        if a.status:
            rows = train_status(a.name)
            if not rows:
                print(f"  no training state for {a.name}")
                return
            print(json.dumps(rows[0], indent=2, default=str))
            return
        if a.follow:
            log = Path("./training/runs") / f"{a.name}.log"
            if not log.is_file():
                print(f"  no log at {log}")
                return
            print(log.read_text(encoding="utf-8", errors="replace")[-4000:])
            return
        launch_train(self.session, a.name, batch_size=a.batch_size, epochs=a.epochs,
                     clean=a.clean, rebuild_vocabs=a.rebuild_vocabs, go=a.go)

    def do_adopt(self, arg):
        """adopt <name> -- back-fill a manifest + vocab snapshot for existing weights.

        No retrain. Pins the model's vocabs so a later train cannot invalidate its embedding
        tables, and records the arch so a mismatched config.py becomes a readable refusal.
        """
        name = arg.strip() or self.session.model
        if not name:
            raise ArgError("usage: adopt <name>")
        from models.manifest import adopt
        adopt(name)
        if self.session.model == name:
            from shell.actions import read_manifest
            self.session.manifest = read_manifest(model_root(name))

    complete_adopt = complete_load

    # ------------------------------------------------------------ STATUS
    def do_status(self, arg):
        """status -- what is loaded, which dials moved, where the last run went."""
        s = self.session
        if not s.loaded:
            print(f"  model      (none loaded)   'load {config.DEFAULT_MODEL}' to start")
        else:
            man = s.manifest
            desc = "no manifest -- run 'adopt'" if not man else (
                ("backfilled" if man.get("backfilled") else "trained")
                + f" {str(man.get('created_at', ''))[:10]}"
                + (f" git {man['git_commit']}" if man.get("git_commit") else ""))
            print(f"  model      {s.model:20} ({s.artifacts_root}, loaded {s.loaded_at}, "
                  f"{len(s.heads)} heads)")
            print(f"  manifest   {desc}")
            print(f"  holdout    {len(s.holdout_ids)} games   (from {s.holdout_source})")
        changed = s.changed_dials
        if changed:
            print(f"  dials      {len(changed)} changed   " + " | ".join(
                f"{k} {a}->{b}" for k, (a, b) in sorted(changed.items())))
        else:
            print("  dials      all at startup values")
        print(f"  last run   {s.last_run_dir or '(none this session)'}")
        print(f"  data       {s.data_dir}"
              + ("  (cleaned frame cached)" if s.cleaned_df is not None else ""))
        try:
            import psutil
            rss = psutil.Process().memory_info().rss / (1 << 30)
            print(f"  memory     {rss:.2f} GB resident")
        except Exception:
            pass

    # ------------------------------------------------------------ misc
    def do_quit(self, arg):
        """quit -- exit the shell."""
        return True

    do_exit = do_quit

    def do_EOF(self, arg):
        print()
        return True
