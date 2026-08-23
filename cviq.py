"""
CourtVisionIQ interactive shell: TRAIN / LOAD / RUN.

    python cviq.py

Three distinct processes over one resident model:

    TRAIN   produce a new named set of weights (launched, never run in-process)
    LOAD    bring an existing model's weights into memory, unloading the previous one
    RUN     evaluate whatever is loaded, into results/<model>/<run name>/

The point of a shell rather than three scripts is that separate ``python x.py`` invocations
cannot share a loaded model. Here the eleven heads stay resident, so changing a dial and
re-predicting costs a rollout instead of a full rebuild.

Kept deliberately free of heavy imports so the prompt appears immediately; TensorFlow is not
imported until the first command that actually needs a model (see shell/heavy.py).
"""
from __future__ import annotations

import os
import sys

# Must precede any TensorFlow import: without it TF grabs the whole GPU up front, which matters
# here because the shell holds a model resident across many runs. Mirrors train.py / evaluate.py.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="cviq", description=__doc__.strip().splitlines()[0])
    ap.add_argument("--preload", metavar="MODEL", nargs="?", const=True,
                    help="load a model before the first prompt (default: config.DEFAULT_MODEL)")
    ap.add_argument("-c", "--command", action="append", metavar="CMD",
                    help="run a command and exit; repeatable, e.g. -c 'load v1.0' -c status")
    args = ap.parse_args(argv)

    from shell.repl import CviqShell

    shell = CviqShell()

    if args.preload:
        import config
        name = config.DEFAULT_MODEL if args.preload is True else args.preload
        shell.onecmd(f"load {name}")

    if args.command:
        for c in args.command:
            print(f"cviq> {c}")
            if shell.onecmd(c):
                break
        return 0

    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        # Ctrl-C at an idle prompt: exit cleanly rather than dumping a traceback. Ctrl-C *during*
        # a command is caught in CviqShell.onecmd and returns to the prompt with the model intact.
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
