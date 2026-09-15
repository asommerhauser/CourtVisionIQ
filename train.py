"""
train.py — CLI for training CourtVisionIQ model versions (engine: training/full_run.py).

Model weights live one-per-version under ``artifacts/v<MAJOR.MINOR>/``. Each command is
user-launched; nothing auto-advances.

  python train.py --full --version 1.1 --batch-size 64 [--epochs 50] [--clean] [--rebuild-vocabs]
      Fresh full train of every head into artifacts/v1.1/. Requires --version and --batch-size.
      --clean re-cleans raw play-by-play into ./data first; --rebuild-vocabs refreezes the shared
      vocab from data (use after a re-clean). Interrupted mid-train? Re-run with --continue.

  python train.py --continue
      Resume an interrupted full train at the next unfinished head (uses the saved run state).

  python train.py --model shot_result [--version 1.1]
      Retrain ONE head in place, keeping the rest. Targets the current run's version.

  python train.py --status
      Show full-run progress.

Evaluation lives in a separate CLI:  python evaluate.py --version 1.1
"""
from __future__ import annotations

import argparse
import os

# Grow the GPU allocation on demand instead of grabbing all VRAM up front — friendlier if anything
# else shares the card. Must be set before TF imports, and importing FullRun pulls in keras.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from training.full_run import DEFAULT_STATE_PATH, FullRun


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CourtVisionIQ models.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true",
                      help="Fresh full train of every head (requires --name and --batch-size).")
    mode.add_argument("--model", metavar="NAME",
                      help="Retrain ONE head in place (keeps the others).")
    mode.add_argument("--continue", dest="cont", action="store_true",
                      help="Resume an interrupted full train at the next unfinished head.")
    mode.add_argument("--status", action="store_true", help="Show full-run progress.")
    mode.add_argument("--extend-holdout", dest="extend_holdout", action="store_true",
                      help="Re-cut the holdout to FINAL_HOLDOUT_GAMES on an already-trained "
                           "model, without re-running setup (which would reset the train). "
                           "Extends forward from the stored train cut only, and refuses if the "
                           "existing ids are not a prefix of the new ones.")

    ap.add_argument("--name", help="Model name, e.g. v1.1 or endgame-feats (required with --full). "
                                   "Free-form; a retrain takes a NEW name rather than overwriting.")
    ap.add_argument("--version", help="Deprecated alias for --name (accepts a bare '1.1').")
    ap.add_argument("--batch-size", type=int, help="Train batch size (required with --full).")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--clean", action="store_true",
                    help="With --full: re-clean raw play-by-play into ./data first.")
    ap.add_argument("--rebuild-vocabs", action="store_true",
                    help="With --full: rebuild+freeze the shared vocab from data (use after --clean).")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--processed-dir", default="./data/processed")
    ap.add_argument("--state", default=DEFAULT_STATE_PATH, help="Full-run state file path.")
    args = ap.parse_args()

    run = FullRun(state_path=args.state)

    name = args.name or args.version

    if args.full:
        if not name or args.batch_size is None:
            ap.error("--full requires --name and --batch-size.")
        if args.clean:
            from data_cleaner import DataCleaner
            from season_context import enrich
            print("[train] --clean: re-cleaning raw play-by-play into ./data ...")
            DataCleaner().run()
            enrich(args.data_dir)
        run.setup(name=name, data_dir=args.data_dir, processed_dir=args.processed_dir,
                  epochs=args.epochs, batch_size=args.batch_size)
        run.train(rebuild_vocabs=args.rebuild_vocabs)
    elif args.model:
        # Retrain one head against the CURRENT run state (which holds the model name + train cut).
        # A --name, if given, must match that state — retraining an arbitrary older model would
        # need its own state and is out of scope.
        if name and run.state.get("version") not in (None, name):
            ap.error(f"--model targets the currently trained model '{run.state.get('version')}', "
                     f"not '{name}'. Run a --full train for that name first.")
        run.retrain_model(args.model, batch_size=args.batch_size)
    elif args.cont:
        run.train()
    elif args.extend_holdout:
        if name and run.state.get("version") not in (None, name):
            ap.error(f"--extend-holdout targets the currently trained model "
                     f"'{run.state.get('version')}', not '{name}'.")
        # data_dir is left to the state: for a trained model the corpus it was cut against is
        # authoritative, and argparse's "./data" default would silently override a different one.
        run.extend_holdout()
    elif args.status:
        run.status()


if __name__ == "__main__":
    main()
