"""
train_full.py — CLI for the staged, user-gated full-corpus curriculum (see training/curriculum.py).

Typical run (each command is launched by you; nothing auto-advances):

    python train_full.py init --fresh         # wipe, re-clean, warmup fit, print schedule
    python train_full.py train                 # train the current stage, then STOP
    python train_full.py eval                  # predict its next 10 games, then STOP + advance
    python train_full.py train                 # ... repeat train/eval per stage ...
    python train_full.py status                # progress at any time
    python train_full.py report                # cross-stage growth report (after the last stage)

Interrupted mid-training? Just re-run ``train`` — it resumes at the next unfinished model.
"""
from __future__ import annotations

import argparse

from training.curriculum import Curriculum, DEFAULT_STATE_PATH


def main() -> None:
    ap = argparse.ArgumentParser(description="Staged full-corpus curriculum trainer.")
    ap.add_argument("--state", default=DEFAULT_STATE_PATH, help="Curriculum state file path.")
    sub = ap.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Wipe + re-clean + warmup fit + build the schedule.")
    p_init.add_argument("--fresh", action="store_true",
                        help="Required: confirms the destructive wipe of data/artifacts/reports.")
    p_init.add_argument("--data-dir", default="./data")
    p_init.add_argument("--processed-dir", default="./data/processed")
    p_init.add_argument("--epochs", type=int, default=50)
    p_init.add_argument("--batch-size", type=int, default=32)

    sub.add_parser("train", help="Train the current stage (warm-started), then stop.")
    sub.add_parser("eval", help="Predict the current stage's holdout, then stop + advance.")
    sub.add_parser("status", help="Show the schedule and per-stage progress.")
    sub.add_parser("report", help="Build the cross-stage growth report (after the final stage).")

    args = ap.parse_args()
    cur = Curriculum(state_path=args.state)

    if args.command == "init":
        if not args.fresh:
            raise SystemExit(
                "init performs a destructive FULL CLEAN + RETRAIN; pass --fresh to confirm."
            )
        cur.init_fresh(data_dir=args.data_dir, processed_dir=args.processed_dir,
                       epochs=args.epochs, batch_size=args.batch_size)
    elif args.command == "train":
        cur.train_stage()
    elif args.command == "eval":
        cur.eval_stage()
    elif args.command == "status":
        cur.status()
    elif args.command == "report":
        cur.growth_report()


if __name__ == "__main__":
    main()
