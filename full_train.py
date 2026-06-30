"""
full_train.py — CLI for the single recency-weighted full train + batched holdout (training/full_run.py).

Reuses the cleaned data + vocab + warmup that `python train_full.py init --fresh` already built — no
re-clean, no re-warmup. Weights go to ./artifacts_full (the curriculum's ./artifacts is left alone;
delete it yourself whenever). Each command is user-launched; nothing auto-advances.

    python full_train.py setup                 # compute the mid-2023 cut + the next-100 holdout
    python full_train.py train                 # one fresh full train (recency-weighted), then STOP
    python full_train.py retrain-shot-type     # retrain ONLY shot_type on existing tensors (live FGs)
    python full_train.py eval                  # predict the next 10 holdout games, then STOP
    python full_train.py eval                  # ... repeat until all 100 are done ...
    python full_train.py eval-all              # predict ALL holdout games straight (paid GPU), report every 10
    python full_train.py status                # progress
    python full_train.py report                # rebuild the aggregate report over finished games

Interrupted mid-train? Re-run `train` — it resumes at the next unfinished model. Interrupted
mid-eval? Re-run `eval` — finished games are skipped.
"""
from __future__ import annotations

import argparse

from training.full_run import DEFAULT_STATE_PATH, FullRun


def main() -> None:
    ap = argparse.ArgumentParser(description="Single full-corpus train + batched holdout eval.")
    ap.add_argument("--state", default=DEFAULT_STATE_PATH, help="Full-run state file path.")
    sub = ap.add_subparsers(dest="command", required=True)

    p_setup = sub.add_parser("setup", help="Compute the train/holdout cut (no reclean/warmup).")
    p_setup.add_argument("--data-dir", default="./data")
    p_setup.add_argument("--processed-dir", default="./data/processed")
    p_setup.add_argument("--epochs", type=int, default=50)
    p_setup.add_argument("--batch-size", type=int, default=64)  # paid GPU has VRAM for it

    sub.add_parser("train", help="One fresh full train of every model (recency-weighted).")
    sub.add_parser("retrain-shot-type",
                   help="Retrain ONLY shot_type on the existing cond_*.npz (live FGs {2pt,3pt} only).")
    sub.add_parser("eval", help="Predict the next batch of holdout games, then stop.")
    sub.add_parser("eval-all",
                   help="Predict ALL holdout games straight through (paid-GPU run), reporting every batch.")
    sub.add_parser("status", help="Show progress.")
    sub.add_parser("report", help="Rebuild the aggregate report over finished games.")

    args = ap.parse_args()
    run = FullRun(state_path=args.state)

    if args.command == "setup":
        run.setup(data_dir=args.data_dir, processed_dir=args.processed_dir,
                  epochs=args.epochs, batch_size=args.batch_size)
    elif args.command == "train":
        run.train()
    elif args.command == "retrain-shot-type":
        run.retrain_shot_type()
    elif args.command == "eval":
        run.eval()
    elif args.command == "eval-all":
        run.eval_all()
    elif args.command == "status":
        run.status()
    elif args.command == "report":
        run.report()


if __name__ == "__main__":
    main()
