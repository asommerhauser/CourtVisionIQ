"""
evaluate.py — CLI for evaluating a trained CourtVisionIQ model version against the holdout.

Writes a results run under results/v<version>/<eval-name>/: report.html + report.json at the root,
queryable parquet under data/, and one folder per game under games/ (box-score CSVs, a game.html
with predicted/actual/variance box scores, and the generated play-by-plays under playbyplay/).

  python evaluate.py --version 1.0
      Evaluate v1.0's holdout (auto-named eval-NNN), STAGE_SIMS sims/game, default concurrency.

  python evaluate.py --version 1.0 --name pace-097 --monte-carlo 21 --concurrency 48
      Named run; 21 sims/game; up to 48 game-sims per GPU forward pass (the VRAM knob). Lower
      --concurrency if you OOM; it is independent of --monte-carlo.

  python evaluate.py --version 1.0 --name pace-097 --games 10
      Predict only the next 10 unfinished holdout games into that run, then stop (batched).

  python evaluate.py --version 1.0 --name pace-097 --report-only
      Rebuild the report over finished games, no new sims.

The holdout set + data paths come from the training run state (set by train.py); the weights come
from --version (artifacts/v<version>/).
"""
from __future__ import annotations

import argparse
import os

# Grow the GPU allocation on demand; must be set before TF imports (importing FullRun pulls keras).
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from training.full_run import DEFAULT_STATE_PATH, FullRun


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a CourtVisionIQ model version on the holdout.")
    ap.add_argument("--version", help="Model version to evaluate (default: the run state's version).")
    ap.add_argument("--name", default=None, help="Eval folder name (default: auto eval-NNN).")
    ap.add_argument("--monte-carlo", type=int, default=None, dest="monte_carlo",
                    help="Sims per game to aggregate (Monte-Carlo count; default: STAGE_SIMS).")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Concurrent game-sims per GPU forward pass (VRAM knob; default 48). Lower it "
                         "if you hit OOM, raise it to use more of the card. Independent of --monte-carlo.")
    ap.add_argument("--games", type=int, default=None,
                    help="Cap NEW games simulated this call (batched / interrupt-friendly). Default: all.")
    ap.add_argument("--report-only", action="store_true",
                    help="Rebuild the report over finished games, no new sims.")
    ap.add_argument("--state", default=DEFAULT_STATE_PATH, help="Full-run state file path.")
    args = ap.parse_args()

    run = FullRun(state_path=args.state)
    if args.report_only:
        run.report(version=args.version, name=args.name)
    else:
        run.eval(version=args.version, name=args.name, n_sims=args.monte_carlo,
                 concurrency=args.concurrency, max_new=args.games)


if __name__ == "__main__":
    main()
