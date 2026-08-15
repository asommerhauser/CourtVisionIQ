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

  python evaluate.py --version 1.0 --name trial1 --shard 1/5
      Run only this shard's slice of the holdout (games 1/5, i.e. holdout[0::5]). Launch N such
      processes (--shard 1/N .. N/N, same --name) to share one GPU: the rollout's worker threads
      are GIL-bound, so process-level sharding is what turns spare CPU cores into throughput.
      Shards write disjoint game folders and skip the aggregate report; when ALL shards finish,
      merge with --report-only. Per-game seeds are position-independent, so the merged run is
      bit-identical to an unsharded one.

  python evaluate.py --version 1.0 --name pace-097 --report-only
      Rebuild the report over finished games, no new sims.

The holdout set + data paths come from the training run state (set by train.py); the weights come
from --version (artifacts/v<version>/).
"""
from __future__ import annotations

import argparse
import os
import re

# Grow the GPU allocation on demand; must be set before TF imports (importing FullRun pulls keras).
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

from training.full_run import DEFAULT_STATE_PATH, FullRun


def parse_shard(value: str) -> tuple[int, int]:
    """Parse ``"I/N"`` (1-based) into ``(i, n)``; raises ``ValueError`` on anything malformed.

    Shard ``i`` of ``n`` owns ``holdout[i-1::n]`` — the N slices are disjoint and together cover
    every holdout game exactly once, so N concurrent processes never race on a game folder.
    """
    m = re.fullmatch(r"(\d+)/(\d+)", value.strip())
    if not m:
        raise ValueError(f"--shard must look like I/N (e.g. 2/5), got {value!r}")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 1 or not 1 <= i <= n:
        raise ValueError(f"--shard needs 1 <= I <= N, got {value!r}")
    return i, n


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
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="Simulate only this 1-based slice of the holdout (holdout[I-1::N]) so N "
                         "processes can share one GPU. Requires --name (all shards must target the "
                         "same run dir); skips the aggregate report — run --report-only after all "
                         "shards finish to merge.")
    ap.add_argument("--state", default=DEFAULT_STATE_PATH, help="Full-run state file path.")
    args = ap.parse_args()

    shard = None
    if args.shard:
        try:
            shard = parse_shard(args.shard)
        except ValueError as e:
            ap.error(str(e))
        if not args.name:
            ap.error("--shard requires --name so every shard resolves the same run dir "
                     "(auto-named eval-NNN dirs would race).")
        if args.report_only:
            ap.error("--shard cannot be combined with --report-only (the merge report always "
                     "covers the full holdout).")

    run = FullRun(state_path=args.state)
    if args.report_only:
        run.report(version=args.version, name=args.name)
    else:
        run.eval(version=args.version, name=args.name, n_sims=args.monte_carlo,
                 concurrency=args.concurrency, max_new=args.games, shard=shard)


if __name__ == "__main__":
    main()
