"""
evaluate.py — CLI for evaluating a trained CourtVisionIQ model version against the holdout.

Writes a results run under results/<model>/<run>/: report.html + report.json at the root,
queryable parquet under data/, and one folder per game under games/ (box-score CSVs, a game.html
with predicted/actual/variance box scores, and the generated play-by-plays under playbyplay/).

  python evaluate.py --model v1.0
      Evaluate v1.0's holdout (auto-named eval-NNN), STAGE_SIMS sims/game, default concurrency.

  python evaluate.py --model v1.0 --run pace-097 --monte-carlo 21 --concurrency 48
      Named run; 21 sims/game; up to 48 game-sims per GPU forward pass (the VRAM knob). Lower
      --concurrency if you OOM; it is independent of --monte-carlo.

  python evaluate.py --model v1.0 --run pace-097 --games 10
      Predict only the next 10 unfinished holdout games into that run, then stop (batched).

  python evaluate.py --model v1.0 --run trial1 --procs 4
      Same thing, supervised: sizes and launches 4 shard processes, shows one merged progress
      line, then merges the report once they finish. --procs auto sizes from usable cores and
      free VRAM. This is the normal way to use a many-core box.

  python evaluate.py --model v1.0 --run trial1 --shard 1/4
      Simulate only this shard's slice of the holdout (games 1/4, i.e. holdout[0::4]). Launch N
      such processes (--shard 1/N .. N/N, same --run) to share one GPU: within a process the
      rollout's worker threads are GIL-bound, so process-level sharding is what turns spare CPU
      cores into throughput. Shards write disjoint game folders and skip the aggregate report;
      once ALL shards finish, merge with --report-only. Per-game seeds are position-independent,
      so the merged run matches an unsharded one.

  python evaluate.py --model v1.0 --run pace-097 --report-only
      Rebuild the report over finished games, no new sims.

The holdout set + data paths come from the training run state (set by train.py); the weights come
from --model (artifacts/<model>/).
"""
from __future__ import annotations

import argparse
import os
import re

# Grow the GPU allocation on demand; must be set before TF imports (importing FullRun pulls keras).
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# config is TF-free; training.full_run is NOT (it pulls models.registry -> every model class ->
# keras), so it is imported inside main() instead. An argument error should not cost a
# TensorFlow import, and a --shard/--dials-only path has no reason to pay for one either.
from config import FULL_RUN_STATE_PATH as DEFAULT_STATE_PATH


def parse_shard(value: str) -> tuple[int, int]:
    """Parse ``"I/N"`` (1-based) into ``(i, n)``; raise ``ValueError`` on anything malformed.

    Shard ``i`` of ``n`` owns ``holdout[i-1::n]`` -- the N slices are disjoint and together cover
    every holdout game exactly once, so N concurrent processes never race on a game folder.
    """
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", value.strip())
    if not m:
        raise ValueError(f"--shard must look like I/N (e.g. 2/5), got {value!r}")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 1 or not 1 <= i <= n:
        raise ValueError(f"--shard needs 1 <= I <= N, got {value!r}")
    return i, n


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate a CourtVisionIQ model on the holdout. For an interactive session "
                    "that keeps the model resident across many runs, use: python cviq.py")
    ap.add_argument("--model", "--version", dest="model", default=None,
                    help="Model to evaluate, e.g. v1.0 (default: the run state's model).")
    ap.add_argument("--run", "--name", dest="run", default=None,
                    help="Run folder name -> results/<model>/<run>/ (default: auto eval-NNN).")
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
                         "processes can share one GPU. Requires --run (all shards must resolve the "
                         "same run dir); skips the aggregate report -- run --report-only after "
                         "every shard finishes to merge.")
    ap.add_argument("--procs", default=None, metavar="N",
                    help="Run N eval processes over disjoint holdout slices, then merge one "
                         "report ('auto' sizes from usable cores + free VRAM). Omit, or 1, for "
                         "the single-process path. Generates the --shard calls for you.")
    ap.add_argument("--dials", default=None, metavar="FILE",
                    help="Apply a dial package (JSON object of DIAL -> value) before simulating. "
                         "How a sharded run pins every process to the SAME inference dials.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Base seed for the Monte-Carlo sims; each (game, sim) pair derives its "
                         "own stream from it, so no two games replay the same draws. Shards and "
                         "pooled runs pass it through, so a re-run reproduces exactly. Change it "
                         "to resample a game set independently of an earlier run.")
    ap.add_argument("--state", default=DEFAULT_STATE_PATH, help="Full-run state file path.")
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    shard = None
    if args.shard:
        try:
            shard = parse_shard(args.shard)
        except ValueError as e:
            ap.error(str(e))
        if not args.run:
            ap.error("--shard requires --run so every shard resolves the same run dir "
                     "(auto-named eval-NNN dirs would race).")
        if args.report_only:
            ap.error("--shard cannot be combined with --report-only (the merge report always "
                     "covers the full holdout).")
        if args.procs:
            ap.error("--procs generates the --shard calls; pass one or the other, not both.")

    if args.procs:
        if args.report_only:
            ap.error("--report-only simulates nothing, so --procs has nothing to parallelize.")
        if args.games:
            ap.error("--games is the single-process interrupt knob; it does not combine with "
                     "--procs. Drop one.")
        if str(args.procs) not in ("auto", ""):
            try:
                if int(args.procs) < 1:
                    raise ValueError
            except ValueError:
                ap.error(f"--procs must be a positive integer or 'auto', got {args.procs!r}")

    if args.dials:
        import config
        try:
            applied = config.apply_dial_file(args.dials)
        except ValueError as e:
            ap.error(str(e))
        print(f"[dials] applied {len(applied)} from {args.dials}")

    if args.procs and str(args.procs) != "1":
        from eval_pool import run_procs          # TF-free supervisor; children do the TF work
        return run_procs(args)

    from training.full_run import FullRun     # deferred: this is the TensorFlow import

    run = FullRun(state_path=args.state)
    if args.report_only:
        run.report(version=args.model, name=args.run)
    else:
        run.eval(version=args.model, name=args.run, n_sims=args.monte_carlo,
                 concurrency=args.concurrency, max_new=args.games, shard=shard,
                 seed=args.seed)


if __name__ == "__main__":
    main()
