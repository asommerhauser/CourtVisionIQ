"""
update_eval_report.py — regenerate an existing evaluation report from its ``report.json``.

The eval harness stores every per-game record losslessly in ``report.json``. This re-derives the
aggregate blocks (so metrics added after the run — e.g. the average-score winner) and rewrites
``report.html`` + the Parquet tables **without re-running any simulations** and without loading any
trained model. It only reads scalars already in the records (margins, scores, box totals), so it's
fast (seconds, not a rollout).

By default it rewrites the report in place (same run id / folder), preserving the original run
identity, timestamp, git commit, and tuning snapshot. Pass ``--out-root`` to write a fresh copy
under a different reports root instead.

    # update the report the conversation referenced, in place
    python -m reporting.update_eval_report reports/evaluation/20260628-074942-a9a72f-full-train

    # update every evaluation report under the default root
    python -m reporting.update_eval_report --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reporting.report_artifacts import DEFAULT_REPORTS_ROOT
from reporting.eval_report import build_report, write_eval_report
from simulation.eval_metrics import _aggregate


def update_report(run_dir: str | Path, *, out_root: str | None = None) -> Path:
    """Rebuild one run's HTML + Parquet from its ``report.json``. Returns the written run dir."""
    run_dir = Path(run_dir)
    report_json = run_dir / "report.json"
    if not report_json.exists():
        raise FileNotFoundError(f"No report.json at {report_json}; nothing to update.")

    report = json.loads(report_json.read_text(encoding="utf-8"))
    records = report.get("records")
    if not records:
        raise ValueError(f"{report_json} has no per-game records to re-aggregate.")

    # Backfill per-game tuning for older runs (predating sim-time tuning capture) from the run-level
    # snapshot, so the progression view shows a single segment tagged with the run's dials rather than
    # an empty/unknown one. Newer records already carry their own per-game tuning and are left as-is.
    run_tuning = report.get("tuning")
    if run_tuning:
        for r in records:
            r.setdefault("tuning", run_tuning)

    aggregate = _aggregate(records)
    rebuilt = build_report(records=records, aggregate=aggregate, n_sims=report["n_sims"],
                           run_name=report.get("run_name"), tuning=report.get("tuning"))
    # build_report mints a fresh run_id / created_at / git_commit; keep the original run's identity
    # and provenance so an in-place update overwrites the same folder and stays traceable.
    rebuilt["run_id"] = run_dir.name
    for field in ("created_at", "git_commit", "platform"):
        if report.get(field) is not None:
            rebuilt[field] = report[field]

    # In place: write back under this run's reports root (<root>/evaluation/<run_id>/).
    reports_root = out_root if out_root is not None else str(run_dir.parent.parent)
    written = write_eval_report(rebuilt, reports_root=reports_root)
    return written


def _discover(reports_root: str) -> list[Path]:
    """Every evaluation run dir (one with a report.json) under ``reports_root``."""
    base = Path(reports_root) / "evaluation"
    if not base.exists():
        return []
    return sorted(p.parent for p in base.glob("*/report.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate eval report(s) from report.json (no sims).")
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="Path to a run dir (the folder containing report.json).")
    ap.add_argument("--all", action="store_true",
                    help="Update every evaluation report under --reports-root.")
    ap.add_argument("--reports-root", default=DEFAULT_REPORTS_ROOT,
                    help="Reports root for --all discovery / default in-place root.")
    ap.add_argument("--out-root", default=None,
                    help="Write the rebuilt report(s) under this root instead of in place.")
    args = ap.parse_args()

    if args.all:
        targets = _discover(args.reports_root)
        if not targets:
            print(f"No evaluation reports found under {args.reports_root}/evaluation/.")
            return
    elif args.run_dir:
        targets = [Path(args.run_dir)]
    else:
        ap.error("pass a run_dir or --all")

    for run_dir in targets:
        written = update_report(run_dir, out_root=args.out_root)
        print(f"updated {written.resolve()}")


if __name__ == "__main__":
    main()
