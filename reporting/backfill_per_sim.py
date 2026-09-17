"""
Backfill the per-sim vectors onto runs that were evaluated before 3.0 recorded them.

``build_game_record`` used to store only the *moments* of the sim distribution -- the mean and sd
of each team total, the mean margin and its sd. That is enough for every accuracy number in the
report and not enough for a single joint one: "do the two teams in a sim share a game?" is a
property of the paired sample, and the sample was averaged away before it was written. So the 2.0
evaluation's central finding -- corr(home, away) = 0.02 against a real 0.35 -- could only be
reached with a scratch script, and no run on disk carries it.

It does not have to stay that way, because the sims themselves are still on disk. ``_PbpSink``
writes every sim's play-by-play as it completes (``simulation/stage_eval.py``), in the same column
schema as the real game, and ``generate_box_score`` is the tally the report already scores against.
Re-reading those CSVs reconstructs exactly the vectors ``build_game_record`` would write today.

    python -m reporting.backfill_per_sim results/version2/v2-run4
    python -m reporting.backfill_per_sim --all

This is idempotent and additive: it writes four new keys into each ``games/*/record.json`` and into
``report.json``'s ``records``, touches nothing else, and skips a game that already has them unless
``--force``. It then re-runs ``update_eval_report`` so the HTML and Parquet pick the new metrics up.

Two guards, because a silent disagreement here would poison a headline metric:

- Every game's reconstructed scores are checked against ``run.json``'s ``per_sim_scores``, which
  was written by the rollout itself at sim time. A mismatch means the CSVs and the record disagree
  about what was simulated, and the game is **skipped with a warning** rather than backfilled.
- ``harvest.py --prune-finished`` deletes play-by-plays to reclaim disk. A pruned game cannot be
  backfilled, and says so, rather than being filled with a plausible-looking guess.
- A run whose play-by-plays predate the current token vocabulary cannot be re-tallied at all: the
  v1.0 runs under ``results/v1.0/`` carry the single ``3pt`` shot token that 2.0's fifteen shot
  zones replaced, and ``points_for_shot`` refuses it. The run is abandoned after its first game,
  with the reason, rather than the whole batch dying on it.

TF-free by construction -- see ``simulation/__init__`` on why that is now true of the whole
evaluation-report stack.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

from simulation.box_score import generate_box_score
from simulation.eval_metrics import PER_SIM_KEYS
from simulation.stats import advanced_stats, team_totals

DEFAULT_RESULTS_ROOT = "./results"
# Tolerance on the run.json cross-check. Scores are integers on both sides, so this is an exact
# comparison with room only for a float round-trip through JSON.
_SCORE_TOL = 0.5


def _side_lines(box, side: str):
    return box.home if side == "home" else box.away


def sim_csvs(game_dir: Path) -> list[Path]:
    """Every sim play-by-play in a game folder, in sim order.

    ``sim_01_playbyplay.csv`` .. ``sim_NN_playbyplay.csv`` -- zero-padded, so lexical order is sim
    order, which is the order ``run.json``'s ``per_sim_scores`` is written in.
    """
    return sorted(game_dir.glob("playbyplay/sim_*_playbyplay.csv"))


def vectors_for_game(game_dir: Path) -> dict[str, list[float]] | None:
    """Rebuild the four per-sim vectors from a game's sim CSVs, or None if they are gone."""
    paths = sim_csvs(game_dir)
    if not paths:
        return None
    out: dict[str, list[float]] = {k: [] for k in PER_SIM_KEYS}
    for path in paths:
        rows = pd.read_csv(path)
        box = generate_box_score(rows)
        totals = {side: team_totals(_side_lines(box, side)) for side in ("home", "away")}
        adv = {
            "home": advanced_stats(totals["home"], totals["away"]),
            "away": advanced_stats(totals["away"], totals["home"]),
        }
        out["per_sim_home_pts"].append(float(box.home_score))
        out["per_sim_away_pts"].append(float(box.away_score))
        out["per_sim_home_pace"].append(float(adv["home"]["pace"]))
        out["per_sim_away_pace"].append(float(adv["away"]["pace"]))
    return out


def _cross_check(game_dir: Path, vectors: dict[str, list[float]]) -> str | None:
    """Compare the rebuilt scores against what the rollout recorded. Returns a reason to skip."""
    run_path = game_dir / "run.json"
    if not run_path.exists():
        return None  # nothing to check against; the CSVs are the only record there is
    try:
        recorded = json.loads(run_path.read_text(encoding="utf-8")).get("per_sim_scores") or []
    except (ValueError, OSError):
        return None
    if not recorded:
        return None
    if len(recorded) != len(vectors["per_sim_home_pts"]):
        return (f"run.json lists {len(recorded)} sims but "
                f"{len(vectors['per_sim_home_pts'])} play-by-plays are on disk")
    for i, entry in enumerate(recorded):
        if (abs(float(entry["home"]) - vectors["per_sim_home_pts"][i]) > _SCORE_TOL
                or abs(float(entry["away"]) - vectors["per_sim_away_pts"][i]) > _SCORE_TOL):
            return (f"sim {i + 1} scores disagree: run.json "
                    f"{entry['home']}-{entry['away']} vs rebuilt "
                    f"{vectors['per_sim_home_pts'][i]:.0f}-{vectors['per_sim_away_pts'][i]:.0f}")
    return None


def backfill_run(run_dir: str | Path, *, force: bool = False, echo=print) -> dict:
    """Write the per-sim vectors into every game record of one run, and into its report.json."""
    run_dir = Path(run_dir)
    games = sorted(p for p in (run_dir / "games").glob("*") if (p / "record.json").exists())
    if not games:
        echo(f"  {run_dir.name}: no finished games")
        return {"filled": 0, "skipped": 0, "already": 0, "pruned": 0, "incompatible": 0}

    filled: dict[int, dict] = {}
    stats = {"filled": 0, "skipped": 0, "already": 0, "pruned": 0, "incompatible": 0}
    for game_dir in games:
        record_path = game_dir / "record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        gid = int(record["game_id"])
        if not force and all(k in record for k in PER_SIM_KEYS):
            stats["already"] += 1
            filled[gid] = {k: record[k] for k in PER_SIM_KEYS}
            continue
        try:
            vectors = vectors_for_game(game_dir)
        except (KeyError, ValueError) as exc:
            # Almost always a token this build cannot read -- a pre-2.0 run's "3pt" against the
            # fifteen zone tokens. Every game in the run will fail the same way, so say it once.
            echo(f"  {run_dir.name}: cannot re-tally these play-by-plays ({exc}); "
                 f"run predates the current vocabulary -- skipping it")
            stats["incompatible"] = len(games)
            return stats
        if vectors is None:
            stats["pruned"] += 1
            continue
        reason = _cross_check(game_dir, vectors)
        if reason is not None:
            echo(f"  SKIP game {gid}: {reason}")
            stats["skipped"] += 1
            continue
        record.update(vectors)
        record_path.write_text(json.dumps(record), encoding="utf-8")
        filled[gid] = vectors
        stats["filled"] += 1

    report_path = run_dir / "report.json"
    if report_path.exists() and filled:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for record in report.get("records", []):
            vectors = filled.get(int(record["game_id"]))
            if vectors is not None:
                record.update(vectors)
        report_path.write_text(json.dumps(report), encoding="utf-8")

    echo(f"  {run_dir.name}: {stats['filled']} filled, {stats['already']} already had them, "
         f"{stats['pruned']} pruned of play-by-play, {stats['skipped']} skipped on mismatch")
    return stats


def discover(results_root: str = DEFAULT_RESULTS_ROOT) -> list[Path]:
    """Every run directory under ``results_root`` that has a report to update."""
    return sorted({Path(p).parent
                   for p in glob.glob(os.path.join(results_root, "*", "*", "report.json"))})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("run_dir", nargs="*", help="run directory, e.g. results/version2/v2-run4")
    ap.add_argument("--all", action="store_true", help="every run under --results-root")
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--force", action="store_true",
                    help="recompute even for games that already carry the vectors")
    ap.add_argument("--no-report", action="store_true",
                    help="write the records but do not regenerate the HTML/Parquet")
    args = ap.parse_args(argv)

    runs = discover(args.results_root) if args.all else [Path(d) for d in args.run_dir]
    if not runs:
        raise SystemExit("nothing to do: pass a run directory or --all")

    for run_dir in runs:
        stats = backfill_run(run_dir, force=args.force)
        if stats["incompatible"]:
            continue
        if not args.no_report:
            from reporting.update_eval_report import update_report

            update_report(run_dir)
            print(f"  {run_dir.name}: report regenerated")


if __name__ == "__main__":
    main()
