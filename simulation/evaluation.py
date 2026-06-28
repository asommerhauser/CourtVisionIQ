"""
evaluation.py — full holdout evaluation harness (sim accuracy vs reality).

Where ``diagnostics.py`` is the *quick baseline* (a few sims, printed aggregates to localize
drift), this is the **scoring** tool. For every game in the holdout manifest it runs the
prediction ``--sims`` times (default 11), then measures how close the simulator is to the real
games along four axes:

  1. **Box score** — an *average* box score (per player + team) over the sims, compared to the
     real game, plus a *standard-deviation* box score (how repeatable each stat is).
  2. **Advanced stats** — four factors + pace (eFG%, TOV%, OREB%, DREB%, FT rate, pace),
     predicted vs actual across the whole set.
  3. **Win prediction** (the headline) — if home wins 6 of 11 sims we predict home at a 54% win
     probability; we measure how often the majority pick is the *correct* winner, plus calibration
     (Brier / log-loss).
  4. **Point spread** — predicted mean margin vs actual margin (MAE, bias, RMSE, correlation,
     within-N hit rates).

Results are written as a self-contained HTML report + queryable Parquet via
``reporting/eval_report.py`` (same on-disk conventions as the training reports).

This drives full rollouts, so run it yourself after the models are trained:

    python -m simulation.evaluation --sims 11
"""
from __future__ import annotations

# Import TensorFlow before pandas-using project modules (see diagnostics.py / main.py).
try:  # noqa: SIM105
    import tensorflow  # noqa: F401
except Exception:
    pass

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from config import HOLDOUT_MANIFEST_NAME, ROLLOUT_BATCH_SIZE, tuning_snapshot
from data_loading import load_all_cleaned
from models.artifacts import DEFAULT_ARTIFACTS_ROOT
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT
from simulation.box_score import BoxScore, generate_box_score
from simulation.controller import GameController
from simulation.game_input import extract_game_input
from simulation.game_simulator import GameSimulator
from simulation.predict_game import _real_starters
from simulation.stats import (
    BOX_STATS,
    ADVANCED_LABELS,
    advanced_stats,
    player_stats,
    team_totals,
)
# Pure (TF-free) scoring math lives in eval_metrics; re-exported here so existing imports
# (stage_eval, tests) keep working and the rollout + metrics stay one import away.
from simulation.eval_metrics import (  # noqa: F401
    spread_metrics, win_metrics, score_win_view, _stat_errors, _aggregate, _BOX_ACCURACY_STATS,
)

DEFAULT_SIMS = 11


# --------------------------------------------------------------------------- rollout

def simulate_repeated(sim: GameSimulator, spec, home_starters, away_starters, *,
                      n_sims: int, seed0: int, home_team: str = "HOME",
                      away_team: str = "AWAY", return_histories: bool = False,
                      batch_size: int = ROLLOUT_BATCH_SIZE, show_progress: bool = False):
    """Play one matchup ``n_sims`` times (seeds ``seed0..seed0+n_sims-1``) -> list of box scores.

    With ``return_histories`` also return the per-sim event histories (so the stage evaluator can
    persist each generated play-by-play): returns ``(boxes, histories)`` instead of just ``boxes``.

    ``batch_size`` > 1 runs the sims through the batched rollout coordinator (one GPU forward pass per
    head across the concurrent sims, see ``simulation/batched_rollout.py``); 1 keeps the original
    one-at-a-time loop. Batching is pure scheduling, so the box scores are unchanged (deterministic
    heads) / distribution-equivalent (real heads).
    """
    if batch_size and batch_size > 1 and n_sims > 1:
        from simulation.batched_rollout import GameJob, run_jobs_batched
        jobs = [GameJob(home_roster=spec.home_roster, away_roster=spec.away_roster,
                        season=str(spec.season), home_starters=home_starters,
                        away_starters=away_starters, season_context=spec.season_context(),
                        seed=seed0 + s) for s in range(n_sims)]
        histories = run_jobs_batched(sim, jobs, batch_size=batch_size, show_progress=show_progress)
        boxes = [generate_box_score(h, home_team=home_team, away_team=away_team) for h in histories]
        return (boxes, histories) if return_histories else boxes

    boxes: list[BoxScore] = []
    histories: list[list[dict]] = []
    for s in range(n_sims):
        ctrl = GameController(sim, seed=seed0 + s)
        ctrl.start(spec.home_roster, spec.away_roster, season=str(spec.season),
                   home_starters=home_starters, away_starters=away_starters,
                   season_context=spec.season_context())
        history = ctrl.run()
        histories.append(history)
        boxes.append(generate_box_score(history, home_team=home_team, away_team=away_team))
    return (boxes, histories) if return_histories else boxes


# --------------------------------------------------------------------------- box aggregation

def _side_lines(box: BoxScore, side: str):
    return box.home if side == "home" else box.away


def _players_on_side(boxes: list[BoxScore], side: str, extra: list[str]) -> list[str]:
    """Union of players appearing on ``side`` across the sims, plus ``extra`` (the actual game)."""
    names: set[str] = set(extra)
    for box in boxes:
        names.update(pl.player for pl in _side_lines(box, side))
    return sorted(names)


def _stack_player_stats(boxes: list[BoxScore], side: str,
                        players: list[str]) -> dict[str, dict[str, list[float]]]:
    """For each player, the list of each stat's value across sims (0 when the sim didn't play him)."""
    cols = {name: {f: [] for f in BOX_STATS} for name in players}
    zero = {f: 0.0 for f in BOX_STATS}
    for box in boxes:
        by_name = {pl.player: pl for pl in _side_lines(box, side)}
        for name in players:
            line = by_name.get(name)
            stats = player_stats(line) if line is not None else zero
            for f in BOX_STATS:
                cols[name][f].append(stats[f])
    return cols


def average_box(boxes: list[BoxScore], side: str,
                players: list[str]) -> dict[str, dict[str, float]]:
    """Per-player mean of every BOX_STAT across the sims."""
    cols = _stack_player_stats(boxes, side, players)
    return {name: {f: float(np.mean(v)) for f, v in stats.items()} for name, stats in cols.items()}


def std_box(boxes: list[BoxScore], side: str,
            players: list[str]) -> dict[str, dict[str, float]]:
    """Per-player population std of every BOX_STAT across the sims (simulation repeatability)."""
    cols = _stack_player_stats(boxes, side, players)
    return {name: {f: float(np.std(v)) for f, v in stats.items()} for name, stats in cols.items()}


def _team_series(boxes: list[BoxScore], side: str) -> dict[str, list[float]]:
    """Per-team-total stat series across sims (one value per sim)."""
    series = {f: [] for f in BOX_STATS}
    for box in boxes:
        totals = team_totals(_side_lines(box, side))
        for f in BOX_STATS:
            series[f].append(totals[f])
    return series


def _advanced_series(boxes: list[BoxScore], side: str) -> dict[str, list[float]]:
    """Per-sim advanced stats for one side (ratios computed per sim, not on averaged totals)."""
    opp = "away" if side == "home" else "home"
    out: dict[str, list[float]] = {k: [] for k in ADVANCED_LABELS}
    for box in boxes:
        adv = advanced_stats(team_totals(_side_lines(box, side)),
                             team_totals(_side_lines(box, opp)))
        for k in ADVANCED_LABELS:
            out[k].append(adv[k])
    return out


# --------------------------------------------------------------------------- per-game

def evaluate_game(sim: GameSimulator, game_df, *, n_sims: int, seed0: int,
                  home_team: str = "HOME", away_team: str = "AWAY",
                  batch_size: int = ROLLOUT_BATCH_SIZE) -> dict:
    """Run ``n_sims`` predictions for one cleaned game and build its evaluation record."""
    spec = extract_game_input(game_df)
    try:
        home_starters, away_starters = _real_starters(game_df)
    except ValueError:
        home_starters = away_starters = None

    boxes = simulate_repeated(sim, spec, home_starters, away_starters, n_sims=n_sims,
                              seed0=seed0, home_team=home_team, away_team=away_team,
                              batch_size=batch_size)
    return build_game_record(game_df, boxes, n_sims=n_sims,
                             home_team=home_team, away_team=away_team)


def build_game_record(game_df, boxes: list[BoxScore], *, n_sims: int,
                      home_team: str = "HOME", away_team: str = "AWAY") -> dict:
    """Assemble one game's evaluation record from its (already-simulated) box scores.

    Split out from ``evaluate_game`` so the stage evaluator can run the sims itself (keeping the
    per-sim histories to persist) and still produce the identical record for the aggregate report.
    """
    actual_box = generate_box_score(game_df, home_team=home_team, away_team=away_team)
    margins = [b.home_score - b.away_score for b in boxes]
    home_wins = sum(1 for m in margins if m > 0)
    win_prob_home = home_wins / len(boxes)
    mean_margin = float(np.mean(margins))
    if win_prob_home > 0.5:
        pred_pick = "home"
    elif win_prob_home < 0.5:
        pred_pick = "away"
    else:  # split decision — break the tie by the average margin
        pred_pick = "home" if mean_margin >= 0 else "away"

    actual_margin = actual_box.home_score - actual_box.away_score
    actual_home_win = actual_margin > 0
    actual_winner = "home" if actual_margin > 0 else "away" if actual_margin < 0 else "tie"

    # Team totals (mean over sims) + actual, per side.
    team_pred = {side: {f: float(np.mean(v)) for f, v in _team_series(boxes, side).items()}
                 for side in ("home", "away")}
    team_std = {side: {f: float(np.std(v)) for f, v in _team_series(boxes, side).items()}
                for side in ("home", "away")}
    team_actual = {side: team_totals(_side_lines(actual_box, side)) for side in ("home", "away")}

    # Advanced stats (mean over per-sim ratios) + actual, per side.
    adv_pred = {side: {k: float(np.mean(v)) for k, v in _advanced_series(boxes, side).items()}
                for side in ("home", "away")}
    adv_actual = {
        "home": advanced_stats(team_actual["home"], team_actual["away"]),
        "away": advanced_stats(team_actual["away"], team_actual["home"]),
    }

    # Per-player average / std box, matched by name (union of sims + actual).
    player_avg, player_std, player_actual, players = {}, {}, {}, {}
    for side in ("home", "away"):
        actual_names = [pl.player for pl in _side_lines(actual_box, side)]
        names = _players_on_side(boxes, side, actual_names)
        players[side] = names
        player_avg[side] = average_box(boxes, side, names)
        player_std[side] = std_box(boxes, side, names)
        player_actual[side] = {pl.player: player_stats(pl) for pl in _side_lines(actual_box, side)}

    return {
        "game_id": int(game_df["game_id"].iloc[0]),
        "n_sims": n_sims,
        # Tuning provenance captured at sim time: which dials produced THIS game's sims, so the
        # report can segment the holdout by distinct tuning (games simmed across retunes keep their
        # own snapshot). evaluated_at orders the progression by when each game was actually run.
        "tuning": tuning_snapshot(),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "win_prob_home": win_prob_home,
        "pred_pick": pred_pick,
        "actual_winner": actual_winner,
        "actual_home_win": actual_home_win,
        "pick_correct": (pred_pick == actual_winner),
        "pred_margin_mean": mean_margin,
        "pred_margin_std": float(np.std(margins)),
        "actual_margin": int(actual_margin),
        "pred_home_score": team_pred["home"]["pts"],
        "pred_away_score": team_pred["away"]["pts"],
        "actual_home_score": actual_box.home_score,
        "actual_away_score": actual_box.away_score,
        "team_pred": team_pred, "team_std": team_std, "team_actual": team_actual,
        "adv_pred": adv_pred, "adv_actual": adv_actual,
        "players": players,
        "player_avg": player_avg, "player_std": player_std, "player_actual": player_actual,
    }


def evaluate_holdout(*, n_sims: int = DEFAULT_SIMS, games: int | None = None,
                     data_dir: str = "./data", processed_dir: str = "./data/processed",
                     artifacts_root: str = DEFAULT_ARTIFACTS_ROOT,
                     reports_root: str = DEFAULT_REPORTS_ROOT, seed0: int = 0,
                     run_name: str | None = None,
                     batch_size: int = ROLLOUT_BATCH_SIZE) -> dict:
    """Evaluate every holdout game ``n_sims`` times, write a report, return the report dict."""
    from reporting.eval_report import build_report, write_eval_report

    manifest = Path(processed_dir) / HOLDOUT_MANIFEST_NAME
    if not manifest.exists():
        raise FileNotFoundError(
            f"No holdout manifest at {manifest}; run preprocess first (it writes {HOLDOUT_MANIFEST_NAME})."
        )
    holdout_ids = [int(g) for g in json.loads(manifest.read_text(encoding="utf-8"))]
    if games is not None:
        holdout_ids = holdout_ids[:games]

    df = load_all_cleaned(data_dir, parse_rosters=True)
    sim = GameSimulator.load(artifacts_root=artifacts_root)

    records: list[dict] = []
    for gid in holdout_ids:
        game = df[df["game_id"] == int(gid)].sort_values("time")
        if game.empty:
            continue
        print(f"  evaluating game {gid} ({n_sims} sims, batch {batch_size})...")
        records.append(evaluate_game(sim, game, n_sims=n_sims, seed0=seed0, batch_size=batch_size))

    aggregate = _aggregate(records)
    report = build_report(records=records, aggregate=aggregate, n_sims=n_sims,
                          run_name=run_name)
    run_dir = write_eval_report(report, reports_root=reports_root)
    _print_summary(aggregate, len(records), n_sims)
    print(f"\nReport -> {run_dir.resolve()}")
    report["run_dir"] = str(run_dir)
    return report


def _print_summary(agg: dict, n_games: int, n_sims: int) -> None:
    h = agg["headline"]
    print(f"\n=== holdout evaluation  ({n_games} games x {n_sims} sims) ===")
    print(f"  win-pick (vote)     {h['pick_accuracy'] * 100:5.1f}%   (Brier {h['brier']:.3f})")
    print(f"  win-pick (score)    {h['score_pick_accuracy'] * 100:5.1f}%   "
          f"(Brier {h['score_brier']:.3f})")
    print(f"  point-spread MAE    {h['spread_mae']:5.1f} pts   (bias {agg['spread']['bias']:+.1f})")
    print(f"  team points MAE     {h['points_mae']:5.1f} pts")


def main():
    ap = argparse.ArgumentParser(description="Full holdout evaluation harness (sim vs reality).")
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS,
                    help=f"Predictions per game (default: {DEFAULT_SIMS}).")
    ap.add_argument("--games", type=int, default=None, help="How many holdout games (default: all).")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--processed-dir", default="./data/processed")
    ap.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    ap.add_argument("--reports-root", default=DEFAULT_REPORTS_ROOT)
    ap.add_argument("--seed0", type=int, default=0, help="First seed (sims use seed0..seed0+sims-1).")
    ap.add_argument("--run-name", default=None, help="Optional human label appended to the run id.")
    ap.add_argument("--batch-size", type=int, default=ROLLOUT_BATCH_SIZE,
                    help=f"Concurrent game-sims batched per GPU pass (default: {ROLLOUT_BATCH_SIZE}; "
                         f"1 = one-at-a-time).")
    args = ap.parse_args()
    evaluate_holdout(n_sims=args.sims, games=args.games, data_dir=args.data_dir,
                     processed_dir=args.processed_dir, artifacts_root=args.artifacts_root,
                     reports_root=args.reports_root, seed0=args.seed0, run_name=args.run_name,
                     batch_size=args.batch_size)


if __name__ == "__main__":
    main()


__all__ = [
    "simulate_repeated", "average_box", "std_box", "evaluate_game", "build_game_record",
    "spread_metrics", "win_metrics", "score_win_view", "evaluate_holdout",
]
