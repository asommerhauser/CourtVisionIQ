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
import math
from pathlib import Path

import numpy as np

from config import HOLDOUT_MANIFEST_NAME
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

DEFAULT_SIMS = 11
# Stat keys (display name -> they map onto BOX_STATS / derived) used in box accuracy tables.
_BOX_ACCURACY_STATS = ("pts", "fga", "fgm", "tpa", "tpm", "fta", "ftm",
                       "oreb", "dreb", "ast", "stl", "blk", "tov", "pf")


# --------------------------------------------------------------------------- rollout

def simulate_repeated(sim: GameSimulator, spec, home_starters, away_starters, *,
                      n_sims: int, seed0: int, home_team: str = "HOME",
                      away_team: str = "AWAY", return_histories: bool = False):
    """Play one matchup ``n_sims`` times (seeds ``seed0..seed0+n_sims-1``) -> list of box scores.

    With ``return_histories`` also return the per-sim event histories (so the stage evaluator can
    persist each generated play-by-play): returns ``(boxes, histories)`` instead of just ``boxes``.
    """
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
                  home_team: str = "HOME", away_team: str = "AWAY") -> dict:
    """Run ``n_sims`` predictions for one cleaned game and build its evaluation record."""
    spec = extract_game_input(game_df)
    try:
        home_starters, away_starters = _real_starters(game_df)
    except ValueError:
        home_starters = away_starters = None

    boxes = simulate_repeated(sim, spec, home_starters, away_starters, n_sims=n_sims,
                              seed0=seed0, home_team=home_team, away_team=away_team)
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


# --------------------------------------------------------------------------- pure metrics

def spread_metrics(pred: list[float], actual: list[float],
                   within=(3, 6, 10)) -> dict:
    """Error of predicted mean margin vs actual margin: MAE, bias, RMSE, correlation, within-N."""
    p, a = np.asarray(pred, float), np.asarray(actual, float)
    if p.size == 0:
        return {"n": 0, "mae": 0.0, "bias": 0.0, "rmse": 0.0, "corr": 0.0,
                "within": {str(w): 0.0 for w in within}}
    err = p - a
    abs_err = np.abs(err)
    corr = float(np.corrcoef(p, a)[0, 1]) if p.size > 1 and p.std() > 0 and a.std() > 0 else 0.0
    return {
        "n": int(p.size),
        "mae": float(abs_err.mean()),
        "bias": float(err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "corr": corr,
        "within": {str(w): float((abs_err <= w).mean()) for w in within},
    }


def win_metrics(win_probs: list[float], outcomes: list[bool], picks_correct: list[bool],
                bins=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> dict:
    """Pick accuracy + probability calibration (Brier, log-loss, binned reliability)."""
    p = np.asarray(win_probs, float)
    y = np.asarray([1.0 if o else 0.0 for o in outcomes], float)
    n = int(p.size)
    if n == 0:
        return {"n": 0, "pick_accuracy": 0.0, "brier": 0.0, "log_loss": 0.0, "calibration": []}
    eps = 1e-12
    clipped = np.clip(p, eps, 1 - eps)
    log_loss = float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean())

    calibration = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        last = math.isclose(hi, bins[-1])
        mask = (p >= lo) & (p <= hi if last else p < hi)
        if mask.any():
            calibration.append({
                "lo": lo, "hi": hi, "n": int(mask.sum()),
                "pred_mean": float(p[mask].mean()), "obs_rate": float(y[mask].mean()),
            })
    return {
        "n": n,
        "pick_accuracy": float(np.mean([1.0 if c else 0.0 for c in picks_correct])),
        "brier": float(((p - y) ** 2).mean()),
        "log_loss": log_loss,
        "calibration": calibration,
    }


def _stat_errors(pairs: list[tuple[float, float]]) -> dict:
    """MAE / bias / predicted-mean / actual-mean for a list of (predicted, actual) pairs."""
    if not pairs:
        return {"n": 0, "pred_mean": 0.0, "actual_mean": 0.0, "mae": 0.0, "bias": 0.0}
    pred = np.array([p for p, _ in pairs], float)
    act = np.array([a for _, a in pairs], float)
    err = pred - act
    return {"n": len(pairs), "pred_mean": float(pred.mean()), "actual_mean": float(act.mean()),
            "mae": float(np.abs(err).mean()), "bias": float(err.mean())}


# --------------------------------------------------------------------------- set-level

def _aggregate(records: list[dict]) -> dict:
    """Roll per-game records up into the report's aggregate blocks."""
    # Win + spread (headline).
    win = win_metrics([r["win_prob_home"] for r in records],
                      [r["actual_home_win"] for r in records],
                      [r["pick_correct"] for r in records])
    spread = spread_metrics([r["pred_margin_mean"] for r in records],
                            [r["actual_margin"] for r in records])

    # Team box accuracy + reliability, pooling both sides of every game.
    team_acc, team_reliability = {}, {}
    for f in _BOX_ACCURACY_STATS:
        pairs, stds = [], []
        for r in records:
            for side in ("home", "away"):
                pairs.append((r["team_pred"][side][f], r["team_actual"][side][f]))
                stds.append(r["team_std"][side][f])
        team_acc[f] = _stat_errors(pairs)
        team_reliability[f] = float(np.mean(stds)) if stds else 0.0

    # Per-player box accuracy + reliability (matched by name; union of sims + actual, zeros for absent).
    player_acc, player_reliability = {}, {}
    for f in _BOX_ACCURACY_STATS:
        pairs, stds = [], []
        for r in records:
            for side in ("home", "away"):
                for name in r["players"][side]:
                    pred = r["player_avg"][side][name][f]
                    actual = r["player_actual"][side].get(name, {}).get(f, 0.0)
                    pairs.append((pred, actual))
                    stds.append(r["player_std"][side][name][f])
        player_acc[f] = _stat_errors(pairs)
        player_reliability[f] = float(np.mean(stds)) if stds else 0.0

    # Advanced (four factors + pace), pooling both sides.
    advanced = {}
    for k in ADVANCED_LABELS:
        pairs = []
        for r in records:
            for side in ("home", "away"):
                pairs.append((r["adv_pred"][side][k], r["adv_actual"][side][k]))
        advanced[k] = _stat_errors(pairs)

    return {
        "win": win,
        "spread": spread,
        "team_accuracy": team_acc,
        "team_reliability": team_reliability,
        "player_accuracy": player_acc,
        "player_reliability": player_reliability,
        "advanced": advanced,
        "headline": {
            "pick_accuracy": win["pick_accuracy"],
            "brier": win["brier"],
            "spread_mae": spread["mae"],
            "points_mae": team_acc["pts"]["mae"],
        },
    }


def evaluate_holdout(*, n_sims: int = DEFAULT_SIMS, games: int | None = None,
                     data_dir: str = "./data", processed_dir: str = "./data/processed",
                     artifacts_root: str = DEFAULT_ARTIFACTS_ROOT,
                     reports_root: str = DEFAULT_REPORTS_ROOT, seed0: int = 0,
                     run_name: str | None = None) -> dict:
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
        print(f"  evaluating game {gid} ({n_sims} sims)...")
        records.append(evaluate_game(sim, game, n_sims=n_sims, seed0=seed0))

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
    print(f"  win-pick accuracy   {h['pick_accuracy'] * 100:5.1f}%   (Brier {h['brier']:.3f})")
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
    args = ap.parse_args()
    evaluate_holdout(n_sims=args.sims, games=args.games, data_dir=args.data_dir,
                     processed_dir=args.processed_dir, artifacts_root=args.artifacts_root,
                     reports_root=args.reports_root, seed0=args.seed0, run_name=args.run_name)


if __name__ == "__main__":
    main()


__all__ = [
    "simulate_repeated", "average_box", "std_box", "evaluate_game", "build_game_record",
    "spread_metrics", "win_metrics", "evaluate_holdout",
]
