"""
diagnostics.py — lightweight sim-vs-real baseline comparison.

This is the *quick* baseline pass, not the full holdout-analysis tool: it simulates each holdout
game a few times and prints **predicted vs actual** team-level aggregates so you can see, in one
table, where the rollout drifts from reality. It exists to localize the shot-attempt inflation:

  * if the **Δt distribution** is too short / the **event histogram** is inflated across the board
    -> it's a PACE problem (too many possessions; look at the time head);
  * if **FG%** is low at a normal possession count -> a MISS-RATE problem (shot_result);
  * if **OREB% / shots-per-possession** is high -> OFFENSIVE-REBOUND chains (no shot clock).

Read-only. Reuses the existing prediction plumbing (loads the model once, then drives
GameController per game/seed) and the box-score decoder; compares against each real game's own
box score and play-by-play.

Run AFTER the models are trained (this drives full rollouts, so run it yourself):

    python -m simulation.diagnostics --games 8 --seeds 3
"""
from __future__ import annotations

# Import TensorFlow before pandas-using project modules. On Windows, importing pandas first can
# break TF's native DLL initialization (see main.py); the rollout below loads TF-backed models.
try:  # noqa: SIM105
    import tensorflow  # noqa: F401
except Exception:
    pass

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import HOLDOUT_MANIFEST_NAME
from data_loading import load_all_cleaned
from models.artifacts import DEFAULT_ARTIFACTS_ROOT
from simulation.box_score import BoxScore, PlayerLine, generate_box_score
from simulation.controller import GameController
from simulation.game_input import extract_game_input
from simulation.game_simulator import GameSimulator
from simulation.predict_game import _real_starters

# Counting fields summed per team from a side's PlayerLine list.
_TEAM_FIELDS = ("fga", "fgm", "tpa", "tpm", "fta", "ftm", "oreb", "dreb", "ast", "tov", "pts")
# Events we histogram (the rest are frame/derived rows).
_EVENT_TYPES = ("shot", "assist", "turnover", "foul", "rebound", "block", "substitution")


def _team_totals(lines: list[PlayerLine]) -> dict[str, float]:
    return {f: float(sum(getattr(pl, f) for pl in lines)) for f in _TEAM_FIELDS}


def _possessions(t: dict[str, float]) -> float:
    """Standard possessions estimate: FGA - OREB + TOV + 0.44*FTA."""
    return t["fga"] - t["oreb"] + t["tov"] + 0.44 * t["fta"]


def _box_team_rows(box: BoxScore) -> list[dict[str, float]]:
    """Two per-team aggregate rows (home, away) with possessions / OREB% / shots-per-poss filled.

    OREB% needs the opponent's DREB, so both sides are computed together.
    """
    home, away = _team_totals(box.home), _team_totals(box.away)
    rows = []
    for t, opp in ((home, away), (away, home)):
        poss = _possessions(t)
        oreb_chances = t["oreb"] + opp["dreb"]
        rows.append({
            **t,
            "poss": poss,
            "fg_pct": (t["fgm"] / t["fga"]) if t["fga"] else 0.0,
            "oreb_pct": (t["oreb"] / oreb_chances) if oreb_chances else 0.0,
            "shots_per_poss": (t["fga"] / poss) if poss else 0.0,
        })
    return rows


def _dt(times) -> np.ndarray:
    """Positive inter-event gaps from a sequence of absolute timestamps."""
    arr = np.asarray(list(times), dtype=np.float64)
    if arr.size < 2:
        return np.empty((0,))
    d = np.diff(arr)
    return d[d > 0]


def _event_hist(events) -> dict[str, float]:
    counts = pd.Series(list(events)).value_counts()
    return {e: float(counts.get(e, 0)) for e in _EVENT_TYPES}


def compare_holdout(*, games: int | None = None, seeds: int = 3,
                    data_dir: str = "./data", processed_dir: str = "./data/processed",
                    artifacts_root: str = DEFAULT_ARTIFACTS_ROOT,
                    seed0: int = 0) -> dict:
    """Simulate each holdout game ``seeds`` times and aggregate predicted-vs-actual stats."""
    manifest = Path(processed_dir) / HOLDOUT_MANIFEST_NAME
    if not manifest.exists():
        raise FileNotFoundError(
            f"No holdout manifest at {manifest}; run preprocess first (it writes {HOLDOUT_MANIFEST_NAME})."
        )
    holdout_ids = json.loads(manifest.read_text(encoding="utf-8"))
    if games is not None:
        holdout_ids = holdout_ids[:games]

    df = load_all_cleaned(data_dir, parse_rosters=True)
    sim = GameSimulator.load(artifacts_root=artifacts_root)

    pred_team, act_team = [], []          # per-team stat rows
    pred_dt, act_dt = [], []              # pooled inter-event gaps
    pred_hist, act_hist = [], []          # per-game event histograms
    pred_events_per_game, act_events_per_game = [], []

    for gid in holdout_ids:
        game = df[df["game_id"] == int(gid)].sort_values("time")
        if game.empty:
            continue
        spec = extract_game_input(game)
        actual_box = generate_box_score(game)
        act_team.extend(_box_team_rows(actual_box))
        act_dt.append(_dt(game["time"]))
        act_hist.append(_event_hist(game["event"]))
        act_events_per_game.append(int((~game["event"].isin(["start", "end"])).sum()))
        try:
            home_starters, away_starters = _real_starters(game)
        except ValueError:
            home_starters = away_starters = None

        for s in range(seeds):
            ctrl = GameController(sim, seed=seed0 + s)
            ctrl.start(spec.home_roster, spec.away_roster, season=str(spec.season),
                       home_starters=home_starters, away_starters=away_starters,
                       season_context=spec.season_context())
            history = ctrl.run()
            pred_box = generate_box_score(history)
            pred_team.extend(_box_team_rows(pred_box))
            pred_dt.append(_dt([r["time"] for r in history]))
            pred_hist.append(_event_hist([r["event"] for r in history]))
            pred_events_per_game.append(sum(1 for r in history
                                            if r["event"] not in ("start", "end")))

    report = {
        "n_games": len(act_hist),
        "seeds": seeds,
        "team_stats": _summarize_team(pred_team, act_team),
        "dt": _summarize_dt(np.concatenate(pred_dt) if pred_dt else np.empty(0),
                            np.concatenate(act_dt) if act_dt else np.empty(0)),
        "events_per_game": {
            "predicted": _mean(pred_events_per_game), "actual": _mean(act_events_per_game),
        },
        "event_histogram": _summarize_hist(pred_hist, act_hist),
    }
    _print_report(report)
    return report


def _mean(xs) -> float:
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else 0.0


def _summarize_team(pred: list[dict], act: list[dict]) -> dict:
    keys = ("fga", "fg_pct", "tpa", "fta", "tov", "oreb_pct", "poss", "shots_per_poss", "pts")
    return {k: {"predicted": _mean([r[k] for r in pred]),
                "actual": _mean([r[k] for r in act])} for k in keys}


def _summarize_dt(pred: np.ndarray, act: np.ndarray) -> dict:
    def stats(a):
        if a.size == 0:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0}
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90))}
    return {"predicted": stats(pred), "actual": stats(act)}


def _summarize_hist(pred: list[dict], act: list[dict]) -> dict:
    return {e: {"predicted": _mean([h[e] for h in pred]),
                "actual": _mean([h[e] for h in act])} for e in _EVENT_TYPES}


def _print_report(r: dict) -> None:
    print(f"\n=== sim-vs-real baseline  ({r['n_games']} holdout games x {r['seeds']} seeds) ===\n")

    def row(label, pred, act, fmt="{:.1f}"):
        delta = (pred - act) / act * 100 if act else float("nan")
        print(f"  {label:<18} pred {fmt.format(pred):>8}   real {fmt.format(act):>8}   "
              f"{delta:+6.1f}%")

    print("-- team per game --")
    labels = {"fga": "FGA", "fg_pct": "FG%", "tpa": "3PA", "fta": "FTA", "tov": "TOV",
              "oreb_pct": "OREB%", "poss": "possessions", "shots_per_poss": "shots/poss",
              "pts": "PTS"}
    for k, lab in labels.items():
        v = r["team_stats"][k]
        fmt = "{:.3f}" if k in ("fg_pct", "oreb_pct", "shots_per_poss") else "{:.1f}"
        row(lab, v["predicted"], v["actual"], fmt)

    print("\n-- inter-event Δt (sec) --")
    for stat in ("mean", "median", "p90"):
        row(f"Δt {stat}", r["dt"]["predicted"][stat], r["dt"]["actual"][stat], "{:.2f}")

    print("\n-- events per game --")
    row("events", r["events_per_game"]["predicted"], r["events_per_game"]["actual"])

    print("\n-- event histogram (per game) --")
    for e in _EVENT_TYPES:
        v = r["event_histogram"][e]
        row(e, v["predicted"], v["actual"])
    print()


def main():
    ap = argparse.ArgumentParser(description="Lightweight sim-vs-real baseline diagnostic.")
    ap.add_argument("--games", type=int, default=None, help="How many holdout games (default: all).")
    ap.add_argument("--seeds", type=int, default=3, help="Sims per game (default: 3).")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--processed-dir", default="./data/processed")
    ap.add_argument("--artifacts-root", default=DEFAULT_ARTIFACTS_ROOT)
    args = ap.parse_args()
    compare_holdout(games=args.games, seeds=args.seeds, data_dir=args.data_dir,
                    processed_dir=args.processed_dir, artifacts_root=args.artifacts_root)


if __name__ == "__main__":
    main()
