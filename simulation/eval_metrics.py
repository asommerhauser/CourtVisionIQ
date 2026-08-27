"""
eval_metrics.py — pure (TensorFlow-free) scoring math for the holdout evaluation.

Split out of ``simulation/evaluation.py`` so the per-game records (already simulated, or reloaded
from a run's ``report.json``) can be rolled up into the aggregate report blocks without importing
the rollout stack (TF / trained models). ``simulation/evaluation.py`` re-exports these so existing
imports keep working, and ``reporting/update_eval_report.py`` reuses them to regenerate a report.

Two winner methods are scored side by side:

  * **majority vote** — the share of sims the home team won is its win probability; the majority is
    the pick (the original headline).
  * **average score** — the winner is the sign of the *mean predicted margin*, and the probability
    comes from the sims' margin distribution (normal approx, :func:`simulation.stats.score_win_prob`).
    More robust when the box-score averages are good but individual sims are coin-flippy.
"""
from __future__ import annotations

import json
import math

import numpy as np

from config import MARGIN_CALIBRATION_INTERCEPT, MARGIN_CALIBRATION_SLOPE
from simulation.stats import ADVANCED_LABELS, MINUTES, score_win_prob, stat_value

# Stat keys used in the team/player box-accuracy tables. "minutes" leads the list the way MIN leads
# a box score; it is derived from the stored ``seconds`` by ``simulation.stats.stat_value``, so it
# scores on records written before minutes was a reported stat.
_BOX_ACCURACY_STATS = (MINUTES, "pts", "fga", "fgm", "tpa", "tpm", "fta", "ftm",
                       "oreb", "dreb", "ast", "stl", "blk", "tov", "pf")


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
    """Pick accuracy + probability calibration (Brier, log-loss, binned reliability).

    Method-agnostic: pass the majority-vote probabilities/picks for the vote headline, or the
    average-score probabilities/picks for the score headline.
    """
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


def calibrate_margin(mean: float) -> float:
    """Post-hoc linear calibration for the SPREAD headline only (config.MARGIN_CALIBRATION_*).

    Predicted margins run too extreme for their information content; this is a report-time
    correction, not a rollout dial -- it must not feed ``score_win_view`` (the win-probability
    methods), since a nonzero intercept can flip the sign of a near-zero margin and that block is
    scored on pick accuracy, not just MAE.
    """
    return MARGIN_CALIBRATION_INTERCEPT + MARGIN_CALIBRATION_SLOPE * mean


def score_win_view(record: dict) -> dict:
    """Average-score winner for one game record (derived from the stored margin scalars).

    Returns ``win_prob_home`` (normal-approx of the sims' margin distribution), ``pick`` (sign of the
    mean margin), and ``pick_correct``. Derived purely from ``pred_margin_mean`` / ``pred_margin_std``
    / ``actual_winner``, so it works on freshly-built records *and* on records reloaded from an older
    ``report.json`` that predate this metric.
    """
    mean = float(record.get("pred_margin_mean", 0.0))
    std = float(record.get("pred_margin_std", 0.0))
    pick = "home" if mean > 0 else "away" if mean < 0 else "tie"
    return {
        "win_prob_home": score_win_prob(mean, std),
        "pick": pick,
        "pick_correct": (pick == record.get("actual_winner")),
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

def _aggregate_core(records: list[dict]) -> dict:
    """Roll per-game records up into the report's aggregate blocks (no progression sub-block).

    Split from :func:`_aggregate` so :func:`progression` can re-aggregate each tuning segment with
    the same math without recursing.
    """
    outcomes = [r["actual_home_win"] for r in records]

    # Winner prediction, two methods: majority vote vs average predicted score.
    win = win_metrics([r["win_prob_home"] for r in records], outcomes,
                      [r["pick_correct"] for r in records])
    score_views = [score_win_view(r) for r in records]
    win_score = win_metrics([v["win_prob_home"] for v in score_views], outcomes,
                            [v["pick_correct"] for v in score_views])
    spread = spread_metrics([calibrate_margin(r["pred_margin_mean"]) for r in records],
                            [r["actual_margin"] for r in records])

    # Team box accuracy + reliability, pooling both sides of every game.
    team_acc, team_reliability = {}, {}
    for f in _BOX_ACCURACY_STATS:
        pairs, stds = [], []
        for r in records:
            for side in ("home", "away"):
                pairs.append((stat_value(r["team_pred"][side], f),
                              stat_value(r["team_actual"][side], f)))
                stds.append(stat_value(r["team_std"][side], f))
        team_acc[f] = _stat_errors(pairs)
        team_reliability[f] = float(np.mean(stds)) if stds else 0.0

    # Per-player box accuracy + reliability (matched by name; union of sims + actual, zeros for absent).
    player_acc, player_reliability = {}, {}
    for f in _BOX_ACCURACY_STATS:
        pairs, stds = [], []
        for r in records:
            for side in ("home", "away"):
                for name in r["players"][side]:
                    pred = stat_value(r["player_avg"][side][name], f)
                    actual = stat_value(r["player_actual"][side].get(name, {}), f)
                    pairs.append((pred, actual))
                    stds.append(stat_value(r["player_std"][side][name], f))
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
        "win_score": win_score,
        "spread": spread,
        "team_accuracy": team_acc,
        "team_reliability": team_reliability,
        "player_accuracy": player_acc,
        "player_reliability": player_reliability,
        "advanced": advanced,
        "headline": {
            "pick_accuracy": win["pick_accuracy"],
            "brier": win["brier"],
            "score_pick_accuracy": win_score["pick_accuracy"],
            "score_brier": win_score["brier"],
            "spread_mae": spread["mae"],
            "points_mae": team_acc["pts"]["mae"],
            # Minutes are a per-player prediction (a team always plays ~240), so the headline
            # tracks the player block -- that is the rotation model's error in the unit it is
            # tuned in.
            "player_minutes_mae": player_acc[MINUTES]["mae"],
        },
    }


# --------------------------------------------------------------------------- tuning progression

# The headline metrics tracked per tuning segment (what the dials actually move).
def _segment_metrics(core: dict) -> dict:
    """Pull the tuning-relevant headline metrics out of an `_aggregate_core` result."""
    h = core["headline"]
    return {
        "pick_accuracy": h["pick_accuracy"],
        "brier": h["brier"],
        "score_pick_accuracy": h["score_pick_accuracy"],
        "score_brier": h["score_brier"],
        "spread_mae": h["spread_mae"],
        "points_mae": h["points_mae"],
        "player_minutes_mae": h["player_minutes_mae"],
        "player_minutes_bias": core["player_accuracy"][MINUTES]["bias"],
        "pace_bias": core["advanced"]["pace"]["bias"],
        "fga_bias": core["team_accuracy"]["fga"]["bias"],
        "efg_bias": core["advanced"]["efg"]["bias"],
    }


def _tuning_key(record: dict) -> str:
    """Stable string identity of a record's tuning snapshot (missing -> empty)."""
    return json.dumps(record.get("tuning") or {}, sort_keys=True)


def progression(records: list[dict]) -> list[dict]:
    """Segment records by **distinct consecutive tuning** and score each segment.

    Records are ordered by ``(evaluated_at, game_id)`` so the trajectory reflects the order games were
    simulated (and thus the order the user retuned). Maximal runs of identical ``tuning`` snapshots
    become one segment; each segment carries the tuning, the dials that **changed vs the previous
    segment**, its game ids, and the tuning-relevant headline metrics (re-aggregated over just that
    segment). Records lacking a ``tuning`` stamp collapse into a single segment with empty tuning.
    """
    if not records:
        return []
    ordered = sorted(records, key=lambda r: (str(r.get("evaluated_at") or ""), r["game_id"]))

    segments: list[list[dict]] = []
    last_key = object()
    for r in ordered:
        key = _tuning_key(r)
        if key != last_key:
            segments.append([])
            last_key = key
        segments[-1].append(r)

    out: list[dict] = []
    prev_tuning: dict | None = None
    for i, seg in enumerate(segments, start=1):
        tuning = seg[0].get("tuning") or {}
        if prev_tuning is None:
            changed = {}                                   # first segment = baseline
        else:
            changed = {k: tuning.get(k) for k in tuning
                       if tuning.get(k) != prev_tuning.get(k)}
        out.append({
            "segment": i,
            "n_games": len(seg),
            "game_ids": [int(r["game_id"]) for r in seg],
            "evaluated_at": seg[0].get("evaluated_at"),
            "tuning": tuning,
            "changed_dials": changed,
            "metrics": _segment_metrics(_aggregate_core(seg)),
        })
        prev_tuning = tuning
    return out


def _aggregate(records: list[dict]) -> dict:
    """Full aggregate: the overall flattened blocks plus a per-tuning ``progression`` trajectory."""
    return {**_aggregate_core(records), "progression": progression(records)}



def reported_sims(records: list[dict], *, default: int) -> int:
    """The sim count the finished games were ACTUALLY run at, not the one a call asked for.

    A merge (``--report-only``, and the pooled run's merge step) simulates nothing, so its
    ``n_sims`` argument is whatever the default happens to be -- which used to stamp a 100-sim
    run as a 21-sim one. The records know: each carries the count it was built with.
    """
    counts = {int(r["n_sims"]) for r in records if r.get("n_sims")}
    if not counts:
        return default
    if len(counts) > 1:
        print(f"  WARNING: this run mixes sim counts ({', '.join(str(c) for c in sorted(counts))}"
              f"); reporting the smallest. Its per-game estimates are not equally precise.")
    return min(counts)


def print_summary(agg: dict, n_games: int, n_sims: int) -> None:
    """The four-line headline every eval path prints. Here rather than in ``evaluation.py`` so the
    pooled supervisor can print it without importing the rollout stack."""
    h = agg["headline"]
    print(f"\n=== holdout evaluation  ({n_games} games x {n_sims} sims) ===")
    print(f"  win-pick (vote)     {h['pick_accuracy'] * 100:5.1f}%   (Brier {h['brier']:.3f})")
    print(f"  win-pick (score)    {h['score_pick_accuracy'] * 100:5.1f}%   "
          f"(Brier {h['score_brier']:.3f})")
    print(f"  point-spread MAE    {h['spread_mae']:5.1f} pts   (bias {agg['spread']['bias']:+.1f})")
    print(f"  team points MAE     {h['points_mae']:5.1f} pts")
    if "player_minutes_mae" in h:
        print(f"  player minutes MAE  {h['player_minutes_mae']:5.1f} min")

__all__ = [
    "spread_metrics", "win_metrics", "score_win_view", "_aggregate", "_aggregate_core",
    "progression", "_stat_errors", "_BOX_ACCURACY_STATS",
]
