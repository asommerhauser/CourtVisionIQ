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
from simulation.stats import ADVANCED_LABELS, MINUTES, REPORT_STATS, score_win_prob, stat_value

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
        return {"n": 0, "pick_accuracy": 0.0, "brier": 0.0, "brier_se": 0.0,
                "log_loss": 0.0, "calibration": []}
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
    # Brier is a mean over games, so it has a standard error -- and at n ~ 100 that error is
    # the whole story: two runs inside +-2 SE of each other are the same model, however
    # different the third decimal looks. Printed next to every Brier (v3_direction.md S8).
    per_game = (p - y) ** 2
    brier_se = float(per_game.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "pick_accuracy": float(np.mean([1.0 if c else 0.0 for c in picks_correct])),
        "brier": float(per_game.mean()),
        "brier_se": brier_se,
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


# --------------------------------------------------------------------------- minutes closeness

def minutes_closeness(records: list[dict], *, within=(2, 5, 10)) -> dict:
    """How close the sims get a player's minutes, pooled over every player-game.

    MAE is the headline: the average miss on one player's minutes. ``within`` adds the same
    within-N read the spread block uses -- the share of player-games called inside N minutes.

    Bias is reported but is near-zero by construction (a team hands out ~240 player-minutes a game,
    so a minute given to the wrong player is taken from the right one); MAE is the number that
    survives that cancellation.
    """
    err = []
    for r in records:
        for side in ("home", "away"):
            for name in r["players"][side]:
                err.append(stat_value(r["player_avg"][side][name], MINUTES)
                           - stat_value(r["player_actual"][side].get(name, {}), MINUTES))
    e = np.asarray(err, float)
    if not e.size:
        return {"n_player_games": 0, "mae": 0.0, "bias": 0.0,
                "within": {str(w): 0.0 for w in within}}
    abs_err = np.abs(e)
    return {
        "n_player_games": int(e.size),
        "mae": float(abs_err.mean()),
        "bias": float(e.mean()),
        "within": {str(w): float((abs_err <= w).mean()) for w in within},
    }



# --------------------------------------------------------------------------- joint structure

# Per-sim vectors a record carries once it was written by a 3.0-era build_game_record. Older
# records have only the moments (mean/std), so every reader here treats them as optional and a run
# that predates them reports n_games = 0 rather than raising -- the same defensive-read convention
# the quarter block already uses.
PER_SIM_KEYS = ("per_sim_home_pts", "per_sim_away_pts", "per_sim_home_pace", "per_sim_away_pace")


def _per_sim(record: dict) -> dict | None:
    """The four per-sim vectors off a record, or None when it predates them / is degenerate."""
    try:
        cols = {k: np.asarray(record[k], dtype=float) for k in PER_SIM_KEYS}
    except (KeyError, TypeError, ValueError):
        return None
    n = cols["per_sim_home_pts"].size
    if n < 2 or any(v.size != n for v in cols.values()):
        return None
    return cols


def joint_metrics(records: list[dict]) -> dict:
    """Do a sim's two teams share a game? The 3.0 headline diagnostic.

    The 2.0 evaluation found corr(home pts, away pts) = 0.02 across the sims of one game against
    0.35 in real games, and pace sd roughly half of real. Every "the model is over-confident"
    symptom follows from that one absence: with independent sides the margin sd is sqrt(2) x the
    per-side sd, which is why it lands near 16.5 against a real 13.7. It is a *missing* shared
    component, not a mis-sized one, so no shrinkage dial is the right fix -- which is exactly what
    this metric exists to keep saying while W3 is built.

    Each quantity is computed within a game (across its sims) and then averaged over games, because
    that is what the simulator controls; pooling every sim of every game would measure the spread
    of matchups instead. ``corr_se`` is the standard error of that game-level mean.
    """
    corrs, pace_sds, margin_sds, total_sds, side_sds = [], [], [], [], []
    for r in records:
        cols = _per_sim(r)
        if cols is None:
            continue
        home, away = cols["per_sim_home_pts"], cols["per_sim_away_pts"]
        if home.std() > 0 and away.std() > 0:
            corrs.append(float(np.corrcoef(home, away)[0, 1]))
        # Both sides of one game see the same possessions to within a rounding, so the game's
        # pace is their mean. Stored per sim as poss/48 (ADVANCED_LABELS["pace"]) rather than raw
        # possessions, so an overtime game is not counted as a fast one.
        pace = (cols["per_sim_home_pace"] + cols["per_sim_away_pace"]) / 2.0
        pace_sds.append(float(pace.std()))
        margin_sds.append(float((home - away).std()))
        total_sds.append(float((home + away).std()))
        side_sds.extend([float(home.std()), float(away.std())])

    def _mean(v):
        return float(np.mean(v)) if v else 0.0

    return {
        "n_games": len(margin_sds),
        "corr_home_away": _mean(corrs),
        "corr_se": (float(np.std(corrs, ddof=1) / math.sqrt(len(corrs))) if len(corrs) > 1 else 0.0),
        "pace_sd": _mean(pace_sds),
        "margin_sd": _mean(margin_sds),
        "total_sd": _mean(total_sds),
        "side_pts_sd": _mean(side_sds),
    }


# Stats the coverage check runs over. Deliberately NOT "reb": a record stores the sd of oreb and
# the sd of dreb separately, and sd(oreb + dreb) needs their covariance, which is not stored. Adding
# a derived "reb" to stat_value would give the right mean and a silently wrong spread -- and this
# function is entirely about the spread. Same reason nothing derived joins this list later.
COVERAGE_STATS = ("pts", "oreb", "dreb", "ast", MINUTES)


def coverage_metrics(records: list[dict], *, stats=COVERAGE_STATS, within=(1, 2)) -> dict:
    """How often the actual lands inside k model sds -- per player, per team total, and on margin.

    A well-calibrated predictive distribution puts 68% inside +-1 sd and 95% inside +-2. The 2.0
    evaluation found the per-player spread almost exactly right (65.7 / 94.5) while the margin was
    far too wide, which is :func:`joint_metrics`'s finding seen from the other side. The margin
    block reports the dispersion ratio outright -- predicted sd over realised residual sd -- because
    that single number is what a shrinkage dial would be fitted to, and the rule is that it reaches
    1.0 by fixing the structure instead.

    Every input is already on every record (``player_avg`` / ``player_std`` / ``player_actual``,
    the team blocks, ``pred_margin_*``), so this retrofits onto historical runs with no
    re-simulation.
    """
    unknown = [f for f in stats if f not in REPORT_STATS]
    if unknown:
        # Without this the loop below finds sd == 0 for every player, skips them all, and reports a
        # confident 0.0% coverage off an empty denominator.
        raise ValueError(f"coverage_metrics: {unknown} are not stored per-stat; "
                         f"pick from {REPORT_STATS}")

    def _rate(hits, n):
        return {str(k): (hits[k] / n if n else 0.0) for k in within}

    player, team = {}, {}
    player_n = team_n = 0
    for f in stats:
        p_hits = {k: 0 for k in within}
        t_hits = {k: 0 for k in within}
        p_n = t_n = 0
        for r in records:
            for side in ("home", "away"):
                for name in r["players"][side]:
                    sd = stat_value(r["player_std"][side][name], f)
                    if sd <= 0:
                        # A player the sims agree on exactly (never plays, or a stat he never
                        # records) carries no distribution to test. Dropping him is the honest
                        # denominator; counting him as a hit would inflate every coverage number.
                        continue
                    mu = stat_value(r["player_avg"][side][name], f)
                    actual = stat_value(r["player_actual"][side].get(name, {}), f)
                    p_n += 1
                    z = abs(actual - mu) / sd
                    for k in within:
                        p_hits[k] += 1 if z <= k else 0
                sd = stat_value(r["team_std"][side], f)
                if sd <= 0:
                    continue
                t_n += 1
                z = abs(stat_value(r["team_actual"][side], f)
                        - stat_value(r["team_pred"][side], f)) / sd
                for k in within:
                    t_hits[k] += 1 if z <= k else 0
        player[f] = _rate(p_hits, p_n)
        team[f] = _rate(t_hits, t_n)
        player_n, team_n = p_n, t_n

    m_hits = {k: 0 for k in within}
    m_n = 0
    pred_sds, residuals = [], []
    for r in records:
        resid = float(r["actual_margin"]) - float(r["pred_margin_mean"])
        residuals.append(resid)
        sd = float(r.get("pred_margin_std") or 0.0)
        if sd <= 0:
            continue
        pred_sds.append(sd)
        m_n += 1
        for k in within:
            m_hits[k] += 1 if abs(resid) / sd <= k else 0
    pred_sd = float(np.mean(pred_sds)) if pred_sds else 0.0
    # ddof=1: the residual sd is estimated from the same games, and at n ~ 100 the correction is
    # not negligible next to the ratio it feeds.
    resid_sd = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    return {
        "player": player, "team": team,
        "n_player_games": player_n, "n_team_games": team_n,
        "margin": {**_rate(m_hits, m_n), "n": m_n,
                   "pred_sd": pred_sd, "resid_sd": resid_sd,
                   "dispersion_ratio": (pred_sd / resid_sd) if resid_sd else 0.0},
    }


def paired_brier(records_a: list[dict], records_b: list[dict], *, score: bool = False) -> dict:
    """Compare two runs' Brier on the games they share, paired game by game.

    An unpaired SE overstates the uncertainty of a *comparison*: two runs on the same 100 games make
    correlated errors, and the sd of the per-game difference is much smaller than either run's own
    sd. This is the test that says whether a Brier of 0.209 and one of 0.233 on the same games are
    actually different -- across the 2.0 runs they were not.

    ``score=True`` compares the Gaussian score probabilities instead of the sim-count vote.
    """
    def _index(records):
        out = {}
        for r in records:
            prob = score_win_view(r)["win_prob_home"] if score else r["win_prob_home"]
            out[int(r["game_id"])] = (float(prob), 1.0 if r["actual_home_win"] else 0.0)
        return out

    a, b = _index(records_a), _index(records_b)
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return {"n": len(shared), "brier_a": 0.0, "brier_b": 0.0,
                "diff": 0.0, "diff_se": 0.0, "z": 0.0}
    ba = np.array([(a[g][0] - a[g][1]) ** 2 for g in shared])
    bb = np.array([(b[g][0] - b[g][1]) ** 2 for g in shared])
    d = ba - bb
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    return {
        "n": int(d.size),
        "brier_a": float(ba.mean()), "brier_b": float(bb.mean()),
        "diff": float(d.mean()), "diff_se": se,
        "z": float(d.mean() / se) if se else 0.0,
    }

# --------------------------------------------------------------------------- set-level

def _aggregate_core(records: list[dict]) -> dict:
    """Roll per-game records up into the report's aggregate blocks (no progression sub-block).

    Split from :func:`_aggregate` so :func:`progression` can re-aggregate each tuning segment with
    the same math without recursing.
    """
    outcomes = [r["actual_home_win"] for r in records]
    joint = joint_metrics(records)
    coverage = coverage_metrics(records)

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
        "joint": joint,
        "coverage": coverage,
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
            # The 3.0 additions: the two W3 gate numbers, and the SE that says whether a
            # Brier difference between two runs is real at all.
            "brier_se": win["brier_se"],
            "score_brier_se": win_score["brier_se"],
            "corr_home_away": joint["corr_home_away"],
            "margin_dispersion_ratio": coverage["margin"]["dispersion_ratio"],
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
        "corr_home_away": core["joint"]["corr_home_away"],
        "pace_sd": core["joint"]["pace_sd"],
        "margin_dispersion_ratio": core["coverage"]["margin"]["dispersion_ratio"],
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
    print(f"  win-pick (vote)     {h['pick_accuracy'] * 100:5.1f}%   "
          f"(Brier {h['brier']:.3f} +- {h.get('brier_se', 0.0):.3f})")
    print(f"  win-pick (score)    {h['score_pick_accuracy'] * 100:5.1f}%   "
          f"(Brier {h['score_brier']:.3f} +- {h.get('score_brier_se', 0.0):.3f})")
    print(f"  point-spread MAE    {h['spread_mae']:5.1f} pts   (bias {agg['spread']['bias']:+.1f})")
    print(f"  team points MAE     {h['points_mae']:5.1f} pts")
    if "player_minutes_mae" in h:
        print(f"  player minutes MAE  {h['player_minutes_mae']:5.1f} min")
    joint = agg.get("joint") or {}
    if joint.get("n_games"):
        # Real 2022-23 for comparison: corr 0.35, pace sd 5.3, margin sd 13.7. Printed here so
        # the gap is legible without opening the report.
        print(f"  corr(home,away)    {joint['corr_home_away']:+.3f}   (real +0.352)   "
              f"pace sd {joint['pace_sd']:.2f} (real 5.3)")
        margin = (agg.get("coverage") or {}).get("margin") or {}
        if margin.get("resid_sd"):
            print(f"  margin sd           {joint['margin_sd']:5.2f}   "
                  f"(actual residual {margin['resid_sd']:.2f}; "
                  f"dispersion {margin['dispersion_ratio']:.2f}x)")

__all__ = [
    "spread_metrics", "win_metrics", "score_win_view", "_aggregate", "_aggregate_core",
    "progression", "_stat_errors", "_BOX_ACCURACY_STATS", "minutes_closeness",
    "joint_metrics", "coverage_metrics", "paired_brier", "PER_SIM_KEYS", "COVERAGE_STATS",
]
