"""Tests for the holdout evaluation harness — pure Python, no TF / trained models."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from simulation.box_score import BoxScore, PlayerLine
from simulation.stats import BOX_STATS, advanced_stats, score_win_prob, team_totals
from simulation import evaluation as ev
from simulation import eval_metrics as em
from reporting import eval_report


# --------------------------------------------------------------------------- helpers

def _box(home_lines, away_lines, hs, as_) -> BoxScore:
    return BoxScore(home=home_lines, away=away_lines, home_score=hs, away_score=as_)


# --------------------------------------------------------------------------- box aggregation

def test_average_and_std_box_match_by_name_with_zeros_for_absent():
    b1 = _box([PlayerLine("A", pts=10, fga=8, fgm=4, seconds=600)], [], 10, 0)
    b2 = _box([PlayerLine("A", pts=20, fga=12, fgm=8), PlayerLine("B", pts=5)], [], 25, 0)
    players = ["A", "B"]

    avg = ev.average_box([b1, b2], "home", players)
    std = ev.std_box([b1, b2], "home", players)

    assert avg["A"]["pts"] == 15.0
    assert avg["B"]["pts"] == 2.5            # absent in b1 -> counts as 0
    assert std["A"]["pts"] == 5.0            # population std of [10, 20]
    assert std["B"]["pts"] == 2.5            # population std of [0, 5]


def test_std_zero_when_identical():
    b = _box([PlayerLine("A", pts=12)], [], 12, 0)
    std = ev.std_box([b, b, b], "home", ["A"])
    assert std["A"]["pts"] == 0.0


# --------------------------------------------------------------------------- advanced stats

def test_advanced_stats_math():
    team = {f: 0.0 for f in BOX_STATS}
    team.update(fga=80, fgm=36, tpm=6, fta=20, oreb=10, tov=14, seconds=14400)  # 240 player-min
    opp = {f: 0.0 for f in BOX_STATS}
    opp.update(dreb=30, oreb=12)

    adv = advanced_stats(team, opp)
    poss = 80 - 10 + 14 + 0.44 * 20             # 92.8
    assert adv["poss"] == pytest.approx(poss)
    assert adv["pace"] == pytest.approx(poss)    # exactly 48 game-minutes
    assert adv["efg"] == pytest.approx((36 + 0.5 * 6) / 80)
    assert adv["tov_pct"] == pytest.approx(14 / poss)
    assert adv["oreb_pct"] == pytest.approx(10 / (10 + 30))
    assert adv["ft_rate"] == pytest.approx(20 / 80)


def test_advanced_stats_zero_safe():
    zero = {f: 0.0 for f in BOX_STATS}
    adv = advanced_stats(zero, zero)
    assert adv["pace"] == 0.0 and adv["efg"] == 0.0 and adv["oreb_pct"] == 0.0


# --------------------------------------------------------------------------- spread / win metrics

def test_spread_metrics():
    m = ev.spread_metrics([3, -2, 10], [5, -1, 4])
    assert m["mae"] == pytest.approx(3.0)
    assert m["bias"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(math.sqrt(41 / 3))
    assert m["within"]["3"] == pytest.approx(2 / 3)


def test_win_metrics():
    w = ev.win_metrics([0.6, 0.4, 0.55], [True, False, True], [True, True, True])
    assert w["pick_accuracy"] == 1.0
    assert w["brier"] == pytest.approx((0.16 + 0.16 + 0.2025) / 3)
    assert w["n"] == 3
    # calibration bins should cover all three games
    assert sum(b["n"] for b in w["calibration"]) == 3


def test_empty_metrics_are_safe():
    assert ev.spread_metrics([], [])["n"] == 0
    assert ev.win_metrics([], [], [])["n"] == 0


def test_score_win_prob():
    # Symmetric: an exactly-even matchup is a coin flip; sign of the margin drives the rest.
    assert score_win_prob(0.0, 5.0) == pytest.approx(0.5)
    assert score_win_prob(5.0, 5.0) == pytest.approx(0.5 * (1 + math.erf(1 / math.sqrt(2))))
    assert score_win_prob(-5.0, 5.0) == pytest.approx(1 - score_win_prob(5.0, 5.0))
    # Zero spread collapses to a hard call by the sign of the mean margin.
    assert score_win_prob(3.0, 0.0) == 1.0
    assert score_win_prob(-3.0, 0.0) == 0.0
    assert score_win_prob(0.0, 0.0) == 0.5


def test_score_win_view_picks_by_mean_margin():
    # Vote would split this game 50/50, but the mean margin is clearly negative -> pick away.
    r = {"pred_margin_mean": -4.0, "pred_margin_std": 6.0, "actual_winner": "away"}
    view = em.score_win_view(r)
    assert view["pick"] == "away"
    assert view["pick_correct"] is True
    assert view["win_prob_home"] < 0.5


# --------------------------------------------------------------------------- report writer

def _make_record(gid: int, win_prob: float, pred_margin: float, actual_margin: int,
                 tuning: dict | None = None, evaluated_at: str | None = None) -> dict:
    """A fully-populated per-game record (the shape evaluate_game returns).

    ``tuning`` / ``evaluated_at`` populate the sim-time provenance the progression view segments on.
    """
    def teamrow(pts):
        row = {f: 0.0 for f in BOX_STATS}
        row.update(pts=float(pts), fga=80.0, fgm=36.0, tpa=18.0, tpm=6.0, fta=20.0, ftm=15.0,
                   oreb=10.0, dreb=30.0, ast=20.0, stl=7.0, blk=4.0, tov=14.0, pf=20.0,
                   seconds=14400.0)
        return row

    def player(pts, secs):
        row = {f: 0.0 for f in BOX_STATS}
        row.update(pts=float(pts), fga=10.0, fgm=5.0, seconds=float(secs))
        return row

    adv = {k: 0.4 for k in eval_report.ADVANCED_LABELS}
    sides = ("home", "away")
    return {
        "game_id": gid, "n_sims": 11,
        "tuning": tuning, "evaluated_at": evaluated_at,
        "win_prob_home": win_prob, "pred_pick": "home" if win_prob > 0.5 else "away",
        "actual_winner": "home" if actual_margin > 0 else "away",
        "actual_home_win": actual_margin > 0,
        "pick_correct": (win_prob > 0.5) == (actual_margin > 0),
        "pred_margin_mean": pred_margin, "pred_margin_std": 5.0, "actual_margin": actual_margin,
        "pred_home_score": 100.0, "pred_away_score": 95.0,
        "actual_home_score": 102, "actual_away_score": 99,
        "team_pred": {s: teamrow(100) for s in sides},
        "team_std": {s: {f: 1.0 for f in BOX_STATS} for s in sides},
        "team_actual": {s: teamrow(101) for s in sides},
        "adv_pred": {s: dict(adv) for s in sides},
        "adv_actual": {s: dict(adv) for s in sides},
        "players": {s: ["A", "B"] for s in sides},
        "player_avg": {s: {"A": player(15, 1800), "B": player(8, 1200)} for s in sides},
        "player_std": {s: {"A": {f: 1.0 for f in BOX_STATS},
                           "B": {f: 0.5 for f in BOX_STATS}} for s in sides},
        "player_actual": {s: {"A": player(16, 1850), "B": player(7, 1100)} for s in sides},
    }


def test_aggregate_and_write_report(tmp_path):
    records = [_make_record(1, 0.7, 4.0, 3), _make_record(2, 0.3, -6.0, -5)]
    agg = ev._aggregate(records)

    assert agg["headline"]["pick_accuracy"] == 1.0
    assert agg["team_accuracy"]["pts"]["n"] == 4          # 2 games x 2 sides
    assert agg["spread"]["n"] == 2

    # Average-score winner is aggregated alongside the vote method (both call these correctly here).
    assert agg["headline"]["score_pick_accuracy"] == 1.0
    assert agg["win_score"]["n"] == 2
    assert 0.0 <= agg["headline"]["score_brier"] <= 1.0

    # No per-game tuning -> a single progression segment (everything collapses).
    assert len(agg["progression"]) == 1
    assert agg["progression"][0]["n_games"] == 2

    report = eval_report.build_report(records=records, aggregate=agg, n_sims=11, run_name="test")
    run_dir = eval_report.write_eval_report(report, reports_root=str(tmp_path))

    assert (run_dir / "report.html").exists()
    assert (run_dir / "report.json").exists()
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Holdout evaluation report" in html and "Win prediction" in html
    assert "average score" in html and "majority vote" in html

    games = pd.read_parquet(run_dir / "games.parquet")
    assert set(games["game_id"]) == {1, 2}
    assert "win_prob_home" in games.columns and "pick_correct" in games.columns
    assert {"score_win_prob_home", "score_pick", "score_pick_correct"}.issubset(games.columns)

    box = pd.read_parquet(run_dir / "box_players.parquet")
    assert {"pred_pts", "std_pts", "actual_pts"}.issubset(box.columns)
    assert len(box) == 2 * 2 * 2                         # games x sides x players

    summary = pd.read_parquet(run_dir / "summary.parquet")
    assert {"scope", "metric", "predicted", "actual", "mae", "bias"}.issubset(summary.columns)
    assert (summary["scope"] == "advanced").any()

    # Tuning capture: the dials that produced the run are recorded in the report + HTML, and a
    # one-row run_summary.parquet joins them to the headline outcomes for cross-run analysis.
    import config
    assert report["tuning"]["DELTA_TIME_SCALE"] == config.DELTA_TIME_SCALE
    assert "Run configuration" in html
    run_summary = pd.read_parquet(run_dir / "run_summary.parquet")
    assert len(run_summary) == 1
    assert {"run_id", "pick_accuracy", "pace_bias", "fga_bias", "efg_bias",
            "DELTA_TIME_SCALE", "SUB_INCOMING_TEMPERATURE"}.issubset(run_summary.columns)
    assert run_summary.iloc[0]["pick_accuracy"] == 1.0
    assert run_summary.iloc[0]["n_tuning_segments"] == 1


# --------------------------------------------------------------------------- tuning progression

def test_progression_segments_by_distinct_tuning():
    tune_a = {"DELTA_TIME_SCALE": 1.0, "HOME_COURT_SHOT_BIAS": 0.0}
    tune_b = {"DELTA_TIME_SCALE": 1.06, "HOME_COURT_SHOT_BIAS": 0.0}
    # Two games under tune A, then one under tune B (in evaluation order via evaluated_at).
    records = [
        _make_record(1, 0.7, 4.0, 3, tuning=tune_a, evaluated_at="2026-06-28T10:00:00"),
        _make_record(2, 0.3, -6.0, -5, tuning=tune_a, evaluated_at="2026-06-28T10:01:00"),
        _make_record(3, 0.6, 2.0, 1, tuning=tune_b, evaluated_at="2026-06-28T11:00:00"),
    ]
    prog = em.progression(records)

    assert [s["segment"] for s in prog] == [1, 2]
    assert prog[0]["n_games"] == 2 and prog[0]["game_ids"] == [1, 2]
    assert prog[1]["n_games"] == 1 and prog[1]["game_ids"] == [3]
    # First segment is the baseline (no prior to diff against); second flags the one dial that moved.
    assert prog[0]["changed_dials"] == {}
    assert prog[1]["changed_dials"] == {"DELTA_TIME_SCALE": 1.06}
    # Each segment's metrics match _aggregate_core over just its records (no recursion / leakage).
    seg2_core = em._aggregate_core(records[2:])
    assert prog[1]["metrics"]["spread_mae"] == pytest.approx(seg2_core["headline"]["spread_mae"])


def test_progression_renders_section_and_parquet(tmp_path):
    tune_a = {"DELTA_TIME_SCALE": 1.0}
    tune_b = {"DELTA_TIME_SCALE": 1.06}
    records = [
        _make_record(1, 0.7, 4.0, 3, tuning=tune_a, evaluated_at="2026-06-28T10:00:00"),
        _make_record(2, 0.3, -6.0, -5, tuning=tune_b, evaluated_at="2026-06-28T11:00:00"),
    ]
    agg = ev._aggregate(records)
    assert len(agg["progression"]) == 2

    report = eval_report.build_report(records=records, aggregate=agg, n_sims=11, run_name="prog")
    run_dir = eval_report.write_eval_report(report, reports_root=str(tmp_path))

    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Tuning progression" in html and "DELTA_TIME_SCALE=1.06" in html

    prog = pd.read_parquet(run_dir / "progression.parquet")
    assert len(prog) == 2
    assert {"segment", "n_games", "spread_mae", "DELTA_TIME_SCALE", "changed_dials"}.issubset(
        prog.columns)
