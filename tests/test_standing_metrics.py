"""
The 3.0 standing metrics: joint structure, distribution coverage, and the Brier standard error.

Pure Python -- no TF, no trained models, no simulation. Records are built by hand so every expected
value is arithmetic a reader can check, in the style of tests/test_evaluation.py.

What these guard, in one line each: that a metric computed from *averaged* fields cannot silently
report a joint property (the reason the per-sim vectors exist at all); that a stat whose spread is
not stored is refused rather than reported as zero coverage; and that a run written before 3.0 reads
as "not measured" rather than as a confident zero.
"""
from __future__ import annotations

import math

import pytest

from simulation.eval_metrics import (
    COVERAGE_STATS,
    PER_SIM_KEYS,
    coverage_metrics,
    joint_metrics,
    paired_brier,
    win_metrics,
)


def _record(game_id: int, *, home_pts, away_pts, home_pace=None, away_pace=None,
            actual_margin=0, win_prob=0.5, per_sim=True) -> dict:
    """A record carrying just enough for the metrics under test.

    ``home_pts`` / ``away_pts`` are the per-sim vectors; the moments are derived from them so the
    record is internally consistent the way a real one is.
    """
    n = len(home_pts)
    margins = [h - a for h, a in zip(home_pts, away_pts)]
    mean_margin = sum(margins) / n
    var = sum((m - mean_margin) ** 2 for m in margins) / n
    rec = {
        "game_id": game_id,
        "n_sims": n,
        "win_prob_home": win_prob,
        "actual_home_win": actual_margin > 0,
        "pick_correct": (win_prob > 0.5) == (actual_margin > 0),
        "pred_margin_mean": mean_margin,
        "pred_margin_std": math.sqrt(var),
        "actual_margin": actual_margin,
    }
    # Empty player/team blocks so the same builder can feed coverage_metrics, which reads them
    # unconditionally -- every real record carries them.
    empty = {f: 0.0 for f in ("seconds", "pts", "oreb", "dreb", "ast")}
    rec.update({
        "players": {"home": [], "away": []},
        "player_avg": {"home": {}, "away": {}},
        "player_std": {"home": {}, "away": {}},
        "player_actual": {"home": {}, "away": {}},
        "team_pred": {"home": dict(empty), "away": dict(empty)},
        "team_std": {"home": dict(empty), "away": dict(empty)},
        "team_actual": {"home": dict(empty), "away": dict(empty)},
    })
    if per_sim:
        rec["per_sim_home_pts"] = list(home_pts)
        rec["per_sim_away_pts"] = list(away_pts)
        rec["per_sim_home_pace"] = list(home_pace if home_pace is not None else [100.0] * n)
        rec["per_sim_away_pace"] = list(away_pace if away_pace is not None else [100.0] * n)
    return rec


def _player_record(game_id: int, *, mu, sd, actual) -> dict:
    """One home player, one stat block, for the coverage tests."""
    stats = {f: 0.0 for f in ("seconds", "pts", "oreb", "dreb", "ast")}
    return {
        "game_id": game_id,
        "players": {"home": ["A"], "away": []},
        "player_avg": {"home": {"A": {**stats, "pts": mu}}, "away": {}},
        "player_std": {"home": {"A": {**stats, "pts": sd}}, "away": {}},
        "player_actual": {"home": {"A": {**stats, "pts": actual}}, "away": {}},
        "team_pred": {"home": dict(stats), "away": dict(stats)},
        "team_std": {"home": dict(stats), "away": dict(stats)},
        "team_actual": {"home": dict(stats), "away": dict(stats)},
        "pred_margin_mean": 0.0,
        "pred_margin_std": 0.0,
        "actual_margin": 0,
    }


# --------------------------------------------------------------------------- joint structure

def test_two_teams_that_move_together_report_a_positive_correlation():
    """The whole point of the metric: a shared pace component shows up as corr > 0."""
    home = [100.0, 110.0, 120.0, 130.0]
    away = [95.0, 105.0, 115.0, 125.0]      # rises in lockstep -> corr = 1
    j = joint_metrics([_record(1, home_pts=home, away_pts=away)])
    assert j["n_games"] == 1
    assert j["corr_home_away"] == pytest.approx(1.0)


def test_two_independent_teams_report_a_correlation_near_zero_and_an_inflated_margin():
    """The 2.0 defect, reproduced in miniature.

    With zero covariance, Var(H-A) and Var(H+A) are both VarH + VarA -- so the margin sd is too
    wide and the total sd too narrow by exactly the same missing term. That identity is the reason
    a shrinkage dial is the wrong fix, so it is worth pinning.
    """
    home = [100.0, 120.0, 100.0, 120.0]
    away = [100.0, 100.0, 120.0, 120.0]     # every combination once -> corr = 0
    j = joint_metrics([_record(1, home_pts=home, away_pts=away)])
    assert j["corr_home_away"] == pytest.approx(0.0, abs=1e-12)
    assert j["margin_sd"] == pytest.approx(j["total_sd"])
    assert j["margin_sd"] == pytest.approx(math.sqrt(2) * j["side_pts_sd"])


def test_pace_sd_is_the_spread_of_the_two_sides_mean_pace():
    j = joint_metrics([_record(1, home_pts=[100.0, 100.0], away_pts=[100.0, 100.0],
                               home_pace=[96.0, 104.0], away_pace=[96.0, 104.0])])
    assert j["pace_sd"] == pytest.approx(4.0)


def test_a_record_written_before_the_per_sim_vectors_is_not_measured_rather_than_zero():
    """Old runs must read as absent, not as a model with perfectly uncorrelated teams."""
    j = joint_metrics([_record(1, home_pts=[100.0, 110.0], away_pts=[100.0, 105.0],
                               per_sim=False)])
    assert j["n_games"] == 0
    assert j["corr_home_away"] == 0.0


def test_a_single_sim_carries_no_joint_information_and_is_skipped():
    """One sim has no spread to correlate. Counting it would divide by zero or report nonsense."""
    assert joint_metrics([_record(1, home_pts=[100.0], away_pts=[99.0])])["n_games"] == 0


def test_every_per_sim_key_is_required_before_a_game_counts():
    """A partially backfilled record must not be half-measured."""
    rec = _record(1, home_pts=[100.0, 110.0], away_pts=[100.0, 105.0])
    del rec[PER_SIM_KEYS[-1]]
    assert joint_metrics([rec])["n_games"] == 0


# --------------------------------------------------------------------------- coverage

def test_coverage_counts_an_actual_inside_one_sd_as_a_hit_at_both_thresholds():
    recs = [_player_record(1, mu=20.0, sd=5.0, actual=24.0)]     # z = 0.8
    cov = coverage_metrics(recs, stats=("pts",))
    assert cov["player"]["pts"]["1"] == pytest.approx(1.0)
    assert cov["player"]["pts"]["2"] == pytest.approx(1.0)


def test_an_actual_between_one_and_two_sd_is_a_hit_only_at_two():
    recs = [_player_record(1, mu=20.0, sd=5.0, actual=27.5)]     # z = 1.5
    cov = coverage_metrics(recs, stats=("pts",))
    assert cov["player"]["pts"]["1"] == pytest.approx(0.0)
    assert cov["player"]["pts"]["2"] == pytest.approx(1.0)


def test_a_player_the_sims_are_unanimous_about_leaves_the_denominator():
    """sd == 0 is no distribution to test. Counting him would inflate every coverage number."""
    recs = [_player_record(1, mu=20.0, sd=5.0, actual=21.0),
            _player_record(2, mu=0.0, sd=0.0, actual=0.0)]
    cov = coverage_metrics(recs, stats=("pts",))
    assert cov["n_player_games"] == 1
    assert cov["player"]["pts"]["1"] == pytest.approx(1.0)


def test_the_margin_dispersion_ratio_is_predicted_sd_over_realised_residual_sd():
    """The number a shrinkage dial would be fitted to -- so it has to be exactly this ratio."""
    recs = [_record(1, home_pts=[110.0, 90.0], away_pts=[100.0, 100.0], actual_margin=10),
            _record(2, home_pts=[110.0, 90.0], away_pts=[100.0, 100.0], actual_margin=-10)]
    cov = coverage_metrics(recs, stats=("pts",))
    margin = cov["margin"]
    assert margin["pred_sd"] == pytest.approx(10.0)
    # Residuals are +10 and -10 about a predicted mean of 0; sample sd (ddof=1) is sqrt(200) ~ 14.14
    assert margin["resid_sd"] == pytest.approx(math.sqrt(200.0))
    assert margin["dispersion_ratio"] == pytest.approx(10.0 / math.sqrt(200.0))


def test_a_stat_whose_spread_is_not_stored_is_refused_rather_than_reported_as_zero():
    """"reb" is the live trap: the mean is derivable, the sd is not without the oreb/dreb
    covariance. Silently returning 0.0% coverage would read as a catastrophic model failure."""
    with pytest.raises(ValueError, match="reb"):
        coverage_metrics([_player_record(1, mu=1.0, sd=1.0, actual=1.0)], stats=("reb",))


def test_the_default_coverage_stats_are_all_actually_stored():
    """Guards the default list itself against the same mistake."""
    coverage_metrics([_player_record(1, mu=1.0, sd=1.0, actual=1.0)], stats=COVERAGE_STATS)


# --------------------------------------------------------------------------- Brier SE

def test_the_brier_standard_error_is_the_sample_sd_of_the_per_game_briers():
    probs = [1.0, 0.0, 1.0, 0.0]
    outcomes = [True, True, True, True]          # per-game Brier: 0, 1, 0, 1
    w = win_metrics(probs, outcomes, [True, False, True, False])
    assert w["brier"] == pytest.approx(0.5)
    # sample sd of (0,1,0,1) is sqrt(1/3); SE divides by sqrt(4)
    assert w["brier_se"] == pytest.approx(math.sqrt(1 / 3) / 2)


def test_a_single_game_has_no_standard_error_rather_than_a_nan():
    assert win_metrics([0.6], [True], [True])["brier_se"] == 0.0


def test_an_empty_run_still_reports_the_standard_error_key():
    """Callers read headline['brier_se'] unconditionally; a missing key would be a KeyError."""
    assert win_metrics([], [], [])["brier_se"] == 0.0


# --------------------------------------------------------------------------- paired Brier

def test_the_paired_comparison_uses_only_the_games_two_runs_share():
    a = [_record(1, home_pts=[1.0, 2.0], away_pts=[1.0, 1.0], win_prob=1.0, actual_margin=1),
         _record(2, home_pts=[1.0, 2.0], away_pts=[1.0, 1.0], win_prob=1.0, actual_margin=1)]
    b = [_record(2, home_pts=[1.0, 2.0], away_pts=[1.0, 1.0], win_prob=0.0, actual_margin=1),
         _record(3, home_pts=[1.0, 2.0], away_pts=[1.0, 1.0], win_prob=0.0, actual_margin=1)]
    assert paired_brier(a, b)["n"] == 1     # only game 2 is in both


def test_two_identical_runs_differ_by_exactly_zero_with_no_uncertainty():
    """The pairing is the point: unpaired SEs would report a spurious band here."""
    recs = [_record(g, home_pts=[1.0, 2.0], away_pts=[1.0, 1.0],
                    win_prob=0.7, actual_margin=1) for g in (1, 2, 3, 4)]
    pb = paired_brier(recs, list(recs))
    assert pb["diff"] == pytest.approx(0.0)
    assert pb["diff_se"] == pytest.approx(0.0)
    assert pb["z"] == 0.0


def test_fewer_than_two_shared_games_reports_no_comparison_rather_than_dividing_by_zero():
    a = [_record(1, home_pts=[1.0, 2.0], away_pts=[1.0, 1.0])]
    b = [_record(9, home_pts=[1.0, 2.0], away_pts=[1.0, 1.0])]
    assert paired_brier(a, b) == {"n": 0, "brier_a": 0.0, "brier_b": 0.0,
                                  "diff": 0.0, "diff_se": 0.0, "z": 0.0}
