"""
Per-head KPI metrics (3.2 W9).

The point of the module is **attribution**: an error in one quantity must move the heads that control it
and no others. A shared scalar punishes a head that did its job for another head's failure, and that
mis-assignment is why a generic score-function estimator needs hundreds of steps to learn anything.

So most of these tests degrade exactly one quantity and assert exactly which heads notice.
"""
import copy

import pytest

from models.head_metrics import (
    EVENT_TOKEN_STATS,
    GAME_SD,
    HEAD_METRICS,
    PROBE_HEADS,
    TEAM_SD,
    count_events,
    head_errors,
    probe_gap,
    total_error,
)


def _side(**over):
    base = {"pts": 114.0, "fga": 88.0, "fgm": 42.0, "tpa": 34.0, "tpm": 12.0, "fta": 23.0,
            "ftm": 18.0, "oreb": 10.0, "dreb": 33.0, "ast": 25.0, "stl": 7.0, "blk": 5.0,
            "tov": 13.0, "pf": 20.0, "efg": 0.546, "tpa_rate": 0.388, "oreb_share": 0.233,
            "possessions": 101.0,
            "players": {"A": {"pts": 30.0, "fga": 20.0, "ast": 5.0, "reb": 6.0, "minutes": 36.0},
                        "B": {"pts": 20.0, "fga": 15.0, "ast": 8.0, "reb": 4.0, "minutes": 32.0},
                        "C": {"pts": 64.0, "fga": 53.0, "ast": 12.0, "reb": 33.0, "minutes": 172.0}}}
    base.update(over)
    return base


def _stats(**over):
    home, away = _side(), _side()
    return {"sides": {"home": home, "away": away},
            "counts": {"substitutions": 54.0, "timeouts": 11.0},
            "margin": 0.0, "counted": True, **over}


def _degrade(real, side_changes=None, **top):
    sim = copy.deepcopy(real)
    for key, delta in (side_changes or {}).items():
        for side in ("home", "away"):
            sim["sides"][side][key] = sim["sides"][side][key] + delta
    for key, value in top.items():
        sim[key] = value
    return sim


# --------------------------------------------------------------------------- the baseline

def test_a_perfect_sim_scores_zero_on_every_head():
    real = _stats()
    errors = head_errors(real, real)
    assert set(errors) == set(HEAD_METRICS)
    assert max(errors.values()) == pytest.approx(0.0)


def test_every_trained_head_has_a_metric():
    """A head with no metric would be invisible to the replay pass, which is a silent no-op for it."""
    from models.registry import STAGE_MODEL_KEYS
    assert set(HEAD_METRICS) == set(STAGE_MODEL_KEYS)


# --------------------------------------------------------------------------- attribution

def test_an_efficiency_error_moves_only_the_shooting_head():
    real = _stats()
    errors = head_errors(real, _degrade(real, {"efg": 0.10}))
    moved = {h for h, v in errors.items() if v > 1e-9}
    assert moved == {"shot_result"}


def test_a_shot_mix_error_moves_only_the_type_head():
    """``shot_type`` controls the mix, and must never be scored on efficiency.

    The cheapest way to win an efficiency metric by changing the mix is to shoot more threes, because
    eFG counts them at 1.5 -- so scoring the mix head on eFG rewards exactly the distortion it is meant
    to prevent.
    """
    real = _stats()
    errors = head_errors(real, _degrade(real, {"tpa_rate": 0.10}))
    moved = {h for h, v in errors.items() if v > 1e-9}
    assert moved == {"shot_type"}
    assert "efg" not in HEAD_METRICS["shot_type"]


def test_a_foul_count_error_moves_both_the_head_that_produces_and_the_head_that_types():
    """``event_time`` emits the ``foul`` token; ``foul_type`` decides what kind. Both own part of it."""
    real = _stats()
    errors = head_errors(real, _degrade(real, {"pf": 6.0}))
    moved = {h for h, v in errors.items() if v > 1e-9}
    assert moved == {"event_time", "foul_type"}


def test_a_rebound_split_error_moves_only_the_rebound_head():
    real = _stats()
    errors = head_errors(real, _degrade(real, {"oreb_share": 0.08}))
    assert {h for h, v in errors.items() if v > 1e-9} == {"rebound_type"}


def test_a_pace_error_moves_the_conditional_time_head():
    real = _stats()
    errors = head_errors(real, _degrade(real, {"possessions": 8.0}))
    assert {h for h, v in errors.items() if v > 1e-9} == {"event_time_cond"}


def test_a_substitution_count_error_moves_the_event_head():
    real = _stats()
    sim = copy.deepcopy(real)
    sim["counts"]["substitutions"] = 80.0
    assert {h for h, v in head_errors(real, sim).items() if v > 1e-9} == {"event_time"}


def test_a_timeout_count_error_moves_both_heads_that_own_timeouts():
    real = _stats()
    sim = copy.deepcopy(real)
    sim["counts"]["timeouts"] = 20.0
    assert {h for h, v in head_errors(real, sim).items() if v > 1e-9} == {"event_time", "timeout_team"}


def test_a_minutes_error_moves_only_the_rotation_heads():
    """And **not** the player head, which is the separation the share metric buys.

    A man playing the wrong number of minutes is a rotation failure. His share of his team's points is a
    different question, and scoring the player head on minutes would charge it for the substitution
    heads' work -- exactly the mis-assignment per-head metrics exist to remove. Minutes enter the player
    head's number only as the REAL weight, deciding whose share matters, never as a quantity it is
    judged on.
    """
    real = _stats()
    sim = copy.deepcopy(real)
    for side in ("home", "away"):
        sim["sides"][side]["players"]["A"]["minutes"] = 12.0
    moved = {h for h, v in head_errors(real, sim).items() if v > 1e-9}
    assert moved == {"substitution", "sub_decision"}


def test_a_share_error_moves_the_player_head_and_not_the_rotation_heads():
    """The other side of the same separation: who scored, with the minutes unchanged."""
    real = _stats()
    sim = copy.deepcopy(real)
    for side in ("home", "away"):
        sim["sides"][side]["players"]["A"]["pts"] = 10.0
        sim["sides"][side]["players"]["B"]["pts"] = 40.0
    moved = {h for h, v in head_errors(real, sim).items() if v > 1e-9}
    assert moved == {"player"}


# --------------------------------------------------------------------------- normalisation

def test_errors_are_expressed_in_standard_deviations():
    """One sd of error on any stat reads as roughly 1.0, which is what makes stats commensurable.

    Raw averaging would let points (sd 12.1 a team-game) drown blocks (sd 2.5).
    """
    real = _stats()
    one_sd = head_errors(real, _degrade(real, {"efg": TEAM_SD["efg"]}))["shot_result"]
    assert one_sd == pytest.approx(1.0, abs=1e-6)


def test_the_shared_margin_term_reaches_every_head():
    """The winner is a joint product of all twelve, so every head carries a little of it.

    Per SIM the shared term is the margin error; a Brier needs a distribution over sims and so belongs
    to the aggregate rather than to one game-sim.
    """
    real = _stats()
    errors = head_errors(real, _degrade(real, margin=GAME_SD["margin"]))
    assert len({round(v, 9) for v in errors.values()}) == 1, "all heads carry the same shared term"
    assert all(v > 0 for v in errors.values())


def test_the_probes_reach_only_the_rotation_adjacent_heads():
    """The one place a behaviour rather than a count enters a head's score."""
    real = _stats()
    probes = {"rows": [{"sim": 0.238, "real": 0.776}]}
    errors = head_errors(real, real, probes=probes)
    moved = {h for h, v in errors.items() if v > 1e-9}
    assert moved == set(PROBE_HEADS)


def test_a_probe_that_could_not_be_computed_contributes_nothing():
    """Absent is not agreement, and a zero real value is not a 100% gap."""
    assert probe_gap(None) == 0.0
    assert probe_gap({"rows": []}) == 0.0
    assert probe_gap({"rows": [{"sim": 1.0, "real": None}]}) == 0.0
    assert probe_gap({"rows": [{"sim": 1.0, "real": 0.0}]}) == 0.0


# --------------------------------------------------------------------------- the vocabulary contract

def test_the_event_tokens_cover_the_heads_own_output_and_nothing_else():
    """Three exclusions follow from the token vocabulary, not from preference."""
    assert "free throw" not in EVENT_TOKEN_STATS, (
        "there is no free-throw event token; attempts come from the rules engine downstream of a "
        "shooting foul, so scoring the head on them charges it for a rule it does not control")
    assert "steal" not in EVENT_TOKEN_STATS, "a steal is turnover_type's sub-type split"
    assert EVENT_TOKEN_STATS["substitution"] == ("substitutions",), (
        "counted, not converted to minutes: the head controls how often an opportunity fires")
    produced = {s for stats in EVENT_TOKEN_STATS.values() for s in stats}
    assert produced <= set(HEAD_METRICS["event_time"])


def test_free_throws_are_not_scored_against_the_event_head():
    real = _stats()
    errors = head_errors(real, _degrade(real, {"fta": 10.0, "ftm": 8.0}))
    assert errors["event_time"] == pytest.approx(0.0), "fta/ftm are the rules engine's, not the head's"


def test_counted_tokens_abstain_when_there_is_no_play_by_play():
    """Scoring a missing count as zero would read as perfect agreement."""
    real = _stats()
    sim = _stats(counted=False)
    sim["counts"] = {}
    assert head_errors(real, sim)["timeout_team"] == pytest.approx(0.0)


def test_count_events_reads_the_two_tokens_that_are_not_box_stats():
    rows = [{"event": "substitution"}, {"event": "timeout"}, {"event": "shot"},
            {"event": "substitution"}]
    assert count_events(rows) == {"substitutions": 2, "timeouts": 1}


# --------------------------------------------------------------------------- the scalar

def test_the_total_is_the_mean_of_the_heads():
    real = _stats()
    sim = _degrade(real, {"efg": 0.10, "pf": 6.0})
    errors = head_errors(real, sim)
    assert total_error(real, sim) == pytest.approx(sum(errors.values()) / len(errors))


def test_a_perfect_sim_totals_zero():
    real = _stats()
    assert total_error(real, real) == pytest.approx(0.0)
