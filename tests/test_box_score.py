"""
Box-score decoder tests.

These exercise the cleaned-data event semantics that the legacy notebook got wrong, plus
minutes accrual and the shared game-split partitioning. They import the decoder directly
(no TensorFlow / trained model needed), so they run fast on CPU.
"""
from __future__ import annotations

import pytest

from data_loading import split_games
from simulation.box_score import (
    generate_box_score, period_box_scores, split_by_period, side_membership,
)

HOME = ["A", "B", "C", "D", "E"]
AWAY = ["F", "G", "H", "I", "J"]


def _row(time, event, player, type_, result, secondary="none",
         home=HOME, away=AWAY):
    return {
        "event": event, "player": player, "type": type_, "result": result,
        "secondary_player": secondary, "time": time,
        "roster_home": list(home), "roster_away": list(away),
    }


def _by_name(box):
    return {pl.player: pl for pl in (*box.home, *box.away)}


def test_scoring_and_shooting_splits():
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "shot", "A", "paint", "made"),      # +2 home
        _row(20, "shot", "A", "paint", "missed"),    # FGA only
        _row(30, "shot", "B", "top3", "made"),      # +3 home
        _row(40, "shot", "F", "top3", "missed"),    # away FGA/3PA only
        _row(50, "shot", "F", "free throw", "made"),    # +1 away
        _row(60, "shot", "F", "free throw", "missed"),  # FTA only
        _row(70, "end", "end", "end", "end"),
    ]
    box = generate_box_score(events)
    p = _by_name(box)

    assert (p["A"].fgm, p["A"].fga, p["A"].pts) == (1, 2, 2)
    assert (p["B"].tpm, p["B"].tpa, p["B"].fgm, p["B"].fga, p["B"].pts) == (1, 1, 1, 1, 3)
    assert (p["F"].tpa, p["F"].tpm) == (1, 0)
    assert (p["F"].ftm, p["F"].fta, p["F"].pts) == (1, 2, 1)
    assert box.home_score == 5
    assert box.away_score == 1
    # Final score equals the sum of each side's player points.
    assert box.home_score == sum(pl.pts for pl in box.home)
    assert box.away_score == sum(pl.pts for pl in box.away)


def test_a_steal_is_one_row_crediting_both_players():
    # One turnover row: A lost the ball, F took it. The stealer rides in secondary_player.
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "turnover", "A", "steal", "cop", secondary="F"),
        _row(20, "end", "end", "end", "end"),
    ]
    p = _by_name(generate_box_score(events))
    assert (p["A"].tov, p["A"].stl) == (1, 0)
    assert (p["F"].stl, p["F"].tov) == (1, 0)


def test_an_offensive_foul_counts_as_a_turnover():
    # No trailing turnover row is emitted, so the box score counts it from the foul.
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "foul", "A", "offensive", "cop"),
        _row(20, "end", "end", "end", "end"),
    ]
    p = _by_name(generate_box_score(events))
    assert (p["A"].tov, p["A"].pf) == (1, 1)


def test_non_steal_turnover_counts():
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "turnover", "A", "violation", "cop"),
        _row(20, "end", "end", "end", "end"),
    ]
    p = _by_name(generate_box_score(events))
    assert (p["A"].tov, p["A"].stl) == (1, 0)


def test_blocked_shot():
    # The shooter's row carries result="blocked" (a missed FGA); the blocker gets a block.
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "shot", "A", "paint", "blocked"),
        _row(10, "block", "F", "paint", "block", secondary="A"),
        _row(20, "end", "end", "end", "end"),
    ]
    p = _by_name(generate_box_score(events))
    assert (p["A"].fga, p["A"].fgm, p["A"].pts) == (1, 0, 0)
    assert p["F"].blk == 1


def test_rebounds_assists_fouls():
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "rebound", "A", "offensive", "null"),
        _row(20, "rebound", "B", "defensive", "cop"),
        _row(30, "assist", "C", "paint", "score"),
        _row(40, "foul", "D", "shooting 2pt", "free throw"),
        _row(50, "foul", "E", "technical", "free throw"),  # technical: NOT a personal foul
        _row(60, "end", "end", "end", "end"),
    ]
    p = _by_name(generate_box_score(events))
    assert (p["A"].oreb, p["A"].reb) == (1, 1)
    assert (p["B"].dreb, p["B"].reb) == (1, 1)
    assert p["C"].ast == 1
    assert p["D"].pf == 1
    assert p["E"].pf == 0


def test_minutes_accrual():
    # 120s of clock with the same five on court => 2.0 minutes each.
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(60, "shot", "A", "paint", "made"),
        _row(120, "shot", "F", "paint", "made"),
        _row(120, "end", "end", "end", "end"),
    ]
    p = _by_name(generate_box_score(events))
    assert p["A"].minutes == 2.0
    assert p["F"].minutes == 2.0


def test_split_games_disjoint_and_deterministic():
    ids = list(range(100))
    train, test, holdout = split_games(ids, seed=42, test_frac=0.2, holdout_frac=0.1)
    assert len(train) == 70 and len(test) == 20 and len(holdout) == 10
    assert train.isdisjoint(test) and train.isdisjoint(holdout) and test.isdisjoint(holdout)
    assert train | test | holdout == set(ids)
    # Same seed reproduces the exact partition (so every model + the box-score tool agree).
    assert split_games(ids, seed=42, test_frac=0.2, holdout_frac=0.1) == (train, test, holdout)


# ---------------------------------------------------------------------------
# Period slicing
# ---------------------------------------------------------------------------

Q = 720          # PERIOD_LENGTH; imported below rather than restated in each test
REG = 4 * Q


def _sum_over(boxes, side, field):
    return sum(getattr(pl, field) for b in boxes
               for pl in (b.home if side == "home" else b.away))


def test_split_by_period_cuts_on_the_quarter_boundaries():
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "shot", "A", "paint", "made"),
        _row(Q + 5, "shot", "B", "paint", "made"),
        _row(2 * Q + 5, "shot", "C", "paint", "made"),
        _row(3 * Q + 5, "shot", "D", "paint", "made"),
    ]
    got = [(p, len(rows)) for p, rows, _ in split_by_period(events)]
    assert got == [(0, 2), (1, 1), (2, 1), (3, 1)]


def test_the_seed_row_is_the_previous_periods_last_row():
    events = [
        _row(10, "shot", "A", "paint", "made"),
        _row(700, "shot", "A", "paint", "missed"),
        _row(Q + 5, "shot", "B", "paint", "made"),
    ]
    slices = split_by_period(events)
    assert slices[0][2] is None                       # nothing precedes the first period
    assert slices[1][2]["time"] == 700


def test_a_buzzer_row_does_not_invent_a_fifth_period():
    """period_index is half-open, so a clock exactly on a boundary opens the next period.

    Right for the game state, wrong for an event: a shot at 0.0 is a buzzer-beater belonging to
    the quarter it ended. Every regulation game has such rows -- the final shot and the `end`
    sentinel both sit at 2880 -- so left alone this invents an overtime in every game.
    """
    events = [
        _row(10, "shot", "A", "paint", "made"),
        _row(REG - 2, "shot", "B", "paint", "missed"),
        _row(REG, "shot", "C", "top3", "made"),       # the buzzer-beater
        _row(REG, "end", "end", "end", "end"),
    ]
    periods = [p for p, _, _ in split_by_period(events)]
    assert periods == [0, 3]
    boxes = period_box_scores(events)
    assert 4 not in boxes
    assert boxes[3].home_score == 3                   # the buzzer three counts in Q4


def test_a_real_overtime_still_gets_its_own_period():
    events = [
        _row(10, "shot", "A", "paint", "made"),
        _row(REG + 60, "shot", "B", "paint", "made"),   # 1:00 into OT1
        _row(REG + 300, "end", "end", "end", "end"),
    ]
    assert [p for p, _, _ in split_by_period(events)] == [0, 4]


def test_period_boxes_sum_to_the_whole_game_box():
    """The independent number: two routes to one quantity, over every counting stat."""
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "shot", "A", "paint", "made"),
        _row(30, "rebound", "F", "defensive", "cop"),
        _row(Q + 5, "shot", "B", "top3", "made"),
        _row(Q + 40, "foul", "G", "personal", "nothing"),
        _row(2 * Q + 5, "turnover", "A", "steal", "cop", secondary="F"),
        _row(3 * Q + 5, "shot", "C", "free throw", "made"),
        _row(REG, "end", "end", "end", "end"),
    ]
    whole = generate_box_score(events)
    parts = list(period_box_scores(events).values())
    for side in ("home", "away"):
        for field in ("pts", "fga", "fgm", "tpa", "tpm", "fta", "ftm",
                      "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "seconds"):
            assert _sum_over([whole], side, field) == _sum_over(parts, side, field), \
                f"{side}.{field} does not sum across periods"
    assert whole.home_score == sum(b.home_score for b in parts)
    assert whole.away_score == sum(b.away_score for b in parts)


def test_minutes_across_a_buzzer_are_credited_to_the_lineup_that_played_them():
    """Without a seed row the interval spanning the break is credited to nobody.

    The player substituted off AT the break is the case that bites: he is on the floor for the
    interval, and he appears in no roster snapshot on the far side of it.
    """
    bench = ["K", "B", "C", "D", "E"]
    events = [
        _row(700, "shot", "A", "paint", "made"),               # A on the floor
        _row(Q + 20, "shot", "K", "paint", "made", home=bench),  # A has been replaced by K
    ]
    parts = period_box_scores(events)
    whole = _by_name(generate_box_score(events))
    got = {}
    for b in parts.values():
        for pl in (*b.home, *b.away):
            got[pl.player] = got.get(pl.player, 0.0) + pl.seconds
    assert got["A"] == whole["A"].seconds
    assert got["A"] == 40.0        # 700 -> 740, the whole interval across the buzzer


def test_a_stat_by_a_player_off_the_floor_still_lands_on_his_side():
    """Side membership is a game-level fact; deriving it per period drops the stat.

    Measured over the cleaned corpus before the fix: one game in 1500 lost turnovers and a foul
    this way, because the player fouled at the buzzer while not in any Q4 roster snapshot.
    """
    without = ["K", "B", "C", "D", "E"]           # A is off the floor for the whole last period
    events = [
        _row(10, "shot", "A", "paint", "made"),   # A appears on the home roster in Q1
        _row(3 * Q + 5, "shot", "K", "paint", "made", home=without),
        _row(REG - 1, "foul", "A", "offensive", "cop", home=without),
    ]
    parts = period_box_scores(events)
    last = parts[3]
    assert any(pl.player == "A" and pl.pf == 1 for pl in last.home), \
        "the foul was dropped: A resolved to no side inside the period"
    assert _sum_over(list(parts.values()), "home", "pf") == \
        _sum_over([generate_box_score(events)], "home", "pf")


def test_out_of_order_periods_raise_rather_than_overwrite():
    """Contiguous runs mean a repeated period is a clock fault; silently merging would hide it."""
    events = [
        _row(10, "shot", "A", "paint", "made"),
        _row(Q + 5, "shot", "B", "paint", "made"),
        _row(20, "shot", "C", "paint", "made"),      # back in period 0
    ]
    with pytest.raises(ValueError, match="out of order"):
        period_box_scores(events)


def test_an_empty_game_yields_no_periods():
    assert split_by_period([]) == []
    assert period_box_scores([]) == {}
