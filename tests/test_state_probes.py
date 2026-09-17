"""
The three game-state behaviour probes, on hand-built rows.

Pure pandas, no TF, no simulation -- the probes read play-by-play and nothing else, which is what
lets them score the simulator and reality with the same function. Rows are built by hand so each
expected value is countable by eye, in the style of tests/test_diagnostics.py.

The thing most worth guarding here is the *denominator*. Every one of these probes is a rate, and
each has a way to look good by dropping the inconvenient half of its sample: a player who never
comes off, a game that never reaches Q4, a stretch of clock spent in the state with no foul in it.
Several tests below exist only to pin those down.
"""
from __future__ import annotations

import pytest

from reporting.state_probes import (
    BLOWOUT_MARGIN,
    FOUL_TROUBLE_WITHIN,
    HALFTIME_SECONDS,
    accumulate,
    foul_trouble_events,
    late_foul_state,
    probe_game,
    q4_starter_seconds,
    summarize,
    _blank,
)

HOME = ["H1", "H2", "H3", "H4", "H5"]
AWAY = ["A1", "A2", "A3", "A4", "A5"]


def _row(time, event="shot", player="H1", type_="rim", result="missed",
         home=None, away=None, secondary=""):
    """A filler row. Deliberately a MISS: a scoring default would move the margin in every fixture
    that is testing something else, which is how three of these tests were wrong the first time."""
    return {
        "game_id": 1, "time": time, "event": event, "player": player,
        "type": type_, "result": result, "secondary_player": secondary,
        "roster_home": list(home if home is not None else HOME),
        "roster_away": list(away if away is not None else AWAY),
        "home/away": "home", "season": 2023, "playoff": 0,
    }


def _fouls(player, times, **kw):
    """Rows that charge ``player`` a personal foul at each time."""
    return [_row(t, event="foul", player=player, type_="shooting", result="", **kw) for t in times]


# --------------------------------------------------------------------------- probe A

def test_a_player_off_the_floor_within_the_window_counts_as_benched():
    bench = [n for n in HOME if n != "H1"] + ["H6"]
    rows = [_row(0), *_fouls("H1", [100, 200, 300]),
            _row(330, home=bench[:5])]          # H1 off, 30 s after his third
    assert foul_trouble_events(rows)[3] == [True]


def test_a_player_who_comes_off_too_late_counts_as_not_benched():
    bench = [n for n in HOME if n != "H1"] + ["H6"]
    rows = [_row(0), *_fouls("H1", [100, 200, 300]),
            _row(300 + FOUL_TROUBLE_WITHIN + 1, home=bench[:5])]
    assert foul_trouble_events(rows)[3] == [False]


def test_a_player_who_never_comes_off_still_counts_rather_than_vanishing():
    """The denominator test. Dropping him would score only the players who were benched, which is
    exactly the population that makes a simulator look like it manages foul trouble."""
    rows = [_row(0), *_fouls("H1", [100, 200, 300]), _row(1000)]
    assert foul_trouble_events(rows)[3] == [False]


def test_a_third_foul_after_halftime_is_not_foul_trouble():
    bench = [n for n in HOME if n != "H1"] + ["H6"]
    rows = [_row(0), *_fouls("H1", [100, 200, HALFTIME_SECONDS + 10]),
            _row(HALFTIME_SECONDS + 20, home=bench[:5])]
    assert foul_trouble_events(rows)[3] == []


def test_technical_and_offensive_fouls_do_not_advance_the_personal_count():
    """Mirrors LineupScan's own rule, so the probe and the model's foul feature cannot drift."""
    rows = [_row(0),
            *_fouls("H1", [100, 200]),
            _row(250, event="foul", player="H1", type_="technical", result=""),
            _row(1000)]
    assert foul_trouble_events(rows)[3] == []


def test_the_fourth_foul_is_tracked_separately_from_the_third():
    """One substitution resolves both pending observations, and they get different answers.

    He takes his third at 300 and his fourth at 400, and comes off at 420 -- 120 s after the third
    (too slow) and 20 s after the fourth (in time). A pending map keyed by player alone would keep
    only the third and report the fourth as never having happened.
    """
    bench = [n for n in HOME if n != "H1"] + ["H6"]
    rows = [_row(0), *_fouls("H1", [100, 200, 300, 400]), _row(420, home=bench[:5])]
    events = foul_trouble_events(rows)
    assert events[3] == [False] and events[4] == [True]


# --------------------------------------------------------------------------- probe B

# Q4 runs [2160, 2880): 2880 is the first second of overtime, not the last of regulation.
Q4_START, Q4_LAST = 2160, 2879


def _through_q4(margin_maker):
    """Rows reaching Q4 with a chosen margin, then a full quarter with the starters on."""
    rows = [_row(0)]
    rows += margin_maker
    # A row at the end of Q3, so the Q4 period box credits only Q4's own interval rather than the
    # whole empty stretch since the last event.
    rows.append(_row(Q4_START - 1))
    rows.append(_row(Q4_START))
    rows.append(_row(Q4_LAST))
    return rows


def test_q4_starter_seconds_come_from_the_period_box_and_the_margin_is_read_at_the_buzzer():
    # Six made rim shots by the home team before Q4 -> +12.
    scoring = [_row(100 + 10 * i, player="H1", type_="rim", result="made") for i in range(6)]
    rows = _through_q4(scoring)
    result = q4_starter_seconds(rows)
    assert result is not None
    seconds, margin = result
    assert margin == 12
    # A full 12:00 quarter for each of the ten. period_box_scores credits from the period
    # boundary, not from the quarter's first event row, which is why this is 720 and not 719.
    assert seconds == pytest.approx(10 * 720.0)


def test_a_game_that_never_reaches_the_fourth_quarter_reports_nothing():
    """Better than a zero: a truncated sim has no Q4 rotation to judge."""
    assert q4_starter_seconds([_row(0), _row(100), _row(500)]) is None


def test_a_game_with_no_full_lineup_reports_nothing_rather_than_empty_starters():
    rows = [dict(_row(t), roster_home=["H1"], roster_away=["A1"])
            for t in (0, Q4_START, Q4_LAST)]
    assert q4_starter_seconds(rows) is None


def test_a_blowout_and_a_close_game_land_in_different_buckets():
    pool = _blank()
    accumulate(pool, {"foul_trouble": {3: [], 4: []},
                      "q4": {"seconds": 1000.0, "margin": BLOWOUT_MARGIN},
                      "late": {"fouls": 0, "seconds": 0.0}})
    accumulate(pool, {"foul_trouble": {3: [], 4: []},
                      "q4": {"seconds": 2000.0, "margin": BLOWOUT_MARGIN - 1},
                      "late": {"fouls": 0, "seconds": 0.0}})
    out = summarize(pool)["blowout_q4"]
    assert out["blowout_games"] == 1 and out["close_games"] == 1
    assert out["ratio"] == pytest.approx(0.5)
    assert out["blowout_frequency"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- probe C

def _lead(points):
    """Home-team made rim shots worth ``points`` (a multiple of 2)."""
    return [_row(100 + 10 * i, player="H1", type_="rim", result="made")
            for i in range(points // 2)]


def test_the_trailing_teams_foul_inside_the_window_is_counted():
    # Home leads by 6 going into the last 2:00, so AWAY is the trailing team.
    rows = [_row(0), *_lead(6),
            _row(2770),                                   # in state: Q4, 110 s left, diff +6
            _row(2790, event="foul", player="A1", type_="shooting", result=""),
            _row(Q4_LAST)]
    fouls, seconds = late_foul_state(rows)
    assert fouls == 1
    assert seconds == pytest.approx(Q4_LAST - 2770)


def test_the_leading_teams_foul_inside_the_window_is_not_counted():
    rows = [_row(0), *_lead(6), _row(2770),
            _row(2790, event="foul", player="H1", type_="shooting", result=""),
            _row(Q4_LAST)]
    assert late_foul_state(rows)[0] == 0


def test_a_deficit_outside_the_band_is_not_the_state():
    """Down 2 you defend; down 15 you have given up. The band is the part that is a strategy."""
    rows = [_row(0), *_lead(2), _row(2770),
            _row(2790, event="foul", player="A1", type_="shooting", result=""),
            _row(Q4_LAST)]
    fouls, seconds = late_foul_state(rows)
    assert fouls == 0 and seconds == 0.0


def test_clock_spent_in_the_state_with_no_foul_still_counts_toward_the_denominator():
    """Otherwise the rate is fouls-per-foul and always 100%."""
    rows = [_row(0), *_lead(6), _row(2770), _row(Q4_LAST)]
    fouls, seconds = late_foul_state(rows)
    assert fouls == 0 and seconds == pytest.approx(Q4_LAST - 2770)


# --------------------------------------------------------------------------- the whole thing

def test_probe_game_returns_all_three_blocks_for_an_ordinary_game():
    rows = _through_q4([_row(100 + 10 * i, player="H1", type_="rim", result="made")
                        for i in range(6)])
    out = probe_game(rows)
    assert set(out) == {"foul_trouble", "q4", "late"}
    assert out["q4"]["margin"] == 12


def test_summarize_reports_none_rather_than_zero_when_nothing_qualified():
    """A rate with an empty denominator is not 0.0 -- it is unmeasured, and must read that way."""
    out = summarize(_blank())
    assert out["foul_trouble_4"]["p_benched"] is None
    assert out["blowout_q4"]["ratio"] is None
    assert out["late_foul"]["rate_per_100s"] is None
