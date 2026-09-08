"""
Tests for the game-state feature pipeline (the TF-free derivation + normalization layer).

Covers ``models.game_state_features``:
  - ``derive_game_state``: inclusive per-row running score (home−away and total), period
    index + seconds-left-in-period from the clock, and per-period team-foul counts that
    reset at each period boundary, with the scoring/fouling team resolved by roster
    membership — the same row shape the cleaned data and ``GameSimulator.history`` share.
  - Normalization by fixed constants (no train-fit stats).
  - ``merge_game_state_features`` aligns per-row arrays positionally over a multi-game frame.
  - Parity: the derived running state at the last row equals what a straight scan of the
    same event stream (box-score scoring semantics) produces.

The model graph + simulator inference are exercised by the TF-dependent suites (run on the
training box); these tests need no TensorFlow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models import game_state_features as gs


HOME = ["H1", "H2", "H3", "H4", "H5"]
AWAY = ["A1", "A2", "A3", "A4", "A5"]


def _row(event, player, time, *, type="none", result="none",
         home=None, away=None):
    return {
        "event": event, "player": player, "type": type, "result": result,
        "secondary_player": "none", "time": float(time),
        "roster_home": list(home or HOME), "roster_away": list(away or AWAY),
    }


# ---------------------------------------------------------------------------
# derive_game_state — running score
# ---------------------------------------------------------------------------

def test_running_score_is_inclusive_and_team_resolved_by_roster():
    rows = [
        _row("start", "start", 0.0),
        _row("shot", "H1", 10.0, type="paint", result="made"),    # home +2
        _row("shot", "A1", 20.0, type="top3", result="made"),    # away +3
        _row("shot", "H2", 30.0, type="paint", result="missed"),  # no points
        _row("shot", "A2", 40.0, type="free throw", result="made"),  # away +1
    ]
    out = gs.derive_game_state(rows)

    # score_diff is inclusive of each row's own event (matches the live controller score).
    assert list(out["score_diff"]) == [0, 2, -1, -1, -2]
    assert list(out["score_total"]) == [0, 2, 5, 5, 6]


def test_missed_and_nonscoring_events_do_not_move_score():
    rows = [
        _row("shot", "H1", 5.0, type="paint", result="missed"),
        _row("rebound", "H2", 6.0, type="offensive"),
        _row("turnover", "H3", 7.0, result="cop"),
        _row("assist", "H4", 8.0),
    ]
    out = gs.derive_game_state(rows)
    assert list(out["score_total"]) == [0, 0, 0, 0]
    assert list(out["score_diff"]) == [0, 0, 0, 0]


# ---------------------------------------------------------------------------
# derive_game_state — period index + time-left
# ---------------------------------------------------------------------------

def test_period_index_and_time_left_across_quarters_and_ot():
    rows = [
        _row("shot", "H1", 0.0),        # Q1
        _row("shot", "H1", 719.0),      # Q1, 1s left
        _row("shot", "H1", 720.0),      # Q2 start
        _row("shot", "H1", 2880.0),     # OT1 start (regulation = 2880)
        _row("shot", "H1", 3179.0),     # OT1, 1s left
    ]
    out = gs.derive_game_state(rows)
    assert list(out["period_idx"]) == [0, 0, 1, 4, 4]
    assert list(out["period_time_left"]) == [720.0, 1.0, 720.0, 300.0, 1.0]


# ---------------------------------------------------------------------------
# derive_game_state — per-period team fouls
# ---------------------------------------------------------------------------

def test_team_fouls_count_by_side_and_reset_each_period():
    rows = [
        _row("foul", "H1", 100.0, type="personal"),   # home 1
        _row("foul", "A1", 200.0, type="shooting 2pt"),   # away 1
        _row("foul", "H2", 300.0, type="offensive"),  # excluded (offensive)
        _row("foul", "H3", 400.0, type="technical"),  # excluded (technical)
        _row("foul", "H4", 500.0, type="loose ball"), # home 2
        _row("shot", "H1", 800.0, type="paint", result="made"),  # Q2 — fouls reset
        _row("foul", "A2", 850.0, type="personal"),   # away 1 (new period)
    ]
    out = gs.derive_game_state(rows)
    assert list(out["team_fouls_home"]) == [1, 1, 1, 1, 2, 0, 0]
    assert list(out["team_fouls_away"]) == [0, 1, 1, 1, 1, 0, 1]


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

def test_normalize_uses_fixed_constants_and_clips():
    raw = {
        "score_diff": np.array([25.0, -25.0, 100.0], dtype=np.float32),
        "score_total": np.array([220.0, 0.0, 0.0], dtype=np.float32),
        "period_idx": np.array([5.0, 0.0, 0.0], dtype=np.float32),
        "period_time_left": np.array([720.0, 0.0, 0.0], dtype=np.float32),
        "team_fouls_home": np.array([6.0, 0.0, 99.0], dtype=np.float32),
        "team_fouls_away": np.array([6.0, 0.0, 0.0], dtype=np.float32),
        "poss_clock": np.array([24.0, 0.0, 90.0], dtype=np.float32),
    }
    out = gs.normalize_game_state(raw)
    assert np.isclose(out["score_diff"][0], 1.0)
    assert np.isclose(out["score_diff"][1], -1.0)
    assert np.isclose(out["score_diff"][2], 60.0 / 25.0)   # clipped at 60
    assert np.isclose(out["score_total"][0], 1.0)
    assert np.isclose(out["period_idx"][0], 1.0)
    assert np.isclose(out["period_time_left"][0], 1.0)
    assert np.isclose(out["team_fouls_home"][0], 1.0)
    assert np.isclose(out["team_fouls_home"][2], 12.0 / 6.0)  # clipped at 12
    assert np.isclose(out["poss_clock"][0], 1.0)
    assert np.isclose(out["poss_clock"][2], 1.0)              # clipped at the 24s clock
    for k in gs.GAME_STATE_KEYS:
        assert out[k].dtype == np.float32


# ---------------------------------------------------------------------------
# poss_clock — the derived shot-clock proxy
# ---------------------------------------------------------------------------

def _clock(rows):
    return list(gs.derive_game_state(rows)["poss_clock"])


def _ends(rows):
    """Possessions completed over ``rows`` -- what the pace check counts."""
    scan = gs.GameStateScan()
    for r in rows:
        scan.step(r)
    return scan.poss_ends


def test_the_clock_runs_from_the_start_of_the_possession():
    rows = [
        _row("start", "start", 0.0),
        _row("shot", "H1", 8.0, type="paint", result="missed"),
        _row("rebound", "H2", 10.0, type="offensive", result="null"),
    ]
    assert _clock(rows)[:2] == [0.0, 8.0]


def test_a_made_field_goal_ends_the_possession():
    rows = [
        _row("shot", "H1", 10.0, type="paint", result="made"),
        _row("shot", "A1", 22.0, type="top3", result="missed"),
    ]
    # The made shot reports its own possession's length; the next possession starts at 0 and the
    # away miss is 12s into it.
    assert _clock(rows) == [10.0, 12.0]


def test_a_defensive_rebound_ends_the_possession():
    rows = [
        _row("shot", "H1", 10.0, type="paint", result="missed"),
        _row("rebound", "A1", 12.0, type="defensive", result="cop"),
        _row("shot", "A2", 20.0, type="rim", result="missed"),
    ]
    assert _clock(rows) == [10.0, 12.0, 8.0]


def test_an_offensive_rebound_resets_the_clock_without_changing_hands():
    rows = [
        _row("shot", "H1", 10.0, type="paint", result="missed"),
        _row("rebound", "H2", 12.0, type="offensive", result="null"),
        _row("shot", "H3", 18.0, type="rim", result="missed"),
    ]
    # The real 14-second reset: same offense, fresh clock. The second shot is 6s into it.
    assert _clock(rows) == [10.0, 12.0, 6.0]


def test_a_turnover_ends_the_possession():
    rows = [
        _row("turnover", "H1", 14.0, type="steal", result="cop", ),
        _row("shot", "A1", 20.0, type="rim", result="made"),
    ]
    assert _clock(rows) == [14.0, 6.0]


def test_a_made_free_throw_trip_ends_the_possession():
    rows = [
        _row("foul", "A1", 10.0, type="shooting 2pt", result="free throw"),
        _row("shot", "H1", 12.0, type="free throw", result="made"),
        _row("shot", "A2", 20.0, type="rim", result="missed"),
    ]
    # The foul leaves the possession running; the trip ends it, at the made attempt's time.
    assert _clock(rows) == [10.0, 12.0, 8.0]


def test_a_two_shot_trip_is_one_possession_not_two():
    rows = [
        _row("foul", "A1", 10.0, type="shooting 2pt", result="free throw"),
        _row("shot", "H1", 12.0, type="free throw", result="made"),
        _row("shot", "H1", 15.0, type="free throw", result="made"),
        _row("shot", "A2", 22.0, type="rim", result="missed"),
    ]
    # Ending on each made attempt counted the trip twice and read the second at ~0. The trip
    # resolves once, at the LAST made attempt, so the next possession is 22 - 15 = 7s along.
    assert _clock(rows) == [10.0, 12.0, 15.0, 7.0]
    assert _ends(rows) == 1


def test_an_and_one_does_not_end_the_possession_twice():
    rows = [
        _row("shot", "H1", 10.0, type="rim", result="made"),
        _row("foul", "A1", 10.0, type="shooting 2pt", result="free throw"),
        _row("shot", "H1", 14.0, type="free throw", result="made"),
        _row("shot", "A2", 20.0, type="rim", result="missed"),
    ]
    # The basket already ended it, so the foul and the bonus shot are measured from the basket
    # (0s and 4s), and the away team's next possession is dated from the basket too: 10s, not
    # the 6s it would read if the free throw had ended the possession a second time.
    assert _ends(rows) == 1
    assert _clock(rows) == [10.0, 0.0, 4.0, 10.0]


def test_a_technical_trip_leaves_the_ball_where_it_was():
    rows = [
        _row("shot", "H1", 5.0, type="rim", result="missed"),
        _row("rebound", "H2", 7.0, type="offensive", result="null"),
        _row("foul", "A1", 10.0, type="technical", result="free throw"),
        _row("shot", "H1", 12.0, type="free throw", result="made"),
        _row("shot", "H3", 18.0, type="rim", result="missed"),
    ]
    # The shooting team keeps the ball and the shot clock resumes, so the possession that
    # started at the offensive rebound is still running: 18 - 7 = 11s.
    assert _ends(rows) == 0
    assert _clock(rows) == [5.0, 7.0, 3.0, 5.0, 11.0]


def test_a_take_foul_trip_leaves_the_ball_where_it_was():
    rows = [
        _row("foul", "A1", 10.0, type="personal take", result="free throw op"),
        _row("shot", "H1", 12.0, type="free throw", result="made"),
        _row("shot", "H2", 18.0, type="rim", result="missed"),
    ]
    assert _ends(rows) == 0
    assert _clock(rows) == [10.0, 12.0, 18.0]


def test_a_missed_free_throw_leaves_the_rebound_to_decide():
    rows = [
        _row("shot", "H1", 12.0, type="free throw", result="missed"),
        _row("rebound", "A1", 14.0, type="defensive", result="cop"),
        _row("shot", "A2", 20.0, type="rim", result="missed"),
    ]
    assert _clock(rows) == [12.0, 14.0, 6.0]


def test_a_defensive_foul_does_not_end_the_possession():
    rows = [
        _row("shot", "H1", 5.0, type="paint", result="missed"),
        _row("rebound", "H2", 7.0, type="offensive", result="null"),
        _row("foul", "A1", 12.0, type="personal", result="nothing"),
        _row("shot", "H3", 15.0, type="rim", result="made"),
    ]
    # Only the offensive rebound restarts the clock; the common foul leaves it running.
    assert _clock(rows) == [5.0, 7.0, 5.0, 8.0]


def test_an_offensive_foul_ends_the_possession():
    rows = [
        _row("foul", "H1", 10.0, type="offensive", result="cop"),
        _row("shot", "A1", 18.0, type="rim", result="made"),
    ]
    assert _clock(rows) == [10.0, 8.0]


def test_a_period_boundary_starts_a_new_possession():
    rows = [
        _row("shot", "H1", 700.0, type="paint", result="missed"),
        _row("rebound", "H2", 715.0, type="offensive", result="null"),
        _row("shot", "H3", 725.0, type="rim", result="missed"),   # Q2
    ]
    # The Q2 row does not carry the tail of a Q1 possession across the buzzer.
    assert _clock(rows) == [700.0, 715.0, 5.0]


def test_the_boundary_rule_classifies_each_cleaned_row_shape():
    end, reset = gs.POSSESSION_END, gs.POSSESSION_RESET
    cases = [
        (("shot", "paint", "made"), end),
        (("turnover", "steal", "cop"), end),
        (("rebound", "defensive", "cop"), end),
        (("rebound", "team defensive", "cop"), end),
        (("foul", "offensive", "cop"), end),
        (("rebound", "offensive", "null"), reset),
        (("rebound", "team offensive", "null"), reset),
        (("shot", "paint", "missed"), None),
        (("shot", "paint", "blocked"), None),
        # Free throws are resolved by the trip, not the row -- three cases say a made one does
        # not end a possession, and none is visible here. See GameStateScan._resolve_free_throws.
        (("shot", "free throw", "made"), None),
        (("shot", "free throw", "missed"), None),
        (("block", "paint", "block"), None),
        (("assist", "paint", "score"), None),
        (("foul", "shooting 2pt", "free throw"), None),
        (("foul", "personal take", "free throw op"), None),
        (("foul", "personal", "nothing"), None),
        (("foul", "loose ball", "op"), None),
        (("timeout", "home", "none"), None),
        (("substitution", "none", "substitution"), None),
    ]
    for (event, etype, result), expected in cases:
        assert gs.possession_boundary(event, etype, result) == expected, (event, etype, result)


def test_the_incremental_scan_matches_the_batch_derivation():
    """The simulator feeds rows one at a time; preprocessing feeds a whole game."""
    rows = [
        _row("start", "start", 0.0),
        _row("shot", "H1", 9.0, type="paint", result="missed"),
        _row("rebound", "H2", 11.0, type="offensive", result="null"),
        _row("shot", "H3", 19.0, type="rim", result="made"),
        _row("turnover", "A1", 26.0, type="bad pass", result="cop"),
        _row("shot", "H4", 31.0, type="top3", result="made"),
    ]
    scan = gs.GameStateScan()
    incremental = [scan.step(r)[-1] for r in rows]
    assert incremental == _clock(rows)


# ---------------------------------------------------------------------------
# merge_game_state_features — positional alignment over a multi-game frame
# ---------------------------------------------------------------------------

def test_merge_aligns_positionally_across_games():
    g1 = [
        _row("shot", "H1", 10.0, type="top3", result="made"),   # +3 home
        _row("shot", "A1", 20.0, type="paint", result="made"),   # +2 away
    ]
    g2 = [
        _row("shot", "H1", 10.0, type="paint", result="made"),   # +2 home (fresh game)
    ]
    df = pd.DataFrame(
        [{**r, "game_id": 1} for r in g1] + [{**r, "game_id": 2} for r in g2]
    )
    cols = {}
    gs.merge_game_state_features(df, cols)

    # Per-game running score restarts each game (no cross-game leakage).
    assert np.allclose(cols["score_diff"] * 25.0, [3.0, 1.0, 2.0])
    assert np.allclose(cols["score_total"] * 220.0, [3.0, 5.0, 2.0])
    for k in gs.GAME_STATE_KEYS:
        assert cols[k].shape == (3,)


def test_merge_preserves_non_contiguous_game_order():
    # Interleaved game ids with a non-default index — merge must align by position.
    df = pd.DataFrame(
        [
            _row("shot", "H1", 10.0, type="paint", result="made"),  # g1: +2
            _row("shot", "H1", 10.0, type="top3", result="made"),  # g2: +3
            _row("shot", "A1", 20.0, type="paint", result="made"),  # g1: -2 (diff back to 0)
        ]
    )
    df["game_id"] = [1, 2, 1]
    df.index = [100, 200, 300]  # non-contiguous labels
    cols = {}
    gs.merge_game_state_features(df, cols)
    assert np.allclose(cols["score_diff"] * 25.0, [2.0, 3.0, 0.0])


# ---------------------------------------------------------------------------
# parity: derive vs an independent box-score-style scan
# ---------------------------------------------------------------------------

def test_final_score_matches_box_score_scan():
    from simulation.box_score import generate_box_score

    rows = [
        _row("start", "start", 0.0),
        _row("shot", "H1", 10.0, type="paint", result="made"),
        _row("shot", "H2", 20.0, type="top3", result="made"),
        _row("shot", "A1", 30.0, type="paint", result="made"),
        _row("shot", "A2", 40.0, type="free throw", result="made"),
        _row("shot", "H3", 50.0, type="paint", result="missed"),
    ]
    out = gs.derive_game_state(rows)
    box = generate_box_score(rows)

    # The inclusive running diff/total at the last row equals the final box score.
    assert out["score_diff"][-1] == box.home_score - box.away_score
    assert out["score_total"][-1] == box.home_score + box.away_score


def test_every_zone_scores_identically_in_both_scans():
    """The two point lookups must not drift — they are now one function, so prove it.

    A per-zone divergence would desync the trained score feature from the box score silently:
    the model would learn a running score that never happened.
    """
    from simulation.box_score import generate_box_score
    from zones import ZONE_POINTS, ZONE_TOKENS

    for token in ZONE_TOKENS:
        rows = [_row("start", "start", 0.0),
                _row("shot", "H1", 10.0, type=token, result="made")]
        out = gs.derive_game_state(rows)
        box = generate_box_score(rows)
        assert out["score_total"][-1] == box.home_score == ZONE_POINTS[token], token


def test_an_unknown_shot_type_raises_in_both_scans():
    """The old `else: # 2pt` catch-all scored any unrecognized token as two, forever."""
    from simulation.box_score import generate_box_score

    rows = [_row("start", "start", 0.0),
            _row("shot", "H1", 10.0, type="mid_nowhere", result="made")]
    with pytest.raises(KeyError):
        gs.derive_game_state(rows)
    with pytest.raises(KeyError):
        generate_box_score(rows)
