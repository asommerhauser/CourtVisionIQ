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
        _row("shot", "H1", 10.0, type="2pt", result="made"),    # home +2
        _row("shot", "A1", 20.0, type="3pt", result="made"),    # away +3
        _row("shot", "H2", 30.0, type="2pt", result="missed"),  # no points
        _row("shot", "A2", 40.0, type="free throw", result="made"),  # away +1
    ]
    out = gs.derive_game_state(rows)

    # score_diff is inclusive of each row's own event (matches the live controller score).
    assert list(out["score_diff"]) == [0, 2, -1, -1, -2]
    assert list(out["score_total"]) == [0, 2, 5, 5, 6]


def test_missed_and_nonscoring_events_do_not_move_score():
    rows = [
        _row("shot", "H1", 5.0, type="2pt", result="missed"),
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
        _row("foul", "A1", 200.0, type="shooting"),   # away 1
        _row("foul", "H2", 300.0, type="offensive"),  # excluded (offensive)
        _row("foul", "H3", 400.0, type="technical"),  # excluded (technical)
        _row("foul", "H4", 500.0, type="loose ball"), # home 2
        _row("shot", "H1", 800.0, type="2pt", result="made"),  # Q2 — fouls reset
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
    for k in gs.GAME_STATE_KEYS:
        assert out[k].dtype == np.float32


# ---------------------------------------------------------------------------
# merge_game_state_features — positional alignment over a multi-game frame
# ---------------------------------------------------------------------------

def test_merge_aligns_positionally_across_games():
    g1 = [
        _row("shot", "H1", 10.0, type="3pt", result="made"),   # +3 home
        _row("shot", "A1", 20.0, type="2pt", result="made"),   # +2 away
    ]
    g2 = [
        _row("shot", "H1", 10.0, type="2pt", result="made"),   # +2 home (fresh game)
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
            _row("shot", "H1", 10.0, type="2pt", result="made"),  # g1: +2
            _row("shot", "H1", 10.0, type="3pt", result="made"),  # g2: +3
            _row("shot", "A1", 20.0, type="2pt", result="made"),  # g1: -2 (diff back to 0)
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
        _row("shot", "H1", 10.0, type="2pt", result="made"),
        _row("shot", "H2", 20.0, type="3pt", result="made"),
        _row("shot", "A1", 30.0, type="2pt", result="made"),
        _row("shot", "A2", 40.0, type="free throw", result="made"),
        _row("shot", "H3", 50.0, type="2pt", result="missed"),
    ]
    out = gs.derive_game_state(rows)
    box = generate_box_score(rows)

    # The inclusive running diff/total at the last row equals the final box score.
    assert out["score_diff"][-1] == box.home_score - box.away_score
    assert out["score_total"][-1] == box.home_score + box.away_score
