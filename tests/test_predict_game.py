"""
Prediction-export + box-score (+/-) tests.

Verify that a generated game serializes to the **exact cleaned-data format** (so it parses with
the same tooling as ``data/season2003.csv``), that the home/away flag is derived correctly, that
the play-by-play round-trips through the box-score decoder, and that plus/minus is credited to
the lineups on the floor. No trained models needed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from simulation.box_score import generate_box_score
from simulation.game_input import GameInput
from simulation.predict_game import (
    CLEANED_COLUMNS, _real_starters, _score_dict, history_to_cleaned_frame,
)

HOME = ["A", "B", "C", "D", "E"]
AWAY = ["F", "G", "H", "I", "J"]


def _row(time, event, player, type_, result, secondary="none", home=HOME, away=AWAY):
    return {"event": event, "player": player, "type": type_, "result": result,
            "secondary_player": secondary, "time": time,
            "roster_home": list(home), "roster_away": list(away)}


def _spec():
    return GameInput(home_roster=HOME, away_roster=AWAY, season=2003, playoff=0)


def test_cleaned_frame_columns_match_real_data():
    df = history_to_cleaned_frame([_row(0, "start", "start", "start", "start")], _spec())
    assert list(df.columns) == CLEANED_COLUMNS
    real = pd.read_csv(Path("data") / "season2003.csv", nrows=1)
    assert list(df.columns) == list(real.columns)


def test_home_away_flag_derivation():
    history = [
        _row(0, "start", "start", "start", "start"),     # boundary → 0
        _row(5, "shot", "A", "2pt", "made"),             # home → 1
        _row(10, "rebound", "F", "defensive", "cop"),    # away → 2
        _row(15, "substitution", "A", "substitution", "substitution", secondary="K",
             home=["K", "B", "C", "D", "E"]),            # keyed off incoming K (home) → 1
    ]
    df = history_to_cleaned_frame(history, _spec())
    assert df["home/away"].tolist() == [0, 1, 2, 1]


def test_playoff_flag_mapping():
    playoff_spec = GameInput(home_roster=HOME, away_roster=AWAY, season=2003, playoff=1)
    df = history_to_cleaned_frame([_row(0, "start", "start", "start", "start")], playoff_spec)
    assert df["playoff"].iloc[0] == 2                    # cleaned: 1=regular, 2=playoff
    df_reg = history_to_cleaned_frame([_row(0, "start", "start", "start", "start")], _spec())
    assert df_reg["playoff"].iloc[0] == 1


def test_roundtrip_through_box_score():
    history = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "shot", "A", "2pt", "made"),
        _row(20, "shot", "F", "3pt", "made"),
        _row(30, "end", "end", "end", "end"),
    ]
    # The exported frame must decode back to the same scores.
    df = history_to_cleaned_frame(history, _spec())
    box = generate_box_score(df)
    assert box.home_score == 2
    assert box.away_score == 3
    assert box.home_score == sum(pl.pts for pl in box.home)


def test_plus_minus_credited_to_on_court_lineups():
    history = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "shot", "A", "2pt", "made"),     # home +2: A..E +2, F..J -2
        _row(20, "shot", "F", "3pt", "made"),      # away +3: F..J +3, A..E -3
        _row(30, "end", "end", "end", "end"),
    ]
    box = generate_box_score(history, home_team="HOM", away_team="AWY")
    pm = {pl.player: pl.pm for pl in (*box.home, *box.away)}
    assert pm["A"] == -1      # +2 then -3
    assert pm["F"] == 1       # -2 then +3
    # A team's net +/- (sum over its five) equals score margin × 5 only if lineups never change;
    # here the simple invariant: each scoring play is zero-sum across the ten on-court players.
    assert sum(pm.values()) == 0


def test_real_starters_read_from_start_row():
    # Six players appear across the game, but the start row's lineup is the real tip-off five.
    home_six = HOME + ["K"]
    away_six = AWAY + ["L"]
    game = pd.DataFrame([
        _row(0, "start", "start", "start", "start", home=HOME, away=AWAY),
        _row(120, "substitution", "A", "substitution", "substitution", secondary="K",
             home=["K", "B", "C", "D", "E"], away=AWAY),
        _row(130, "substitution", "F", "substitution", "substitution", secondary="L",
             home=["K", "B", "C", "D", "E"], away=["L", "G", "H", "I", "J"]),
    ])
    home_starters, away_starters = _real_starters(game)
    assert home_starters == HOME            # the start-row five, not the union with bench
    assert away_starters == AWAY
    assert "K" not in home_starters and "L" not in away_starters


def test_real_starters_filters_non_players_and_caps_at_five():
    game = pd.DataFrame([
        _row(0, "start", "start", "start", "start",
             home=HOME + ["none", "K"], away=AWAY),
    ])
    home_starters, _ = _real_starters(game)
    assert home_starters == HOME            # "none" dropped, capped at five


def test_score_dict_picks_winner():
    home = _score_dict("HOME", "AWAY", 102, 98)
    assert home["winner"] == "HOME"
    assert home["line"] == "HOME 102 - 98 AWAY"
    away = _score_dict("HOME", "AWAY", 90, 99)
    assert away["winner"] == "AWAY"
    tie = _score_dict("HOME", "AWAY", 100, 100)
    assert tie["winner"] == "TIE"


def test_box_row_has_full_nba_columns():
    box = generate_box_score([
        _row(0, "start", "start", "start", "start"),
        _row(10, "shot", "A", "2pt", "made"),
        _row(20, "end", "end", "end", "end"),
    ])
    frame = box.to_frame("home")
    for col in ("MIN", "FG", "FG%", "3PT", "FT", "OREB", "DREB", "REB", "+/-", "PTS"):
        assert col in frame.columns
    assert frame.iloc[-1]["Player"] == "TEAM"      # totals row appended
