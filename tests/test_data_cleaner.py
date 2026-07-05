"""
Tests for DataCleaner.

Verifies the invariants that downstream models depend on:
  - All required columns are present in cleaned output
  - Rosters are NaN-free lists that survive ast.literal_eval
  - End-of-game sentinel uses the last on-court 5-player lineup (not cumulative)
  - Assist home/away reflects the assisting player's team, not the shooter's
  - Time values are always numeric (never the string "null")
  - Block creates two events (shot + block); shot.result == "blocked"
  - Assist is emitted before the shot it belongs to
  - Steal creates two events (steal-side + turnover-side)
  - Team rebounds produce no event
  - "no turnover" type produces no event
  - Substitution with both players NaN is skipped
  - game_id is monotonically increasing across game boundaries
  - Each game is framed by a "start" event and an "end" event
  - season and playoff are parsed correctly from data_set
  - Time conversion is correct for regulation quarters and OT
"""

import ast
import os
import tempfile

import pandas as pd
import pytest

from data_cleaner import DataCleaner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HOME = ["Alice", "Bob", "Charlie", "Dave", "Eve"]
AWAY = ["Frank", "Grace", "Hank", "Ivy", "Jack"]

_DEFAULT_ROW = {
    "event_type": "shot",
    "period": 1,
    "elapsed": "0:00:30",
    "player": "Alice",
    "assist": None,
    "block": None,
    "steal": None,
    "entered": None,
    "left": None,
    "type": "2pt",
    "result": "made",
    "h1": HOME[0], "h2": HOME[1], "h3": HOME[2], "h4": HOME[3], "h5": HOME[4],
    "a1": AWAY[0], "a2": AWAY[1], "a3": AWAY[2], "a4": AWAY[3], "a5": AWAY[4],
    "data_set": "2002-03 regular season",
    # columns that get dropped:
    "game_id": 1, "away_score": 0, "home_score": 0, "remaining_time": None,
    "play_length": None, "play_id": None, "team": None, "outof": None,
    "possession": None, "shot_distance": None,
    "original_x": None, "original_y": None,
    "converted_x": None, "converted_y": None,
    "description": None,
}


def _make_csv(tmp_path, rows):
    """
    Write a list of row-override dicts as a raw CSV file and return its path.
    Each entry in `rows` overrides _DEFAULT_ROW fields; a leading 'start of
    period 1' row is prepended automatically so every file has a valid game.
    """
    start_row = {**_DEFAULT_ROW, "event_type": "start of period", "period": 1, "elapsed": "0:00:00"}
    full_rows = [start_row] + [{**_DEFAULT_ROW, **r} for r in rows]
    path = str(tmp_path / "raw.csv")
    pd.DataFrame(full_rows).to_csv(path, index=False)
    return path


def _parse(tmp_path, rows):
    """Run DataCleaner.parse_file on a synthetic CSV and return cleaned_df."""
    csv_path = _make_csv(tmp_path, rows)
    dc = DataCleaner()
    dc.season = 2003
    _, cleaned = dc.parse_file(csv_path)
    return cleaned


def _roster_list(cell):
    """Parse a roster cell (list or stringified list) back to a Python list."""
    if isinstance(cell, list):
        return cell
    return ast.literal_eval(cell)


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------

def test_output_columns_present(tmp_path):
    cleaned = _parse(tmp_path, [])
    expected = {"game_id", "roster_home", "roster_away", "time", "event",
                "player", "type", "result", "secondary_player", "home/away", "season", "playoff"}
    assert expected.issubset(set(cleaned.columns))


# ---------------------------------------------------------------------------
# Roster integrity
# ---------------------------------------------------------------------------

def test_rosters_are_lists_with_no_nan(tmp_path):
    """Every roster cell must be a list of strings with no NaN / float values."""
    cleaned = _parse(tmp_path, [{"event_type": "shot", "result": "made"}])
    for col in ("roster_home", "roster_away"):
        for cell in cleaned[col]:
            players = _roster_list(cell)
            assert isinstance(players, list)
            for p in players:
                # Must be a plain string, never float / NaN
                assert isinstance(p, str), f"Non-string in roster: {p!r}"


def test_roster_with_nan_slots_still_valid(tmp_path):
    """A row that has some NaN lineup slots (h4/h5 missing) must still produce a
    valid, parseable roster — not crash ast.literal_eval."""
    row = {"event_type": "shot", "h4": None, "h5": None}
    cleaned = _parse(tmp_path, [row])
    # The shot event (index 1, after the start event)
    shot = cleaned[cleaned["event"] == "shot"].iloc[0]
    players = _roster_list(shot["roster_home"])
    assert all(isinstance(p, str) for p in players)
    assert len(players) <= 5


# ---------------------------------------------------------------------------
# End-event roster
# ---------------------------------------------------------------------------

def test_end_event_roster_is_last_known_lineup(tmp_path):
    """The 'end' sentinel must carry the last on-court 5-player lineup, not
    the cumulative list of all players who appeared in the game."""
    # Game starts with HOME/AWAY; then a sub brings in "Zach" for "Alice".
    rows = [
        {"event_type": "substitution", "player": "Bob", "entered": "Zach", "left": "Alice",
         "h1": "Zach", "h2": HOME[1], "h3": HOME[2], "h4": HOME[3], "h5": HOME[4]},
        # A later shot so the last-known lineup sticks.
        {"event_type": "shot", "player": "Zach",
         "h1": "Zach", "h2": HOME[1], "h3": HOME[2], "h4": HOME[3], "h5": HOME[4]},
    ]
    cleaned = _parse(tmp_path, rows)
    end_row = cleaned[cleaned["event"] == "end"].iloc[0]
    home_roster = _roster_list(end_row["roster_home"])
    # "Zach" must be in the end roster; "Alice" (subbed out) must not.
    assert "Zach" in home_roster
    assert "Alice" not in home_roster


# ---------------------------------------------------------------------------
# run(): file discovery (ignore filtering) + idempotent output
# ---------------------------------------------------------------------------

def _write_raw(dir_path, name, rows=None):
    """Write a synthetic raw master CSV (one game) into dir_path/name."""
    start_row = {**_DEFAULT_ROW, "event_type": "start of period", "period": 1, "elapsed": "0:00:00"}
    full_rows = [start_row] + [{**_DEFAULT_ROW, **r} for r in (rows or [])]
    path = dir_path / name
    pd.DataFrame(full_rows).to_csv(path, index=False)
    return path


def test_input_files_excludes_truncated(tmp_path):
    """Sample/Truncated files and non-CSVs are filtered out of the input set."""
    (tmp_path / "master.csv").write_text("x")
    (tmp_path / "master Truncated.csv").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    dc = DataCleaner(data_path=str(tmp_path))
    assert dc._input_files() == ["master.csv"]


def test_input_files_filters_before_slicing(tmp_path):
    """start/end index into the meaningful files (after ignore-filtering)."""
    for n in ("a.csv", "b.csv", "c.csv", "b Truncated.csv"):
        (tmp_path / n).write_text("x")
    dc = DataCleaner(start=1, data_path=str(tmp_path))
    assert dc._input_files() == ["b.csv", "c.csv"]


def test_run_is_idempotent(tmp_path, monkeypatch):
    """Re-running clean regenerates the season file instead of duplicating it."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_raw(raw, "master.csv", [{"event_type": "shot", "result": "made"}])
    monkeypatch.chdir(tmp_path)  # run() writes to ./data relative to cwd

    DataCleaner(data_path=str(raw)).run()
    out = tmp_path / "data" / "season2003.csv"
    n1 = len(pd.read_csv(out))

    DataCleaner(data_path=str(raw)).run()
    n2 = len(pd.read_csv(out))

    assert n1 == n2 and n1 > 0  # second run overwrote, did not append


def test_run_excludes_truncated_within_a_run(tmp_path, monkeypatch):
    """A Truncated sample beside the master file is not processed (no dup games)."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_raw(raw, "master.csv", [{"event_type": "shot", "result": "made"}])
    _write_raw(raw, "master Truncated.csv", [{"event_type": "shot", "result": "made"}])
    monkeypatch.chdir(tmp_path)

    DataCleaner(data_path=str(raw)).run()
    cleaned = pd.read_csv(tmp_path / "data" / "season2003.csv")
    # One master file = one game; if Truncated were processed there'd be two.
    assert cleaned["game_id"].nunique() == 1


def test_end_event_roster_never_exceeds_five(tmp_path):
    """Even if many subs occur, end-event roster is at most 5 players."""
    # All subs happen after the start; last lineup is still max 5.
    rows = [
        {"event_type": "substitution", "entered": "P6", "left": "Alice",
         "h1": "P6", "h2": HOME[1], "h3": HOME[2], "h4": HOME[3], "h5": HOME[4]},
    ]
    cleaned = _parse(tmp_path, rows)
    end_row = cleaned[cleaned["event"] == "end"].iloc[0]
    assert len(_roster_list(end_row["roster_home"])) <= 5


# ---------------------------------------------------------------------------
# Assist home/away
# ---------------------------------------------------------------------------

def test_assist_home_away_reflects_assister_not_shooter(tmp_path):
    """
    Shooter is 'Alice' (home). Assister is 'Frank' (away).
    The assist event must have home/away == 2 (home_indicator's away code), not 1
    (home) — i.e. attribution follows the assister, not the shooter.
    """
    row = {"event_type": "shot", "player": "Alice", "assist": "Frank", "result": "made"}
    cleaned = _parse(tmp_path, [row])
    assist_event = cleaned[cleaned["event"] == "assist"].iloc[0]
    assert assist_event["home/away"] == 2  # Frank is away (home_indicator: 1=home, 2=away)

def test_assist_home_away_when_assister_is_home(tmp_path):
    """Assister 'Bob' is home → home/away == 1."""
    row = {"event_type": "shot", "player": "Alice", "assist": "Bob", "result": "made"}
    cleaned = _parse(tmp_path, [row])
    assist_event = cleaned[cleaned["event"] == "assist"].iloc[0]
    assert assist_event["home/away"] == 1


# ---------------------------------------------------------------------------
# Time values
# ---------------------------------------------------------------------------

def test_time_is_always_numeric(tmp_path):
    """time column must contain only int/float, never the string 'null'."""
    rows = [
        {"event_type": "shot", "elapsed": None},          # unparseable time
        {"event_type": "rebound", "type": "rebound defensive"},
    ]
    cleaned = _parse(tmp_path, rows)
    for val in cleaned["time"]:
        assert isinstance(val, (int, float)), f"Non-numeric time: {val!r}"
        assert val != "null"


def test_unparseable_time_carries_forward_last_known(tmp_path):
    """When a row's time can't be parsed, the event gets the last valid time."""
    rows = [
        {"event_type": "shot", "elapsed": "0:01:00"},    # time_val = 60
        {"event_type": "shot", "elapsed": None},          # unparseable → carry 60
    ]
    cleaned = _parse(tmp_path, rows)
    shots = cleaned[cleaned["event"] == "shot"]
    assert shots.iloc[1]["time"] == shots.iloc[0]["time"]


# ---------------------------------------------------------------------------
# Time conversion math
# ---------------------------------------------------------------------------

def test_time_conversion_q1():
    dc = DataCleaner()
    # 30 seconds into Q1 = 30
    assert dc.convert_time(1, "0:00:30") == 30

def test_time_conversion_q2():
    dc = DataCleaner()
    # 0 seconds into Q2 = 12*60 = 720
    assert dc.convert_time(2, "0:00:00") == 720

def test_time_conversion_q4():
    dc = DataCleaner()
    # 1:30 into Q4 = 3*720 + 90 = 2250
    assert dc.convert_time(4, "0:01:30") == 2250

def test_time_conversion_ot1():
    dc = DataCleaner()
    # 0 seconds into OT1 (period 5) = 4*12*60 = 2880
    assert dc.convert_time(5, "0:00:00") == 2880

def test_time_conversion_ot2():
    dc = DataCleaner()
    # 2:00 into OT2 (period 6) = 2880 + 300 + 120 = 3300
    assert dc.convert_time(6, "0:02:00") == 3300

def test_time_conversion_invalid():
    dc = DataCleaner()
    assert dc.convert_time(None, "0:00:30") is None
    assert dc.convert_time(1, None) is None
    assert dc.convert_time(1, "bad") is None


# ---------------------------------------------------------------------------
# Block events
# ---------------------------------------------------------------------------

def test_block_creates_shot_and_block_events(tmp_path):
    row = {"event_type": "shot", "player": "Alice", "block": "Frank", "result": "missed"}
    cleaned = _parse(tmp_path, [row])
    shot = cleaned[cleaned["event"] == "shot"].iloc[0]
    block = cleaned[cleaned["event"] == "block"].iloc[0]

    assert shot["result"] == "blocked"
    assert block["player"] == "Frank"
    assert block["type"] == "2pt"              # shot sub-type (not the victim's name)
    assert block["secondary_player"] == "Alice"  # blocked shooter goes here
    assert block["result"] == "block"


def test_block_home_away_is_opposite_of_shooter(tmp_path):
    """Shooter Alice is home (1); blocker Frank is away (2)."""
    row = {"event_type": "shot", "player": "Alice", "block": "Frank"}
    cleaned = _parse(tmp_path, [row])
    shot = cleaned[cleaned["event"] == "shot"].iloc[0]
    block = cleaned[cleaned["event"] == "block"].iloc[0]
    assert shot["home/away"] == 1
    assert block["home/away"] == 2  # home_indicator: 1=home, 2=away


# ---------------------------------------------------------------------------
# Assist ordering
# ---------------------------------------------------------------------------

def test_assist_emitted_before_shot(tmp_path):
    row = {"event_type": "shot", "player": "Alice", "assist": "Bob", "result": "made"}
    cleaned = _parse(tmp_path, [row])
    idx_assist = cleaned.index[cleaned["event"] == "assist"][0]
    idx_shot = cleaned.index[cleaned["event"] == "shot"][0]
    assert idx_assist < idx_shot


# ---------------------------------------------------------------------------
# Steal / turnover
# ---------------------------------------------------------------------------

def test_steal_creates_two_turnover_events(tmp_path):
    row = {
        "event_type": "turnover", "player": "Alice",
        "steal": "Frank", "type": None,
    }
    cleaned = _parse(tmp_path, [row])
    turnovers = cleaned[cleaned["event"] == "turnover"]
    assert len(turnovers) == 2

    steal_evt = turnovers[turnovers["result"] == "steal"].iloc[0]
    cop_evt = turnovers[turnovers["result"] == "cop"].iloc[0]

    assert steal_evt["player"] == "Frank"
    assert steal_evt["type"] == "steal"
    assert cop_evt["player"] == "Alice"
    assert cop_evt["type"] == "steal"


def test_steal_home_away_for_both_events(tmp_path):
    """Frank (away) steals from Alice (home)."""
    row = {"event_type": "turnover", "player": "Alice", "steal": "Frank"}
    cleaned = _parse(tmp_path, [row])
    turnovers = cleaned[cleaned["event"] == "turnover"]
    steal_evt = turnovers[turnovers["result"] == "steal"].iloc[0]
    cop_evt = turnovers[turnovers["result"] == "cop"].iloc[0]
    assert steal_evt["home/away"] == 2   # Frank is away (home_indicator: 1=home, 2=away)
    assert cop_evt["home/away"] == 1     # Alice is home


def test_no_turnover_type_produces_no_event(tmp_path):
    row = {"event_type": "turnover", "player": "Alice", "type": "no turnover"}
    cleaned = _parse(tmp_path, [row])
    assert cleaned[cleaned["event"] == "turnover"].empty


def test_unrecognized_turnover_type_skipped(tmp_path):
    row = {"event_type": "turnover", "player": "Alice", "type": "mystery error xyz"}
    cleaned = _parse(tmp_path, [row])
    assert cleaned[cleaned["event"] == "turnover"].empty


# ---------------------------------------------------------------------------
# Rebound
# ---------------------------------------------------------------------------

def test_team_rebound_produces_no_event(tmp_path):
    row = {"event_type": "rebound", "type": "team rebound", "player": None}
    cleaned = _parse(tmp_path, [row])
    assert cleaned[cleaned["event"] == "rebound"].empty


def test_defensive_rebound_result_is_cop(tmp_path):
    row = {"event_type": "rebound", "type": "rebound defensive", "player": "Alice"}
    cleaned = _parse(tmp_path, [row])
    reb = cleaned[cleaned["event"] == "rebound"].iloc[0]
    assert reb["type"] == "defensive"
    assert reb["result"] == "cop"


def test_offensive_rebound_result_is_null(tmp_path):
    row = {"event_type": "rebound", "type": "rebound offensive", "player": "Alice"}
    cleaned = _parse(tmp_path, [row])
    reb = cleaned[cleaned["event"] == "rebound"].iloc[0]
    assert reb["type"] == "offensive"
    assert reb["result"] == "null"


# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

def test_substitution_both_nan_skipped(tmp_path):
    row = {"event_type": "substitution", "entered": None, "left": None}
    cleaned = _parse(tmp_path, [row])
    assert cleaned[cleaned["event"] == "substitution"].empty


def test_substitution_player_is_outgoing_secondary_is_incoming(tmp_path):
    """Convention: `player` = outgoing (left), `secondary_player` = incoming (entered)."""
    row = {"event_type": "substitution", "entered": "Zach", "left": "Alice",
           "h1": "Zach", "h2": HOME[1], "h3": HOME[2], "h4": HOME[3], "h5": HOME[4]}
    cleaned = _parse(tmp_path, [row])
    sub = cleaned[cleaned["event"] == "substitution"].iloc[0]
    assert sub["player"] == "Alice"             # outgoing player
    assert sub["secondary_player"] == "Zach"    # incoming player
    assert sub["type"] == "substitution"
    assert sub["result"] == "substitution"


def test_substitution_home_away_identifies_team(tmp_path):
    """home/away identifies the substituting team via the incoming player, who is on
    the post-sub five (the outgoing player has already left it). Home sub → 1."""
    row = {"event_type": "substitution", "entered": "Zach", "left": "Alice",
           "h1": "Zach", "h2": HOME[1], "h3": HOME[2], "h4": HOME[3], "h5": HOME[4]}
    cleaned = _parse(tmp_path, [row])
    sub = cleaned[cleaned["event"] == "substitution"].iloc[0]
    assert sub["home/away"] == 1


def test_substitution_only_incoming_present(tmp_path):
    """Only `entered` present: outgoing unknown → player='null', incoming kept."""
    row = {"event_type": "substitution", "entered": "Zach", "left": None}
    cleaned = _parse(tmp_path, [row])
    sub = cleaned[cleaned["event"] == "substitution"].iloc[0]
    assert sub["player"] == "null"              # no outgoing → "null" token
    assert sub["type"] == "substitution"        # clean type (not the leaving player)
    assert sub["secondary_player"] == "Zach"    # incoming player


def test_substitution_only_outgoing_present(tmp_path):
    """Only `left` present: incoming unknown → secondary_player='none'."""
    row = {"event_type": "substitution", "entered": None, "left": "Alice"}
    cleaned = _parse(tmp_path, [row])
    sub = cleaned[cleaned["event"] == "substitution"].iloc[0]
    assert sub["player"] == "Alice"             # outgoing player
    assert sub["secondary_player"] == "none"    # no one entered → "none" token


# ---------------------------------------------------------------------------
# Foul normalization
# ---------------------------------------------------------------------------

def test_foul_type_offensive_charge(tmp_path):
    row = {"event_type": "foul", "player": "Alice", "type": "offensive charge"}
    cleaned = _parse(tmp_path, [row])
    foul = cleaned[cleaned["event"] == "foul"].iloc[0]
    assert foul["type"] == "offensive"
    assert foul["result"] == "cop"


def test_foul_type_technical(tmp_path):
    row = {"event_type": "foul", "player": "Alice", "type": "non-unsportsmanlike technical"}
    cleaned = _parse(tmp_path, [row])
    foul = cleaned[cleaned["event"] == "foul"].iloc[0]
    assert foul["type"] == "technical"
    assert foul["result"] == "free throw"


def test_foul_unknown_type_raises(tmp_path):
    dc = DataCleaner()
    dc.season = 2003
    with pytest.raises(ValueError, match="Unknown foul type"):
        dc.determine_foul_result("mystery foul")


# ---------------------------------------------------------------------------
# Game boundaries & IDs
# ---------------------------------------------------------------------------

def test_each_game_has_start_and_end(tmp_tmp=None):
    """parse_file must emit exactly one 'start' and one 'end' event."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = type("P", (), {"__truediv__": lambda s, x: os.path.join(tmp, x)})()
        cleaned = _parse(tmp_path, [{"event_type": "shot"}])
    assert (cleaned["event"] == "start").sum() == 1
    assert (cleaned["event"] == "end").sum() == 1


def test_game_id_increments_across_game_boundaries(tmp_path):
    """Two games in one file → game_id 1 then game_id 2."""
    start2 = {**_DEFAULT_ROW, "event_type": "start of period", "period": 1}
    rows = [
        {"event_type": "shot"},
        start2,                       # second game boundary
        {"event_type": "shot"},
    ]
    csv_path = _make_csv(tmp_path, rows)
    dc = DataCleaner()
    dc.season = 2003
    _, cleaned = dc.parse_file(csv_path)

    game_ids = sorted(cleaned["game_id"].unique())
    assert game_ids == list(range(game_ids[0], game_ids[0] + 2))  # two consecutive IDs


# ---------------------------------------------------------------------------
# Season / playoff
# ---------------------------------------------------------------------------

def test_regular_season_parsed(tmp_path):
    row = {"event_type": "shot", "data_set": "2002-03 regular season"}
    cleaned = _parse(tmp_path, [row])
    # season = 2002+1 = 2003 (set by run(); verify parse_file inherits it)
    assert (cleaned["playoff"] == 1).all()  # playoff column: 1=regular season, 2=playoffs


def test_playoff_flag_set(tmp_path):
    """data_set ending in something other than 'n' → playoff=1."""
    start_row = {**_DEFAULT_ROW,
                 "event_type": "start of period", "period": 1,
                 "data_set": "2002-03 playoffs"}
    rows_override = [{"event_type": "shot", "data_set": "2002-03 playoffs"}]
    full_rows = [start_row] + [{**_DEFAULT_ROW, **r} for r in rows_override]
    csv_path = str(tmp_path / "raw.csv")
    pd.DataFrame(full_rows).to_csv(csv_path, index=False)

    dc = DataCleaner()
    dc.season = 2003
    _, cleaned = dc.parse_file(csv_path)
    # The start-of-period row should flag the playoffs; all events in that game inherit
    # it (playoff column: 1=regular season, 2=playoffs).
    game_events = cleaned[cleaned["event"] != "end"]
    assert (game_events["playoff"] == 2).all()


# ---------------------------------------------------------------------------
# Free throw normalization
# ---------------------------------------------------------------------------

def test_free_throw_event_is_shot(tmp_path):
    row = {"event_type": "free throw", "player": "Alice", "type": None, "result": "made"}
    cleaned = _parse(tmp_path, [row])
    ft = cleaned[cleaned["event"] == "shot"].iloc[0]
    assert ft["type"] == "free throw"
    assert ft["result"] == "made"


# ---------------------------------------------------------------------------
# home/away for shots
# ---------------------------------------------------------------------------

def test_shot_home_player_has_home_indicator_1(tmp_path):
    row = {"event_type": "shot", "player": "Alice"}  # Alice is in HOME
    cleaned = _parse(tmp_path, [row])
    shot = cleaned[cleaned["event"] == "shot"].iloc[0]
    assert shot["home/away"] == 1


def test_shot_away_player_has_home_indicator_2(tmp_path):
    row = {"event_type": "shot", "player": "Frank"}  # Frank is in AWAY
    cleaned = _parse(tmp_path, [row])
    shot = cleaned[cleaned["event"] == "shot"].iloc[0]
    assert shot["home/away"] == 2  # home_indicator: 1=home, 2=away
