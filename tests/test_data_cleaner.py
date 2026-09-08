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
from zones import ZONE_TOKENS


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
    "type": "jump shot",          # raw free text; zones.marker_is_three reads the 3pt prefix
    "result": "made",
    "h1": HOME[0], "h2": HOME[1], "h3": HOME[2], "h4": HOME[3], "h5": HOME[4],
    "a1": AWAY[0], "a2": AWAY[1], "a3": AWAY[2], "a4": AWAY[3], "a5": AWAY[4],
    "data_set": "2002-03 regular season",
    # columns that get dropped:
    "game_id": 1, "away_score": 0, "home_score": 0, "remaining_time": None,
    "play_length": None, "play_id": None, "team": None,
    # KEPT from 2.0 on — a shooting foul's free-throw count is read off the following trip's
    # `outof` (and, for an and-1, the preceding basket's `points`). See _label_shooting_fouls.
    "outof": None, "num": None, "points": None,
    # KEPT from 2.0 on — the fouled player, written into a foul row's secondary_player.
    "opponent": None,
    "possession": None,
    "original_x": None, "original_y": None,
    "description": None,
    # KEPT from 2.0 on — the shot zone is derived from these (zones.py). (25, 10) is 4.75 ft
    # straight out from the near hoop: inside the lane, outside the rim -> "paint".
    "shot_distance": 5, "converted_x": 25.0, "converted_y": 10.0,
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
# Shot zones — the fifteen spatial tokens replacing the 2pt/3pt binary
# ---------------------------------------------------------------------------

def test_shot_rows_carry_a_zone_token(tmp_path):
    """The zone comes from the coordinates, not the raw type text."""
    cleaned = _parse(tmp_path, [{"event_type": "shot", "player": "Alice",
                                 "converted_x": 25.0, "converted_y": 7.0,
                                 "shot_distance": 2, "result": "made"}])
    shot = cleaned[cleaned["event"] == "shot"].iloc[0]
    assert shot["type"] == "rim"          # 1.75 ft from the hoop


def test_the_raw_marker_decides_the_family_not_the_coordinates(tmp_path):
    """A 3pt-marked attempt gets a three zone; the same spot unmarked gets a two zone."""
    spot = {"converted_x": 25.0, "converted_y": 30.0, "shot_distance": 25}
    three = _parse(tmp_path, [{"event_type": "shot", "player": "Alice",
                               "type": "3pt jump shot", "result": "missed", **spot}])
    two = _parse(tmp_path, [{"event_type": "shot", "player": "Alice",
                             "type": "jump shot", "result": "missed", **spot}])
    assert three[three["event"] == "shot"].iloc[0]["type"] == "top3"
    assert two[two["event"] == "shot"].iloc[0]["type"] == "mid_top"


def test_a_shot_with_no_coordinates_falls_back_rather_than_failing(tmp_path):
    cleaned = _parse(tmp_path, [{"event_type": "shot", "player": "Alice", "result": "made",
                                 "converted_x": None, "converted_y": None,
                                 "shot_distance": None}])
    shot = cleaned[cleaned["event"] == "shot"].iloc[0]
    assert shot["type"] == "mid_base_l"   # zones.FALLBACK_TWO


def test_assist_and_block_share_the_shot_row_zone(tmp_path):
    """All three rows describe the same attempt, so all three carry the same token."""
    cleaned = _parse(tmp_path, [{"event_type": "shot", "player": "Alice", "assist": "Bob",
                                 "converted_x": 25.0, "converted_y": 7.0,
                                 "shot_distance": 2, "result": "made"}])
    assert cleaned[cleaned["event"] == "shot"].iloc[0]["type"] == "rim"
    assert cleaned[cleaned["event"] == "assist"].iloc[0]["type"] == "rim"

    blocked = _parse(tmp_path, [{"event_type": "shot", "player": "Alice", "block": "Frank",
                                 "converted_x": 25.0, "converted_y": 7.0,
                                 "shot_distance": 2, "result": "missed"}])
    assert blocked[blocked["event"] == "block"].iloc[0]["type"] == "rim"


def test_non_shot_rows_never_get_a_zone(tmp_path):
    """The old binary ran on every raw row and typed turnovers and fouls as '2pt'."""
    cleaned = _parse(tmp_path, [
        {"event_type": "turnover", "player": "Alice", "type": "bad pass", "result": None},
        {"event_type": "foul", "player": "Bob", "type": "shooting", "result": None},
    ])
    for _, row in cleaned.iterrows():
        if row["event"] in ("turnover", "foul"):
            assert row["type"] not in ZONE_TOKENS, f"{row['event']} typed as {row['type']}"


# ---------------------------------------------------------------------------
# Schema cleanup — one row per play
# ---------------------------------------------------------------------------

def test_a_steal_is_one_row_naming_the_stealer(tmp_path):
    """It used to be two rows -- a grammar the controller then had to reproduce exactly."""
    cleaned = _parse(tmp_path, [
        {"event_type": "turnover", "player": "Alice", "steal": "Frank", "type": "lost ball",
         "result": None},
    ])
    tos = cleaned[cleaned["event"] == "turnover"]
    assert len(tos) == 1
    row = tos.iloc[0]
    assert (row["player"], row["type"], row["result"]) == ("Alice", "steal", "cop")
    assert row["secondary_player"] == "Frank"          # the stealer, as a block row carries one


def test_a_plain_turnover_names_nobody(tmp_path):
    cleaned = _parse(tmp_path, [
        {"event_type": "turnover", "player": "Alice", "steal": None, "type": "bad pass",
         "result": None},
    ])
    row = cleaned[cleaned["event"] == "turnover"].iloc[0]
    assert (row["type"], row["secondary_player"]) == ("error", "none")


def test_an_offensive_foul_emits_no_trailing_turnover(tmp_path):
    """The raw data pairs the two for 100% of them; the box score counts the TOV from the foul."""
    cleaned = _parse(tmp_path, [
        {"event_type": "foul", "player": "Cara", "type": "offensive charge", "opponent": "Gus",
         "result": None},
        {"event_type": "turnover", "player": "Cara", "steal": None, "type": "offensive foul",
         "result": None},
    ])
    assert cleaned[cleaned["event"] == "turnover"].empty
    assert cleaned[cleaned["event"] == "foul"].iloc[0]["type"] == "offensive"


def test_a_standalone_technical_becomes_a_technical_foul_row(tmp_path):
    """Defensive three seconds and double technicals file under their own raw event_type."""
    cleaned = _parse(tmp_path, [
        {"event_type": "technical foul", "player": "Gus", "type": "defensive 3 seconds",
         "result": None},
    ])
    row = cleaned[cleaned["event"] == "foul"].iloc[0]
    assert (row["player"], row["type"], row["result"]) == ("Gus", "technical", "free throw")


def test_a_technical_with_no_player_is_dropped(tmp_path):
    """Coach technicals name no actor to attribute it to."""
    cleaned = _parse(tmp_path, [
        {"event_type": "technical foul", "player": None, "type": "coach technical foul",
         "result": None},
    ])
    assert cleaned[cleaned["event"] == "foul"].empty


def test_the_emitted_schema_is_enforced(tmp_path):
    """A missing key would otherwise land as a silent all-NaN column in the season file."""
    from data_cleaner import OUTPUT_COLUMNS
    cleaned = _parse(tmp_path, [{"event_type": "shot", "player": "Alice", "result": "made"}])
    assert list(cleaned.columns) == list(OUTPUT_COLUMNS)

    with pytest.raises(ValueError, match="cleaned schema"):
        DataCleaner._check_schema([{"game_id": 1}])


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

def _teamed(rows):
    """Prepend action rows so the cleaner can resolve both team abbreviations."""
    return [
        {"event_type": "shot", "player": "Alice", "team": "LAL", "result": "made"},
        {"event_type": "shot", "player": "Frank", "team": "BOS", "result": "missed"},
        *rows,
    ]


def test_a_timeout_becomes_a_row_naming_the_calling_side(tmp_path):
    cleaned = _parse(tmp_path, _teamed([
        {"event_type": "timeout", "player": None, "team": "LAL", "type": "timeout: regular",
         "result": None},
        {"event_type": "timeout", "player": None, "team": "BOS", "type": "timeout: regular",
         "result": None},
    ]))
    tos = cleaned[cleaned["event"] == "timeout"]
    assert list(tos["type"]) == ["home", "away"]
    assert list(tos["player"]) == ["none", "none"]
    assert list(tos["home/away"]) == [1, 2]


def test_a_timeout_from_an_unknown_team_is_dropped(tmp_path):
    """A guard, not a path the data takes: both abbreviations resolve before any timeout."""
    cleaned = _parse(tmp_path, _teamed([
        {"event_type": "timeout", "player": None, "team": "XXX", "type": "timeout: regular",
         "result": None},
    ]))
    assert cleaned[cleaned["event"] == "timeout"].empty


def test_a_jump_ball_does_not_bind_a_side(tmp_path):
    """A jump-ball row credits the team that WON the tip while naming one of the two jumpers.

    The jumpers are opponents by definition, so about half the time the row pairs an away
    player with the home abbreviation. Binding from it put both sides on one string in ~47% of
    real games in every era, which dropped every timeout by the unbound team and labelled every
    surviving one "home". The existing timeout tests bound from shot rows, so none of them
    reached this path.

    Here Frank is away and the tip went to LAL (home): the jump ball must bind nothing, leaving
    the two shot rows to resolve LAL=home and BOS=away.
    """
    cleaned = _parse(tmp_path, [
        {"event_type": "jump ball", "player": "Frank", "team": "LAL", "result": None},
        {"event_type": "shot", "player": "Alice", "team": "LAL", "result": "made"},
        {"event_type": "shot", "player": "Frank", "team": "BOS", "result": "missed"},
        {"event_type": "timeout", "player": None, "team": "LAL", "type": "timeout: regular",
         "result": None},
        {"event_type": "timeout", "player": None, "team": "BOS", "type": "timeout: regular",
         "result": None},
    ])
    tos = cleaned[cleaned["event"] == "timeout"]
    assert list(tos["type"]) == ["home", "away"]
    # The context columns fill in as each side resolves (null until then, never backfilled),
    # so read the first resolved value of each rather than row 0.
    assert cleaned["home_team"].dropna().iloc[0] == "LAL"
    assert cleaned["away_team"].dropna().iloc[0] == "BOS"


def test_the_two_sides_never_share_an_abbreviation(tmp_path):
    """The guard behind the fix: a binding that would collapse the sides is refused."""
    cleaned = _parse(tmp_path, [
        # A malformed pair crediting one abbreviation to both sides. Home binds; away must not.
        {"event_type": "shot", "player": "Alice", "team": "LAL", "result": "made"},
        {"event_type": "shot", "player": "Frank", "team": "LAL", "result": "missed"},
        {"event_type": "shot", "player": "Grace", "team": "BOS", "result": "missed"},
    ])
    assert cleaned["home_team"].dropna().iloc[0] == "LAL"
    assert cleaned["away_team"].dropna().iloc[0] == "BOS"


# ---------------------------------------------------------------------------
# Team rebounds
# ---------------------------------------------------------------------------

def test_a_playerless_rebound_becomes_a_team_rebound_token(tmp_path):
    """It used to be emitted as a player row literally named "null", ~11.9k times a season."""
    cleaned = _parse(tmp_path, [
        {"event_type": "rebound", "player": None, "type": "rebound offensive", "result": None},
        {"event_type": "rebound", "player": None, "type": "rebound defensive", "result": None},
    ])
    reb = cleaned[cleaned["event"] == "rebound"]
    assert list(reb["type"]) == ["team offensive", "team defensive"]
    assert list(reb["player"]) == ["none", "none"]
    assert list(reb["result"]) == ["null", "cop"]


def test_a_credited_rebound_is_unchanged(tmp_path):
    cleaned = _parse(tmp_path, [
        {"event_type": "rebound", "player": "Alice", "type": "rebound offensive", "result": None},
    ])
    reb = cleaned[cleaned["event"] == "rebound"].iloc[0]
    assert (reb["type"], reb["player"]) == ("offensive", "Alice")


def _miss(player, team):
    return {"event_type": "shot", "player": player, "team": team, "result": "missed"}


_BARE = {"event_type": "rebound", "player": None, "type": "team rebound", "result": None,
         "team": None}


def test_a_bare_team_rebounds_side_is_recovered_from_the_next_possession(tmp_path):
    """The raw type records no side; the next possession-bearing event does."""
    kept = _parse(tmp_path, [
        _miss("Alice", "LAL"), _miss("Frank", "BOS"),      # resolve both abbreviations
        _miss("Frank", "BOS"), _BARE, _miss("Gus", "BOS"),   # BOS kept it -> offensive
        _miss("Frank", "BOS"), _BARE, _miss("Alice", "LAL"),  # LAL got it -> defensive
    ])
    reb = kept[kept["event"] == "rebound"]
    assert list(reb["type"]) == ["team offensive", "team defensive"]
    assert list(reb["result"]) == ["null", "cop"]


def test_a_team_rebound_between_free_throws_is_not_a_rebound(tmp_path):
    """6,336 of 2022-23's 9,374 bare rows are this: the ball is dead, the shooter shoots again."""
    cleaned = _parse(tmp_path, [
        _miss("Alice", "LAL"), _miss("Frank", "BOS"),
        {"event_type": "free throw", "player": "Alice", "team": "LAL", "result": "missed",
         "num": 1, "outof": 2},
        _BARE,
        {"event_type": "free throw", "player": "Alice", "team": "LAL", "result": "made",
         "num": 2, "outof": 2},
    ])
    assert cleaned[cleaned["event"] == "rebound"].empty


def test_a_team_rebound_with_no_following_possession_is_dropped(tmp_path):
    """End-of-period boards: nobody ever gets the ball, so there is no side to record."""
    cleaned = _parse(tmp_path, [
        _miss("Alice", "LAL"), _miss("Frank", "BOS"),
        _miss("Frank", "BOS"), _BARE,
    ])
    assert cleaned[cleaned["event"] == "rebound"].empty


def test_a_foul_is_never_read_as_possession(tmp_path):
    """A foul is usually committed by the team WITHOUT the ball — counting it inverts the side."""
    cleaned = _parse(tmp_path, [
        _miss("Alice", "LAL"), _miss("Frank", "BOS"),
        _miss("Frank", "BOS"), _BARE,
        {"event_type": "foul", "player": "Alice", "team": "LAL", "type": "personal",
         "opponent": "Gus", "result": None},
        _miss("Gus", "BOS"),                               # BOS actually had it -> offensive
    ])
    reb = cleaned[cleaned["event"] == "rebound"].iloc[0]
    assert reb["type"] == "team offensive"


# ---------------------------------------------------------------------------
# The fouled player — foul rows name who drew the foul
# ---------------------------------------------------------------------------

def test_foul_rows_carry_the_fouled_player(tmp_path):
    """Before 2.0 every foul row was secondary_player="none" — no ground truth anywhere."""
    cleaned = _parse(tmp_path, [
        {"event_type": "foul", "player": "Frank", "type": "personal", "opponent": "Bob",
         "result": None},
    ])
    foul = cleaned[cleaned["event"] == "foul"].iloc[0]
    assert foul["player"] == "Frank" and foul["secondary_player"] == "Bob"


def test_a_technical_has_no_fouled_player(tmp_path):
    """The raw `opponent` column is empty for 100% of technicals — nobody is fouled."""
    cleaned = _parse(tmp_path, [
        {"event_type": "foul", "player": "Frank", "type": "technical", "opponent": None,
         "result": None},
    ])
    assert cleaned[cleaned["event"] == "foul"].iloc[0]["secondary_player"] == "none"


def test_an_offensive_foul_names_the_defender_who_drew_it(tmp_path):
    cleaned = _parse(tmp_path, [
        {"event_type": "foul", "player": "Alice", "type": "offensive charge",
         "opponent": "Ivy", "result": None},
    ])
    foul = cleaned[cleaned["event"] == "foul"].iloc[0]
    assert (foul["type"], foul["secondary_player"]) == ("offensive", "Ivy")


def test_a_blank_opponent_falls_back_to_none(tmp_path):
    for blank in (None, "", "   "):
        cleaned = _parse(tmp_path, [
            {"event_type": "foul", "player": "Frank", "type": "personal", "opponent": blank,
             "result": None},
        ])
        assert cleaned[cleaned["event"] == "foul"].iloc[0]["secondary_player"] == "none"


# ---------------------------------------------------------------------------
# Learned free-throw counts — the shooting foul splits into 2pt / 3pt
# ---------------------------------------------------------------------------

def _ft(num, outof, player="Alice"):
    return {"event_type": "free throw", "player": player, "type": "free throw",
            "num": num, "outof": outof, "result": "made"}


def test_shooting_foul_is_labelled_from_the_following_trips_outof(tmp_path):
    two = _parse(tmp_path, [
        {"event_type": "foul", "player": "Frank", "type": "shooting", "result": None},
        _ft(1, 2), _ft(2, 2),
    ])
    three = _parse(tmp_path, [
        {"event_type": "foul", "player": "Frank", "type": "shooting", "result": None},
        _ft(1, 3), _ft(2, 3), _ft(3, 3),
    ])
    assert two[two["event"] == "foul"].iloc[0]["type"] == "shooting 2pt"
    assert three[three["event"] == "foul"].iloc[0]["type"] == "shooting 3pt"


def test_the_trip_is_found_across_an_intervening_substitution(tmp_path):
    cleaned = _parse(tmp_path, [
        {"event_type": "foul", "player": "Frank", "type": "shooting", "result": None},
        {"event_type": "substitution", "player": None, "entered": "Kim", "left": "Bob",
         "type": None, "result": None},
        _ft(1, 3), _ft(2, 3), _ft(3, 3),
    ])
    assert cleaned[cleaned["event"] == "foul"].iloc[0]["type"] == "shooting 3pt"


def test_an_and_one_is_labelled_from_the_basket_it_followed(tmp_path):
    """outof == 1 is an and-1 (24% of shooting fouls); the attempt's value is the made basket's."""
    two = _parse(tmp_path, [
        {"event_type": "shot", "player": "Alice", "type": "jump shot", "result": "made",
         "points": 2, "converted_x": 25.0, "converted_y": 10.0, "shot_distance": 5},
        {"event_type": "foul", "player": "Frank", "type": "shooting", "result": None},
        _ft(1, 1),
    ])
    three = _parse(tmp_path, [
        {"event_type": "shot", "player": "Alice", "type": "3pt jump shot", "result": "made",
         "points": 3, "converted_x": 25.0, "converted_y": 30.0, "shot_distance": 25},
        {"event_type": "foul", "player": "Frank", "type": "shooting", "result": None},
        _ft(1, 1),
    ])
    assert two[two["event"] == "foul"].iloc[0]["type"] == "shooting 2pt"
    assert three[three["event"] == "foul"].iloc[0]["type"] == "shooting 3pt"


def test_a_shooting_foul_with_no_trip_falls_back_to_two(tmp_path):
    cleaned = _parse(tmp_path, [
        {"event_type": "foul", "player": "Frank", "type": "shooting", "result": None},
    ])
    assert cleaned[cleaned["event"] == "foul"].iloc[0]["type"] == "shooting 2pt"


def test_the_bare_shooting_token_is_gone_from_the_cleaned_data(tmp_path):
    cleaned = _parse(tmp_path, [
        {"event_type": "foul", "player": "Frank", "type": "shooting", "result": None},
        _ft(1, 2), _ft(2, 2),
    ])
    assert "shooting" not in set(cleaned["type"])


def test_other_foul_types_are_untouched(tmp_path):
    cleaned = _parse(tmp_path, [
        {"event_type": "foul", "player": "Frank", "type": "personal", "result": None},
        {"event_type": "foul", "player": "Gus", "type": "offensive charge", "result": None},
    ])
    assert list(cleaned[cleaned["event"] == "foul"]["type"]) == ["personal", "offensive"]


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
    assert block["type"] == "paint"            # the shot's ZONE (not the victim's name)
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

def test_steal_home_away_is_the_ball_losers(tmp_path):
    """Frank (away) steals from Alice (home): the row belongs to Alice, who lost it."""
    row = {"event_type": "turnover", "player": "Alice", "steal": "Frank"}
    cleaned = _parse(tmp_path, [row])
    turnovers = cleaned[cleaned["event"] == "turnover"]
    assert len(turnovers) == 1
    turnover = turnovers.iloc[0]
    assert turnover["home/away"] == 1     # Alice is home (home_indicator: 1=home, 2=away)
    assert turnover["secondary_player"] == "Frank"


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
