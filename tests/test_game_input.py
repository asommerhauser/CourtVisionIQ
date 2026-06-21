"""
Game-input extractor tests.

Verify the whole-roster union (no starter/bench distinction), the playoff 2->1 / 1->0
mapping, defensive roster parsing, and the holdout-manifest round-trip. These import the
extractor directly (no TensorFlow / trained model needed) and run fast.
"""
from __future__ import annotations

import json

from config import HOLDOUT_MANIFEST_NAME
from simulation.game_input import (
    GameInput,
    extract_game_input,
    holdout_game_inputs,
    write_holdout_inputs,
)

HOME = ["A", "B", "C", "D", "E"]
AWAY = ["F", "G", "H", "I", "J"]


def _row(time, event, player, type_, result, season, playoff,
         secondary="none", home=HOME, away=AWAY):
    return {
        "event": event, "player": player, "type": type_, "result": result,
        "secondary_player": secondary, "time": time, "season": season, "playoff": playoff,
        "roster_home": list(home), "roster_away": list(away),
    }


def test_whole_roster_union_no_starter_distinction():
    # A substitution brings bench players X (home) and Y (away) on; the whole roster must
    # include them with no starter/bench split.
    home_after = ["A", "B", "C", "D", "X"]
    away_after = ["F", "G", "H", "I", "Y"]
    events = [
        _row(0, "start", "start", "start", "start", 2003, 1),
        _row(10, "shot", "A", "2pt", "made", 2003, 1),
        _row(20, "substitution", "X", "substitution", "substitution", 2003, 1,
             secondary="E", home=home_after),
        _row(30, "substitution", "Y", "substitution", "substitution", 2003, 1,
             secondary="J", away=away_after),
        _row(40, "end", "end", "end", "end", 2003, 1, home=home_after, away=away_after),
    ]
    gi = extract_game_input(events)
    assert gi.home_roster == sorted(["A", "B", "C", "D", "E", "X"])
    assert gi.away_roster == sorted(["F", "G", "H", "I", "J", "Y"])
    assert gi.season == 2003


def test_playoff_mapping():
    regular = [_row(0, "start", "start", "start", "start", 2003, 1)]
    playoff = [_row(0, "start", "start", "start", "start", 2004, 2)]
    assert extract_game_input(regular).playoff == 0
    assert extract_game_input(playoff).playoff == 1
    assert extract_game_input(playoff).season == 2004


def test_roster_parsing_accepts_strings():
    # Cleaned CSVs carry rosters as string literals; both forms must work.
    rows = [{
        "event": "start", "player": "start", "type": "start", "result": "start",
        "secondary_player": "none", "time": 0, "season": 2003, "playoff": 1,
        "roster_home": "['A', 'B', 'C', 'D', 'E']",
        "roster_away": "['F', 'G', 'H', 'I', 'J']",
    }]
    gi = extract_game_input(rows)
    assert gi.home_roster == sorted(HOME)
    assert gi.away_roster == sorted(AWAY)


def test_holdout_inputs_roundtrip(tmp_path):
    # Tiny cleaned CSV with two games; only game 2 is in the holdout manifest.
    import pandas as pd

    rows = []
    for gid, season, playoff in [(1, 2003, 1), (2, 2004, 2)]:
        for t, ev in [(0, "start"), (10, "shot"), (20, "end")]:
            rows.append({
                "game_id": gid, "roster_home": str(HOME), "roster_away": str(AWAY),
                "time": t, "event": ev, "player": "A" if ev == "shot" else ev,
                "type": "2pt" if ev == "shot" else ev,
                "result": "made" if ev == "shot" else ev,
                "secondary_player": "none", "season": season, "playoff": playoff,
            })
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(data_dir / "season_test.csv", index=False)
    (processed_dir / HOLDOUT_MANIFEST_NAME).write_text(json.dumps([2]), encoding="utf-8")

    inputs = holdout_game_inputs(data_dir=str(data_dir), processed_dir=str(processed_dir))
    assert set(inputs) == {2}
    gi = inputs[2]
    assert gi.season == 2004 and gi.playoff == 1
    assert gi.home_roster == sorted(HOME)

    out_path = write_holdout_inputs(data_dir=str(data_dir), processed_dir=str(processed_dir))
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(payload) == {"2"}
    assert GameInput.from_dict(payload["2"]) == gi
