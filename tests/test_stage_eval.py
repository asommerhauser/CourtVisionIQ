"""Stage-eval pure helpers: descriptive game labels + averaged predicted box score."""
import pandas as pd

from simulation.stage_eval import _averaged_box, _game_labels
from simulation.stats import BOX_STATS


def _record(gid=5):
    home = {"A": {f: 1.0 for f in BOX_STATS}, "B": {f: 0.0 for f in BOX_STATS}}
    away = {"F": {f: 2.0 for f in BOX_STATS}}
    return {
        "game_id": gid,
        "player_avg": {"home": home, "away": away},
        "pred_home_score": 100.4, "pred_away_score": 98.6,
    }


def test_game_labels_from_teams_and_date():
    game = pd.DataFrame([{"game_id": 5, "home_team": "PHI", "away_team": "ORL",
                          "game_date": "2002-11-01"}])
    home, away, label = _game_labels(game)
    assert (home, away) == ("PHI", "ORL")
    assert label.startswith("game5_") and "ORLatPHI" in label
    # No path separators or spaces leak into the folder name.
    assert "/" not in label and " " not in label


def test_game_labels_fall_back_when_team_missing():
    game = pd.DataFrame([{"game_id": 7, "home_team": None, "away_team": "", "game_date": None}])
    home, away, label = _game_labels(game)
    assert (home, away) == ("HOME", "AWAY")
    assert label == "game7_AWAYatHOME"


def test_averaged_box_renders_with_score():
    box = _averaged_box(_record(), "PHI", "ORL")
    assert box.home_team == "PHI" and box.away_team == "ORL"
    assert box.home_score == 100.4 and box.away_score == 98.6
    # Averaged float stats render through the standard box-score path without error.
    frame = box.to_frame("home")
    assert "TEAM" in frame["Player"].tolist()
    assert "PHI" in box.render()
