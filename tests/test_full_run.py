"""Full-run driver: the mid-last-season cut + next-N holdout, and command gating."""
import json

import pandas as pd

from config import FINAL_HOLDOUT_GAMES, HOLDOUT_WINDOW_GAMES, FULL_ARTIFACTS_ROOT
from training.chronology import game_index
from training.full_run import FullRun


def _season_csv(path, season, n_reg, n_po):
    rows, gid = [], 0
    base = pd.Timestamp(f"{season - 1}-10-15")
    for phase, count in ((1, n_reg), (2, n_po)):
        for _ in range(count):
            gid += 1
            date = (base + pd.Timedelta(days=gid)).strftime("%Y-%m-%d")
            for i, ev in enumerate(("start", "shot", "end")):
                rows.append({
                    "game_id": gid, "roster_home": str(["A", "B", "C", "D", "E"]),
                    "roster_away": str(["F", "G", "H", "I", "J"]), "time": i * 10, "event": ev,
                    "player": "A", "type": "t", "result": "r", "secondary_player": "none",
                    "season": season, "playoff": phase, "game_date": date,
                    "rest_home": str([2.0] * 5), "rest_away": str([2.0] * 5),
                    "home_games_played": 0.5, "away_games_played": 0.5,
                    "home_days_rest": 2.0, "away_days_rest": 2.0,
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_setup_cuts_mid_last_season(tmp_path):
    data_dir = tmp_path / "data"; data_dir.mkdir()
    _season_csv(data_dir / "season2003.csv", 2003, 20, 4)
    _season_csv(data_dir / "season2004.csv", 2004, 20, 4)
    _season_csv(data_dir / "season2005.csv", 2005, 220, 10)   # big last season for a 100 holdout

    run = FullRun(state_path=str(tmp_path / "state.json"))
    run.setup(data_dir=str(data_dir), processed_dir=str(tmp_path / "proc"))

    st = run.state
    assert st["status"] == "setup"
    assert st["artifacts_root"] == FULL_ARTIFACTS_ROOT
    # Cut at 50% of 2005's 220 regular games: 2005 reg starts at pos 48 -> boundary 48 + 110 = 158.
    assert st["boundary_idx"] == 158
    # The pool is a TARGET, clamped to the tail that actually exists. FINAL_HOLDOUT_GAMES is 700
    # since 3.0 (seven rotating 100-game windows) against 702 games after the cut on the real
    # corpus -- two games of slack. A small corpus like this fixture takes what it has rather than
    # refusing to set up, and window_ids() warns when a window comes out short.
    available = len(idx_all := game_index(str(data_dir))) - st["boundary_idx"]
    assert len(st["holdout_game_ids"]) == min(FINAL_HOLDOUT_GAMES, available)
    assert len(st["holdout_game_ids"]) == available < FINAL_HOLDOUT_GAMES

    # The pool is exactly the contiguous tail after the cut -- that is the invariant, and it is
    # what makes the rotating windows disjoint slices of a known range.
    ordered = [int(g) for g in idx_all["game_id"]]
    assert st["holdout_game_ids"] == ordered[st["boundary_idx"]:]

    # Window 0 -- the games every pre-3.0 run scored -- is all 2005 regular season. Later windows
    # of a pool this small reach into the playoffs, which is correct: the pool is "every untrained
    # game", and a window is reported with its k precisely so that is legible.
    idx = idx_all.set_index("game_id")
    window0 = st["holdout_game_ids"][:HOLDOUT_WINDOW_GAMES]
    for g in window0:
        assert int(idx.loc[g, "season"]) == 2005 and int(idx.loc[g, "playoff"]) == 1


def _state(tmp_path, status):
    s = {"data_dir": "./data", "processed_dir": "./data/processed",
         "artifacts_root": FULL_ARTIFACTS_ROOT, "reports_root": "./reports",
         "epochs": 1, "batch_size": 8, "n_games": 300, "boundary_idx": 158,
         "holdout_game_ids": [1, 2, 3], "eval_batch": 10, "run_name": "full_train",
         "status": status, "trained_models": []}
    p = tmp_path / "state.json"; p.write_text(json.dumps(s), encoding="utf-8")
    return str(p)


def test_eval_before_train_is_a_noop(tmp_path):
    run = FullRun(state_path=_state(tmp_path, status="setup"))
    run.eval()                          # not trained -> returns before touching the simulator
    assert run.state["status"] == "setup"


def test_train_when_trained_is_a_noop(tmp_path):
    run = FullRun(state_path=_state(tmp_path, status="trained"))
    run.train()                         # already trained -> no retrain
    assert run.state["status"] == "trained"
