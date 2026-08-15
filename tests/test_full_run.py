"""Full-run driver: the mid-last-season cut + next-N holdout, and command gating."""
import json

import pandas as pd

from config import FINAL_HOLDOUT_GAMES, FULL_ARTIFACTS_ROOT
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
    assert len(st["holdout_game_ids"]) == FINAL_HOLDOUT_GAMES

    # Every holdout game is a 2005 regular-season game, contiguous right after the boundary.
    from training.chronology import game_index
    idx = game_index(str(data_dir)).set_index("game_id")
    for g in st["holdout_game_ids"]:
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


def test_parse_shard():
    from evaluate import parse_shard
    assert parse_shard("1/1") == (1, 1)
    assert parse_shard("3/5") == (3, 5)
    for bad in ("0/5", "6/5", "0/0", "a/b", "5", "1/5/2", "-1/5"):
        try:
            parse_shard(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse_shard({bad!r}) should have raised")


def test_shard_slices_are_disjoint_and_complete():
    ids = list(range(103))              # deliberately not divisible by n
    n = 5
    slices = [ids[i - 1::n] for i in range(1, n + 1)]
    flat = [g for s in slices for g in s]
    assert len(flat) == len(ids)        # disjoint (no double-simulated game)
    assert sorted(flat) == ids          # complete (no dropped game)


def test_eval_shard_slices_holdout_and_skips_report(tmp_path, monkeypatch):
    """A sharded eval must pass its slice + write_report=False and leave the run state untouched."""
    state = _state(tmp_path, status="trained")
    run = FullRun(state_path=state)
    run.state["version"] = "9.9"
    run.state["holdout_game_ids"] = list(range(10))

    captured = {}

    def fake_evaluate_stage(stage_name, *, holdout_ids, write_report, **kwargs):
        captured["holdout_ids"] = holdout_ids
        captured["write_report"] = write_report
        return {"done": len(holdout_ids), "total": len(holdout_ids),
                "run_dir": str(tmp_path / "run")}

    import reporting.eval_report as eval_report
    import simulation.stage_eval as stage_eval
    monkeypatch.setattr(stage_eval, "evaluate_stage", fake_evaluate_stage)
    monkeypatch.setattr(eval_report, "resolve_results_run_dir",
                        lambda *a, **k: tmp_path / "run")

    run.eval(name="sharded", shard=(2, 3))
    assert captured["holdout_ids"] == [1, 4, 7]      # holdout[1::3]
    assert captured["write_report"] is False
    assert "last_eval_name" not in run.state          # sharded calls don't touch the state

    run.eval(name="whole")                            # unsharded: full list, report written
    assert captured["holdout_ids"] == list(range(10))
    assert captured["write_report"] is True
    assert run.state["last_eval_name"] == "run"


def test_train_when_trained_is_a_noop(tmp_path):
    run = FullRun(state_path=_state(tmp_path, status="trained"))
    run.train()                         # already trained -> no retrain
    assert run.state["status"] == "trained"
