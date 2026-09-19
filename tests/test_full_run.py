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


# ---------------------------------------------------------------- rung 2's pre-flight
# The first 3.2 full train died three epochs into 'event_time': ROLLOUT_SELECTION was on, the
# callback fired at ROLLOUT_EVAL_EVERY, and GameSimulator.load raised on an artifacts root the
# train itself was still filling. A rollout scores the WHOLE bundle, so rung 2 belongs to the
# second pass -- these pin both halves of that: the refusal, and the channel the second pass needs.

def _rollout_state(tmp_path, artifacts_root, **over):
    s = {"data_dir": "./data", "processed_dir": str(tmp_path / "proc"),
         "artifacts_root": str(artifacts_root), "reports_root": str(tmp_path / "reports"),
         "epochs": 1, "batch_size": 8, "n_games": 300, "boundary_idx": 158,
         "holdout_game_ids": [1, 2, 3], "train_tail_game_ids": [1, 2],
         "eval_batch": 10, "run_name": "r", "status": "setup", "trained_models": []}
    s.update(over)
    p = tmp_path / "state.json"; p.write_text(json.dumps(s), encoding="utf-8")
    return str(p)


def _fake_bundle(root, keys):
    """Weight files with no weights in them: the pre-flight tests existence, never contents."""
    for k in keys:
        (root / k).mkdir(parents=True, exist_ok=True)
        (root / k / f"{k}.weights.h5").write_bytes(b"")


def test_missing_heads_counts_what_is_not_on_disk(tmp_path):
    from models.registry import STAGE_MODEL_KEYS
    from training.full_run import missing_heads

    root = tmp_path / "artifacts"
    assert missing_heads(root) == list(STAGE_MODEL_KEYS)
    _fake_bundle(root, STAGE_MODEL_KEYS[:-1])
    assert missing_heads(root) == [STAGE_MODEL_KEYS[-1]]
    _fake_bundle(root, STAGE_MODEL_KEYS)
    assert missing_heads(root) == []


def test_rung2_refuses_a_bundle_still_being_built(tmp_path, monkeypatch, capsys):
    import config

    monkeypatch.setattr(config, "ROLLOUT_SELECTION", True, raising=False)
    run = FullRun(state_path=_rollout_state(tmp_path, tmp_path / "artifacts"))

    assert run._rollout_score_factory() == (None, None)
    out = capsys.readouterr().out
    # The refusal has to name the remedy, or it reads as the feature silently not working.
    assert "rung 2" in out and "python train.py --model event_time" in out


def test_rung2_arms_itself_on_a_finished_bundle(tmp_path, monkeypatch):
    import config
    from models.registry import STAGE_MODEL_KEYS

    monkeypatch.setattr(config, "ROLLOUT_SELECTION", True, raising=False)
    root = tmp_path / "artifacts"
    _fake_bundle(root, STAGE_MODEL_KEYS)
    run = FullRun(state_path=_rollout_state(tmp_path, root))

    factory, holder = run._rollout_score_factory()
    assert factory is not None and holder == {}


def test_retrain_hands_the_rollout_channel_to_run_stage(tmp_path, monkeypatch):
    """Arm 2 is `train.py --model event_time`. It used to build no factory and pass none."""
    import models.pipeline as pipeline
    import config
    import training.full_run as full_run_mod
    from models.registry import STAGE_MODEL_KEYS

    monkeypatch.setattr(config, "ROLLOUT_SELECTION", True, raising=False)
    root = tmp_path / "artifacts"
    _fake_bundle(root, STAGE_MODEL_KEYS)
    run = FullRun(state_path=_rollout_state(tmp_path, root))

    monkeypatch.setattr(FullRun, "_subset_games", lambda self, *, tag: {1, 2})
    monkeypatch.setattr(full_run_mod, "game_index", lambda data_dir: "idx")
    monkeypatch.setattr(full_run_mod, "sequential_partition", lambda *a, **k: ({1}, {2}, {3}))
    seen = {}
    monkeypatch.setattr(pipeline, "run_stage", lambda *a, **k: seen.update(k))

    run.retrain_model("event_time")
    assert seen["rollout_score_fn_factory"] is not None
