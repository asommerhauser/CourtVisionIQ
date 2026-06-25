"""Curriculum state-machine gating: no training is launched on the wrong command/stage."""
import json

from training.curriculum import Curriculum, STAGE_MODEL_KEYS


def _state(tmp_path, current_stage=1, stage_statuses=None):
    schedule = [
        {"stage": 1, "boundary_type": "frac:0.25", "season": 2005, "boundary_idx": 100,
         "train_games": 100, "holdout_game_ids": [101, 102, 103]},
        {"stage": 2, "boundary_type": "final_full", "season": 2023, "boundary_idx": 200,
         "train_games": 200, "holdout_game_ids": []},
    ]
    stages = stage_statuses or {}
    state = {
        "created_at": "now", "data_dir": "./data", "processed_dir": "./data/processed",
        "artifacts_root": "./artifacts", "reports_root": "./reports",
        "epochs": 1, "batch_size": 8, "n_games": 200,
        "current_stage": current_stage, "schedule": schedule, "stages": stages,
    }
    p = tmp_path / "state.json"
    p.write_text(json.dumps(state), encoding="utf-8")
    return p


def test_eval_before_train_is_a_noop(tmp_path):
    cur = Curriculum(state_path=str(_state(tmp_path, current_stage=1)))
    cur.eval_stage()                       # stage 1 is "pending" -> should not advance or evaluate
    assert cur.state["current_stage"] == 1
    assert cur.state["stages"]["1"]["status"] == "pending"


def test_train_when_already_trained_is_a_noop(tmp_path):
    p = _state(tmp_path, current_stage=1,
               stage_statuses={"1": {"status": "trained", "trained_models": STAGE_MODEL_KEYS}})
    cur = Curriculum(state_path=str(p))
    cur.train_stage()                      # already trained -> stays trained, no retrain
    assert cur.state["stages"]["1"]["status"] == "trained"
    assert cur.state["current_stage"] == 1


def test_final_full_eval_advances_without_holdout(tmp_path):
    # Stage 2 is the final full-corpus stage (empty holdout): eval just marks it done + advances,
    # without running the (model-dependent) stage evaluation.
    p = _state(tmp_path, current_stage=2,
               stage_statuses={"2": {"status": "trained", "trained_models": STAGE_MODEL_KEYS}})
    cur = Curriculum(state_path=str(p))
    cur.eval_stage()
    assert cur.state["stages"]["2"]["status"] == "evaluated"
    assert cur.state["current_stage"] == 3       # advanced past the last stage

    # Reloading from disk sees the persisted advance.
    again = Curriculum(state_path=str(p))
    assert again.state["current_stage"] == 3


def test_status_runs(tmp_path, capsys):
    cur = Curriculum(state_path=str(_state(tmp_path)))
    cur.status()
    out = capsys.readouterr().out
    assert "frac:0.25" in out and "final_full" in out
