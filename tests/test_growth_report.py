"""Growth report stitches per-stage training loss + eval headline into one table."""
import json

import pandas as pd

from reporting.growth_report import build_growth_report


def _write_training_run(reports_root, run_name, model_key, best_val_loss):
    d = reports_root / model_key / f"{model_key}-run"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "run_name": run_name, "model_key": model_key, "best_val_loss": best_val_loss,
        "started_at": "2026-06-24T00:00:00",
    }]).to_parquet(d / "run.parquet", index=False)


def _write_eval_report(eval_dir, headline):
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "report.json").write_text(
        json.dumps({"aggregate": {"headline": headline}}), encoding="utf-8")


def test_growth_report_collects_training_and_eval(tmp_path):
    reports_root = tmp_path / "reports"
    run_name = "stage01_frac0.25_s2005"
    _write_training_run(reports_root, run_name, "event_time", 1.23)
    _write_training_run(reports_root, run_name, "player", 0.50)

    eval_dir = tmp_path / "eval_run"
    _write_eval_report(eval_dir, {"pick_accuracy": 0.7, "brier": 0.21,
                                  "spread_mae": 9.5, "points_mae": 6.1})

    state = {
        "n_games": 200,
        "schedule": [{"stage": 1, "boundary_type": "frac:0.25", "season": 2005,
                      "train_games": 100, "holdout_game_ids": [1, 2]}],
        "stages": {"1": {"status": "evaluated", "report_run_name": run_name,
                         "eval_run_dir": str(eval_dir)}},
    }

    run_dir = build_growth_report(state, reports_root=str(reports_root))
    df = pd.read_parquet(__import__("pathlib").Path(run_dir) / "growth.parquet")

    row = df.iloc[0]
    assert row["train_games"] == 100
    assert abs(row["val_event_time"] - 1.23) < 1e-9
    assert abs(row["mean_best_val_loss"] - (1.23 + 0.50) / 2) < 1e-9
    assert abs(row["pick_accuracy"] - 0.7) < 1e-9 and abs(row["spread_mae"] - 9.5) < 1e-9
    assert (__import__("pathlib").Path(run_dir) / "growth.html").exists()


def test_growth_report_handles_no_runs(tmp_path):
    # No training runs / no eval yet: still emits a table of pending stages without crashing.
    state = {
        "n_games": 50,
        "schedule": [{"stage": 1, "boundary_type": "frac:0.25", "season": 2005,
                      "train_games": 30, "holdout_game_ids": [1]}],
        "stages": {},
    }
    run_dir = build_growth_report(state, reports_root=str(tmp_path / "reports"))
    df = pd.read_parquet(__import__("pathlib").Path(run_dir) / "growth.parquet")
    assert df.iloc[0]["status"] == "pending"
