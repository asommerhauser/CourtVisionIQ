"""
Write a TrainingReport into the queryable Parquet data model.

Two files per run (kept in the run folder, see report_artifacts.py):

  run.parquet     One row: the run-level summary. Scalar columns for the common
                  fields everyone filters on (model_key, status, batch_size, lr,
                  best_val_loss, duration ...) plus nested config/env/data/model
                  as JSON strings so nothing is lost while the schema stays flat.

  epochs.parquet  Long format: one row per (epoch, metric). Columns:
                  [run_id, model_key, epoch, split, metric, value, duration_sec,
                  learning_rate]. Long format unions cleanly across models that
                  emit different metrics — the schema never changes when a new
                  model is added.

Aggregate querying is done by globbing every run/epochs parquet across the tree
(see query.py); pandas/pyarrow reads the whole partitioned set in one call.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reporting.schema import TrainingReport


def run_summary_row(report: TrainingReport) -> dict:
    """Flatten a report into the single row stored in run.parquet."""
    cfg = report.config
    env = report.environment
    data = report.data
    minfo = report.model
    return {
        "run_id": report.run_id,
        "model_key": report.model_key,
        "status": report.status,
        "started_at": report.started_at,
        "ended_at": report.ended_at,
        "duration_sec": report.duration_sec,
        "epochs_planned": cfg.epochs_planned if cfg else None,
        "epochs_run": report.epochs_run,
        "best_epoch": report.best_epoch,
        "best_val_loss": report.best_val_loss,
        "batch_size": cfg.batch_size if cfg else None,
        "lr": cfg.lr if cfg else None,
        "time_loss_weight": cfg.time_loss_weight if cfg else None,
        "device": env.device if env else None,
        "git_commit": env.git_commit if env else None,
        "total_params": minfo.total_params if minfo else None,
        "train_games": data.train_games if data else None,
        "test_games": data.test_games if data else None,
        # Full nested state preserved as JSON so the row schema stays flat/stable.
        "config_json": json.dumps(cfg.to_dict()) if cfg else None,
        "environment_json": json.dumps(env.to_dict()) if env else None,
        "data_json": json.dumps(data.to_dict()) if data else None,
        "model_json": json.dumps(minfo.to_dict()) if minfo else None,
        "final_test_metrics_json": json.dumps(report.final_test_metrics),
    }


def epoch_rows(report: TrainingReport) -> list[dict]:
    """Explode a report's epochs into long-format rows for epochs.parquet."""
    rows: list[dict] = []
    for rec in report.epochs:
        for metric, value in rec.metrics.items():
            split = "val" if metric.startswith("val_") else "train"
            name = metric[4:] if metric.startswith("val_") else metric
            rows.append({
                "run_id": report.run_id,
                "model_key": report.model_key,
                "epoch": rec.epoch,
                "split": split,
                "metric": name,
                "value": float(value),
                "duration_sec": rec.duration_sec,
                "learning_rate": rec.learning_rate,
            })
    return rows


def write_run(report: TrainingReport, path: str | Path) -> Path:
    path = Path(path)
    pd.DataFrame([run_summary_row(report)]).to_parquet(path, index=False)
    return path


def write_epochs(report: TrainingReport, path: str | Path) -> Path:
    path = Path(path)
    rows = epoch_rows(report)
    # Stable column order even when there are no epochs (failed run).
    cols = ["run_id", "model_key", "epoch", "split", "metric",
            "value", "duration_sec", "learning_rate"]
    pd.DataFrame(rows, columns=cols).to_parquet(path, index=False)
    return path
