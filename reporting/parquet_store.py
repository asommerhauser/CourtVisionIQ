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
    """Flatten a report into the single row stored in run.parquet.

    Every field that could drive a comparison query is a native typed column.
    The *_json blobs remain for lossless round-trips and future fields.
    """
    cfg = report.config
    env = report.environment
    data = report.data
    minfo = report.model
    arch = (cfg.arch or {}) if cfg else {}
    vocab = (data.vocab_sizes or {}) if data else {}
    norm = (data.norm_stats or {}) if data else {}
    tm = report.final_test_metrics or {}

    return {
        # --- identity ---
        "run_id": report.run_id,
        "run_name": report.run_name,
        "model_key": report.model_key,
        "status": report.status,
        "started_at": report.started_at,
        "ended_at": report.ended_at,
        "duration_sec": report.duration_sec,
        "git_commit": env.git_commit if env else None,

        # --- training outcome ---
        "epochs_planned": cfg.epochs_planned if cfg else None,
        "epochs_run": report.epochs_run,
        "best_epoch": report.best_epoch,
        "best_val_loss": report.best_val_loss,

        # --- optimizer / schedule hyperparameters ---
        "lr": cfg.lr if cfg else None,
        "batch_size": cfg.batch_size if cfg else None,
        "time_loss_weight": cfg.time_loss_weight if cfg else None,
        "patience": cfg.patience if cfg else None,
        "mixed_precision": cfg.mixed_precision if cfg else None,
        "jit_compile": cfg.jit_compile if cfg else None,

        # --- architecture hyperparameters ---
        "model_dim": arch.get("model_dim"),
        "num_layers": arch.get("num_layers"),
        "num_heads": arch.get("num_heads"),
        "ff_dim": arch.get("ff_dim"),
        "dropout": arch.get("dropout"),
        "sequence_length": arch.get("sequence_length"),
        "roster_dim": arch.get("roster_dim"),

        # --- model size ---
        "total_params": minfo.total_params if minfo else None,
        "trainable_params": minfo.trainable_params if minfo else None,
        "non_trainable_params": minfo.non_trainable_params if minfo else None,

        # --- data split ---
        "train_games": data.train_games if data else None,
        "test_games": data.test_games if data else None,

        # --- vocab sizes (flat, for spotting data-scale differences) ---
        "vocab_event": vocab.get("event"),
        "vocab_player": vocab.get("player"),
        "vocab_type": vocab.get("type"),
        "vocab_result": vocab.get("result"),
        "vocab_season": vocab.get("season"),

        # --- time normalization stats ---
        "norm_max_time": norm.get("max_time"),
        "norm_delta_mean": norm.get("delta_mean"),
        "norm_delta_std": norm.get("delta_std"),

        # --- hardware ---
        "device": env.device if env else None,

        # --- final held-out test metrics (flat for leaderboard queries) ---
        "test_loss": tm.get("loss"),
        "test_event_loss": tm.get("event_output_loss"),
        "test_time_loss": tm.get("time_output_loss"),
        "test_event_acc": tm.get("event_output_acc"),
        "test_time_mae": tm.get("time_output_mae"),

        # --- full nested state (lossless; schema-stable escape hatch) ---
        "config_json": json.dumps(cfg.to_dict()) if cfg else None,
        "environment_json": json.dumps(env.to_dict()) if env else None,
        "data_json": json.dumps(data.to_dict()) if data else None,
        "model_json": json.dumps(minfo.to_dict()) if minfo else None,
        "final_test_metrics_json": json.dumps(tm),
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
