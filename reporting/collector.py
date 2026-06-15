"""
Orchestrates report capture around a `model.fit` call.

A single `ReportCollector` is created inside a model's `train()`. It:
  1. exposes `.callback` to attach to `model.fit` (per-epoch timing + metrics),
  2. snapshots environment, data, and model size before/around training,
  3. on `finalize()`, assembles a `TrainingReport` and writes all four output
     files (report.html, report.json, run.parquet, epochs.parquet).

Keeping all capture logic here — driven only by the generic `RunConfig` and the
captured structures — is what lets every model reuse reporting unchanged.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime

import keras
import tensorflow as tf

from reporting.report_artifacts import ReportArtifacts, new_run_id, DEFAULT_REPORTS_ROOT
from reporting.schema import (
    TrainingReport, RunConfig, EnvInfo, DataInfo, ModelInfo,
)
from reporting import html_report, parquet_store


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _capture_environment() -> EnvInfo:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        device = f"GPU x{len(gpus)}"
        names = [g.name for g in gpus]
    else:
        device = "CPU"
        names = []
    try:
        policy = keras.mixed_precision.global_policy().name
    except Exception:
        policy = ""
    return EnvInfo(
        python_version=sys.version.split()[0],
        tensorflow_version=tf.__version__,
        keras_version=keras.__version__,
        platform=platform.platform(),
        device=device,
        device_names=names,
        mixed_precision_policy=policy,
        git_commit=_git_commit(),
    )


class ReportCollector:
    """Capture state across a training run and emit the standardized report."""

    def __init__(self, config: RunConfig, run_id: str | None = None,
                 run_name: str | None = None,
                 reports_root: str = DEFAULT_REPORTS_ROOT):
        self.config = config
        self.run_id = run_id or new_run_id(run_name)
        self.artifacts = ReportArtifacts.for_run(
            config.model_key, self.run_id, reports_root
        )
        from reporting.callback import ReportingCallback  # local import: keras dep
        self.callback = ReportingCallback()

        self.environment: EnvInfo = _capture_environment()
        self.data: DataInfo | None = None
        self.model: ModelInfo | None = None
        self._started = datetime.now()

    # --- capture hooks (called from train) ---

    def capture_data(self, train_games: int, test_games: int,
                     sequence_length: int, vocab_sizes: dict,
                     norm_stats: dict | None) -> None:
        self.data = DataInfo(
            train_games=int(train_games),
            test_games=int(test_games),
            sequence_length=int(sequence_length),
            vocab_sizes=dict(vocab_sizes or {}),
            norm_stats=dict(norm_stats or {}),
        )

    def capture_model(self, model) -> None:
        try:
            trainable = int(sum(v.numpy().size for v in model.trainable_variables))
            non_trainable = int(sum(v.numpy().size for v in model.non_trainable_variables))
        except Exception:
            trainable = int(model.count_params())
            non_trainable = 0
        self.model = ModelInfo(
            total_params=trainable + non_trainable,
            trainable_params=trainable,
            non_trainable_params=non_trainable,
            num_layers=len(model.layers),
        )

    # --- finalize ---

    def finalize(self, status: str = "completed",
                 final_test_metrics: dict | None = None) -> ReportArtifacts:
        ended = datetime.now()
        records = self.callback.records

        best_epoch = None
        best_val_loss = None
        for rec in records:
            vl = rec.metrics.get("val_loss")
            if vl is not None and (best_val_loss is None or vl < best_val_loss):
                best_val_loss = vl
                best_epoch = rec.epoch

        report = TrainingReport(
            run_id=self.run_id,
            model_key=self.config.model_key,
            status=status,
            started_at=self._started.isoformat(timespec="seconds"),
            ended_at=ended.isoformat(timespec="seconds"),
            duration_sec=round((ended - self._started).total_seconds(), 2),
            epochs_run=len(records),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            config=self.config,
            environment=self.environment,
            data=self.data,
            model=self.model,
            epochs=records,
            final_test_metrics=final_test_metrics or {},
        )
        self.write(report)
        return self.artifacts

    def write(self, report: TrainingReport) -> ReportArtifacts:
        self.artifacts.ensure_dir()
        self.artifacts.json_path.write_text(report.to_json(), encoding="utf-8")
        self.artifacts.html_path.write_text(html_report.render(report), encoding="utf-8")
        parquet_store.write_run(report, self.artifacts.run_parquet_path)
        parquet_store.write_epochs(report, self.artifacts.epochs_parquet_path)
        return self.artifacts
