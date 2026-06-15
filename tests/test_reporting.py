"""
Unit coverage for the standardized reporting layer (reporting/ package).

These tests exercise the report pipeline without training the real model: a tiny
synthetic Keras model drives the per-epoch callback, and a hand-built
TrainingReport drives the writers/renderer. Like the persistence tests, the
design is model-agnostic — nothing here is specific to the Event/Time model, so
any future model inherits the same guarantees.
"""
from __future__ import annotations

import keras
import numpy as np
import pandas as pd
import pytest

from reporting.schema import (
    TrainingReport, RunConfig, EnvInfo, DataInfo, ModelInfo, EpochRecord,
)
from reporting.report_artifacts import ReportArtifacts, new_run_id
from reporting.callback import ReportingCallback
from reporting import html_report, parquet_store


def _make_report(run_id: str = "20260614-000000-unit") -> TrainingReport:
    cfg = RunConfig(
        model_key="event_time", epochs_planned=2, batch_size=8, lr=1e-3,
        time_loss_weight=0.25, patience=3, mixed_precision=False,
        jit_compile=False, arch={"model_dim": 16},
    )
    return TrainingReport(
        run_id=run_id, model_key="event_time", status="completed",
        started_at="2026-06-14T00:00:00", ended_at="2026-06-14T00:01:00",
        duration_sec=60.0, epochs_run=2, best_epoch=1, best_val_loss=0.5,
        config=cfg, environment=EnvInfo(device="CPU"),
        data=DataInfo(train_games=10, test_games=3, sequence_length=16,
                      vocab_sizes={"event": 7, "player": 50}),
        model=ModelInfo(total_params=1000, trainable_params=1000, num_layers=5),
        epochs=[
            EpochRecord(0, 1.0, 1e-3, {"loss": 1.0, "acc": 0.4, "val_loss": 0.8, "val_acc": 0.45}),
            EpochRecord(1, 1.0, 1e-3, {"loss": 0.6, "acc": 0.7, "val_loss": 0.5, "val_acc": 0.72}),
        ],
        final_test_metrics={"loss": 0.5, "acc": 0.72},
    )


def test_run_id_is_unique_and_sortable():
    a, b = new_run_id(), new_run_id()
    assert a != b
    assert new_run_id("My Run!").endswith("my-run")


def test_callback_records_one_epoch_each_with_timing():
    model = keras.Sequential([keras.layers.Dense(1, input_shape=(3,))])
    model.compile(optimizer="sgd", loss="mse")
    x = np.random.rand(16, 3).astype("float32")
    y = np.random.rand(16, 1).astype("float32")

    cb = ReportingCallback()
    model.fit(x, y, validation_data=(x, y), epochs=3, verbose=0, callbacks=[cb])

    assert len(cb.records) == 3
    for i, rec in enumerate(cb.records):
        assert rec.epoch == i
        assert rec.duration_sec >= 0.0
        assert "loss" in rec.metrics and "val_loss" in rec.metrics
        assert all(np.isfinite(v) for v in rec.metrics.values())


def test_schema_json_round_trip():
    rep = _make_report()
    rt = TrainingReport.from_dict(rep.to_dict())
    assert rt.run_id == rep.run_id
    assert rt.best_val_loss == rep.best_val_loss
    assert len(rt.epochs) == len(rep.epochs)
    assert rt.config.batch_size == rep.config.batch_size


def test_html_is_self_contained_with_graphs():
    html = html_report.render(_make_report())
    assert _make_report().run_id in html
    assert "data:image/png;base64," in html      # embedded graph
    assert "Epoch-by-epoch metrics" in html


def test_parquet_round_trip(tmp_path):
    rep = _make_report()
    run_path = tmp_path / "run.parquet"
    eps_path = tmp_path / "epochs.parquet"
    parquet_store.write_run(rep, run_path)
    parquet_store.write_epochs(rep, eps_path)

    runs = pd.read_parquet(run_path)
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == rep.run_id
    assert runs.iloc[0]["best_val_loss"] == 0.5
    assert runs.iloc[0]["batch_size"] == 8

    eps = pd.read_parquet(eps_path)
    assert list(eps.columns) == [
        "run_id", "model_key", "epoch", "split", "metric",
        "value", "duration_sec", "learning_rate",
    ]
    # 2 epochs x (loss + acc) x (train + val) = 8 rows.
    assert len(eps) == 8
    assert set(eps["split"]) == {"train", "val"}
    assert set(eps["metric"]) == {"loss", "acc"}


def test_report_artifacts_paths(tmp_path):
    arts = ReportArtifacts.for_run("event_time", "run123", tmp_path)
    assert arts.run_dir == tmp_path / "event_time" / "run123"
    assert arts.html_path.name == "report.html"
    assert arts.json_path.name == "report.json"
    assert arts.run_parquet_path.name == "run.parquet"
    assert arts.epochs_parquet_path.name == "epochs.parquet"
    arts.ensure_dir()
    assert arts.run_dir.exists()


def test_query_helpers_glob_tree(tmp_path):
    from reporting import query

    # Two runs under one model.
    for rid in ("20260614-000001-a", "20260614-000002-b"):
        arts = ReportArtifacts.for_run("event_time", rid, tmp_path)
        arts.ensure_dir()
        rep = _make_report(rid)
        parquet_store.write_run(rep, arts.run_parquet_path)
        parquet_store.write_epochs(rep, arts.epochs_parquet_path)

    runs = query.load_runs(tmp_path)
    assert len(runs) == 2
    eps = query.load_epochs(tmp_path)
    assert len(eps) == 16

    curve = query.learning_curve("20260614-000001-a", "loss", tmp_path)
    assert "train" in curve.columns and "val" in curve.columns
    assert len(curve) == 2
