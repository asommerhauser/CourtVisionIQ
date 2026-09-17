"""
Standardized training/testing reports for CourtVisionIQ models.

Every model reports through one shared contract (mirroring models/artifacts.py):
attach a ReportCollector around `model.fit` and call `finalize()`. Each run emits
a self-contained HTML report plus a queryable Parquet data model.

  from reporting import ReportCollector, RunConfig
  from reporting.query import load_runs, load_epochs

**``ReportCollector`` resolves lazily.** It subclasses a Keras callback, so importing it imports
TensorFlow — and this package also holds the *evaluation* report stack, which is pure
pandas/matplotlib: ``eval_report``, ``update_eval_report``, ``baseline_comparison``, ``query``.
Eagerly importing the collector here meant that re-aggregating a finished run's ``report.json``,
or running a baseline pass over cleaned CSVs, loaded TF and (beside a live eval pool) took a CUDA
context's worth of VRAM from the workers. The two stacks are described in the module docstrings;
only the training one needs Keras, so only it pays for it.

``from reporting import ReportCollector`` still works and still imports TF.
"""
from reporting.schema import (
    TrainingReport, RunConfig, EnvInfo, DataInfo, ModelInfo, EpochRecord,
)
from reporting.report_artifacts import ReportArtifacts, new_run_id

# Keras-backed names, resolved on first access (PEP 562) rather than at import.
_LAZY = {
    "ReportCollector": "reporting.collector",
    "ReportingCallback": "reporting.callback",
}


def __getattr__(name: str):
    """Resolve a Keras-backed name by importing only the module that provides it."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'reporting' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "TrainingReport", "RunConfig", "EnvInfo", "DataInfo", "ModelInfo",
    "EpochRecord", "ReportArtifacts", "new_run_id", "ReportCollector", "ReportingCallback",
]
