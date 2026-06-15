"""
Standardized training/testing reports for CourtVisionIQ models.

Every model reports through one shared contract (mirroring models/artifacts.py):
attach a ReportCollector around `model.fit` and call `finalize()`. Each run emits
a self-contained HTML report plus a queryable Parquet data model.

  from reporting import ReportCollector, RunConfig
  from reporting.query import load_runs, load_epochs
"""
from reporting.schema import (
    TrainingReport, RunConfig, EnvInfo, DataInfo, ModelInfo, EpochRecord,
)
from reporting.report_artifacts import ReportArtifacts, new_run_id
from reporting.collector import ReportCollector

__all__ = [
    "TrainingReport", "RunConfig", "EnvInfo", "DataInfo", "ModelInfo",
    "EpochRecord", "ReportArtifacts", "new_run_id", "ReportCollector",
]
