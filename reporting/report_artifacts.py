"""
Shared on-disk layout for generated training/testing reports.

Deliberately parallels models/artifacts.py: every run persists into its own
subdirectory under a common reports root, keyed by model and run, so a single
manager (see query.py) can discover and load any run the same way:

    <root>/<model_key>/<run_id>/report.html      self-contained HTML report
    <root>/<model_key>/<run_id>/report.json      full structured TrainingReport
    <root>/<model_key>/<run_id>/run.parquet      1-row run summary (queryable)
    <root>/<model_key>/<run_id>/epochs.parquet   long-format per-epoch metrics

The per-run-folder layout (rather than one appended Parquet) avoids rewrite/race
issues and lets queries glob the whole tree. New models reuse this verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import uuid

DEFAULT_REPORTS_ROOT = "./reports"


def new_run_id(name: str | None = None) -> str:
    """A sortable, unique run id: ``YYYYMMDD-HHMMSS-<short>``.

    An optional human name is slugged and appended for at-a-glance recognition.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short = uuid.uuid4().hex[:6]
    if name:
        slug = "".join(c if c.isalnum() else "-" for c in name).strip("-").lower()
        if slug:
            return f"{stamp}-{short}-{slug}"
    return f"{stamp}-{short}"


@dataclass(frozen=True)
class ReportArtifacts:
    """Resolved report paths for a single run under a reports root."""

    key: str
    run_id: str
    root: Path

    @classmethod
    def for_run(cls, key: str, run_id: str,
                root: str | Path = DEFAULT_REPORTS_ROOT) -> "ReportArtifacts":
        return cls(key=key, run_id=run_id, root=Path(root))

    @property
    def model_dir(self) -> Path:
        return self.root / self.key

    @property
    def run_dir(self) -> Path:
        return self.model_dir / self.run_id

    @property
    def html_path(self) -> Path:
        return self.run_dir / "report.html"

    @property
    def json_path(self) -> Path:
        return self.run_dir / "report.json"

    @property
    def run_parquet_path(self) -> Path:
        return self.run_dir / "run.parquet"

    @property
    def epochs_parquet_path(self) -> Path:
        return self.run_dir / "epochs.parquet"

    def ensure_dir(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def exists(self) -> bool:
        return self.json_path.exists()
