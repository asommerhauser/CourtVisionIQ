"""
The training/testing report data model.

These dataclasses are the single, model-agnostic schema every model in the
project reports through. The design mirror's the project's persistence ethos
(see models/artifacts.py): one shared, model-keyed contract that any model reuses
verbatim. A new model gets reporting "for free" by populating the same structures
— it does not invent its own report format.

Everything is plain dataclasses with `to_dict()` so a `TrainingReport` serializes
losslessly to JSON (report.json) and feeds the Parquet writers (parquet_store.py)
and the HTML renderer (html_report.py).

Per-epoch metrics are intentionally a free-form ``dict[str, float]`` rather than
named fields: the Event/Time model emits ``event_output_acc`` / ``time_output_mae``,
a future model emits its own keys, and both flow through unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class RunConfig:
    """Everything that defines *how* a run was configured (hyperparameters)."""

    model_key: str
    epochs_planned: int
    batch_size: int
    lr: float
    time_loss_weight: float
    patience: int
    mixed_precision: bool
    jit_compile: bool
    # Model architecture knobs (so two runs with different arch are comparable).
    arch: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EnvInfo:
    """The machine/software environment a run executed in."""

    python_version: str = ""
    tensorflow_version: str = ""
    keras_version: str = ""
    platform: str = ""
    device: str = ""                 # e.g. "GPU x1" or "CPU"
    device_names: list = field(default_factory=list)
    mixed_precision_policy: str = ""
    git_commit: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataInfo:
    """Shape/size of the data a run trained and validated on."""

    train_games: int = 0
    test_games: int = 0
    sequence_length: int = 0
    vocab_sizes: dict = field(default_factory=dict)
    norm_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelInfo:
    """Size of the compiled model."""

    total_params: int = 0
    trainable_params: int = 0
    non_trainable_params: int = 0
    num_layers: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpochRecord:
    """One epoch's outcome: timing, learning rate, and all train/val metrics."""

    epoch: int
    duration_sec: float
    learning_rate: float
    metrics: dict = field(default_factory=dict)   # all keras logs (train + val_*)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingReport:
    """The complete record of a single training/testing run."""

    run_id: str
    model_key: str
    status: str = "completed"        # completed | early_stopped | failed
    started_at: str = ""
    ended_at: str = ""
    duration_sec: float = 0.0
    epochs_run: int = 0
    best_epoch: int | None = None
    best_val_loss: float | None = None
    config: RunConfig | None = None
    environment: EnvInfo | None = None
    data: DataInfo | None = None
    model: ModelInfo | None = None
    epochs: list = field(default_factory=list)        # list[EpochRecord]
    final_test_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "model_key": self.model_key,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_sec": self.duration_sec,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict() if self.config else None,
            "environment": self.environment.to_dict() if self.environment else None,
            "data": self.data.to_dict() if self.data else None,
            "model": self.model.to_dict() if self.model else None,
            "epochs": [e.to_dict() for e in self.epochs],
            "final_test_metrics": self.final_test_metrics,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingReport":
        cfg = d.get("config")
        env = d.get("environment")
        data = d.get("data")
        model = d.get("model")
        return cls(
            run_id=d["run_id"],
            model_key=d["model_key"],
            status=d.get("status", "completed"),
            started_at=d.get("started_at", ""),
            ended_at=d.get("ended_at", ""),
            duration_sec=d.get("duration_sec", 0.0),
            epochs_run=d.get("epochs_run", 0),
            best_epoch=d.get("best_epoch"),
            best_val_loss=d.get("best_val_loss"),
            config=RunConfig(**cfg) if cfg else None,
            environment=EnvInfo(**env) if env else None,
            data=DataInfo(**data) if data else None,
            model=ModelInfo(**model) if model else None,
            epochs=[EpochRecord(**e) for e in d.get("epochs", [])],
            final_test_metrics=d.get("final_test_metrics", {}),
        )
