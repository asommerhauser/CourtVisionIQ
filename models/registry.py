"""
Central registry of trainable models, keyed by their stable artifact KEY.

Adding a new model is a one-line entry here. Anything that needs to act over "all
models" (the ModelBundle manager, the parametrized persistence tests) iterates this
mapping rather than hard-coding model classes.

Each registered class must implement the persistence contract:
  - class attribute `KEY: str`
  - `save_artifacts(self, model, root=...) -> ModelArtifacts`
  - classmethod `from_artifacts(cls, root=..., encoder=None, **kwargs) -> (instance, model)`
"""
from __future__ import annotations

from models.event_time_model import EventTimeModel

# key -> model wrapper class
MODEL_REGISTRY: dict[str, type] = {
    EventTimeModel.KEY: EventTimeModel,
}
