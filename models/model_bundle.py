"""
ModelBundle: load and hold all trained models together.

This is the forward-facing "use the models together" entry point. Today the registry
contains only the Event/Time model, but ModelBundle is written to scale: it iterates
MODEL_REGISTRY, reloads every model that has artifacts on disk, and exposes them by
key. Models without artifacts yet are skipped, so the bundle works incrementally as
new models come online.

    bundle = ModelBundle.load("./artifacts")
    model = bundle.models["event_time"]      # or bundle["event_time"]
    inst  = bundle.instances["event_time"]   # the wrapper (norm_stats, encoder, ...)
"""
from __future__ import annotations

from models.artifacts import ModelArtifacts, DEFAULT_ARTIFACTS_ROOT
from models.registry import MODEL_REGISTRY


class ModelBundle:
    def __init__(self, models: dict, instances: dict):
        self.models = models        # key -> compiled/loaded keras model
        self.instances = instances  # key -> model wrapper (EventTimeModel, ...)

    def __getitem__(self, key: str):
        return self.models[key]

    def __contains__(self, key: str) -> bool:
        return key in self.models

    def keys(self):
        return self.models.keys()

    @classmethod
    def load(cls, root: str = DEFAULT_ARTIFACTS_ROOT, encoder=None, **kwargs) -> "ModelBundle":
        """
        Reload every registered model that has artifacts under `root`. Shared
        `encoder`/constructor kwargs flow to each model's from_artifacts. Models
        with no saved weights are skipped.
        """
        models: dict = {}
        instances: dict = {}
        for key, model_cls in MODEL_REGISTRY.items():
            if not ModelArtifacts.for_key(key, root).exists():
                continue
            inst, model = model_cls.from_artifacts(root=root, encoder=encoder, **kwargs)
            instances[key] = inst
            models[key] = model
        return cls(models, instances)
