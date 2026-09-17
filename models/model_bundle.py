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
from models.manifest import feature_mismatch, read_manifest
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

        Refuses up front when the bundle's manifest records a different INPUT SIGNATURE than this
        build produces. Until 3.0 the manifest's arch snapshot was advisory -- nothing read it at
        load, and ``from_artifacts`` rebuilt the graph from the live ``config.py`` and called
        ``load_weights`` unconditionally. Most signature changes do fail there, because
        ``fusion_projection``'s kernel is a function of the concatenated input width; but a change
        that swaps one key for another of the same width loads silently and means something
        different on every row. This turns that into the readable refusal the manifest's own
        docstring promises, naming the keys that moved.
        """
        problems = feature_mismatch((read_manifest(root) or {}).get("features"))
        if problems:
            raise ValueError(
                f"the bundle at {root} was trained with different model inputs than this "
                f"build produces:\n  " + "\n  ".join(problems) + "\n"
                "Check out the commit that trained it, or retrain. Loading anyway would "
                "rebuild the graph from the current config.py and reinterpret the weights.")

        models: dict = {}
        instances: dict = {}
        for key, model_cls in MODEL_REGISTRY.items():
            if not ModelArtifacts.for_key(key, root).exists():
                continue
            inst, model = model_cls.from_artifacts(root=root, encoder=encoder, **kwargs)
            instances[key] = inst
            models[key] = model
        return cls(models, instances)
