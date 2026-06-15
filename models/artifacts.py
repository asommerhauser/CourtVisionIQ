"""
Shared on-disk layout for trained model artifacts.

Every model persists into its own subdirectory under a common artifacts root, with
a consistent, model-keyed naming scheme so a single manager (see model_bundle.py)
can discover and load any model the same way:

    <root>/<key>/<key>.keras          full Keras model (graph + weights)
    <root>/<key>/<key>.weights.h5     weights only (Keras 3 requires the .weights.h5 suffix)
    <root>/<key>/norm_stats.json      per-model normalization stats / aux state

The weights-only file is the robust reload path: rebuild the architecture in Python
(deterministic from the frozen vocabs) and restore weights, sidestepping custom-object
deserialization. The .keras file is the convenient single-file path.

New models reuse this verbatim by declaring a `KEY` and going through ModelArtifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_ARTIFACTS_ROOT = "./artifacts"


@dataclass(frozen=True)
class ModelArtifacts:
    """Resolved artifact paths for a single model under an artifacts root."""

    key: str
    root: Path

    @classmethod
    def for_key(cls, key: str, root: str | Path = DEFAULT_ARTIFACTS_ROOT) -> "ModelArtifacts":
        return cls(key=key, root=Path(root))

    @property
    def model_dir(self) -> Path:
        return self.root / self.key

    @property
    def keras_path(self) -> Path:
        """Full single-file Keras model."""
        return self.model_dir / f"{self.key}.keras"

    @property
    def weights_path(self) -> Path:
        """Weights-only file (Keras 3 mandates the `.weights.h5` suffix)."""
        return self.model_dir / f"{self.key}.weights.h5"

    @property
    def norm_stats_path(self) -> Path:
        """Per-model normalization stats / auxiliary JSON state."""
        return self.model_dir / "norm_stats.json"

    def ensure_dir(self) -> Path:
        """Create the model directory if needed and return it."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        return self.model_dir

    def exists(self) -> bool:
        """True when the weights file is present (the minimum needed to reload)."""
        return self.weights_path.exists()
