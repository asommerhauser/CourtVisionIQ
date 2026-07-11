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

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ARTIFACTS_ROOT = "./artifacts"

# Parent directory that holds one subdirectory per model VERSION: ``<VERSIONS_ROOT>/v<MAJOR.MINOR>/``
# (e.g. ``./artifacts/v1.0/``). A full train writes a new version dir; a single-model retrain
# overwrites just one head inside an existing version. Keeping each version in its own dir preserves
# history for comparison across trains.
VERSIONS_ROOT = "./artifacts"


def version_root(version: str, versions_root: str = VERSIONS_ROOT) -> str:
    """Resolve a version label (e.g. ``"1.0"`` or ``"v1.0"``) to its artifacts root string.

    Returned as a plain ``./``-prefixed string (not a ``Path``) so it compares equal to the
    hard-coded defaults recorded in run state / config.
    """
    v = str(version).strip()
    v = v if v.startswith("v") else f"v{v}"
    return f"{versions_root.rstrip('/')}/{v}"


def latest_version(versions_root: str = VERSIONS_ROOT) -> str | None:
    """Newest ``v<MAJOR.MINOR>`` under ``versions_root`` by numeric (major, minor) order.

    Returns the bare label (``"1.0"``, no ``v`` prefix) or ``None`` when no version dir exists.
    Used as the default target for a single-model retrain (``train.py --model <name>``).
    """
    root = Path(versions_root)
    if not root.is_dir():
        return None
    best: tuple[int, int] | None = None
    best_label: str | None = None
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = re.fullmatch(r"v(\d+)\.(\d+)", child.name)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)))
        if best is None or key > best:
            best, best_label = key, f"{m.group(1)}.{m.group(2)}"
    return best_label


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


def warm_start_weights(model, key: str, init_weights_root) -> bool:
    """Load a model's weights from a prior artifacts root before fitting (curriculum warm-start).

    Each curriculum stage continues training the previous stage's weights rather than starting
    fresh; ``init_weights_root`` points at that prior stage's artifacts. Returns True if weights
    were loaded. Shapes match across stages because the vocab is built + frozen once up front, so
    every stage rebuilds the identical architecture. A no-op (returns False) when
    ``init_weights_root`` is falsy or no prior weights exist (the first stage trains fresh).
    """
    if not init_weights_root:
        return False
    arts = ModelArtifacts.for_key(key, init_weights_root)
    if arts.weights_path.exists():
        model.load_weights(arts.weights_path)
        print(f"[warm-start] '{key}': loaded weights from {arts.weights_path.resolve()}")
        return True
    print(f"[warm-start] '{key}': no prior weights at {arts.weights_path.resolve()}; training fresh")
    return False
