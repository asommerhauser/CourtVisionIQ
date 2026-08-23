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

One level up, `<MODELS_ROOT>/<name>/` holds one such artifacts root per trained model; see
`model_root` / `list_models` / `active_model`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ARTIFACTS_ROOT = "./artifacts"

# Parent directory holding one subdirectory per MODEL: ``<MODELS_ROOT>/<name>/`` (e.g.
# ``./artifacts/v1.0/``). A full train writes a new model dir; a single-head retrain overwrites just
# one head inside an existing one. Each model keeps its own dir so trains stay comparable.
MODELS_ROOT = "./artifacts"
VERSIONS_ROOT = MODELS_ROOT  # deprecated alias, kept for existing callers

# Records which model is currently loaded, inside the gitignored artifacts tree (per-machine).
ACTIVE_MARKER = "ACTIVE"

# Model names are free-form slugs. Anchored so a name can never contain a path separator or be
# ".."/"." -- model_root() interpolates it straight into a path.
_VALID_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def version_root(version: str, versions_root: str = VERSIONS_ROOT) -> str:
    """Deprecated: resolve a ``v<MAJOR>.<MINOR>`` label to its artifacts root.

    Superseded by :func:`model_root`, which takes a free-form name. Kept because it also normalizes
    a bare ``"1.0"`` to ``"v1.0"``, which existing callers (training.full_run, evaluate.py) rely on.
    """
    v = str(version).strip()
    v = v if v.startswith("v") else f"v{v}"
    return model_root(v, versions_root)


def model_root(name: str, models_root: str = MODELS_ROOT) -> str:
    """Resolve a model NAME to its artifacts root string.

    Names are free-form slugs -- ``"v1.0"``, ``"endgame-feats"``, ``"relative-encoding"``. A model
    name is the train identity: retraining always produces a NEW name rather than overwriting an
    existing one, so a set of weights and the runs evaluated against it never drift apart.

    Returned as a plain ``./``-prefixed string (not a ``Path``) so it compares equal to the
    hard-coded defaults recorded in run state / config.
    """
    slug = str(name).strip()
    if not _VALID_NAME.fullmatch(slug):
        raise ValueError(
            f"invalid model name {name!r}: expected letters/digits then any of "
            f"[A-Za-z0-9._-] (e.g. 'v1.0', 'endgame-feats'). Path separators and '..' are "
            f"rejected so a name can never escape the artifacts root."
        )
    return f"{models_root.rstrip('/')}/{slug}"


def list_models(models_root: str = MODELS_ROOT) -> list[str]:
    """Every loadable model name under ``models_root``, sorted.

    A directory counts only when it holds ``event_time`` weights -- the head
    :class:`~simulation.game_simulator.GameSimulator` requires. That check also keeps stray
    per-head dirs out of the listing: the legacy ``main.py`` path writes an unversioned
    ``./artifacts/<head>/``, which would otherwise look like a model named e.g. "player".
    """
    root = Path(models_root)
    if not root.is_dir():
        return []
    names = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not _VALID_NAME.fullmatch(child.name):
            continue
        if ModelArtifacts.for_key("event_time", child).exists():
            names.append(child.name)
    return names


def active_model(models_root: str = MODELS_ROOT) -> str | None:
    """The model name recorded by :func:`set_active_model`, or None.

    Stored in ``<models_root>/ACTIVE`` -- inside the gitignored artifacts tree, so "which model is
    loaded" stays per-machine rather than travelling in the repo. Returns None when the file is
    absent or names a model that no longer exists.
    """
    marker = Path(models_root) / ACTIVE_MARKER
    if not marker.is_file():
        return None
    name = marker.read_text(encoding="utf-8").strip()
    return name if name and name in list_models(models_root) else None


def set_active_model(name: str, models_root: str = MODELS_ROOT) -> str:
    """Record ``name`` as the active model. Returns the name."""
    model_root(name, models_root)  # validate before writing
    root = Path(models_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / ACTIVE_MARKER).write_text(name.strip() + "\n", encoding="utf-8")
    return name


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
