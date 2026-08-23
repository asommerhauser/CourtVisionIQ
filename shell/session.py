"""
The resident session: one loaded model plus the dial state that ``run`` will use.

This is what makes LOAD and RUN distinct commands rather than one blocking call. The models stay
in memory between runs, so re-tuning a dial and re-predicting costs a rollout, not a rebuild of
eleven heads.
"""
from __future__ import annotations

import copy
import gc
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import config
from config import HOLDOUT_MANIFEST_NAME


def holdout_from_results(model, results_root="./results") -> list:
    """Holdout game ids recovered from a model's previous run folders.

    Per-game folders are named ``game<ID>_<date>_<away>at<home>`` (see
    ``simulation.stage_eval._game_labels``), so the ids survive even when the processed manifest
    that produced them does not. Ordered by id for a stable, resumable sequence.
    """
    base = Path(results_root) / str(model)
    if not base.is_dir():
        return []
    ids = set()
    for run in base.iterdir():
        games = run / "games"
        if not games.is_dir():
            continue
        for d in games.iterdir():
            m = re.match(r"game(\d+)_", d.name)
            if m and d.is_dir():
                ids.add(int(m.group(1)))
    return sorted(ids)


@dataclass
class Session:
    """Everything the shell holds between commands.

    ``sim`` is a :class:`~simulation.game_simulator.GameSimulator`, typed loosely so this module
    imports without TensorFlow.
    """

    model: str | None = None
    artifacts_root: str | None = None
    sim: object | None = None
    manifest: dict = field(default_factory=dict)
    heads: tuple[str, ...] = ()
    loaded_at: str | None = None

    holdout_ids: list[int] = field(default_factory=list)
    holdout_source: str = ""

    data_dir: str = "./data"
    processed_dir: str = "./data/processed"
    # Parsed cleaned frame, cached across runs: load_all_cleaned() re-reads ~20 season CSVs and is
    # the largest per-run cost after model load. Invalidated when data_dir changes.
    cleaned_df: object | None = None
    cleaned_df_key: str | None = None

    baseline: dict = field(default_factory=dict)
    last_run_dir: Path | None = None

    def __post_init__(self):
        # Snapshot the startup dial values so `reset` restores exactly, including the dict dials.
        if not self.baseline:
            self.baseline = config.get_dials()

    # ------------------------------------------------------------------ dials
    @property
    def changed_dials(self) -> dict:
        """Dials differing from their startup values, as ``{name: (before, after)}``."""
        now = config.get_dials()
        return {k: (v, now[k]) for k, v in self.baseline.items() if now[k] != v}

    def reset_dials(self) -> int:
        n = len(self.changed_dials)
        config.apply_dials(copy.deepcopy(self.baseline))
        return n

    # ------------------------------------------------------------------ model
    @property
    def loaded(self) -> bool:
        return self.sim is not None

    def unload(self) -> str | None:
        """Release the resident model. Returns the name that was unloaded, or None.

        Order matters. ``clear_session()`` resets Keras' global graph/name state but frees nothing
        still referenced from Python, so dropping the references is what actually reclaims the
        ~2.7 GB. The compiled-inference cache is popped first: ``_compiled_forward`` builds
        ``tf.function(lambda x, _m=model: ...)``, capturing each head in a default argument, so a
        surviving cache pins every one of them.

        Note this returns host RAM, not VRAM -- TF's BFC allocator keeps its pool. With
        ``TF_FORCE_GPU_ALLOW_GROWTH`` the next load reuses that pool rather than double-allocating,
        so repeated swaps plateau instead of climbing.
        """
        if self.sim is None:
            return None
        was = self.model
        sim, self.sim = self.sim, None
        sim.__dict__.pop("_tf_infer_cache", None)
        try:
            sim.heads.clear()
        except AttributeError:
            pass
        sim.model = None
        sim.instance = None
        del sim
        self.model = self.artifacts_root = self.loaded_at = None
        self.manifest, self.heads = {}, ()
        self.holdout_ids, self.holdout_source = [], ""
        gc.collect()
        try:
            import keras
            keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()
        return was

    # ---------------------------------------------------------------- holdout
    def resolve_holdout(self) -> tuple[list[int], str]:
        """The holdout game ids to evaluate, and a label for where they came from.

        Deliberately does **not** require ``training/full_run_state.json``: that file is written by
        a full train and is absent on a machine that only ever loads someone else's weights. The
        manifest each preprocess writes is what makes ``run`` work against existing weights with no
        train state at all.
        """
        man = self.manifest.get("holdout_game_ids")
        if man:
            return [int(g) for g in man], f"{self.model}/manifest.json"

        path = Path(self.processed_dir) / HOLDOUT_MANIFEST_NAME
        if path.is_file():
            ids = [int(g) for g in json.loads(path.read_text(encoding="utf-8"))]
            if ids:
                return ids, str(path)

        state = Path("./training/full_run_state.json")
        if state.is_file():
            ids = json.loads(state.read_text(encoding="utf-8")).get("holdout_game_ids") or []
            if ids:
                return [int(g) for g in ids], str(state)

        # Last resort: recover the ids from a previous run's per-game folders. The processed
        # manifest is rewritten by preprocess and can end up empty (a re-preprocess with a
        # different cut truncates it), which would otherwise strand a perfectly loadable model
        # with nothing to evaluate. The folders are named game<ID>_<date>_<away>at<home>.
        ids = holdout_from_results(self.model)
        if ids:
            return ids, f"results/{self.model}/ (recovered from a previous run)"

        raise FileNotFoundError(
            f"No holdout game list found. Looked at: the model's manifest.json, {path} "
            f"(present but empty), ./training/full_run_state.json, and previous runs under "
            f"results/{self.model}/. Re-run preprocess, or evaluate a model that has one."
        )
