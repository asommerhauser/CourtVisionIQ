"""
Simulation / generation layer.

Holds the trained model(s) and drives game *generation* — the self-feeding loop
that the training side (encoder + EventTimeModel + ModelBundle) was built to enable.
This package is the home for the rollout driver, the Controller and the Monte-Carlo
harness (see docs/technical_specs.md → Evaluation Strategy).

**Nothing here is imported eagerly.** ``game_simulator`` imports TensorFlow, and this package
also holds modules that are pure pandas — ``box_score``, ``stats``, ``eval_metrics``, ``probes``.
An eager ``from simulation.game_simulator import GameSimulator`` at the top of this file meant
that *any* touch of the package loaded TF: ``reporting/update_eval_report.py`` re-aggregating a
finished run, ``reporting/baseline_comparison.py`` doing a pandas pass over cleaned CSVs, and the
W1 probes reading play-by-play CSVs all paid for a CUDA context they never used — and, beside a
live eval pool, took that VRAM away from it. ``scripts/peek_sims.py`` worked around it by loading
``box_score.py`` through ``importlib`` behind the package's back.

So the TF-backed names resolve lazily, through the module ``__getattr__`` below (PEP 562).
``from simulation import GameSimulator`` still works and still imports TF; ``from
simulation.eval_metrics import _aggregate`` now does not.

One Windows caveat this makes explicit rather than hides: importing pandas before TF breaks TF's
native DLL initialization ("DLL load failed while importing _pywrap_tensorflow_internal"). That
was never fixed by this file's import order — a caller who imported pandas first was already
broken. The entry points that need TF import it first themselves (``main.py``, ``train.py``,
``evaluate.py``); anything else in this package is TF-free and does not care.
"""
from __future__ import annotations

# Names that cost a TensorFlow import, mapped to the module that provides them.
_LAZY = {
    "GameSimulator": "simulation.game_simulator",
}

# Names that are pure pandas/stdlib and safe to import on demand without TF.
_TF_FREE = {
    "BoxScore": "simulation.box_score",
    "PlayerLine": "simulation.box_score",
    "box_score_for_game": "simulation.box_score",
    "generate_box_score": "simulation.box_score",
    "GameInput": "simulation.game_input",
    "extract_game_input": "simulation.game_input",
    "game_input_for_game": "simulation.game_input",
    "holdout_game_inputs": "simulation.game_input",
    "write_holdout_inputs": "simulation.game_input",
}


def __getattr__(name: str):
    """Resolve a package-level name by importing only the module that provides it."""
    module = _LAZY.get(name) or _TF_FREE.get(name)
    if module is None:
        raise AttributeError(f"module 'simulation' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "GameSimulator",
    "BoxScore",
    "PlayerLine",
    "generate_box_score",
    "box_score_for_game",
    "GameInput",
    "extract_game_input",
    "game_input_for_game",
    "holdout_game_inputs",
    "write_holdout_inputs",
]
