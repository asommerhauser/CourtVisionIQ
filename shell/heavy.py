"""
The single gate for importing TensorFlow.

The shell must reach its prompt in well under a second, so ``cviq.py`` starts the REPL without
importing tensorflow, keras, pandas, or anything under ``models.``/``simulation.``. Commands that
genuinely need a model (``load``, ``run``, ``diagnose``) call :func:`ensure_tf` first; the read-only
ones (``status``, ``dials``, ``models``, ``runs``, ``help``) never do.

TF is imported **before** pandas on purpose. On Windows, importing pandas first can break TF's
native DLL initialization:

    ImportError: DLL load failed while importing _pywrap_tensorflow_internal

The same workaround appears in ``tests/conftest.py`` and ``simulation/stage_eval.py``. For that
reason this is deliberately *not* done on a background thread -- the ordering is the whole point.
"""
from __future__ import annotations

_READY = False


def ensure_tf(verbose: bool = True) -> None:
    """Import TensorFlow once, printing a one-line notice the first time. Idempotent."""
    global _READY
    if _READY:
        return
    if verbose:
        print("  importing tensorflow (first heavy command only) ...", flush=True)
    import tensorflow  # noqa: F401
    _READY = True


def is_ready() -> bool:
    """True once TF has been imported, without importing it."""
    return _READY
