"""
Shared pytest setup.

Import TensorFlow before any other heavy library. On Windows, importing pandas
(pulled in by several non-TF test modules collected earlier alphabetically) before
TensorFlow can break TF's native DLL initialization:

    ImportError: DLL load failed while importing _pywrap_tensorflow_internal

Forcing TF to load first here makes the full suite order-independent. Wrapped in
try/except so environments without TF (or non-Windows) are unaffected.
"""
try:  # noqa: SIM105
    import tensorflow  # noqa: F401
except Exception:
    pass


import copy

import pytest

import config as _config


@pytest.fixture(autouse=True)
def _restore_dials():
    """Snapshot and restore every rollout dial around each test.

    The dials are module globals that the shell (and ``config.dials()``) rebind at runtime, so a
    test that sets one would otherwise leak into every test that runs after it. Deep-copied
    because three of them are dicts.
    """
    saved = {k: copy.deepcopy(getattr(_config, k)) for k in _config._TUNING_KEYS}
    yield
    for k, v in saved.items():
        setattr(_config, k, v)
