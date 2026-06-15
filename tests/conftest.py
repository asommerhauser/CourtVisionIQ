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
