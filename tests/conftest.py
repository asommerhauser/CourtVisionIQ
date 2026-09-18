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
from pathlib import Path

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


# Constants that are NOT rollout dials but that tests do rebind: corpus bounds and architecture
# switches. They cannot join _TUNING_KEYS -- config.py:629 spells out why a training knob does not
# belong there (nothing at sim time reads it, and it is in neither _TUNING_KEYS nor ARCH_KEYS) -- so
# they get their own restore list. Names are listed whether or not this build defines them yet, so a
# workstream that adds one does not also have to remember to add it here.
_BUILD_CONSTANTS = (
    "MIN_TRAIN_SEASON",
    "MIN_PLAYER_SUBSET_GAMES",
    "SUBSET_MODEL_KEYS",
    "FILM_ENABLED",
    "CROSS_ROSTER_ENABLED",
    "ROLLOUT_SELECTION",
)


@pytest.fixture(autouse=True)
def _restore_build_constants():
    """Snapshot and restore the corpus/architecture constants around each test.

    Same argument as ``_restore_dials``: they are module globals, a test that sets one would leak
    into every test after it, and the leak is silent -- a corpus floor left set turns 18 modules'
    2003 fixtures into empty corpora, and the failure reads as "this fixture is broken" rather than
    "the previous test did not clean up".
    """
    saved = {k: copy.deepcopy(getattr(_config, k))
             for k in _BUILD_CONSTANTS if hasattr(_config, k)}
    # The training corpus floor is OFF for tests. 18 modules build 2003 fixtures, and the production
    # floor of 2008 would empty every one of them -- so the default here is "no floor" and the tests
    # that exercise the floor set it themselves. That keeps the floor's own behaviour explicitly
    # tested rather than incidentally relied on.
    if hasattr(_config, "MIN_TRAIN_SEASON"):
        _config.MIN_TRAIN_SEASON = None
    yield
    for k, v in saved.items():
        setattr(_config, k, v)


@pytest.fixture(scope="session", autouse=True)
def _preserve_committed_vocabs():
    """Restore ``encoder/vocabs/`` at the end of the session.

    A net, not a fix: **as of 3.2 it catches nothing.** Every test that builds an ``Encoder`` passes
    an isolated ``vocab_dir=tmp_path``, and the shared ``norm_stats.json`` path is keyed off
    ``encoder.vocab_dir`` rather than ``config.NORM_STATS_PATH``
    (``models/event_time_model.py:259-266``), so nothing in the suite currently writes the committed
    directory. That is the *result* of fixing tests one at a time -- ``tests/test_backbone.py:143``
    still carries the comment someone wrote after learning it the hard way ("a default Encoder()
    rewrites the committed encoder/vocabs/") -- and it is a convention, enforced only by every
    future test author remembering it.

    This makes it an invariant instead. It earns its keep in two places: a new test that forgets the
    tmp dir fails loudly here rather than silently editing the tree, and W4 deletes and rebuilds the
    vocabs on purpose, so development against that code runs writers that really do target this
    directory.

    Bytes, not text: the files are rewritten exactly, so the restore cannot flip a line ending
    (standing rule 6). Files the session creates are removed; files it deletes come back.

    It is a session teardown, so it does **not** survive a kill or a Ctrl-C -- which is why the
    handover keeps ``git status --porcelain encoder/vocabs/`` as a belt-and-braces check before a
    train rather than trusting this alone.
    """
    vocab_dir = Path(_config.VOCAB_DIR)
    before = ({p.name: p.read_bytes() for p in sorted(vocab_dir.glob("*.json"))}
              if vocab_dir.is_dir() else {})
    yield
    if not vocab_dir.is_dir():
        return
    for name, blob in before.items():
        path = vocab_dir / name
        if not path.exists() or path.read_bytes() != blob:
            path.write_bytes(blob)
    for path in sorted(vocab_dir.glob("*.json")):
        if path.name not in before:
            path.unlink()
