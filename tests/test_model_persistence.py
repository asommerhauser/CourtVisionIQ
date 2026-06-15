"""
Model persistence round-trip tests: train -> save -> reload -> infer.

These are written for the *multi-model* future. Each model contributes one
`ModelTestAdapter` describing how to build a tiny instance, synthesize cleaned data,
construct a forward-pass input batch, and assert its output shapes. The actual tests
are parametrized over `MODEL_TEST_ADAPTERS`, so a new model gets full save/load/infer
coverage by appending a single adapter here — no new test file.

Covered for every registered model:
  - from_artifacts(): rebuild graph + restore weights (the robust reload path)
  - keras.models.load_model(<key>.keras): full single-file reload
  - ModelBundle.load(): the combined "all models together" manager
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import keras
import numpy as np
import pandas as pd
import pytest

from encoder.encoder import Encoder
from models.artifacts import ModelArtifacts
from models.event_time_model import EventTimeModel
from models.model_bundle import ModelBundle

# Tiny config so the round trip runs fast on CPU.
TEST_SEQ_LEN = 16


@dataclass
class ModelTestAdapter:
    key: str
    model_cls: type
    seq_len: int
    make_csv: Callable[[Path], None]
    # (encoder, data_dir, processed_dir) -> wrapper instance
    build: Callable[[Encoder, str, str], object]
    # (instance, model) -> dict of output_name -> array
    forward: Callable[[object, object], dict]
    # (instance, outputs) -> None  (raise on mismatch)
    assert_outputs: Callable[[object, dict], None]


# --------------------------------------------------------------------------- #
# Event/Time model adapter
# --------------------------------------------------------------------------- #

def _roster(players):
    return str(list(players))


def _event_time_csv(path: Path) -> None:
    """Three short games so an 80/20-ish split yields both train and test games."""
    rows = []
    for gid, n in [(1, 5), (2, 4), (3, 4)]:
        for i in range(n):
            rows.append({
                "game_id": gid,
                "roster_home": _roster(["A", "B", "C", "D", "E"]),
                "roster_away": _roster(["F", "G", "H", "I", "J"]),
                "time": i * 10,
                "event": f"ev{gid}_{i}",
                "player": "A",
                "type": "t",
                "result": "r",
                "season": "2003",
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _event_time_build(encoder, data_dir, processed_dir):
    return EventTimeModel(
        encoder,
        sequence_length=TEST_SEQ_LEN,
        path=str(data_dir),
        processed_dir=str(processed_dir),
    )


def _event_time_forward(inst, model) -> dict:
    split = inst._load_processed("test.npz")
    inputs = {k: split[k] for k in EventTimeModel.INPUT_KEYS}
    out = model(inputs, training=False)
    return {k: np.asarray(v) for k, v in out.items()}


def _event_time_assert(inst, outputs) -> None:
    assert set(outputs.keys()) == {"event_output", "time_output"}
    event_vocab = inst.encoder.event_vocab.next_token
    ev = outputs["event_output"]
    tm = outputs["time_output"]
    assert ev.ndim == 3 and ev.shape[1] == TEST_SEQ_LEN and ev.shape[2] == event_vocab
    assert tm.ndim == 3 and tm.shape[1] == TEST_SEQ_LEN and tm.shape[2] == 1
    assert ev.shape[0] == tm.shape[0] >= 1
    assert np.isfinite(ev).all() and np.isfinite(tm).all()


EVENT_TIME_ADAPTER = ModelTestAdapter(
    key=EventTimeModel.KEY,
    model_cls=EventTimeModel,
    seq_len=TEST_SEQ_LEN,
    make_csv=_event_time_csv,
    build=_event_time_build,
    forward=_event_time_forward,
    assert_outputs=_event_time_assert,
)

# New models: append their adapter here to inherit the round-trip coverage below.
MODEL_TEST_ADAPTERS = [EVENT_TIME_ADAPTER]


# --------------------------------------------------------------------------- #
# Shared fixtures / helpers
# --------------------------------------------------------------------------- #

def _train_tiny(adapter: ModelTestAdapter, tmp_path: Path):
    """Train a 1-epoch model for `adapter` and return (artifacts_root, vocab_dir, data_dir)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    adapter.make_csv(data_dir / "season_clean.csv")

    vocab_dir = tmp_path / "vocabs"
    artifacts_root = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"

    enc = Encoder(vocab_dir=vocab_dir)
    inst = adapter.build(enc, data_dir, processed_dir)
    inst.preprocess(rebuild_vocabs=True, test_frac=0.34)  # ~1 of 3 games to test
    inst.train(
        epochs=1,
        batch_size=2,
        artifacts_root=str(artifacts_root),
        mixed_precision=False,  # CPU-friendly, deterministic dtype
    )
    return artifacts_root, vocab_dir, data_dir, processed_dir


@pytest.mark.parametrize("adapter", MODEL_TEST_ADAPTERS, ids=lambda a: a.key)
def test_from_artifacts_roundtrip(adapter, tmp_path):
    """Rebuild graph + restore weights, then run inference with valid output shapes."""
    artifacts_root, vocab_dir, data_dir, processed_dir = _train_tiny(adapter, tmp_path)

    enc = Encoder(vocab_dir=vocab_dir)
    inst, model = adapter.model_cls.from_artifacts(
        root=str(artifacts_root),
        encoder=enc,
        sequence_length=adapter.seq_len,
        path=str(data_dir),
        processed_dir=str(processed_dir),
    )
    outputs = adapter.forward(inst, model)
    adapter.assert_outputs(inst, outputs)


@pytest.mark.parametrize("adapter", MODEL_TEST_ADAPTERS, ids=lambda a: a.key)
def test_full_keras_model_loads(adapter, tmp_path):
    """The single-file <key>.keras reloads (validates custom-layer serialization)."""
    artifacts_root, _, _, _ = _train_tiny(adapter, tmp_path)
    arts = ModelArtifacts.for_key(adapter.key, artifacts_root)
    assert arts.keras_path.exists()
    assert arts.weights_path.exists()
    loaded = keras.models.load_model(arts.keras_path)
    assert loaded is not None


@pytest.mark.parametrize("adapter", MODEL_TEST_ADAPTERS, ids=lambda a: a.key)
def test_model_bundle_loads(adapter, tmp_path):
    """ModelBundle reloads the model from artifacts and exposes it by key."""
    artifacts_root, vocab_dir, data_dir, processed_dir = _train_tiny(adapter, tmp_path)

    bundle = ModelBundle.load(
        root=str(artifacts_root),
        encoder=Encoder(vocab_dir=vocab_dir),
        sequence_length=adapter.seq_len,
        path=str(data_dir),
        processed_dir=str(processed_dir),
    )
    assert adapter.key in bundle
    outputs = adapter.forward(bundle.instances[adapter.key], bundle.models[adapter.key])
    adapter.assert_outputs(bundle.instances[adapter.key], outputs)
