"""The shared transformer backbone's layer-name contract.

Weight reload matches **by layer name**: ``from_artifacts`` rebuilds the graph from config and
restores weights into it, and the single-file ``<key>.keras`` reload does the same. So the names
in ``models/backbone.py`` are a persistence contract, not an implementation detail -- renaming
one is silent at build time and only surfaces later as a failed or partially-restored reload.

These tests pin the names, their order, and the weight shapes they carry, for the builder itself
and for every registered head that uses it. ``scripts/dump_layer_names.py`` is the one-off
before/after form of the same check; this is the permanent one.

Build-only: no ``.fit``, no CUDA.
"""
from __future__ import annotations

import keras
import numpy as np
import pytest
from keras import Input

from encoder.encoder import Encoder
from models.backbone import build_backbone

from test_model_persistence import MODEL_TEST_ADAPTERS

SEQ = 8
D = 16
FF = 24
HEADS = 2


def backbone_layer_names(num_layers: int) -> list[str]:
    """The canonical backbone layer names, in graph order."""
    names = ["fusion_concat", "fusion_projection", "fusion_ln",
             "positional_embedding", "emb_dropout", "attn_pad_mask"]
    for i in range(num_layers):
        names += [f"block{i}_ln1", f"block{i}_mha", f"block{i}_res1",
                  f"block{i}_ln2", f"block{i}_ff1", f"block{i}_ffdrop",
                  f"block{i}_ff2", f"block{i}_res2"]
    return names + ["final_ln"]


def _tiny_backbone(num_layers=2):
    """A standalone two-part backbone, the smallest thing that exercises the builder."""
    a = Input(shape=(SEQ, 5), dtype="float32", name="a")
    b = Input(shape=(SEQ, 3), dtype="float32", name="b")
    pad = Input(shape=(SEQ,), dtype="float32", name="pad_mask")
    out = build_backbone([a, b], pad, seq_len=SEQ, d_model=D,
                         num_layers=num_layers, num_heads=HEADS, ff_dim=FF, dropout=0.0)
    return keras.Model(inputs=[a, b, pad], outputs=out, name="TinyBackbone")


# --------------------------------------------------------------------------- #
# --- The builder itself                                                    -- #
# --------------------------------------------------------------------------- #

def test_backbone_emits_the_canonical_layer_names_in_order():
    model = _tiny_backbone(num_layers=2)
    order = {l.name: i for i, l in enumerate(model.layers)}
    expected = backbone_layer_names(2)
    missing = [n for n in expected if n not in order]
    assert not missing, f"backbone lost layer(s): {missing}"
    positions = [order[n] for n in expected]
    assert positions == sorted(positions), "backbone layers are out of graph order"


def test_backbone_weight_shapes_follow_d_model_and_ff_dim():
    model = _tiny_backbone(num_layers=1)
    by_name = {l.name: l for l in model.layers}
    # fusion_projection maps the concatenated parts (5 + 3) to d_model.
    assert by_name["fusion_projection"].weights[0].shape == (8, D)
    assert by_name["positional_embedding"].weights[0].shape == (SEQ, D)
    assert by_name["block0_ff1"].weights[0].shape == (D, FF)
    assert by_name["block0_ff2"].weights[0].shape == (FF, D)
    for ln in ("fusion_ln", "block0_ln1", "block0_ln2", "final_ln"):
        assert by_name[ln].weights[0].shape == (D,)


def test_backbone_block_count_follows_num_layers():
    for n in (1, 3):
        names = {l.name for l in _tiny_backbone(num_layers=n).layers}
        assert f"block{n - 1}_mha" in names
        assert f"block{n}_mha" not in names


def test_backbone_output_is_the_encoded_sequence():
    model = _tiny_backbone(num_layers=1)
    assert tuple(model.output.shape) == (None, SEQ, D)
    out = model({"a": np.zeros((2, SEQ, 5), np.float32),
                 "b": np.zeros((2, SEQ, 3), np.float32),
                 "pad_mask": np.ones((2, SEQ), np.float32)}, training=False)
    assert np.isfinite(np.asarray(out)).all()


def test_padded_keys_do_not_reach_later_rows():
    """The key-padding mask is wired: a masked-out key cannot influence a later real row.

    The probe holes out row 3 rather than the trailing rows, because attention is causal --
    a trailing row is invisible to every earlier row whether it is masked or not, so
    scribbling on it would prove nothing. Row 5 attends over 0..5, row 3 included, so if the
    mask is dropped this changes row 5's output.
    """
    model = _tiny_backbone(num_layers=1)
    pad = np.ones((1, SEQ), np.float32)
    pad[0, 3] = 0.0
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1, SEQ, 5)).astype(np.float32)
    b = np.zeros((1, SEQ, 3), np.float32)
    base = np.asarray(model({"a": a, "b": b, "pad_mask": pad}, training=False))

    a2 = a.copy()
    a2[0, 3] = 999.0
    other = np.asarray(model({"a": a2, "b": b, "pad_mask": pad}, training=False))
    assert np.allclose(base[0, 4:], other[0, 4:], atol=1e-5)


# --------------------------------------------------------------------------- #
# --- Every registered head                                                 -- #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("adapter", MODEL_TEST_ADAPTERS, ids=lambda a: a.key)
def test_head_carries_the_backbone_layer_names(adapter, tmp_path):
    """Each head's graph contains the shared backbone, named exactly as the contract says."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    adapter.make_csv(data_dir / "season_clean.csv")

    # Isolated vocab dir: a default Encoder() rewrites the committed encoder/vocabs/.
    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    inst = adapter.build(enc, data_dir, tmp_path / "processed")
    inst.preprocess(rebuild_vocabs=True, test_frac=0.34)
    inst.model_dim = D
    model = inst.model(num_layers=2, num_heads=HEADS, ff_dim=FF)

    order = {l.name: i for i, l in enumerate(model.layers)}
    expected = backbone_layer_names(2)
    missing = [n for n in expected if n not in order]
    assert not missing, f"{adapter.key} lost backbone layer(s): {missing}"
    positions = [order[n] for n in expected]
    assert positions == sorted(positions), f"{adapter.key} backbone is out of graph order"
    assert model.get_layer("positional_embedding").weights[0].shape == (inst.sequence_length, D)
