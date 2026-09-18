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


# The mask layer hangs off the ``pad_mask`` input rather than off the running tensor, so Keras is
# free to place it anywhere after that input in the topological sort -- it is not a link in the
# chain and carries no ordering guarantee. Its presence is asserted separately, and
# ``test_padded_keys_do_not_reach_later_rows`` proves it is actually wired into the attention.
SIDE_BRANCH_LAYERS = ("attn_pad_mask",)

# 3.2 W6's FiLM layers. The two Dense projections per hook hang off the CONTEXT tensor, not off the
# running one, so like ``attn_pad_mask`` they carry no ordering guarantee -- only the Multiply and the
# Add sit on the main path. ``film_concat`` / ``film_ctx`` are the shared bottleneck, built once.
FILM_SIDE_LAYERS = ("film_concat", "film_ctx")


def film_side_layers(num_layers: int) -> list[str]:
    out = list(FILM_SIDE_LAYERS)
    for i in range(num_layers):
        for slot in (1, 2):
            out += [f"block{i}_film{slot}_scale", f"block{i}_film{slot}_shift"]
    return out


def backbone_chain_names(num_layers: int, *, film: bool = False) -> list[str]:
    """The canonical backbone layer names that sit on the main tensor path, in graph order."""
    names = ["fusion_concat", "fusion_projection", "fusion_ln",
             "positional_embedding", "emb_dropout"]
    for i in range(num_layers):
        names += [f"block{i}_ln1"]
        if film:
            names += [f"block{i}_film1_mul", f"block{i}_film1"]
        names += [f"block{i}_mha", f"block{i}_res1", f"block{i}_ln2"]
        if film:
            names += [f"block{i}_film2_mul", f"block{i}_film2"]
        names += [f"block{i}_ff1", f"block{i}_ffdrop", f"block{i}_ff2", f"block{i}_res2"]
    return names + ["final_ln"]


def backbone_layer_names(num_layers: int, *, film: bool = False) -> list[str]:
    """Every canonical backbone layer name (chain plus side branches)."""
    extra = film_side_layers(num_layers) if film else []
    return backbone_chain_names(num_layers, film=film) + list(SIDE_BRANCH_LAYERS) + extra


def assert_backbone_names(model, num_layers: int, label: str, *, film: bool = False) -> None:
    """Every canonical name is present, and the main path is in graph order."""
    order = {l.name: i for i, l in enumerate(model.layers)}
    missing = [n for n in backbone_layer_names(num_layers, film=film) if n not in order]
    assert not missing, f"{label} lost backbone layer(s): {missing}"
    positions = [order[n] for n in backbone_chain_names(num_layers, film=film)]
    assert positions == sorted(positions), f"{label} backbone chain is out of graph order"


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
    assert_backbone_names(_tiny_backbone(num_layers=2), 2, "backbone")


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

    # film=config.FILM_ENABLED, so the heads' FiLM layers are really checked rather than merely
    # tolerated: with film=False the non-FiLM names are still present and ordered, so the assertion
    # would pass while saying nothing about W6.
    import config
    assert_backbone_names(model, 2, adapter.key, film=bool(config.FILM_ENABLED))
    assert model.get_layer("positional_embedding").weights[0].shape == (inst.sequence_length, D)


# --------------------------------------------------------------------------- #
# --- W6: context modulation (FiLM)                                         -- #
# --------------------------------------------------------------------------- #

def _film_backbone(num_layers=2, *, film=True, ctx_width=3):
    """The tiny backbone, with one of its parts also serving as the FiLM context."""
    a = Input(shape=(SEQ, 5), dtype="float32", name="a")
    ctx = Input(shape=(SEQ, ctx_width), dtype="float32", name="ctx")
    pad = Input(shape=(SEQ,), dtype="float32", name="pad_mask")
    out = build_backbone([a, ctx], pad, seq_len=SEQ, d_model=D, num_layers=num_layers,
                         num_heads=HEADS, ff_dim=FF, dropout=0.0,
                         film_context=[ctx] if film else None)
    return keras.Model(inputs=[a, ctx, pad], outputs=out, name="FilmBackbone")


def test_film_is_the_identity_at_initialisation():
    """**The property that makes the A/B honest.**

    Scale and shift are zero-initialised in both kernel and bias, and applied as
    ``h * (1 + gamma) + beta``, so before any training the modulation does nothing at all and a FiLM
    graph is numerically the same as one built without it. Predicting the scale directly instead would
    perturb the residual stream at initialisation, and the comparison would then be between two
    different initialisations rather than between FiLM and no FiLM.
    """
    off, on = _film_backbone(film=False), _film_backbone(film=True)
    shared = {l.name: l for l in off.layers}
    for layer in on.layers:
        src = shared.get(layer.name)
        if src is not None and src.weights and len(src.weights) == len(layer.weights):
            layer.set_weights(src.get_weights())

    rng = np.random.default_rng(0)
    xa = rng.normal(size=(3, SEQ, 5)).astype("float32")
    xc = rng.normal(size=(3, SEQ, 3)).astype("float32")
    pad = np.ones((3, SEQ), dtype="float32")

    np.testing.assert_allclose(off.predict([xa, xc, pad], verbose=0),
                               on.predict([xa, xc, pad], verbose=0), atol=1e-5)


def test_the_scale_and_shift_start_at_exactly_zero():
    model = _film_backbone()
    for name in ("block0_film1_scale", "block0_film1_shift",
                 "block1_film2_scale", "block1_film2_shift"):
        kernel, bias = model.get_layer(name).get_weights()
        assert not kernel.any() and not bias.any(), f"{name} must start at zero"


def test_no_context_reproduces_the_original_graph_exactly():
    """``film_context=None`` must leave the graph as it was, the way a zero
    ``LOCAL_ATTENTION_HEADS`` does -- otherwise every pre-3.2 bundle stops reloading."""
    off = _film_backbone(film=False)
    assert not [l.name for l in off.layers if "film" in l.name]
    assert_backbone_names(off, 2, "film-off", film=False)


def test_the_feature_flag_turns_it_off_without_touching_the_call_sites(monkeypatch):
    """The heads always pass a context; ``FILM_ENABLED`` is what decides whether it is used.

    Read at call time, not imported at module scope -- 3.0 lost a day to knobs frozen at import.
    """
    import config
    monkeypatch.setattr(config, "FILM_ENABLED", False)
    model = _film_backbone(film=True)
    assert not [l.name for l in model.layers if "film" in l.name]


def test_the_modulation_reaches_the_output_once_the_scale_is_not_zero():
    """Zero-init must not mean permanently inert: a trained gamma has to change the result."""
    model = _film_backbone()
    rng = np.random.default_rng(1)
    xa = rng.normal(size=(2, SEQ, 5)).astype("float32")
    xc = rng.normal(size=(2, SEQ, 3)).astype("float32")
    pad = np.ones((2, SEQ), dtype="float32")
    before = model.predict([xa, xc, pad], verbose=0)

    scale = model.get_layer("block0_film1_scale")
    kernel, bias = scale.get_weights()
    scale.set_weights([kernel, bias + 0.5])          # a constant gamma of 0.5

    after = model.predict([xa, xc, pad], verbose=0)
    assert not np.allclose(before, after, atol=1e-6), "a non-zero gamma must change the output"


def test_the_context_only_enters_through_the_modulation():
    """The FiLM bottleneck is built once and shared across blocks, not rebuilt per block."""
    model = _film_backbone(num_layers=3)
    assert len([l for l in model.layers if l.name == "film_ctx"]) == 1
    # Two hooks per block, each with its own scale and shift.
    assert len([l for l in model.layers if l.name.endswith("_scale")]) == 3 * 2
    assert len([l for l in model.layers if l.name.endswith("_shift")]) == 3 * 2


def test_film_is_recorded_in_the_arch_keys():
    """A flag that is off produces a graph MISSING layers, which ``load_weights`` skips silently."""
    from models.manifest import ARCH_KEYS
    assert "FILM_ENABLED" in ARCH_KEYS and "FILM_DIM" in ARCH_KEYS
