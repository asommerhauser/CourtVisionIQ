"""Dump every head's layer names and weight shapes, for before/after refactor diffs.

Weight reload is **by layer name**: ``from_artifacts`` rebuilds the graph from config and
restores weights into it, and the single-file ``<key>.keras`` reload matches the same way. So a
refactor of the shared transformer backbone is safe if and only if it leaves every layer's name,
order and weight shapes untouched -- a rename is silent at build time and only surfaces as a
reload failure (or, worse, a partially-restored model) much later.

This prints a stable, sortable text record of exactly that, so::

    git checkout <pre-refactor-commit> && python scripts/dump_layer_names.py > before.txt
    git checkout <post-refactor-commit> && python scripts/dump_layer_names.py > after.txt
    diff before.txt after.txt

is the direct check. An empty diff is the pass.

Graph construction only -- no ``.fit``, no CUDA, no cleaned data. It reuses the per-model build
adapters from ``tests/test_model_persistence.py`` (the one place that already knows how to stand
up every registered head from synthetic rows), against an isolated vocab directory so it can
never overwrite the committed ``encoder/vocabs/``.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

# A small graph is enough: layer names do not depend on width, and this keeps the dump seconds-fast.
MODEL_DIM = 32
NUM_LAYERS = 2
NUM_HEADS = 2
FF_DIM = 32


def _build(adapter, tmp: Path):
    """Stand up ``adapter``'s head and return its built Keras model."""
    from encoder.encoder import Encoder

    data_dir = tmp / adapter.key / "data"
    data_dir.mkdir(parents=True)
    adapter.make_csv(data_dir / "season_clean.csv")

    # Isolated vocab dir: a default Encoder() would rewrite the committed encoder artifacts.
    enc = Encoder(vocab_dir=tmp / adapter.key / "vocabs")
    inst = adapter.build(enc, data_dir, tmp / adapter.key / "processed")
    # preprocess prints a summary line carrying the temp directory, which differs on every
    # run and would show up as a spurious hunk in the before/after diff this script exists
    # to produce. Only layer lines belong on stdout.
    with contextlib.redirect_stdout(sys.stderr):
        inst.preprocess(rebuild_vocabs=True, test_frac=0.34)
    inst.model_dim = MODEL_DIM
    return inst.model(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_dim=FF_DIM)


def _dump(key: str, model, out) -> None:
    """One line per layer: position, name, class, then each weight's shape."""
    for i, layer in enumerate(model.layers):
        shapes = " ".join(str(tuple(w.shape)) for w in layer.weights)
        print(f"{key}\t{i:03d}\t{layer.name}\t{type(layer).__name__}\t{shapes}", file=out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None,
                    help="Comma-separated model keys to dump (default: every registered head).")
    args = ap.parse_args()

    from test_model_persistence import MODEL_TEST_ADAPTERS

    adapters = MODEL_TEST_ADAPTERS
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        adapters = [a for a in adapters if a.key in wanted]
        missing = wanted - {a.key for a in adapters}
        if missing:
            print(f"unknown model key(s): {sorted(missing)}", file=sys.stderr)
            return 2

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for adapter in adapters:
            _dump(adapter.key, _build(adapter, tmp), sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
