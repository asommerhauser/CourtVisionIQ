"""RowFF: row-wise semantics, and the graph shape that keeps the roster encoder in VRAM.

The second test is the one that matters. Train 3 OOM'd 298 steps into epoch 1 on a 2.34 GiB
tensor named

    gradient_tape/.../roster_vec/roster_encoder/pma/mab/rff/fc2/MatMul/MatMul_1

which is TF's gradient for a *broadcasting* BatchMatMul: one kernel gradient per batch
element, (B, d_model, d_ff), reduced afterwards to the kernel's real (d_model, d_ff). Keras 3's
Dense lowers to that op whenever its input is rank > 2, and inside the roster encoder the batch
axis is B*SEQ = 38400 at batch 64. RowFF therefore flattens to rank 2 first, and this asserts
the flattening by looking at the emitted graph rather than at the output -- an output test
cannot tell the two graphs apart, which is exactly why this shipped.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from layers.row_ff import RowFF


def _layer(d_model=8, d_ff=16):
    ff = RowFF(d_model=d_model, d_ff=d_ff, dropout=0.0)
    ff.build((None, None, d_model))
    return ff


def test_rowff_is_applied_per_row():
    """(B, N, D) is the same as N independent (B, D) applications -- the 'row-wise' claim."""
    ff = _layer()
    x = tf.constant(np.random.RandomState(0).randn(3, 5, 8), dtype=tf.float32)

    batched = ff(x, training=False).numpy()
    per_row = np.stack(
        [ff(x[:, i, :], training=False).numpy() for i in range(5)], axis=1
    )

    np.testing.assert_allclose(batched, per_row, rtol=1e-5, atol=1e-6)


def test_rowff_preserves_leading_axes():
    ff = _layer()
    for shape in [(4, 8), (4, 5, 8), (2, 3, 5, 8)]:
        out = ff(tf.zeros(shape), training=False)
        assert tuple(out.shape) == (*shape[:-1], 8), shape


def test_rowff_emits_no_batched_matmul():
    """Regression: a rank-3 Dense would put BatchMatMul in the graph, and its gradient is
    the multi-gigabyte per-example kernel gradient that killed train 3."""
    ff = _layer()

    @tf.function
    def fwd_bwd(x):
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(ff(x, training=True))
        return tape.gradient(loss, ff.trainable_variables)

    graph = fwd_bwd.get_concrete_function(
        tf.TensorSpec([None, 5, 8], tf.float32)
    ).graph
    batched = sorted({op.type for op in graph.get_operations()
                      if op.type.startswith("BatchMatMul")})
    assert not batched, f"RowFF emitted {batched}; Dense is seeing a rank > 2 input"


def test_rowff_kernel_gradients_are_kernel_shaped():
    """The gradients that reach the optimizer are (in, out) -- not (B, in, out)."""
    ff = _layer()
    x = tf.constant(np.random.RandomState(1).randn(4, 5, 8), dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(ff(x, training=True))
    grads = tape.gradient(loss, ff.trainable_variables)

    for g, v in zip(grads, ff.trainable_variables):
        assert tuple(g.shape) == tuple(v.shape)
