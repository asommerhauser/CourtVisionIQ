"""
Keras callback that captures what `model.fit`'s History object leaves out.

History gives us per-epoch metric values but no wall-clock timing and no tidy
per-epoch learning-rate trace. This callback records both, plus a snapshot of the
full logs dict (every train + ``val_`` metric Keras reports), into a list of
``EpochRecord``s. It is the single piece attached inside `model.fit`; the
collector reads ``.records`` afterward to assemble the report.

Being a generic Callback (no model-specific keys) is what keeps it reusable: any
model's metrics flow through ``logs`` unchanged.
"""
from __future__ import annotations

import time

import keras

from reporting.schema import EpochRecord


def _current_lr(model) -> float:
    """Best-effort read of the optimizer's current learning rate as a float."""
    try:
        lr = model.optimizer.learning_rate
        # LR can be a schedule or a variable; resolve to a python float.
        if callable(getattr(lr, "__call__", None)):
            try:
                lr = lr(model.optimizer.iterations)
            except Exception:
                pass
        return float(keras.ops.convert_to_numpy(lr)) if hasattr(keras, "ops") else float(lr)
    except Exception:
        try:
            return float(model.optimizer.learning_rate)
        except Exception:
            return 0.0


class ReportingCallback(keras.callbacks.Callback):
    """Collect per-epoch timing, learning rate, and all metrics during fit()."""

    def __init__(self):
        super().__init__()
        self.records: list[EpochRecord] = []
        self._epoch_start: float = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.perf_counter() - self._epoch_start
        logs = dict(logs or {})
        # LR is captured as its own field; drop it from metrics so it isn't double
        # counted (Keras logs it as `lr` and/or `learning_rate` across versions).
        skip = {"lr", "learning_rate"}
        metrics = {k: float(v) for k, v in logs.items() if k not in skip}
        self.records.append(
            EpochRecord(
                epoch=int(epoch),
                duration_sec=round(duration, 4),
                learning_rate=_current_lr(self.model),
                metrics=metrics,
            )
        )
