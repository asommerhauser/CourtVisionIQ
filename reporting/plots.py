"""
Render training curves as base64-encoded PNG strings.

Returning base64 (rather than writing loose .png files) lets the HTML report be
fully self-contained — graphs embed directly via ``<img src="data:image/png;..."``.

The curve builders are metric-name driven, not model-specific: give any train
metric and its ``val_`` counterpart and you get a curve. ``auto_curves`` discovers
every train/val metric pair present in a report, so a new model's metrics are
plotted automatically with no code changes here.
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")  # headless: no display needed during training
import matplotlib.pyplot as plt

from reporting.schema import TrainingReport


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _series(report: TrainingReport, metric: str):
    xs, ys = [], []
    for rec in report.epochs:
        if metric in rec.metrics:
            xs.append(rec.epoch + 1)        # 1-based for display
            ys.append(rec.metrics[metric])
    return xs, ys


def metric_curve(report: TrainingReport, train_metric: str,
                 title: str | None = None, ylabel: str | None = None) -> str | None:
    """Plot a train metric and its ``val_`` counterpart over epochs -> base64 PNG.

    Returns None if the metric is absent so callers can skip empty plots.
    """
    tx, ty = _series(report, train_metric)
    vx, vy = _series(report, f"val_{train_metric}")
    if not tx and not vx:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    if tx:
        ax.plot(tx, ty, marker="o", ms=3, label="train")
    if vx:
        ax.plot(vx, vy, marker="o", ms=3, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel or train_metric)
    ax.set_title(title or train_metric)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _fig_to_base64(fig)


# Friendly titles for the Event/Time model's known heads; anything not listed
# still gets plotted by auto_curves using its raw metric name.
_PRETTY = {
    "loss": ("Total loss", "loss"),
    "event_output_loss": ("Event head loss (cross-entropy)", "loss"),
    "event_output_acc": ("Event prediction accuracy", "accuracy"),
    "time_output_loss": ("Time head loss (MAE)", "loss"),
    "time_output_mae": ("Time prediction MAE", "MAE"),
}


def auto_curves(report: TrainingReport) -> list[tuple[str, str]]:
    """Discover every plottable metric and render it.

    Returns a list of ``(title, base64_png)``. Loss is placed first; remaining
    metrics follow in a stable order. Validation-only keys are folded into their
    base metric (so ``val_loss`` and ``loss`` share one chart).
    """
    base_metrics: list[str] = []
    seen = set()
    for rec in report.epochs:
        for k in rec.metrics:
            base = k[4:] if k.startswith("val_") else k
            if base not in seen:
                seen.add(base)
                base_metrics.append(base)

    # Loss first, then the rest in discovery order.
    base_metrics.sort(key=lambda m: (m != "loss", m))

    charts: list[tuple[str, str]] = []
    for m in base_metrics:
        title, ylabel = _PRETTY.get(m, (m, m))
        png = metric_curve(report, m, title=title, ylabel=ylabel)
        if png:
            charts.append((title, png))
    return charts


def epoch_duration_curve(report: TrainingReport) -> str | None:
    """Bar chart of per-epoch wall-clock time -> base64 PNG."""
    xs = [rec.epoch + 1 for rec in report.epochs]
    ys = [rec.duration_sec for rec in report.epochs]
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(xs, ys, color="#4C78A8")
    ax.set_xlabel("epoch")
    ax.set_ylabel("seconds")
    ax.set_title("Epoch wall-clock time")
    ax.grid(True, axis="y", alpha=0.3)
    return _fig_to_base64(fig)
