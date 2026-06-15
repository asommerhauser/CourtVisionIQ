"""
Render a TrainingReport into a single, self-contained HTML file.

Zero templating dependency (pure stdlib + f-strings) so the report stays easy to
generate anywhere. Graphs are embedded as base64 PNGs, so the .html opens
standalone in any browser with nothing else alongside it.

Layout is metric-agnostic: the epoch table columns are derived from whatever
metrics the run actually produced, so a new model's report renders correctly with
no changes here.
"""
from __future__ import annotations

import html as _html

from reporting.schema import TrainingReport
from reporting import plots


def _esc(v) -> str:
    return _html.escape(str(v))


def _kv_table(title: str, rows: dict) -> str:
    body = "".join(
        f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows.items()
    )
    return f"<h2>{_esc(title)}</h2><table class='kv'>{body}</table>"


def _metric_columns(report: TrainingReport) -> list[str]:
    cols: list[str] = []
    seen = set()
    for rec in report.epochs:
        for k in rec.metrics:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    # Stable, readable ordering: loss family first, val_ paired after train.
    cols.sort(key=lambda c: (c.replace("val_", ""), c.startswith("val_")))
    return cols


def _epoch_table(report: TrainingReport) -> str:
    cols = _metric_columns(report)
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    header = f"<tr><th>epoch</th><th>time (s)</th><th>lr</th>{head}</tr>"

    best = report.best_epoch
    rows = []
    for rec in report.epochs:
        cells = []
        for c in cols:
            val = rec.metrics.get(c)
            cells.append(f"<td>{val:.5f}</td>" if isinstance(val, (int, float)) else "<td>—</td>")
        cls = " class='best'" if best is not None and rec.epoch == best else ""
        rows.append(
            f"<tr{cls}><td>{rec.epoch + 1}</td>"
            f"<td>{rec.duration_sec:.2f}</td>"
            f"<td>{rec.learning_rate:.2e}</td>"
            f"{''.join(cells)}</tr>"
        )
    return f"<h2>Epoch-by-epoch metrics</h2><table class='epochs'>{header}{''.join(rows)}</table>"


_STYLE = """
:root { --fg:#1f2329; --muted:#6b7280; --line:#e5e7eb; --accent:#4C78A8; --best:#fff7e6; }
* { box-sizing:border-box; }
body { font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       color:var(--fg); margin:0; padding:32px; max-width:1100px; }
h1 { margin:0 0 4px; font-size:26px; }
h2 { margin:32px 0 10px; font-size:18px; border-bottom:2px solid var(--line); padding-bottom:6px; }
.sub { color:var(--muted); margin:0 0 8px; }
.badge { display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px;
         font-weight:600; color:#fff; }
.badge.completed { background:#2ca02c; }
.badge.early_stopped { background:#ff7f0e; }
.badge.failed { background:#d62728; }
table { border-collapse:collapse; font-size:13px; }
table.kv th { text-align:left; color:var(--muted); font-weight:600; padding:4px 16px 4px 0; vertical-align:top; }
table.kv td { padding:4px 0; }
table.epochs { width:100%; }
table.epochs th, table.epochs td { border:1px solid var(--line); padding:5px 8px; text-align:right; }
table.epochs th { background:#f9fafb; text-align:center; }
table.epochs tr.best { background:var(--best); font-weight:600; }
.cards { display:flex; gap:16px; flex-wrap:wrap; margin:8px 0 4px; }
.card { border:1px solid var(--line); border-radius:10px; padding:14px 18px; min-width:160px; }
.card .label { color:var(--muted); font-size:12px; }
.card .value { font-size:22px; font-weight:700; margin-top:2px; }
.charts img { max-width:100%; border:1px solid var(--line); border-radius:8px; margin:10px 0; }
footer { margin-top:40px; color:var(--muted); font-size:12px; }
"""


def _summary_cards(report: TrainingReport) -> str:
    def card(label, value):
        return f"<div class='card'><div class='label'>{_esc(label)}</div><div class='value'>{_esc(value)}</div></div>"

    best_val = f"{report.best_val_loss:.5f}" if report.best_val_loss is not None else "—"
    best_ep = (report.best_epoch + 1) if report.best_epoch is not None else "—"
    dur = f"{report.duration_sec:.1f}s"
    if report.duration_sec >= 60:
        dur = f"{report.duration_sec / 60:.1f} min"
    cards = [
        card("Epochs run", report.epochs_run),
        card("Best val_loss", best_val),
        card("Best epoch", best_ep),
        card("Total time", dur),
    ]
    return f"<div class='cards'>{''.join(cards)}</div>"


def render(report: TrainingReport) -> str:
    """Build the full self-contained HTML document for a run."""
    cfg = report.config
    env = report.environment
    data = report.data
    minfo = report.model

    header = (
        f"<h1>{_esc(report.model_key)} — training report</h1>"
        f"<p class='sub'>Run <code>{_esc(report.run_id)}</code> · "
        f"<span class='badge {_esc(report.status)}'>{_esc(report.status)}</span> · "
        f"{_esc(report.started_at)} → {_esc(report.ended_at)}</p>"
    )

    sections = [header, _summary_cards(report)]

    if cfg:
        rows = {
            "epochs (planned)": cfg.epochs_planned,
            "batch size": cfg.batch_size,
            "learning rate": cfg.lr,
            "time loss weight": cfg.time_loss_weight,
            "early-stop patience": cfg.patience,
            "mixed precision": cfg.mixed_precision,
            "jit compile": cfg.jit_compile,
        }
        for k, v in (cfg.arch or {}).items():
            rows[f"arch.{k}"] = v
        sections.append(_kv_table("Hyperparameters", rows))

    if env:
        sections.append(_kv_table("Environment", {
            "device": f"{env.device}" + (f" ({', '.join(env.device_names)})" if env.device_names else ""),
            "mixed precision policy": env.mixed_precision_policy,
            "python": env.python_version,
            "tensorflow": env.tensorflow_version,
            "keras": env.keras_version,
            "platform": env.platform,
            "git commit": env.git_commit,
        }))

    if data:
        rows = {
            "train games": data.train_games,
            "test games": data.test_games,
            "sequence length": data.sequence_length,
        }
        for k, v in (data.vocab_sizes or {}).items():
            rows[f"vocab.{k}"] = v
        for k, v in (data.norm_stats or {}).items():
            rows[f"norm.{k}"] = v
        sections.append(_kv_table("Data", rows))

    if minfo:
        sections.append(_kv_table("Model size", {
            "total params": f"{minfo.total_params:,}",
            "trainable params": f"{minfo.trainable_params:,}",
            "non-trainable params": f"{minfo.non_trainable_params:,}",
            "layers": minfo.num_layers,
        }))

    # Graphs.
    charts = plots.auto_curves(report)
    dur_png = plots.epoch_duration_curve(report)
    if charts or dur_png:
        imgs = "".join(
            f"<img alt='{_esc(title)}' src='data:image/png;base64,{png}'/>"
            for title, png in charts
        )
        if dur_png:
            imgs += f"<img alt='epoch time' src='data:image/png;base64,{dur_png}'/>"
        sections.append(f"<h2>Graphs</h2><div class='charts'>{imgs}</div>")

    if report.final_test_metrics:
        sections.append(_kv_table(
            "Final test metrics",
            {k: (f"{v:.5f}" if isinstance(v, (int, float)) else v)
             for k, v in report.final_test_metrics.items()},
        ))

    sections.append(_epoch_table(report))
    sections.append("<footer>Generated by CourtVisionIQ reporting.</footer>")

    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{_esc(report.model_key)} — {_esc(report.run_id)}</title>"
        f"<style>{_STYLE}</style></head><body>{''.join(sections)}</body></html>"
    )
