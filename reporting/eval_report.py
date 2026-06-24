"""
eval_report.py — render a holdout *evaluation* run into a self-contained report.

The training reports (``html_report.py`` / ``parquet_store.py``) are epoch/loss shaped; an
evaluation run is a different beast (per-game win/spread accuracy, predicted-vs-actual box and
advanced stats). This module is the evaluation analogue: it consumes the records + aggregates
produced by ``simulation/evaluation.py`` and writes, under the *same* on-disk convention as the
training reports (``<reports_root>/evaluation/<run_id>/``):

  * ``report.html``        — self-contained HTML (cards, win/spread/box/advanced tables, embedded
                             calibration + margin-scatter + bias PNGs, an example avg/std box).
  * ``report.json``        — the full report dict, lossless.
  * ``games.parquet``      — one row per holdout game (win/spread/score scalars).
  * ``box_players.parquet``— one row per (game, side, player): predicted mean & std + actual.
  * ``summary.parquet``    — long-format aggregate metrics (scope, metric, predicted, actual, …).

Self-contained and TensorFlow-free so it can be unit-tested without trained models.
"""
from __future__ import annotations

import base64
import html as _html
import io
import json
import platform
import subprocess
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

from reporting.report_artifacts import ReportArtifacts, new_run_id, DEFAULT_REPORTS_ROOT
from simulation.stats import ADVANCED_LABELS, BOX_STATS

# Friendly labels for the box-accuracy stat keys.
_STAT_LABELS = {
    "pts": "PTS", "fga": "FGA", "fgm": "FGM", "tpa": "3PA", "tpm": "3PM",
    "fta": "FTA", "ftm": "FTM", "oreb": "OREB", "dreb": "DREB", "ast": "AST",
    "stl": "STL", "blk": "BLK", "tov": "TO", "pf": "PF",
}
# Which advanced metrics are percentages (formatted ×100 with a % feel).
_ADV_PCT = {"efg", "tov_pct", "oreb_pct", "dreb_pct"}


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def build_report(*, records: list[dict], aggregate: dict, n_sims: int,
                 run_name: str | None = None) -> dict:
    """Package the harness output into a serializable report dict."""
    return {
        "run_id": new_run_id(run_name),
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "n_games": len(records),
        "n_sims": n_sims,
        "aggregate": aggregate,
        "records": records,
    }


# --------------------------------------------------------------------------- plots

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _calibration_plot(win: dict) -> str | None:
    cal = win.get("calibration") or []
    if not cal:
        return None
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="perfect")
    xs = [c["pred_mean"] for c in cal]
    ys = [c["obs_rate"] for c in cal]
    sizes = [20 + 12 * c["n"] for c in cal]
    ax.scatter(xs, ys, s=sizes, color="#4C78A8", zorder=3, label="observed")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("predicted home win probability")
    ax.set_ylabel("observed home win rate")
    ax.set_title("Win-probability calibration")
    ax.grid(True, alpha=0.3); ax.legend()
    return _fig_to_base64(fig)


def _margin_plot(records: list[dict]) -> str | None:
    if not records:
        return None
    pred = [r["pred_margin_mean"] for r in records]
    act = [r["actual_margin"] for r in records]
    lim = max(1.0, max(abs(v) for v in pred + act) * 1.1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([-lim, lim], [-lim, lim], "--", color="#9ca3af")
    ax.axhline(0, color="#e5e7eb"); ax.axvline(0, color="#e5e7eb")
    ax.scatter(act, pred, color="#4C78A8", alpha=0.7, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("actual margin (home − away)")
    ax.set_ylabel("predicted mean margin")
    ax.set_title("Point spread: predicted vs actual")
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _bias_plot(team_acc: dict) -> str | None:
    keys = [k for k in _STAT_LABELS if k in team_acc]
    if not keys:
        return None
    labels = [_STAT_LABELS[k] for k in keys]
    biases = [team_acc[k]["bias"] for k in keys]
    colors = ["#d62728" if b > 0 else "#2ca02c" for b in biases]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(labels, biases, color=colors)
    ax.axhline(0, color="#1f2329", linewidth=0.8)
    ax.set_ylabel("predicted − actual (per team)")
    ax.set_title("Team box-score bias by stat")
    ax.grid(True, axis="y", alpha=0.3)
    return _fig_to_base64(fig)


# --------------------------------------------------------------------------- HTML

def _esc(v) -> str:
    return _html.escape(str(v))


def _cards(headline: dict) -> str:
    def card(label, value):
        return (f"<div class='card'><div class='label'>{_esc(label)}</div>"
                f"<div class='value'>{_esc(value)}</div></div>")
    cards = [
        card("Win-pick accuracy", f"{headline['pick_accuracy'] * 100:.1f}%"),
        card("Brier score", f"{headline['brier']:.3f}"),
        card("Point-spread MAE", f"{headline['spread_mae']:.1f} pts"),
        card("Team PTS MAE", f"{headline['points_mae']:.1f} pts"),
    ]
    return f"<div class='cards'>{''.join(cards)}</div>"


def _table(headers: list[str], rows: list[list], cls: str = "data") -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table class='{cls}'><tr>{head}</tr>{body}</table>"


def _win_section(agg: dict, records: list[dict]) -> str:
    win = agg["win"]
    rows = []
    for r in sorted(records, key=lambda x: x["game_id"]):
        mark = "✓" if r["pick_correct"] else "✗"
        rows.append([
            r["game_id"], f"{r['win_prob_home'] * 100:.0f}%", r["pred_pick"],
            r["actual_winner"], mark,
            f"{r['pred_home_score']:.0f}-{r['pred_away_score']:.0f}",
            f"{r['actual_home_score']}-{r['actual_away_score']}",
        ])
    table = _table(["game", "P(home win)", "pick", "actual", "correct",
                    "pred score (H-A)", "actual score"], rows)
    cal = _calibration_plot(win)
    img = (f"<div class='charts'><img alt='calibration' "
           f"src='data:image/png;base64,{cal}'/></div>") if cal else ""
    summary = _table(["metric", "value"], [
        ["games", win["n"]],
        ["pick accuracy", f"{win['pick_accuracy'] * 100:.1f}%"],
        ["Brier score", f"{win['brier']:.4f}"],
        ["log-loss", f"{win['log_loss']:.4f}"],
    ], cls="kv2")
    return ("<h2>Win prediction (headline)</h2>"
            "<p class='sub'>Majority of the per-game sims is the pick; the share that picks home "
            "is its win probability.</p>" + summary + img + table)


def _spread_section(agg: dict, records: list[dict]) -> str:
    sp = agg["spread"]
    within = " · ".join(f"≤{w}: {sp['within'][w] * 100:.0f}%" for w in sp["within"])
    summary = _table(["metric", "value"], [
        ["MAE", f"{sp['mae']:.2f} pts"],
        ["bias (pred − actual)", f"{sp['bias']:+.2f} pts"],
        ["RMSE", f"{sp['rmse']:.2f} pts"],
        ["correlation", f"{sp['corr']:.3f}"],
        ["within-N hit rate", within],
    ], cls="kv2")
    scatter = _margin_plot(records)
    img = (f"<div class='charts'><img alt='margin' "
           f"src='data:image/png;base64,{scatter}'/></div>") if scatter else ""
    return "<h2>Point spread</h2>" + summary + img


def _accuracy_section(title: str, acc: dict, reliability: dict, note: str) -> str:
    rows = []
    for k, lab in _STAT_LABELS.items():
        if k not in acc:
            continue
        a = acc[k]
        rows.append([lab, f"{a['pred_mean']:.1f}", f"{a['actual_mean']:.1f}",
                     f"{a['mae']:.2f}", f"{a['bias']:+.2f}", f"{reliability.get(k, 0.0):.2f}"])
    table = _table(["stat", "pred", "actual", "MAE", "bias", "sim std"], rows)
    return f"<h2>{_esc(title)}</h2><p class='sub'>{_esc(note)}</p>" + table


def _advanced_section(agg: dict) -> str:
    rows = []
    for k, lab in ADVANCED_LABELS.items():
        a = agg["advanced"][k]
        scale = 100.0 if k in _ADV_PCT else 1.0
        suffix = "%" if k in _ADV_PCT else ""
        rows.append([lab, f"{a['pred_mean'] * scale:.1f}{suffix}",
                     f"{a['actual_mean'] * scale:.1f}{suffix}",
                     f"{a['mae'] * scale:.2f}{suffix}", f"{a['bias'] * scale:+.2f}{suffix}"])
    table = _table(["stat", "pred", "actual", "MAE", "bias"], rows)
    bias = _bias_plot(agg["team_accuracy"])
    img = (f"<div class='charts'><img alt='bias' "
           f"src='data:image/png;base64,{bias}'/></div>") if bias else ""
    return ("<h2>Advanced stats — four factors + pace</h2>"
            "<p class='sub'>Predicted (mean over sims, per team) vs actual, pooled across the "
            "holdout set.</p>" + table + img)


def _example_box_section(records: list[dict]) -> str:
    if not records:
        return ""
    r = sorted(records, key=lambda x: x["game_id"])[0]
    out = [f"<h2>Example average & std box score — game {_esc(r['game_id'])}</h2>",
           "<p class='sub'>Per-player mean over the sims with the simulation std in parentheses, "
           "next to the actual line.</p>"]
    headers = ["Player", "MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "STL", "BLK", "TO", "PF"]
    for side in ("home", "away"):
        rows = []
        avg, std, actual = r["player_avg"][side], r["player_std"][side], r["player_actual"][side]
        for name in sorted(r["players"][side], key=lambda n: avg[n]["pts"], reverse=True):
            m, s = avg[name], std[name]
            a = actual.get(name, {f: 0.0 for f in BOX_STATS})

            def cell(stat):  # predicted mean (± std)  /  actual
                return f"{m[stat]:.1f}±{s[stat]:.1f} / {a[stat]:.0f}"

            rows.append([
                name, f"{m['seconds'] / 60:.1f} / {a['seconds'] / 60:.0f}",
                cell("pts"),
                f"{m['fgm']:.1f}-{m['fga']:.1f} / {a['fgm']:.0f}-{a['fga']:.0f}",
                f"{m['tpm']:.1f}-{m['tpa']:.1f} / {a['tpm']:.0f}-{a['tpa']:.0f}",
                f"{m['ftm']:.1f}-{m['fta']:.1f} / {a['ftm']:.0f}-{a['fta']:.0f}",
                f"{m['oreb'] + m['dreb']:.1f} / {a['oreb'] + a['dreb']:.0f}",
                cell("ast"), cell("stl"), cell("blk"), cell("tov"), cell("pf"),
            ])
        out.append(f"<h3>{_esc(side)} (pred mean±std / actual)</h3>")
        out.append(_table(headers, rows))
    return "".join(out)


_STYLE = """
:root { --fg:#1f2329; --muted:#6b7280; --line:#e5e7eb; --accent:#4C78A8; }
* { box-sizing:border-box; }
body { font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       color:var(--fg); margin:0; padding:32px; max-width:1100px; }
h1 { margin:0 0 4px; font-size:26px; }
h2 { margin:34px 0 8px; font-size:19px; border-bottom:2px solid var(--line); padding-bottom:6px; }
h3 { margin:18px 0 6px; font-size:14px; color:var(--muted); text-transform:capitalize; }
.sub { color:var(--muted); margin:0 0 10px; font-size:13px; }
.cards { display:flex; gap:16px; flex-wrap:wrap; margin:14px 0; }
.card { border:1px solid var(--line); border-radius:10px; padding:14px 18px; min-width:170px; }
.card .label { color:var(--muted); font-size:12px; }
.card .value { font-size:24px; font-weight:700; margin-top:2px; }
table { border-collapse:collapse; font-size:13px; margin:6px 0 4px; }
table.data th, table.data td { border:1px solid var(--line); padding:5px 9px; text-align:right; }
table.data th { background:#f9fafb; text-align:center; }
table.data td:first-child, table.data th:first-child { text-align:left; }
table.kv2 th { text-align:left; color:var(--muted); font-weight:600; padding:3px 18px 3px 0; }
table.kv2 td { padding:3px 0; }
.charts img { max-width:100%; border:1px solid var(--line); border-radius:8px; margin:10px 0; }
footer { margin-top:42px; color:var(--muted); font-size:12px; }
"""


def render_html(report: dict) -> str:
    agg = report["aggregate"]
    records = report["records"]
    name = f" · <b>{_esc(report['run_name'])}</b>" if report.get("run_name") else ""
    header = (
        f"<h1>Holdout evaluation report</h1>"
        f"<p class='sub'>Run <code>{_esc(report['run_id'])}</code>{name} · "
        f"{_esc(report['n_games'])} games × {_esc(report['n_sims'])} sims · "
        f"{_esc(report['created_at'])} · commit {_esc(report['git_commit'] or '—')}</p>"
    )
    sections = [
        header,
        _cards(agg["headline"]),
        _win_section(agg, records),
        _spread_section(agg, records),
        _accuracy_section("Team box-score accuracy", agg["team_accuracy"],
                          agg["team_reliability"],
                          "Predicted (mean over sims) vs actual per team; 'sim std' is the average "
                          "per-game spread across sims."),
        _advanced_section(agg),
        _accuracy_section("Per-player box-score accuracy", agg["player_accuracy"],
                          agg["player_reliability"],
                          "Players matched by name across sims and the real game (absent = 0)."),
        _example_box_section(records),
        "<footer>Generated by CourtVisionIQ evaluation harness.</footer>",
    ]
    return ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>evaluation — {_esc(report['run_id'])}</title>"
            f"<style>{_STYLE}</style></head><body>{''.join(sections)}</body></html>")


# --------------------------------------------------------------------------- Parquet

def _games_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append({
            "game_id": r["game_id"], "n_sims": r["n_sims"],
            "win_prob_home": r["win_prob_home"], "pred_pick": r["pred_pick"],
            "actual_winner": r["actual_winner"], "pick_correct": r["pick_correct"],
            "pred_margin_mean": r["pred_margin_mean"], "pred_margin_std": r["pred_margin_std"],
            "actual_margin": r["actual_margin"],
            "pred_home_score": r["pred_home_score"], "pred_away_score": r["pred_away_score"],
            "actual_home_score": r["actual_home_score"], "actual_away_score": r["actual_away_score"],
        })
    return pd.DataFrame(rows)


def _box_players_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        for side in ("home", "away"):
            for name in r["players"][side]:
                avg, std = r["player_avg"][side][name], r["player_std"][side][name]
                actual = r["player_actual"][side].get(name, {f: 0.0 for f in BOX_STATS})
                row = {"game_id": r["game_id"], "side": side, "player": name}
                for f in BOX_STATS:
                    row[f"pred_{f}"] = avg[f]
                    row[f"std_{f}"] = std[f]
                    row[f"actual_{f}"] = actual[f]
                rows.append(row)
    return pd.DataFrame(rows)


def _summary_frame(agg: dict) -> pd.DataFrame:
    rows = []
    # Headline / win / spread scalars.
    rows.append({"scope": "win", "metric": "pick_accuracy",
                 "predicted": agg["win"]["pick_accuracy"], "actual": None, "mae": None, "bias": None})
    rows.append({"scope": "win", "metric": "brier",
                 "predicted": agg["win"]["brier"], "actual": None, "mae": None, "bias": None})
    for m in ("mae", "bias", "rmse", "corr"):
        rows.append({"scope": "spread", "metric": m, "predicted": None, "actual": None,
                     "mae": agg["spread"]["mae"] if m == "mae" else None,
                     "bias": agg["spread"]["bias"] if m == "bias" else None})
    # Per-stat accuracy blocks (team, player, advanced).
    for scope, block in (("team", agg["team_accuracy"]),
                         ("player", agg["player_accuracy"]),
                         ("advanced", agg["advanced"])):
        for metric, a in block.items():
            rows.append({"scope": scope, "metric": metric, "predicted": a["pred_mean"],
                         "actual": a["actual_mean"], "mae": a["mae"], "bias": a["bias"]})
    return pd.DataFrame(rows)


def write_eval_report(report: dict, *, reports_root: str = DEFAULT_REPORTS_ROOT):
    """Write report.html / report.json / the three Parquet tables. Returns the run dir Path."""
    arts = ReportArtifacts.for_run("evaluation", report["run_id"], root=reports_root)
    run_dir = arts.ensure_dir()

    arts.html_path.write_text(render_html(report), encoding="utf-8")
    arts.json_path.write_text(json.dumps(report, indent=2, default=_json_default),
                              encoding="utf-8")
    _games_frame(report["records"]).to_parquet(run_dir / "games.parquet", index=False)
    _box_players_frame(report["records"]).to_parquet(run_dir / "box_players.parquet", index=False)
    _summary_frame(report["aggregate"]).to_parquet(run_dir / "summary.parquet", index=False)
    return run_dir


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


__all__ = ["build_report", "render_html", "write_eval_report"]
