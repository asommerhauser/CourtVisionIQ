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
import re
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

from reporting.report_artifacts import ReportArtifacts, new_run_id, DEFAULT_REPORTS_ROOT
from simulation.stats import ADVANCED_LABELS, BOX_STATS
from simulation.eval_metrics import score_win_view

# Evaluation ("test run") outputs live under results/, split from the model-training reports/ tree.
# One folder per eval run: results/v<version>/<eval-name>/ with report.html + report.json at the
# root and the queryable parquet under data/. (See resolve_results_run_dir / write_eval_report.)
DEFAULT_RESULTS_ROOT = "./results"


def resolve_results_run_dir(version: str, *, name: str | None = None,
                            holdout_total: int | None = None,
                            results_root: str = DEFAULT_RESULTS_ROOT) -> Path:
    """Pick (and create) the results run dir for an evaluation under results/v<version>/.

    With ``name`` -> results/v<version>/<name> (stable; a re-run resumes it). Without a name -> the
    latest ``eval-NNN`` if it is still incomplete (fewer per-game ``record.json`` than
    ``holdout_total``), so a batched eval keeps filling one folder; otherwise the next ``eval-NNN``.
    """
    base = Path(results_root) / f"v{version}"
    base.mkdir(parents=True, exist_ok=True)

    if name:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name)).strip("-") or "eval"
        run = base / slug
        run.mkdir(parents=True, exist_ok=True)
        return run

    existing = sorted((d for d in base.iterdir()
                       if d.is_dir() and re.fullmatch(r"eval-\d+", d.name)),
                      key=lambda d: int(d.name.split("-")[1]))
    if existing:
        latest = existing[-1]
        done = len(list((latest / "games").glob("*/record.json")))
        if holdout_total is None or done < holdout_total:
            return latest  # resume the in-progress run instead of spawning a new folder
    nxt = (int(existing[-1].name.split("-")[1]) + 1) if existing else 1
    run = base / f"eval-{nxt:03d}"
    run.mkdir(parents=True, exist_ok=True)
    return run

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
                 run_name: str | None = None, tuning: dict | None = None) -> dict:
    """Package the harness output into a serializable report dict.

    ``tuning`` is the snapshot of rollout dials that produced this run (DELTA_TIME_SCALE, the
    temperatures, the rotation/clock knobs, …); it defaults to the live values from ``config`` so
    every eval report records exactly the tuning behind it for cross-run analysis.
    """
    if tuning is None:
        from config import tuning_snapshot
        tuning = tuning_snapshot()
    return {
        "run_id": new_run_id(run_name),
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "platform": platform.platform(),
        "n_games": len(records),
        "n_sims": n_sims,
        "tuning": tuning,
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


def _calibration_plot(win: dict, title: str = "Win-probability calibration") -> str | None:
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
    ax.set_title(title)
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


def _progression_chart(prog: list[dict]) -> str | None:
    """Trend of the key tuning-target metrics across the distinct-tuning segments."""
    if len(prog) < 2:
        return None
    xs = [p["segment"] for p in prog]
    panels = [("score_brier", "Score Brier", "#4C78A8"),
              ("spread_mae", "Spread MAE (pts)", "#E45756"),
              ("pace_bias", "Pace bias (pred−actual)", "#54A24B")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (key, title, color) in zip(axes, panels):
        ys = [p["metrics"][key] for p in prog]
        ax.plot(xs, ys, "-o", color=color)
        if key == "pace_bias":
            ax.axhline(0, color="#9ca3af", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("tuning segment")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


# --------------------------------------------------------------------------- HTML

def _esc(v) -> str:
    return _html.escape(str(v))


def _cards(headline: dict) -> str:
    def card(label, value):
        return (f"<div class='card'><div class='label'>{_esc(label)}</div>"
                f"<div class='value'>{_esc(value)}</div></div>")
    cards = [
        card("Win-pick (vote)", f"{headline['pick_accuracy'] * 100:.1f}%"),
        card("Brier (vote)", f"{headline['brier']:.3f}"),
    ]
    # Score-based winner (mean predicted margin) — shown alongside the vote headline when present.
    if "score_pick_accuracy" in headline:
        cards += [
            card("Win-pick (score)", f"{headline['score_pick_accuracy'] * 100:.1f}%"),
            card("Brier (score)", f"{headline['score_brier']:.3f}"),
        ]
    cards += [
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
    win_score = agg.get("win_score")

    # Method comparison: majority vote vs average predicted score, side by side.
    comp_rows = [["pick accuracy",
                  f"{win['pick_accuracy'] * 100:.1f}%",
                  f"{win_score['pick_accuracy'] * 100:.1f}%" if win_score else "—"],
                 ["Brier score",
                  f"{win['brier']:.4f}",
                  f"{win_score['brier']:.4f}" if win_score else "—"],
                 ["log-loss",
                  f"{win['log_loss']:.4f}",
                  f"{win_score['log_loss']:.4f}" if win_score else "—"]]
    comparison = _table(["metric", "majority vote", "average score"], comp_rows)

    rows = []
    for r in sorted(records, key=lambda x: x["game_id"]):
        sv = score_win_view(r)
        vote_mark = "✓" if r["pick_correct"] else "✗"
        score_mark = "✓" if sv["pick_correct"] else "✗"
        rows.append([
            r["game_id"], r["actual_winner"],
            f"{r['win_prob_home'] * 100:.0f}%", r["pred_pick"], vote_mark,
            f"{sv['win_prob_home'] * 100:.0f}%", sv["pick"], score_mark,
            f"{r['pred_home_score']:.0f}-{r['pred_away_score']:.0f}",
            f"{r['actual_home_score']}-{r['actual_away_score']}",
        ])
    table = _table(["game", "actual",
                    "vote P(home)", "vote pick", "✓",
                    "score P(home)", "score pick", "✓",
                    "pred score (H-A)", "actual score"], rows)

    charts = []
    cal_vote = _calibration_plot(win, "Calibration — majority vote")
    if cal_vote:
        charts.append(f"<img alt='calibration (vote)' src='data:image/png;base64,{cal_vote}'/>")
    if win_score:
        cal_score = _calibration_plot(win_score, "Calibration — average score")
        if cal_score:
            charts.append(
                f"<img alt='calibration (score)' src='data:image/png;base64,{cal_score}'/>")
    img = f"<div class='charts'>{''.join(charts)}</div>" if charts else ""

    return ("<h2>Win prediction (headline)</h2>"
            "<p class='sub'>Two ways to call the winner. <b>Majority vote</b>: the share of sims the "
            "home team won is its win probability, and the majority is the pick. <b>Average score</b>: "
            "the winner is the sign of the mean predicted margin, with the probability from the sims' "
            "margin spread (normal approx) — more robust when the box-score averages are good but "
            "individual sims are coin-flippy.</p>" + comparison + img + table)


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


def _fmt_dial(v) -> str:
    """Compact display of a tuning dial value (floats trimmed; dict-strings passed through)."""
    if isinstance(v, float):
        return f"{v:g}"
    return _esc(v)


def _progression_section(agg: dict) -> str:
    """How the headline metrics moved across distinct-tuning segments (alongside the overall)."""
    prog = agg.get("progression") or []
    if not prog:
        return ""
    rows = []
    for p in prog:
        m = p["metrics"]
        gids = p["game_ids"]
        span = f"{min(gids)}–{max(gids)} ({p['n_games']})" if gids else f"({p['n_games']})"
        changed = p["changed_dials"]
        changed_txt = ("(baseline)" if p["segment"] == 1 and not changed
                       else " · ".join(f"{k}={_fmt_dial(v)}" for k, v in changed.items()) or "—")
        rows.append([
            p["segment"], span, changed_txt,
            f"{m['pace_bias']:+.1f}", f"{m['fga_bias']:+.1f}", f"{m['efg_bias'] * 100:+.1f}%",
            f"{m['pick_accuracy'] * 100:.0f}%", f"{m['brier']:.3f}",
            f"{m['score_pick_accuracy'] * 100:.0f}%", f"{m['score_brier']:.3f}",
            f"{m['spread_mae']:.1f}", f"{m['points_mae']:.1f}",
        ])
    table = _table(["seg", "games", "tuning change", "pace bias", "FGA bias", "eFG bias",
                    "vote pick", "vote Brier", "score pick", "score Brier",
                    "spread MAE", "PTS MAE"], rows)
    chart = _progression_chart(prog)
    img = (f"<div class='charts'><img alt='progression' "
           f"src='data:image/png;base64,{chart}'/></div>") if chart else ""
    note = ("Holdout segmented by <b>distinct tuning</b> in evaluation order — each row is a stretch of "
            "games simmed under one set of dials, with what changed vs the prior segment. Tracks how "
            "the model evolved as you retuned, alongside the overall aggregate above. "
            "(One row only = the whole holdout ran under a single tuning, or per-game tuning wasn't "
            "recorded.)")
    return f"<h2>Tuning progression</h2><p class='sub'>{note}</p>{table}{img}"


def _tuning_section(report: dict) -> str:
    """Run-configuration / tuning dials used for this eval (recorded for cross-run analysis)."""
    tuning = report.get("tuning") or {}
    if not tuning:
        return ""
    rows = [[k, v] for k, v in tuning.items()]
    table = _table(["dial", "value"], rows, cls="kv2")
    return ("<h2>Run configuration (tuning)</h2>"
            "<p class='sub'>Rollout dials used for this run, captured from <code>config</code>. "
            "Also written to <code>run_summary.parquet</code> (one row per run) with the headline "
            "outcomes for knobs→results analysis across runs.</p>" + table)


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
        _progression_section(agg),
        _accuracy_section("Team box-score accuracy", agg["team_accuracy"],
                          agg["team_reliability"],
                          "Predicted (mean over sims) vs actual per team; 'sim std' is the average "
                          "per-game spread across sims."),
        _advanced_section(agg),
        _accuracy_section("Per-player box-score accuracy", agg["player_accuracy"],
                          agg["player_reliability"],
                          "Players matched by name across sims and the real game (absent = 0)."),
        _example_box_section(records),
        _tuning_section(report),
        "<footer>Generated by CourtVisionIQ evaluation harness.</footer>",
    ]
    return ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>evaluation — {_esc(report['run_id'])}</title>"
            f"<style>{_STYLE}</style></head><body>{''.join(sections)}</body></html>")


# --------------------------------------------------------------------------- Parquet

def _games_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        sv = score_win_view(r)  # average-score winner (derived from the stored margin scalars)
        rows.append({
            "game_id": r["game_id"], "n_sims": r["n_sims"],
            "win_prob_home": r["win_prob_home"], "pred_pick": r["pred_pick"],
            "actual_winner": r["actual_winner"], "pick_correct": r["pick_correct"],
            "score_win_prob_home": sv["win_prob_home"], "score_pick": sv["pick"],
            "score_pick_correct": sv["pick_correct"],
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
    # Average-score winner (parallel to the vote scope above).
    if agg.get("win_score"):
        for metric in ("pick_accuracy", "brier", "log_loss"):
            rows.append({"scope": "win_score", "metric": metric,
                         "predicted": agg["win_score"][metric],
                         "actual": None, "mae": None, "bias": None})
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


def _progression_frame(agg: dict) -> pd.DataFrame:
    """One row per distinct-tuning segment: metrics + the tuning dials + what changed.

    Complements the one-row ``run_summary.parquet`` (which records a single run-level tuning) by
    capturing the *within-run* trajectory when the dials change between eval batches.
    """
    rows = []
    for p in (agg.get("progression") or []):
        gids = p["game_ids"]
        row = {
            "segment": p["segment"], "n_games": p["n_games"],
            "game_id_min": min(gids) if gids else None,
            "game_id_max": max(gids) if gids else None,
            "evaluated_at": p.get("evaluated_at"),
            "changed_dials": json.dumps(p["changed_dials"], sort_keys=True),
            **p["metrics"],
            **(p["tuning"] or {}),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _run_summary_frame(report: dict) -> pd.DataFrame:
    """One row joining the run's tuning dials to its headline outcomes (cross-run analysis table).

    Concatenating these single-row tables across runs gives a knobs→results table: every tuning
    dial alongside pick accuracy, Brier, spread/points MAE, and the pace / FGA / eFG biases.
    """
    agg = report["aggregate"]

    def _bias(block: str, metric: str):
        return agg.get(block, {}).get(metric, {}).get("bias")

    row = {
        "run_id": report["run_id"],
        "run_name": report.get("run_name"),
        "created_at": report.get("created_at"),
        "git_commit": report.get("git_commit"),
        "n_games": report["n_games"],
        "n_sims": report["n_sims"],
        # How many distinct tunings the holdout spanned (1 = single tuning; >1 = retuned mid-run, so
        # the run-level dials below are only the last segment's — see progression.parquet).
        "n_tuning_segments": len(agg.get("progression") or []),
        # Headline outcomes (both winner methods).
        "pick_accuracy": agg["headline"]["pick_accuracy"],
        "brier": agg["headline"]["brier"],
        "score_pick_accuracy": agg["headline"].get("score_pick_accuracy"),
        "score_brier": agg["headline"].get("score_brier"),
        "spread_mae": agg["headline"]["spread_mae"],
        "points_mae": agg["headline"]["points_mae"],
        # Key biases the tuning targets (predicted − actual, per team).
        "pace_bias": _bias("advanced", "pace"),
        "fga_bias": _bias("team_accuracy", "fga"),
        "efg_bias": _bias("advanced", "efg"),
        # Every tuning dial that produced the run.
        **(report.get("tuning") or {}),
    }
    return pd.DataFrame([row])


def write_eval_report(report: dict, *, reports_root: str = DEFAULT_REPORTS_ROOT,
                      run_dir: str | Path | None = None):
    """Write report.html / report.json / the Parquet tables. Returns the run dir Path.

    ``run_dir`` (the new results layout): write report.html + report.json at the folder root and the
    parquet under ``<run_dir>/data/``. Without it (legacy), write everything flat under
    ``<reports_root>/evaluation/<run_id>/``.
    """
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        html_path, json_path = run_dir / "report.html", run_dir / "report.json"
        data_dir = run_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        arts = ReportArtifacts.for_run("evaluation", report["run_id"], root=reports_root)
        run_dir = arts.ensure_dir()
        html_path, json_path, data_dir = arts.html_path, arts.json_path, run_dir

    html_path.write_text(render_html(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    _games_frame(report["records"]).to_parquet(data_dir / "games.parquet", index=False)
    _box_players_frame(report["records"]).to_parquet(data_dir / "box_players.parquet", index=False)
    _summary_frame(report["aggregate"]).to_parquet(data_dir / "summary.parquet", index=False)
    _run_summary_frame(report).to_parquet(data_dir / "run_summary.parquet", index=False)
    prog = _progression_frame(report["aggregate"])
    if not prog.empty:
        prog.to_parquet(data_dir / "progression.parquet", index=False)
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
