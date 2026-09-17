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
  * ``box_quarters.parquet``— one row per (game, period, side, stat): predicted vs actual.

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
import numpy as np
import pandas as pd

from reporting.report_artifacts import ReportArtifacts, new_run_id, DEFAULT_REPORTS_ROOT
from simulation.stats import ADVANCED_LABELS, BOX_STATS, MINUTES, REPORT_STATS, stat_value
from simulation.eval_metrics import minutes_closeness, score_win_view

# Evaluation ("test run") outputs live under results/, split from the model-training reports/ tree.
# One folder per eval run: results/<model>/<eval-name>/ with report.html + report.json at the
# root and the queryable parquet under data/. (See resolve_results_run_dir / write_eval_report.)
DEFAULT_RESULTS_ROOT = "./results"


def _slug(value: str) -> str:
    """Filesystem-safe folder name: anything outside [A-Za-z0-9_.-] collapses to a dash."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "eval"


def resolve_results_run_dir(model: str, *, name: str | None = None,
                            holdout_total: int | None = None,
                            results_root: str = DEFAULT_RESULTS_ROOT) -> Path:
    """Pick (and create) the results run dir for an evaluation under results/<model>/.

    ``model`` is the model name, already carrying any ``v`` prefix ("v1.0", "endgame-feats"), so
    runs sit beside the weights they came from.

    With ``name`` -> results/<model>/<name> (stable; a re-run resumes it). Without a name -> the
    latest ``eval-NNN`` if it is still incomplete (fewer per-game ``record.json`` than
    ``holdout_total``), so a batched eval keeps filling one folder; otherwise the next ``eval-NNN``.
    """
    base = Path(results_root) / _slug(model)
    base.mkdir(parents=True, exist_ok=True)

    if name:
        slug = _slug(name)
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


# The game ids a run covers, pinned inside the run dir. Written the first time a run is created and
# authoritative from then on: resumes, --shard children, --report-only and the pooled merge all read
# it instead of re-deriving from the training state, so a subset run's denominators stay right with
# no flag to remember and the run stays self-describing.
RUN_HOLDOUT_NAME = "holdout.json"
# Which rotating window the run scored, pinned beside the ids. A run dir written before 3.0
# has no such file and reads as window 0, which is true of every one of them.
RUN_WINDOW_NAME = "window.json"


def subset_holdout(ids, n: int | None):
    """Every ``len(ids)//n``-th id, ``n`` of them -- a subset that spans the whole holdout window.

    ``n=None`` returns the list unchanged. Stride rather than a head slice because the holdout is
    chronological: ``ids[:n]`` would draw every game from one narrow stretch of the calendar, with
    the same teams, rest states and injury context correlated across the whole sample.

    When ``n`` does not divide ``len(ids)`` the stride is rounded down and the tail is trimmed, so
    the sample can stop short of the last game (100 -> 30 covers ids[0::3][:30], i.e. through
    index 87). Exact divisors -- 100 -> 20, 50, 25, 10 -- span the full range.
    """
    ids = list(ids)
    if n is None:
        return ids
    if n < 1:
        raise ValueError(f"holdout subset must be >= 1, got {n}")
    if n > len(ids):
        raise ValueError(f"holdout subset of {n} asked for, but the holdout has {len(ids)} games")
    return ids[::len(ids) // n][:n]


def run_window(run_dir) -> int:
    """The rotating-window index a run dir was scored at; 0 for any run written before 3.0."""
    path = Path(run_dir) / RUN_WINDOW_NAME
    if not path.is_file():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8")).get("k", 0))


def pin_run_holdout(run_dir, full_holdout, *, subset: int | None = None,
                    window: int | None = None) -> list[int]:
    """The game ids this run covers, pinned to ``run_dir/holdout.json`` on first use.

    An existing pin wins: that is what makes a resume, a ``--shard`` child and a later
    ``--report-only`` agree on the denominator without being told the subset again. Asking for a
    ``subset`` that disagrees with the pin is an error rather than a silent re-slice -- the run's
    finished games were simulated against the pinned set, and re-slicing would report them under a
    total they never belonged to.

    ``window`` gets the same treatment, and needs it more. A subset that disagrees with the pin at
    least changes the game COUNT, so it is visible; a window that disagrees selects a different 100
    games of the same size, and without this check a resume under the wrong ``--window`` would
    quietly append games from another stretch of the calendar to a finished run and report the mix
    under one headline. The pinned ids would not even catch it, because the child reads them back
    from the pin -- it is the report's ``window`` field that would be the lie.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / RUN_HOLDOUT_NAME
    wanted = [int(g) for g in subset_holdout(full_holdout, subset)]

    window_path = run_dir / RUN_WINDOW_NAME
    if window is not None:
        if window_path.is_file():
            pinned_k = int(json.loads(window_path.read_text(encoding="utf-8")).get("k", 0))
            if pinned_k != int(window):
                raise ValueError(
                    f"{window_path} pins window k={pinned_k} for this run, but --window "
                    f"{window} was asked for. A run scores one window; use a new --run name.")
        else:
            window_path.write_text(json.dumps({"k": int(window), "size": len(wanted)}, indent=2),
                                   encoding="utf-8")

    if path.is_file():
        pinned = [int(g) for g in json.loads(path.read_text(encoding="utf-8"))]
        if subset is not None and pinned != wanted:
            raise ValueError(
                f"{path} pins {len(pinned)} games for this run, but --holdout {subset} selects "
                f"{len(wanted)}. Use a new --run name, or drop --holdout to keep the pinned set.")
        return pinned

    path.write_text(json.dumps(wanted, indent=2), encoding="utf-8")
    return wanted


# Friendly labels for the box-accuracy stat keys. "MIN" leads the way it does on a box score; it is
# derived from the stored ``seconds`` (see ``simulation.stats.stat_value``), so it renders for runs
# simulated before minutes was a reported stat.
_STAT_LABELS = {
    MINUTES: "MIN",
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


def build_report(*, records: list[dict], aggregate: dict, n_sims: int, window: int = 0,
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
        # Which rotating window of the untrained tail these games came from. Recorded on every
        # report because later windows sit further from the train cut -- March games judged on
        # January weights -- so drift with k is a finding, not noise (docs/v3_direction.md §4).
        "window": int(window),
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


def _minutes_plots(records: list[dict]) -> str | None:
    """Predicted vs actual player minutes: the scatter, and the signed-error distribution."""
    pred, act = [], []
    for r in records:
        for side in ("home", "away"):
            for name in r["players"][side]:
                pred.append(stat_value(r["player_avg"][side][name], MINUTES))
                act.append(stat_value(r["player_actual"][side].get(name, {}), MINUTES))
    if not pred:
        return None
    pred_a, act_a = np.asarray(pred, float), np.asarray(act, float)
    lim = max(1.0, float(max(pred_a.max(), act_a.max())) * 1.05)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax.plot([0, lim], [0, lim], "--", color="#9ca3af", zorder=1)
    ax.scatter(act_a, pred_a, s=14, alpha=0.35, color="#4C78A8", zorder=3)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("actual minutes")
    ax.set_ylabel("predicted minutes (mean over sims)")
    ax.set_title("Player minutes: predicted vs actual", fontsize=10)
    ax.grid(True, alpha=0.3)

    err = pred_a - act_a
    ax2.hist(err, bins=40, color="#B279A2")
    ax2.axvline(0, color="#1f2329", linewidth=0.8)
    ax2.set_xlabel("predicted − actual (minutes)")
    ax2.set_ylabel("player-games")
    ax2.set_title("Minutes error distribution", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _progression_chart(prog: list[dict]) -> str | None:
    """Trend of the key tuning-target metrics across the distinct-tuning segments."""
    if len(prog) < 2:
        return None
    xs = [p["segment"] for p in prog]
    panels = [("score_brier", "Score Brier", "#4C78A8"),
              ("spread_mae", "Spread MAE (pts)", "#E45756"),
              ("pace_bias", "Pace bias (pred−actual)", "#54A24B"),
              ("player_minutes_mae", "Player MIN MAE (min)", "#B279A2")]
    # Older reports predate the minutes panel; drop panels this run has no metric for.
    panels = [pn for pn in panels if pn[0] in prog[0]["metrics"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 3.2))
    for ax, (key, title, color) in zip(np.atleast_1d(axes), panels):
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
    # Player minutes — the rotation prediction, headlined next to the scoring ones.
    if "player_minutes_mae" in headline:
        cards.append(card("Player MIN MAE", f"{headline['player_minutes_mae']:.1f} min"))
    # The 3.0 gate numbers. corr(home, away) is the one metric that says whether the two teams
    # in a sim share a game at all; the real-game value is +0.35, and the target is printed
    # with it so the card is readable on its own.
    if headline.get("corr_home_away") is not None:
        cards.append(card("corr(home,away)",
                          f"{headline['corr_home_away']:+.3f} / {REAL_CORR_HOME_AWAY:+.2f}"))
    if headline.get("margin_dispersion_ratio"):
        cards.append(card("Margin dispersion",
                          f"{headline['margin_dispersion_ratio']:.2f}×"))
    return f"<div class='cards'>{''.join(cards)}</div>"



# Real 2022-23 values, measured over all 1,320 games with the same generate_box_score tally the
# report scores against (reporting/backfill_per_sim.py rebuilds the sim side the same way). These
# are the targets in docs/v3_direction.md SS3 W3, shown next to every sim number so the gap is
# legible without a second document.
REAL_CORR_HOME_AWAY = 0.3517
REAL_PACE_SD = 4.80
REAL_MARGIN_SD = 13.66
REAL_TOTAL_SD = 19.73
REAL_SIDE_PTS_SD = 12.07


def _joint_section(agg: dict) -> str:
    """Does a sim's two teams share a game? The W3 gate, and the clearest single diagnostic."""
    joint = agg.get("joint") or {}
    if not joint.get("n_games"):
        # A run evaluated before per-sim vectors were recorded. reporting/backfill_per_sim.py
        # rebuilds them from the sim play-by-plays still on disk.
        return ""
    rows = [
        ["corr(home pts, away pts)", f"{joint['corr_home_away']:+.4f} ± {joint['corr_se']:.4f}",
         f"{REAL_CORR_HOME_AWAY:+.4f}",
         "Do the two teams share a game? The whole joint structure in one number."],
        ["Pace sd (poss/48)", f"{joint['pace_sd']:.2f}", f"{REAL_PACE_SD:.2f}",
         "Within-game spread of tempo across sims, against the across-game spread in reality."],
        ["Margin sd", f"{joint['margin_sd']:.2f}", f"{REAL_MARGIN_SD:.2f}",
         "Too WIDE when the sides are independent: sqrt(2) × the per-side sd."],
        ["Total sd", f"{joint['total_sd']:.2f}", f"{REAL_TOTAL_SD:.2f}",
         "Too NARROW for the same reason — the other half of the same defect."],
        ["Per-side points sd", f"{joint['side_pts_sd']:.2f}", f"{REAL_SIDE_PTS_SD:.2f}",
         "The marginal. This one is right, which is what isolates the fault to the joint."],
    ]
    return (
        "<h2>Joint structure — do the two teams share a game?</h2>"
        "<p class='note'>Each quantity is computed <i>within</i> a game across its sims, then "
        "averaged over games. With independent sides, Var(H−A) and Var(H+A) are both "
        "VarH + VarA, so the margin comes out too wide and the total too narrow by the same "
        "missing covariance — which is why no shrinkage dial is the right fix. The real column "
        "is all 1,320 games of 2022-23, tallied the same way.</p>"
        + _table(["Quantity", "This run", "Real", "What it means"],
                 rows) +
        f"<p class='note'>Computed over {joint['n_games']} games carrying per-sim vectors.</p>"
    )


def _coverage_section(agg: dict) -> str:
    """Is the predicted spread the right size? Per player, per team total, and on the margin."""
    cov = agg.get("coverage") or {}
    if not cov.get("n_player_games"):
        return ""
    rows = []
    for scope, label in (("player", "Per player"), ("team", "Team total")):
        for stat, block in (cov.get(scope) or {}).items():
            rows.append([f"{label} {stat}", f"{block.get('1', 0.0) * 100:.1f}%",
                         f"{block.get('2', 0.0) * 100:.1f}%"])
    margin = cov.get("margin") or {}
    if margin.get("n"):
        rows.append(["Game margin", f"{margin.get('1', 0.0) * 100:.1f}%",
                     f"{margin.get('2', 0.0) * 100:.1f}%"])
    body = _table(["Scope", "within ±1 sd (ideal 68.3%)", "within ±2 sd (ideal 95.4%)"], rows)
    tail = ""
    if margin.get("resid_sd"):
        tail = (
            "<p class='note'>Margin: predicted sd "
            f"<b>{margin['pred_sd']:.2f}</b> against a realised residual sd of "
            f"<b>{margin['resid_sd']:.2f}</b> — a dispersion ratio of "
            f"<b>{margin['dispersion_ratio']:.2f}×</b>. This is the number a shrinkage dial would "
            "be fitted to; the rule is that it reaches 1.00 by fixing the structure instead "
            "(docs/v3_direction.md §3 W3).</p>"
        )
    return (
        "<h2>Distribution coverage — is the spread the right size?</h2>"
        "<p class='note'>How often the actual value landed inside k model standard deviations. "
        "Players and stats the sims are unanimous about carry no distribution to test and are "
        "excluded from the denominator rather than counted as hits.</p>"
        + body + tail +
        f"<p class='note'>{cov['n_player_games']} player-games, "
        f"{cov['n_team_games']} team-games.</p>"
    )

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


def _player_minutes_section(records: list[dict]) -> str:
    """How close the sims get a player's minutes — MAE, within-N, and the error distribution."""
    if not records:
        return ""
    c = minutes_closeness(records)
    if not c["n_player_games"]:
        return ""

    within = " · ".join(f"≤{w} min: {v * 100:.0f}%" for w, v in c["within"].items())
    summary = _table(["metric", "value"], [
        ["MAE", f"{c['mae']:.2f} min"],
        ["within-N hit rate", within],
        ["bias (pred − actual)", f"{c['bias']:+.2f} min"],
        ["player-games", c["n_player_games"]],
    ], cls="kv2")

    chart = _minutes_plots(records)
    img = (f"<div class='charts'><img alt='minutes' "
           f"src='data:image/png;base64,{chart}'/></div>") if chart else ""

    note = ("How close we get a player's minutes: <b>MAE is the average miss on one player-game</b>, "
            "and within-N is the share called inside N minutes. Bias sits near zero by construction "
            "— a team hands out ~240 player-minutes a game, so a minute given to the wrong player is "
            "taken from the right one — which is why MAE is the number to read.")
    return f"<h2>Player minutes accuracy</h2><p class='sub'>{note}</p>" + summary + img


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
            f"{m['player_minutes_mae']:.2f}" if "player_minutes_mae" in m else "—",
            f"{m['player_minutes_bias']:+.2f}" if "player_minutes_bias" in m else "—",
        ])
    table = _table(["seg", "games", "tuning change", "pace bias", "FGA bias", "eFG bias",
                    "vote pick", "vote Brier", "score pick", "score Brier",
                    "spread MAE", "PTS MAE", "MIN MAE", "MIN bias"], rows)
    chart = _progression_chart(prog)
    img = (f"<div class='charts'><img alt='progression' "
           f"src='data:image/png;base64,{chart}'/></div>") if chart else ""
    note = ("Holdout segmented by <b>distinct tuning</b> in evaluation order — each row is a stretch of "
            "games simmed under one set of dials, with what changed vs the prior segment. Tracks how "
            "the model evolved as you retuned, alongside the overall aggregate above. "
            "(One row only = the whole holdout ran under a single tuning, or per-game tuning wasn't "
            "recorded.)")
    return f"<h2>Tuning progression</h2><p class='sub'>{note}</p>{table}{img}"


# Stats worth reading a quarter at a time. Deliberately shorter than _STAT_LABELS: the
# per-quarter question is "does the model play end-game basketball?", which is pace, shot mix,
# free throws and scoring -- not a per-quarter steals column nobody reads.
_QUARTER_STATS = ("pts", "fga", "fgm", "tpa", "tpm", "fta", "ftm", "oreb", "dreb", "ast", "tov", "pf")

_PERIOD_LABELS = {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"}


def _period_label(period) -> str:
    p = int(period)
    return _PERIOD_LABELS.get(p, f"OT{p - 3}")


def quarter_rows(records: list[dict]) -> list[dict]:
    """Long-format (game, period, side, stat) -> predicted / actual, over every record.

    Shared by the HTML section and ``box_quarters.parquet`` so the page and the queryable frame
    cannot disagree. Records written before workstream 13 carry no quarter block and are skipped
    rather than faked -- an old run is partially readable, not silently wrong.
    """
    out = []
    for r in records:
        actual = r.get("quarter_actual") or {}
        pred = r.get("quarter_pred") or {}
        for period in sorted(set(actual) | set(pred), key=lambda x: int(x)):
            for side in ("home", "away"):
                a = (actual.get(period) or {}).get(side) or {}
                q = (pred.get(period) or {}).get(side) or {}
                for stat in _QUARTER_STATS:
                    if stat not in a and stat not in q:
                        continue
                    out.append({
                        "game_id": r["game_id"],
                        "period": int(period),
                        "period_label": _period_label(period),
                        "side": side,
                        "stat": stat,
                        "pred": float(q[stat]) if stat in q else float("nan"),
                        "actual": float(a[stat]) if stat in a else float("nan"),
                    })
    return out


def _box_quarters_frame(records: list[dict]) -> pd.DataFrame:
    rows = quarter_rows(records)
    if not rows:
        return pd.DataFrame(columns=["game_id", "period", "period_label", "side",
                                     "stat", "pred", "actual"])
    return pd.DataFrame(rows)


def _quarter_section(records: list[dict]) -> str:
    """Per-quarter team accuracy: what the model does early against what it does late.

    This is the only view in the report that can answer whether end-game basketball is modelled
    at all -- whether a trailing team fouls, whether threes get hunted late. It is also the
    evidence the clutch loss weighting was deferred to wait for rather than guessed at.
    """
    frame = _box_quarters_frame(records)
    if frame.empty:
        return ("<h2>Per-quarter team accuracy</h2>"
                "<p class='sub'>No quarter data in these records — they predate the per-quarter "
                "split. Re-run the eval to populate it.</p>")

    both = frame.dropna(subset=["pred", "actual"])
    if both.empty:
        return ("<h2>Per-quarter team accuracy</h2>"
                "<p class='sub'>These records carry the actual per-quarter box but no predicted "
                "one — they were built without the per-sim histories.</p>")

    rows = []
    for period, chunk in both.groupby("period", sort=True):
        cells = [_period_label(period)]
        for stat in ("pts", "fga", "tpa", "fta", "tov", "pf"):
            part = chunk[chunk["stat"] == stat]
            if part.empty:
                cells.append("—")
                continue
            pred, actual = part["pred"].mean(), part["actual"].mean()
            # _table escapes every cell, so this stays plain text rather than markup.
            cells.append(f"{pred:.1f} / {actual:.1f} ({pred - actual:+.1f})")
        rows.append(cells)

    table = _table(["period", "PTS", "FGA", "3PA", "FTA", "TO", "PF"], rows)
    note = ("Predicted / actual per team per period, averaged over the holdout, bias in "
            "brackets. A model that does not play end-game basketball shows it here: flat FTA and "
            "3PA into Q4 means no intentional fouling and no three-point hunting when trailing. "
            "Also written to box_quarters.parquet, one row per (game, period, side, stat).")
    return f"<h2>Per-quarter team accuracy</h2><p class='sub'>{_esc(note)}</p>" + table


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
        f"window k={_esc(report.get('window', 0))} · "
        f"{_esc(report['created_at'])} · commit {_esc(report['git_commit'] or '—')}</p>"
    )
    sections = [
        header,
        _cards(agg["headline"]),
        _win_section(agg, records),
        _spread_section(agg, records),
        _joint_section(agg),
        _coverage_section(agg),
        _progression_section(agg),
        _accuracy_section("Team box-score accuracy", agg["team_accuracy"],
                          agg["team_reliability"],
                          "Predicted (mean over sims) vs actual per team; 'sim std' is the average "
                          "per-game spread across sims. MIN is the team's total player-minutes "
                          "(~240 in regulation), so its error is really a check on simulated game "
                          "length / overtime."),
        _advanced_section(agg),
        _accuracy_section("Per-player box-score accuracy", agg["player_accuracy"],
                          agg["player_reliability"],
                          "Players matched by name across sims and the real game (absent = 0). MIN "
                          "is the rotation prediction — how many minutes the sims gave each player "
                          "vs how many they actually played."),
        _quarter_section(records),
        _player_minutes_section(records),
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
                # REPORT_STATS = the stored box fields plus derived "minutes", so the queryable
                # table carries pred/std/actual minutes next to the raw seconds.
                for f in REPORT_STATS:
                    row[f"pred_{f}"] = stat_value(avg, f)
                    row[f"std_{f}"] = stat_value(std, f)
                    row[f"actual_{f}"] = stat_value(actual, f)
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
    # Brier standard error, beside the Brier it qualifies (docs/v3_direction.md §8: two runs
    # within ±2 SE on the same games are the same model).
    if agg["win"].get("brier_se") is not None:
        rows.append({"scope": "win", "metric": "brier_se",
                     "predicted": agg["win"]["brier_se"],
                     "actual": None, "mae": None, "bias": None})
    if (agg.get("win_score") or {}).get("brier_se") is not None:
        rows.append({"scope": "win_score", "metric": "brier_se",
                     "predicted": agg["win_score"]["brier_se"],
                     "actual": None, "mae": None, "bias": None})
    # Joint structure: predicted = this run, actual = the real 2022-23 value, so the long
    # table reads the same way as every accuracy row above it.
    joint = agg.get("joint") or {}
    if joint.get("n_games"):
        for metric, real in (("corr_home_away", REAL_CORR_HOME_AWAY),
                             ("pace_sd", REAL_PACE_SD),
                             ("margin_sd", REAL_MARGIN_SD),
                             ("total_sd", REAL_TOTAL_SD),
                             ("side_pts_sd", REAL_SIDE_PTS_SD)):
            rows.append({"scope": "joint", "metric": metric, "predicted": joint[metric],
                         "actual": real, "mae": None, "bias": joint[metric] - real})
        rows.append({"scope": "joint", "metric": "corr_se", "predicted": joint["corr_se"],
                     "actual": None, "mae": None, "bias": None})
    cov = agg.get("coverage") or {}
    if cov.get("n_player_games"):
        for scope in ("player", "team"):
            for stat, block in (cov.get(scope) or {}).items():
                for k, rate in block.items():
                    rows.append({"scope": f"coverage_{scope}", "metric": f"{stat}_within_{k}sd",
                                 "predicted": rate,
                                 "actual": 0.683 if k == "1" else 0.954,
                                 "mae": None, "bias": rate - (0.683 if k == "1" else 0.954)})
        margin = cov.get("margin") or {}
        for metric in ("pred_sd", "resid_sd", "dispersion_ratio"):
            rows.append({"scope": "coverage_margin", "metric": metric,
                         "predicted": margin.get(metric),
                         "actual": 1.0 if metric == "dispersion_ratio" else None,
                         "mae": None, "bias": None})
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
        # The window this run scored. Pooling run_summary rows across runs is the whole point of
        # rotating windows, and without this column the pooled table cannot tell six distinct
        # 100-game windows from six re-runs of the same one.
        "window_k": report.get("window", 0),
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
        "player_minutes_mae": agg["headline"].get("player_minutes_mae"),
        "player_minutes_bias": _bias("player_accuracy", MINUTES),
        # Brier standard error: the cross-run table is exactly where two Briers get compared,
        # so the thing that says whether the difference is real belongs in the same row.
        "brier_se": agg["headline"].get("brier_se"),
        "score_brier_se": agg["headline"].get("score_brier_se"),
        # Joint structure (the W3 gate) and the margin dispersion it produces.
        "corr_home_away": (agg.get("joint") or {}).get("corr_home_away"),
        "corr_se": (agg.get("joint") or {}).get("corr_se"),
        "pace_sd": (agg.get("joint") or {}).get("pace_sd"),
        "margin_sd": (agg.get("joint") or {}).get("margin_sd"),
        "total_sd": (agg.get("joint") or {}).get("total_sd"),
        "margin_dispersion_ratio": ((agg.get("coverage") or {}).get("margin") or {})
                                   .get("dispersion_ratio"),
        "margin_resid_sd": ((agg.get("coverage") or {}).get("margin") or {}).get("resid_sd"),
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
    _box_quarters_frame(report["records"]).to_parquet(data_dir / "box_quarters.parquet",
                                                      index=False)
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
