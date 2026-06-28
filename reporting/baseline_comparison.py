"""
baseline_comparison.py — does the model beat "predict the player's season average"?

The eval report shows the model's per-player box-score MAE, but a low MAE only means something if it
beats the dumb baseline: **predict each player's season-to-date average every game**. NBA box scores
are heavily mean-reverting, so that baseline is strong; if the model doesn't clearly beat it, the
"we predict specific-game box scores well" pitch is really just restating mean reversion.

This is a **read-only, zero-simulation** check. It reuses the per-game records already persisted in a
run's ``report.json`` (the model's 11-sim ``player_avg`` and the ``player_actual`` lines, matched by
name — apples-to-apples with the headline report) and builds the baseline from the cleaned data:

  * **season-to-date (STD)** — the player's mean line over their games *earlier in the same season*
    (the fair "what you'd actually know" baseline; falls back to the full-season mean when the player
    has fewer than ``--min-prior-games`` prior games).
  * **full-season** — the player's mean over *all* their games that season (an optimistic upper bound;
    it peeks at future games, so beating this is the harder bar).

For every real player-game it pairs three predictions of the actual line — model / STD / full-season —
and reports MAE per stat plus the paired win-rate (how often the model's error beats the baseline's),
for two cohorts: **all matched players** and **rotation only** (actual minutes ≥ ``--rotation-min``).

    python -m reporting.baseline_comparison reports/evaluation/<run_id>
"""
from __future__ import annotations

import argparse
import html as _html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_loading import load_all_cleaned
from simulation.box_score import generate_box_score
from simulation.stats import BOX_STATS, player_stats
from simulation.eval_metrics import _stat_errors
from training.chronology import game_index

# Stats we compare (the report's counting stats; seconds is used only for the rotation filter).
COMPARE_STATS = ("pts", "fga", "fgm", "tpm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf")
_STAT_LABELS = {
    "pts": "PTS", "fga": "FGA", "fgm": "FGM", "tpm": "3PM", "fta": "FTA", "oreb": "OREB",
    "dreb": "DREB", "ast": "AST", "stl": "STL", "blk": "BLK", "tov": "TO", "pf": "PF",
}
_ZERO = {f: 0.0 for f in BOX_STATS}


def _season_player_games(df: pd.DataFrame, meta: pd.DataFrame,
                         seasons: set[int]) -> dict[str, list[tuple[int, dict]]]:
    """For each player, the list of ``(pos, stat-dict)`` over every game in ``seasons``.

    ``pos`` is the chronological rank from ``game_index`` so season-to-date slices are exact (and
    same-date ties are ordered deterministically). Box-scores both sides of each game.
    """
    pos_by_id = dict(zip(meta["game_id"].astype(int), meta["pos"].astype(int)))
    ids = meta.loc[meta["season"].astype(int).isin(seasons), "game_id"].astype(int).tolist()
    out: dict[str, list[tuple[int, dict]]] = {}
    for gid in ids:
        game = df[df["game_id"] == gid]
        if game.empty:
            continue
        box = generate_box_score(game.sort_values("time"))
        pos = pos_by_id[gid]
        for line in (*box.home, *box.away):
            out.setdefault(line.player, []).append((pos, player_stats(line)))
    return out


def _mean_line(lines: list[dict]) -> dict:
    """Mean of a list of stat-dicts (empty -> zeros)."""
    if not lines:
        return dict(_ZERO)
    return {f: float(np.mean([ln[f] for ln in lines])) for f in BOX_STATS}


def _baselines_for(player: str, pos: int, hist: dict[str, list[tuple[int, dict]]],
                   min_prior: int) -> tuple[dict, dict]:
    """(season-to-date, full-season) baseline lines for ``player`` at chronological ``pos``."""
    games = hist.get(player, [])
    prior = [s for p, s in games if p < pos]
    full = [s for _, s in games]
    std = _mean_line(prior) if len(prior) >= min_prior else _mean_line(full)
    return std, _mean_line(full)


def compare(run_dir: str | Path, *, data_dir: str = "./data", min_prior: int = 3,
            rotation_min: float = 15.0) -> dict:
    """Build the model-vs-baseline comparison for a run. Returns the report dict."""
    run_dir = Path(run_dir)
    report_path = run_dir / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"No report.json at {report_path}.")
    records = json.loads(report_path.read_text(encoding="utf-8")).get("records") or []
    if not records:
        raise ValueError(f"{report_path} has no per-game records.")

    meta = game_index(data_dir)
    season_by_id = dict(zip(meta["game_id"].astype(int), meta["season"].astype(int)))
    pos_by_id = dict(zip(meta["game_id"].astype(int), meta["pos"].astype(int)))
    seasons = {season_by_id[int(r["game_id"])] for r in records if int(r["game_id"]) in season_by_id}

    df = load_all_cleaned(data_dir)
    hist = _season_player_games(df, meta, seasons)

    rotation_seconds = rotation_min * 60.0
    # Paired predictions of each actual line, per cohort: list of (pred, actual) per stat per predictor.
    cohorts = ("all", "rotation")
    predictors = ("model", "std", "full")
    pairs = {c: {p: {f: [] for f in COMPARE_STATS} for p in predictors} for c in cohorts}
    wins = {c: {f: [] for f in COMPARE_STATS} for c in cohorts}  # 1 model better, 0.5 tie, 0 worse
    n_player_games = {c: 0 for c in cohorts}

    for r in records:
        gid = int(r["game_id"])
        pos = pos_by_id.get(gid, 1 << 30)
        for side in ("home", "away"):
            actual = r["player_actual"][side]
            model_side = r["player_avg"][side]
            for name, act in actual.items():
                std, full = _baselines_for(name, pos, hist, min_prior)
                model = model_side.get(name, _ZERO)
                in_rotation = float(act.get("seconds", 0.0)) >= rotation_seconds
                row_cohorts = ("all", "rotation") if in_rotation else ("all",)
                for c in row_cohorts:
                    n_player_games[c] += 1
                    for f in COMPARE_STATS:
                        a = float(act.get(f, 0.0))
                        pairs[c]["model"][f].append((float(model.get(f, 0.0)), a))
                        pairs[c]["std"][f].append((float(std[f]), a))
                        pairs[c]["full"][f].append((float(full[f]), a))
                        me = abs(float(model.get(f, 0.0)) - a)
                        be = abs(float(std[f]) - a)
                        wins[c][f].append(1.0 if me < be else 0.5 if me == be else 0.0)

    table = {c: {f: {p: _stat_errors(pairs[c][p][f])["mae"] for p in predictors}
                 for f in COMPARE_STATS} for c in cohorts}
    winrate = {c: {f: (float(np.mean(wins[c][f])) if wins[c][f] else 0.0) for f in COMPARE_STATS}
               for c in cohorts}

    report = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_games": len(records),
        "seasons": sorted(seasons),
        "min_prior_games": min_prior,
        "rotation_min": rotation_min,
        "n_player_games": n_player_games,
        "mae": table,            # mae[cohort][stat][predictor]
        "model_winrate_vs_std": winrate,
        "_predictors": list(predictors),
        "_cohorts": list(cohorts),
    }
    return report


# --------------------------------------------------------------------------- render

def _verdict(report: dict) -> str:
    """One-line read on the rotation cohort's points: does the model beat season-to-date?"""
    rot = report["mae"]["rotation"]
    m, s = rot["pts"]["model"], rot["pts"]["std"]
    wr = report["model_winrate_vs_std"]["rotation"]["pts"] * 100
    if s <= 0:
        return "No rotation player-games to judge."
    rel = (m - s) / s * 100
    if m < s:
        return (f"Rotation PTS: model MAE {m:.2f} beats season-to-date {s:.2f} "
                f"({rel:+.1f}%), winning {wr:.0f}% of player-games. The pitch holds on points.")
    return (f"Rotation PTS: model MAE {m:.2f} vs season-to-date {s:.2f} ({rel:+.1f}%), "
            f"winning only {wr:.0f}% of player-games - not clearly beating mean reversion.")


def _esc(v) -> str:
    return _html.escape(str(v))


def _html_table(report: dict, cohort: str) -> str:
    mae, wr = report["mae"][cohort], report["model_winrate_vs_std"][cohort]
    head = "".join(f"<th>{h}</th>" for h in
                   ("stat", "model MAE", "STD MAE", "full-season MAE", "model−STD", "model win% vs STD"))
    rows = []
    for f in COMPARE_STATS:
        m, s, fu = mae[f]["model"], mae[f]["std"], mae[f]["full"]
        delta = m - s
        cls = "good" if delta < 0 else "bad"
        rows.append(
            f"<tr><td>{_STAT_LABELS[f]}</td><td>{m:.2f}</td><td>{s:.2f}</td><td>{fu:.2f}</td>"
            f"<td class='{cls}'>{delta:+.2f}</td><td>{wr[f] * 100:.0f}%</td></tr>")
    n = report["n_player_games"][cohort]
    return (f"<h2>{cohort.capitalize()} players "
            f"<span class='sub'>({n} player-games)</span></h2>"
            f"<table><tr>{head}</tr>{''.join(rows)}</table>")


_STYLE = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1f2329;
 margin:0;padding:32px;max-width:900px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:26px 0 8px;border-bottom:2px solid #e5e7eb;padding-bottom:6px}
.sub{color:#6b7280;font-size:13px;font-weight:400}
.verdict{font-size:15px;padding:12px 16px;border:1px solid #e5e7eb;border-radius:10px;margin:14px 0;background:#f9fafb}
table{border-collapse:collapse;font-size:13px;margin:6px 0}
th,td{border:1px solid #e5e7eb;padding:5px 10px;text-align:right}
th{background:#f9fafb;text-align:center}td:first-child,th:first-child{text-align:left}
td.good{color:#15803d;font-weight:600}td.bad{color:#b91c1c;font-weight:600}
footer{margin-top:36px;color:#6b7280;font-size:12px}
"""


def render_html(report: dict) -> str:
    body = [
        f"<h1>Box-score baseline comparison</h1>",
        f"<p class='sub'>Run <code>{_esc(Path(report['run_dir']).name)}</code> · "
        f"{report['n_games']} games · seasons {report['seasons']} · "
        f"baseline = season-to-date avg (min {report['min_prior_games']} prior games) · "
        f"rotation ≥ {report['rotation_min']:.0f} min · {_esc(report['created_at'])}</p>",
        f"<div class='verdict'>{_esc(_verdict(report))}</div>",
        "<p class='sub'>Lower MAE is better. <b>model−STD</b> negative (green) = model beats the "
        "season-to-date baseline on that stat. The rotation cohort is the honest test (the all-players "
        "pool is padded with bench/zero lines both predictors trivially nail).</p>",
        _html_table(report, "rotation"),
        _html_table(report, "all"),
        "<footer>Generated by CourtVisionIQ baseline_comparison — read-only, no simulation.</footer>",
    ]
    return ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>baseline comparison — {_esc(Path(report['run_dir']).name)}</title>"
            f"<style>{_STYLE}</style></head><body>{''.join(body)}</body></html>")


def _to_frame(report: dict) -> pd.DataFrame:
    rows = []
    for c in report["_cohorts"]:
        for f in COMPARE_STATS:
            mae = report["mae"][c][f]
            rows.append({
                "cohort": c, "stat": f,
                "model_mae": mae["model"], "std_mae": mae["std"], "full_mae": mae["full"],
                "model_minus_std": mae["model"] - mae["std"],
                "model_winrate_vs_std": report["model_winrate_vs_std"][c][f],
                "n_player_games": report["n_player_games"][c],
            })
    return pd.DataFrame(rows)


def _print_summary(report: dict) -> None:
    print(f"\n=== baseline comparison ({report['n_games']} games, seasons {report['seasons']}) ===")
    for c in report["_cohorts"]:
        print(f"\n-- {c} players ({report['n_player_games'][c]} player-games) --")
        print(f"  {'stat':5} {'model':>7} {'STD':>7} {'full':>7} {'d(m-STD)':>9} {'win%':>6}")
        for f in COMPARE_STATS:
            mae = report["mae"][c][f]
            wr = report["model_winrate_vs_std"][c][f] * 100
            print(f"  {_STAT_LABELS[f]:5} {mae['model']:7.2f} {mae['std']:7.2f} {mae['full']:7.2f} "
                  f"{mae['model'] - mae['std']:+9.2f} {wr:6.0f}")
    print(f"\n  {_verdict(report)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Model vs season-average box-score baseline (no sims).")
    ap.add_argument("run_dir", help="Eval run dir (the folder containing report.json).")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--min-prior-games", type=int, default=3,
                    help="Min prior in-season games before season-to-date is used (else full-season).")
    ap.add_argument("--rotation-min", type=float, default=15.0,
                    help="Actual minutes threshold for the rotation cohort.")
    args = ap.parse_args()

    report = compare(args.run_dir, data_dir=args.data_dir, min_prior=args.min_prior_games,
                     rotation_min=args.rotation_min)
    run_dir = Path(args.run_dir)
    (run_dir / "baseline_comparison.html").write_text(render_html(report), encoding="utf-8")
    (run_dir / "baseline_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _to_frame(report).to_parquet(run_dir / "baseline_comparison.parquet", index=False)
    _print_summary(report)
    print(f"\nReport -> {(run_dir / 'baseline_comparison.html').resolve()}")


if __name__ == "__main__":
    main()
