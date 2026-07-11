"""
game_report.py — a single game's HTML box-score report (predicted / actual / variance).

For each evaluated holdout game the eval harness writes a ``game.html`` next to its CSV box scores
and play-by-plays. It renders three full box scores from the game's evaluation record (the dict
``simulation.evaluation.build_game_record`` produces):

  1. Predicted — per-player mean over the sims (the aggregated prediction).
  2. Actual    — the real box score, raw.
  3. Variance  — predicted mean − actual (signed), so misses are visible at a glance.

Self-contained and TensorFlow-free (mirrors reporting/eval_report.py) so it can be unit-tested
without trained models.
"""
from __future__ import annotations

import html as _html

# Display columns -> BOX_STATS key. "seconds" is rendered as minutes (÷60).
_COLS: list[tuple[str, str]] = [
    ("MIN", "seconds"), ("PTS", "pts"), ("FGM", "fgm"), ("FGA", "fga"),
    ("3PM", "tpm"), ("3PA", "tpa"), ("FTM", "ftm"), ("FTA", "fta"),
    ("OREB", "oreb"), ("DREB", "dreb"), ("AST", "ast"), ("STL", "stl"),
    ("BLK", "blk"), ("TO", "tov"), ("PF", "pf"),
]

_STYLE = """
:root { --fg:#1f2329; --muted:#6b7280; --line:#e5e7eb; --accent:#4C78A8; --neg:#b4232a; --pos:#1a7f37; }
* { box-sizing:border-box; }
body { font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:var(--fg);
       margin:24px; line-height:1.4; }
h1 { font-size:22px; margin:0 0 2px; } h2 { font-size:17px; margin:26px 0 6px; }
h3 { font-size:14px; margin:14px 0 4px; color:var(--muted); }
p.sub { color:var(--muted); margin:0 0 12px; font-size:13px; }
.scoreline { font-size:15px; margin:6px 0 4px; }
.scoreline b { font-size:17px; }
.tablewrap { overflow-x:auto; }
table { border-collapse:collapse; font-size:12.5px; margin:2px 0 10px; min-width:640px; }
th, td { padding:3px 8px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
thead th { color:var(--muted); font-weight:600; border-bottom:2px solid var(--line); }
tr.team td { font-weight:700; border-top:2px solid var(--line); border-bottom:none; }
td.neg { color:var(--neg); } td.pos { color:var(--pos); }
footer { margin-top:32px; color:var(--muted); font-size:12px; }
"""


def _esc(x) -> str:
    return _html.escape(str(x))


def _val(stats: dict, key: str) -> float:
    x = float(stats.get(key, 0.0) or 0.0)
    return x / 60.0 if key == "seconds" else x


def _row(label: str, stats: dict, *, fmt, team: bool = False, signed: bool = False) -> str:
    cells = [f"<td>{_esc(label)}</td>"]
    for _, key in _COLS:
        v = _val(stats, key)
        cls = ""
        if signed:
            cls = " class='neg'" if v < -0.05 else (" class='pos'" if v > 0.05 else "")
        cells.append(f"<td{cls}>{fmt(v)}</td>")
    tr = " class='team'" if team else ""
    return f"<tr{tr}>{''.join(cells)}</tr>"


def _table(side_label: str, player_rows: list[str], team_row: str) -> str:
    head = "".join(f"<th>{_esc(lbl)}</th>" for lbl, _ in _COLS)
    body = "".join(player_rows) + team_row
    return (f"<h3>{_esc(side_label)}</h3><div class='tablewrap'><table>"
            f"<thead><tr><th>Player</th>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def _box_section(title: str, sub: str, record: dict, home_team: str, away_team: str,
                 *, cell: str) -> str:
    """One box score (predicted / actual / variance) with home + away tables.

    ``cell`` selects the per-player value source: "pred" (mean), "actual" (raw), or
    "variance" (mean − actual).
    """
    fmt_mean = (lambda v: f"{v:.1f}")
    fmt_int = (lambda v: f"{v:.0f}")
    fmt_signed = (lambda v: f"{v:+.1f}")
    out = [f"<h2>{_esc(title)}</h2><p class='sub'>{_esc(sub)}</p>"]
    for side, team in (("home", home_team), ("away", away_team)):
        avg = record["player_avg"][side]
        actual = record["player_actual"][side]
        names = sorted(record["players"][side], key=lambda n: avg.get(n, {}).get("pts", 0.0),
                       reverse=True)
        rows = []
        for name in names:
            a = actual.get(name, {})
            m = avg.get(name, {})
            if cell == "pred":
                rows.append(_row(name, m, fmt=fmt_mean))
            elif cell == "actual":
                rows.append(_row(name, a, fmt=fmt_int))
            else:  # variance = mean - actual
                diff = {k: _val(m, k) - _val(a, k) for _, k in _COLS}
                # _val already converted seconds->min; store back under the raw keys the row reads.
                diff_stats = {k: (diff[k] * 60.0 if k == "seconds" else diff[k]) for _, k in _COLS}
                rows.append(_row(name, diff_stats, fmt=fmt_signed, signed=True))
        # Team totals row.
        if cell == "pred":
            team_row = _row(f"{team} (TEAM)", record["team_pred"][side], fmt=fmt_mean, team=True)
        elif cell == "actual":
            team_row = _row(f"{team} (TEAM)", record["team_actual"][side], fmt=fmt_int, team=True)
        else:
            tp, ta = record["team_pred"][side], record["team_actual"][side]
            tdiff = {k: (_val(tp, k) - _val(ta, k)) for _, k in _COLS}
            tdiff_stats = {k: (tdiff[k] * 60.0 if k == "seconds" else tdiff[k]) for _, k in _COLS}
            team_row = _row(f"{team} (TEAM)", tdiff_stats, fmt=fmt_signed, team=True, signed=True)
        out.append(_table(f"{team} ({side})", rows, team_row))
    return "".join(out)


def render_game_html(record: dict, *, home_team: str = "HOME", away_team: str = "AWAY") -> str:
    """Render one game's predicted / actual / variance box scores into a self-contained HTML doc."""
    gid = record.get("game_id")
    ph, pa = record.get("pred_home_score", 0.0), record.get("pred_away_score", 0.0)
    ah, aa = record.get("actual_home_score", 0), record.get("actual_away_score", 0)
    wp = record.get("win_prob_home")
    n_sims = record.get("n_sims")

    scoreline = (
        f"<p class='scoreline'>Predicted <b>{_esc(away_team)} {pa:.0f} @ {_esc(home_team)} "
        f"{ph:.0f}</b> &nbsp;·&nbsp; Actual <b>{_esc(away_team)} {aa:.0f} @ {_esc(home_team)} "
        f"{ah:.0f}</b></p>"
    )
    meta = []
    if wp is not None:
        meta.append(f"home win prob {wp:.0%}")
    if n_sims is not None:
        meta.append(f"{_esc(n_sims)} sims")
    sub = " · ".join(meta)

    sections = [
        f"<h1>Game {_esc(gid)} — {_esc(away_team)} @ {_esc(home_team)}</h1>",
        f"<p class='sub'>{sub}</p>" if sub else "",
        scoreline,
        _box_section("Predicted box score", "Per-player mean over the sims (aggregated prediction).",
                     record, home_team, away_team, cell="pred"),
        _box_section("Actual box score", "The real game, raw.",
                     record, home_team, away_team, cell="actual"),
        _box_section("Variance (predicted − actual)",
                     "Predicted mean minus actual; red = under, green = over.",
                     record, home_team, away_team, cell="variance"),
        "<footer>Generated by CourtVisionIQ evaluation harness.</footer>",
    ]
    return ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>game {_esc(gid)} — {_esc(away_team)} @ {_esc(home_team)}</title>"
            f"<style>{_STYLE}</style></head><body>{''.join(sections)}</body></html>")


__all__ = ["render_game_html"]
