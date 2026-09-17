"""
Three game-state behaviours, measured on the simulator and on reality, from play-by-play alone.

2.0 gave the model the inputs a coach reacts to -- each player's personal fouls, the score, the
clock, the period. Whether it learned to *behave* on them is a different question from whether the
loss went down, and it is not visible in any box-score error. These are the three behaviours that
are large, rule-driven and unambiguous, so if game state reaches the rollout at all they should be
there (docs/v3_direction.md SS3 W1):

  A. Foul trouble -> bench.   P(off the floor within 60 s of taking his Kth personal foul, K
                              before halftime). A coach sits a player in early foul trouble; the
                              event rate matters as much as the conditional, because a simulator
                              that never sits anyone also reaches a 4th first-half foul far too
                              often.
  B. Blowout -> starters out. Total Q4 seconds for the ten starters when the margin at the start
                              of Q4 is >= 20, against when it is not. Reported as a ratio, so it
                              does not depend on how long the sim's games are.
  C. Late and behind -> foul. The trailing team's foul rate in the last 2:00 of Q4 while down 4-9,
                              per 100 seconds spent in that state -- elapsed clock is the only
                              denominator that is well defined on both sides without a second
                              possession tally.

Everything here runs on the play-by-play CSVs already on disk: ``_PbpSink`` writes every sim's
rows as it completes, in the same column schema as the real game, so **one function scores both
sides and no re-simulation is needed**. State comes from the vetted scans -- ``GameStateScan`` for
score/period/clock, ``LineupScan`` for personal fouls, ``period_box_scores`` for Q4 minutes -- never
from a second tally written here.

    python -m reporting.state_probes results/version2/v2-run4
    python -m reporting.state_probes results/version2/v2-run4 --seasons 2023

Writes ``state_probes.{json,html,parquet}`` into the run directory, alongside the report. Read-only
and TF-free, so it is safe to run beside a live eval pool.

Two sizing notes that decide how the output is read:

- The real side is the **whole season**, never the holdout window. A 100-game window holds ~4 real
  fourth-fouls-before-half; the rate is only readable against a full season (SS8).
- The 4th-foul probe is reported with a **3rd-foul companion**, which has ~30x the events. The 4th
  is the doc's probe; the 3rd is the one that can actually move.
"""
from __future__ import annotations

import argparse
import ast
import glob
import html as _html
import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

from models.game_state_features import GameStateScan, period_index
from models.rotation_features import LineupScan
from simulation.box_score import period_box_scores, side_membership

# --- Probe A ----------------------------------------------------------------------------------
# The doc's probe is the 4th foul; the 3rd is the higher-n companion that makes the number legible
# at any realistic sample size.
FOUL_TROUBLE_KS = (3, 4)
# Halftime. Two 12:00 quarters -- a foul after this is ordinary, not trouble.
HALFTIME_SECONDS = 2 * 2 * 720 / 2
FOUL_TROUBLE_WITHIN = 60.0

# --- Probe B ----------------------------------------------------------------------------------
BLOWOUT_MARGIN = 20
Q4 = 3                      # period_box_scores keys 0-3 then OT

# --- Probe C ----------------------------------------------------------------------------------
LATE_SECONDS = 120.0        # last 2:00 of the period
LATE_DEFICIT = (4, 9)

PROBE_LABELS = {
    "foul_trouble_3": "3rd foul before half -> off within 60 s",
    "foul_trouble_4": "4th foul before half -> off within 60 s",
    "blowout_q4_ratio": "Q4 starter seconds, blowout / close",
    "late_foul_rate": "Trailing-team fouls per 100 s (last 2:00, down 4-9)",
}


def _roster(cell) -> list[str]:
    """One roster snapshot as a list of names, from a real list or a list literal."""
    if isinstance(cell, list):
        return [str(n) for n in cell]
    if isinstance(cell, str) and cell.startswith("["):
        try:
            return [str(n) for n in ast.literal_eval(cell)]
        except (ValueError, SyntaxError):
            return []
    return []


def _rows(frame: pd.DataFrame) -> list[dict]:
    return frame.to_dict("records")


# --------------------------------------------------------------------------- the three probes

def foul_trouble_events(rows: list[dict]) -> dict[int, list[bool]]:
    """Per K, one bool per qualifying foul: was he off the floor within the window?

    A forward pass. When a personal foul takes a player to exactly K before halftime, he becomes
    pending; he resolves the first row on which he is no longer among the ten on the floor. A
    player still pending at the final row never came off, which is a real outcome (False), not a
    dropped sample -- dropping it would score only the players who were benched.
    """
    scan = LineupScan()
    # Keyed by (player, K), not by player: a third and a fourth foul before half are two separate
    # observations of the same man, and one substitution resolves both. Keying by name alone would
    # silently drop whichever came second -- which is always the fourth, the rarer and more
    # interesting one.
    pending: dict[tuple[str, int], float] = {}
    out: dict[int, list[bool]] = {k: [] for k in FOUL_TROUBLE_KS}
    for row in rows:
        before = dict(scan.fouls)
        scan.step(row)
        t = float(row.get("time") or 0.0)
        on_court = set(_roster(row.get("roster_home"))) | set(_roster(row.get("roster_away")))

        # Resolve anyone pending who has left the floor.
        for key in [k for k in pending if k[0] not in on_court]:
            when = pending.pop(key)
            out[key[1]].append((t - when) <= FOUL_TROUBLE_WITHIN)

        # A foul that lands a player on exactly K, before halftime, while he is on the floor.
        for name, count in scan.fouls.items():
            if count == before.get(name, 0):
                continue                      # he did not foul on this row
            if count in FOUL_TROUBLE_KS and t < HALFTIME_SECONDS and name in on_court:
                pending.setdefault((name, count), t)
    for (_name, k) in pending:
        out[k].append(False)                  # never came off
    return out


def q4_starter_seconds(rows: list[dict]) -> tuple[float, int] | None:
    """``(total Q4 seconds for the ten starters, margin at the start of Q4)``, or None.

    Starters are the five per side on the first row that shows a full lineup: the cleaner repairs
    the snapshot, and both real and simulated rows carry it, so this is the same definition on both
    sides. Q4 minutes come from ``period_box_scores``, whose per-period seconds are verified to sum
    to the whole-game box -- the alternative would be a second minutes accounting.
    """
    starters: set[str] = set()
    for row in rows:
        home, away = _roster(row.get("roster_home")), _roster(row.get("roster_away"))
        if len(home) == 5 and len(away) == 5:
            starters = set(home) | set(away)
            break
    if not starters:
        return None

    # Margin as Q4 begins: the score as of the last row before the period turns over.
    scan = GameStateScan()
    margin_at_q4 = None
    prev_diff = 0
    for row in rows:
        if period_index(float(row.get("time") or 0.0)) >= Q4 and margin_at_q4 is None:
            margin_at_q4 = prev_diff
        prev_diff = scan.step(row)[0]
    if margin_at_q4 is None:
        return None                            # game never reached the fourth quarter

    try:
        boxes = period_box_scores(rows)
    except ValueError:
        return None                            # out-of-order times; say nothing rather than guess
    box = boxes.get(Q4)
    if box is None:
        return None
    seconds = sum(line.seconds for line in (*box.home, *box.away) if line.player in starters)
    return float(seconds), int(margin_at_q4)


def late_foul_state(rows: list[dict]) -> tuple[int, float]:
    """``(fouls by the trailing team, seconds spent in the state)`` for one game.

    The state is: fourth period or later, at most 2:00 left, and the score within 4-9 points. Time
    is credited over the interval between consecutive rows, to the state that held at the start of
    it -- the same interval convention ``LineupScan`` and ``generate_box_score`` use for minutes.
    """
    home_names, away_names = side_membership(rows)
    scan = GameStateScan()
    fouls = 0
    seconds = 0.0
    prev_t = None
    prev_in_state = False
    prev_trailing = None
    for row in rows:
        t = float(row.get("time") or 0.0)
        if prev_t is not None and prev_in_state and t > prev_t:
            seconds += t - prev_t
        diff, _total, period, time_left, *_ = scan.step(row)

        if prev_in_state and str(row.get("event") or "").strip().lower() == "foul":
            fouler = str(row.get("player") or "").strip()
            side = "home" if fouler in home_names else "away" if fouler in away_names else None
            if side is not None and side == prev_trailing:
                fouls += 1

        lo, hi = LATE_DEFICIT
        in_state = (period >= Q4 and time_left <= LATE_SECONDS and lo <= abs(diff) <= hi)
        prev_in_state = in_state
        prev_trailing = ("away" if diff > 0 else "home") if in_state else None
        prev_t = t
    return fouls, seconds


def probe_game(rows: list[dict]) -> dict:
    """All three probes over one game's rows."""
    out: dict = {"foul_trouble": foul_trouble_events(rows)}
    q4 = q4_starter_seconds(rows)
    out["q4"] = None if q4 is None else {"seconds": q4[0], "margin": q4[1]}
    fouls, seconds = late_foul_state(rows)
    out["late"] = {"fouls": fouls, "seconds": seconds}
    return out


# --------------------------------------------------------------------------- pooling

def _blank() -> dict:
    return {
        "n_games": 0,
        "foul_events": {k: 0 for k in FOUL_TROUBLE_KS},
        "foul_benched": {k: 0 for k in FOUL_TROUBLE_KS},
        "q4_blowout_seconds": 0.0, "q4_blowout_games": 0,
        "q4_close_seconds": 0.0, "q4_close_games": 0,
        "late_fouls": 0, "late_seconds": 0.0,
    }


def accumulate(pool: dict, game: dict) -> dict:
    pool["n_games"] += 1
    for k, results in game["foul_trouble"].items():
        pool["foul_events"][k] += len(results)
        pool["foul_benched"][k] += sum(1 for hit in results if hit)
    if game["q4"] is not None:
        blowout = abs(game["q4"]["margin"]) >= BLOWOUT_MARGIN
        key = "blowout" if blowout else "close"
        pool[f"q4_{key}_seconds"] += game["q4"]["seconds"]
        pool[f"q4_{key}_games"] += 1
    pool["late_fouls"] += game["late"]["fouls"]
    pool["late_seconds"] += game["late"]["seconds"]
    return pool


def summarize(pool: dict) -> dict:
    """Pool counts -> the rates the report reads."""
    n = max(pool["n_games"], 1)
    out: dict = {"n_games": pool["n_games"]}
    for k in FOUL_TROUBLE_KS:
        events = pool["foul_events"][k]
        out[f"foul_trouble_{k}"] = {
            "n_events": events,
            "events_per_game": events / n,
            "p_benched": (pool["foul_benched"][k] / events) if events else None,
        }
    blow_g, close_g = pool["q4_blowout_games"], pool["q4_close_games"]
    blow = pool["q4_blowout_seconds"] / blow_g if blow_g else None
    close = pool["q4_close_seconds"] / close_g if close_g else None
    out["blowout_q4"] = {
        "blowout_games": blow_g, "close_games": close_g,
        "blowout_frequency": blow_g / max(blow_g + close_g, 1),
        "starter_seconds_blowout": blow, "starter_seconds_close": close,
        "ratio": (blow / close) if (blow and close) else None,
    }
    out["late_foul"] = {
        "fouls": pool["late_fouls"],
        "state_seconds": pool["late_seconds"],
        "state_seconds_per_game": pool["late_seconds"] / n,
        "rate_per_100s": (100.0 * pool["late_fouls"] / pool["late_seconds"])
                         if pool["late_seconds"] else None,
    }
    return out


# --------------------------------------------------------------------------- drivers

def probe_sims(run_dir: str | Path, *, limit_games: int | None = None, echo=print) -> dict:
    """Pool the three probes over every sim play-by-play in a finished run."""
    run_dir = Path(run_dir)
    game_dirs = sorted((run_dir / "games").glob("*"))
    if limit_games:
        game_dirs = game_dirs[:limit_games]
    pool = _blank()
    n_sims = 0
    for i, game_dir in enumerate(game_dirs, 1):
        for path in sorted(game_dir.glob("playbyplay/sim_*_playbyplay.csv")):
            accumulate(pool, probe_game(_rows(pd.read_csv(path))))
            n_sims += 1
        if echo and i % 25 == 0:
            echo(f"    {i}/{len(game_dirs)} game folders, {n_sims} sims")
    pool["n_sims"] = n_sims
    return pool


def probe_real(data_dir: str = "./data", *, seasons=("2023",), echo=print) -> dict:
    """Pool the same probes over the real cleaned seasons."""
    pool = _blank()
    for season in seasons:
        path = Path(data_dir) / f"season{season}.csv"
        if not path.exists():
            raise FileNotFoundError(f"no cleaned season at {path}")
        frame = pd.read_csv(path)
        for gid, game in frame.groupby("game_id", sort=False):
            accumulate(pool, probe_game(_rows(game)))
        if echo:
            echo(f"    season {season}: {pool['n_games']} games so far")
    return pool


def compare(run_dir: str | Path, *, data_dir: str = "./data", seasons=("2023",),
            limit_games: int | None = None, echo=print) -> dict:
    """Run both sides and pair them up."""
    if echo:
        echo("  sims:")
    sim = summarize(probe_sims(run_dir, limit_games=limit_games, echo=echo))
    if echo:
        echo("  real:")
    real = summarize(probe_real(data_dir, seasons=seasons, echo=echo))
    return {"run": str(run_dir), "seasons": list(seasons), "sim": sim, "real": real}


# --------------------------------------------------------------------------- output

def _rows_for_frame(report: dict) -> list[dict]:
    sim, real = report["sim"], report["real"]
    rows = []
    for k in FOUL_TROUBLE_KS:
        s, r = sim[f"foul_trouble_{k}"], real[f"foul_trouble_{k}"]
        rows.append({"probe": f"foul_trouble_{k}", "metric": "p_benched_within_60s",
                     "sim": s["p_benched"], "real": r["p_benched"],
                     "sim_n": s["n_events"], "real_n": r["n_events"]})
        rows.append({"probe": f"foul_trouble_{k}", "metric": "events_per_game",
                     "sim": s["events_per_game"], "real": r["events_per_game"],
                     "sim_n": s["n_events"], "real_n": r["n_events"]})
    s, r = sim["blowout_q4"], real["blowout_q4"]
    for metric in ("starter_seconds_blowout", "starter_seconds_close", "ratio",
                   "blowout_frequency"):
        rows.append({"probe": "blowout_q4", "metric": metric, "sim": s[metric], "real": r[metric],
                     "sim_n": s["blowout_games"] + s["close_games"],
                     "real_n": r["blowout_games"] + r["close_games"]})
    s, r = sim["late_foul"], real["late_foul"]
    for metric in ("rate_per_100s", "state_seconds_per_game"):
        rows.append({"probe": "late_foul", "metric": metric, "sim": s[metric], "real": r[metric],
                     "sim_n": s["fouls"], "real_n": r["fouls"]})
    return rows


def to_frame(report: dict) -> pd.DataFrame:
    frame = pd.DataFrame(_rows_for_frame(report))
    frame["delta"] = frame["sim"] - frame["real"]
    return frame


def render_html(report: dict) -> str:
    def esc(v):
        return _html.escape("" if v is None else str(v))

    def num(v, spec=".3f"):
        return "—" if v is None else format(v, spec)

    body = []
    for row in _rows_for_frame(report):
        body.append(
            "<tr>"
            f"<td>{esc(PROBE_LABELS.get(row['probe'], row['probe']))}</td>"
            f"<td>{esc(row['metric'])}</td>"
            f"<td>{num(row['sim'])}</td><td>{num(row['real'])}</td>"
            f"<td>{esc(row['sim_n'])}</td><td>{esc(row['real_n'])}</td>"
            "</tr>"
        )
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<title>game-state probes</title><style>"
        "body{font:14px/1.5 system-ui,sans-serif;margin:2rem;max-width:60rem}"
        "table{border-collapse:collapse;width:100%}"
        "td,th{border-bottom:1px solid #ddd;padding:.4rem .6rem;text-align:left}"
        "th{background:#f5f5f5}.note{color:#555}</style></head><body>"
        "<h1>Game-state behaviour probes</h1>"
        f"<p class='note'>Run <code>{esc(report['run'])}</code> against real seasons "
        f"{esc(', '.join(report['seasons']))}. Sim rates come from the run's per-sim "
        "play-by-plays; the real side is the whole season, never the holdout window.</p>"
        "<table><tr><th>Probe</th><th>Metric</th><th>Sim</th><th>Real</th>"
        "<th>sim n</th><th>real n</th></tr>" + "".join(body) + "</table>"
        "</body></html>"
    )


def write(report: dict, run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    (run_dir / "state_probes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (run_dir / "state_probes.html").write_text(render_html(report), encoding="utf-8")
    to_frame(report).to_parquet(run_dir / "state_probes.parquet", index=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seasons", default="2023",
                    help="comma-separated cleaned seasons for the real side")
    ap.add_argument("--limit-games", type=int, default=None,
                    help="only the first N game folders (a quick check, not a result)")
    args = ap.parse_args(argv)

    report = compare(args.run_dir, data_dir=args.data_dir,
                     seasons=tuple(s.strip() for s in args.seasons.split(",") if s.strip()),
                     limit_games=args.limit_games)
    write(report, args.run_dir)
    frame = to_frame(report)
    print(frame.to_string(index=False))
    print(f"\nwrote state_probes.{{json,html,parquet}} -> {args.run_dir}")


if __name__ == "__main__":
    main()
