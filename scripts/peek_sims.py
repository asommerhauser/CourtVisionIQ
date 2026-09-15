"""
peek_sims.py -- read a live eval run's finished sims, before any record.json exists.

``record.json`` is written once a whole CHUNK of games finishes (simulation/stage_eval.py), and
every report path keys off it, so a run an hour into a six-game chunk has no readable results even
though hundreds of sims are already on disk. Their play-by-play is, though: ``_PbpSink`` writes
each sim's CSV the moment that sim completes. This rebuilds box scores from those CSVs and
compares them to the real game sitting in the same folder.

Read-only, pandas + stdlib only, no TensorFlow: safe to run beside a live pool.

    python scripts/peek_sims.py --run results/version2/v2-run1

It is a sanity check, not a report. The sims it can see are the ones that finished first, which is
a biased sample (short games complete sooner), so treat the numbers as "is this sane" rather than
as the run's result. The real report supersedes it.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


def _load_box_score():
    """Load simulation/box_score.py WITHOUT importing the simulation package.

    ``simulation/__init__.py`` imports GameSimulator, which imports TensorFlow -- so a plain
    ``from simulation.box_score import ...`` would load TF and take a CUDA context's worth of VRAM
    away from the very pool this script is meant to watch (the rule eval_pool.py and harvest.py
    already follow). box_score itself needs only pandas, zones and models.game_state_features, all
    of which are TF-free, so loading the file directly is enough.
    """
    path = os.path.join(_ROOT, "simulation", "box_score.py")
    spec = importlib.util.spec_from_file_location("_cviq_box_score", path)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: @dataclass resolves its class's __module__ through sys.modules, and
    # an unregistered module makes that lookup return None.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.generate_box_score


generate_box_score = _load_box_score()

ROTATION_RANKS = 12
ROTATION_MIN = 10.0          # "in the rotation" threshold, matching workstream 11a's gate


def _label(df: pd.DataFrame, col: str, fallback: str) -> str:
    """One team label from a cleaned frame, or ``fallback``.

    Some cleaned seasons carry an empty team column, and a bare ``str()`` on that turns NaN into
    the literal string "nan" -- which then shows up as a team name in the table.
    """
    if col not in df.columns or not len(df):
        return fallback
    val = df[col].dropna()
    return str(val.iloc[0]) if len(val) else fallback


def _teams(df: pd.DataFrame) -> tuple[str, str]:
    """Team labels from the cleaned frame, falling back to the generic HOME/AWAY."""
    return _label(df, "home_team", "HOME"), _label(df, "away_team", "AWAY")


def _new_rotation() -> dict:
    return {"by_rank": {r: [] for r in range(1, ROTATION_RANKS + 1)},
            "n_rotation": [], "top_pts": []}


def _collect_rotation(acc: dict, box) -> None:
    """Accumulate one box score's per-team minutes-by-rank, rotation size and top scorer.

    Per TEAM, not per game: a team's rotation is the unit every rotation target is expressed in,
    and pooling both sides doubles the sample for free. Rank is by minutes descending, so rank 1
    is that team's most-used player in THAT game -- the only way to compare a sim's rotation SHAPE
    to the real one without needing the two to agree on who the starters are.
    """
    for side in (box.home, box.away):
        if not side:
            continue
        mins = sorted((p.minutes for p in side), reverse=True)
        for r in range(1, ROTATION_RANKS + 1):
            if r <= len(mins):
                acc["by_rank"][r].append(mins[r - 1])
        acc["n_rotation"].append(sum(1 for m in mins if m >= ROTATION_MIN))
        acc["top_pts"].append(max((p.pts for p in side), default=0))


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _print_rotation(sim_rot: dict, act_rot: dict) -> None:
    print()
    print("--- rotation shape (per team, minutes by rank) ---")
    print(f"{'rank':>5} {'sim':>8} {'actual':>8} {'diff':>8}")
    for r in range(1, ROTATION_RANKS + 1):
        s_m, a_m = _mean(sim_rot["by_rank"][r]), _mean(act_rot["by_rank"][r])
        if s_m != s_m and a_m != a_m:
            continue
        print(f"{r:>5} {s_m:>8.1f} {a_m:>8.1f} {s_m - a_m:>+8.1f}")
    s_n, a_n = _mean(sim_rot["n_rotation"]), _mean(act_rot["n_rotation"])
    s_p, a_p = _mean(sim_rot["top_pts"]), _mean(act_rot["top_pts"])
    print()
    print(f"  players >= {ROTATION_MIN:.0f} min   sim {s_n:5.1f}   actual {a_n:5.1f}   "
          f"diff {s_n - a_n:+.1f}")
    print(f"  team-high points   sim {s_p:5.1f}   actual {a_p:5.1f}   diff {s_p - a_p:+.1f}")
    print()
    print("  A FLAT profile (rank 1-5 under, rank 8+ over) means minutes are spread too evenly.")
    print("  That compresses the talent gap between teams, and with it the predicted margins --")
    print("  which is the mechanism behind a low margin correlation and a coin-flip win pick.")
    print("  SUB_INCOMING_TEMPERATURE is the dial: <1 sharpens who checks in (v1.0 used 0.45).")


def peek(run_dir: str) -> int:
    folders = sorted(glob.glob(os.path.join(run_dir, "games", "*")))
    rows = []
    sim_rot, act_rot = _new_rotation(), _new_rotation()
    for folder in folders:
        pbp = os.path.join(folder, "playbyplay")
        sims = sorted(glob.glob(os.path.join(pbp, "sim_*_playbyplay.csv")))
        actual_path = os.path.join(pbp, "actual_playbyplay.csv")
        if not sims or not os.path.isfile(actual_path):
            continue
        adf = pd.read_csv(actual_path)
        home, away = _teams(adf)
        abox = generate_box_score(adf, home_team=home, away_team=away)

        pred_home, pred_away, pred_rows = [], [], []
        for s in sims:
            try:
                sdf = pd.read_csv(s)
                b = generate_box_score(sdf, home_team=home, away_team=away)
            except Exception as e:      # noqa: BLE001 - one unreadable sim must not stop the peek
                print(f"  ! {os.path.basename(s)}: {type(e).__name__}: {e}")
                continue
            pred_home.append(b.home_score)
            pred_away.append(b.away_score)
            pred_rows.append(len(sdf))
            _collect_rotation(sim_rot, b)
        if not pred_home:
            continue
        _collect_rotation(act_rot, abox)

        rows.append({
            "game": os.path.basename(folder),
            "sims": len(pred_home),
            "pred_home": sum(pred_home) / len(pred_home),
            "pred_away": sum(pred_away) / len(pred_away),
            "act_home": abox.home_score,
            "act_away": abox.away_score,
            "pred_rows": sum(pred_rows) / len(pred_rows),
            "act_rows": len(adf),
        })

    if not rows:
        print(f"No finished sims under {run_dir}/games/*/playbyplay/ yet.")
        return 1

    df = pd.DataFrame(rows)
    df["pred_total"] = df.pred_home + df.pred_away
    df["act_total"] = df.act_home + df.act_away

    pd.set_option("display.width", 200)
    print(df[["game", "sims", "pred_home", "act_home", "pred_away", "act_away",
              "pred_rows", "act_rows"]].to_string(index=False,
                                                  float_format=lambda v: f"{v:.1f}"))

    n_sims = int(df["sims"].sum())
    print()
    print(f"{len(df)} games with at least one finished sim, {n_sims} sims total")
    print("(biased early sample: short games finish first)")
    print()

    # Per TEAM, which is how the eval report and every dial target is expressed.
    pred_team = pd.concat([df.pred_home, df.pred_away]).mean()
    act_team = pd.concat([df.act_home, df.act_away]).mean()
    print(f"  team points   pred {pred_team:6.1f}   actual {act_team:6.1f}   "
          f"bias {pred_team - act_team:+.1f}")
    print(f"  total points  pred {df.pred_total.mean():6.1f}   actual {df.act_total.mean():6.1f}   "
          f"bias {df.pred_total.mean() - df.act_total.mean():+.1f}")
    print(f"  events/game   pred {df.pred_rows.mean():6.1f}   actual {df.act_rows.mean():6.1f}   "
          f"ratio {df.pred_rows.mean() / max(df.act_rows.mean(), 1e-9):.3f}")
    print()
    print("  events/game ratio is the pace signal: >1 means the sim generates more events than")
    print("  the real game, which DELTA_TIME_SCALE is the lever on (config.py).")

    _print_rotation(sim_rot, act_rot)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, metavar="RUNDIR",
                    help="Eval run dir, e.g. results/version2/v2-run1")
    args = ap.parse_args()
    raise SystemExit(peek(args.run))


if __name__ == "__main__":
    main()
