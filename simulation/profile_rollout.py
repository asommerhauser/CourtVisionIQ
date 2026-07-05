"""
profile_rollout.py — find the real per-game bottleneck in a single rollout.

The batched coordinator showed the rollout is CPU/Python-bound, not GPU-bound (low batch occupancy,
slow forward/s, maxed CPU). This profiles ONE single-game rollout to localize the cost: is it
``GameSimulator.build_model_inputs`` (per-step Python input construction) or ``_infer`` (the model
forward pass / TF)? A warm-up run first absorbs one-time graph tracing, so the profiled run reflects
steady-state per-step cost.

Run on WSL (loads the trained model):

    python -m simulation.profile_rollout                  # first full-train holdout game
    python -m simulation.profile_rollout --game-id 298335
"""
from __future__ import annotations

try:  # TF before pandas-using modules (see evaluation.py / main.py)
    import tensorflow  # noqa: F401
except Exception:
    pass

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

from config import FULL_ARTIFACTS_ROOT
from data_loading import load_all_cleaned
from simulation.controller import GameController
from simulation.game_input import extract_game_input
from simulation.game_simulator import GameSimulator
from simulation.predict_game import _real_starters


def _first_holdout_game(state_path: str) -> int | None:
    p = Path(state_path)
    if not p.exists():
        return None
    ids = json.loads(p.read_text(encoding="utf-8")).get("holdout_game_ids") or []
    return int(ids[0]) if ids else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile one single-game rollout (no batching).")
    ap.add_argument("--game-id", type=int, default=None, help="Cleaned game_id to simulate.")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--artifacts-root", default=FULL_ARTIFACTS_ROOT)
    ap.add_argument("--state-path", default="./training/full_run_state.json")
    ap.add_argument("--top", type=int, default=25, help="Rows of profile output.")
    args = ap.parse_args()

    gid = args.game_id or _first_holdout_game(args.state_path)
    if gid is None:
        raise SystemExit("No --game-id and no holdout ids in state; pass --game-id.")

    df = load_all_cleaned(args.data_dir, parse_rosters=True)
    game = df[df["game_id"] == int(gid)].sort_values("time")
    if game.empty:
        raise SystemExit(f"game {gid} not found in cleaned data.")
    spec = extract_game_input(game)
    try:
        home_starters, away_starters = _real_starters(game)
    except ValueError:
        home_starters = away_starters = None

    print(f"[profile] loading model from {args.artifacts_root} ...")
    sim = GameSimulator.load(artifacts_root=args.artifacts_root)

    def run_once(seed: int):
        ctrl = GameController(sim, seed=seed)
        ctrl.start(spec.home_roster, spec.away_roster, season=str(spec.season),
                   home_starters=home_starters, away_starters=away_starters,
                   season_context=spec.season_context())
        return ctrl.run()

    print(f"[profile] warm-up run (absorbs graph tracing) on game {gid} ...")
    t0 = time.monotonic()
    warm = run_once(seed=0)
    t_warm = time.monotonic() - t0
    print(f"[profile] warm-up: {len(warm)} events in {t_warm:.1f}s")

    print("[profile] profiled run ...")
    pr = cProfile.Profile()
    t0 = time.monotonic()
    pr.enable()
    hist = run_once(seed=1)
    pr.disable()
    t_prof = time.monotonic() - t0
    print(f"[profile] profiled: {len(hist)} events in {t_prof:.1f}s "
          f"({1000 * t_prof / max(len(hist), 1):.1f} ms/event)\n")

    st = pstats.Stats(pr)
    print("=" * 70, "\nBy CUMULATIVE time (where the wall-clock goes):\n", "=" * 70, sep="")
    st.sort_stats("cumulative").print_stats(args.top)
    print("=" * 70, "\nBy TOTAL time (self-time hot spots):\n", "=" * 70, sep="")
    st.sort_stats("tottime").print_stats(15)


if __name__ == "__main__":
    main()
