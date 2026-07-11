"""
subset.py — extract a compact, representative training subset for the small heads.

The big player-vocab heads (player / substitution / stint_length) want every game; the small
categorical/regression heads (event/type/result/conditional-time) saturate well before they've seen
the whole corpus and then overfit. This module carves a slice of the train pool for those small
heads with two properties:

  * **Per-season sample rate, modern-heavy** — each season contributes a fixed fraction of its
    games. The most recent seasons are sampled heavily so current players get a large sample
    (``SUBSET_RECENT_SEASON_RATES``, newest first — e.g. 70% / 40% / 25%); older seasons decay
    gently from there (halving every ``SUBSET_RECENCY_HALFLIFE_SEASONS`` seasons). So the modern
    game dominates the sample without the old game vanishing.
  * **Coverage-complete** — every player who appears anywhere in the train pool is guaranteed at
    least one game in the subset, so no player embedding trains on zero rows. Rare / old-only players
    pull in the older games they need regardless of their season's rate.

The selection is deterministic (seeded) and persisted to ``SUBSET_GAMES_PATH`` as a flat list of
game ids plus the parameters + coverage stats that produced it. ``extract`` reads the live
``full_run_state.json`` so the subset is carved from *exactly* the same train pool the full run
trains on (same boundary cut), then training routes ``SUBSET_MODEL_KEYS`` to it (see
``models.pipeline.run_stage`` / ``training.full_run``).

CLI:
    python -m training.subset extract           # build + persist the subset from full_run_state.json
    python -m training.subset extract --frac 0.10 --halflife 6
    python -m training.subset show              # print the saved subset's summary
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from config import (
    FINAL_HOLDOUT_GAMES, SUBSET_GAMES_PATH, SUBSET_RECENCY_HALFLIFE_SEASONS,
    SUBSET_RECENT_SEASON_RATES, SUBSET_SEED, TEST_FRAC,
)
from data_loading import ROSTER_STR_COLS, load_all_cleaned
from training.chronology import game_index, sequential_partition


def _game_players_and_season(data_dir: str) -> tuple[dict[int, set], dict[int, int]]:
    """Map each game_id to (set of players who appear in it, season).

    Players come from the union of the home/away roster cells across the game's rows — everyone who
    was on the floor at any point, which is exactly the coverage target. Season comes from the
    chronological game index.
    """
    df = load_all_cleaned(data_dir, parse_rosters=True)
    seasons = {int(g): int(s) for g, s in df.groupby("game_id")["season"].first().items()} \
        if "season" in df.columns else {}

    players: dict[int, set] = defaultdict(set)
    for col in ROSTER_STR_COLS:
        if col not in df.columns:
            continue
        for gid, roster in zip(df["game_id"].to_numpy(), df[col].to_numpy()):
            if roster:
                players[int(gid)].update(p for p in roster if p and p != "PAD")
    return players, seasons


def season_sample_rates(seasons, recent_rates=SUBSET_RECENT_SEASON_RATES,
                         halflife: float = SUBSET_RECENCY_HALFLIFE_SEASONS) -> dict[int, float]:
    """Per-season sample rate: the newest seasons take ``recent_rates`` (newest first), then the
    rest decay from the last recent rate, halving every ``halflife`` seasons.

    Rank 0 = newest season. For rank ``r < len(recent_rates)`` the rate is ``recent_rates[r]``; for
    older seasons it is ``recent_rates[-1] * 0.5 ** ((r - (len(recent_rates)-1)) / halflife)`` — a
    smooth exponential tail anchored at the last recent rate (so the curve is continuous).
    """
    desc = sorted(set(int(s) for s in seasons), reverse=True)
    base = recent_rates[-1]
    anchor = len(recent_rates) - 1
    rates: dict[int, float] = {}
    for rank, s in enumerate(desc):
        if rank < len(recent_rates):
            rates[s] = float(recent_rates[rank])
        else:
            rates[s] = float(base * 0.5 ** ((rank - anchor) / max(halflife, 1e-6)))
    return rates


def build_subset(train_games, game_players: dict[int, set], game_season: dict[int, int], *,
                 recent_rates=SUBSET_RECENT_SEASON_RATES,
                 halflife: float = SUBSET_RECENCY_HALFLIFE_SEASONS,
                 seed: int = SUBSET_SEED) -> tuple[list[int], dict]:
    """Select a per-season-rate, coverage-complete subset of ``train_games``.

    Returns ``(sorted_game_ids, stats)``. Two phases:
      1. **Coverage** — walk players rarest-first; for any not yet covered, add one game containing
         them, picked recency-weighted among their games. Guarantees every train player appears
         (rare / old-only players anchor the older games they need).
      2. **Per-season fill** — bring each season up to ``round(rate(season) * games_in_season)`` by
         adding more of that season's games at random (within a season every game is equally recent,
         so the fill is uniform). A season already over its target from coverage keeps its games.
    """
    rng = np.random.default_rng(seed)
    train = sorted(int(g) for g in train_games)
    if not train:
        return [], {"n_train": 0, "n_subset": 0, "n_players": 0, "n_covered": 0}

    newest = max(game_season.get(g, 0) for g in train)
    recency_w = {g: 0.5 ** ((newest - game_season.get(g, newest)) / max(halflife, 1e-6)) for g in train}

    # Games grouped by season, and the target count per season from its sample rate.
    season_games: dict[int, list[int]] = defaultdict(list)
    for g in train:
        season_games[game_season.get(g, -1)].append(g)
    rates = season_sample_rates(season_games.keys(), recent_rates, halflife)
    targets = {s: min(len(gs), round(rates.get(s, 0.0) * len(gs))) for s, gs in season_games.items()}

    # Player -> list of train games containing them; and per-player frequency (for rarest-first).
    player_games: dict[str, list[int]] = defaultdict(list)
    for g in train:
        for p in game_players.get(g, ()):  # only players that actually appear
            player_games[p].append(g)
    freq = Counter({p: len(gs) for p, gs in player_games.items()})

    chosen: set[int] = set()
    covered: set[str] = set()

    # Phase 1: coverage, rarest players first (so scarce/old-only players anchor the old games).
    for p in sorted(player_games, key=lambda p: freq[p]):
        if p in covered:
            continue
        cands = [g for g in player_games[p] if g not in chosen]
        if not cands:
            covered.add(p)
            continue
        w = np.array([recency_w[g] for g in cands], dtype=np.float64)
        pick = int(cands[rng.choice(len(cands), p=w / w.sum())])
        chosen.add(pick)
        covered.update(game_players.get(pick, ()))

    # Phase 2: per-season fill up to each season's target (uniform within the season).
    for s, gs in season_games.items():
        need = targets[s] - sum(1 for g in gs if g in chosen)
        if need <= 0:
            continue
        pool = [g for g in gs if g not in chosen]
        if not pool:
            continue
        picks = rng.choice(len(pool), size=min(need, len(pool)), replace=False)
        chosen.update(int(pool[i]) for i in picks)

    out = sorted(chosen)
    # Per-season breakdown: chosen / total (achieved rate) so the modern-heavy tilt is verifiable.
    by_season = {
        str(s): {"chosen": sum(1 for g in gs if g in chosen), "total": len(gs),
                 "rate": round(rates.get(s, 0.0), 3)}
        for s, gs in sorted(season_games.items())
    }
    stats = {
        "n_train": len(train),
        "n_subset": len(out),
        "subset_frac_actual": round(len(out) / len(train), 4),
        "n_players": len(player_games),
        "n_covered": len(covered),
        "uncovered_players": len(player_games) - len(covered),
        "by_season": by_season,
    }
    return out, stats


def extract(*, recent_rates=SUBSET_RECENT_SEASON_RATES,
            halflife: float = SUBSET_RECENCY_HALFLIFE_SEASONS,
            seed: int = SUBSET_SEED, out_path: str = SUBSET_GAMES_PATH,
            state_path: str = "./training/full_run_state.json") -> dict:
    """Build the subset from the full-run train pool and persist it to ``out_path``.

    Reads ``full_run_state.json`` for the data dir + boundary cut so the subset is carved from the
    exact same train games the full run uses (val/holdout are left untouched — the subset only ever
    shrinks the *train* set for the small heads).
    """
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    data_dir = state["data_dir"]
    boundary = int(state["boundary_idx"])

    idx = game_index(data_dir)
    train_games, _, _ = sequential_partition(
        idx, boundary, n_holdout=FINAL_HOLDOUT_GAMES, val_frac=TEST_FRAC, seed=state.get("seed", seed),
    )
    print(f"[subset] full-run train pool = {len(train_games)} games (boundary {boundary}); "
          f"reading rosters to map players...")
    game_players, game_season = _game_players_and_season(data_dir)

    subset_ids, stats = build_subset(
        train_games, game_players, game_season,
        recent_rates=recent_rates, halflife=halflife, seed=seed,
    )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": data_dir, "boundary_idx": boundary,
        "recent_season_rates": list(recent_rates), "halflife_seasons": halflife, "seed": seed,
        **stats,
        "subset_game_ids": subset_ids,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[subset] {stats['n_subset']} / {stats['n_train']} train games "
          f"({stats['subset_frac_actual']:.1%} overall), every player covered: "
          f"{stats['uncovered_players'] == 0} "
          f"({stats['n_covered']}/{stats['n_players']} players).")
    for s, b in stats["by_season"].items():
        print(f"[subset]   {s}: {b['chosen']:>4}/{b['total']:<4} ({b['chosen']/b['total']:.0%}, "
              f"target rate {b['rate']:.0%})")
    print(f"[subset] saved -> {Path(out_path).resolve()}")
    print("[subset] now (re)start training:  python train.py --full --version <X.Y> --batch-size <N>")
    return payload


def load_subset_games(path: str = SUBSET_GAMES_PATH) -> set[int] | None:
    """Return the persisted subset game-id set, or None if it hasn't been extracted yet."""
    p = Path(path)
    if not p.exists():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {int(g) for g in payload.get("subset_game_ids", [])}


def show(path: str = SUBSET_GAMES_PATH) -> None:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"No subset at {p.resolve()} — run:  python -m training.subset extract")
    payload = json.loads(p.read_text(encoding="utf-8"))
    print(f"Subset: {payload['n_subset']}/{payload['n_train']} train games "
          f"({payload.get('subset_frac_actual', 0):.1%} overall), "
          f"recent_rates={payload.get('recent_season_rates')}, "
          f"halflife={payload['halflife_seasons']}, seed={payload['seed']}")
    print(f"Players covered: {payload['n_covered']}/{payload['n_players']} "
          f"(uncovered {payload['uncovered_players']})")
    for s, b in payload["by_season"].items():
        print(f"  {s}: {b['chosen']}/{b['total']} ({b['rate']:.0%} target)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract the representative training subset.")
    sub = ap.add_subparsers(dest="command", required=True)
    pe = sub.add_parser("extract", help="Build + persist the subset from full_run_state.json.")
    pe.add_argument("--rates", type=float, nargs="+", default=None,
                    help="Per-season rates, newest first (default config.SUBSET_RECENT_SEASON_RATES).")
    pe.add_argument("--halflife", type=float, default=SUBSET_RECENCY_HALFLIFE_SEASONS,
                    help="Older-season decay halflife in seasons.")
    pe.add_argument("--seed", type=int, default=SUBSET_SEED)
    pe.add_argument("--out", default=SUBSET_GAMES_PATH)
    pe.add_argument("--state", default="./training/full_run_state.json")
    sub.add_parser("show", help="Print the saved subset summary.")

    args = ap.parse_args()
    if args.command == "extract":
        rates = tuple(args.rates) if args.rates else SUBSET_RECENT_SEASON_RATES
        extract(recent_rates=rates, halflife=args.halflife, seed=args.seed,
                out_path=args.out, state_path=args.state)
    elif args.command == "show":
        show()


if __name__ == "__main__":
    main()
