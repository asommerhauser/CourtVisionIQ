"""
subset.py — extract a compact, representative training subset for the small heads.

The big player-vocab heads (player / substitution / sub_decision) want every game; the small
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
import config
from config import VOCAB_DIR
from data_loading import ROSTER_STR_COLS, load_training_corpus
from player_floor import (
    ANON_FILENAME,
    build_alias_map,
    n_slots as n_anon_slots,
    save_aliases,
)
from training.chronology import game_index, sequential_partition


def _game_players_and_season(data_dir: str) -> tuple[dict[int, set], dict[int, int]]:
    """Map each game_id to (set of players who appear in it, season).

    Players come from the union of the home/away roster cells across the game's rows — everyone who
    was on the floor at any point, which is exactly the coverage target. Season comes from the
    chronological game index.
    """
    df = load_training_corpus(data_dir, parse_rosters=True)
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
    """Select a per-season-rate subset of ``train_games``.

    Returns ``(sorted_game_ids, stats)``. One phase: bring each season up to
    ``round(rate(season) * games_in_season)`` by sampling that season's games uniformly (within a
    season every game is equally recent, so there is nothing to weight).

    **Coverage-completeness was retired in 3.2.** It used to run first: walk players rarest-first and,
    for any not yet covered, add one game containing them, so every player in the train pool was
    guaranteed at least one game and no embedding trained on zero rows. Under W4's vocabulary floor
    that guarantee is **inert** -- anyone it rescues with a single game falls below the floor anyway
    and maps to an anonymous slot -- and it was actively harmful, because it dragged old games into
    the sample for players who will be anonymous regardless. See docs/v3_2_direction.md 2.2.

    ``stats["players"]`` carries the per-player game count **within the subset**, which is what W4's
    floor reads. It is produced here because this is the only place the (game -> players) map is
    already in hand.
    """
    rng = np.random.default_rng(seed)
    train = sorted(int(g) for g in train_games)
    if not train:
        return [], {"n_train": 0, "n_subset": 0, "n_players": 0, "players": {}}

    # Per-game recency weights are gone with coverage-completeness: they only ever broke ties when
    # choosing WHICH game to add for a rare player, and the per-season fill is uniform by
    # construction. The modern tilt now lives entirely in the per-season RATES.

    # Games grouped by season, and the target count per season from its sample rate.
    season_games: dict[int, list[int]] = defaultdict(list)
    for g in train:
        season_games[game_season.get(g, -1)].append(g)
    rates = season_sample_rates(season_games.keys(), recent_rates, halflife)
    targets = {s: min(len(gs), round(rates.get(s, 0.0) * len(gs))) for s, gs in season_games.items()}

    chosen: set[int] = set()

    # Per-season fill up to each season's target (uniform within the season).
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
    # Per-player game count INSIDE the subset -- W4's vocabulary floor is defined against this, not
    # against the corpus and not against the train pool. Counted after selection for that reason.
    subset_players: Counter = Counter()
    for g in out:
        subset_players.update(game_players.get(g, ()))
    stats = {
        "n_train": len(train),
        "n_subset": len(out),
        "subset_frac_actual": round(len(out) / len(train), 4),
        "n_players": len(subset_players),
        "players": dict(sorted(subset_players.items())),
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

    # --- W4: the player vocabulary floor, and the anonymous slots below it ---
    # Built here because this is the only step that already holds BOTH halves: the per-player subset
    # game counts the floor is defined against, and the per-game name sets the slot assignment needs
    # to colour. Doing it in a head's preprocess would mean parsing every roster cell a second time.
    #
    # ``game_players`` covers the training corpus only, which is exactly right: it is floored at
    # MIN_TRAIN_SEASON, and no game below that floor ever reaches a head. Train, val and holdout are
    # all inside it, so no game anywhere can contain two players sharing a slot.
    floor = getattr(config, "MIN_PLAYER_SUBSET_GAMES", None)
    aliases = build_alias_map(stats["players"], game_players, floor,
                              max_slots=getattr(config, "ANON_SLOTS_MAX", None))
    if aliases:
        save_aliases(VOCAB_DIR, aliases, floor=floor)
        print(f"[subset] vocabulary floor {floor}: "
              f"{len(stats['players']) - len(aliases)} players keep their own embedding row, "
              f"{len(aliases)} are aliased to {n_anon_slots(aliases)} anonymous slots "
              f"-> {Path(VOCAB_DIR) / ANON_FILENAME}")
    else:
        # Either the floor is off, or every player clears it. Remove any stale map so a later train
        # cannot pick up a floor that is no longer configured -- that would be invisible.
        stale = Path(VOCAB_DIR) / ANON_FILENAME
        if stale.is_file():
            stale.unlink()
            print(f"[subset] no vocabulary floor in effect; removed stale {stale}")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": data_dir, "boundary_idx": boundary,
        "recent_season_rates": list(recent_rates), "halflife_seasons": halflife, "seed": seed,
        "player_floor": floor, "n_anon_slots": n_anon_slots(aliases), "n_aliased": len(aliases),
        **stats,
        "subset_game_ids": subset_ids,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    counts = sorted(stats["players"].values())
    med = counts[len(counts) // 2] if counts else 0
    print(f"[subset] {stats['n_subset']} / {stats['n_train']} train games "
          f"({stats['subset_frac_actual']:.1%} overall), {stats['n_players']} players in the subset, "
          f"median {med} games each.")
    # The distribution, because W4's vocabulary floor is chosen against it and a floor at the median
    # halves the embedding table. Printed so the number is read before the floor is set, not after.
    if counts:
        import numpy as _np
        qs = [int(_np.percentile(counts, q)) for q in (10, 25, 50, 75, 90)]
        print(f"[subset]   games per player: p10 {qs[0]}  p25 {qs[1]}  p50 {qs[2]}  "
              f"p75 {qs[3]}  p90 {qs[4]}  max {counts[-1]}")
        for floor in (10, 15, 20, 25, 30):
            keep = sum(1 for c in counts if c >= floor)
            print(f"[subset]   floor {floor:>3}: {keep:>5} players kept, "
                  f"{len(counts) - keep:>5} -> anonymous ({keep / len(counts):.1%} kept)")
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
    counts = sorted((payload.get("players") or {}).values())
    med = counts[len(counts) // 2] if counts else 0
    print(f"Players in the subset: {payload['n_players']} (median {med} games each)")
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
