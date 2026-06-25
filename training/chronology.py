"""
chronology.py — chronological game ordering, sequential (next-N) splits, and the curriculum
stop-point schedule.

The curriculum trains on *contiguous, cumulative* slices of the corpus and, after each slice,
predicts the **next block of real games** as a holdout (no random split). All three concepts
need one agreed chronological ordering of games, which this module owns:

  * ``game_index``         — one row per game (season, playoff flag, date, within-season regular
                             ordinal) ordered chronologically; ``pos`` is the global rank.
  * ``sequential_partition`` — given a boundary position, return ``(train, val, holdout)`` game-id
                             sets in the *same shape* ``data_loading.split_games`` returns, so it
                             drops straight into every model's ``preprocess(game_partition=...)``.
  * ``build_schedule``     — the repeating stop-point cycle (25% / 50% / pre-playoffs per season,
                             starting after a bootstrap), each entry carrying its boundary + the
                             next-N holdout game ids.

Ordering matches ``data_loading.load_all_cleaned`` (the global ``game_id`` numbering); a game's
``pos`` here is its index in that chronological order, so ``holdout`` ids and ``train`` slices line
up with what every model and the box-score tooling load.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BOUNDARY_CYCLE, HOLDOUT_GAMES, SEASONS_PER_STAGE, SEED, TEST_FRAC
from data_loading import load_all_cleaned


def game_index(data_dir: str = "./data") -> pd.DataFrame:
    """One row per game, ordered chronologically; ``pos`` is the global chronological rank.

    Columns: ``game_id, season, playoff, game_date, is_regular, pos, season_reg_count,
    season_reg_ordinal``. Sorted by ``(season, playoff, game_date, game_id)`` so regular-season
    games precede that season's playoffs and dates break ties within a phase. ``season_reg_ordinal``
    is the 0-based rank among a season's regular games (−1 for playoff games).
    """
    df = load_all_cleaned(data_dir)
    meta = df.groupby("game_id", sort=False).first().reset_index()[
        [c for c in ("game_id", "season", "playoff", "game_date") if c in df.columns]
    ]
    meta["game_id"] = meta["game_id"].astype(int)
    meta["season"] = meta["season"].astype(int) if "season" in meta else 0
    # Cleaned data: playoff flag 1 = regular season, 2 = playoff. Default to regular if absent.
    meta["playoff"] = meta["playoff"].astype(int) if "playoff" in meta else 1
    meta["game_date"] = (
        pd.to_datetime(meta["game_date"], errors="coerce") if "game_date" in meta else pd.NaT
    )
    meta["is_regular"] = meta["playoff"] == 1

    # Stable chronological sort; NaT dates fall back to game_id order (filled to a min sentinel).
    meta["_date_key"] = meta["game_date"].fillna(pd.Timestamp.min)
    meta = meta.sort_values(["season", "playoff", "_date_key", "game_id"],
                            kind="stable").drop(columns="_date_key").reset_index(drop=True)
    meta["pos"] = np.arange(len(meta), dtype=int)

    reg = meta[meta["is_regular"]]
    meta["season_reg_count"] = (
        meta["season"].map(reg.groupby("season").size()).fillna(0).astype(int)
    )
    meta["season_reg_ordinal"] = -1
    if not reg.empty:
        meta.loc[meta["is_regular"], "season_reg_ordinal"] = (
            meta[meta["is_regular"]].groupby("season").cumcount()
        )
    return meta


def sequential_partition(index: pd.DataFrame, boundary_idx: int, *,
                         n_holdout: int = HOLDOUT_GAMES, val_frac: float = TEST_FRAC,
                         seed: int = SEED) -> tuple[set, set, set]:
    """``(train, val, holdout)`` game-id sets for a chronological cut at ``boundary_idx``.

    Train = every game before ``boundary_idx``; holdout = the next ``n_holdout`` games (the block
    we predict for this stage); val = a seeded random subset of train, used only for early
    stopping. Matches the ``(train, test, holdout)`` shape of ``data_loading.split_games`` so it
    feeds directly into ``preprocess(game_partition=...)``.
    """
    ordered = index["game_id"].to_numpy()
    n = len(ordered)
    boundary_idx = int(max(0, min(boundary_idx, n)))

    train_ids = ordered[:boundary_idx]
    holdout_ids = ordered[boundary_idx:boundary_idx + n_holdout]

    train_arr = np.array(sorted(int(g) for g in train_ids))
    n_val = max(1, int(round(len(train_arr) * val_frac))) if (val_frac and len(train_arr) > 1) else 0
    n_val = min(n_val, max(0, len(train_arr) - 1))  # never let val swallow the whole train pool
    perm = train_arr.copy()
    np.random.default_rng(seed).shuffle(perm)
    val_set = set(int(g) for g in perm[:n_val])
    train_set = set(int(g) for g in perm[n_val:])
    return train_set, val_set, set(int(g) for g in holdout_ids)


def build_schedule(index: pd.DataFrame, *, seasons_per_stage: int = SEASONS_PER_STAGE,
                   cycle: tuple[str, ...] = BOUNDARY_CYCLE,
                   n_holdout: int = HOLDOUT_GAMES) -> list[dict]:
    """The curriculum stop-point schedule: a list of stage dicts in chronological order.

    A stop is placed every ``seasons_per_stage`` seasons — the first after the first
    ``seasons_per_stage`` seasons — and the stop *point* cycles through ``cycle`` across those
    stops. So with 3 and the default cycle: train ~3 seasons → stop 25% into the next season →
    +3 seasons → stop 50% in → +3 seasons → stop pre-playoffs → repeat. ``frac:f`` cuts at the
    game ``f`` of the way through that season's regular games; ``pre_playoffs`` cuts at the first
    playoff game (so the holdout is the first ``n_holdout`` playoff games — skipped, advancing the
    cycle, if that season has no playoffs). Boundaries are forced strictly increasing; any stop
    without room for a full ``n_holdout`` holdout is dropped. A final ``final_full`` stage trains
    on the entire corpus.

    Each entry: ``{stage, boundary_idx, boundary_type, season, train_games, holdout_game_ids}``.
    """
    total = len(index)
    ordered_ids = index["game_id"].to_numpy()
    seasons = sorted(index["season"].unique().tolist())

    schedule: list[dict] = []

    def _last_boundary() -> int:
        return schedule[-1]["boundary_idx"] if schedule else 0

    # Stop k (0-based) lands in the season ``seasons_per_stage`` * (k+1) along, with its point
    # taken from ``cycle[k % len(cycle)]`` — so stops are spaced ~seasons_per_stage seasons apart
    # while the 25% / 50% / pre-playoffs point rotates.
    k = 0
    while seasons_per_stage * (k + 1) < len(seasons):
        s = seasons[seasons_per_stage * (k + 1)]
        entry = cycle[k % len(cycle)]
        k += 1

        srows = index[index["season"] == s]
        reg = srows[srows["is_regular"]]
        if reg.empty:
            continue
        first_reg_pos = int(reg["pos"].min())
        reg_count = int(len(reg))

        if entry.startswith("frac:"):
            f = float(entry.split(":", 1)[1])
            boundary_idx = first_reg_pos + int(f * reg_count)
        elif entry == "pre_playoffs":
            playoff_rows = srows[~srows["is_regular"]]
            if playoff_rows.empty:
                continue
            boundary_idx = int(playoff_rows["pos"].min())
        else:
            raise ValueError(f"unknown boundary cycle entry: {entry!r}")

        if boundary_idx <= _last_boundary():
            continue                                       # keep boundaries strictly increasing
        if boundary_idx + n_holdout > total:
            continue                                       # not enough games left for a holdout
        schedule.append({
            "stage": len(schedule) + 1,
            "boundary_idx": boundary_idx,
            "boundary_type": entry,
            "season": int(s),
            "train_games": boundary_idx,
            "holdout_game_ids": [int(g) for g in ordered_ids[boundary_idx:boundary_idx + n_holdout]],
        })

    # Final stage: train on the whole corpus (no further holdout to predict).
    if _last_boundary() < total:
        schedule.append({
            "stage": len(schedule) + 1,
            "boundary_idx": total,
            "boundary_type": "final_full",
            "season": int(seasons[-1]) if seasons else 0,
            "train_games": total,
            "holdout_game_ids": [],
        })
    return schedule


def format_schedule(index: pd.DataFrame, schedule: list[dict]) -> str:
    """Human-readable schedule table for the user to approve at ``init``."""
    by_id = index.set_index("game_id")
    lines = [f"{'stage':>5}  {'type':<14}  {'season':>6}  {'train_games':>11}  holdout"]
    for st in schedule:
        hid = st["holdout_game_ids"]
        if hid:
            first = by_id.loc[hid[0]]
            tag = f"{len(hid)} games from g{hid[0]} (S{int(first['season'])}" \
                  f"{'/PO' if int(first['playoff']) != 1 else ''})"
        else:
            tag = "— (full corpus)"
        lines.append(f"{st['stage']:>5}  {st['boundary_type']:<14}  {st['season']:>6}  "
                     f"{st['train_games']:>11}  {tag}")
    return "\n".join(lines)


__all__ = ["game_index", "sequential_partition", "build_schedule", "format_schedule"]
