"""
Shared cleaned-data loading + game splitting.

``EventTimeModel`` and ``PlayerModel`` both turn the cleaned season CSVs into model
tensors, and the box-score tool reads the *same* games back out to validate against. All
three must agree on (a) which CSVs count as cleaned data, (b) the globally-unique
``game_id`` numbering, and (c) the deterministic train/val/holdout partition — otherwise a
"holdout game 42" in the split manifest would not be the same rows the box-score tool
loads. This module is that single source of truth.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

# Roster columns hold Python-list literals (e.g. "['A', 'B', ...]") in the cleaned CSVs.
ROSTER_STR_COLS = ("roster_home", "roster_away")


def cleaned_csvs(data_dir) -> list[Path]:
    """Return the cleaned season CSVs in ``data_dir`` (those with game_id + rosters).

    Sorted so the global ``game_id`` numbering in ``load_all_cleaned`` is deterministic.
    """
    out = []
    for p in sorted(Path(data_dir).glob("*.csv")):
        try:
            cols = pd.read_csv(p, nrows=0).columns
        except Exception:
            continue
        if "game_id" in cols and "roster_home" in cols:
            out.append(p)
    return out


def load_all_cleaned(data_dir, parse_rosters: bool = False) -> pd.DataFrame:
    """Concatenate all cleaned CSVs, keeping ``game_id`` globally unique across files.

    Each file's ``game_id`` is offset by the running max so ids never collide across
    seasons; the numbering depends only on the (sorted) file order, so it is stable for a
    given ``data_dir``. With ``parse_rosters`` the roster string-lists are decoded to real
    Python lists (what the box-score tool consumes).
    """
    frames = []
    offset = 0
    for p in cleaned_csvs(data_dir):
        df = pd.read_csv(p)
        df["game_id"] = df["game_id"].astype(int) + offset
        offset = int(df["game_id"].max()) + 1
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No cleaned CSVs found in {Path(data_dir).resolve()}")
    out = pd.concat(frames, ignore_index=True)
    if parse_rosters:
        for col in ROSTER_STR_COLS:
            if col in out.columns:
                out[col] = out[col].apply(_parse_roster)
    return out


def _parse_roster(value):
    """Decode a roster cell to a list of player names (already-a-list passes through)."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def split_games(unique_games, seed: int, test_frac: float, holdout_frac: float):
    """Partition game ids into disjoint (train, test, holdout) sets, deterministically.

    The shuffle is seeded, then the **holdout** is carved off first, then ``test``, with
    everything else going to ``train``. Carving holdout first keeps it fully reserved: it
    is excluded from training *and* from the early-stopping validation (``test``) split.
    Fractions are taken over the full game count.
    """
    games = np.array(sorted(set(np.asarray(unique_games).tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(games)

    n = len(games)
    n_holdout = int(round(n * holdout_frac)) if holdout_frac else 0
    n_test = max(1, int(round(n * test_frac))) if test_frac else 0
    # Guard tiny game counts: never let holdout+test swallow the whole pool.
    n_holdout = min(n_holdout, max(0, n - n_test - 1))

    holdout = set(games[:n_holdout].tolist())
    test = set(games[n_holdout:n_holdout + n_test].tolist())
    train = set(games[n_holdout + n_test:].tolist())
    return train, test, holdout
