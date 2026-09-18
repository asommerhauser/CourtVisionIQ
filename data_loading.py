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
# Season-context rest columns are roster-parallel numeric-list literals (e.g. "[1, 3, 1]")
# added by season_context.enrich; decoded the same way as rosters when requested.
REST_STR_COLS = ("rest_home", "rest_away")


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


_OFFSET_CACHE: dict[tuple, dict[Path, int]] = {}


def season_offsets(data_dir) -> dict[Path, int]:
    """The per-file ``game_id`` offset that makes ids globally unique. **The** definition.

    Raw ``game_id`` values collide across season files -- they are not even ordered by season
    (2016 holds 1313-2628, 2019 holds 1-1312, 2003 holds 15556-16832) -- so each file's ids are
    shifted past every earlier file's maximum. The numbering therefore depends only on the sorted
    file order, and is stable for a given ``data_dir``.

    This exists as its own function because it had two implementations and one of them was missing.
    ``load_all_cleaned`` did the walk inline; ``player_priors.build`` read each season CSV directly
    and applied no offset at all, so the priors sidecar was keyed by *per-season* id while every
    training row carried the *cumulative* id. Measured over the real corpus, the two agreed on
    **1,277 of 26,969 games** -- season 2003, the one file whose offset is zero -- and
    ``prior_features.merge_prior_features`` raises on that, so a full train aborted before its first
    epoch. A rule read in two places drifts; this is the one place.

    Memoized on each file's (size, mtime) so the repeated ``load_all_cleaned`` calls across a
    train's preprocess passes cost one extra column-only read, not one per call, and a re-clean
    invalidates it.
    """
    paths = cleaned_csvs(data_dir)
    key = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    hit = _OFFSET_CACHE.get(key)
    if hit is not None:
        return dict(hit)
    offsets: dict[Path, int] = {}
    offset = 0
    for p in paths:
        offsets[p] = offset
        ids = pd.read_csv(p, usecols=["game_id"])["game_id"].astype(int)
        offset = int(ids.max()) + offset + 1
    _OFFSET_CACHE[key] = dict(offsets)
    return offsets


def load_all_cleaned(data_dir, parse_rosters: bool = False, *,
                     min_season: int | None = None) -> pd.DataFrame:
    """Concatenate all cleaned CSVs, keeping ``game_id`` globally unique across files.

    The numbering is ``season_offsets``'; see there for why it is not computed here any more.
    With ``parse_rosters`` the roster string-lists are decoded to real Python lists (what the
    box-score tool consumes).

    ``min_season`` drops rows from seasons before it **after** the offset walk, so every surviving
    game keeps the id it already had. Filtering the file list instead would renumber the whole
    corpus; see ``config.MIN_TRAIN_SEASON``. ``None`` (the default) means the whole corpus, which is
    what the box-score tool, the shell and the report stack want -- the floor is a *training* bound,
    not a corpus one, so it is opt-in and ``load_training_corpus`` is the thing that opts in.
    """
    offsets = season_offsets(data_dir)
    frames = []
    for p in cleaned_csvs(data_dir):
        df = pd.read_csv(p)
        df["game_id"] = df["game_id"].astype(int) + offsets[p]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No cleaned CSVs found in {Path(data_dir).resolve()}")
    out = pd.concat(frames, ignore_index=True)
    if min_season is not None and "season" in out.columns:
        kept = out["season"].astype(int) >= int(min_season)
        if not kept.any():
            raise ValueError(
                f"min_season={min_season} leaves no games: the cleaned data in "
                f"{Path(data_dir).resolve()} spans seasons "
                f"{int(out['season'].min())}-{int(out['season'].max())}")
        out = out[kept].reset_index(drop=True)
    if parse_rosters:
        for col in (*ROSTER_STR_COLS, *REST_STR_COLS):
            if col in out.columns:
                out[col] = out[col].apply(_parse_roster)
    return out


def training_min_season() -> int | None:
    """``config.MIN_TRAIN_SEASON``, read at call time.

    Read here rather than imported at module scope on purpose: 3.0 lost a day to
    ``from config import X`` freezing three knobs at import time, so switching one off in a test or
    on the command line did nothing and the "feature disabled" path was silently untested.
    """
    import config
    return getattr(config, "MIN_TRAIN_SEASON", None)


def load_training_corpus(data_dir, parse_rosters: bool = False) -> pd.DataFrame:
    """The corpus **as training sees it**: ``load_all_cleaned`` floored at the training season bound.

    One name so the floor cannot be applied in some training paths and not others. Every head's
    ``_load_all`` and ``training.chronology.game_index`` come through here; anything that genuinely
    wants all 21 seasons -- the box-score validator, the shell, the priors sidecar, season context --
    keeps calling ``load_all_cleaned`` and says so by doing it.
    """
    return load_all_cleaned(data_dir, parse_rosters, min_season=training_min_season())


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


def resolve_partition(game_partition, game_id, seed, test_frac, holdout_frac):
    """Return the ``(train, test, holdout)`` game-id sets a model's preprocess should use.

    When ``game_partition`` is given (the curriculum's explicit chronological split — see
    ``training.chronology.sequential_partition``) it is normalized to int sets and used verbatim;
    otherwise this falls back to the random, seeded ``split_games`` over all games. Centralized so
    every model resolves the split the same way.
    """
    if game_partition is not None:
        train, test, holdout = game_partition
        return ({int(g) for g in train}, {int(g) for g in test}, {int(g) for g in holdout})
    return split_games(np.unique(game_id), seed=seed, test_frac=test_frac, holdout_frac=holdout_frac)
