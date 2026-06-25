"""
Shared season-context feature plumbing (player rest, team rest, season progress).

Single source of truth for the six season-level inputs so the four model wrappers
(Event/Time, Player, Conditional-Type, Substitution) stay in sync. Mirrors exactly how
``time_abs`` / ``delta_time`` are normalized and projected:

  * per-player rest (``rest_home`` / ``rest_away``, shape ``(SEQ, ROSTER_SIZE)``) is fed
    into the roster set-encoder alongside the player ids, so every head — including player
    selection — sees per-player freshness;
  * the four team scalars (``home_games_played`` / ``away_games_played`` /
    ``home_days_rest`` / ``away_days_rest``, shape ``(SEQ, 1)``) are ``Dense``-projected and
    concatenated into the fusion, just like the existing continuous features.

Rest is clipped at ``REST_CLIP_DAYS`` (beyond that is uniformly "long layoff") then z-scored
with train-only ``rest_mean`` / ``rest_std`` (persisted into ``norm_stats``); games-played
arrives already normalized by season length (see ``season_context.py``), so it is fed raw.
"""
from __future__ import annotations

import ast

import numpy as np

from config import ROSTER_SIZE

# Roster-parallel per-player rest inputs (shape (SEQ, ROSTER_SIZE), fed to the roster encoder).
REST_LIST_COLS = ("rest_home", "rest_away")
# Team-level scalar inputs (shape (SEQ, 1), projected + concatenated into the fusion).
TEAM_SCALAR_COLS = ("home_games_played", "away_games_played", "home_days_rest", "away_days_rest")
# All six, appended to each model's INPUT_KEYS (order stable).
SEASON_INPUT_KEYS = (*REST_LIST_COLS, *TEAM_SCALAR_COLS)

# Days beyond which extra rest carries no extra signal (injury layoffs all look the same).
REST_CLIP_DAYS = 30.0
# Matches season_context.DEFAULT_REST_DAYS — a sane fallback when no value is available.
DEFAULT_REST_DAYS = 3.0


def _to_list(value):
    """Decode a rest cell (already-a-list, NaN, or a list literal string) to a list."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def pad_rest(value, size: int = ROSTER_SIZE) -> np.ndarray:
    """Roster-parallel rest list -> fixed-length ``size`` float array (PAD slots are 0)."""
    buf = np.zeros((size,), dtype=np.float32)
    vals = _to_list(value)[:size]
    for i, v in enumerate(vals):
        buf[i] = float(v)
    return buf


# =====================
# --- Preprocessing ---
# =====================

def build_raw_season_cols(df) -> dict:
    """Raw (un-normalized) season-context arrays from the enriched cleaned ``df``."""
    cols = {c: np.stack(df[c].apply(pad_rest).to_numpy()) for c in REST_LIST_COLS}  # (N, 5)
    for c in TEAM_SCALAR_COLS:
        cols[c] = df[c].to_numpy(dtype=np.float32)                                   # (N,)
    return cols


def compute_rest_stats(raw, rosters, pad_player, train_mask):
    """``(rest_mean, rest_std)`` over valid (non-PAD) player-rest slots on TRAIN rows.

    Computed off the per-player arrays (the richest sample) with PAD slots excluded so the
    empty-slot zeros don't skew the stats; rest is clipped first to match the model transform.
    """
    parts = []
    for rest_key, roster_key in (("rest_home", "home_roster"), ("rest_away", "away_roster")):
        rest = np.clip(raw[rest_key][train_mask], 0.0, REST_CLIP_DAYS)
        valid = rosters[roster_key][train_mask] != pad_player
        if valid.any():
            parts.append(rest[valid])
    if not parts:
        return DEFAULT_REST_DAYS, 1.0
    allv = np.concatenate(parts)
    return float(allv.mean()), (float(allv.std()) or 1.0)


def standardize_season_cols(raw, rest_mean, rest_std) -> dict:
    """Clip+z-score the rest arrays and team day-rest scalars; leave games-played raw."""
    for c in REST_LIST_COLS:
        raw[c] = ((np.clip(raw[c], 0.0, REST_CLIP_DAYS) - rest_mean) / rest_std).astype(np.float32)
    for c in ("home_days_rest", "away_days_rest"):
        raw[c] = ((np.clip(raw[c], 0.0, REST_CLIP_DAYS) - rest_mean) / rest_std).astype(np.float32)
    for c in ("home_games_played", "away_games_played"):
        raw[c] = raw[c].astype(np.float32)
    return raw


def merge_season_features(df, cols, rosters, pad_player, train_mask, norm_stats,
                          refit: bool = True) -> dict:
    """Build, normalize, and merge the season-context arrays into ``cols``.

    When ``refit`` (the default), computes ``rest_mean`` / ``rest_std`` from the train rows and
    persists them into ``norm_stats`` (like the time stats) so inference applies the identical
    transform. When ``refit`` is False (staged curriculum runs), reuses the already-persisted
    ``rest_mean`` / ``rest_std`` from ``norm_stats`` so standardization stays fixed across stages.
    Mutates and returns ``cols``.
    """
    raw = build_raw_season_cols(df)
    if refit:
        rest_mean, rest_std = compute_rest_stats(raw, rosters, pad_player, train_mask)
        norm_stats["rest_mean"] = rest_mean
        norm_stats["rest_std"] = rest_std
    else:
        rest_mean, rest_std = norm_stats["rest_mean"], norm_stats["rest_std"]
    standardize_season_cols(raw, rest_mean, rest_std)
    cols.update(raw)
    return cols


def append_season_batches(batches, cols, idx, n, SEQ) -> None:
    """Pad/stack the season-context arrays for one game into ``batches`` (mirrors _build_split)."""
    for k in REST_LIST_COLS:
        buf = np.zeros((SEQ, ROSTER_SIZE), dtype=np.float32)
        buf[:n] = cols[k][idx]
        batches[k].append(buf)
    for k in TEAM_SCALAR_COLS:
        buf = np.zeros((SEQ, 1), dtype=np.float32)
        buf[:n, 0] = cols[k][idx]
        batches[k].append(buf)


# =====================
# --- Model graph   ---
# =====================

def make_season_inputs(SEQ):
    """Keras Inputs for the season-context features: (rest_home, rest_away, team_scalars dict)."""
    from keras import Input  # local import: keep the preprocessing helpers TF-free.

    rest_home = Input(shape=(SEQ, ROSTER_SIZE), dtype="float32", name="rest_home")
    rest_away = Input(shape=(SEQ, ROSTER_SIZE), dtype="float32", name="rest_away")
    team = {c: Input(shape=(SEQ, 1), dtype="float32", name=c) for c in TEAM_SCALAR_COLS}
    return rest_home, rest_away, team


def season_team_projections(team) -> list:
    """Dense(16) projection of each team scalar (mirrors the time_abs/delta_time projections)."""
    from keras import layers  # local import: keep the preprocessing helpers TF-free.

    return [layers.Dense(16, name=f"{c}_proj")(team[c]) for c in TEAM_SCALAR_COLS]
