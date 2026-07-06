"""
Shared game-state feature plumbing (running score, period/clock, per-period team fouls).

The models were score-blind: the fusion saw event tokens, rosters, absolute game time, and
season context, but never the *state of the game* — who is ahead, by how much, which period,
how long is left in it, how close each team is to the penalty. That state drives real
basketball (leaders sit on leads, trailers foul and shoot threes, garbage time compresses
margins, the bonus manufactures free throws), and a transformer cannot reconstruct the
running score by exact arithmetic over ~500 event tokens — so we hand it to the model
explicitly, exactly the way ``season_features`` hands over rest / games-played.

Six per-row scalars, each ``Dense(16)``-projected and concatenated into the fusion (like the
``time_abs`` / ``delta_time`` / team-scalar projections):

  * ``score_diff``   — home minus away points so far (signed);
  * ``score_total``  — combined points so far (a pace / game-phase proxy);
  * ``period_idx``   — 0–3 regulation, 4+ per overtime;
  * ``period_time_left`` — seconds left in the current period;
  * ``team_fouls_home`` / ``team_fouls_away`` — personal fouls this period per side
    (the penalty/bonus proximity).

Normalization is by **fixed constants** (below), not train-fit stats — the quantities have
known natural scales, so there are no new ``norm_stats`` keys to persist and inference needs
no loaded statistics. The single ``derive_game_state`` scan is used by BOTH preprocessing
(over the cleaned event rows) and the simulator (over ``self.history``), so train/inference
parity holds by construction. State at row *i* is inclusive of row *i*'s own event, matching
the live controller score when row *i* is the last event in history.
"""
from __future__ import annotations

import ast

import numpy as np

# --- Period geometry (mirrors simulation/controller.py constants) ---
PERIOD_LENGTH = 720          # 12:00 regulation quarter (seconds)
OT_LENGTH = 300              # 5:00 overtime period
REGULATION = 4 * PERIOD_LENGTH  # 2880s

# Fouls that do NOT add to a team's per-period penalty count (offensive fouls are turnovers,
# technicals are bench/tech fouls) — everything else personal counts, matching the bonus intent.
NON_TEAM_FOUL_TYPES = {"technical", "offensive"}
# Non-play frames carry no state contribution.
_SKIP_EVENTS = {"start", "end", "none", "PAD", "UNK", ""}

# Per-row scalar inputs (shape (SEQ, 1), projected + concatenated into the fusion).
GAME_STATE_KEYS = (
    "score_diff", "score_total", "period_idx", "period_time_left",
    "team_fouls_home", "team_fouls_away",
)
GAME_STATE_INPUT_KEYS = GAME_STATE_KEYS

# Fixed normalization: (clip_lo, clip_hi, divisor). Chosen so typical values land ~[-1, 1].
_NORM = {
    "score_diff": (-60.0, 60.0, 25.0),
    "score_total": (0.0, 300.0, 220.0),
    "period_idx": (0.0, 8.0, 5.0),
    "period_time_left": (0.0, 900.0, 720.0),
    "team_fouls_home": (0.0, 12.0, 6.0),
    "team_fouls_away": (0.0, 12.0, 6.0),
}


def _period_index(t: float) -> int:
    """Monotonic period id at clock ``t`` (0–3 regulation, then one per OT)."""
    if t < REGULATION:
        return int(t // PERIOD_LENGTH)
    return 4 + int((t - REGULATION) // OT_LENGTH)


def _period_end(t: float) -> float:
    """Clock at the end of the period containing ``t`` (seconds left = this − t)."""
    if t < REGULATION:
        return (int(t // PERIOD_LENGTH) + 1) * PERIOD_LENGTH
    return REGULATION + (int((t - REGULATION) // OT_LENGTH) + 1) * OT_LENGTH


def _roster(value) -> list:
    """Coerce a roster cell (list or "['A','B']" string) to a list of names (mirrors box_score)."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def _norm(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


# =====================
# --- Derivation    ---
# =====================

def derive_game_state(rows) -> dict[str, np.ndarray]:
    """Per-row running game state for one game's ordered event ``rows`` (inclusive of each row).

    ``rows`` is an ordered iterable of dict-like events carrying at least ``event, player,
    type, result, time`` and the per-row ``roster_home`` / ``roster_away`` snapshots — the
    shape produced by both the cleaned data and ``GameSimulator`` history. Scoring follows the
    box-score semantics (made shot → 2 / 3 / 1 by type; scoring team = the player's side by
    roster membership). Returns raw (un-normalized) ``(N,)`` float arrays keyed by
    ``GAME_STATE_KEYS``.
    """
    rows = list(rows)
    n = len(rows)
    out = {k: np.zeros((n,), dtype=np.float32) for k in GAME_STATE_KEYS}

    home_pts = away_pts = 0
    fouls_home = fouls_away = 0
    cur_period = -1
    for i, row in enumerate(rows):
        t = float(row.get("time") or 0.0)
        period = _period_index(t)
        if period != cur_period:            # per-period team-foul reset (controller parity)
            fouls_home = fouls_away = 0
            cur_period = period

        event = _norm(row.get("event"))
        if event not in _SKIP_EVENTS:
            player = _norm(row.get("player"))
            home_roster = _roster(row.get("roster_home"))
            away_roster = _roster(row.get("roster_away"))
            team = ("home" if player in home_roster
                    else "away" if player in away_roster else None)
            etype = _norm(row.get("type"))
            result = _norm(row.get("result"))
            if event == "shot" and result == "made":
                pts = 3 if etype == "3pt" else 1 if etype == "free throw" else 2
                if team == "home":
                    home_pts += pts
                elif team == "away":
                    away_pts += pts
            elif event == "foul" and etype not in NON_TEAM_FOUL_TYPES:
                if team == "home":
                    fouls_home += 1
                elif team == "away":
                    fouls_away += 1

        out["score_diff"][i] = home_pts - away_pts
        out["score_total"][i] = home_pts + away_pts
        out["period_idx"][i] = period
        out["period_time_left"][i] = _period_end(t) - t
        out["team_fouls_home"][i] = fouls_home
        out["team_fouls_away"][i] = fouls_away
    return out


def normalize_game_state(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Clip + scale each raw game-state array by its fixed constant (no train-fit stats)."""
    out = {}
    for k in GAME_STATE_KEYS:
        lo, hi, div = _NORM[k]
        out[k] = (np.clip(raw[k], lo, hi) / div).astype(np.float32)
    return out


# =====================
# --- Preprocessing ---
# =====================

def merge_game_state_features(df, cols) -> dict:
    """Derive, normalize, and merge the game-state arrays into ``cols`` (positional over ``df``).

    Scans each game (grouped, original row order preserved) and writes the six normalized
    per-row arrays into ``cols`` aligned to ``df``'s positional index — the same layout as the
    encoded categorical / season columns. Needs no train mask or ``norm_stats`` (fixed-constant
    normalization). Mutates and returns ``cols``.
    """
    n = len(df)
    game_ids = df["game_id"].to_numpy()
    records = df.to_dict("records")
    raw = {k: np.zeros((n,), dtype=np.float32) for k in GAME_STATE_KEYS}
    for g in _ordered_unique(game_ids):
        pos = np.where(game_ids == g)[0]
        gs = derive_game_state([records[i] for i in pos])
        for k in GAME_STATE_KEYS:
            raw[k][pos] = gs[k]
    cols.update(normalize_game_state(raw))
    return cols


def _ordered_unique(arr):
    """Unique values in first-appearance order (game groups stay in the cleaned-file order)."""
    seen = set()
    order = []
    for v in arr:
        if v not in seen:
            seen.add(v)
            order.append(v)
    return order


def append_game_state_batches(batches, cols, idx, n, SEQ) -> None:
    """Pad/stack the game-state arrays for one game into ``batches`` (mirrors append_season_batches)."""
    for k in GAME_STATE_KEYS:
        buf = np.zeros((SEQ, 1), dtype=np.float32)
        buf[:n, 0] = cols[k][idx]
        batches[k].append(buf)


# =====================
# --- Model graph   ---
# =====================

def make_game_state_inputs(SEQ) -> dict:
    """Keras Inputs for the six game-state scalars (shape (SEQ, 1) each)."""
    from keras import Input  # local import: keep the preprocessing helpers TF-free.

    return {k: Input(shape=(SEQ, 1), dtype="float32", name=k) for k in GAME_STATE_KEYS}


def game_state_projections(inputs: dict) -> list:
    """Dense(16) projection of each game-state scalar (mirrors the time/team-scalar projections)."""
    from keras import layers  # local import: keep the preprocessing helpers TF-free.

    return [layers.Dense(16, name=f"{k}_proj")(inputs[k]) for k in GAME_STATE_KEYS]
