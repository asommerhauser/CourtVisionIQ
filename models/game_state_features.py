"""
Shared game-state feature plumbing (running score, period/clock, per-period team fouls).

The models were score-blind: the fusion saw event tokens, rosters, absolute game time, and
season context, but never the *state of the game* — who is ahead, by how much, which period,
how long is left in it, how close each team is to the penalty. That state drives real
basketball (leaders sit on leads, trailers foul and shoot threes, garbage time compresses
margins, the bonus manufactures free throws), and a transformer cannot reconstruct the
running score by exact arithmetic over ~500 event tokens — so we hand it to the model
explicitly, exactly the way ``season_features`` hands over rest / games-played.

Seven per-row scalars, each ``Dense(16)``-projected and concatenated into the fusion (like the
``time_abs`` / ``delta_time`` / team-scalar projections):

  * ``score_diff``   — home minus away points so far (signed);
  * ``score_total``  — combined points so far (a pace / game-phase proxy);
  * ``period_idx``   — 0–3 regulation, 4+ per overtime;
  * ``period_time_left`` — seconds left in the current period;
  * ``team_fouls_home`` / ``team_fouls_away`` — personal fouls this period per side
    (the penalty/bonus proximity);
  * ``poss_clock``   — seconds since the current possession started, a shot-clock proxy.

The shot clock drives shot selection, shot-clock turnovers and the timing of the next event, and
attention cannot reconstruct it: doing so means finding the last possession change and
subtracting, over a ~500-token history. There is no shot clock in the source data, so it is
derived from the event stream — see :func:`possession_boundary` for the rule and its one known
inaccuracy.

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

from zones import points_for_shot

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
    "team_fouls_home", "team_fouls_away", "poss_clock",
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
    # Clipped at the shot clock itself: past 24s the exact value carries no information the model
    # can act on, and the clip is what keeps a stale possession (a data gap, a long dead ball)
    # from reading as an extreme input.
    "poss_clock": (0.0, 24.0, 24.0),
}


# Possession-boundary outcomes, returned by :func:`possession_boundary`.
POSSESSION_END = "end"      # the ball changes hands; a new possession starts on the next row
POSSESSION_RESET = "reset"  # same offense, fresh clock (an offensive rebound)


def possession_boundary(event: str, etype: str, result: str):
    """Whether a cleaned row ends the possession, only resets the clock, or neither.

    Read off the cleaned-data semantics, which are the authority: section 7 removed the
    ``possession`` column, and no flip-result set survives anywhere in the repo. What is left is
    the ``result`` token, which the cleaner already uses to say what happened to the ball:

      * ``cop`` — change of possession. A turnover, a defensive or team-defensive rebound, or an
        offensive foul (``determine_foul_result`` maps it there). Ends the possession.
      * a made shot — ``event="shot", result="made"``. Ends it. Free throws are normalized under
        ``shot`` too, so a made free throw ends it as well.
      * a rebound that is not ``cop`` — an offensive or team-offensive board. The offense keeps
        the ball but the shot clock restarts, so the clock resets without the possession ending.

    A defensive foul (``free throw`` / ``free throw op``), a common foul (``nothing``), a loose
    ball foul (``op``), a missed shot, a block and an assist all leave the possession running.

    **Known inaccuracy.** The cleaned data carries no free-throw index — ``num``/``outof`` are a
    v3 item — so a made *first* free throw of a two-shot trip is read as ending the possession,
    and the second free throw's clock reads ~0. It is confined to free-throw rows, where the shot
    clock is off and the value means nothing; live play resumes correctly either way, because the
    last made free throw ends the possession and a missed one leaves the rebound to decide.
    """
    if result == "cop":
        return POSSESSION_END
    if event == "shot" and result == "made":
        return POSSESSION_END
    if event == "rebound":
        return POSSESSION_RESET
    return None


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


def _period_start(t: float) -> float:
    """Clock at the start of the period containing ``t``.

    Where a possession is taken to begin when there is no earlier evidence -- the game's first
    row, and the first row after each period boundary. Anchoring on the period rather than on
    that row's own timestamp is the truthful reading: play resumes at the tip or the inbound, so
    an event ten seconds into a quarter is ten seconds into its possession, not zero.
    """
    if t < REGULATION:
        return int(t // PERIOD_LENGTH) * PERIOD_LENGTH
    return REGULATION + int((t - REGULATION) // OT_LENGTH) * OT_LENGTH


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

class GameStateScan:
    """Incremental form of :func:`derive_game_state`: feed rows in order, get each row's raw state.

    ``derive_game_state`` is a thin loop over this, so preprocessing (a whole game at once) and
    the simulator (one row at a time, as the rollout appends) execute literally the same code —
    the train/inference parity this module promises holds by construction rather than by
    convention. State is inclusive of the row just fed.
    """

    __slots__ = ("home_pts", "away_pts", "fouls_home", "fouls_away", "cur_period", "poss_start")

    def __init__(self) -> None:
        self.home_pts = 0
        self.away_pts = 0
        self.fouls_home = 0
        self.fouls_away = 0
        self.cur_period = -1
        # Always overwritten on the first row (period -1 never matches a real period), but a
        # number rather than None so a misuse is a wrong value, not a TypeError.
        self.poss_start = 0.0

    def step(self, row) -> tuple:
        """Fold one event row in; return its raw state values in ``GAME_STATE_KEYS`` order."""
        t = float(row.get("time") or 0.0)
        period = _period_index(t)
        if period != self.cur_period:       # per-period team-foul reset (controller parity)
            self.fouls_home = self.fouls_away = 0
            self.cur_period = period
            # A new period starts a new possession, anchored at the buzzer rather than at
            # whenever the first event of the period happens to land. Covers the game's first
            # row too, since period -1 never matches.
            self.poss_start = _period_start(t)

        boundary = None
        event = _norm(row.get("event"))
        if event not in _SKIP_EVENTS:
            player = _norm(row.get("player"))
            home_roster = _roster(row.get("roster_home"))
            away_roster = _roster(row.get("roster_away"))
            team = ("home" if player in home_roster
                    else "away" if player in away_roster else None)
            etype = _norm(row.get("type"))
            result = _norm(row.get("result"))
            boundary = possession_boundary(event, etype, result)
            if event == "shot" and result == "made":
                # Through zones.points_for_shot, which simulation/box_score.py also calls, so
                # the trained score feature and the box score cannot drift apart.
                pts = points_for_shot(etype)
                if team == "home":
                    self.home_pts += pts
                elif team == "away":
                    self.away_pts += pts
            elif event == "foul" and etype not in NON_TEAM_FOUL_TYPES:
                if team == "home":
                    self.fouls_home += 1
                elif team == "away":
                    self.fouls_away += 1

        # Measured BEFORE the boundary is applied: a row that ends a possession belongs to the
        # possession it ends, and reports how long that one lasted. The next row starts at zero.
        poss_clock = t - self.poss_start
        if boundary is not None:
            self.poss_start = t

        return (self.home_pts - self.away_pts,
                self.home_pts + self.away_pts,
                period,
                _period_end(t) - t,
                self.fouls_home,
                self.fouls_away,
                poss_clock)


def derive_game_state(rows) -> dict[str, np.ndarray]:
    """Per-row running game state for one game's ordered event ``rows`` (inclusive of each row).

    ``rows`` is an ordered iterable of dict-like events carrying at least ``event, player,
    type, result, time`` and the per-row ``roster_home`` / ``roster_away`` snapshots — the
    shape produced by both the cleaned data and ``GameSimulator`` history. Scoring follows the
    box-score semantics (made shot -> 2 / 3 / 1 by type; scoring team = the player's side by
    roster membership). Returns raw (un-normalized) ``(N,)`` float arrays keyed by
    ``GAME_STATE_KEYS``. A thin driver over :class:`GameStateScan`.
    """
    rows = list(rows)
    n = len(rows)
    out = {k: np.zeros((n,), dtype=np.float32) for k in GAME_STATE_KEYS}
    scan = GameStateScan()
    for i, row in enumerate(rows):
        for k, v in zip(GAME_STATE_KEYS, scan.step(row)):
            out[k][i] = v
    return out


def normalize_game_state(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Clip + scale each raw game-state array by its fixed constant (no train-fit stats)."""
    out = {}
    for k in GAME_STATE_KEYS:
        lo, hi, div = _NORM[k]
        out[k] = (np.clip(raw[k], lo, hi) / div).astype(np.float32)
    return out


def normalize_game_state_row(raw_row) -> tuple:
    """Normalize ONE row's raw values (``GameStateScan.step`` output) to ``GAME_STATE_KEYS`` order.

    Deliberately routed through :func:`normalize_game_state` on length-1 arrays rather than
    hand-rolled: the batch path stores raw values into a float32 array before clipping, so going
    through the same float32 clip/divide is what makes the incremental simulator path bit-identical
    to preprocessing. Six tiny numpy ops, paid once per appended event (not once per head call).
    """
    one = {k: np.array([v], dtype=np.float32) for k, v in zip(GAME_STATE_KEYS, raw_row)}
    out = normalize_game_state(one)
    return tuple(out[k][0] for k in GAME_STATE_KEYS)


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
    """Keras Inputs for the game-state scalars (shape (SEQ, 1) each)."""
    from keras import Input  # local import: keep the preprocessing helpers TF-free.

    return {k: Input(shape=(SEQ, 1), dtype="float32", name=k) for k in GAME_STATE_KEYS}


def game_state_projections(inputs: dict) -> list:
    """Dense(16) projection of each game-state scalar (mirrors the time/team-scalar projections)."""
    from keras import layers  # local import: keep the preprocessing helpers TF-free.

    return [layers.Dense(16, name=f"{k}_proj")(inputs[k]) for k in GAME_STATE_KEYS]


# =====================
# --- Measurement   ---
# =====================
#
# ``poss_clock`` is derived, not recorded: there is no shot clock in the source data, so the
# rule in :func:`possession_boundary` is an inference from the event stream and could be wrong
# in a way no unit test would catch. Real basketball has a known pace, so counting possessions
# per team per game against it is the check that bites. Follows the ``python -m zones`` pattern:
# a cheap TF-free CPU pass, re-runnable whenever the rule is touched.
#
#     python -m models.game_state_features --seasons 2003,2013,2023

# NBA pace has ranged roughly 89-101 possessions per team per game across the 21 cleaned seasons
# (slowest in the mid-2010s, fastest in the early 2020s). A derivation landing outside this band
# is not a pace observation -- it means the rule is counting the wrong rows.
PACE_GATE = (85.0, 108.0)


def _percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[k]


def _scan_season(path):
    """Per-game possession counts and per-possession durations for one cleaned season file."""
    import pandas as pd

    df = pd.read_csv(path)
    per_game_ends, lengths, clipped, rows = [], [], 0, 0
    for _, game in df.groupby("game_id", sort=False):
        records = game.to_dict("records")
        scan = GameStateScan()
        ends = 0
        for row in records:
            *_, clock = scan.step(row)
            rows += 1
            if clock > 24.0:
                clipped += 1
            event = _norm(row.get("event"))
            if event in _SKIP_EVENTS:
                continue
            if possession_boundary(event, _norm(row.get("type")),
                                   _norm(row.get("result"))) == POSSESSION_END:
                ends += 1
                lengths.append(clock)
        per_game_ends.append(ends)
    return per_game_ends, lengths, clipped, rows


def _main(argv=None) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="python -m models.game_state_features",
        description="Validate the derived possession clock against real NBA pace.",
    )
    parser.add_argument("--seasons", default="2003,2013,2023",
                        help="comma-separated season labels, as in data/season<YYYY>.csv")
    parser.add_argument("--data-dir", default="./data",
                        help="directory of cleaned season CSVs")
    args = parser.parse_args(argv)

    rows = []
    failures = []
    for label in (x.strip() for x in args.seasons.split(",") if x.strip()):
        path = os.path.join(args.data_dir, f"season{label}.csv")
        if not os.path.isfile(path):
            print(f"WARNING: no cleaned file at {path}")
            continue
        print(f"scanning {label}: {path} ...", flush=True)
        ends, lengths, clipped, n = _scan_season(path)
        if not ends:
            failures.append(f"{label}: no games found")
            continue
        games = len(ends)
        per_team = sum(ends) / games / 2.0     # both teams' possessions end; pace is per team
        rows.append((label, games, per_team,
                     sum(lengths) / len(lengths) if lengths else float("nan"),
                     _percentile(lengths, 0.50), _percentile(lengths, 0.95),
                     100.0 * clipped / n if n else 0.0))
        lo, hi = PACE_GATE
        if not lo <= per_team <= hi:
            failures.append(
                f"{label}: {per_team:.1f} possessions per team per game, outside {lo}-{hi} "
                "-- the boundary rule is counting the wrong rows")

    if not rows:
        print("Nothing scanned.")
        return 1

    print()
    print(f"{'season':>8} {'games':>7} {'poss/team':>10} {'mean s':>8} {'p50 s':>7} "
          f"{'p95 s':>7} {'>24s':>7}")
    for label, games, per_team, mean, p50, p95, pct in rows:
        print(f"{label:>8} {games:>7} {per_team:>10.1f} {mean:>8.1f} {p50:>7.1f} "
              f"{p95:>7.1f} {pct:>6.1f}%")

    print()
    if failures:
        print("GATE FAILURES -- the possession rule is wrong, stop and fix it:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
