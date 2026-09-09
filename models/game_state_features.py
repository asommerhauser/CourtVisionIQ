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


# Fouls whose free throws leave the ball with the shooting team: a technical, and the
# "free throw op" family (personal take, transition take, flagrant-1). The trip does not end a
# possession, and the shot clock resumes rather than restarting.
_RETAINING_FOUL_RESULTS = {"free throw op"}
_RETAINING_FOUL_TYPES = {"technical"}
# Events that count as live play when deciding whether a foul was drawn on a made basket.
# Substitutions, timeouts and the game sentinels can sit between the basket and the foul.
_LIVE_EVENTS = {"shot", "rebound", "turnover", "block", "assist"}


def possession_boundary(event: str, etype: str, result: str):
    """Whether a cleaned row ends the possession, only resets the clock, or neither.

    Read off the cleaned-data semantics, which are the authority: section 7 removed the
    ``possession`` column, and no flip-result set survives anywhere in the repo. What is left is
    the ``result`` token, which the cleaner already uses to say what happened to the ball:

      * ``cop`` — change of possession. A turnover, a defensive or team-defensive rebound, or an
        offensive foul (``determine_foul_result`` maps it there). Ends the possession.
      * a made field goal — ``event="shot", result="made"`` with a zone type. Ends it.
      * a rebound that is not ``cop`` — an offensive or team-offensive board. The offense keeps
        the ball but the shot clock restarts, so the clock resets without the possession ending.

    A defensive foul (``free throw`` / ``free throw op``), a common foul (``nothing``), a loose
    ball foul (``op``), a missed shot, a block and an assist all leave the possession running.

    **Free throws are deliberately not decided here.** A made free throw usually does end a
    possession, but three cases say otherwise and none of them is visible in a single row, so
    :class:`GameStateScan` resolves the trip as a whole — see ``_resolve_free_throws``. Reading
    the shot row alone over-counted possessions by 12%: 109.4 per team per game in 2022-23,
    against 99.4 by the standard formula on the same file.
    """
    if result == "cop":
        return POSSESSION_END
    if event == "shot" and result == "made" and etype != "free throw":
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

    __slots__ = ("home_pts", "away_pts", "fouls_home", "fouls_away", "cur_period", "poss_start",
                 "ft_made_at", "ft_retains", "ft_after_basket", "prev_live", "poss_ends")

    def __init__(self) -> None:
        self.home_pts = 0
        self.away_pts = 0
        self.fouls_home = 0
        self.fouls_away = 0
        self.cur_period = -1
        # Always overwritten on the first row (period -1 never matches a real period), but a
        # number rather than None so a misuse is a wrong value, not a TypeError.
        self.poss_start = 0.0
        # Open free-throw trip (see _resolve_free_throws).
        self.ft_made_at = None          # time of the latest made free throw in the open trip
        self.ft_retains = False         # technical / flagrant / take: the shooting team keeps it
        self.ft_after_basket = False    # an and-1: the basket it followed already ended it
        self.prev_live = None           # (event, type, result) of the last live-play row
        # Possessions completed so far. Not a feature -- it is what the pace check counts, and it
        # lives here so the check counts the same events the clock resets on, by construction.
        self.poss_ends = 0

    def _track_free_throws(self, event, etype, result, t) -> None:
        """Accumulate what the open free-throw trip will need when it resolves.

        A foul row opens the bookkeeping: whether its free throws leave the ball with the
        shooting team, and whether it followed a made basket (an and-1). Each made free throw
        then records its time; the latest one is where the next possession starts, so a trip
        resolves once whatever its length, with no need for the ``num``/``outof`` index the
        cleaned data does not carry.

        A missed free throw clears the pending time: the trip's outcome is no longer settled by
        the trip, it is settled by the rebound that follows, which decides on its own.
        """
        if event == "foul":
            self.ft_retains = (result in _RETAINING_FOUL_RESULTS
                               or etype in _RETAINING_FOUL_TYPES)
            prev = self.prev_live
            self.ft_after_basket = bool(
                prev and prev[0] == "shot" and prev[1] != "free throw" and prev[2] == "made")
            return
        if event == "shot" and etype == "free throw":
            self.ft_made_at = t if result == "made" else None
            return
        if event in _LIVE_EVENTS:
            self.prev_live = (event, etype, result)

    def _resolve_free_throws(self) -> None:
        """Close an open free-throw trip, moving the possession start if the ball changed hands.

        Three trips do NOT end a possession, and each was measured over the 2022-23 file:

          * a trip that is not over -- the next row is another of its free throws. Ending on each
            made attempt counted a two-shot trip twice (~5.7 possessions per team per game).
          * an and-1. The made basket already ended the possession; counting the bonus shot again
            double-counted it (3.52 per team per game).
          * a technical, flagrant or take foul, where the shooting team keeps the ball and the
            shot clock resumes rather than restarting (0.86 per team per game).

        Together those were the whole 12% over-count.
        """
        if self.ft_made_at is not None and not (self.ft_retains or self.ft_after_basket):
            self.poss_start = self.ft_made_at
            self.poss_ends += 1
        self.ft_made_at = None
        self.ft_retains = False
        self.ft_after_basket = False

    def step(self, row) -> tuple:
        """Fold one event row in; return its raw state values in ``GAME_STATE_KEYS`` order."""
        t = float(row.get("time") or 0.0)
        period = _period_index(t)
        if period != self.cur_period:       # per-period team-foul reset (controller parity)
            self.fouls_home = self.fouls_away = 0
            self.cur_period = period
            # A new period starts a new possession, anchored at the buzzer rather than at
            # whenever the first event of the period happens to land. Covers the game's first
            # row too, since period -1 never matches. Not counted as a possession end: nobody
            # completed a trip, the clock simply restarts.
            self.poss_start = _period_start(t)
            self.ft_made_at = None
            self.ft_retains = False
            self.ft_after_basket = False

        boundary = None
        event = _norm(row.get("event"))
        etype_raw = _norm(row.get("type"))
        # A trip resolves on the first row that is not one of its own free throws, so its outcome
        # is known (last shot seen, foul kind, what preceded it) before this row's clock is read.
        if not (event == "shot" and etype_raw == "free throw"):
            self._resolve_free_throws()
        if event not in _SKIP_EVENTS:
            player = _norm(row.get("player"))
            home_roster = _roster(row.get("roster_home"))
            away_roster = _roster(row.get("roster_away"))
            team = ("home" if player in home_roster
                    else "away" if player in away_roster else None)
            etype = _norm(row.get("type"))
            result = _norm(row.get("result"))
            boundary = possession_boundary(event, etype, result)
            self._track_free_throws(event, etype, result, t)
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
            if boundary == POSSESSION_END:
                self.poss_ends += 1

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

    Scans each game (grouped, original row order preserved) and writes the seven normalized
    per-row arrays into ``cols`` aligned to ``df``'s positional index — the same layout as the
    encoded categorical / season columns. Needs no train mask or ``norm_stats`` (fixed-constant
    normalization). Mutates and returns ``cols``.
    """
    n = len(df)
    raw = {k: np.zeros((n,), dtype=np.float32) for k in GAME_STATE_KEYS}
    for pos, records in iter_game_rows(df):
        gs = derive_game_state(records)
        for k in GAME_STATE_KEYS:
            raw[k][pos] = gs[k]
    cols.update(normalize_game_state(raw))
    return cols


def iter_game_rows(df):
    """Yield ``(positions, records)`` for each game in file order, one game live at a time.

    The obvious form of this — ``df.to_dict("records")`` once, then ``np.where(game_ids == g)``
    per game — is quadratic and enormous at corpus scale, and both costs are paid on every
    preprocess. Over 21 seasons that is 13.4M row dicts held at once (tens of GB), plus one
    17ms full-array scan per game across 27k games, twice over between here and ``_build_split``:
    about sixteen minutes of pure index scanning before any work happens. Grouping once and
    mapping index labels to positions is linear and keeps only one game's dicts alive.
    """
    import pandas as pd  # local: the derivation helpers above stay import-light.

    pos_of = pd.Series(np.arange(len(df)), index=df.index)
    for _, game in df.groupby("game_id", sort=False):
        yield pos_of.loc[game.index].to_numpy(), game.to_dict("records")


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

# How far the derived count may sit from the box-score formula on the same file, in possessions
# per team per game. The two measure the same quantity by different routes -- one walks the event
# stream, the other is FGA - OREB + TOV + 0.44*FTA -- so they should agree closely; the 0.44
# coefficient is itself an approximation of free-throw trips, which is most of the slack here.
# A gate against published pace was the first attempt and was worse: those figures are normalized
# per 48 minutes and exclude playoffs, so the band had to be loose enough to hide real errors.
PACE_TOLERANCE = 3.0


def _percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[k]


def _scan_season(path):
    """Per-game possession counts, durations, and the formula's estimate, for one season file."""
    import pandas as pd

    df = pd.read_csv(path)
    per_game_ends, lengths, clipped, rows = [], [], 0, 0
    for _, game in df.groupby("game_id", sort=False):
        scan = GameStateScan()
        before = 0
        for row in game.to_dict("records"):
            *_, clock = scan.step(row)
            rows += 1
            if clock > 24.0:
                clipped += 1
            if scan.poss_ends != before:      # this row completed a possession
                lengths.append(clock)
                before = scan.poss_ends
        per_game_ends.append(scan.poss_ends)
    return per_game_ends, lengths, clipped, rows, _formula_possessions(df)


def _formula_possessions(df) -> float:
    """Possessions per team per game by the standard box-score estimate.

    ``FGA - OREB + TOV + 0.44 * FTA`` -- the accepted approximation, and the only reference the
    check needs that does not come from the rule being checked. Computed on the same file, so it
    tracks era, pace and this data's own quirks; published pace figures are normalized per 48
    minutes and exclude playoff games, so they are an anchor rather than a target. On 2022-23 this
    lands at 99.4 against a published 99.2.
    """
    shots = df[df["event"] == "shot"]
    fga = int((shots["type"] != "free throw").sum())
    fta = int((shots["type"] == "free throw").sum())
    rebounds = df[df["event"] == "rebound"]
    oreb = int(rebounds["type"].isin(("offensive", "team offensive")).sum())
    tov = int((df["event"] == "turnover").sum()) + int(
        ((df["event"] == "foul") & (df["type"] == "offensive")).sum())
    games = df["game_id"].nunique()
    return (fga - oreb + tov + 0.44 * fta) / games / 2.0


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
        ends, lengths, clipped, n, formula = _scan_season(path)
        if not ends:
            failures.append(f"{label}: no games found")
            continue
        games = len(ends)
        per_team = sum(ends) / games / 2.0     # both teams' possessions end; pace is per team
        rows.append((label, games, per_team, formula,
                     sum(lengths) / len(lengths) if lengths else float("nan"),
                     _percentile(lengths, 0.50), _percentile(lengths, 0.95),
                     100.0 * clipped / n if n else 0.0))
        if abs(per_team - formula) > PACE_TOLERANCE:
            failures.append(
                f"{label}: {per_team:.1f} possessions per team per game against {formula:.1f} "
                f"by the box-score formula, a gap of {per_team - formula:+.1f} "
                f"(tolerance {PACE_TOLERANCE}) -- the boundary rule is counting the wrong rows")

    if not rows:
        print("Nothing scanned.")
        return 1

    print()
    print(f"{'season':>8} {'games':>7} {'poss/team':>10} {'formula':>8} {'gap':>6} "
          f"{'mean s':>8} {'p50 s':>7} {'p95 s':>7} {'>24s':>7}")
    for label, games, per_team, formula, mean, p50, p95, pct in rows:
        print(f"{label:>8} {games:>7} {per_team:>10.1f} {formula:>8.1f} {per_team-formula:>+6.1f} "
              f"{mean:>8.1f} {p50:>7.1f} {p95:>7.1f} {pct:>6.1f}%")

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
