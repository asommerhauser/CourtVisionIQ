"""
Per-player on-court state: stint seconds, seconds played, personal fouls.

The roster encoder takes exactly one number per on-court player today -- days of rest, which is
constant for the whole game. Everything that actually drives a rotation is invisible: how long
this player has been on the floor, how many minutes he has already played, how much foul trouble
he is in. Player minutes are the single largest remaining box-score error, and they are currently
produced by a timer with a dial on it, so these three are the inputs that let the model produce
them instead.

Three roster-parallel scalars per side, each ``(SEQ, ROSTER_SIZE)`` and aligned slot-for-slot
with ``home_roster`` / ``away_roster``, the same shape ``rest_home`` / ``rest_away`` already use:

  * ``stint_seconds``  -- seconds since this player last came on;
  * ``played_seconds`` -- seconds on the floor so far this game;
  * ``court_fouls``    -- personal fouls charged so far.

Normalization is by **fixed constants** (below), not train-fit stats, for the same reason
``game_state_features`` uses them: the quantities have known natural scales, so there are no new
``norm_stats`` keys to persist and inference needs no loaded statistics. The single
:class:`LineupScan` is used by BOTH preprocessing (over the cleaned event rows) and the simulator
(over ``self.history``), so train/inference parity holds by construction.

State at row *i* is inclusive of row *i*: minutes are credited up to that row's clock, and a foul
on that row is already counted.

This module is TF-free and imports nothing from ``simulation`` -- ``simulation/__init__.py``
imports ``GameSimulator``, so a single import from that package would pull in TensorFlow and make
the measurement pass at the bottom of this file unrunnable without CUDA. That is the same
reasoning that puts ``zones.py`` at root level.
"""
from __future__ import annotations

import ast

import numpy as np

from config import ROSTER_SIZE
from models.game_state_features import iter_game_rows

# Technicals are bench/team fouls and do not count toward the personal-foul disqualification --
# the same rule ``simulation/controller.py:_charge_foul`` applies, and the reason this set lives
# here rather than being spelled out twice.
NON_PERSONAL_FOUL_TYPES = {"technical"}

# Non-play frames carry no state contribution (mirrors game_state_features._SKIP_EVENTS).
_SKIP_EVENTS = {"start", "end", "none", "PAD", "UNK", ""}

# Roster-parallel per-player inputs, one pair per scalar. Order is stable, and is the order
# LineupScan.step returns them in.
ROSTER_STATE_KEYS = (
    "stint_seconds_home", "stint_seconds_away",
    "played_seconds_home", "played_seconds_away",
    "court_fouls_home", "court_fouls_away",
)

# Fixed normalization: (clip_lo, clip_hi, divisor), chosen so typical values land ~[0, 1].
_NORM = {
    # A stint past 20 minutes is a starter who has not come off; past that the exact value stops
    # carrying anything the model can act on, and the clip keeps a data gap from reading as an
    # extreme input.
    "stint_seconds_home": (0.0, 1200.0, 600.0),
    "stint_seconds_away": (0.0, 1200.0, 600.0),
    # A full regulation game is 2880s; the divisor puts a 24-minute night at 1.0.
    "played_seconds_home": (0.0, 3600.0, 1440.0),
    "played_seconds_away": (0.0, 3600.0, 1440.0),
    # Six is disqualification (config.FOUL_OUT_LIMIT), so three -- foul trouble -- is 1.0.
    "court_fouls_home": (0.0, 6.0, 3.0),
    "court_fouls_away": (0.0, 6.0, 3.0),
}


def _roster(value) -> list:
    """Coerce a roster cell (a list, or its list-literal string form) to a list of names."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def _norm_str(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


# =====================
# --- Derivation    ---
# =====================

class LineupScan:
    """Feed rows in order, get each row's per-player on-court state.

    Reads the roster snapshot off each row rather than folding substitution rows: the cleaner
    repairs the snapshot and emits a substitution for every real change, so the two agree, and
    reading the snapshot is what the box score already does. Simulator history rows carry the
    same two columns, so the simulator drives this identically.

    Minutes are credited over the interval BETWEEN rows, to the five that held the floor for it
    -- the same accounting as ``simulation/box_score.py``. A player's stint therefore starts at
    the clock of the row he first appears on.
    """

    __slots__ = ("played", "entered_at", "fouls", "prev_time", "prev_five")

    def __init__(self) -> None:
        self.played: dict[str, float] = {}
        self.entered_at: dict[str, float] = {}
        self.fouls: dict[str, int] = {}
        self.prev_time = None
        self.prev_five: tuple[list, list] = ([], [])

    def step(self, row) -> tuple:
        """Fold one row in; return six roster-parallel lists in ``ROSTER_STATE_KEYS`` order."""
        t = float(row.get("time") or 0.0)
        home = _roster(row.get("roster_home"))
        away = _roster(row.get("roster_away"))

        # --- Minutes: credit the lineup that held the floor over the elapsed interval. ---
        if self.prev_time is not None and t > self.prev_time:
            elapsed = t - self.prev_time
            for name in (*self.prev_five[0], *self.prev_five[1]):
                self.played[name] = self.played.get(name, 0.0) + elapsed

        # --- Stints: anyone newly on the floor starts one here, anyone off it loses his. ---
        on_court = set(home) | set(away)
        for name in on_court:
            if name not in self.entered_at:
                self.entered_at[name] = t
        for name in [n for n in self.entered_at if n not in on_court]:
            del self.entered_at[name]

        # --- Fouls: inclusive of this row, matching the game-state features' convention. ---
        event = _norm_str(row.get("event"))
        if event == "foul" and _norm_str(row.get("type")) not in NON_PERSONAL_FOUL_TYPES:
            fouler = _norm_str(row.get("player"))
            if fouler and fouler not in _SKIP_EVENTS:
                self.fouls[fouler] = self.fouls.get(fouler, 0) + 1

        self.prev_time = t
        self.prev_five = (home, away)

        return (
            [t - self.entered_at.get(p, t) for p in home],
            [t - self.entered_at.get(p, t) for p in away],
            [self.played.get(p, 0.0) for p in home],
            [self.played.get(p, 0.0) for p in away],
            [float(self.fouls.get(p, 0)) for p in home],
            [float(self.fouls.get(p, 0)) for p in away],
        )


def derive_lineup_state(rows) -> dict[str, np.ndarray]:
    """Per-row per-player on-court state for one game's ordered event ``rows``.

    Returns raw (un-normalized) ``(N, ROSTER_SIZE)`` float arrays keyed by ``ROSTER_STATE_KEYS``,
    slot-aligned to that row's ``roster_home`` / ``roster_away``. A thin driver over
    :class:`LineupScan`, exactly as ``derive_game_state`` is over ``GameStateScan``.
    """
    rows = list(rows)
    n = len(rows)
    out = {k: np.zeros((n, ROSTER_SIZE), dtype=np.float32) for k in ROSTER_STATE_KEYS}
    scan = LineupScan()
    for i, row in enumerate(rows):
        for k, values in zip(ROSTER_STATE_KEYS, scan.step(row)):
            for j, v in enumerate(values[:ROSTER_SIZE]):
                out[k][i, j] = v
    return out


def normalize_lineup_state(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Clip + scale each raw array by its fixed constant (no train-fit stats)."""
    out = {}
    for k in ROSTER_STATE_KEYS:
        lo, hi, div = _NORM[k]
        out[k] = (np.clip(raw[k], lo, hi) / div).astype(np.float32)
    return out


def normalize_lineup_state_row(raw_row) -> tuple:
    """Normalize ONE row's raw values (``LineupScan.step`` output) to ``ROSTER_STATE_KEYS`` order.

    Routed through :func:`normalize_lineup_state` on length-1 arrays rather than hand-rolled, for
    the reason ``game_state_features.normalize_game_state_row`` gives: the batch path stores raw
    values into a float32 array before clipping, and going through the same float32 clip/divide
    is what makes the incremental simulator path bit-identical to preprocessing.

    Each entry of ``raw_row`` is a per-slot list, which may be shorter than ``ROSTER_SIZE``; the
    remaining slots stay at zero, matching the PAD slots the roster encoder masks out anyway.
    """
    one = {}
    for k, values in zip(ROSTER_STATE_KEYS, raw_row):
        buf = np.zeros((1, ROSTER_SIZE), dtype=np.float32)
        buf[0, :len(values)] = values[:ROSTER_SIZE]
        one[k] = buf
    out = normalize_lineup_state(one)
    return tuple(out[k][0] for k in ROSTER_STATE_KEYS)


# =====================
# --- Preprocessing ---
# =====================

def merge_rotation_features(df, cols) -> dict:
    """Derive, normalize, and merge the roster-parallel arrays into ``cols``.

    Scans each game (grouped, original row order preserved) and writes the six normalized
    ``(N, ROSTER_SIZE)`` arrays into ``cols`` aligned to ``df``'s positional index -- the layout
    ``merge_game_state_features`` uses, at the shape ``merge_season_features`` uses for rest.
    Needs no train mask and no ``norm_stats`` (fixed-constant normalization). Mutates ``cols``.
    """
    n = len(df)
    raw = {k: np.zeros((n, ROSTER_SIZE), dtype=np.float32) for k in ROSTER_STATE_KEYS}
    for pos, records in iter_game_rows(df):
        ls = derive_lineup_state(records)
        for k in ROSTER_STATE_KEYS:
            raw[k][pos] = ls[k]
    cols.update(normalize_lineup_state(raw))
    return cols


def append_rotation_batches(batches, cols, idx, n, SEQ) -> None:
    """Pad/stack the roster-parallel arrays for one game (mirrors append_season_batches)."""
    for k in ROSTER_STATE_KEYS:
        buf = np.zeros((SEQ, ROSTER_SIZE), dtype=np.float32)
        buf[:n] = cols[k][idx]
        batches[k].append(buf)


# =====================
# --- Model graph   ---
# =====================

def make_rotation_inputs(SEQ) -> dict:
    """Keras Inputs for the roster-parallel scalars (shape (SEQ, ROSTER_SIZE) each)."""
    from keras import Input  # local import: keep the preprocessing helpers TF-free.

    return {k: Input(shape=(SEQ, ROSTER_SIZE), dtype="float32", name=k)
            for k in ROSTER_STATE_KEYS}


def side_scalars(rest, rotation_inputs, side: str) -> list:
    """The per-player scalars for one side, in the order the roster encoder expects them.

    Rest first, so the single-scalar ordering the encoder had before 2.0 is a prefix of this one
    and the meaning of scalar 0 does not move. The rest follow ``ROSTER_STATE_KEYS``.
    """
    return [rest] + [rotation_inputs[f"{stem}_{side}"]
                     for stem in ("stint_seconds", "played_seconds", "court_fouls")]


# Number of per-player scalars the roster encoder is built for: rest plus the three above.
# Feeds RosterEncoderParams.num_scalars, whose kernel shape then encodes the count.
NUM_ROSTER_SCALARS = 4


# =====================
# --- Measurement   ---
# =====================
#
# Everything above is derived from the on-court five, and the five is the quantity the raw data
# is least reliable about: its lineup snapshots flicker and it omits every quarter-break
# substitution (see ``data_cleaner._repair_fives``). Unit tests did not find either of the two
# real bugs at Gate B; one number against an independent source found both. So this is that
# number for the rotation work, and like ``python -m zones`` and
# ``python -m models.game_state_features`` it is a cheap TF-free CPU pass, re-runnable whenever
# the derivation is touched:
#
#     python -m models.rotation_features --seasons 2003,2013,2023
#
# Check 2 is the one that bites. Minutes can be read two ways off the same file -- from each
# row's roster snapshot, and by folding the substitution rows forward from the opening five --
# and after the repair those are the same quantity by independent routes, so they must agree.
# Before the repair they disagreed on 156.6 rows a game.

# Rows a game on which the two routes may disagree. Zero is the honest target; the allowance
# exists so a single malformed game in twenty-one seasons does not fail the gate outright.
LINEUP_TOLERANCE = 0.1
# Substitutions per team per game. The 2022-23 cleaned file carries 23.2 before the repair and
# gains the ~3.9 a team the quarter breaks were never recording; the band is wide enough to hold
# every era (the older files substitute less) and narrow enough that a doubling fails.
SUBS_PER_TEAM_LO, SUBS_PER_TEAM_HI = 15.0, 35.0
# Rows a side is not five, as a percentage. The source itself omits one half of a handful of
# substitutions a season -- four null ``entered`` and two null ``left`` in 2002-03, none at all
# in 2022-23 -- and a lineup the data records as four cannot be repaired without inventing the
# fifth player. So this is an allowance for a known data defect, not slack.
#
# Measured after the repair: 0.0021% in 2002-03, 0.0027% in 2012-13, 0.0000% in 2022-23, which
# is the source's own 9 / 13 / 0 short rows carried a row or two further. The allowance is four
# times the worst of those and no more. It was 0.05 first, calibrated on an 86-game slice, and
# the full season came in at 0.0573% -- a real defect that a threshold set from a small sample
# had very nearly waved through. Widening this is almost never the right response to a failure.
SHORT_LINEUP_PCT = 0.01


def _fold_subs(rows):
    """On-court five per row by folding substitution rows forward from the opening snapshot.

    The independent route. It never reads a roster column after the first row, so it agrees with
    :class:`LineupScan` only if every lineup change in the file is explained by a substitution.

    A substitution row carrying ``none`` on one side is the cleaner's record of a raw row that
    named only one half of the swap: ``none`` as the outgoing player means someone came on
    without anyone leaving, ``none`` as the incoming means someone left without a replacement.
    Those change the lineup's SIZE rather than its membership, and the fold has to follow or it
    disagrees for the rest of the game.
    """
    home = list(_roster(rows[0].get("roster_home")))
    away = list(_roster(rows[0].get("roster_away")))
    out = []
    for row in rows:
        if _norm_str(row.get("event")) == "substitution":
            outgoing = _norm_str(row.get("player"))
            incoming = _norm_str(row.get("secondary_player"))
            if outgoing == "none":
                # Nobody left, so membership says nothing about the side -- the row's own
                # home/away indicator is the only thing that does.
                (home if _as_int(row.get("home/away")) == 1 else away).append(incoming)
            else:
                for five in (home, away):
                    if outgoing in five:
                        if incoming == "none":
                            five.remove(outgoing)
                        else:
                            five[five.index(outgoing)] = incoming
                        break
        out.append((list(home), list(away)))
    return out


def _as_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _scan_game(rows):
    """One game: (disagreeing, short, duplicated, substitutions, played seconds, end clock)."""
    scan = LineupScan()
    folded = _fold_subs(rows)
    disagree = short = duplicated = 0
    for i, row in enumerate(rows):
        scan.step(row)
        home = _roster(row.get("roster_home"))
        away = _roster(row.get("roster_away"))
        if set(home) != set(folded[i][0]) or set(away) != set(folded[i][1]):
            disagree += 1
        if len(home) != ROSTER_SIZE or len(away) != ROSTER_SIZE:
            short += 1
        # A player in two slots at once. Nothing in the raw data does this -- it is what a
        # substitution applied against the wrong lineup produces, and because every comparison
        # here is by membership it is invisible until the five silently grows to six.
        if len(set(home)) != len(home) or len(set(away)) != len(away):
            duplicated += 1
    subs = sum(1 for r in rows if _norm_str(r.get("event")) == "substitution")
    return disagree, short, duplicated, subs, scan.played, float(rows[-1].get("time") or 0.0)


def _scan_season(path):
    import pandas as pd

    df = pd.read_csv(path, low_memory=False)
    games = rows_total = 0
    disagree = short = duplicated = subs = minute_failures = 0
    starters, rotation = [], []
    for _, game in df.groupby("game_id", sort=False):
        rows = game.to_dict("records")
        if not rows:
            continue
        games += 1
        rows_total += len(rows)
        d, sh, dup, s, played, end = _scan_game(rows)
        disagree += d
        short += sh
        duplicated += dup
        subs += s
        # Ten players on the floor at every instant means total player-seconds is exactly ten
        # times the game's length -- an identity, not a band. It only holds where every lineup
        # is five, so a game carrying a short one is excluded rather than counted as a failure:
        # the source really does record four players there, and inventing a fifth would be
        # fabricating the very quantity this gate exists to check.
        if not sh and abs(sum(played.values()) - end * 10.0) > 1e-3:
            minute_failures += 1
        ordered = sorted(played.values(), reverse=True)
        if len(ordered) >= 10:
            starters.extend(v / 60.0 for v in ordered[:10])
        rotation.append(sum(1 for v in played.values() if v >= 600.0) / 2.0)
    return (games, rows_total, disagree, short, duplicated, subs, minute_failures,
            starters, rotation)


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def _main(argv=None) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="python -m models.rotation_features",
        description="Validate the derived on-court state against the substitution rows.",
    )
    parser.add_argument("--seasons", default="2003,2013,2023",
                        help="comma-separated season labels, as in data/season<YYYY>.csv")
    parser.add_argument("--data-dir", default="./data",
                        help="directory of cleaned season CSVs")
    args = parser.parse_args(argv)

    rows, failures = [], []
    for label in (x.strip() for x in args.seasons.split(",") if x.strip()):
        path = os.path.join(args.data_dir, f"season{label}.csv")
        if not os.path.isfile(path):
            print(f"WARNING: no cleaned file at {path}")
            continue
        print(f"scanning {label}: {path} ...", flush=True)
        (games, rows_total, disagree, short, duplicated, subs,
         minute_failures, starters, rotation) = _scan_season(path)
        if not games:
            failures.append(f"{label}: no games found")
            continue
        per_game_disagree = disagree / games
        per_team_subs = subs / games / 2.0
        short_pct = 100.0 * short / rows_total if rows_total else 0.0
        rows.append((label, games, per_game_disagree, per_team_subs, minute_failures,
                     short_pct, duplicated, _mean(starters), _mean(rotation)))

        if per_game_disagree > LINEUP_TOLERANCE:
            failures.append(
                f"{label}: {per_game_disagree:.2f} rows a game where folding the substitutions "
                f"disagrees with the roster snapshot (tolerance {LINEUP_TOLERANCE}) -- a lineup "
                f"change somewhere has no substitution row to explain it")
        if minute_failures:
            failures.append(
                f"{label}: {minute_failures} of {games} games have five a side throughout and "
                f"still do not total ten players' minutes -- the scan is losing time somewhere")
        if duplicated:
            failures.append(
                f"{label}: {duplicated} rows put one player in two slots at once -- a "
                f"substitution has been applied against a lineup that did not match it")
        if short_pct > SHORT_LINEUP_PCT:
            failures.append(
                f"{label}: {short_pct:.4f}% of rows carry a side that is not five "
                f"(allowance {SHORT_LINEUP_PCT}%) -- the source omits a handful of these a "
                f"season, but a jump means the repair is dropping players")
        if not SUBS_PER_TEAM_LO <= per_team_subs <= SUBS_PER_TEAM_HI:
            failures.append(
                f"{label}: {per_team_subs:.1f} substitutions per team per game, outside "
                f"[{SUBS_PER_TEAM_LO}, {SUBS_PER_TEAM_HI}]")

    if not rows:
        print("Nothing scanned.")
        return 1

    print()
    print(f"{'season':>8} {'games':>7} {'disagree/g':>11} {'subs/team':>10} {'min fails':>10} "
          f"{'short %':>9} {'dup rows':>9} {'top10 min':>10} {'10+ min':>8}")
    for label, games, disagree, subs, fails, short_pct, dup, top10, depth in rows:
        print(f"{label:>8} {games:>7} {disagree:>11.2f} {subs:>10.1f} {fails:>10} "
              f"{short_pct:>9.4f} {dup:>9} {top10:>10.1f} {depth:>8.1f}")
    print()
    print("top10 min is the mean of each game's ten highest per-player minutes (starters plus "
          "the first bench unit); 10+ min is players per team over ten minutes. Neither is "
          "gated -- they are reported so a rotation that collapses is visible.")

    print()
    if failures:
        print("GATE FAILURES -- the on-court derivation is wrong, stop and fix it:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
