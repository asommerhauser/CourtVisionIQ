"""
zones.py — the shot-zone geometry: fifteen court zones from a raw shot's coordinates.

CourtVisionIQ 2.0 replaces the ``shot_type`` head's ``2pt`` / ``3pt`` binary with fifteen
spatial tokens (see ``docs/v2_planned_changes.md`` §3). Shot selection and shot efficiency are
spatial: with one make rate per player per point value, a rim finisher and a long-two shooter
look identical to the model, and rim make rate competes with long-mid frequency for a single
eFG dial. Per zone, the rebound head also learns where a miss came from, and the three-point
revolution becomes a learnable spatial fact instead of something absorbed into player
embeddings.

**Root level on purpose.** Both ``data_cleaner`` (which has no business importing from
``simulation/``) and the TF-free simulation layer import this. It cannot live in
``simulation/stats.py``, which already imports ``simulation/box_score.py`` — this module's
biggest consumer — so that would cycle.

The fold
--------
Both baskets fold onto one half court **oriented as the offense attacks**, so a team gets the
same token regardless of which way it is going. The raw frame is the full court:
``converted_x`` 0–50 and ``converted_y`` 0–94 feet, with the hoops at ``(25, 5.25)`` and
``(25, 88.75)``. The ``nx`` flip at the far basket is what makes left/right a real signal
(handedness, shooter court preference) rather than noise.

The marker wins
---------------
The raw ``type`` text (its ``3pt`` prefix) is the **authority** on point value; geometry only
picks the zone *within* the 2pt or 3pt family. Measured disagreement between derived geometry
and the raw marker is 0.42% in 2002-03 and 0.14% in 2022-23, so the two nearly always agree —
but where they do not, the marker is what the league scored and what the box score must match.

Run ``python -m zones --seasons 2003,2013,2023`` for the validation table (see §3): it prints
each zone's share of shots and make rate per era, plus the derived-vs-marker disagreement.
"""
from __future__ import annotations

import math

# --- Court geometry (feet, raw "converted" frame) ---------------------------------------------
COURT_LENGTH = 94.0
HALF_COURT = COURT_LENGTH / 2.0      # y < 47 is the "near" basket's end
CENTER_X = 25.0
HOOP_NEAR_Y = 5.25                   # hoop center, near end
HOOP_FAR_Y = 88.75                   # hoop center, far end

# --- Zone boundaries --------------------------------------------------------------------------
RIM_RADIUS = 4.0                     # rim: within 4 ft of the hoop
LANE_HALF_WIDTH = 8.0                # the lane is 16 ft wide
LANE_DEPTH = 14.0                    # free-throw line, measured from the hoop
MID_NEAR_FAR = 16.0                  # splits the short mid-range from the long
CORNER3_MAX_DEPTH = 8.75             # the straight-line portion of the arc
HEAVE_DISTANCE = 32.0                # beyond this a 3 is a heave, not a shot
NARROW_ANGLE = 25.0                  # degrees off straight-on: "top" / above-the-break
WIDE_ANGLE = 55.0                    # degrees: beyond this is the deep corner

# --- The fifteen tokens -----------------------------------------------------------------------
# Names align radially, so each mid-range zone pairs with the arc zone directly behind it.
ZONE_TOKENS = (
    "rim", "paint",
    "mid_base_l", "mid_base_r",
    "mid_wing_l", "mid_wing_r",
    "mid_corner_l", "mid_corner_r",
    "mid_top",
    "corner3_l", "corner3_r",
    "wing3_l", "wing3_r",
    "top3",
    "heave",
)

THREE_TOKENS = frozenset({"corner3_l", "corner3_r", "wing3_l", "wing3_r", "top3", "heave"})
TWO_TOKENS = frozenset(ZONE_TOKENS) - THREE_TOKENS

# Point value per token. This is the single lookup that ``simulation/box_score.py`` and
# ``models/game_state_features.py`` must BOTH read, or the trained score feature desyncs from
# the box score (pinned by tests/test_game_state_features.py).
ZONE_POINTS = {t: (3 if t in THREE_TOKENS else 2) for t in ZONE_TOKENS}

# Where a shot goes when its coordinates are unusable. Coverage is >=98.4% in every one of the
# 21 seasons (most under 0.1% null), so this path is rare by construction.
FALLBACK_TWO = "mid_base_l"
FALLBACK_THREE = "wing3_l"


def is_three(token: str) -> bool:
    """Is ``token`` a three-point zone? Unknown tokens are not — callers must not score them."""
    return token in THREE_TOKENS


def points_for(token: str) -> int:
    """Point value of a made shot from ``token``. Raises on an unknown token.

    Deliberately strict: a silent ``else: 2`` fallback is exactly how an unrecognized token
    would quietly score two points forever.
    """
    try:
        return ZONE_POINTS[token]
    except KeyError:
        raise KeyError(f"unknown shot zone {token!r}; expected one of {ZONE_TOKENS}") from None


def marker_is_three(type_text) -> bool:
    """Does the raw ``type`` text mark a three? The authority on point value.

    The raw vocabulary is free text ("3pt jump shot", "driving layup", "turnaround fadeaway");
    every three-point attempt is prefixed ``3pt``. Shared so the cleaner and this module cannot
    disagree about what counts as a three.
    """
    if type_text is None:
        return False
    text = str(type_text)
    if text.strip().lower() in ("nan", "none", "null", ""):
        return False
    return text.lower().startswith("3pt")


def fold(x: float, y: float, shot_distance=None) -> tuple[float, float, float, float]:
    """Fold a full-court ``(x, y)`` onto one half court, oriented as the offense attacks.

    Returns ``(nx, ny, dist, phi)``:

    * ``nx``   — signed offset from the hoop's center line; **positive is the shooter's left**
    * ``ny``   — depth from the hoop (can be slightly negative from under/behind the rim)
    * ``dist`` — straight-line distance from the hoop, feet
    * ``phi``  — degrees off straight-on; 0 is dead centre, positive is left

    **Which basket is being attacked** is decided by ``shot_distance`` when it is available:
    whichever hoop the league's own measurement agrees with. Deciding by which half the shot
    came from instead is right for essentially every normal shot but wrong for the one case
    that matters — a genuine backcourt heave, launched from the shooter's *own* end. Those
    exist in the data (2022-23 has rows like ``shot_distance=65`` at ``(20.3, 24.4)``), and the
    half-court rule folds them to the near hoop at 19.7 ft, landing a 5-15% prayer in ``top3``
    where it drags a real zone's make rate down. That is precisely what ``heave`` exists to
    prevent, so the token would have been unreachable for the shots that need it most.

    Falls back to the half-court rule when ``shot_distance`` is missing.

    ``ny`` is floored inside the ``atan2`` only, so a shot from behind the backboard still gets
    a sane angle instead of flipping into the opposite half plane.
    """
    x = float(x)
    y = float(y)
    dist_near = math.hypot(x - CENTER_X, y - HOOP_NEAR_Y)
    dist_far = math.hypot(x - CENTER_X, y - HOOP_FAR_Y)

    measured = _usable(shot_distance)
    if measured is None:
        near_end = y < HALF_COURT
    else:
        near_end = abs(dist_near - measured) <= abs(dist_far - measured)

    if near_end:
        ny = y - HOOP_NEAR_Y
        nx = x - CENTER_X
    else:                                    # far basket — the flip that makes left/right real
        ny = HOOP_FAR_Y - y
        nx = CENTER_X - x
    dist = math.hypot(nx, ny)
    phi = math.degrees(math.atan2(nx, max(ny, 1e-6)))
    return nx, ny, dist, phi


def _side(nx: float) -> str:
    """``l`` or ``r`` suffix. Dead centre (``nx == 0``) is arbitrarily left."""
    return "l" if nx >= 0 else "r"


def _two_point_zone(nx: float, ny: float, dist: float, phi: float) -> str:
    """Pick the zone for a shot the marker says is a two."""
    if dist <= RIM_RADIUS:
        return "rim"
    if abs(nx) <= LANE_HALF_WIDTH and ny <= LANE_DEPTH:
        return "paint"
    # Outside the paint from here down.
    if abs(phi) <= NARROW_ANGLE:
        return "mid_top"
    if dist < MID_NEAR_FAR:
        return f"mid_base_{_side(nx)}"
    if abs(phi) <= WIDE_ANGLE:
        return f"mid_wing_{_side(nx)}"
    return f"mid_corner_{_side(nx)}"


def _three_point_zone(nx: float, ny: float, dist: float, phi: float) -> str:
    """Pick the zone for a shot the marker says is a three.

    ``heave`` is its own token rather than folded into ``top3`` because heaves are 0.3–0.4% of
    shots at 5–15% — an order of magnitude worse than a real three — and would otherwise drag
    every above-the-break make rate down by about a point.
    """
    if dist >= HEAVE_DISTANCE:
        return "heave"
    if ny <= CORNER3_MAX_DEPTH:              # the straight-line portion of the arc
        return f"corner3_{_side(nx)}"
    if abs(phi) > NARROW_ANGLE:
        return f"wing3_{_side(nx)}"
    return "top3"


def _usable(value) -> float | None:
    """Coerce a raw cell to a float, or ``None`` if it is missing/blank/NaN."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() in ("nan", "none", "null"):
            return None
        try:
            value = float(text)
        except ValueError:
            return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _zone_from_distance(dist: float, three: bool) -> str:
    """Coordinate-free fallback: the little a bare ``shot_distance`` can honestly tell us.

    Distance alone fixes no angle, so it can only separate the radial bands and otherwise
    degrades to the same defaults as :func:`zone_for`'s last resort. Kept deliberately coarse
    rather than inventing an angle we do not have.
    """
    if three:
        return "heave" if dist >= HEAVE_DISTANCE else FALLBACK_THREE
    if dist <= RIM_RADIUS:
        return "rim"
    if dist <= LANE_DEPTH:
        return "paint"
    return FALLBACK_TWO


def zone_for(x=None, y=None, *, three: bool, shot_distance=None) -> str:
    """The zone token for one shot. ``three`` comes from the raw marker and is authoritative.

    Resolution order, best information first:

    1. usable ``(x, y)`` — the real geometry;
    2. usable ``shot_distance`` — radial bands only (see :func:`_zone_from_distance`);
    3. neither — ``mid_base_l`` for a two, ``wing3_l`` for a three.

    Geometry never overrides ``three``: a shot the league scored as a two that lands beyond the
    arc stays a two, in the nearest two-point zone.
    """
    fx, fy = _usable(x), _usable(y)
    if fx is not None and fy is not None:
        nx, ny, dist, phi = fold(fx, fy, shot_distance)
        return _three_point_zone(nx, ny, dist, phi) if three \
            else _two_point_zone(nx, ny, dist, phi)

    dist = _usable(shot_distance)
    if dist is not None:
        return _zone_from_distance(dist, three)

    return FALLBACK_THREE if three else FALLBACK_TWO


def geometry_says_three(x, y, shot_distance=None) -> bool | None:
    """Would the geometry alone call this a three? ``None`` when coordinates are unusable.

    Only used to *measure* agreement with the raw marker (the validation table below). It is
    never allowed to decide point value — see the module docstring.
    """
    fx, fy = _usable(x), _usable(y)
    if fx is None or fy is None:
        return None
    nx, ny, dist, _ = fold(fx, fy, shot_distance)
    if ny <= CORNER3_MAX_DEPTH:              # corner: the arc is a straight line at 22 ft
        return abs(nx) >= 22.0
    return dist >= 23.75                     # above the break


# =============================================================================================
# --- Validation table (python -m zones)
# =============================================================================================

def _season_of(path) -> int | None:
    """Season label for a raw master file, matching ``data_cleaner``: first year found, +1."""
    import csv
    import re

    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            match = re.search(r"\d{4}", str(row.get("data_set", "")))
            return int(match.group()) + 1 if match else None
    return None


def _scan(path) -> tuple[dict, int, int]:
    """Tally ``{zone: [attempts, makes]}`` plus (disagreements, coord-covered) for one file."""
    import csv

    tally: dict[str, list[int]] = {t: [0, 0] for t in ZONE_TOKENS}
    disagree = covered = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            if row.get("event_type") != "shot":
                continue
            three = marker_is_three(row.get("type"))
            zone = zone_for(row.get("converted_x"), row.get("converted_y"),
                            three=three, shot_distance=row.get("shot_distance"))
            tally[zone][0] += 1
            if str(row.get("result", "")).strip().lower() == "made":
                tally[zone][1] += 1

            derived = geometry_says_three(row.get("converted_x"), row.get("converted_y"),
                                          row.get("shot_distance"))
            if derived is not None:
                covered += 1
                disagree += derived != three
    return tally, disagree, covered


def _print_table(results: dict) -> None:
    seasons = sorted(results)
    width = max(len(t) for t in ZONE_TOKENS) + 2
    header = "zone".ljust(width) + "".join(f"{s:>18}" for s in seasons)
    print(header)
    print("-" * len(header))
    for token in ZONE_TOKENS:
        line = token.ljust(width)
        for season in seasons:
            tally, total = results[season]["tally"], results[season]["shots"]
            att, made = tally[token]
            share = 100.0 * att / total if total else 0.0
            rate = 100.0 * made / att if att else 0.0
            line += f"{share:>8.1f}% @{rate:>5.1f}%"
        print(line)
    print("-" * len(header))

    line = "3PA share".ljust(width)
    for season in seasons:
        tally, total = results[season]["tally"], results[season]["shots"]
        threes = sum(tally[t][0] for t in ZONE_TOKENS if is_three(t))
        line += f"{100.0 * threes / total if total else 0.0:>16.1f}%"
    print(line)

    line = "marker disagree".ljust(width)
    for season in seasons:
        r = results[season]
        pct = 100.0 * r["disagree"] / r["covered"] if r["covered"] else 0.0
        line += f"{pct:>16.2f}%"
    print(line)

    line = "coord coverage".ljust(width)
    for season in seasons:
        r = results[season]
        pct = 100.0 * r["covered"] / r["shots"] if r["shots"] else 0.0
        line += f"{pct:>16.1f}%"
    print(line)


def _check_gates(results: dict) -> list[str]:
    """The §3 gates. Returns the failures; empty means the geometry looks right."""
    failures = []
    for season in sorted(results):
        tally = results[season]["tally"]

        missing = [t for t in ZONE_TOKENS if tally[t][0] == 0]
        if missing:
            failures.append(f"{season}: no shots at all in {missing}")

        def rate(token):
            att, made = tally[token]
            return made / att if att else 0.0

        # corner3 > wing3 > top3, in every era.
        corner = (rate("corner3_l") + rate("corner3_r")) / 2
        wing = (rate("wing3_l") + rate("wing3_r")) / 2
        if not (corner > wing > rate("top3")):
            failures.append(
                f"{season}: expected corner3 > wing3 > top3, got "
                f"{corner:.3f} / {wing:.3f} / {rate('top3'):.3f}"
            )

        # Left/right volumes should be near-symmetric (within a fifth of each other).
        for left, right in (("mid_base_l", "mid_base_r"), ("mid_wing_l", "mid_wing_r"),
                            ("corner3_l", "corner3_r"), ("wing3_l", "wing3_r")):
            a, b = tally[left][0], tally[right][0]
            if a and b and not (0.8 <= a / b <= 1.25):
                failures.append(f"{season}: {left}/{right} volumes lopsided ({a} vs {b})")

        if results[season]["covered"]:
            pct = 100.0 * results[season]["disagree"] / results[season]["covered"]
            if pct >= 1.0:
                failures.append(f"{season}: derived-vs-marker disagreement {pct:.2f}% (want <1%)")
    return failures


def _main(argv=None) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="python -m zones",
        description="Print the shot-zone validation table (docs/v2_planned_changes.md §3).",
    )
    parser.add_argument("--seasons", default="2003,2013,2023",
                        help="comma-separated season labels, as in data/season<YYYY>.csv")
    parser.add_argument("--raw", default="./RawData/MasterFiles",
                        help="directory of raw combined-stats master files")
    args = parser.parse_args(argv)

    wanted = {int(s) for s in args.seasons.split(",") if s.strip()}
    files = sorted(f for f in os.listdir(args.raw)
                   if f.endswith(".csv") and "Truncated" not in f)

    results: dict[int, dict] = {}
    for name in files:
        path = os.path.join(args.raw, name)
        season = _season_of(path)
        if season not in wanted:
            continue
        print(f"scanning {season}: {name} ...", flush=True)
        tally, disagree, covered = _scan(path)
        shots = sum(a for a, _ in tally.values())
        results[season] = {"tally": tally, "shots": shots,
                           "disagree": disagree, "covered": covered}

    missing = wanted - set(results)
    if missing:
        print(f"\nWARNING: no raw file found for season(s) {sorted(missing)}")
    if not results:
        print("Nothing scanned.")
        return 1

    print()
    _print_table(results)

    failures = _check_gates(results)
    print()
    if failures:
        print("GATE FAILURES — the geometry is wrong, stop and fix it:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
