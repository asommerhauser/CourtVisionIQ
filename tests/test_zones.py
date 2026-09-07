"""
Shot-zone geometry tests.

Pure geometry — no models, no data files, no TF. The three things that must hold:

  1. **Both baskets fold to the same token.** A team gets the same zone regardless of which
     way it is attacking, and left stays left after the flip.
  2. **Every boundary lands where the spec says** (docs/v2_planned_changes.md §3).
  3. **The raw marker wins.** Geometry picks the zone *within* the 2pt or 3pt family; it never
     decides point value.
"""
from __future__ import annotations

import math

import pytest

import zones
from zones import (
    CENTER_X, HOOP_FAR_Y, HOOP_NEAR_Y, ZONE_POINTS, ZONE_TOKENS,
    fold, is_three, marker_is_three, points_for, zone_for,
)


def near(x, y):
    """A shot at the near basket."""
    return zone_for(x, y, three=False)


def mirror(x, y):
    """The same shot at the far basket: mirrored through the court's centre point."""
    return 50.0 - x, 94.0 - y


# ===================================================================== #
# The token set itself
# ===================================================================== #

def test_there_are_fifteen_distinct_tokens():
    assert len(ZONE_TOKENS) == 15
    assert len(set(ZONE_TOKENS)) == 15


def test_every_token_has_a_point_value_and_the_split_is_right():
    assert set(ZONE_POINTS) == set(ZONE_TOKENS)
    threes = {t for t in ZONE_TOKENS if is_three(t)}
    assert threes == {"corner3_l", "corner3_r", "wing3_l", "wing3_r", "top3", "heave"}
    assert all(ZONE_POINTS[t] == 3 for t in threes)
    assert all(ZONE_POINTS[t] == 2 for t in set(ZONE_TOKENS) - threes)


def test_an_unknown_token_raises_rather_than_quietly_scoring_two():
    """The silent `else: 2` fallback is exactly how a bad token scores forever."""
    with pytest.raises(KeyError):
        points_for("mid_nowhere")
    assert is_three("mid_nowhere") is False


# ===================================================================== #
# The fold — both baskets, and left/right
# ===================================================================== #

@pytest.mark.parametrize("x,y", [
    (25.0, 5.25),      # dead on the rim
    (25.0, 12.0),      # paint, straight on
    (36.8, 10.0),      # baseline mid-range, left
    (13.4, 29.1),      # above the break, right
    (1.8, 9.9),        # deep corner
    (25.0, 30.0),      # top of the key
])
def test_both_baskets_map_to_the_same_token(x, y):
    three = zones.geometry_says_three(x, y)
    fx, fy = mirror(x, y)
    assert zone_for(x, y, three=three) == zone_for(fx, fy, three=three)


def test_the_far_basket_flip_preserves_left_and_right():
    """Without the nx flip, left/right would be noise instead of a real signal."""
    left_near = zone_for(36.0, 10.0, three=False)
    left_far = zone_for(*mirror(36.0, 10.0), three=False)
    assert left_near.endswith("_l") and left_far.endswith("_l")

    right_near = zone_for(14.0, 10.0, three=False)
    right_far = zone_for(*mirror(14.0, 10.0), three=False)
    assert right_near.endswith("_r") and right_far.endswith("_r")


def test_fold_reports_depth_distance_and_angle():
    nx, ny, dist, phi = fold(CENTER_X, HOOP_NEAR_Y + 10.0)
    assert nx == pytest.approx(0.0)
    assert ny == pytest.approx(10.0)
    assert dist == pytest.approx(10.0)
    assert phi == pytest.approx(0.0)          # straight on

    nx, ny, dist, phi = fold(CENTER_X + 10.0, HOOP_NEAR_Y + 10.0)
    assert nx == pytest.approx(10.0)          # positive is the shooter's left
    assert phi == pytest.approx(45.0)
    assert dist == pytest.approx(math.hypot(10.0, 10.0))


def test_a_shot_from_behind_the_backboard_still_gets_a_sane_angle():
    """ny goes negative under the rim; the atan2 floor keeps phi from flipping half planes."""
    _, ny, _, phi = fold(CENTER_X + 6.0, HOOP_NEAR_Y - 1.0)
    assert ny < 0
    assert 0.0 < phi <= 90.0


def test_the_far_hoop_is_where_the_spec_says():
    _, ny, dist, _ = fold(CENTER_X, HOOP_FAR_Y)
    assert ny == pytest.approx(0.0) and dist == pytest.approx(0.0)


# ===================================================================== #
# Two-point boundaries
# ===================================================================== #

def test_rim_is_within_four_feet():
    assert near(CENTER_X, HOOP_NEAR_Y + 3.9) == "rim"
    assert near(CENTER_X, HOOP_NEAR_Y + 4.0) == "rim"          # inclusive
    assert near(CENTER_X, HOOP_NEAR_Y + 4.1) != "rim"


def test_paint_is_the_lane_outside_the_rim():
    assert near(CENTER_X + 7.9, HOOP_NEAR_Y + 10.0) == "paint"  # inside the lane
    assert near(CENTER_X + 8.1, HOOP_NEAR_Y + 10.0) != "paint"  # outside its width
    assert near(CENTER_X, HOOP_NEAR_Y + 14.0) == "paint"        # to the free-throw line
    assert near(CENTER_X, HOOP_NEAR_Y + 14.1) != "paint"        # and no further


def test_mid_top_is_straight_on_outside_the_paint():
    assert near(CENTER_X, HOOP_NEAR_Y + 18.0) == "mid_top"
    # Widen past 25 degrees at the same depth and it stops being "top".
    depth = 18.0
    wide = depth * math.tan(math.radians(30.0))
    assert near(CENTER_X + wide, HOOP_NEAR_Y + depth) != "mid_top"


def test_short_wide_twos_are_baseline_and_long_ones_are_wing_or_corner():
    # 12 ft out at a wide angle: short of the 16 ft split -> baseline.
    assert near(CENTER_X + 11.0, HOOP_NEAR_Y + 4.0) == "mid_base_l"
    # 18 ft at ~40 degrees -> wing.
    depth = 18.0 * math.cos(math.radians(40.0))
    across = 18.0 * math.sin(math.radians(40.0))
    assert near(CENTER_X + across, HOOP_NEAR_Y + depth) == "mid_wing_l"
    # 18 ft at ~70 degrees -> the deep corner two.
    depth = 18.0 * math.cos(math.radians(70.0))
    across = 18.0 * math.sin(math.radians(70.0))
    assert near(CENTER_X + across, HOOP_NEAR_Y + depth) == "mid_corner_l"


def test_every_two_point_position_resolves_to_a_two_point_token():
    """Sweep the half court: no gaps, and nothing leaks into a three-point token."""
    for x in range(0, 51):
        for y in range(0, 47):
            token = zone_for(float(x), float(y), three=False)
            assert token in ZONE_TOKENS
            assert not is_three(token), f"({x},{y}) -> {token}"


# ===================================================================== #
# Three-point boundaries
# ===================================================================== #

def test_corner_three_is_the_straight_line_portion_of_the_arc():
    assert zone_for(2.0, HOOP_NEAR_Y + 8.0, three=True) == "corner3_r"
    assert zone_for(2.0, HOOP_NEAR_Y + 8.75, three=True) == "corner3_r"   # inclusive
    assert zone_for(2.0, HOOP_NEAR_Y + 8.8, three=True) != "corner3_r"


def test_above_the_break_splits_into_wing_and_top():
    # Straight on, well beyond the corner depth -> top3.
    assert zone_for(CENTER_X, HOOP_NEAR_Y + 25.0, three=True) == "top3"
    # Same depth, swung past 25 degrees -> a wing three.
    across = 25.0 * math.tan(math.radians(35.0))
    assert zone_for(CENTER_X + across, HOOP_NEAR_Y + 25.0, three=True) == "wing3_l"
    assert zone_for(CENTER_X - across, HOOP_NEAR_Y + 25.0, three=True) == "wing3_r"


def test_a_heave_is_its_own_token():
    """Heaves are 0.3-0.4% of shots at 5-15%; folded into top3 they drag it down a point."""
    assert zone_for(CENTER_X, HOOP_NEAR_Y + 31.9, three=True) != "heave"
    assert zone_for(CENTER_X, HOOP_NEAR_Y + 32.0, three=True) == "heave"


def test_a_backcourt_heave_is_not_mistaken_for_an_ordinary_three():
    """The case the half-court fold gets wrong: launched from the shooter's OWN end.

    Both rows are real 2022-23 attempts. Deciding the basket by which half the shot came from
    puts the first at 19.7 ft (``top3``) and the second at 17.6 ft (``wing3_r``) — a 5-15%
    prayer landing in a zone whose make rate it then drags down, which is the whole reason
    ``heave`` is a separate token.
    """
    assert zone_for(20.3, 24.4, three=True, shot_distance=65) == "heave"
    assert zone_for(38.2, 77.1, three=True, shot_distance=73) == "heave"

    # Without the distance column there is nothing to detect it with, and the half-court rule
    # stands — documented, not silently wrong.
    assert zone_for(20.3, 24.4, three=True) != "heave"


def test_shot_distance_picks_the_basket_but_never_moves_an_ordinary_shot():
    """A normal attempt lands in the same zone with or without the distance column."""
    for x, y, three, dist in [(36.8, 10.0, False, 13), (20.0, 18.5, False, 14),
                              (1.8, 84.1, True, 24), (13.4, 29.1, True, 27),
                              (25.0, 5.0, False, 0)]:
        assert zone_for(x, y, three=three) == zone_for(x, y, three=three, shot_distance=dist)


def test_every_three_point_position_resolves_to_a_three_point_token():
    for x in range(0, 51):
        for y in range(0, 47):
            token = zone_for(float(x), float(y), three=True)
            assert is_three(token), f"({x},{y}) -> {token}"


# ===================================================================== #
# The raw marker is the authority on point value
# ===================================================================== #

def test_the_marker_decides_the_family_not_the_geometry():
    """A shot the league scored as a two stays a two even if it lands beyond the arc."""
    deep_x, deep_y = CENTER_X, HOOP_NEAR_Y + 26.0
    assert zones.geometry_says_three(deep_x, deep_y) is True
    assert zone_for(deep_x, deep_y, three=False) == "mid_top"      # a two, in a two-point zone
    assert zone_for(deep_x, deep_y, three=True) == "top3"

    # ...and the reverse: a marked three from point-blank range is still a three.
    assert zone_for(CENTER_X, HOOP_NEAR_Y + 1.0, three=True) == "corner3_l"


def test_marker_is_three_reads_the_3pt_prefix():
    assert marker_is_three("3pt jump shot") is True
    assert marker_is_three("3PT PULLUP") is True
    assert marker_is_three("driving layup") is False
    assert marker_is_three("turnaround fadeaway") is False
    for empty in (None, "", "  ", "nan", "NaN", "null"):
        assert marker_is_three(empty) is False


def test_geometry_says_three_is_none_without_coordinates():
    assert zones.geometry_says_three(None, None) is None
    assert zones.geometry_says_three("", "") is None


# ===================================================================== #
# Fallbacks — coverage is >=98.4% per season, so these are the rare tail
# ===================================================================== #

def test_missing_coordinates_fall_back_to_shot_distance():
    assert zone_for(None, None, three=False, shot_distance=1) == "rim"
    assert zone_for(None, None, three=False, shot_distance=10) == "paint"
    assert zone_for(None, None, three=False, shot_distance=20) == "mid_base_l"
    assert zone_for(None, None, three=True, shot_distance=24) == "wing3_l"
    assert zone_for(None, None, three=True, shot_distance=40) == "heave"


def test_with_neither_coordinates_nor_distance_the_defaults_apply():
    assert zone_for(None, None, three=False) == "mid_base_l"
    assert zone_for(None, None, three=True) == "wing3_l"


@pytest.mark.parametrize("bad", [None, "", "  ", "nan", "NaN", "null", "n/a", float("nan")])
def test_unusable_cells_are_treated_as_missing_not_as_zero(bad):
    """A blank coordinate read as 0.0 would put the shot in the far corner, not nowhere."""
    assert zone_for(bad, bad, three=False) == "mid_base_l"
    assert zone_for(25.0, bad, three=False) == "mid_base_l"
    assert zone_for(bad, 10.0, three=False) == "mid_base_l"


# ===================================================================== #
# Real rows sampled from RawData, cross-checked against shot_distance
# ===================================================================== #

@pytest.mark.parametrize("x,y,three,raw_distance,expected", [
    (25.0, 5.0, False, 0, "rim"),            # a layup, just under the rim
    (36.8, 10.0, False, 13, "mid_base_l"),   # baseline turnaround
    (20.0, 18.5, False, 14, "paint"),        # floater in the lane
    (1.8, 84.1, True, 24, "corner3_l"),      # far basket, left corner
    (13.4, 29.1, True, 27, "wing3_r"),       # near basket, right wing
])
def test_real_raw_rows_land_where_the_distance_column_agrees(x, y, three, raw_distance, expected):
    assert zone_for(x, y, three=three) == expected
    _, _, dist, _ = fold(x, y)
    assert dist == pytest.approx(raw_distance, abs=1.0)   # raw column is rounded to the foot
