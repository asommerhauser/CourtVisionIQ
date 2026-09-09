"""
Tests for models/rotation_features.py -- the per-player on-court state the rotation model needs.

Three scalars per on-court player, derived by one scan that both preprocessing and the simulator
drive, so they cannot diverge: seconds in the current stint, seconds played, personal fouls.

What is worth pinning here is the accounting, not the arithmetic: minutes belong to the lineup
that held the floor over an interval (not the one on the row that ends it), a stint starts when a
player comes on, and a technical is not a personal foul. The measurement pass in the same module
is what checks the derivation against the data; these check it against its own definition.
"""

import numpy as np
import pytest

from config import ROSTER_SIZE
from models.rotation_features import (
    ROSTER_STATE_KEYS,
    LineupScan,
    _fold_subs,
    _scan_game,
    derive_lineup_state,
    normalize_lineup_state,
)

HOME = ["Alice", "Bob", "Charlie", "Dave", "Eve"]
AWAY = ["Frank", "Grace", "Hank", "Ivy", "Jack"]


def _row(time, event="shot", player="Alice", etype="rim", result="made",
         home=None, away=None, secondary="none"):
    return {
        "time": time, "event": event, "player": player, "type": etype, "result": result,
        "secondary_player": secondary,
        "roster_home": list(home or HOME), "roster_away": list(away or AWAY),
    }


# ---------------------------------------------------------------------------
# Minutes
# ---------------------------------------------------------------------------

def test_minutes_are_credited_to_the_lineup_that_held_the_floor():
    """The interval belongs to the five that was on for it, not the five on the closing row."""
    swapped = ["Kate"] + HOME[1:]
    scan = LineupScan()
    scan.step(_row(0, event="start", player="start"))
    scan.step(_row(100))                                   # Alice on court for 0..100
    scan.step(_row(160, home=swapped))                     # Kate replaces Alice at 160
    scan.step(_row(200, home=swapped))                     # Kate on court for 160..200

    assert scan.played["Alice"] == pytest.approx(160.0)
    assert scan.played["Kate"] == pytest.approx(40.0)


def test_every_second_of_the_game_is_credited_to_exactly_ten_players():
    """The identity the measurement pass gates on, at the scale of one synthetic game."""
    scan = LineupScan()
    rows = [_row(0, event="start", player="start"), _row(300), _row(700),
            _row(700, home=["Kate"] + HOME[1:]), _row(1440, home=["Kate"] + HOME[1:])]
    for row in rows:
        scan.step(row)
    assert sum(scan.played.values()) == pytest.approx(1440.0 * 10)


def test_a_row_that_does_not_advance_the_clock_credits_nobody():
    """Several events share one timestamp constantly; none of them is playing time."""
    scan = LineupScan()
    scan.step(_row(0, event="start", player="start"))
    scan.step(_row(50))
    before = dict(scan.played)
    scan.step(_row(50, event="rebound", player="Bob"))
    assert scan.played == before


# ---------------------------------------------------------------------------
# Stints
# ---------------------------------------------------------------------------

def test_a_stint_starts_when_the_player_comes_on():
    swapped = ["Kate"] + HOME[1:]
    scan = LineupScan()
    scan.step(_row(0, event="start", player="start"))
    scan.step(_row(600, home=swapped))
    stint_home, _, _, _, _, _ = scan.step(_row(900, home=swapped))

    assert stint_home[0] == pytest.approx(300.0)   # Kate came on at 600
    assert stint_home[1] == pytest.approx(900.0)   # Bob has not come off


def test_coming_back_on_starts_a_new_stint_rather_than_resuming_the_old_one():
    swapped = ["Kate"] + HOME[1:]
    scan = LineupScan()
    scan.step(_row(0, event="start", player="start"))
    scan.step(_row(300, home=swapped))              # Alice off
    scan.step(_row(900, home=HOME))                 # Alice back on
    stint_home, *_ = scan.step(_row(1000, home=HOME))

    assert stint_home[0] == pytest.approx(100.0)
    # ... but the minutes from the first stint are still hers.
    assert scan.played["Alice"] == pytest.approx(300.0 + 100.0)


# ---------------------------------------------------------------------------
# Fouls
# ---------------------------------------------------------------------------

def test_a_personal_foul_counts_against_the_fouler_on_the_row_it_happens():
    scan = LineupScan()
    scan.step(_row(0, event="start", player="start"))
    *_, fouls_home, _ = scan.step(_row(100, event="foul", player="Bob", etype="personal",
                                       result="nothing"))
    assert fouls_home[HOME.index("Bob")] == 1.0


def test_a_technical_is_not_a_personal_foul():
    """It is a bench foul: simulation/controller.py:_charge_foul returns early on it, and the
    disqualification limit this feature exists to represent never counts one."""
    scan = LineupScan()
    scan.step(_row(0, event="start", player="start"))
    *_, fouls_home, _ = scan.step(_row(100, event="foul", player="Bob", etype="technical",
                                       result="free throw"))
    assert fouls_home[HOME.index("Bob")] == 0.0


# ---------------------------------------------------------------------------
# Array driver and normalization
# ---------------------------------------------------------------------------

def test_derive_returns_one_slot_aligned_array_per_key():
    rows = [_row(0, event="start", player="start"), _row(120), _row(240)]
    out = derive_lineup_state(rows)
    assert set(out) == set(ROSTER_STATE_KEYS)
    for key in ROSTER_STATE_KEYS:
        assert out[key].shape == (len(rows), ROSTER_SIZE)
        assert out[key].dtype == np.float32
    # Slot 0 of the home arrays is Alice, who has been on since tip-off.
    assert out["played_seconds_home"][-1, 0] == pytest.approx(240.0)


def test_normalization_is_fixed_constants_and_clips_at_the_top():
    """No train-fit statistics, so nothing has to be persisted or reloaded at inference."""
    raw = {k: np.zeros((1, ROSTER_SIZE), dtype=np.float32) for k in ROSTER_STATE_KEYS}
    raw["court_fouls_home"][0, 0] = 3.0
    raw["court_fouls_home"][0, 1] = 99.0          # beyond the limit: must clip, not explode
    raw["played_seconds_home"][0, 0] = 1440.0
    out = normalize_lineup_state(raw)
    assert out["court_fouls_home"][0, 0] == pytest.approx(1.0)
    assert out["court_fouls_home"][0, 1] == pytest.approx(2.0)
    assert out["played_seconds_home"][0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The measurement pass
# ---------------------------------------------------------------------------

def test_folding_the_substitutions_reproduces_the_roster_snapshots():
    """The gate's independent route. It reads no roster column after the first row, so it agrees
    only if every lineup change in the file is explained by a substitution row."""
    swapped = ["Kate"] + HOME[1:]
    rows = [
        _row(0, event="start", player="start"),
        _row(300),
        _row(600, event="substitution", player="Alice", etype="substitution",
             result="substitution", secondary="Kate", home=swapped),
        _row(900, home=swapped),
    ]
    folded = _fold_subs(rows)
    for row, (home, away) in zip(rows, folded):
        assert set(home) == set(row["roster_home"])
        assert set(away) == set(row["roster_away"])

    disagree, short, subs, played, end = _scan_game(rows)
    assert disagree == 0
    assert short == 0
    assert subs == 1
    assert sum(played.values()) == pytest.approx(end * 10.0)


def test_a_lineup_change_with_no_substitution_row_is_caught():
    """What the gate exists to detect: the five moves and nothing explains it."""
    rows = [
        _row(0, event="start", player="start"),
        _row(300),
        _row(600, home=["Kate"] + HOME[1:]),      # Kate appears, no substitution row
        _row(900, home=["Kate"] + HOME[1:]),
    ]
    disagree, *_ = _scan_game(rows)
    assert disagree == 2


def test_a_substitution_with_no_incoming_player_shrinks_the_fold_too():
    """The sentinel rows have to move the fold, or it disagrees for the rest of the game."""
    short_five = HOME[1:]
    rows = [
        _row(0, event="start", player="start"),
        _row(300, event="substitution", player="Alice", etype="substitution",
             result="substitution", secondary="none", home=short_five),
        _row(600, home=short_five),
    ]
    disagree, short, *_ = _scan_game(rows)
    assert disagree == 0
    assert short == 2                              # the two rows a side is four
