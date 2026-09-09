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
    BENCH_ID_KEYS,
    BENCH_STATE_KEYS,
    SUB_COUNT_CLASSES,
    NUM_ROSTER_SCALARS,
    ROSTER_STATE_KEYS,
    LineupScan,
    _fold_subs,
    _scan_game,
    can_substitute,
    dead_ball_after,
    derive_lineup_state,
    derive_sub_decisions,
    merge_rotation_features,
    normalize_lineup_state,
    normalize_lineup_state_row,
    side_scalars,
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

    disagree, short, duplicated, subs, played, end = _scan_game(rows)
    assert disagree == 0
    assert short == 0
    assert duplicated == 0
    assert subs == 1
    assert sum(played.values()) == pytest.approx(end * 10.0)


def test_a_player_in_two_slots_is_caught():
    """Nothing in the raw data does this; it is what a substitution applied against the wrong
    lineup produces, and membership comparisons hide it until the five grows to six."""
    rows = [
        _row(0, event="start", player="start"),
        _row(300, home=["Alice", "Bob", "Charlie", "Dave", "Alice"]),
    ]
    _, _, duplicated, *_ = _scan_game(rows)
    assert duplicated == 1


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
    disagree, short, duplicated, *_ = _scan_game(rows)
    assert disagree == 0
    assert duplicated == 0
    assert short == 2                              # the two rows a side is four


# ---------------------------------------------------------------------------
# Train / inference parity
# ---------------------------------------------------------------------------

def test_the_incremental_path_matches_the_batch_path_bit_for_bit():
    """The property the whole design rests on.

    Preprocessing folds a whole game at once through merge_rotation_features; the simulator
    folds one row at a time through LineupScan + normalize_lineup_state_row. They are the same
    scan, so they must agree exactly -- not approximately, since a drift here is a model fed
    one thing in training and another at rollout, with nothing to report it.
    """
    import pandas as pd

    rows = [
        _row(0, event="start", player="start"),
        _row(120),
        _row(300, event="foul", player="Bob", etype="personal", result="nothing"),
        _row(600, home=["Kate"] + HOME[1:]),
        _row(900, home=["Kate"] + HOME[1:]),
    ]
    df = pd.DataFrame([{**r, "game_id": 1} for r in rows])

    cols = {}
    merge_rotation_features(df, cols)

    scan = LineupScan()
    for i, row in enumerate(rows):
        incremental = normalize_lineup_state_row(scan.step(row))
        for key, values in zip(ROSTER_STATE_KEYS, incremental):
            assert np.array_equal(cols[key][i], values), f"{key} differs at row {i}"


def test_the_scalars_reach_the_encoder_with_rest_first():
    """Rest stays scalar 0, so the single-scalar ordering from before 2.0 is a prefix of this
    one and the meaning of a slot does not move under a model that predates the others."""
    rotation = {k: f"<{k}>" for k in ROSTER_STATE_KEYS}
    assert side_scalars("<rest_home>", rotation, "home") == [
        "<rest_home>", "<stint_seconds_home>", "<played_seconds_home>", "<court_fouls_home>",
    ]
    assert len(side_scalars("<rest_away>", rotation, "away")) == NUM_ROSTER_SCALARS


# ---------------------------------------------------------------------------
# The bench bundle
# ---------------------------------------------------------------------------

BENCH = ["Kate", "Liam", "Mia"]


def _avail():
    return (HOME + BENCH, list(AWAY))


def test_the_bench_is_who_is_available_and_not_on_the_floor():
    scan = LineupScan(_avail())
    scan.step(_row(0, event="start", player="start"))
    names, *_ = scan.bench_state(0)
    assert names == BENCH

    scan.step(_row(300, home=["Kate"] + HOME[1:]))
    names, *_ = scan.bench_state(0)
    assert set(names) == {"Alice", "Liam", "Mia"}, "Alice sat down, Kate came on"


def test_bench_rest_runs_from_sitting_down_and_has_played_says_which():
    """A starter resting two minutes and a deep bench player who has not moved all night both
    read as a long time; only the flag separates them."""
    scan = LineupScan(_avail())
    scan.step(_row(0, event="start", player="start"))
    scan.step(_row(300, home=["Kate"] + HOME[1:]))          # Alice off at 300
    scan.step(_row(500, home=["Kate"] + HOME[1:]))

    names, rest, played, fouls, has_played = scan.bench_state(0)
    by_name = dict(zip(names, rest))
    assert by_name["Alice"] == pytest.approx(200.0)          # sat down at 300, now 500
    assert by_name["Liam"] == pytest.approx(500.0)           # never played: measured from tip-off
    flags = dict(zip(names, has_played))
    assert flags["Alice"] == 1.0
    assert flags["Liam"] == 0.0
    assert dict(zip(names, played))["Alice"] == pytest.approx(300.0)


def test_a_scan_with_no_available_set_has_no_bench():
    """Every on-court-only caller builds LineupScan bare, and must not pay for a bench."""
    scan = LineupScan()
    scan.step(_row(0, event="start", player="start"))
    names, *_ = scan.bench_state(0)
    assert names == []


def test_bench_ids_are_encoded_and_pad_filled_to_bench_size():
    from config import BENCH_SIZE

    rows = [_row(0, event="start", player="start"), _row(300)]
    seen = {}
    encode = lambda names: (
        [seen.setdefault(n, len(seen) + 1) for n in names[:BENCH_SIZE]]
        + [0] * (BENCH_SIZE - len(names[:BENCH_SIZE])))

    out = derive_lineup_state(rows, encode_bench=encode)
    for key in BENCH_ID_KEYS:
        assert out[key].shape == (len(rows), BENCH_SIZE)
        assert out[key].dtype == np.int32
    for key in BENCH_STATE_KEYS:
        assert out[key].shape == (len(rows), BENCH_SIZE)
    # Only the five who appear on the floor are available in this fixture, so with the five on
    # court the bench is empty and every slot is PAD.
    assert not out["bench_home"].any()


def test_bench_ids_are_not_normalized():
    """They are tokens, not quantities: a clip-and-divide would corrupt every player id."""
    raw = {
        "bench_home": np.array([[7, 9, 0]], dtype=np.int32),
        "bench_fouls_home": np.array([[3.0, 99.0, 0.0]], dtype=np.float32),
    }
    out = normalize_lineup_state(raw)
    assert np.array_equal(out["bench_home"], raw["bench_home"])
    assert out["bench_fouls_home"][0, 0] == pytest.approx(1.0)
    assert out["bench_fouls_home"][0, 1] == pytest.approx(2.0)      # clipped at the foul limit


# ---------------------------------------------------------------------------
# Sub-decision positions and targets
# ---------------------------------------------------------------------------

def _sub(time, side="home", out_="Alice", in_="Kate"):
    row = _row(time, event="substitution", player=out_, etype="substitution",
               result="substitution", secondary=in_)
    row["home/away"] = 1 if side == "home" else 2
    return row


def test_the_dead_ball_rule_matches_the_table_in_section_2():
    late = dict(period_idx=0, seconds_left=30.0, ends_free_throws=True)
    early = dict(period_idx=0, seconds_left=400.0, ends_free_throws=True)

    assert dead_ball_after("foul", "personal", "free throw", **early)
    assert dead_ball_after("timeout", "home", "none", **early)
    assert dead_ball_after("turnover", "violation", "cop", **early)
    assert not dead_ball_after("turnover", "steal", "cop", **early), "a steal is live"
    assert dead_ball_after("rebound", "team defensive", "cop", **early)
    assert not dead_ball_after("rebound", "defensive", "cop", **early), "a live board"
    assert not dead_ball_after("shot", "rim", "missed", **early)
    # The clock stops after a made basket only late in the period.
    assert not dead_ball_after("shot", "rim", "made", **early)
    assert dead_ball_after("shot", "rim", "made", **late)
    # Mid-trip a free throw leaves the ball dead: the shooter simply shoots again.
    assert dead_ball_after("shot", "free throw", "missed",
                           period_idx=0, seconds_left=400.0, ends_free_throws=False)


def test_a_made_field_goal_is_a_dead_ball_but_never_a_substitution_opportunity():
    """NBA Rule 3 Section V clause 10, which has no last-two-minutes exception.

    The one row type where the clock notion and the substitution notion part company. Conflating
    them adds ~24 opportunities a game at which no substitution is legal.
    """
    assert dead_ball_after("shot", "rim", "made",
                           period_idx=3, seconds_left=30.0, ends_free_throws=True)
    assert not can_substitute("shot", "rim", "made", ends_free_throws=True)


def test_the_substitution_rule_permits_exactly_what_clause_10_allows():
    """Clause 10 names the exceptions: personal foul, technical foul, timeout, violation."""
    assert can_substitute("foul", "personal", "free throw", ends_free_throws=True)
    assert can_substitute("foul", "technical", "free throw", ends_free_throws=True)
    assert can_substitute("timeout", "home", "none", ends_free_throws=True)
    assert can_substitute("turnover", "violation", "cop", ends_free_throws=True)
    assert can_substitute("rebound", "team defensive", "cop", ends_free_throws=True)
    # Live play is never an opportunity, whoever has the ball.
    assert not can_substitute("turnover", "steal", "cop", ends_free_throws=True)
    assert not can_substitute("shot", "rim", "missed", ends_free_throws=True)


def test_there_is_no_possession_condition_on_substituting():
    """Rule 3 has none. After a defensive rebound neither team may substitute -- because the
    ball is live, not because of who holds it."""
    for side in ("defensive", "offensive"):
        assert not can_substitute("rebound", side, "cop", ends_free_throws=True)


def test_the_free_throw_window_follows_clause_9():
    """Substitutes enter prior to the final attempt if the ball will remain in play, or after it
    if it will not."""
    assert can_substitute("shot", "free throw", "made", ends_free_throws=True)
    # A missed last attempt leaves the ball live, so clause 9 puts its window BEFORE it -- which
    # in an event stream is the foul that awarded the trip.
    assert not can_substitute("shot", "free throw", "missed", ends_free_throws=True)
    assert not can_substitute("shot", "free throw", "made", ends_free_throws=False)


def test_the_target_counts_the_substitution_run_that_follows_per_side():
    rows = [
        _row(0, event="start", player="start"),
        _row(300, event="foul", player="Bob", etype="personal", result="free throw"),
        _sub(300, "home"), _sub(300, "away"), _sub(300, "away"),
        _row(320),
    ]
    out = derive_sub_decisions(rows)
    assert out["can_sub"][1] == 1.0
    assert out["subs_home"][1] == 1.0
    assert out["subs_away"][1] == 2.0
    # The substitution rows themselves are never query positions.
    assert out["can_sub"][2] == 0.0


def test_the_count_is_capped_at_the_last_class():
    rows = [_row(0, event="start", player="start"),
            _row(300, event="timeout", player="none", etype="home", result="none"),
            *[_sub(300, "home") for _ in range(5)],
            _row(320)]
    out = derive_sub_decisions(rows)
    assert out["subs_home"][1] == SUB_COUNT_CLASSES - 1


def test_substitutions_are_credited_to_an_opportunity_not_to_the_row_above_them():
    """13.8% of substitution runs sit where no row is a legal opportunity, because the raw file
    appends a stoppage's substitutions after the play that drew the whistle. Attributing by
    position would drop those from the target and teach a rate well below the truth."""
    rows = [
        _row(0, event="start", player="start"),
        _row(200, event="foul", player="Bob", etype="personal", result="free throw"),
        _row(300, event="shot", player="Frank", etype="free throw", result="missed"),
        _sub(300, "home"),
        _row(320),
    ]
    out = derive_sub_decisions(rows)
    # The missed last free throw is live, so it is not an opportunity ...
    assert out["can_sub"][2] == 0.0
    assert out["subs_home"][2] == 0.0
    # ... and the substitution is credited back to the foul, which is one.
    assert out["can_sub"][1] == 1.0
    assert out["subs_home"][1] == 1.0


def test_a_live_row_is_never_a_query_position():
    rows = [_row(0, event="start", player="start"),
            _row(300, event="shot", player="Alice", etype="rim", result="missed"),
            _row(320)]
    out = derive_sub_decisions(rows)
    assert out["can_sub"][1] == 0.0
