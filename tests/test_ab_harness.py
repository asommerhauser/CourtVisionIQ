"""
The A/B harness and its run-state records (3.2 W12).

Most of this file is about **refusing comparisons that are not comparisons**. The 2.0 cycle spent real
effort on Brier differences that were inside the noise, and on confidence buckets read across runs with
different sim counts; each of those is a specific, checkable mistake.
"""
import json

import pytest

from reporting.ab_harness import (
    ARMS,
    BRIER_SD,
    PAIRED_DIFF_SD,
    assert_comparable,
    compare_arms,
    describe_arm,
    detection_threshold,
    readable_at,
    record_phase,
    replay_pass_record,
)


# --------------------------------------------------------------------------- the statistics

def test_the_thresholds_reproduce_the_direction_documents_table():
    """§7.2: 0.036 unpaired at n = 100, 0.017 paired at n = 100, 0.007 paired at n = 700."""
    assert detection_threshold(100, paired=False) == pytest.approx(0.036, abs=0.002)
    assert detection_threshold(100) == pytest.approx(0.017, abs=0.001)
    assert detection_threshold(700) == pytest.approx(0.007, abs=0.001)


def test_pairing_halves_the_threshold():
    """**The distinction the harness exists to get right.**

    0.174 is ONE run's per-game Brier sd. A comparison rests on the sd of the per-game *difference*,
    which is about half that (0.087) because two arms on the same games make correlated errors. Using the
    single-run figure would double the threshold and hide every gain 3.2 expects.
    """
    assert PAIRED_DIFF_SD == pytest.approx(BRIER_SD / 2, abs=0.005)
    assert detection_threshold(100) < detection_threshold(100, paired=False)


def test_the_expected_gain_is_borderline_at_100_games_and_clears_at_700():
    """Which is the entire argument for reading the final evaluation on the 700-game pool."""
    assert not readable_at(100, 0.010), "0.010 does not clear two standard errors at n = 100"
    assert readable_at(700, 0.010)
    assert not readable_at(700, 0.005), "the bottom of the expected range is still out of reach"


def test_no_games_is_not_a_zero_threshold():
    """Zero would make any difference 'separated', which is the worst possible default."""
    assert detection_threshold(0) == float("inf")


# --------------------------------------------------------------------------- refusing bad comparisons

def test_arms_scoring_different_windows_are_refused():
    a = describe_arm("retrained", model="v3.2", run="a", window=0, seed=42, monte_carlo=200)
    b = describe_arm("rung2", model="v3.2", run="b", window=1, seed=42, monte_carlo=200)
    with pytest.raises(ValueError, match="different games"):
        assert_comparable([a, b])


def test_arms_with_different_sim_counts_are_refused():
    """Selecting on p_hat >= 0.70 from n sims is biased even for a perfect model -- +9pp at 20 sims,
    +4.6 at 50 -- so buckets are not comparable across sim counts, and Brier's Monte-Carlo inflation
    differs too."""
    a = describe_arm("retrained", model="v3.2", run="a", window=0, seed=42, monte_carlo=200)
    b = describe_arm("kpi", model="v3.2", run="b", window=0, seed=42, monte_carlo=50)
    with pytest.raises(ValueError, match="sim count"):
        assert_comparable([a, b])


def test_arms_with_different_seeds_are_refused():
    """The one place the standing rule inverts.

    §8 says repeat runs on the same window use a different --seed so they are independent draws. That is
    for repeats of ONE model. For a model comparison a shared seed makes both arms face the same
    Monte-Carlo draw and removes that variance from the difference.
    """
    a = describe_arm("retrained", model="v3.2", run="a", window=0, seed=42, monte_carlo=200)
    b = describe_arm("rung2", model="v3.2", run="b", window=0, seed=7, monte_carlo=200)
    with pytest.raises(ValueError, match="independent Monte-Carlo draw"):
        assert_comparable([a, b])


def test_a_properly_matched_pair_passes():
    arms = [describe_arm(n, model="v3.2", run=f"r-{n}", window=0, seed=42, monte_carlo=200)
            for n in ("retrained", "kpi")]
    assert_comparable(arms)


def test_a_single_arm_is_not_a_comparison_and_is_not_refused():
    assert_comparable([describe_arm("retrained", model="v3.2", run="a", window=0, seed=1,
                                    monte_carlo=200)]) is None


def test_an_unknown_arm_is_refused():
    with pytest.raises(ValueError, match="unknown arm"):
        describe_arm("multi_scale_time", model="v3.3", run="a", window=0, seed=1, monte_carlo=200)


def test_the_arms_are_a_chain_each_a_superset_of_the_last():
    assert ARMS == ("retrained", "rung2", "kpi")


# --------------------------------------------------------------------------- the comparison itself

def _records(probs, actual):
    return [{"game_id": i, "win_prob_home": p, "actual_home_win": a}
            for i, (p, a) in enumerate(zip(probs, actual))]


def test_a_difference_inside_two_standard_errors_reads_as_the_same_model():
    """Which is the correct reading, and the one the 2.0 cycle kept not taking."""
    actual = [1, 0] * 50
    a = _records([0.6, 0.4] * 50, actual)
    b = _records([0.61, 0.39] * 50, actual)
    out = compare_arms(a, b, label_a="retrained", label_b="kpi")
    assert out["n"] == 100
    assert out["verdict"] == "the same model"
    assert out["separated"] is False


def test_a_large_difference_separates():
    actual = [1, 0] * 50
    a = _records([0.9, 0.1] * 50, actual)
    b = _records([0.5, 0.5] * 50, actual)
    out = compare_arms(a, b, label_a="retrained", label_b="kpi")
    assert out["separated"] is True and out["verdict"] == "separated"


def test_the_comparison_carries_the_threshold_it_had_to_clear():
    """So a reader never has to reconstruct which n and which sd produced the verdict."""
    actual = [1, 0] * 10
    out = compare_arms(_records([0.6, 0.4] * 10, actual), _records([0.6, 0.4] * 10, actual),
                       label_a="a", label_b="b")
    assert out["threshold_2se"] == pytest.approx(detection_threshold(20))
    assert out["a"] == "a" and out["b"] == "b"


# --------------------------------------------------------------------------- the run-state record

def test_the_replay_record_keeps_the_per_head_kept_counts():
    """Per head, because the filter is per head: a head every sim failed equally gets no update, and
    without this number that is indistinguishable from a pass that trained on everything."""
    rec = replay_pass_record(games=520, sims_per_game=10,
                             head_kept={"shot_type": 2600, "foul_type": 0}, seed=42)
    assert rec["game_sims"] == 5200
    assert rec["head_kept"] == {"foul_type": 0, "shot_type": 2600}
    assert rec["kept_total"] == 2600


def test_the_record_lands_in_the_run_state(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"version": "v3.2", "checkpoint_selection": {"x": 1}}),
                     encoding="utf-8")
    record_phase(state, "kpi", replay_pass_record(games=1, sims_per_game=10,
                                                  head_kept={"player": 5}, seed=0))
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["replay_pass"]["game_sims"] == 10
    assert saved["checkpoint_selection"] == {"x": 1}, "beside it, not over it"


def test_recording_twice_merges_rather_than_replaces(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({}), encoding="utf-8")
    record_phase(state, "kpi", {"games": 1})
    record_phase(state, "kpi", {"seed": 9})
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert saved["replay_pass"] == {"games": 1, "seed": 9}


def test_a_missing_state_file_is_silent(tmp_path):
    """A record is worth having, and worth nothing at the cost of failing a finished run over it."""
    record_phase(tmp_path / "absent.json", "kpi", {"games": 1})     # must not raise
