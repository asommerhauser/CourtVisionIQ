"""
The weighted replay pass (3.2 W11).

Two things are load-bearing and both are tested as properties rather than as arithmetic.

**The estimator cannot diverge.** The direction document specifies a signed sample weight; a negative
weight on cross-entropy is ``-w·log p`` with ``w < 0``, minimised by driving that action's probability to
zero and the loss to ``-inf``. Filtering to positive advantages is bounded by construction, which is why
every weight this module produces is strictly positive.

**The advantage travels down the channel that already exists.** Every head multiplies its loss mask by a
per-game weight through ``season_features.apply_recency``, which reads ``split["recency_weight"]`` -- so
the pass is a reweighting of the ordinary training path, not a second one.
"""
import numpy as np
import pytest

from models.replay import (
    MIN_SIBLINGS,
    advantages,
    apply_advantages,
    head_weights,
    keep_positive,
    kept_fraction,
    positive_weights,
    replay_weight_array,
)


# --------------------------------------------------------------------------- the advantage

def test_the_advantage_is_the_sibling_mean_minus_this_sim():
    """Errors, so lower is better and a positive advantage means MORE realistic than its siblings."""
    scores = [1.0, 2.0, 3.0]
    advs = advantages(scores)
    assert advs[0] == pytest.approx((2.0 + 3.0) / 2 - 1.0)      # best sim, positive
    assert advs[2] == pytest.approx((1.0 + 2.0) / 2 - 3.0)      # worst sim, negative
    assert advs[0] > 0 > advs[2]


def test_the_baseline_leaves_this_sim_out():
    """Including itself would shrink every advantage toward zero by 1/n and bias the filter."""
    scores = [1.0, 5.0]
    advs = advantages(scores)
    assert advs[0] == pytest.approx(5.0 - 1.0)
    assert advs[1] == pytest.approx(1.0 - 5.0)


def test_identical_sims_have_no_advantage_either_way():
    advs = advantages([2.0, 2.0, 2.0])
    assert advs == pytest.approx([0.0, 0.0, 0.0])
    assert keep_positive(advs) == [], "indistinguishable from the baseline teaches nothing"


def test_a_game_with_no_siblings_yields_zero_not_a_fabricated_advantage():
    """One sim has no leave-one-out baseline. Zero is the honest answer and the filter drops it."""
    assert advantages([3.0]) == [0.0]
    assert keep_positive(advantages([3.0])) == []
    assert MIN_SIBLINGS == 2


def test_the_advantages_of_a_game_sum_to_zero():
    """A property of the leave-one-out baseline, and the reason roughly half of any batch is kept."""
    advs = advantages([1.0, 4.0, 2.5, 9.0, 0.5])
    assert sum(advs) == pytest.approx(0.0)


def test_about_half_a_batch_survives_the_filter():
    rng = np.random.default_rng(0)
    fractions = [kept_fraction(advantages(rng.normal(size=10).tolist())) for _ in range(50)]
    assert 0.3 < float(np.mean(fractions)) < 0.7


# --------------------------------------------------------------------------- the filter

def test_every_weight_is_strictly_positive():
    """**The property that makes the pass unable to diverge.**

    A negative weight on cross-entropy is minimised by driving the sampled action's probability to zero
    and the loss to -inf, so a signed estimator needs a clip and a KL leash to stay stable. There is no
    budget for tuning three constants inside one pass, so the filter is the bound.
    """
    weights = positive_weights(advantages([1.0, 2.0, 3.0, 9.0]))
    assert weights and all(w > 0 for w in weights.values())


def test_exactly_zero_is_not_kept():
    assert positive_weights([0.0, 0.0]) == {}


def test_all_sims_equally_bad_produces_no_pass_at_all():
    """Nothing to learn from is a legitimate outcome, and must not be an error."""
    assert positive_weights(advantages([5.0, 5.0, 5.0])) == {}


def test_the_kept_weights_average_one():
    """So the pass's effective learning rate does not ride on the arbitrary scale of the score.

    Without it the step size would depend on whether errors happen to be measured in points or in
    standard deviations, and one pass is not the place to discover that.
    """
    weights = positive_weights(advantages([0.1, 0.2, 10.0, 20.0]))
    assert float(np.mean(list(weights.values()))) == pytest.approx(1.0)


def test_scaling_every_score_leaves_the_relative_weights_alone():
    a = positive_weights(advantages([1.0, 2.0, 4.0]))
    b = positive_weights(advantages([10.0, 20.0, 40.0]))
    assert set(a) == set(b)
    for k in a:
        assert a[k] == pytest.approx(b[k])


# --------------------------------------------------------------------------- per-head weighting

def test_each_head_is_weighted_by_its_own_advantage():
    """**The point of per-head metrics.**

    A sim that got the shot mix right and the rotation wrong should teach ``shot_type`` and not
    ``substitution``. One shared scalar hands both the same weight and reintroduces exactly the
    mis-assignment head_metrics exists to remove.
    """
    errors = [
        {"shot_type": 0.1, "substitution": 9.0},   # good mix, bad rotation
        {"shot_type": 9.0, "substitution": 0.1},   # the reverse
    ]
    weights = head_weights(errors)
    assert list(weights["shot_type"]) == [0], "only the sim with the better mix teaches shot_type"
    assert list(weights["substitution"]) == [1]


def test_a_head_every_sim_failed_equally_gets_no_update():
    errors = [{"foul_type": 4.0}, {"foul_type": 4.0}, {"foul_type": 4.0}]
    assert head_weights(errors)["foul_type"] == {}


def test_head_weights_cover_every_head_in_the_scores():
    from models.head_metrics import HEAD_METRICS
    errors = [{h: float(i + 1) for h in HEAD_METRICS} for i in range(3)]
    assert set(head_weights(errors)) == set(HEAD_METRICS)


def test_no_scores_is_not_an_error():
    assert head_weights([]) == {}


# --------------------------------------------------------------------------- the injection seam

def test_the_weight_array_is_aligned_to_the_splits_game_order():
    arr = replay_weight_array([7, 8, 9], {7: 1.5, 9: 0.5})
    np.testing.assert_allclose(arr, [1.5, 0.0, 0.5])
    assert arr.dtype == np.float32


def test_a_filtered_out_sim_contributes_nothing_without_being_removed():
    """Zero weight rather than dropped rows: the tensors keep their shape and the mask does the work."""
    arr = replay_weight_array([1, 2], {1: 2.0})
    assert arr[1] == 0.0


def test_the_advantage_uses_the_per_game_channel_every_head_already_honours():
    """Not a second mechanism. ``apply_recency`` reads exactly this key, in all twelve heads."""
    split = {"loss_mask": np.ones((2, 4), dtype="float32")}
    out = apply_advantages(split, [1, 2], {1: 3.0})
    assert "recency_weight" in out
    np.testing.assert_allclose(out["recency_weight"], [3.0, 0.0])

    from models.season_features import apply_recency
    masked = apply_recency(out["loss_mask"], out)
    np.testing.assert_allclose(masked[0], [3.0] * 4)
    np.testing.assert_allclose(masked[1], [0.0] * 4)


def test_applying_advantages_does_not_mutate_the_split_it_was_given():
    split = {"recency_weight": np.array([1.0], dtype="float32")}
    out = apply_advantages(split, [1], {1: 5.0})
    assert split["recency_weight"][0] == 1.0 and out["recency_weight"][0] == 5.0


def test_recency_is_overwritten_rather_than_multiplied():
    """These are sims of games from one narrow slice, not a corpus spanning twenty-one seasons, so the
    corpus-age discount has nothing to say about them."""
    split = {"recency_weight": np.array([0.05, 0.05], dtype="float32")}
    out = apply_advantages(split, [1, 2], {1: 1.0, 2: 1.0})
    np.testing.assert_allclose(out["recency_weight"], [1.0, 1.0])
