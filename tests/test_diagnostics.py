"""
Shot-mix diagnostic tests (workstream 13, the per-zone half of §11).

The eval's box-score tables are per-player accuracy stats; shot MIX is a distribution
comparison, which is why it lives in ``simulation.diagnostics`` beside the event histogram and
the Δt buckets rather than in the eval report.

What it exists to catch: the model can match total FGA and total eFG on a completely different
shot mix -- more long twos, fewer corner threes -- and every game-level number in the report
would look right. The zone tokens were introduced in Phase 2 precisely to make that question
askable, and nothing asked it until now.
"""
from __future__ import annotations

from simulation.diagnostics import _zone_mix, _summarize_zone_mix
from zones import ZONE_TOKENS


def _mix(shots):
    """shots: (zone, made) pairs -> the three parallel columns _zone_mix reads."""
    events = ["shot"] * len(shots)
    types = [z for z, _ in shots]
    results = ["made" if m else "missed" for _, m in shots]
    return _zone_mix(events, types, results)


def test_zone_mix_counts_attempts_and_makes_per_zone():
    hist = _mix([("rim", True), ("rim", False), ("top3", True)])
    assert hist["rim.fga"] == 2 and hist["rim.fgm"] == 1
    assert hist["top3.fga"] == 1 and hist["top3.fgm"] == 1
    assert hist["paint.fga"] == 0


def test_every_zone_is_present_even_at_zero():
    """A missing key would make the summary's mean silently skip a game."""
    hist = _mix([("rim", True)])
    for zone in ZONE_TOKENS:
        assert f"{zone}.fga" in hist and f"{zone}.fgm" in hist


def test_free_throws_are_not_shot_selection():
    hist = _mix([("rim", True)])
    hist2 = _zone_mix(["shot", "shot"], ["rim", "free throw"], ["made", "made"])
    assert hist2["rim.fga"] == hist["rim.fga"]
    assert sum(hist2[f"{z}.fga"] for z in ZONE_TOKENS) == 1


def test_non_shot_events_are_ignored():
    hist = _zone_mix(["rebound", "foul", "shot"], ["rim", "rim", "rim"],
                     ["cop", "nothing", "made"])
    assert hist["rim.fga"] == 1


def test_summary_reports_shares_not_counts():
    """A pace difference must not read as a shot-selection difference: they are separate faults.

    Twice as many shots, identical mix -> identical shares.
    """
    real = [_mix([("rim", True), ("top3", False)])]
    fast = [_mix([("rim", True), ("rim", True), ("top3", False), ("top3", False)])]
    summ = _summarize_zone_mix(fast, real)
    assert summ["rim"]["share_pred"] == summ["rim"]["share_actual"] == 0.5
    assert summ["top3"]["share_pred"] == summ["top3"]["share_actual"] == 0.5
    # the raw counts still differ, so a pace problem is still visible
    assert summ["rim"]["fga_pred"] == 2 and summ["rim"]["fga_actual"] == 1


def test_summary_catches_a_mix_shift_that_leaves_the_totals_alone():
    """The failure this diagnostic exists for: same FGA, same makes, different shots."""
    real = [_mix([("rim", True), ("corner3_l", False)])]
    pred = [_mix([("mid_top", True), ("rim", False)])]
    summ = _summarize_zone_mix(pred, real)
    assert summ["corner3_l"]["share_actual"] == 0.5
    assert summ["corner3_l"]["share_pred"] == 0.0
    assert summ["mid_top"]["share_pred"] == 0.5
    assert summ["mid_top"]["share_actual"] == 0.0


def test_shares_sum_to_one_and_fg_pct_is_within_zone():
    hist = [_mix([("rim", True), ("rim", False), ("rim", True), ("top3", False)])]
    summ = _summarize_zone_mix(hist, hist)
    assert abs(sum(v["share_actual"] for v in summ.values()) - 1.0) < 1e-9
    assert summ["rim"]["fg_pct_actual"] == 2 / 3        # within the zone, not of all shots
    assert summ["top3"]["fg_pct_actual"] == 0.0


def test_an_empty_run_does_not_divide_by_zero():
    summ = _summarize_zone_mix([], [])
    assert all(v["share_pred"] == 0.0 and v["fg_pct_actual"] == 0.0 for v in summ.values())
