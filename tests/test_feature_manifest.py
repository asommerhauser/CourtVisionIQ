"""
The manifest's input signature, and the refusal it enables.

``ARCH_KEYS`` records how big a model is. It has never recorded WHICH INPUTS it was trained with,
and ``read_manifest`` was advisory -- nothing consulted it at load, and ``from_artifacts`` rebuilt
the graph from the live ``config.py`` and called ``load_weights`` unconditionally.

3.0 widened the fusion three times in one retrain (eight team priors, running_pace, the regime
latent) and took the roster encoder's per-player scalar count from 4 to 14. Most of that does fail
loudly at ``load_weights``, because ``fusion_projection``'s kernel is a function of the concatenated
width. But a change that swaps one key for another of the same width loads with no error at all and
means something different on every row -- the same class of failure ``LOCAL_ATTENTION_*`` was added
to ``ARCH_KEYS`` to catch.

The compatibility half matters as much as the refusal: a v1.0 or 2.0 bundle records no signature,
and an absence must read as an absence rather than as a mismatch, or every existing model stops
loading the day this lands.
"""
from __future__ import annotations

import pytest

from models.manifest import ARCH_KEYS, SCHEMA, feature_mismatch, feature_snapshot


def test_the_signature_records_the_input_names_not_just_their_count():
    """Counts alone cannot tell a swap from a match."""
    snap = feature_snapshot()
    assert "running_pace" in snap["game_state_keys"]
    assert "prior_home" in snap["prior_input_keys"]
    assert snap["regime_key"] == "regime"
    assert snap["num_roster_scalars"] == 14


def test_the_schema_was_bumped_so_an_old_manifest_is_identifiable():
    assert SCHEMA >= 2


def test_the_latent_is_in_the_arch_snapshot_because_it_changes_weight_shapes():
    """REGIME_DIM sizes a table and widens the fusion, so it is capacity, not a dial."""
    assert "REGIME_DIM" in ARCH_KEYS and "REGIME_ENABLED" in ARCH_KEYS


def test_a_matching_build_reports_nothing():
    assert feature_mismatch(feature_snapshot()) == []


def test_a_manifest_written_before_this_existed_is_an_absence_not_a_mismatch():
    """Every v1.0 and 2.0 bundle. They must keep loading."""
    assert feature_mismatch(None) == []
    assert feature_mismatch({}) == []


def test_a_changed_scalar_count_is_named():
    stale = dict(feature_snapshot(), num_roster_scalars=4)
    problems = feature_mismatch(stale)
    assert len(problems) == 1
    assert "num_roster_scalars" in problems[0]
    assert "trained with 4" in problems[0] and "this build has 14" in problems[0]


def test_a_same_width_key_swap_is_caught():
    """The case that produces no shape error anywhere, and is the reason names are recorded."""
    snap = feature_snapshot()
    swapped = dict(snap, game_state_keys=[*snap["game_state_keys"][:-1], "something_else"])
    assert any("game_state_keys" in p for p in feature_mismatch(swapped))


def test_a_key_the_recorded_signature_does_not_carry_is_ignored():
    """Forward compatibility: a manifest from an older 3.0 build that predates a later key must not
    be refused over the key it could not have known about."""
    snap = feature_snapshot()
    partial = {"num_roster_scalars": snap["num_roster_scalars"]}
    assert feature_mismatch(partial) == []


def test_every_recorded_key_is_json_serialisable():
    """It is written into manifest.json, so a tuple or a numpy scalar would break the write."""
    import json

    json.dumps(feature_snapshot())
