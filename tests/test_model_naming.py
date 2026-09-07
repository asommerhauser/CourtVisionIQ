"""
Model naming, the results layout, and the manifest.

A model NAME is the train identity: retraining picks a new name rather than overwriting, so a set
of weights and the runs evaluated against it never drift apart. These tests pin the three places
that assumption shows up on disk -- ``artifacts/<name>/``, ``results/<model>/<run>/``, and
``manifest.json`` -- plus the path-escape guard, since a name is interpolated straight into a path.
"""
from __future__ import annotations

import json

import pytest

import config
from models.artifacts import (ModelArtifacts, active_model, list_models, model_root,
                              set_active_model, version_root)


# --------------------------------------------------------------------------- #
# --- Names                                                                 -- #
# --------------------------------------------------------------------------- #

def test_model_root_matches_the_legacy_string_exactly():
    """Existing run state records this literal; a Path or trailing slash would compare unequal."""
    assert model_root("v1.0") == "./artifacts/v1.0"


def test_model_root_accepts_a_free_form_name():
    assert model_root("endgame-feats") == "./artifacts/endgame-feats"


def test_version_root_still_normalizes_a_bare_version():
    assert version_root("1.0") == version_root("v1.0") == "./artifacts/v1.0"


@pytest.mark.parametrize("bad", ["../etc", "a/b", "a\\b", "", ".", "..", ".hidden", "-lead"])
def test_model_root_rejects_path_escapes(bad):
    with pytest.raises(ValueError, match="invalid model name"):
        model_root(bad)


def test_config_default_model_and_root_cannot_disagree():
    """FULL_ARTIFACTS_ROOT is derived, so the invariant holds by construction, not by convention."""
    assert config.FULL_ARTIFACTS_ROOT == model_root(config.DEFAULT_MODEL)


def test_list_models_requires_event_time_weights(tmp_path):
    """Skips stray per-head dirs -- legacy main.py writes an unversioned ./artifacts/<head>/."""
    (tmp_path / "good" / "event_time").mkdir(parents=True)
    (tmp_path / "good" / "event_time" / "event_time.weights.h5").write_bytes(b"x")
    (tmp_path / "player").mkdir()          # looks like a model, is a stray head dir
    (tmp_path / "empty").mkdir()
    assert list_models(str(tmp_path)) == ["good"]


def test_active_model_round_trips(tmp_path):
    (tmp_path / "m1" / "event_time").mkdir(parents=True)
    (tmp_path / "m1" / "event_time" / "event_time.weights.h5").write_bytes(b"x")
    assert active_model(str(tmp_path)) is None
    set_active_model("m1", str(tmp_path))
    assert active_model(str(tmp_path)) == "m1"


def test_active_model_ignores_a_deleted_model(tmp_path):
    (tmp_path / "ACTIVE").write_text("ghost\n", encoding="utf-8")
    assert active_model(str(tmp_path)) is None


# --------------------------------------------------------------------------- #
# --- Results layout                                                        -- #
# --------------------------------------------------------------------------- #

def test_named_run_sits_under_the_model(tmp_path):
    from reporting.eval_report import resolve_results_run_dir
    d = resolve_results_run_dir("endgame-feats", name="trial2", results_root=str(tmp_path))
    assert d.parent.name == "endgame-feats" and d.name == "trial2"


def test_v_prefixed_model_keeps_its_existing_folder(tmp_path):
    """results/v1.0/ already exists on disk; the model name carries the v, nothing re-adds it."""
    from reporting.eval_report import resolve_results_run_dir
    d = resolve_results_run_dir("v1.0", name="trial2", results_root=str(tmp_path))
    assert d.parent.name == "v1.0"


def test_auto_named_run_resumes_while_incomplete(tmp_path):
    from reporting.eval_report import resolve_results_run_dir
    a = resolve_results_run_dir("m", holdout_total=10, results_root=str(tmp_path))
    b = resolve_results_run_dir("m", holdout_total=10, results_root=str(tmp_path))
    assert a == b and a.name == "eval-001"


def test_auto_named_run_increments_once_complete(tmp_path):
    from reporting.eval_report import resolve_results_run_dir
    a = resolve_results_run_dir("m", holdout_total=1, results_root=str(tmp_path))
    g = a / "games" / "game1"
    g.mkdir(parents=True)
    (g / "record.json").write_text("{}", encoding="utf-8")
    b = resolve_results_run_dir("m", holdout_total=1, results_root=str(tmp_path))
    assert b.name == "eval-002"


# --------------------------------------------------------------------------- #
# --- Manifest                                                              -- #
# --------------------------------------------------------------------------- #

def test_read_manifest_is_empty_when_absent(tmp_path):
    """Pre-manifest models must still load, so this returns {} rather than raising."""
    from models.manifest import read_manifest
    assert read_manifest(tmp_path) == {}


def test_write_manifest_merges(tmp_path):
    from models.manifest import read_manifest, write_manifest
    write_manifest(tmp_path, name="m", epochs=10)
    write_manifest(tmp_path, epochs=20, batch_size=64)
    d = read_manifest(tmp_path)
    assert d["name"] == "m" and d["epochs"] == 20 and d["batch_size"] == 64


def test_new_manifest_captures_the_arch():
    """The arch is what turns a config.py edit into a readable refusal instead of a shape error."""
    from models.manifest import new_manifest
    m = new_manifest("m")
    assert m["arch"]["MODEL_DIM"] == config.MODEL_DIM
    assert m["arch"]["NUM_LAYERS"] == config.NUM_LAYERS


def test_new_manifest_records_the_dials_in_force():
    from models.manifest import new_manifest
    config.set_dial("DELTA_TIME_SCALE", 0.99)
    assert new_manifest("m")["recommended_dials"]["DELTA_TIME_SCALE"] == 0.99


def test_record_head_appends(tmp_path):
    from models.manifest import read_manifest, record_head, write_manifest
    write_manifest(tmp_path, name="m", heads={})
    arts = ModelArtifacts.for_key("player", tmp_path)
    arts.ensure_dir()
    arts.weights_path.write_bytes(b"weights")
    record_head(tmp_path, "player")
    heads = read_manifest(tmp_path)["heads"]
    assert "player" in heads and heads["player"]["weights_bytes"] == 7


def test_snapshot_vocabs_copies_every_vocab(tmp_path):
    from encoder.encoder import Encoder
    from models.manifest import snapshot_vocabs
    enc = Encoder(vocab_dir=tmp_path / "src")
    enc.save_all()
    dest = snapshot_vocabs(enc, tmp_path / "model")
    assert {p.name for p in dest.glob("*.json")} >= {"player_vocab.json", "event_vocab.json"}


def test_holdout_recovers_from_previous_run_folders(tmp_path):
    """Folder names carry the ids, so a truncated processed manifest is not fatal."""
    from shell.session import holdout_from_results
    for gid in (298324, 298325):
        (tmp_path / "v1.0" / "run1" / "games" / f"game{gid}_2023-01-10_AWAYatHOME").mkdir(parents=True)
    assert holdout_from_results("v1.0", results_root=str(tmp_path)) == [298324, 298325]


def test_holdout_prefers_the_manifest(tmp_path):
    from shell.session import Session
    s = Session(processed_dir=str(tmp_path))
    s.model = "v1.0"
    s.manifest = {"holdout_game_ids": [7, 8, 9]}
    ids, src = s.resolve_holdout()
    assert ids == [7, 8, 9] and "manifest" in src


def test_holdout_skips_an_empty_processed_manifest(tmp_path, monkeypatch):
    """The real failure seen on disk: holdout_games.json present but containing [].

    ``processed_dir`` is not the only place ``resolve_holdout`` looks: two of its fallbacks --
    ``./training/full_run_state.json`` and ``./results/<model>/`` -- are resolved against the
    CWD, so on a machine that has actually trained, the real run state answers first and this
    test never reaches the raise. ``chdir`` into ``tmp_path`` isolates both.
    """
    from shell.session import Session
    monkeypatch.chdir(tmp_path)
    (tmp_path / config.HOLDOUT_MANIFEST_NAME).write_text("[]", encoding="utf-8")
    s = Session(processed_dir=str(tmp_path))
    s.model = "nonexistent-model"
    with pytest.raises(FileNotFoundError, match="present but empty"):
        s.resolve_holdout()
