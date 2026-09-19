"""The replay pass's seams: how a weight reaches a split, and what happens when it cannot."""
import numpy as np
import pytest

from training.replay_corpus import sim_game_id
from training.replay_pass import merge_weights, reweight_split, sim_probe_report


def _split(tmp_path, name, n_games, **extra):
    path = tmp_path / name
    np.savez_compressed(path, recency_weight=np.ones((n_games,), dtype=np.float32),
                        pad_mask=np.ones((n_games, 3), dtype=np.float32), **extra)
    return path


def test_weights_land_in_sorted_game_order(tmp_path):
    ids = [sim_game_id(7, 2), sim_game_id(7, 0), sim_game_id(7, 1)]
    _split(tmp_path, "train.npz", 3)
    reweight_split(tmp_path, ids, {sim_game_id(7, 1): 2.5}, echo=lambda *_: None)

    with np.load(tmp_path / "train.npz") as data:
        weights = data["recency_weight"]
    # sorted(ids) is sim 0, 1, 2 -- so the weighted sim is the middle row, not the one listed second.
    assert weights.tolist() == [0.0, 2.5, 0.0]


def test_a_filtered_sim_contributes_nothing_rather_than_being_removed(tmp_path):
    ids = [sim_game_id(7, i) for i in range(4)]
    _split(tmp_path, "player_train.npz", 4)
    reweight_split(tmp_path, ids, {}, echo=lambda *_: None)

    with np.load(tmp_path / "player_train.npz") as data:
        assert data["recency_weight"].tolist() == [0.0] * 4
        assert data["pad_mask"].shape == (4, 3)      # the tensors themselves are untouched


def test_every_head_train_file_is_rewritten_whatever_it_is_called(tmp_path):
    ids = [sim_game_id(7, i) for i in range(2)]
    for name in ("train.npz", "player_train.npz", "cond_train.npz"):
        _split(tmp_path, name, 2)
    _split(tmp_path, "test.npz", 2)                  # validation split: left alone

    touched = reweight_split(tmp_path, ids, {ids[0]: 1.0}, echo=lambda *_: None)
    assert {p.name for p in touched} == {"train.npz", "player_train.npz", "cond_train.npz"}
    with np.load(tmp_path / "test.npz") as data:
        assert data["recency_weight"].tolist() == [1.0, 1.0]


def test_a_length_mismatch_refuses_instead_of_misaligning(tmp_path):
    """The failure this check exists for: every weight after a dropped game lands one row early."""
    _split(tmp_path, "train.npz", 5)
    with pytest.raises(ValueError, match="cannot be aligned"):
        reweight_split(tmp_path, [sim_game_id(7, i) for i in range(6)], {}, echo=lambda *_: None)


def test_merge_weights_keeps_both_chunks():
    into = {"player": {1: 1.0}}
    merge_weights(into, {"player": {2: 2.0}, "shot_type": {3: 3.0}})
    assert into == {"player": {1: 1.0, 2: 2.0}, "shot_type": {3: 3.0}}


def test_probe_report_is_in_the_shape_probe_gap_reads():
    from models.head_metrics import probe_gap

    rows = [{"game_id": 1, "time": 0, "event": "start", "player": "start", "type": "start",
             "result": "start", "secondary_player": "none",
             "roster_home": str(["A", "B", "C", "D", "E"]),
             "roster_away": str(["F", "G", "H", "I", "J"])}]
    real = {"foul_trouble_3": {"n_events": 10, "events_per_game": 1.0, "p_benched": 0.8},
            "foul_trouble_4": {"n_events": 10, "events_per_game": 1.0, "p_benched": 0.9},
            "foul_trouble_5": {"n_events": 10, "events_per_game": 1.0, "p_benched": 0.9},
            "blowout_q4": {"blowout_games": 5, "close_games": 5, "blowout_frequency": 0.5,
                           "starter_seconds_blowout": 300.0, "starter_seconds_close": 600.0,
                           "ratio": 0.5},
            "late_foul": {"fouls": 4, "state_seconds": 100.0, "state_seconds_per_game": 10.0,
                          "rate_per_100s": 4.0},
            "n_games": 10}
    report = sim_probe_report(rows, real)
    assert isinstance(report["rows"], list) and report["rows"]
    # A one-game sim fills almost none of the probe rows; probe_gap has to survive that, because most
    # single games contain no blowout fourth quarter and no late-foul state at all.
    assert probe_gap(report) >= 0.0
