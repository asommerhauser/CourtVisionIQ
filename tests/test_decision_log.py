"""
Decision logging during rollout (3.2 W10).

Two properties carry the design. **A thin index, not a tensor dump** -- the context is re-derived by
replaying the sim's own play-by-play rather than stored, because ``build_model_inputs`` serves a dict
memoised per row and shared across the ~5 head calls at one position, and the prior columns alone run to
megabytes per position at ``SEQ = 600``. And **one log per worker**, because up to ``ROLLOUT_BATCH_SIZE``
slots sample concurrently on their own threads and a shared list would lose the sim identity the
advantage calculation cannot do without.
"""
import json

import pytest

from simulation.decision_log import (
    FIELDS,
    LOG_FILENAME,
    DecisionLog,
    group_by_sim,
    read_log,
    write_log,
)


# --------------------------------------------------------------------------- the record

def test_a_decision_records_where_and_what_only():
    """No tensors: the fields are the sim's identity, the position, the head and the token."""
    log = DecisionLog(game_id=7, sim_index=3)
    log.record("event_time", "event_output", 41, "shot")
    assert log.as_dicts() == [{"game_id": 7, "sim_index": 3, "position": 41,
                               "head": "event_time", "output": "event_output", "token": "shot"}]
    assert set(FIELDS) == set(log.as_dicts()[0])


def test_a_disabled_log_records_nothing():
    log = DecisionLog(1, 0, enabled=False)
    log.record("player", "player_output", 5, "A")
    assert len(log) == 0


def test_an_empty_log_is_still_truthy():
    """``if log:`` must not silently stop recording on the first decision of a game."""
    assert bool(DecisionLog(1, 0)) is True
    assert len(DecisionLog(1, 0)) == 0


def test_tokens_are_stringified_so_a_count_and_a_name_store_alike():
    """``sub_decision`` samples an integer count; every other head samples a name."""
    log = DecisionLog(1, 0)
    log.record("sub_decision", "sub_count_home_output", 9, 2)
    assert log.as_dicts()[0]["token"] == "2"


# --------------------------------------------------------------------------- persistence

def test_the_log_round_trips_through_jsonl(tmp_path):
    log = DecisionLog(11, 2)
    log.record("event_time", "event_output", 0, "shot")
    log.record("shot_result", "result_output", 1, "made")
    path = write_log(log.rows, tmp_path)

    assert path.name == LOG_FILENAME
    rows = read_log(path)
    assert [r["token"] for r in rows] == ["shot", "made"]
    assert all(set(r) == set(FIELDS) for r in rows)


def test_writing_appends_so_sims_of_one_game_accumulate(tmp_path):
    """Each sim finishes on its own thread at its own time; the file is the join point."""
    for sim_index in range(3):
        log = DecisionLog(4, sim_index)
        log.record("player", "player_output", sim_index, f"P{sim_index}")
        write_log(log.rows, tmp_path)
    rows = read_log(tmp_path / LOG_FILENAME)
    assert len(rows) == 3
    assert sorted(r["sim_index"] for r in rows) == [0, 1, 2]


def test_a_missing_log_reads_as_empty_rather_than_raising(tmp_path):
    assert read_log(tmp_path / "nope.jsonl") == []


def test_the_log_is_written_beside_the_play_by_play_not_inside_it(tmp_path):
    """``harvest.py`` prunes ``playbyplay/`` on a live run, which would take the labels with it."""
    game_dir = tmp_path / "games" / "0001"
    (game_dir / "playbyplay").mkdir(parents=True)
    path = write_log([("1", 0, 0, "event_time", "event_output", "shot")], game_dir)
    assert path.parent == game_dir
    assert "playbyplay" not in path.parts


def test_decisions_group_by_the_grain_the_advantage_is_computed_at(tmp_path):
    rows = []
    for sim_index in (0, 1):
        log = DecisionLog(9, sim_index)
        log.record("event_time", "event_output", 0, "shot")
        log.record("event_time", "event_output", 1, "rebound")
        rows.extend(log.rows)
    grouped = group_by_sim(rows)
    assert set(grouped) == {(9, 0), (9, 1)}
    assert all(len(v) == 2 for v in grouped.values())


# --------------------------------------------------------------------------- the simulator hook

class _Sim:
    """The two attributes ``_log_decision`` reads, and nothing else."""
    sequence_length = 600

    def __init__(self, n_history, log=None):
        self.history = [{}] * n_history
        self.decision_log = log

    _log_decision = None  # bound below


def _bind():
    from simulation.game_simulator import GameSimulator
    _Sim._log_decision = GameSimulator._log_decision


def test_the_hook_is_a_no_op_without_a_log():
    """Every ordinary eval runs through these call sites, so off must cost nothing but a lookup."""
    _bind()
    sim = _Sim(10, log=None)
    sim._log_decision("event_time", "event_output", "shot")   # must not raise


def test_the_logged_position_is_the_decision_index():
    """``n - 1``: the same row the heads read their logits at, so a replay of the play-by-play through
    preprocess lands on the position the head was actually asked about."""
    _bind()
    log = DecisionLog(1, 0)
    sim = _Sim(43, log=log)
    sim._log_decision("player", "player_output", "A")
    assert log.as_dicts()[0]["position"] == 42


def test_the_position_is_clamped_to_the_model_window():
    """Past ``SEQ`` the model sees a sliding window, and the decision index is its last row."""
    _bind()
    log = DecisionLog(1, 0)
    sim = _Sim(_Sim.sequence_length + 250, log=log)
    sim._log_decision("event_time", "event_output", "shot")
    assert log.as_dicts()[0]["position"] == _Sim.sequence_length - 1


# --------------------------------------------------------------------------- per-worker isolation

def test_each_worker_gets_its_own_log(monkeypatch):
    """**The property the threading model requires.**

    ``run_jobs_batched`` fills a caller-owned list, following ``boxes_out``'s shape so the return type is
    unchanged. A shared log would need a lock and would lose which sim each decision came from.
    """
    import inspect
    from simulation.batched_rollout import run_jobs_batched
    params = inspect.signature(run_jobs_batched).parameters
    assert "decisions_out" in params and "job_identity" in params
    src = inspect.getsource(run_jobs_batched)
    assert "wsim.decision_log = DecisionLog(gid, sim_index)" in src, (
        "the log must hang off the per-worker sim, not off the master")
    assert "decisions_out[job_idx] = wsim.decision_log" in src


def test_the_identity_callback_supplies_game_and_sim():
    """Without it the job index stands in, which is enough to group but not to name a game."""
    import inspect
    from simulation.batched_rollout import run_jobs_batched
    src = inspect.getsource(run_jobs_batched)
    assert "job_identity(job_idx) if job_identity is not None" in src


# --------------------------------------------------------------------------- every head is hooked

@pytest.mark.parametrize("head,output", [
    ("event_time", "event_output"),
    ("player", "player_output"),
    ("shot_result", "result_output"),
    ("substitution", "secondary_player_output"),
    ("sub_decision", "output"),
])
def test_every_sampled_head_reaches_the_log(head, output):
    """A head that never logs is a head the replay pass silently cannot train.

    The event head is checked in the controller, because it samples from there rather than from a
    ``predict_*`` wrapper -- and it is the head whose own output histogram W9 scores.
    """
    import inspect
    from simulation import controller, game_simulator
    src = inspect.getsource(game_simulator) + inspect.getsource(controller)
    assert f'_log_decision("{head}"' in src or f"_log_decision({head}" in src or \
        f'_log_decision(PlayerModel.KEY, "{output}"' in src or \
        f'_log_decision(SubstitutionModel.KEY, "{output}"' in src or \
        f"_log_decision(SubDecisionModel.KEY, {output}" in src, \
        f"{head} samples but never logs"


def test_the_conditional_type_heads_log_under_their_own_key():
    """Five heads share one ``predict_type``, so the key has to come from the argument."""
    import inspect
    from simulation.game_simulator import GameSimulator
    src = inspect.getsource(GameSimulator.predict_type)
    assert '_log_decision(key, "type_output", pick)' in src
