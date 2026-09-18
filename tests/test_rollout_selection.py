"""
W4 rung 2: selecting a checkpoint by what it simulates, and coexisting with EarlyStopping.

The interaction is the whole risk here, and it is silent when it goes wrong.
``EarlyStopping(restore_best_weights=True)`` restores at ``on_train_end`` -- after every callback's
``on_epoch_end`` and before ``save_artifacts``. A selector that restores the weights it likes is
therefore overwritten, and the run reports a rollout-selected epoch that is not what reached disk.
Nothing downstream can tell. So the selector never restores, ``train()`` does, and the test below
proves the order rather than assuming it.

The scoring policy is driven with an injected ``score_fn``, so the whole selection mechanism is
testable without a simulator, a GPU or a trained bundle.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import config
from models.rollout_selection import (
    STATE_KEY,
    apply_selection,
    build_selector,
    eval_game_ids,
    record_selection,
    rollout_score,
)


# --------------------------------------------------------------------------- the score

def _agg(points_mae=7.0, dispersion=1.0):
    return {"headline": {"points_mae": points_mae},
            "coverage": {"margin": {"dispersion_ratio": dispersion}}}


def test_a_perfectly_calibrated_run_scores_its_box_error_and_nothing_else():
    assert rollout_score(_agg(7.0, 1.0), {"rows": [{"sim": 0.5, "real": 0.5}]}) == pytest.approx(7.0)


def test_over_dispersion_costs_something():
    """W3 is trying to move this, so a checkpoint must not be able to ignore it."""
    assert rollout_score(_agg(7.0, 1.36)) > rollout_score(_agg(7.0, 1.0))


def test_collapsing_the_spread_to_flatter_the_box_does_not_win():
    """The failure this term exists for: a checkpoint that predicts the mean every time has a
    decent box MAE and no distribution at all."""
    tight = rollout_score(_agg(6.5, 0.4))
    honest = rollout_score(_agg(7.0, 1.0))
    assert tight > honest


def test_missing_game_state_behaviour_costs_something():
    """Without this term nothing in the objective notices that the simulator never benches a
    player in foul trouble -- the measured sim rate is 0.238 against a real 0.776."""
    absent = rollout_score(_agg(), {"rows": [{"sim": 0.238, "real": 0.776}]})
    present = rollout_score(_agg(), {"rows": [{"sim": 0.776, "real": 0.776}]})
    assert absent > present


def test_a_probe_that_could_not_be_computed_contributes_nothing_rather_than_agreement():
    """A None sim rate must not read as a perfect match, which is what a 0.0 would do."""
    missing = rollout_score(_agg(), {"rows": [{"sim": None, "real": 0.776}]})
    assert missing == pytest.approx(rollout_score(_agg()))


def test_an_empty_aggregate_does_not_raise():
    assert rollout_score({}) == 0.0
    assert rollout_score({}, None) == 0.0


# --------------------------------------------------------------------------- the eval game set

def _state(n=800, boundary=700):
    return {"boundary_idx": boundary,
            "train_tail_game_ids": list(range(1000, 1000 + n)),
            "holdout_game_ids": list(range(1000 + n, 1000 + n + 700))}


def test_the_eval_games_come_from_the_training_tail_and_never_the_holdout():
    """Selecting a checkpoint against holdout games turns the report into a training metric. It is
    the most expensive way to fool yourself here, because nothing downstream would look wrong."""
    state = _state()
    games = eval_game_ids(state, n_games=20)
    assert len(games) == 20
    assert set(games) <= set(state["train_tail_game_ids"])
    assert not set(games) & set(state["holdout_game_ids"])


def test_the_sample_is_drawn_from_the_recent_tail_not_the_whole_corpus():
    state = _state(n=5000)
    games = eval_game_ids(state, n_games=20)
    assert min(games) >= state["train_tail_game_ids"][-config.ROLLOUT_EVAL_TAIL]


def test_the_sample_is_stable_for_a_seed_and_moves_with_it():
    state = _state()
    assert eval_game_ids(state, seed=0) == eval_game_ids(state, seed=0)
    assert eval_game_ids(state, seed=0) != eval_game_ids(state, seed=1)


def test_a_state_with_no_tail_recorded_yields_nothing_rather_than_guessing():
    assert eval_game_ids({"boundary_idx": 10}) == []


# --------------------------------------------------------------------------- selection + restore

class _Model:
    """Just enough model for the callback: weights that can be read and written."""

    def __init__(self):
        self.weights = [np.zeros(3)]

    def get_weights(self):
        return [np.array(w, copy=True) for w in self.weights]

    def set_weights(self, w):
        self.weights = [np.array(x, copy=True) for x in w]


class _History:
    def __init__(self, val_losses):
        self.history = {"val_loss": list(val_losses)}


def _run(scores, monkeypatch, every=1):
    """Drive the callback by hand over len(scores) epochs; return (model, selector)."""
    monkeypatch.setattr(config, "ROLLOUT_SELECTION", True)
    model = _Model()
    selector = build_selector(model, score_fn=lambda e: scores[e], every=every)
    for epoch in range(len(scores)):
        model.weights = [np.full(3, float(epoch))]
        logs = {}
        selector.on_epoch_end(epoch, logs)
    return model, selector


def test_the_selector_keeps_the_best_scoring_epoch(monkeypatch):
    model, selector = _run([5.0, 3.0, 9.0, 4.0], monkeypatch)
    assert selector.selection.best_epoch == 1
    assert selector.best_weights[0].tolist() == [1.0, 1.0, 1.0]


def test_the_selector_does_not_restore_on_its_own(monkeypatch):
    """If it did, EarlyStopping would overwrite it at on_train_end and the record would lie."""
    model, selector = _run([5.0, 3.0, 9.0], monkeypatch)
    assert model.weights[0].tolist() == [2.0, 2.0, 2.0], "the model was mutated mid-training"


def test_the_restore_happens_after_early_stopping_would_have_run(monkeypatch):
    """The ordering, end to end: EarlyStopping restores epoch 2, then apply_selection wins."""
    model, selector = _run([5.0, 3.0, 9.0, 4.0], monkeypatch)
    # EarlyStopping's restore, simulated: it puts back the best-val_loss epoch's weights.
    model.set_weights([np.full(3, 2.0)])
    record = apply_selection(model, selector, history=_History([9.0, 8.0, 1.0, 7.0]))
    assert model.weights[0].tolist() == [1.0, 1.0, 1.0], "rung 2 must have the last word"
    assert record["nll_best_epoch"] == 2
    assert record["rollout_best_epoch"] == 1
    assert record["epochs_disagree"] is True


def test_when_the_two_criteria_agree_the_record_says_so(monkeypatch):
    """docs/v3_direction.md §6 step 6 fires only on disagreement, so this has to be a number."""
    model, selector = _run([5.0, 3.0, 9.0], monkeypatch)
    record = apply_selection(model, selector, history=_History([9.0, 1.0, 7.0]))
    assert record["nll_best_epoch"] == record["rollout_best_epoch"] == 1
    assert record["epochs_disagree"] is False
    assert record["rollout_score_at_nll_best"] == pytest.approx(3.0)


def test_only_one_snapshot_is_held_at_a_time(monkeypatch):
    """A full head is ~250 MB; keeping every candidate would cost more RAM than the train."""
    _model, selector = _run([5.0, 4.0, 3.0, 2.0], monkeypatch)
    assert isinstance(selector.best_weights, list) and len(selector.best_weights) == 1


def test_the_selector_only_scores_every_nth_epoch(monkeypatch):
    monkeypatch.setattr(config, "ROLLOUT_SELECTION", True)
    seen = []
    model = _Model()
    selector = build_selector(model, score_fn=lambda e: (seen.append(e), 1.0)[1], every=3)
    for epoch in range(9):
        selector.on_epoch_end(epoch, {})
    assert seen == [2, 5, 8]


def test_with_rung_two_off_there_is_no_selector_and_nothing_is_restored(monkeypatch):
    """Selection off: weights are exactly what EarlyStopping chose and the run is unchanged.

    3.2 W8 turned ROLLOUT_SELECTION on, so this sets it explicitly rather than leaning on the default
    -- which is what a test of the OFF path should have done from the start.
    """
    monkeypatch.setattr(config, "ROLLOUT_SELECTION", False)
    assert build_selector(_Model(), score_fn=lambda e: 1.0) is None
    model = _Model()
    model.set_weights([np.full(3, 7.0)])
    assert apply_selection(model, None, history=None) is None
    assert model.weights[0].tolist() == [7.0, 7.0, 7.0]


def test_a_selector_that_never_scored_leaves_the_weights_alone(monkeypatch):
    """Training shorter than one evaluation interval must not blank the model."""
    monkeypatch.setattr(config, "ROLLOUT_SELECTION", True)
    model = _Model()
    model.set_weights([np.full(3, 7.0)])
    selector = build_selector(model, score_fn=lambda e: 1.0, every=10)
    selector.on_epoch_end(0, {})
    assert apply_selection(model, selector) is None
    assert model.weights[0].tolist() == [7.0, 7.0, 7.0]


# --------------------------------------------------------------------------- the record

def test_the_record_merges_into_the_run_state_without_disturbing_it(tmp_path):
    path = tmp_path / "full_run_state.json"
    path.write_text(json.dumps({"status": "trained", "trained_models": ["event_time"]}))
    record_selection(path, "event_time", {"rollout_best_epoch": 4})
    state = json.loads(path.read_text())
    assert state["trained_models"] == ["event_time"]
    assert state[STATE_KEY]["event_time"]["rollout_best_epoch"] == 4


# --------------------------------------------------------------------------- #
# --- W8: the bridge that makes any of the above run                        -- #
# --------------------------------------------------------------------------- #

def _probe_report(sim_bench=0.238, real_bench=0.776):
    """A ``compare()``-shaped report, with every row ``_rows_for_frame`` reads."""
    def side(bench):
        return {
            "foul_trouble_3": {"p_benched": bench, "events_per_game": 1.4, "n_events": 1570},
            "foul_trouble_4": {"p_benched": bench, "events_per_game": 0.37, "n_events": 51},
            "blowout_q4": {"starter_seconds_blowout": 3321.0, "starter_seconds_close": 3963.0,
                           "ratio": 0.838, "blowout_frequency": 0.22,
                           "blowout_games": 20, "close_games": 80},
            "late_foul": {"rate_per_100s": 1.62, "state_seconds_per_game": 30.1, "fouls": 100},
        }
    return {"sim": side(sim_bench), "real": side(real_bench)}


def test_the_scored_rows_are_the_deliberate_subset():
    """``_rows_for_frame`` emits eight rows; three describe one behaviour and one is a denominator.

    ``rollout_score`` averages relative gaps, so units cancel -- but weighting the Q4 rotation three
    times over (blowout seconds, close seconds, and their ratio) would make it three fifths of the
    behaviour term on its own.
    """
    from models.rollout_bridge import SCORED_PROBES, scored_probe_rows
    rows = scored_probe_rows(_probe_report())["rows"]
    assert {(r["probe"], r["metric"]) for r in rows} == set(SCORED_PROBES)
    assert len(rows) == 5

    from reporting.state_probes import _rows_for_frame
    assert len(_rows_for_frame(_probe_report())) == 10, "the full set is wider than what is scored"


def test_the_excluded_rows_really_are_excluded():
    from models.rollout_bridge import scored_probe_rows
    scored = {(r["probe"], r["metric"]) for r in scored_probe_rows(_probe_report())["rows"]}
    for excluded in (("blowout_q4", "starter_seconds_blowout"),
                     ("blowout_q4", "starter_seconds_close"),
                     ("blowout_q4", "blowout_frequency"),
                     ("foul_trouble_3", "events_per_game"),
                     ("late_foul", "state_seconds_per_game")):
        assert excluded not in scored


def test_the_selected_rows_feed_the_behaviour_term():
    """The end the bridge exists for: a probe gap has to cost something in the score."""
    from models.rollout_bridge import scored_probe_rows
    agg = _agg(7.0, 1.0) if "_agg" in globals() else {"headline": {"points_mae": 7.0}}
    matched = rollout_score(agg, scored_probe_rows(_probe_report(0.776, 0.776)))
    missed = rollout_score(agg, scored_probe_rows(_probe_report(0.238, 0.776)))
    assert missed > matched


def test_a_run_state_without_the_train_tail_abstains_rather_than_scoring_nothing():
    """The 3.0 gap: ``eval_game_ids`` read a key nothing wrote, so it returned an empty list.

    Building a score function over no games would have selected a checkpoint on a constant, and
    nothing downstream would have looked wrong.
    """
    from models.rollout_bridge import build_rollout_score_fn
    said = []
    fn = build_rollout_score_fn({"boundary_idx": 100}, make_sim=lambda: None, echo=said.append)
    assert fn is None
    assert said and "train_tail_game_ids" in said[0]


def test_setup_records_the_train_tail(tmp_path):
    """The other half: ``FullRun.setup`` has to write the key ``eval_game_ids`` reads."""
    import json
    import config
    from training.full_run import FullRun
    from test_chronology import _build_corpus

    data_dir = _build_corpus(tmp_path)
    state = tmp_path / "state.json"
    run = FullRun(str(state))
    run.setup(name="probe", data_dir=str(data_dir), processed_dir=str(tmp_path / "processed"))

    saved = json.loads(state.read_text(encoding="utf-8"))
    tail = saved["train_tail_game_ids"]
    assert tail, "setup must record the training-era tail"
    assert len(tail) <= config.ROLLOUT_EVAL_TAIL
    assert eval_game_ids(saved, n_games=2), "and eval_game_ids must be able to sample from it"


def test_the_tail_never_reaches_into_the_holdout(tmp_path):
    """Selecting against holdout games turns the report into a training metric."""
    import json
    from training.full_run import FullRun
    from test_chronology import _build_corpus

    data_dir = _build_corpus(tmp_path)
    state = tmp_path / "state.json"
    FullRun(str(state)).setup(name="probe", data_dir=str(data_dir),
                             processed_dir=str(tmp_path / "processed"))
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert not (set(saved["train_tail_game_ids"]) & set(saved["holdout_game_ids"]))


def test_the_pipeline_asks_the_factory_for_the_head_it_built():
    """``run_stage`` constructs the heads, so a ready callable could not reach this epoch's weights.

    The factory is handed the head instance, whose ``_live_model`` the score function reads.
    """
    import inspect
    from models.pipeline import run_stage
    assert "rollout_score_fn_factory" in inspect.signature(run_stage).parameters
    src = inspect.getsource(run_stage)
    assert "rollout_score_fn_factory(model)" in src
    assert 'extra["rollout_score_fn"]' in src


def test_only_the_event_time_head_is_offered_the_score_function():
    """The other five head classes have the identical signature WITHOUT the parameter."""
    import inspect
    from models.event_time_model import EventTimeModel
    from models.player_model import PlayerModel
    from models.substitution_model import SubstitutionModel

    assert "rollout_score_fn" in inspect.signature(EventTimeModel.train).parameters
    for cls in (PlayerModel, SubstitutionModel):
        assert "rollout_score_fn" not in inspect.signature(cls.train).parameters


def test_the_head_exposes_the_inner_model_not_the_trainer():
    """``build_trainer`` may add a regime table, so the wrapper's weight list is longer than the
    simulator's graph. Copying from the wrapper would raise on length."""
    import inspect
    from models.event_time_model import EventTimeModel
    src = inspect.getsource(EventTimeModel.train)
    assert "self._live_model = inner" in src, (
        "_live_model must be the inner functional model; `model` is rebound to the trainer, whose "
        "weight list also holds the regime latent table the simulator's graph does not have")
    assert "self._live_model = model" not in src


def test_the_sim_side_of_the_probes_is_pooled_from_histories():
    """Computed in memory: writing thousands of play-by-play CSVs per scored epoch to read them
    straight back would cost more than the rollouts."""
    from models.rollout_bridge import sim_probe_summary
    summary = sim_probe_summary([])
    assert set(summary) >= {"foul_trouble_3", "foul_trouble_4", "blowout_q4", "late_foul"}
