"""
Rung 2's bridge: the defects found by actually running it, over two nights.

On 2026-09-22 ``python train.py --model event_time --name version3.2`` reached epoch 3 -- the first
``ROLLOUT_EVAL_EVERY`` boundary -- and went quiet. Nine and a half hours later it had produced no
output of any kind, held 20 GB of RSS, and was burning 1.2 cores at 1% GPU utilisation. On
2026-09-23, with the inference path fixed and a heartbeat added, the same evaluation got all the way
to the scoring step in 62 minutes and died there on a shape mismatch nobody had ever reached. Each
defect gets a test here:

* **Nothing timed or bounded an evaluation.** A rollout that never returns and a rollout that is
  merely slow look identical from outside, so the run could not be judged while it was running.
  ``_check_budget`` turns the second into an abort carrying the measured number.
* **Nothing reported progress.** One evaluation was a single blocking call. It now runs in
  ``simulate_games`` streaming mode, which hands each finished sim back as it lands, so the bridge
  can project the total within the first minute. That rewiring also moves where the probe histories
  come from, which is the silent-wrong risk the streaming test covers.
* **The corpus was parsed whole to keep twenty games.** ``load_all_cleaned(parse_rosters=True)``
  runs ``ast.literal_eval`` per row; doing that over 21 seasons to slice out the scored games is
  where the RSS went. The slice now happens inside the load.

* **The real side was never summarized.** ``probe_real`` returns counts and the scorer reads rates,
  so the two arguments to one comparison had different shapes -- invisible until an evaluation
  survived long enough to score.
* **The compiled path retraced per call.** ``reduce_retracing`` alone did not collapse a sequence
  axis that grows one event at a time; the signature now declares it dynamic.

The budget, corpus and summarize groups are pure and need no simulator. The streaming test is the
elaborate one, built on stubs: if it fails, suspect the stubs before the code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
import models.rollout_bridge as rb
from models.rollout_bridge import RolloutBudgetExceeded, _check_budget


# --------------------------------------------------------------------------- the budget

def test_an_evaluation_inside_the_budget_says_nothing(monkeypatch):
    monkeypatch.setattr(config, "ROLLOUT_EVAL_BUDGET_MIN", 25.0)
    assert _check_budget(24.9, epoch=2) is None


def test_an_evaluation_over_the_budget_aborts_with_the_measured_number(monkeypatch):
    """The abort must carry the number, because the number is the decision.

    A bare "too slow" leaves the operator exactly where they started. The message names the measured
    minutes, the budget they broke, and the per-epoch cost implied for the rest of the run.
    """
    monkeypatch.setattr(config, "ROLLOUT_EVAL_BUDGET_MIN", 25.0)
    monkeypatch.setattr(config, "ROLLOUT_EVAL_EVERY", 3)
    with pytest.raises(RolloutBudgetExceeded) as excinfo:
        _check_budget(570.0, epoch=2)          # the 9.5 hours actually observed
    message = str(excinfo.value)
    assert "570.0 min" in message
    assert "25 min budget" in message
    assert "epoch 3" in message                # 0-based epoch, reported 1-based


def test_a_zero_budget_disables_the_guard(monkeypatch):
    """Zero means "no ceiling", so a deliberately long pass is expressible without editing code."""
    monkeypatch.setattr(config, "ROLLOUT_EVAL_BUDGET_MIN", 0)
    assert _check_budget(10_000.0, epoch=0) is None


# --------------------------------------------------------------------------- the corpus slice

def _cleaned_csv(path, game_ids, season=2023):
    """A minimal cleaned season file -- enough columns for ``cleaned_csvs`` to accept it."""
    rows = []
    for gid in game_ids:
        for i in range(3):
            rows.append({"game_id": gid, "season": season, "event_num": i,
                         "roster_home": "['A', 'B']", "roster_away": "['C', 'D']"})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_the_loader_slices_before_it_parses_rosters(tmp_path):
    """``game_ids`` keeps only the wanted games, and the survivors still get real lists.

    The ordering is the point. Parsing first and filtering second gives the same answer at the cost
    of an ``ast.literal_eval`` per row over the whole corpus, in the training process, for rows the
    caller is about to discard.
    """
    from data_loading import load_all_cleaned
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _cleaned_csv(data_dir / "season2023.csv", [1, 2, 3, 4])

    out = load_all_cleaned(str(data_dir), parse_rosters=True, game_ids=[2, 3])

    assert sorted(int(g) for g in out["game_id"].unique()) == [2, 3]
    assert out["roster_home"].iloc[0] == ["A", "B"]     # decoded, not left as a string
    assert len(out) == 6                                # two games x three rows


def test_no_game_ids_still_means_the_whole_corpus(tmp_path):
    """The filter is opt-in: every other caller must see exactly what it saw before."""
    from data_loading import load_all_cleaned
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _cleaned_csv(data_dir / "season2023.csv", [1, 2, 3, 4])

    assert len(load_all_cleaned(str(data_dir))) == 12


# --------------------------------------------------------------------------- streaming histories

class _FakeSim:
    """Just enough simulator for the bridge, which only sets an attribute on it."""


def test_the_probes_read_the_streamed_histories(monkeypatch):
    """Streaming mode returns EMPTY history lists by contract, so the probes must read the stream.

    This is the silent-wrong case in the rewiring. ``simulate_games(on_sim=...)`` hands each history
    over as it lands and returns ``[]`` in its place, so a bridge that kept reading the return value
    would pool the probes over nothing: every behaviour term would score against an empty sim side,
    the selection would still produce a number, and nothing in the output would look wrong.
    """
    seen: dict = {}

    def _fake_simulate_games(sim, games, *, n_sims, seed0, batch_size, game_ids, on_sim):
        for g in range(len(games)):
            for s in range(n_sims):
                on_sim(g, s, [{"event": "SHOT"}], object())
        return [([object()] * n_sims, []) for _ in games]   # boxes kept, histories emptied

    def _capture(histories):
        seen["histories"] = list(histories)
        return {}

    import simulation.eval_metrics as em
    import simulation.evaluation as ev
    import simulation.game_input as gi
    import reporting.state_probes as sp

    monkeypatch.setattr(ev, "simulate_games", _fake_simulate_games)
    monkeypatch.setattr(ev, "_real_starters", lambda df: (["A"], ["C"]))
    monkeypatch.setattr(ev, "build_game_record", lambda df, boxes, n_sims: {"gid": 1})
    monkeypatch.setattr(em, "_aggregate", lambda records: {"headline": {}})
    monkeypatch.setattr(gi, "extract_game_input", lambda df, data_dir: object())
    # A blank POOL, not a summary: the bridge summarizes it itself, and stubbing the summarized
    # shape here would hide exactly the bug that killed the first real evaluation.
    monkeypatch.setattr(sp, "probe_real", lambda *a, **kw: sp._blank())
    monkeypatch.setattr(rb, "sim_probe_summary", _capture)
    monkeypatch.setattr(rb, "scored_probe_rows", lambda report: {"rows": []})
    monkeypatch.setattr(rb, "rollout_score", lambda aggregate, probes: 1.25)
    monkeypatch.setattr(config, "ROLLOUT_EVAL_BUDGET_MIN", 0)

    frame = pd.DataFrame({"game_id": [7, 7], "event_num": [0, 1]})
    monkeypatch.setattr("data_loading.load_all_cleaned", lambda *a, **kw: frame)

    state = {"boundary_idx": 10, "train_tail_game_ids": [7]}
    score_fn = rb.build_rollout_score_fn(
        state, make_sim=_FakeSim, live_model=None, n_games=1, n_sims=2, echo=None)

    assert score_fn(0) == 1.25
    assert len(seen["histories"]) == 2                  # one per sim, from the stream
    assert seen["histories"][0] == [{"event": "SHOT"}]  # and the rows survived the handover


# --------------------------------------------------------------------------- the real side

def test_the_real_side_is_summarized_not_the_raw_pool():
    """``probe_real`` returns COUNTS; the scorer reads RATES. The bridge has to summarize.

    This is what actually killed the first evaluation that ever reached the scoring step
    (``KeyError: 'foul_trouble_3'``, 2026-09-23). It survived review because the two sides of the
    same comparison were built by different code: the sim side goes through ``sim_probe_summary``,
    which summarizes, and the real side did not. A pool has ``foul_events``; a summary has
    ``foul_trouble_3``.
    """
    import reporting.state_probes as sp
    pool = sp._blank()
    assert "foul_trouble_3" not in pool          # the shape the bridge used to hand over
    assert "foul_trouble_3" in sp.summarize(pool)  # the shape _rows_for_frame requires


# --------------------------------------------------------------------------- the inference path

def test_the_compiled_forward_honours_a_per_simulator_override(monkeypatch):
    """Rung 2 opts ONE simulator in without flipping ``CVIQ_TF_INFER`` for every other path.

    ``enabled=True`` against a model that only accepts numpy proves the compiled branch was entered:
    the ``tf.function`` call raises, and the signature is pinned to ``_EAGER``. A cache left empty
    means the branch was never taken at all.
    """
    import simulation.game_simulator as gs
    monkeypatch.setattr(gs, "_TF_INFER_ENABLED", False)

    class _NumpyOnlyModel:
        def __call__(self, inputs, training=False):
            if not isinstance(inputs["x"], np.ndarray):
                raise TypeError("tensors not accepted")
            return {"y": np.zeros((1, 1), dtype="float32")}

    inputs = {"x": np.zeros((1, 3), dtype="float32")}

    default_cache: dict = {}
    gs._compiled_forward(default_cache, _NumpyOnlyModel(), "m", inputs)
    assert default_cache == {}                       # module default off -> straight to eager

    opted_in: dict = {}
    gs._compiled_forward(opted_in, _NumpyOnlyModel(), "m", inputs, enabled=True)
    assert opted_in[("m", ("x",))] is gs._EAGER      # compiled branch taken, then pinned back

    monkeypatch.setattr(gs, "_TF_INFER_ENABLED", True)
    opted_out: dict = {}
    gs._compiled_forward(opted_out, _NumpyOnlyModel(), "m", inputs, enabled=False)
    assert opted_out == {}                           # explicit False beats the module default


def test_the_trace_signature_frees_the_batch_and_sequence_axes():
    """One trace per key-set, or the rollout pays a ~10-minute compile per new sequence length.

    A rollout grows its sequence one event at a time and its batch shrinks as sims finish, so a
    signature that pins either axis retraces for hundreds of distinct shapes. Everything past the
    sequence axis stays static, because the built layers were constructed against those sizes.
    """
    import tensorflow as tf
    from simulation.game_simulator import _dynamic_specs

    specs = _dynamic_specs(tf, {
        "event": np.zeros((4, 37), dtype="int32"),            # (batch, seq)
        "home_roster": np.zeros((4, 37, 5), dtype="int32"),   # (batch, seq, roster)
        "prior_home": np.zeros((4, 37, 5, 17), dtype="float32"),
    })

    assert specs["event"].shape.as_list() == [None, None]
    assert specs["home_roster"].shape.as_list() == [None, None, 5]
    assert specs["prior_home"].shape.as_list() == [None, None, 5, 17]
    assert specs["prior_home"].dtype == tf.float32
    assert specs["event"].dtype == tf.int32
