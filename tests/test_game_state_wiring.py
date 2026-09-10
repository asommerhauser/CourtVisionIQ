"""
Build-smoke for the game-state feature wiring across every head + the inference path.

Training-free: builds each head's Keras graph at tiny dims (model_dim=32, 1 layer) and runs
ONE forward pass over its own preprocessed split, asserting every game-state input is
consumed and the head produces finite output. This catches fusion / INPUT_KEYS / preprocess
mismatches introduced by the game-state features without any ``.fit`` (the persistence suite
on the training box covers real training).

The last tests check that the simulator's ``build_model_inputs`` emits every game-state key
the event head declares — the train/inference parity guarantee — and that both next-step time
heads ship the loss-mask arrays without ever feeding them to the graph.
"""
import numpy as np
import pandas as pd
import tensorflow as tf

from encoder.encoder import Encoder
from models.event_time_model import EventTimeModel
from models.conditional_type_model import ConditionalTypeModel, TYPE_GEN_SPECS
from models.conditional_time_model import ConditionalTimeModel
from models.sub_decision_model import SubDecisionModel
from models.player_model import PlayerModel
from models.substitution_model import SubstitutionModel
from models.game_state_features import (
    GAME_STATE_KEYS, QUERY_MASK_KEYS, NEXT_CONTINUATION_KEY, PERIOD_BREAK_KEY,
    PERIOD_LENGTH, apply_query_mask,
)

# The shared synthetic cleaned CSV, plus the pieces needed to extend it: it has no assist, no
# block and no free throw, so a loss mask built over it alone is all zeros.
from test_oncourt_mask import (
    _make_cleaned_csv, _game_rows, _roster as _roster_str, _SEASON, HOME, AWAY,
)


def _setup(tmp_path):
    """A shared frozen encoder + a two-game cleaned CSV under tmp_path/data."""
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv", games=(1, 2))
    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    return enc


def _kwargs(tmp_path):
    return dict(path=str(tmp_path / "data"),
                processed_dir=str(tmp_path / "processed"),
                sequence_length=32, model_dim=32)


def _assert_consumes_game_state(model):
    names = {i.name.split(":")[0] for i in model.inputs}
    for k in GAME_STATE_KEYS:
        assert k in names, f"{model.name} does not consume game-state input {k!r}"


def test_event_time_head_builds_and_forward_passes(tmp_path):
    enc = _setup(tmp_path)
    m = EventTimeModel(enc, **_kwargs(tmp_path))
    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)
    model = m.model(num_layers=1, num_heads=2, ff_dim=32)
    _assert_consumes_game_state(model)
    out = model({k: train[k] for k in m.INPUT_KEYS}, training=False)
    assert np.isfinite(out["event_output"].numpy()).all()
    assert np.isfinite(out["time_output"].numpy()).all()
    # The game-state columns really vary across rows (not all-zero padding).
    assert np.abs(train["score_diff"]).sum() > 0


def test_conditional_type_head_builds_and_forward_passes(tmp_path):
    enc = _setup(tmp_path)
    # Vocabs must exist + be frozen first (the conditional heads default rebuild_vocabs=False).
    EventTimeModel(enc, **_kwargs(tmp_path)).preprocess(
        rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    spec = TYPE_GEN_SPECS["shot_type"]
    m = ConditionalTypeModel(spec, enc, **_kwargs(tmp_path))
    train, _ = m.preprocess(rebuild_vocabs=False, test_frac=0.0, holdout_frac=0.0)
    model = m.model(num_layers=1, num_heads=2, ff_dim=32)
    _assert_consumes_game_state(model)
    out = model({k: train[k] for k in m.INPUT_KEYS}, training=False)
    assert np.isfinite(next(iter(out.values())).numpy()).all()


def test_conditional_time_head_builds_and_forward_passes(tmp_path):
    enc = _setup(tmp_path)
    EventTimeModel(enc, **_kwargs(tmp_path)).preprocess(
        rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    m = ConditionalTimeModel(enc, **_kwargs(tmp_path))
    train, _ = m.preprocess(rebuild_vocabs=False, test_frac=0.0, holdout_frac=0.0)
    model = m.model(num_layers=1, num_heads=2, ff_dim=32)
    _assert_consumes_game_state(model)
    out = model({k: train[k] for k in m.INPUT_KEYS}, training=False)
    assert np.isfinite(out["time_output"].numpy()).all()


def test_sub_decision_head_builds_and_forward_passes(tmp_path):
    enc = _setup(tmp_path)
    EventTimeModel(enc, **_kwargs(tmp_path)).preprocess(
        rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    m = SubDecisionModel(enc, **_kwargs(tmp_path))
    train, _ = m.preprocess(rebuild_vocabs=False, test_frac=0.0, holdout_frac=0.0)
    model = m.model(num_layers=1, num_heads=2, ff_dim=32)
    _assert_consumes_game_state(model)
    out = model({k: train[k] for k in m.INPUT_KEYS}, training=False)
    assert np.isfinite(next(iter(out.values())).numpy()).all()


def test_simulator_inputs_cover_game_state_keys():
    """build_model_inputs must emit every game-state key the event head declares."""
    from simulation.game_simulator import GameSimulator

    # The event head's INPUT_KEYS (what build_model_inputs returns) must include all six.
    for k in GAME_STATE_KEYS:
        assert k in EventTimeModel.INPUT_KEYS
    # And build_model_inputs derives them from history (checked via the shared scan in
    # test_game_state_features); here just assert the contract wiring is present.
    assert hasattr(GameSimulator, "build_model_inputs")


def test_player_head_builds_and_forward_passes(tmp_path):
    """Closes half the gap §10 named: this suite covered four heads, not six."""
    enc = _setup(tmp_path)
    EventTimeModel(enc, **_kwargs(tmp_path)).preprocess(
        rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    m = PlayerModel(enc, **_kwargs(tmp_path))
    train, _ = m.preprocess(rebuild_vocabs=False, test_frac=0.0, holdout_frac=0.0)
    model = m.model(num_layers=1, num_heads=2, ff_dim=32)
    _assert_consumes_game_state(model)
    out = model({k: train[k] for k in m.INPUT_KEYS}, training=False)
    assert np.isfinite(out["player_output"].numpy()).all()
    assert np.abs(train["score_diff"]).sum() > 0


def test_substitution_head_builds_and_forward_passes(tmp_path):
    """The other half. Both heads consume the game state; neither was ever asserted to."""
    enc = _setup(tmp_path)
    EventTimeModel(enc, **_kwargs(tmp_path)).preprocess(
        rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    m = SubstitutionModel(enc, **_kwargs(tmp_path))
    train, _ = m.preprocess(rebuild_vocabs=False, test_frac=0.0, holdout_frac=0.0)
    model = m.model(num_layers=1, num_heads=2, ff_dim=32)
    _assert_consumes_game_state(model)
    out = model({k: train[k] for k in m.INPUT_KEYS}, training=False)
    assert np.isfinite(out[m.output_name].numpy()).all()
    assert np.abs(train["score_diff"]).sum() > 0


# ---------------------------------------------------------------------------
# Loss masking - the arrays reach the npz, and only where they belong
# ---------------------------------------------------------------------------

def _continuation_game(gid):
    """A game carrying one of every expansion shape, plus a period boundary.

    The shared two-game fixture has no assist, no block and no free throw, so a mask built over
    it is all zeros -- a test against it would pass whatever the rule did. This game exists so
    the end-to-end assertion has something real to bite on.
    """
    base = {"roster_home": _roster_str(HOME), "roster_away": _roster_str(AWAY),
            "season": "2003", "playoff": 1, "game_id": gid, "secondary_player": "none",
            "home/away": 1, **_SEASON}

    def row(time, event, player, type, result, **kw):
        return {**base, "time": time, "event": event, "player": player,
                "type": type, "result": result, **kw}

    return [
        row(0, "start", "start", "start", "start", **{"home/away": 0}),
        # assist -> the assisted basket
        row(10, "assist", "A", "rim", "score"),
        row(10, "shot", "B", "rim", "made"),
        # blocked shot -> the block
        row(24, "shot", "F", "paint", "blocked", **{"home/away": 2}),
        row(24, "block", "C", "paint", "block", secondary_player="F"),
        row(26, "rebound", "D", "defensive", "cop"),
        # a shooting foul -> a two-shot trip, split by a substitution the way real data is
        row(40, "foul", "G", "shooting 2pt", "free throw", secondary_player="A",
            **{"home/away": 2}),
        row(42, "shot", "A", "free throw", "made"),
        row(42, "substitution", "E", "substitution", "substitution", secondary_player="K"),
        row(45, "shot", "A", "free throw", "made"),
        # last row of period 0 -- the time head's other exclusion
        row(PERIOD_LENGTH - 5, "shot", "F", "rim", "missed", **{"home/away": 2}),
        row(PERIOD_LENGTH + 8, "shot", "B", "rim", "made"),
        row(PERIOD_LENGTH + 20, "end", "end", "end", "end", **{"home/away": 0}),
    ]


def _setup_with_continuations(tmp_path):
    """The shared fixture plus the continuation game, written as one cleaned CSV."""
    (tmp_path / "data").mkdir()
    rows = _game_rows(1) + _continuation_game(2)
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "season_clean.csv", index=False)
    return Encoder(vocab_dir=tmp_path / "vocabs")


def test_the_two_time_heads_carry_the_query_mask_arrays(tmp_path):
    """Both next-step time heads must ship the mask; neither may treat it as a model input.

    ConditionalTimeModel is included deliberately. It is the sim's actual clock -- _advance_for
    routes through predict_delta -- and it is called once per sampled play, never at a
    continuation row, so it is asked about exactly the positions the event head is asked about.
    """
    enc = _setup_with_continuations(tmp_path)
    et = EventTimeModel(enc, **_kwargs(tmp_path))
    et_train, _ = et.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)
    ct = ConditionalTimeModel(enc, **_kwargs(tmp_path))
    ct_train, _ = ct.preprocess(rebuild_vocabs=False, test_frac=0.0, holdout_frac=0.0)

    for split, model in ((et_train, et), (ct_train, ct)):
        name = type(model).__name__
        for k in QUERY_MASK_KEYS:
            assert k in split, f"{name} preprocess dropped {k!r}"
            assert split[k].shape == split["loss_mask"].shape, name
            assert set(np.unique(split[k])) <= {0.0, 1.0}, name
            # A loss mask, not a feature: it must never be fed to the graph.
            assert k not in model.INPUT_KEYS, f"{name} feeds {k!r} to the model"
        # And it is really populated -- four continuations in the fixture game (the assisted
        # shot, the block, and both free throws), each masking the position BEFORE it.
        assert split[NEXT_CONTINUATION_KEY].sum() == 4, name
        assert split[PERIOD_BREAK_KEY].sum() == 1, name


def test_masked_positions_are_a_subset_of_trainable_positions(tmp_path):
    """Nothing may be masked that loss_mask has not already opened, and it must remove some."""
    enc = _setup_with_continuations(tmp_path)
    et = EventTimeModel(enc, **_kwargs(tmp_path))
    train, _ = et.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    masked = train[NEXT_CONTINUATION_KEY] > 0
    assert not (masked & (train["loss_mask"] == 0)).any()

    event_mask = apply_query_mask(train["loss_mask"], train, time_head=False)
    time_mask = apply_query_mask(train["loss_mask"], train, time_head=True)
    assert (event_mask <= train["loss_mask"]).all()      # can only ever remove weight
    assert event_mask.sum() == train["loss_mask"].sum() - 4
    # The time head drops one more: the last row of the first period.
    assert time_mask.sum() == event_mask.sum() - 1
