"""
Tests for StintLengthModel.

Covers the stint-specific behavior layered on the shared substitution machinery:
  - the realized-stint target math (entry -> exit, with right-censoring at the final whistle);
  - preprocess writes stint_*.npz with the conditioning + continuous next-step target arrays;
  - the loss conditions on the fully decided sub and is masked to substitution rows;
  - the output head is a single regression scalar.

Train / from_artifacts / full-keras-load / ModelBundle round-trip coverage lives in
tests/test_model_persistence.py (the shared parametrized adapter for every model).
"""
import numpy as np
import pandas as pd

from encoder.encoder import Encoder
from models.stint_length_model import StintLengthModel
from models.substitution_model import SUB_EVENT, START_TOKEN


HOME = ["A", "B", "C", "D", "E"]
AWAY = ["F", "G", "H", "I", "J"]

# Season-context columns the (post-season-enrichment) preprocess consumes — supplied here so the
# synthetic data matches the enriched cleaned-data schema the models train on.
_SEASON = {
    "rest_home": str([2.0] * 5), "rest_away": str([2.0] * 5),
    "home_games_played": 0.5, "away_games_played": 0.5,
    "home_days_rest": 2.0, "away_days_rest": 2.0,
}


def _roster(players):
    return str(list(players))


def _game_rows(gid):
    """start (full fives) -> shots -> a real sub (A->K) -> end, so A has a closed stint and K
    is on the floor at the whistle (right-censored)."""
    full = {"roster_home": _roster(HOME), "roster_away": _roster(AWAY),
            "season": "2003", "playoff": 1, **_SEASON}
    return [
        {**full, "game_id": gid, "time": 0, "event": "start", "player": "start",
         "type": "start", "result": "start", "secondary_player": "none"},
        {**full, "game_id": gid, "time": 12, "event": "shot", "player": "A",
         "type": "2pt", "result": "made", "secondary_player": "none"},
        {**full, "game_id": gid, "time": 30, "event": "shot", "player": "F",
         "type": "2pt", "result": "missed", "secondary_player": "none"},
        {**full, "game_id": gid, "roster_home": _roster(["K", "B", "C", "D", "E"]),
         "time": 48, "event": SUB_EVENT, "player": "A", "type": SUB_EVENT,
         "result": SUB_EVENT, "secondary_player": "K"},
        {**full, "game_id": gid, "time": 60, "event": "end", "player": "end",
         "type": "end", "result": "end", "secondary_player": "none"},
    ]


def _make_cleaned_csv(path, games=(1,)):
    rows = []
    for gid in games:
        rows.extend(_game_rows(gid))
    pd.DataFrame(rows).to_csv(path, index=False)


def _model(tmp_path, **kwargs):
    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    return StintLengthModel(
        enc,
        path=str(tmp_path / "data"),
        processed_dir=str(tmp_path / "processed"),
        sequence_length=kwargs.pop("sequence_length", 32),
        model_dim=kwargs.pop("model_dim", 32),
    )


# ---------------------------------------------------------------------------
# Realized-stint target math
# ---------------------------------------------------------------------------

def test_stint_seconds_entry_exit_and_censoring():
    """A closed stint = exit - entry; a player still on court at the whistle is censored to the
    game's last event time. Both opening starters and in-game subs seed entries."""
    # Build the per-game frame the method walks: two openers (start->A, start->F at t=0), then
    # an in-game sub A->K at t=300, ending at t=600.
    df = pd.DataFrame([
        {"game_id": 1, "time": 0, "event": "start", "player": "start", "secondary_player": "none"},
        {"game_id": 1, "time": 0, "event": SUB_EVENT, "player": START_TOKEN, "secondary_player": "A"},
        {"game_id": 1, "time": 0, "event": SUB_EVENT, "player": START_TOKEN, "secondary_player": "F"},
        {"game_id": 1, "time": 100, "event": "shot", "player": "A", "secondary_player": "none"},
        {"game_id": 1, "time": 300, "event": SUB_EVENT, "player": "A", "secondary_player": "K"},
        {"game_id": 1, "time": 600, "event": "end", "player": "end", "secondary_player": "none"},
    ])

    inst = StintLengthModel.__new__(StintLengthModel)  # no __init__ needed for this pure helper
    out = inst._stint_seconds_per_row(df)

    assert out[1] == 300.0   # A entered at 0, pulled at 300 -> 300s (closed)
    assert out[2] == 600.0   # F entered at 0, never pulled -> censored to last time 600
    assert out[4] == 300.0   # K entered at 300, never pulled -> censored to 600 -> 300s
    # Non-entry rows carry no stint.
    assert out[0] == 0.0 and out[3] == 0.0 and out[5] == 0.0


# ---------------------------------------------------------------------------
# Preprocess arrays + masking
# ---------------------------------------------------------------------------

def test_preprocess_writes_stint_target_chain(tmp_path):
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv", games=(1, 2))
    m = _model(tmp_path)

    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    for key in ("next_event", "next_player", "next_secondary_player",
                "next_delta_time", "next_stint_target", "loss_mask", "pad_mask"):
        assert key in train

    # The head's own log-stint normalization stats are persisted.
    assert "stint_log_mean" in m.norm_stats and "stint_log_std" in m.norm_stats

    # Every substitution placement has a finite, non-PAD target; the opening subs (5 per team)
    # plus the in-game sub all contribute a stint target.
    sub_id = m.encoder.encode_event(SUB_EVENT)
    is_sub = train["next_event"] == sub_id
    assert is_sub.sum() >= 2 * (2 * 5)  # >= 10 opening subs per game across two games
    assert np.isfinite(train["next_stint_target"]).all()


def test_make_dataset_masks_non_substitution_rows(tmp_path):
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv", games=(1, 2))
    m = _model(tmp_path)
    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    ds = m._make_dataset(train, batch_size=2, shuffle=False)
    inputs, targets, weights = next(iter(ds))
    w = weights[m.output_name].numpy()
    ne = inputs["next_event"].numpy()
    sub_id = m.encoder.encode_event(SUB_EVENT)
    # The regression loss only counts substitution placements; everything else is zero-weight.
    assert w[ne != sub_id].sum() == 0.0
    assert m.output_name == "stint_output"
    assert set(targets.keys()) == {"stint_output"}


# ---------------------------------------------------------------------------
# Model graph
# ---------------------------------------------------------------------------

def test_model_output_is_single_regression_scalar(tmp_path):
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv")
    m = _model(tmp_path)
    m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    model = m.model(num_layers=1, num_heads=2, ff_dim=32)
    out = model.outputs[0]
    assert m.output_name == "stint_output"
    assert out.shape[-1] == 1
    # It conditions on the fully decided sub (outgoing + incoming).
    input_keys = set(model.input.keys())
    assert "next_player" in input_keys
    assert "next_secondary_player" in input_keys
