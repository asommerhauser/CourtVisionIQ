"""
Tests for SubstitutionModel.

Covers the substitution-specific behavior layered on top of the shared conditional-head
machinery:
  - opening `start -> starter` subs are synthesized in-memory (rosters fill 0->5, outgoing
    = "start", incoming = each starter), with the start frame blanked;
  - preprocess writes sub_*.npz with the conditioning + target chain arrays;
  - loss conditions on the outgoing `player` and targets the incoming `secondary_player`,
    over the player vocab, masked to substitution rows;
  - a 1-epoch train smoke + from_artifacts round-trip, and ModelBundle picks it up.
"""
import numpy as np
import pandas as pd

from config import ROSTER_SIZE
from encoder.encoder import Encoder
from models.substitution_model import SubstitutionModel, SUB_EVENT, START_TOKEN


HOME = ["A", "B", "C", "D", "E"]
AWAY = ["F", "G", "H", "I", "J"]
BENCH = ["K", "L"]  # players who come in via real subs

# Season-context columns the (post-season-enrichment) preprocess consumes; every row carries them
# so the fixture matches the enriched cleaned-data schema the models train on.
_SEASON = {
    "rest_home": str([2.0] * 5), "rest_away": str([2.0] * 5),
    "home_games_played": 0.5, "away_games_played": 0.5,
    "home_days_rest": 2.0, "away_days_rest": 2.0,
}


def _roster(players):
    return str(list(players))


def _game_rows(gid):
    """A small game: start (full fives) -> two shots -> a real sub -> end."""
    full = {"roster_home": _roster(HOME), "roster_away": _roster(AWAY),
            "season": "2003", "playoff": 1, **_SEASON}
    rows = [
        {**full, "game_id": gid, "time": 0, "event": "start", "player": "start",
         "type": "start", "result": "start", "secondary_player": "none", "home/away": 0},
        {**full, "game_id": gid, "time": 12, "event": "shot", "player": "A",
         "type": "2pt", "result": "made", "secondary_player": "none", "home/away": 1},
        {**full, "game_id": gid, "time": 30, "event": "shot", "player": "F",
         "type": "2pt", "result": "missed", "secondary_player": "none", "home/away": 2},
        # real in-game sub: A (outgoing, on the five) -> K (incoming, off the bench)
        {"game_id": gid, "roster_home": _roster(["K", "B", "C", "D", "E"]),
         "roster_away": _roster(AWAY), "season": "2003", "playoff": 1, "time": 48,
         "event": SUB_EVENT, "player": "A", "type": SUB_EVENT, "result": SUB_EVENT,
         "secondary_player": "K", "home/away": 1, **_SEASON},
        {**full, "game_id": gid, "time": 60, "event": "end", "player": "end",
         "type": "end", "result": "end", "secondary_player": "none", "home/away": 0},
    ]
    return rows


def _make_cleaned_csv(path, games=(1,)):
    rows = []
    for gid in games:
        rows.extend(_game_rows(gid))
    pd.DataFrame(rows).to_csv(path, index=False)


def _model(tmp_path, **kwargs):
    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    return SubstitutionModel(
        enc,
        path=str(tmp_path / "data"),
        processed_dir=str(tmp_path / "processed"),
        sequence_length=kwargs.pop("sequence_length", 32),
        model_dim=kwargs.pop("model_dim", 32),
    )


# ---------------------------------------------------------------------------
# Opening-sub synthesis
# ---------------------------------------------------------------------------

def test_augment_inserts_opening_subs(tmp_path):
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv")
    m = _model(tmp_path)

    df = m._load_all()
    aug = m._augment_with_opening_subs(df)
    game = aug[aug["game_id"] == 1].reset_index(drop=True)

    # start frame is blanked, then exactly 10 opening subs (5 home + 5 away).
    assert game.loc[0, "event"] == "start"
    assert m.encoder.str_to_list(game.loc[0, "roster_home"]) == []
    assert m.encoder.str_to_list(game.loc[0, "roster_away"]) == []

    openers = game.iloc[1:11]
    assert (openers["event"] == SUB_EVENT).all()
    assert (openers["player"] == START_TOKEN).all()  # outgoing = "start"
    # incoming = the starters, in order (home five then away five).
    assert list(openers["secondary_player"]) == HOME + AWAY

    # rosters fill incrementally 0->5 for the substituting team.
    home_openers = game.iloc[1:6]
    assert [len(m.encoder.str_to_list(r)) for r in home_openers["roster_home"]] == [1, 2, 3, 4, 5]
    away_openers = game.iloc[6:11]
    assert [len(m.encoder.str_to_list(r)) for r in away_openers["roster_away"]] == [1, 2, 3, 4, 5]
    # during the home build the away roster is still empty.
    assert all(m.encoder.str_to_list(r) == [] for r in home_openers["roster_away"])

    # the real game resumes after the openers (the first shot).
    assert game.loc[11, "event"] == "shot"


# ---------------------------------------------------------------------------
# Preprocess arrays + masking
# ---------------------------------------------------------------------------

def test_preprocess_writes_arrays_with_chain(tmp_path):
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv", games=(1, 2))
    m = _model(tmp_path)

    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    for key in ("next_event", "next_player", "next_secondary_player",
                "next_delta_time", "loss_mask", "pad_mask"):
        assert key in train

    sub_id = m.encoder.encode_event(SUB_EVENT)
    start_id = m.encoder.encode_player(START_TOKEN)

    # Across both games, every substitution placement whose outgoing is "start" is an
    # opening sub; there must be exactly 10 per game (5 home + 5 away).
    is_sub = train["next_event"] == sub_id
    opening = is_sub & (train["next_player"] == start_id)
    assert int(opening.sum()) == 2 * (2 * ROSTER_SIZE)

    # The opening targets (next_secondary_player) are real starters, never PAD/"start".
    pad_player = m.encoder.encode_player("PAD")
    targets = train["next_secondary_player"][opening]
    assert (targets != pad_player).all()
    assert (targets != start_id).all()


def test_make_dataset_masks_non_substitution(tmp_path):
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv", games=(1, 2))
    m = _model(tmp_path)
    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    ds = m._make_dataset(train, batch_size=2, shuffle=False)
    inputs, targets, weights = next(iter(ds))
    w = weights[m.output_name].numpy()
    ne = inputs["next_event"].numpy()
    sub_id = m.encoder.encode_event(SUB_EVENT)
    # Every nonzero-weight step is a substitution placement; non-subs are zeroed.
    assert np.all((w > 0) == ((ne == sub_id) & (w > 0)))
    assert w[ne != sub_id].sum() == 0.0


# ---------------------------------------------------------------------------
# Model graph
# ---------------------------------------------------------------------------

def test_model_output_is_over_player_vocab(tmp_path):
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv")
    m = _model(tmp_path)
    m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    model = m.model(num_layers=1, num_heads=2, ff_dim=32)
    out = model.outputs[0]
    assert m.output_name == "secondary_player_output"
    assert out.shape[-1] == m.encoder.player_vocab.next_token


def test_avail_mask_restricts_logits_to_available_players(tmp_path):
    """The per-game availability mask is emitted and the head's logits for unavailable
    players (e.g. PAD) are pushed to ~-inf, while available players stay finite."""
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv", games=(1, 2))
    m = _model(tmp_path)
    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    V = m.encoder.player_vocab.next_token
    n_games = train["pad_mask"].shape[0]
    assert train["avail_mask"].shape == (n_games, V)

    pad = m.encoder.encode_player("PAD")
    a_id = m.encoder.encode_player("A")  # a real on-court player -> available every game
    assert (train["avail_mask"][:, pad] == 0.0).all()
    assert (train["avail_mask"][:, a_id] == 1.0).all()

    # Forward pass: masked logits at unavailable ids are strongly negative; available finite.
    model = m.model(num_layers=1, num_heads=2, ff_dim=32)
    inputs = {k: train[k] for k in m.INPUT_KEYS}
    logits = model(inputs, training=False)[m.output_name].numpy()  # (G, SEQ, V)
    assert logits[:, :, pad].max() < -1e3      # masked, large-negative but float16-safe finite
    assert np.isfinite(logits[:, :, a_id]).all() and logits[:, :, a_id].max() > -1e3

# Train / from_artifacts / full-keras-load / ModelBundle round-trip coverage lives in
# tests/test_model_persistence.py (the shared parametrized adapter for every model).
