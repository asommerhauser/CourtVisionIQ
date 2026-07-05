"""
Tests for OnCourtCandidateMask + the target-in-candidates loss guards (Train 2.5).

The Player and Substitution heads' softmax is now masked in-graph to the row's legal
candidate set (train/inference legality parity):
  - Player (actor/outgoing) head: candidates = the row's on-court ten;
  - Substitution (incoming) head: candidates = game availability minus the on-court ten
    (the legal bench at the decision).
Rows whose target falls outside the candidate set are zeroed out of the loss by
``target_on_court``-based guards in each head's ``_build_split`` — otherwise the target
would sit on a -1e9 logit and train on noise.
"""
import numpy as np
import pandas as pd
import tensorflow as tf

from encoder.encoder import Encoder
from models.event_time_model import OnCourtCandidateMask, target_on_court
from models.player_model import PlayerModel
from models.substitution_model import SubstitutionModel, SUB_EVENT


HOME = ["A", "B", "C", "D", "E"]
AWAY = ["F", "G", "H", "I", "J"]

_SEASON = {
    "rest_home": str([2.0] * 5), "rest_away": str([2.0] * 5),
    "home_games_played": 0.5, "away_games_played": 0.5,
    "home_days_rest": 2.0, "away_days_rest": 2.0,
}


def _roster(players):
    return str(list(players))


def _game_rows(gid):
    """A small game: start (full fives) -> two shots -> a real sub (A -> K) -> end."""
    full = {"roster_home": _roster(HOME), "roster_away": _roster(AWAY),
            "season": "2003", "playoff": 1, **_SEASON}
    return [
        {**full, "game_id": gid, "time": 0, "event": "start", "player": "start",
         "type": "start", "result": "start", "secondary_player": "none", "home/away": 0},
        {**full, "game_id": gid, "time": 12, "event": "shot", "player": "A",
         "type": "2pt", "result": "made", "secondary_player": "none", "home/away": 1},
        {**full, "game_id": gid, "time": 30, "event": "shot", "player": "F",
         "type": "2pt", "result": "missed", "secondary_player": "none", "home/away": 2},
        # real in-game sub: A (outgoing, on the five) -> K (incoming, off the bench);
        # sub rows carry the POST-sub lineup, matching the cleaned-data convention.
        {"game_id": gid, "roster_home": _roster(["K", "B", "C", "D", "E"]),
         "roster_away": _roster(AWAY), "season": "2003", "playoff": 1, "time": 48,
         "event": SUB_EVENT, "player": "A", "type": SUB_EVENT, "result": SUB_EVENT,
         "secondary_player": "K", "home/away": 1, **_SEASON},
        {"game_id": gid, "roster_home": _roster(["K", "B", "C", "D", "E"]),
         "roster_away": _roster(AWAY), "season": "2003", "playoff": 1, "time": 55,
         "event": "shot", "player": "K", "type": "2pt", "result": "made",
         "secondary_player": "none", "home/away": 1, **_SEASON},
        {"game_id": gid, "roster_home": _roster(["K", "B", "C", "D", "E"]),
         "roster_away": _roster(AWAY), "season": "2003", "playoff": 1, "time": 60,
         "event": "end", "player": "end", "type": "end", "result": "end",
         "secondary_player": "none", "home/away": 0, **_SEASON},
    ]


def _make_cleaned_csv(path, games=(1,)):
    rows = []
    for gid in games:
        rows.extend(_game_rows(gid))
    pd.DataFrame(rows).to_csv(path, index=False)


def _player_model(tmp_path):
    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    return PlayerModel(enc, path=str(tmp_path / "data"),
                       processed_dir=str(tmp_path / "processed"),
                       sequence_length=32, model_dim=32)


def _sub_model(tmp_path):
    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    return SubstitutionModel(enc, path=str(tmp_path / "data"),
                             processed_dir=str(tmp_path / "processed"),
                             sequence_length=32, model_dim=32)


# ---------------------------------------------------------------------------
# Layer math
# ---------------------------------------------------------------------------

def test_layer_restricts_to_on_court(tmp_path):
    """Restrict mode: only the on-court ids keep finite logits; PAD never a candidate."""
    V = 8
    logits = tf.zeros((1, 2, V))
    home = tf.constant([[[1, 2], [1, 2]]], dtype=tf.int32)   # (B, SEQ, 2)
    away = tf.constant([[[3, 0], [3, 0]]], dtype=tf.int32)   # PAD-padded slot
    out = OnCourtCandidateMask()(logits, home, away).numpy()[0, 0]
    on = {1, 2, 3}
    for v in range(V):
        if v in on:
            assert out[v] > -1e8, f"on-court id {v} should stay finite"
        else:
            assert out[v] < -1e8, f"id {v} (incl. PAD 0) should be masked"


def test_layer_excludes_on_court_from_avail(tmp_path):
    """Exclude mode: candidates = avail minus on-court (the legal bench)."""
    V = 8
    logits = tf.zeros((1, 1, V))
    home = tf.constant([[[1, 2]]], dtype=tf.int32)
    away = tf.constant([[[3, 0]]], dtype=tf.int32)
    avail = tf.constant([[0, 1, 1, 1, 1, 1, 0, 0]], dtype=tf.float32)  # ids 1..5 in the game
    out = OnCourtCandidateMask(exclude_on_court=True)(logits, home, away, avail).numpy()[0, 0]
    assert out[4] > -1e8 and out[5] > -1e8          # available, off-court -> bench
    for v in (0, 1, 2, 3, 6, 7):                    # PAD / on-court / out of game
        assert out[v] < -1e8, f"id {v} should be masked"


def test_layer_serialization_round_trip():
    layer = OnCourtCandidateMask(exclude_on_court=True)
    clone = OnCourtCandidateMask.from_config(layer.get_config())
    assert clone.exclude_on_court is True


def test_target_on_court_helper():
    home = np.array([[1, 2], [1, 2]], dtype=np.int32)
    away = np.array([[3, 0], [3, 0]], dtype=np.int32)
    target = np.array([2, 5], dtype=np.int32)
    assert target_on_court(target, home, away).tolist() == [1.0, 0.0]


# ---------------------------------------------------------------------------
# Player head: loss guard + in-graph mask
# ---------------------------------------------------------------------------

def test_player_loss_mask_guards_offcourt_targets(tmp_path):
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv")
    m = _player_model(tmp_path)
    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    loss = train["loss_mask"][0]
    # Rows (no opening-sub augmentation on this head):
    # 0 start -> target A (on court)          : kept
    # 1 shot A -> target F (on court)          : kept
    # 2 shot F -> target A (outgoing, on court): kept  <- the outgoing-pick training row
    # 3 sub    -> target K (on court post-sub) : kept
    # 4 shot K -> target "end" (never on court): GUARDED to 0
    # 5 end    -> no next target               : 0 (pre-existing rule)
    assert loss[:4].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert loss[4] == 0.0  # pre-"end" sentinel row no longer trains on noise
    assert loss[5] == 0.0


def test_player_logits_masked_to_on_court(tmp_path):
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv")
    m = _player_model(tmp_path)
    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    model = m.model(num_layers=1, num_heads=2, ff_dim=32)
    inputs = {k: train[k] for k in m.INPUT_KEYS}
    assert "avail_mask" not in m.INPUT_KEYS  # replaced by the in-graph on-court mask
    logits = model(inputs, training=False)["player_output"].numpy()

    a_id = m.encoder.encode_player("A")
    k_id = m.encoder.encode_player("K")
    pad = m.encoder.encode_player("PAD")
    # Row 1 (first shot): full fives on court, K on the bench.
    assert logits[0, 1, a_id] > -1e8
    assert logits[0, 1, k_id] < -1e8
    # Row 4 (post-sub shot): K on court, A subbed out.
    assert logits[0, 4, k_id] > -1e8
    assert logits[0, 4, a_id] < -1e8
    assert logits[:, :, pad].max() < -1e8


def test_player_masked_loss_is_finite(tmp_path):
    """NaN regression: guarded rows keep the total masked loss finite even though their
    target logit sits at -1e9."""
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv")
    m = _player_model(tmp_path)
    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    model = m.model(num_layers=1, num_heads=2, ff_dim=32)
    logits = model({k: train[k] for k in m.INPUT_KEYS}, training=False)["player_output"]
    lf = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = lf(train["player_target"], logits, sample_weight=train["loss_mask"])
    assert np.isfinite(float(loss))


# ---------------------------------------------------------------------------
# Substitution head: loss guard
# ---------------------------------------------------------------------------

def test_sub_loss_mask_keeps_legal_incoming_targets(tmp_path):
    (tmp_path / "data").mkdir()
    _make_cleaned_csv(tmp_path / "data" / "season_clean.csv")
    m = _sub_model(tmp_path)
    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    sub_id = m.encoder.encode_event(SUB_EVENT)
    k_id = m.encoder.encode_secondary_player("K")
    is_sub_target = train["next_event"][0] == sub_id
    # Every substitution placement row (opening synth + the real A->K sub) survives the
    # guard: the incoming player is available and NOT yet on the floor at the decision.
    assert train["loss_mask"][0][is_sub_target].min() == 1.0
    # The real sub's placement row targets K specifically.
    real = is_sub_target & (train["next_secondary_player"][0] == k_id)
    assert real.any() and train["loss_mask"][0][real].all()


def test_sub_guard_zeroes_target_already_on_court(tmp_path):
    """A (mis-cleaned) sub whose incoming player is already on the floor is guarded out."""
    (tmp_path / "data").mkdir()
    rows = _game_rows(1)
    rows[3]["secondary_player"] = "B"  # "incoming" B is already on court -> illegal target
    pd.DataFrame(rows).to_csv(tmp_path / "data" / "season_clean.csv", index=False)
    m = _sub_model(tmp_path)
    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    sub_id = m.encoder.encode_event(SUB_EVENT)
    b_id = m.encoder.encode_secondary_player("B")
    a_id = m.encoder.encode_player("A")
    # The real in-game sub (outgoing A, not an opening "start" sub): its incoming target B
    # is already on the floor, so the guard zeroes it. B's legitimate opening-sub row (where
    # B checks into a partial lineup) keeps its loss.
    bad = ((train["next_event"][0] == sub_id)
           & (train["next_secondary_player"][0] == b_id)
           & (train["next_player"][0] == a_id))
    assert bad.any()
    assert train["loss_mask"][0][bad].max() == 0.0
