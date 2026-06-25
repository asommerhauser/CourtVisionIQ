import numpy as np
import pandas as pd

from encoder.encoder import Encoder
from models.event_time_model import EventTimeModel
from config import MAX_SEQUENCE_LENGTH


def _roster(players):
    return str(list(players))


# Season-context columns the (post-season-enrichment) preprocess consumes.
_SEASON = {
    "rest_home": str([2.0] * 5), "rest_away": str([2.0] * 5),
    "home_games_played": 0.5, "away_games_played": 0.5,
    "home_days_rest": 2.0, "away_days_rest": 2.0,
}


def _make_cleaned_csv(path):
    """Two games with monotonically increasing time per game (resets each game)."""
    rows = []
    for gid, n in [(1, 4), (2, 3)]:
        for i in range(n):
            rows.append({
                "game_id": gid,
                "roster_home": _roster(["A", "B", "C", "D", "E"]),
                "roster_away": _roster(["F", "G", "H", "I", "J"]),
                "time": i * 10,
                "event": f"ev{gid}_{i}",
                "player": "A",
                "type": "t",
                "result": "r",
                "secondary_player": "none",
                "season": "2003",
                **_SEASON,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_sub_game_csv(path):
    """One game with a substitution mid-sequence, so the step whose NEXT event is the sub can be
    checked for zero loss-weight."""
    home, away = ["A", "B", "C", "D", "E"], ["F", "G", "H", "I", "J"]
    base = {"game_id": 1, "roster_home": _roster(home), "roster_away": _roster(away),
            "season": "2003", **_SEASON}
    rows = [
        {**base, "time": 0, "event": "start", "player": "start", "type": "start",
         "result": "start", "secondary_player": "none"},
        {**base, "time": 10, "event": "shot", "player": "A", "type": "2pt",
         "result": "missed", "secondary_player": "none"},
        {**base, "roster_home": _roster(["K", "B", "C", "D", "E"]), "time": 20,
         "event": "substitution", "player": "A", "type": "substitution",
         "result": "substitution", "secondary_player": "K"},
        {**base, "roster_home": _roster(["K", "B", "C", "D", "E"]), "time": 30,
         "event": "shot", "player": "K", "type": "2pt", "result": "made",
         "secondary_player": "none"},
        {**base, "roster_home": _roster(["K", "B", "C", "D", "E"]), "time": 40,
         "event": "end", "player": "end", "type": "end", "result": "end",
         "secondary_player": "none"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_substitution_targets_are_zero_weighted_in_loss(tmp_path):
    """The event/time heads must not be trained to emit substitution: any step whose NEXT event
    is a substitution is zero-weighted in BOTH heads' loss (subs are scheduler-owned)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_sub_game_csv(data_dir / "season_clean.csv")

    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    m = EventTimeModel(enc, path=str(data_dir), processed_dir=str(tmp_path / "processed"))
    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0, holdout_frac=0.0)

    ds = m._make_dataset(train, batch_size=4, shuffle=False)
    _inputs, targets, weights = next(iter(ds))
    et = targets["event_output"].numpy()
    sub_id = enc.encode_event("substitution")
    assert (et == sub_id).any()  # the fixture really does contain a substitution target
    for head in ("event_output", "time_output"):
        w = weights[head].numpy()
        assert w[et == sub_id].sum() == 0.0   # substitution targets contribute no loss
    # non-substitution real steps are still trained on.
    assert (weights["event_output"].numpy() > 0).any()


def test_preprocess_no_cross_game_target_leakage(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cleaned_csv(data_dir / "season_clean.csv")

    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    m = EventTimeModel(
        enc,
        path=str(data_dir),
        processed_dir=str(tmp_path / "processed"),
    )
    # Single split so both games land together and we can inspect deterministically.
    train, test = m.preprocess(rebuild_vocabs=True, test_frac=0.0)

    SEQ = MAX_SEQUENCE_LENGTH
    pad_event = enc.encode_event("PAD")

    # Every game's last real row must have no next-step target (PAD) and zero weight.
    for b in range(train["event"].shape[0]):
        n = int(train["pad_mask"][b].sum())
        assert n > 0
        # last real row's target is PAD (target shift does not pull next game's event)
        assert train["event_target"][b][n - 1] == pad_event
        # loss_mask excludes the last real row
        assert train["loss_mask"][b][n - 1] == 0.0
        assert train["loss_mask"][b][: n - 1].sum() == n - 1


def test_preprocess_delta_resets_each_game(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_cleaned_csv(data_dir / "season_clean.csv")

    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    m = EventTimeModel(
        enc,
        path=str(data_dir),
        processed_dir=str(tmp_path / "processed"),
    )
    train, _ = m.preprocess(rebuild_vocabs=True, test_frac=0.0)

    # First timestep delta of every game is the normalized value of raw delta 0.
    stats = m.norm_stats
    expected_first = (0.0 - stats["delta_mean"]) / stats["delta_std"]
    for b in range(train["delta_time"].shape[0]):
        assert np.isclose(train["delta_time"][b][0, 0], expected_first, atol=1e-5)
