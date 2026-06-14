import numpy as np
import pandas as pd

from encoder.encoder import Encoder
from models.event_time_model import EventTimeModel
from config import MAX_SEQUENCE_LENGTH


def _roster(players):
    return str(list(players))


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
                "season": "2003",
            })
    pd.DataFrame(rows).to_csv(path, index=False)


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
