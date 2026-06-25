"""Recency weighting: per-game weight decays with age, applied in the sample-weight mask."""
import numpy as np
import pandas as pd

from encoder.encoder import Encoder
from models.event_time_model import EventTimeModel
from models.season_features import apply_recency, recency_weight
from config import RECENCY_FLOOR

_SEASON = {
    "rest_home": str([2.0] * 5), "rest_away": str([2.0] * 5),
    "home_games_played": 0.5, "away_games_played": 0.5,
    "home_days_rest": 2.0, "away_days_rest": 2.0,
}


def _corpus(path, seasons):
    """One cleaned CSV; `seasons` is a per-game season list (game i -> seasons[i])."""
    rows = []
    for gid, season in enumerate(seasons, start=1):
        for i in range(4):
            rows.append({
                "game_id": gid,
                "roster_home": str(["A", "B", "C", "D", "E"]),
                "roster_away": str(["F", "G", "H", "I", "J"]),
                "time": i * 10, "event": "shot" if i else "start",
                "player": "A", "type": "t", "result": "r", "secondary_player": "none",
                "season": season, **_SEASON,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_recency_weight_decays_with_age():
    assert recency_weight(2023, 2023) == 1.0          # newest season -> full weight
    assert recency_weight(2017, 2023) < 1.0           # older -> less
    assert recency_weight(2017, 2023) > recency_weight(2005, 2023)   # monotonic
    assert recency_weight(1950, 2023) == RECENCY_FLOOR  # floored, never zero


def test_apply_recency_scales_rows():
    split = {"recency_weight": np.array([0.5, 1.0], dtype=np.float32)}
    out = apply_recency(np.ones((2, 3), dtype=np.float32), split)
    assert np.allclose(out[0], 0.5) and np.allclose(out[1], 1.0)
    # No-op when the key is absent (older npz / flag off).
    assert np.allclose(apply_recency(np.ones((2, 3), dtype=np.float32), {}), 1.0)


def test_preprocess_attaches_recency(tmp_path):
    data_dir = tmp_path / "data"; data_dir.mkdir()
    _corpus(data_dir / "c.csv", seasons=[2003, 2003, 2004, 2004])  # 2 older, 2 newer
    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    m = EventTimeModel(enc, path=str(data_dir), processed_dir=str(tmp_path / "processed"))
    train, _ = m.preprocess(rebuild_vocabs=True, game_partition=({1, 2, 3, 4}, set(), set()))

    rw = train["recency_weight"]
    assert rw.shape == (4,)
    # games ordered 1,2,3,4 -> 2003,2003,2004,2004; newest == 1.0, older strictly less.
    assert rw[2] == 1.0 and rw[3] == 1.0
    assert rw[0] < 1.0 and np.isclose(rw[0], rw[1])
    assert (rw >= RECENCY_FLOOR).all()


def test_single_season_is_unweighted(tmp_path):
    data_dir = tmp_path / "data"; data_dir.mkdir()
    _corpus(data_dir / "c.csv", seasons=[2003, 2003, 2003])
    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    m = EventTimeModel(enc, path=str(data_dir), processed_dir=str(tmp_path / "processed"))
    train, _ = m.preprocess(rebuild_vocabs=True, game_partition=({1, 2, 3}, set(), set()))
    assert np.allclose(train["recency_weight"], 1.0)   # one season -> everything weight 1.0
