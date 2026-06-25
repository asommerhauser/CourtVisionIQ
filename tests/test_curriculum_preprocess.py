"""Curriculum preprocess: explicit chronological partition + fixed (loaded) norm stats."""
import numpy as np
import pandas as pd

from encoder.encoder import Encoder
from models.event_time_model import EventTimeModel

_SEASON = {
    "rest_home": str([2.0] * 5), "rest_away": str([2.0] * 5),
    "home_games_played": 0.5, "away_games_played": 0.5,
    "home_days_rest": 2.0, "away_days_rest": 2.0,
}


def _make_corpus(path, n_games=6):
    """n_games, each with a distinct time scale so train-slice norm stats differ by slice."""
    rows = []
    for gid in range(1, n_games + 1):
        for i in range(4):
            rows.append({
                "game_id": gid,
                "roster_home": str(["A", "B", "C", "D", "E"]),
                "roster_away": str(["F", "G", "H", "I", "J"]),
                "time": i * 5 * gid,          # later games run on a larger time scale
                "event": "shot" if i else "start",
                "player": "A", "type": "t", "result": "r", "secondary_player": "none",
                "season": "2003", **_SEASON,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_preprocess_honors_explicit_partition(tmp_path):
    data_dir = tmp_path / "data"; data_dir.mkdir()
    _make_corpus(data_dir / "season_clean.csv")

    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    m = EventTimeModel(enc, path=str(data_dir), processed_dir=str(tmp_path / "processed"))
    partition = ({1, 2, 3, 4}, {5}, {6})
    m.preprocess(rebuild_vocabs=True, game_partition=partition)

    import json
    manifest = json.loads((tmp_path / "processed" / "holdout_games.json").read_text())
    assert manifest == [6]   # the explicit holdout, not a random split


def test_refit_false_loads_fixed_norm_stats(tmp_path):
    data_dir = tmp_path / "data"; data_dir.mkdir()
    _make_corpus(data_dir / "season_clean.csv")
    processed = str(tmp_path / "processed")

    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    # Warmup fit over the full corpus: persists norm stats spanning every game.
    warm = EventTimeModel(enc, path=str(data_dir), processed_dir=processed)
    warm.preprocess(rebuild_vocabs=True, game_partition=({1, 2, 3, 4, 5, 6}, set(), set()))
    full_max_time = warm.norm_stats["max_time"]

    # A staged run on just the first two games must NOT recompute the (smaller) stats; it loads
    # the warmup stats so standardization stays fixed for warm-start.
    staged = EventTimeModel(enc, path=str(data_dir), processed_dir=processed)
    staged.preprocess(rebuild_vocabs=False, refit_norm_stats=False,
                      game_partition=({1, 2}, set(), {3}))
    assert staged.norm_stats["max_time"] == full_max_time
    # Sanity: games 1-2 alone would give a strictly smaller max_time, proving it didn't refit.
    assert full_max_time > 5 * 2 * 3   # game 2's own max is 5*3*2=30; full max is 5*3*6=90
