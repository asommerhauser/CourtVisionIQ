"""
Tests for the season-context feature pipeline (the TF-free data + preprocessing layer).

Covers:
  - DataCleaner stamps game_date / home_team / away_team (team resolved from the raw
    ``team`` column via roster membership).
  - season_context.enrich_df: per-team games-played, per-team days-rest, per-player rest
    (true day deltas, first-appearance default of 3), and roster-parallel alignment of the
    rest_home / rest_away columns — including after a substitution changes the lineup.
  - data_loading decodes the rest_home / rest_away list columns.
  - models.season_features numpy helpers (padding, stats, standardization, batching).

The model graph + simulator inference are exercised by the TF-dependent suites (run on the
training box); these tests need no TensorFlow.
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from data_cleaner import DataCleaner
from season_context import DEFAULT_REST_DAYS, enrich_df
from models import season_features as sf


# ---------------------------------------------------------------------------
# Helpers: synthetic raw rows (for the cleaner) and cleaned rows (for enrich)
# ---------------------------------------------------------------------------

_RAW_COLS = ["data_set", "date", "event_type", "period", "elapsed", "team", "player",
             "type", "result", "h1", "h2", "h3", "h4", "h5", "a1", "a2", "a3", "a4", "a5",
             "assist", "block", "steal", "entered", "left"]


def _raw(**kw):
    row = {c: None for c in _RAW_COLS}
    row.update(kw)
    return row


def _clean_raw(rows) -> pd.DataFrame:
    """Run DataCleaner.parse_file over synthetic raw rows; return the cleaned frame."""
    import os
    import tempfile

    d = tempfile.mkdtemp()
    path = os.path.join(d, "mini.csv")
    pd.DataFrame(rows)[_RAW_COLS].to_csv(path, index=False)
    dc = DataCleaner()
    dc.season = 2003
    _, cleaned = dc.parse_file(path)
    return cleaned


def _cleaned_row(game_id, date, home, away, roster_home, roster_away, *, season=2003):
    """One synthetic cleaned event row (constant-per-game context columns included)."""
    return dict(
        game_id=game_id, roster_home=str(roster_home), roster_away=str(roster_away),
        time=0, event="shot", player="x", type="2pt", result="made",
        secondary_player="none", **{"home/away": 1}, season=season, playoff=1,
        game_date=date, home_team=home, away_team=away,
    )


# ---------------------------------------------------------------------------
# DataCleaner: game_date / home_team / away_team
# ---------------------------------------------------------------------------

def test_cleaner_stamps_date_and_resolves_teams():
    rows = [
        _raw(data_set="2002-03 Regular Season", date="2002-10-29",
             event_type="start of period", period=1, elapsed="0:00:00",
             h1="A", h2="B", h3="C", h4="D", h5="E", a1="V", a2="W", a3="X", a4="Y", a5="Z"),
        _raw(data_set="2002-03 Regular Season", date="2002-10-29", event_type="shot",
             period=1, elapsed="0:00:20", team="LAL", player="A", type="2pt", result="missed",
             h1="A", h2="B", h3="C", h4="D", h5="E", a1="V", a2="W", a3="X", a4="Y", a5="Z"),
        _raw(data_set="2002-03 Regular Season", date="2002-10-29", event_type="rebound",
             period=1, elapsed="0:00:22", team="BOS", player="V", type="rebound defensive",
             h1="A", h2="B", h3="C", h4="D", h5="E", a1="V", a2="W", a3="X", a4="Y", a5="Z"),
    ]
    cleaned = _clean_raw(rows)
    assert {"game_date", "home_team", "away_team"}.issubset(cleaned.columns)
    assert (cleaned["game_date"] == "2002-10-29").all()
    # Home team fixed by a home actor (A in h1..h5); away by an away actor (V in a1..a5).
    teams = cleaned[cleaned["event"].isin(["shot", "rebound", "end"])]
    assert (teams["home_team"] == "LAL").all()
    assert (teams[teams["event"].isin(["rebound", "end"])]["away_team"] == "BOS").all()


# ---------------------------------------------------------------------------
# enrich_df: games-played, team rest, per-player rest
# ---------------------------------------------------------------------------

def test_enrich_games_played_increments_per_team():
    rows = [
        _cleaned_row(1, "2003-01-01", "LAL", "BOS", ["A", "B"], ["X", "Y"]),
        _cleaned_row(2, "2003-01-05", "LAL", "BOS", ["A", "B"], ["X", "Y"]),
        _cleaned_row(3, "2003-01-09", "LAL", "DEN", ["A", "B"], ["M", "N"]),
    ]
    out = enrich_df(pd.DataFrame(rows))
    by_game = out.groupby("game_id")[["home_games_played", "away_games_played"]].first()
    # LAL: 0, 1, 2 games played before each game (normalized by 82; stored float32).
    np.testing.assert_allclose(by_game["home_games_played"], [0.0, 1 / 82.0, 2 / 82.0], rtol=1e-6)
    # DEN debuts in game 3 -> 0 prior games.
    assert by_game.loc[3, "away_games_played"] == 0.0


def test_enrich_rest_deltas_and_debut_default():
    rows = [
        _cleaned_row(1, "2003-01-01", "LAL", "BOS", ["A", "B"], ["X", "Y"]),
        _cleaned_row(2, "2003-01-02", "LAL", "BOS", ["A", "C"], ["X", "Y"]),   # back-to-back
        _cleaned_row(3, "2003-01-20", "LAL", "DEN", ["A", "B"], ["M", "N"]),
    ]
    out = enrich_df(pd.DataFrame(rows))
    g = {gid: rows.iloc[0] for gid, rows in out.groupby("game_id")}

    # Openers default to 3 days (fresh).
    assert g[1]["home_days_rest"] == DEFAULT_REST_DAYS
    assert ast.literal_eval(str(g[1]["rest_home"])) == [3, 3]

    # Game 2 is a back-to-back: team rest 1; A played g1 (1 day), C debuts (3).
    assert g[2]["home_days_rest"] == 1
    assert ast.literal_eval(str(g[2]["rest_home"])) == [1, 3]

    # Game 3: A last played g2 (Jan 2 -> 18), B last played g1 (Jan 1 -> 19); DEN opener (3).
    assert ast.literal_eval(str(g[3]["rest_home"])) == [18, 19]
    assert g[3]["away_days_rest"] == DEFAULT_REST_DAYS


def test_enrich_rest_aligns_with_roster_after_substitution():
    """rest_home must stay slot-aligned with roster_home even when the lineup changes."""
    rows = [
        _cleaned_row(1, "2003-01-01", "LAL", "BOS", ["A", "B"], ["X", "Y"]),
        _cleaned_row(2, "2003-01-04", "LAL", "BOS", ["A", "B"], ["X", "Y"]),
    ]
    # Game 2 second row: B subbed out for bench player C (who debuts this game).
    sub_row = _cleaned_row(2, "2003-01-04", "LAL", "BOS", ["A", "C"], ["X", "Y"])
    sub_row["time"] = 100
    rows.append(sub_row)

    out = enrich_df(pd.DataFrame(rows))
    g2 = out[out["game_id"] == 2].reset_index(drop=True)
    for _, row in g2.iterrows():
        roster = ast.literal_eval(str(row["roster_home"]))
        rest = ast.literal_eval(str(row["rest_home"]))
        assert len(roster) == len(rest)              # slot-for-slot alignment
    # A rested 3 days (played g1 on Jan 1); C is a debut -> 3.
    first, last = g2.iloc[0], g2.iloc[-1]
    assert ast.literal_eval(str(first["rest_home"])) == [3, 3]      # [A, B]
    assert ast.literal_eval(str(last["rest_home"])) == [3, 3]       # [A, C] (C debut=3)


# ---------------------------------------------------------------------------
# data_loading: rest columns decode to lists
# ---------------------------------------------------------------------------

def test_data_loading_parses_rest_columns(tmp_path):
    from data_loading import load_all_cleaned

    rows = [
        _cleaned_row(1, "2003-01-01", "LAL", "BOS", ["A", "B"], ["X", "Y"]),
        _cleaned_row(1, "2003-01-01", "LAL", "BOS", ["A", "B"], ["X", "Y"]),
    ]
    df = enrich_df(pd.DataFrame(rows))
    (tmp_path / "season2003.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    loaded = load_all_cleaned(str(tmp_path), parse_rosters=True)
    assert isinstance(loaded["rest_home"].iloc[0], list)
    assert isinstance(loaded["rest_away"].iloc[0], list)


# ---------------------------------------------------------------------------
# season_features: numpy helpers
# ---------------------------------------------------------------------------

def test_pad_rest_pads_to_roster_size():
    out = sf.pad_rest("[1, 20, 3]")
    assert out.shape == (5,)
    assert list(out) == [1.0, 20.0, 3.0, 0.0, 0.0]


def test_compute_rest_stats_excludes_pad_slots():
    raw = {
        "rest_home": np.array([[1.0, 3.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "rest_away": np.array([[5.0, 5.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    }
    rosters = {
        "home_roster": np.array([[7, 8, 0, 0, 0]], np.int32),   # 2 valid slots
        "away_roster": np.array([[1, 2, 0, 0, 0]], np.int32),   # 2 valid slots
    }
    mean, std = sf.compute_rest_stats(raw, rosters, pad_player=0,
                                      train_mask=np.array([True]))
    # Only the 4 valid slots [1, 3, 5, 5] feed the stats (pad zeros excluded).
    assert mean == np.mean([1, 3, 5, 5])
    assert std > 0


def test_merge_and_standardize_round_trip():
    df = pd.DataFrame({
        "rest_home": ["[2, 2]", "[10]"],
        "rest_away": ["[1, 1]", "[40]"],          # 40 will be clipped to 30
        "home_games_played": [0.0, 0.5],
        "away_games_played": [0.0, 0.5],
        "home_days_rest": [3, 1],
        "away_days_rest": [3, 2],
    })
    rosters = {
        "home_roster": np.array([[7, 8, 0, 0, 0], [9, 0, 0, 0, 0]], np.int32),
        "away_roster": np.array([[1, 2, 0, 0, 0], [3, 0, 0, 0, 0]], np.int32),
    }
    norm = {}
    cols = sf.merge_season_features(df, {}, rosters, 0, np.array([True, True]), norm)
    assert set(norm) == {"rest_mean", "rest_std"}
    # All season inputs present and the right dtype.
    for key in sf.SEASON_INPUT_KEYS:
        assert key in cols
    assert cols["rest_home"].dtype == np.float32
    # Clip: the raw 40 must standardize as if it were 30.
    clipped = (30.0 - norm["rest_mean"]) / norm["rest_std"]
    assert np.isclose(cols["rest_away"][1, 0], clipped)


def test_append_season_batches_shapes():
    cols = sf.build_raw_season_cols(pd.DataFrame({
        "rest_home": ["[1, 2]"], "rest_away": ["[3]"],
        "home_games_played": [0.1], "away_games_played": [0.2],
        "home_days_rest": [1], "away_days_rest": [2],
    }))
    batches = {k: [] for k in sf.SEASON_INPUT_KEYS}
    sf.append_season_batches(batches, cols, np.array([0]), n=1, SEQ=4)
    assert np.stack(batches["rest_home"]).shape == (1, 4, 5)
    assert np.stack(batches["home_games_played"]).shape == (1, 4, 1)
