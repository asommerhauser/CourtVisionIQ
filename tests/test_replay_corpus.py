"""The replay corpus: sim ids, the columns a simulation cannot know, and the sidecar beside it."""
import pandas as pd
import pytest

from training.replay_corpus import (assert_ids_fit, attach_context, real_game_id, rest_by_player,
                                    select_replay_games, sim_game_id, sim_index_of, write_corpus,
                                    write_priors)


def _real_game(gid=5084, season=2023):
    """Two rows of a real game: five men a side, one substitution's worth of difference."""
    return pd.DataFrame([
        {"game_id": gid, "roster_home": str(["A", "B", "C", "D", "E"]),
         "roster_away": str(["F", "G", "H", "I", "J"]), "time": 0, "event": "start",
         "player": "start", "type": "start", "result": "start", "secondary_player": "none",
         "home/away": 0, "season": season, "playoff": 1, "game_date": "2022-10-18",
         "home_team": "BOS", "away_team": "PHI", "home_games_played": 0.0,
         "away_games_played": 0.0, "home_days_rest": 3, "away_days_rest": 3,
         "rest_home": str([3, 3, 3, 3, 3]), "rest_away": str([2, 2, 2, 2, 2])},
        {"game_id": gid, "roster_home": str(["A", "B", "C", "D", "X"]),
         "roster_away": str(["F", "G", "H", "I", "J"]), "time": 60, "event": "shot",
         "player": "A", "type": "2pt", "result": "made", "secondary_player": "none",
         "home/away": 1, "season": season, "playoff": 1, "game_date": "2022-10-18",
         "home_team": "BOS", "away_team": "PHI", "home_games_played": 0.0,
         "away_games_played": 0.0, "home_days_rest": 3, "away_days_rest": 3,
         "rest_home": str([3, 3, 3, 3, 9]), "rest_away": str([2, 2, 2, 2, 2])},
    ])


def test_sim_ids_round_trip():
    sid = sim_game_id(297709, 7)
    assert real_game_id(sid) == 297709 and sim_index_of(sid) == 7
    # Distinct games and distinct sims never collide, which is the whole point of the offset.
    assert len({sim_game_id(g, s) for g in (1, 2, 297709) for s in range(10)}) == 30


def test_sim_index_must_fit_the_multiplier():
    with pytest.raises(ValueError):
        sim_game_id(1, 100)


def test_ids_that_would_collide_are_refused():
    assert_ids_fit([1, 2, 297709])          # the real corpus, with room to spare
    with pytest.raises(ValueError):
        assert_ids_fit([10_000_000_000])    # a corpus that reaches the base


def test_select_is_deterministic_and_sized():
    pool = range(1000)
    first = select_replay_games(pool, fraction=0.1, seed=7)
    assert first == select_replay_games(pool, fraction=0.1, seed=7)
    assert first != select_replay_games(pool, fraction=0.1, seed=8)
    assert len(first) == 100 and first == sorted(first)


def test_select_always_returns_at_least_one_game():
    assert len(select_replay_games([4, 5], fraction=0.001, seed=1)) == 1


def test_rest_is_read_over_every_row_not_just_the_first():
    rest, home_default, away_default = rest_by_player(_real_game())
    # X only ever appears in the second row; a first-row read would miss him entirely.
    assert rest["X"] == 9.0 and rest["A"] == 3.0 and rest["F"] == 2.0
    assert home_default == 3.0 and away_default == 2.0


def test_context_is_carried_and_rest_follows_the_sims_own_roster():
    real = _real_game()
    # A sim that benched E for X from the very first row: the real row's rest list is aligned to the
    # wrong man, which is exactly the failure this re-lay exists to prevent.
    sim = pd.DataFrame([
        {"game_id": sim_game_id(5084, 0), "roster_home": str(["A", "B", "C", "D", "X"]),
         "roster_away": str(["F", "G", "H", "I", "J"]), "time": 0, "event": "start",
         "player": "start", "type": "start", "result": "start", "secondary_player": "none",
         "home/away": 0, "season": 2023, "playoff": 1},
    ])
    out = attach_context(sim, real)
    assert out["game_date"].iloc[0] == "2022-10-18"
    assert out["home_team"].iloc[0] == "BOS" and out["away_days_rest"].iloc[0] == 3
    assert out["rest_home"].iloc[0] == str([3.0, 3.0, 3.0, 3.0, 9.0])
    assert out["rest_away"].iloc[0] == str([2.0, 2.0, 2.0, 2.0, 2.0])


def test_an_unknown_player_falls_to_the_sides_median_rest():
    real = _real_game()
    sim = pd.DataFrame([
        {"game_id": sim_game_id(5084, 1), "roster_home": str(["A", "B", "C", "D", "ZZ"]),
         "roster_away": str(["F", "G", "H", "I", "J"]), "time": 0, "event": "start",
         "player": "start", "type": "start", "result": "start", "secondary_player": "none",
         "home/away": 0, "season": 2023, "playoff": 1},
    ])
    out = attach_context(sim, real)
    assert out["rest_home"].iloc[0].endswith("3.0]")


def test_corpus_is_written_one_file_per_season(tmp_path):
    a = attach_context(_real_game(season=2022).assign(game_id=sim_game_id(1, 0)), _real_game())
    b = attach_context(_real_game(season=2023).assign(game_id=sim_game_id(2, 0)), _real_game())
    written = write_corpus([a, b], tmp_path / "corpus")
    assert {p.name for p in written} == {"season2022.csv", "season2023.csv"}
    back = pd.read_csv(tmp_path / "corpus" / "season2023.csv")
    assert set(back["game_id"]) == {sim_game_id(2, 0)}


def _priors(tmp_path, game_ids):
    root = tmp_path / "data" / "priors"
    root.mkdir(parents=True)
    pd.DataFrame({"game_id": game_ids, "player": ["A"] * len(game_ids),
                  "min_pg": [30.0] * len(game_ids)}).to_parquet(root / "players_2023.parquet")
    pd.DataFrame({"game_id": game_ids, "side": ["home"] * len(game_ids),
                  "pace": [99.0] * len(game_ids)}).to_parquet(root / "teams_2023.parquet")
    return tmp_path / "data"


def test_priors_are_rekeyed_onto_the_sim_ids(tmp_path):
    data_dir = _priors(tmp_path, [5084])
    sim_ids = [sim_game_id(5084, i) for i in range(3)]
    write_priors(data_dir, tmp_path / "corpus", sim_ids)

    players = pd.read_parquet(tmp_path / "corpus" / "priors" / "players_2023.parquet")
    assert sorted(players["game_id"]) == sorted(sim_ids)
    # merge_prior_features RAISES on partial coverage, so every replayed sim must be present.
    assert len(players) == 3


def test_priors_that_cover_nothing_are_refused(tmp_path):
    data_dir = _priors(tmp_path, [999999])
    with pytest.raises(ValueError):
        write_priors(data_dir, tmp_path / "corpus", [sim_game_id(5084, 0)])
