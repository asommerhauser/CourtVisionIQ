import numpy as np
import pandas as pd

from training.chronology import (
    build_schedule,
    game_index,
    sequential_partition,
)


def _roster(players):
    return str(list(players))


def _season_csv(path, season, n_regular, n_playoff, start_day):
    """Write one cleaned season CSV: n_regular regular + n_playoff playoff games, 3 rows each.

    game_dates increase across games; playoff games are dated after the regular season so the
    chronological sort keeps them last within the season.
    """
    rows = []
    gid = 0
    day = start_day
    for phase_flag, count in ((1, n_regular), (2, n_playoff)):
        for _ in range(count):
            gid += 1
            date = f"{season - 1}-12-{day:02d}" if day <= 28 else f"{season}-01-{day - 28:02d}"
            for i, ev in enumerate(("start", "shot", "end")):
                rows.append({
                    "game_id": gid,
                    "roster_home": _roster(["A", "B", "C", "D", "E"]),
                    "roster_away": _roster(["F", "G", "H", "I", "J"]),
                    "time": i * 10,
                    "event": ev,
                    "player": "A",
                    "type": "t",
                    "result": "r",
                    "secondary_player": "none",
                    "season": season,
                    "playoff": phase_flag,
                    "game_date": date,
                })
            day += 1
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_corpus(tmp_path, seasons=(2003, 2004, 2005, 2006), n_regular=8, n_playoff=4):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for k, s in enumerate(seasons):
        _season_csv(data_dir / f"season{s}.csv", s, n_regular, n_playoff, start_day=1 + k)
    return data_dir


def test_game_index_orders_chronologically(tmp_path):
    data_dir = _build_corpus(tmp_path)
    idx = game_index(str(data_dir))

    # 4 seasons x (8 + 4) games.
    assert len(idx) == 4 * 12
    # Seasons appear in non-decreasing order, and within a season regular precedes playoff.
    assert idx["season"].is_monotonic_increasing
    for s, grp in idx.groupby("season"):
        assert grp["playoff"].is_monotonic_increasing
        assert (grp[grp["is_regular"]]["season_reg_ordinal"].to_numpy()
                == np.arange(8)).all()
        assert (grp["season_reg_count"] == 8).all()
    # pos is a dense 0..N-1 chronological rank.
    assert (idx["pos"].to_numpy() == np.arange(len(idx))).all()


def test_sequential_partition_is_chronological_and_disjoint(tmp_path):
    data_dir = _build_corpus(tmp_path)
    idx = game_index(str(data_dir))
    ordered = idx["game_id"].to_numpy()

    boundary = 20
    train, val, holdout = sequential_partition(idx, boundary, n_holdout=5, val_frac=0.25, seed=42)

    # Holdout is exactly the next 5 games after the boundary, in order.
    assert sorted(holdout) == sorted(int(g) for g in ordered[boundary:boundary + 5])
    # Train + val together are exactly the games before the boundary.
    assert train | val == set(int(g) for g in ordered[:boundary])
    # The three sets are pairwise disjoint.
    assert not (train & val) and not (train & holdout) and not (val & holdout)
    # Val is a non-empty minority carved for early stopping.
    assert 0 < len(val) < boundary


def test_build_schedule_cycles_and_steps(tmp_path):
    seasons = tuple(range(2003, 2013))  # 10 seasons
    data_dir = _build_corpus(tmp_path, seasons=seasons)
    idx = game_index(str(data_dir))

    # seasons_per_stage=1 -> a stop every season, the 25/50/pre-playoffs point rotating.
    sched1 = build_schedule(idx, seasons_per_stage=1, n_holdout=2)
    bounds = [st["boundary_idx"] for st in sched1 if st["boundary_type"] != "final_full"]
    assert bounds == sorted(bounds) and len(set(bounds)) == len(bounds)
    assert [st["boundary_type"] for st in sched1][:3] == ["frac:0.25", "frac:0.50", "pre_playoffs"]
    assert sched1[-1]["boundary_type"] == "final_full"
    assert sched1[-1]["boundary_idx"] == len(idx) and sched1[-1]["holdout_game_ids"] == []
    for st in sched1[:-1]:
        assert len(st["holdout_game_ids"]) == 2

    # seasons_per_stage=3 -> stops land 3 seasons apart (seasons[3], [6], [9]) -> far fewer stages.
    sched3 = build_schedule(idx, seasons_per_stage=3, n_holdout=2)
    stop_seasons = [st["season"] for st in sched3 if st["boundary_type"] != "final_full"]
    assert stop_seasons == [seasons[3], seasons[6], seasons[9]]
    assert len(sched3) < len(sched1)


def test_pre_playoffs_holdout_is_first_playoff_games(tmp_path):
    data_dir = _build_corpus(tmp_path, seasons=tuple(range(2003, 2013)))
    idx = game_index(str(data_dir))
    sched = build_schedule(idx, seasons_per_stage=1, n_holdout=2)

    by_id = idx.set_index("game_id")
    for st in sched:
        if st["boundary_type"] == "pre_playoffs":
            # The held-out games right after a pre-playoffs boundary must be playoff games.
            assert all(int(by_id.loc[g, "playoff"]) == 2 for g in st["holdout_game_ids"])
