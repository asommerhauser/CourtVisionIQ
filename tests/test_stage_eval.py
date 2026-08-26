"""Stage-eval pure helpers: descriptive game labels + averaged predicted box score, plus the
report's sim count and the pool width."""
import json

import pandas as pd
import pytest

from simulation.stage_eval import _averaged_box, _game_labels
from simulation.stats import BOX_STATS


def _record(gid=5):
    home = {"A": {f: 1.0 for f in BOX_STATS}, "B": {f: 0.0 for f in BOX_STATS}}
    away = {"F": {f: 2.0 for f in BOX_STATS}}
    return {
        "game_id": gid,
        "player_avg": {"home": home, "away": away},
        "pred_home_score": 100.4, "pred_away_score": 98.6,
    }


def test_game_labels_from_teams_and_date():
    game = pd.DataFrame([{"game_id": 5, "home_team": "PHI", "away_team": "ORL",
                          "game_date": "2002-11-01"}])
    home, away, label = _game_labels(game)
    assert (home, away) == ("PHI", "ORL")
    assert label.startswith("game5_") and "ORLatPHI" in label
    # No path separators or spaces leak into the folder name.
    assert "/" not in label and " " not in label


def test_game_labels_fall_back_when_team_missing():
    game = pd.DataFrame([{"game_id": 7, "home_team": None, "away_team": "", "game_date": None}])
    home, away, label = _game_labels(game)
    assert (home, away) == ("HOME", "AWAY")
    assert label == "game7_AWAYatHOME"


def test_averaged_box_renders_with_score():
    box = _averaged_box(_record(), "PHI", "ORL")
    assert box.home_team == "PHI" and box.away_team == "ORL"
    assert box.home_score == 100.4 and box.away_score == 98.6
    # Averaged float stats render through the standard box-score path without error.
    frame = box.to_frame("home")
    assert "TEAM" in frame["Player"].tolist()
    assert "PHI" in box.render()


# --------------------------------------------------------------------------- #
# --- Reported sim count + pool width                                      --- #
# --------------------------------------------------------------------------- #

def _run_stage(tmp_path, monkeypatch, *, records, n_sims):
    """Drive evaluate_stage over already-finished games (max_new=0) and capture the report."""
    import simulation.stage_eval as se

    seen = {}

    def fake_build_report(*, records, aggregate, n_sims, run_name):
        seen["n_sims"] = n_sims
        return {"n_sims": n_sims, "n_games": len(records)}

    monkeypatch.setattr("reporting.eval_report.build_report", fake_build_report)
    monkeypatch.setattr("reporting.eval_report.write_eval_report",
                        lambda rep, **kw: tmp_path / "run")
    monkeypatch.setattr(se, "_aggregate", lambda recs: {})
    monkeypatch.setattr(se, "_print_summary", lambda *a, **kw: None)
    monkeypatch.setattr(se, "load_all_cleaned", lambda *a, **kw: _cleaned(records))

    run_dir = tmp_path / "run"
    for rec in records:
        d = run_dir / "games" / f"game{rec['game_id']}_AWAYatHOME"
        d.mkdir(parents=True, exist_ok=True)
        (d / "record.json").write_text(json.dumps(rec), encoding="utf-8")

    se.evaluate_stage("v1.0", holdout_ids=[r["game_id"] for r in records], n_sims=n_sims,
                      max_new=0, results_run_dir=run_dir, data_dir=str(tmp_path))
    return seen


def _cleaned(records):
    return pd.DataFrame([{"game_id": r["game_id"], "time": 0, "home_team": "HOME",
                          "away_team": "AWAY", "game_date": None} for r in records])


def test_report_uses_the_sim_count_the_games_were_run_at(tmp_path, monkeypatch):
    """A merge is called with whatever n_sims happens to default; the records know the truth."""
    records = [{"game_id": 1, "n_sims": 100}, {"game_id": 2, "n_sims": 100}]
    seen = _run_stage(tmp_path, monkeypatch, records=records, n_sims=21)
    assert seen["n_sims"] == 100


def test_mixed_sim_counts_report_the_smallest_and_warn(tmp_path, monkeypatch, capsys):
    records = [{"game_id": 1, "n_sims": 100}, {"game_id": 2, "n_sims": 21}]
    seen = _run_stage(tmp_path, monkeypatch, records=records, n_sims=100)
    assert seen["n_sims"] == 21
    assert "mixes sim counts" in capsys.readouterr().out


def test_pre_seed_records_fall_back_to_the_requested_count(tmp_path, monkeypatch):
    """Records written before n_sims was recorded carry nothing to read back."""
    records = [{"game_id": 1}, {"game_id": 2}]
    seen = _run_stage(tmp_path, monkeypatch, records=records, n_sims=21)
    assert seen["n_sims"] == 21


@pytest.mark.parametrize("n_sims,expected", [(21, 6), (100, 1), (50, 2), (1, 6)])
def test_pool_width_is_capped_in_sims_not_games(n_sims, expected):
    """run_jobs_batched holds every finished history until the pool drains, so the memory peak is
    games x sims. 21 sims must keep today's 6-game pool exactly; 100 sims must collapse to 1."""
    from config import EVAL_GAMES_PER_BATCH, EVAL_POOL_JOBS

    assert max(1, min(EVAL_GAMES_PER_BATCH, EVAL_POOL_JOBS // max(1, n_sims))) == expected
