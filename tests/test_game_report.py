"""Per-game HTML box report + the results-run directory resolver."""
from reporting.eval_report import resolve_results_run_dir
from reporting.game_report import render_game_html
from simulation.stats import BOX_STATS


def _line(v):
    return {f: float(v) for f in BOX_STATS}


def _record():
    return {
        "game_id": 42, "n_sims": 21, "win_prob_home": 0.6,
        "pred_home_score": 100.4, "pred_away_score": 98.6,
        "actual_home_score": 108, "actual_away_score": 95,
        "players": {"home": ["A", "B"], "away": ["F"]},
        "player_avg": {"home": {"A": _line(10), "B": _line(5)}, "away": {"F": _line(8)}},
        "player_std": {"home": {"A": _line(1), "B": _line(0.5)}, "away": {"F": _line(2)}},
        "player_actual": {"home": {"A": _line(12)}, "away": {"F": _line(6)}},
        "team_pred": {"home": _line(100), "away": _line(98)},
        "team_std": {"home": _line(3), "away": _line(3)},
        "team_actual": {"home": _line(108), "away": _line(95)},
    }


def test_game_html_has_three_box_scores_and_team_rows():
    html = render_game_html(_record(), home_team="PHI", away_team="ORL")
    assert "Predicted box score" in html
    assert "Actual box score" in html
    assert "Variance (predicted" in html
    assert "PHI (TEAM)" in html and "ORL (TEAM)" in html
    assert html.startswith("<!DOCTYPE html>") and "game 42" in html.lower()


def test_results_run_dir_auto_increments_and_resumes(tmp_path):
    root = str(tmp_path)
    d1 = resolve_results_run_dir("1.0", holdout_total=3, results_root=root)
    assert d1.name == "eval-001"

    # An incomplete run (0 < 3 records) is resumed, not replaced.
    (d1 / "games").mkdir(parents=True, exist_ok=True)
    assert resolve_results_run_dir("1.0", holdout_total=3, results_root=root) == d1

    # Once complete, the next call opens a fresh run.
    for i in range(3):
        g = d1 / "games" / f"g{i}"
        g.mkdir(parents=True)
        (g / "record.json").write_text("{}", encoding="utf-8")
    assert resolve_results_run_dir("1.0", holdout_total=3, results_root=root).name == "eval-002"


def test_results_run_dir_named_is_stable(tmp_path):
    root = str(tmp_path)
    a = resolve_results_run_dir("1.0", name="pace-097", results_root=root)
    b = resolve_results_run_dir("1.0", name="pace-097", results_root=root)
    assert a == b and a.name == "pace-097"
