"""The arm comparison driver: what it reads, what it refuses, and what it writes."""
import json

import pytest

from reporting.ab_report import compare, pairs_for, read_arm, render_html


def _run(tmp_path, name, *, n=6, window=0, seed=0, sims=200, flip=0):
    """A results dir with ``n`` finished games. ``flip`` games are predicted backwards."""
    run_dir = tmp_path / "results" / "version3.2" / name
    for i in range(n):
        game = run_dir / "games" / f"game{1000 + i}"
        game.mkdir(parents=True)
        p_home = 0.2 if i < flip else 0.8
        (game / "record.json").write_text(json.dumps({
            "game_id": 1000 + i, "n_sims": sims, "seed_base": seed,
            "win_prob_home": p_home, "actual_home_win": True,
            "pred_margin_mean": 4.0, "pred_margin_std": 11.0, "actual_margin": 3,
        }), encoding="utf-8")
    (run_dir / "report.json").write_text(
        json.dumps({"n_sims": sims, "window": window, "run_name": name}), encoding="utf-8")
    return run_dir


def test_pairs_are_the_increments_plus_the_whole():
    assert pairs_for(3) == [(0, 1), (1, 2), (0, 2)]
    # With two arms the increment IS the whole; reporting it twice would suggest two findings.
    assert pairs_for(2) == [(0, 1)]
    assert pairs_for(1) == []


def test_identity_comes_from_the_run_not_the_command_line(tmp_path):
    identity, records = read_arm(_run(tmp_path, "v32-a1", window=2, seed=11, sims=50), "retrained")
    assert identity["arm"] == "retrained" and identity["window"] == 2
    assert identity["seed"] == 11 and identity["monte_carlo"] == 50
    assert identity["n_games"] == len(records) == 6


def test_an_unevaluated_arm_is_refused(tmp_path):
    empty = tmp_path / "results" / "version3.2" / "v32-a9"
    empty.mkdir(parents=True)
    with pytest.raises(SystemExit, match="no finished games"):
        read_arm(empty, "kpi")


def test_a_run_that_mixes_seed_bases_is_not_one_arm(tmp_path):
    run_dir = _run(tmp_path, "v32-mixed")
    stray = run_dir / "games" / "game9999"
    stray.mkdir(parents=True)
    (stray / "record.json").write_text(json.dumps({
        "game_id": 9999, "n_sims": 200, "seed_base": 77,
        "win_prob_home": 0.5, "actual_home_win": True}), encoding="utf-8")
    with pytest.raises(SystemExit, match="mixes seed bases"):
        read_arm(run_dir, "rung2")


def test_arms_on_different_windows_are_refused(tmp_path):
    a = _run(tmp_path, "v32-a1", window=0)
    b = _run(tmp_path, "v32-a2", window=3)
    with pytest.raises(ValueError, match="window"):
        compare([a, b], echo=lambda *_: None)


def test_arms_at_different_sim_counts_are_refused(tmp_path):
    a = _run(tmp_path, "v32-a1", sims=200)
    b = _run(tmp_path, "v32-a2", sims=50)
    with pytest.raises(ValueError, match="monte_carlo"):
        compare([a, b], echo=lambda *_: None)


def test_a_comparable_set_is_compared_and_written(tmp_path):
    a = _run(tmp_path, "v32-a1", flip=3)     # half its games predicted backwards
    b = _run(tmp_path, "v32-a2")             # all of them right
    c = _run(tmp_path, "v32-a3")
    report = compare([a, b, c], out_dir=tmp_path / "out", echo=lambda *_: None)

    assert [x["arm"] for x in report["arms"]] == ["retrained", "rung2", "kpi"]
    # Three pairs, each in both the vote and the score view.
    assert len(report["comparisons"]) == 6
    first = report["comparisons"][0]
    assert first["a"] == "retrained" and first["b"] == "rung2"
    assert first["brier_a"] > first["brier_b"] and first["separated"]
    # The two arms that are byte-identical cannot separate, whatever the threshold.
    same = [c for c in report["comparisons"] if (c["a"], c["b"]) == ("rung2", "kpi")]
    assert same and all(c["verdict"] == "the same model" for c in same)
    assert (tmp_path / "out" / "ab_report.json").is_file()
    assert "<h1>3.2 A/B arms</h1>" in (tmp_path / "out" / "ab_report.html").read_text(encoding="utf-8")


def test_the_comparison_is_recorded_in_the_run_state(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"version": "version3.2"}), encoding="utf-8")
    compare([_run(tmp_path, "v32-a1"), _run(tmp_path, "v32-a2")],
            state_path=str(state), out_dir=tmp_path / "out", echo=lambda *_: None)

    saved = json.loads(state.read_text(encoding="utf-8"))
    assert "ab_arms" in saved and len(saved["ab_arms"]["arms"]) == 2


def test_html_survives_a_missing_number():
    report = {"arms": [{"arm": "kpi", "description": "d", "window": 0, "seed": 0,
                        "monte_carlo": 200, "n_games": 3, "run_dir": "x"}],
              "comparisons": [{"a": "retrained", "b": "kpi", "n": 0, "score_view": False,
                               "threshold_2se": 0.01, "separated": False,
                               "verdict": "the same model"}]}
    assert "the same model" in render_html(report)
