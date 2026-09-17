"""
Rotating holdout windows: the pool, the window index, and every place k has to reach.

Four v2 runs scored the same 100 games, so their numbers are four looks at one sample rather than
four samples -- pooling them is pseudo-replication, and the direction document's SS1e quantifies how
unreadable that makes a confidence bucket. 3.0 rotates: the pool is every untrained game, a run
scores one ~100-game window of it, and the window index is recorded so drift with k is legible.

The design point worth stating, because it is what kept the change small: the pool is a SUPERSET of
the old holdout, so ``extend_holdout``'s prefix invariant is untouched. 100 is a prefix of 700.
Window 0 is byte-identical to what every earlier run scored, and no guard was weakened to get there.

Pure Python -- no TF, no simulation. ``full_run`` imports TensorFlow at module scope through its
pipeline import, so the FullRun tests below build the state dict and call the one method directly
rather than going through the CLI.
"""
from __future__ import annotations

import json

import pytest

import config
from reporting.eval_report import RUN_WINDOW_NAME, pin_run_holdout, run_window, subset_holdout


# --------------------------------------------------------------------------- the constants

def test_the_pool_divides_into_whole_windows():
    assert config.FINAL_HOLDOUT_GAMES == config.HOLDOUT_WINDOW_GAMES * config.HOLDOUT_WINDOWS


def test_the_pool_fits_in_the_untrained_tail():
    """26,969 games, cut at 26,267 -> 702 available. A pool larger than that cannot be extended to."""
    assert config.FINAL_HOLDOUT_GAMES <= 702


# --------------------------------------------------------------------------- window selection

def _window_ids(pool, k):
    """Mirror of FullRun.window_ids, exercised through the real implementation."""
    from training.full_run import FullRun

    run = FullRun.__new__(FullRun)
    run.state = {"holdout_game_ids": list(pool)}
    return FullRun.window_ids(run, k)


POOL = list(range(1000, 1000 + 700))


def test_window_zero_is_the_first_games_after_the_train_cut():
    """The compatibility guarantee: k=0 reproduces what every run before 3.0 scored."""
    assert _window_ids(POOL, 0) == POOL[:100]


def test_windows_are_disjoint_and_together_cover_the_pool():
    seen = []
    for k in range(config.HOLDOUT_WINDOWS):
        seen.extend(_window_ids(POOL, k))
    assert seen == POOL
    assert len(set(seen)) == len(seen)


def test_a_window_past_the_end_of_the_pool_says_how_to_widen_it():
    """The failure a user will actually hit, so the message has to carry the fix."""
    with pytest.raises(SystemExit, match="extend-holdout"):
        _window_ids(POOL[:100], 3)


def test_a_negative_window_is_refused():
    with pytest.raises(SystemExit):
        _window_ids(POOL, -1)


def test_a_short_final_window_is_returned_with_a_warning_not_silently_padded(capsys):
    """A pool of 650 gives a 50-game window 6. Returning it is right; pretending it is 100 is not."""
    short = POOL[:650]
    got = _window_ids(short, 6)
    assert got == short[600:]
    assert "not 100" in capsys.readouterr().out


# --------------------------------------------------------------------------- the run-dir pin

def test_the_window_is_pinned_on_first_use(tmp_path):
    pin_run_holdout(tmp_path, POOL[:100], window=2)
    assert json.loads((tmp_path / RUN_WINDOW_NAME).read_text())["k"] == 2
    assert run_window(tmp_path) == 2


def test_a_run_dir_written_before_windows_existed_reads_as_window_zero(tmp_path):
    """Every historical run scored the games immediately after the cut, which IS window 0."""
    assert run_window(tmp_path) == 0


def test_resuming_a_run_under_a_different_window_is_refused(tmp_path):
    """The dangerous case. A mismatched --holdout changes the game COUNT, so it is visible; a
    mismatched --window selects a different 100 games of the same size, and every id still looks
    legitimate. Without this, a resume would append games from another stretch of the calendar to
    a finished run and report the mixture under one headline."""
    pin_run_holdout(tmp_path, POOL[:100], window=1)
    with pytest.raises(ValueError, match="pins window k=1"):
        pin_run_holdout(tmp_path, POOL[100:200], window=3)


def test_resuming_without_naming_a_window_keeps_the_pinned_one(tmp_path):
    """A --report-only merge does not pass --window; it must not clear the pin either."""
    pin_run_holdout(tmp_path, POOL[:100], window=4)
    pin_run_holdout(tmp_path, POOL[:100])
    assert run_window(tmp_path) == 4


# --------------------------------------------------------------------------- order of operations

def test_the_window_is_applied_before_the_subset_stride():
    """"window 2, 50 games" must mean a 50-game subset OF window 2 -- never a subset of the whole
    pool that happens to overlap window 2. Both paths (FullRun.eval and eval_pool.run_procs) apply
    them in this order; reversing either would silently score the wrong games under a correct
    looking window.json."""
    window = _window_ids(POOL, 2)
    got = subset_holdout(window, 50)
    assert len(got) == 50
    assert set(got) <= set(window)
    assert set(got).isdisjoint(_window_ids(POOL, 0))


# --------------------------------------------------------------------------- the CLI and shards

def test_shard_commands_forward_the_window():
    """A child that re-reads the state file without --window would select a different 100 games of
    the same size. Same failure --dials exists to prevent, and the same fix."""
    from eval_pool import shard_commands

    shards = shard_commands(model="version3", run="v3-run1", n=2, dials_path="dials.json",
                            state_path="state.json", run_dir="results/version3/v3-run1",
                            window=3)
    for shard in shards:
        assert "--window" in shard.cmd
        assert shard.cmd[shard.cmd.index("--window") + 1] == "3"


def test_shard_commands_omit_the_window_when_it_is_zero():
    """Window 0 is the default on both sides, so the command line stays as it was."""
    from eval_pool import shard_commands

    shards = shard_commands(model="version3", run="v3-run1", n=2, dials_path="dials.json",
                            state_path="state.json", run_dir="results/version3/v3-run1")
    assert all("--window" not in shard.cmd for shard in shards)


def test_the_evaluate_parser_accepts_a_window_and_defaults_it_to_zero():
    from evaluate import build_parser

    assert build_parser().parse_args([]).window == 0
    assert build_parser().parse_args(["--window", "5"]).window == 5


# --------------------------------------------------------------------------- the report

def _aggregate_stub() -> dict:
    """The smallest aggregate _run_summary_frame will read."""
    return {
        "headline": {"pick_accuracy": 0.6, "brier": 0.23, "spread_mae": 9.9, "points_mae": 7.0},
        "progression": [],
    }


def test_the_report_records_the_window_and_the_run_summary_carries_it():
    """Pooling run_summary rows across runs is the point of rotating windows; without the column
    the pooled table cannot tell six distinct windows from six re-runs of the same one."""
    from reporting.eval_report import _run_summary_frame, build_report

    report = build_report(records=[], aggregate=_aggregate_stub(),
                          n_sims=50, window=4, run_name="v3-run5", tuning={})
    assert report["window"] == 4
    assert _run_summary_frame(report)["window_k"].iloc[0] == 4


def test_a_report_written_before_windows_existed_reads_as_window_zero():
    """update_eval_report re-aggregates historical report.json files, which carry no window field."""
    from reporting.eval_report import _run_summary_frame

    report = {"run_id": "x", "n_games": 0, "n_sims": 50, "aggregate": _aggregate_stub()}
    assert _run_summary_frame(report)["window_k"].iloc[0] == 0
