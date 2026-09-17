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

    def fake_build_report(*, records, aggregate, n_sims, run_name, window=0):
        seen["n_sims"] = n_sims
        seen["window"] = window
        return {"n_sims": n_sims, "n_games": len(records)}

    monkeypatch.setattr("reporting.eval_report.build_report", fake_build_report)
    monkeypatch.setattr("reporting.eval_report.write_eval_report",
                        lambda rep, **kw: tmp_path / "run")
    monkeypatch.setattr(se, "_aggregate", lambda recs: {})
    monkeypatch.setattr(se, "print_summary", lambda *a, **kw: None)
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


# --------------------------------------------------------------------------- #
# --- Durability: streaming sink + salvage on a killed process             --- #
# --------------------------------------------------------------------------- #

def _inflight(tmp_path, monkeypatch, *, boxes, n_sims=10, gid=42):
    """An _InFlight over a stubbed record builder, so the salvage logic is what's under test."""
    import simulation.stage_eval as se

    written = {}

    def fake_build_game_record(game, bx, *, n_sims, seed_base, home_team, away_team):
        return {"game_id": gid, "n_sims": n_sims, "seed_base": seed_base}

    def fake_write(out_dir, game, bx, record, home_team, away_team):
        out_dir.mkdir(parents=True, exist_ok=True)
        name = "record.partial.json" if record.get("partial") else "record.json"
        (out_dir / name).write_text(json.dumps(record), encoding="utf-8")
        written["record"] = record
        written["name"] = name

    monkeypatch.setattr(se, "build_game_record", fake_build_game_record)
    monkeypatch.setattr(se, "_write_game_folder", fake_write)

    g = se._InFlight(gid=gid, out_dir=tmp_path / "game42", game=object(),
                     home_team="PHI", away_team="ORL", n_sims=n_sims, seed0=7, boxes=boxes)
    return se, g, written


def test_salvage_writes_a_partial_record_never_the_completion_marker(tmp_path, monkeypatch):
    """The whole design rests on this: record.json stays the sole 'this game is done' marker.

    If salvage wrote record.json, finished_games would count the game, the remainder wave would
    never re-run it, and reported_sims (which takes the min) would relabel the entire run at the
    salvaged count. A partial file keeps the data and keeps the game re-runnable.
    """
    se, g, written = _inflight(tmp_path, monkeypatch, boxes=[object()] * 4, n_sims=10)

    se._salvage_one(g, "signal 15")

    assert written["name"] == "record.partial.json"
    assert (g.out_dir / "record.partial.json").exists()
    assert not (g.out_dir / "record.json").exists(), "a salvaged game is NOT finished"
    assert written["record"]["partial"] is True
    assert written["record"]["n_sims"] == 4, "the record says what it was actually built from"
    assert written["record"]["requested_n_sims"] == 10


def test_salvage_skips_a_game_that_already_finished(tmp_path, monkeypatch):
    se, g, written = _inflight(tmp_path, monkeypatch, boxes=[object()] * 4)
    g.out_dir.mkdir(parents=True, exist_ok=True)
    (g.out_dir / "record.json").write_text("{}", encoding="utf-8")

    se._salvage_one(g, "atexit")

    assert written == {}, "nothing to salvage - the game is already done"


def test_salvage_skips_a_game_with_nothing_finished(tmp_path, monkeypatch):
    se, g, written = _inflight(tmp_path, monkeypatch, boxes=[None, None, None])

    se._salvage_one(g, "atexit")

    assert written == {}
    assert not (g.out_dir / "record.partial.json").exists()


def test_salvage_ignores_unfinished_sims_and_counts_only_real_boxes(tmp_path, monkeypatch):
    """boxes_out is pre-sized to n_sims and fills in live, so most of it is None mid-game."""
    boxes = [object(), None, object(), None, None]
    se, g, written = _inflight(tmp_path, monkeypatch, boxes=boxes, n_sims=5)

    se._salvage_one(g, "signal 15")

    assert written["record"]["n_sims"] == 2


def test_salvage_all_runs_once_and_survives_a_failing_game(tmp_path, monkeypatch):
    """It runs from a signal handler: a raising game must not stop the others, or fire twice."""
    import simulation.stage_eval as se

    calls = []

    def flaky(g, reason):
        calls.append(g.gid)
        if g.gid == 2:
            raise RuntimeError("disk full")

    monkeypatch.setattr(se, "_salvage_one", flaky)
    monkeypatch.setattr(se, "_SALVAGED", False)
    monkeypatch.setattr(se, "_INFLIGHT", [
        se._InFlight(gid=i, out_dir=tmp_path / str(i), game=None, home_team="H", away_team="A",
                     n_sims=4, seed0=0, boxes=[object()]) for i in (1, 2, 3)])

    se._salvage_all("signal 15")
    se._salvage_all("atexit")       # atexit fires after the handler; must be a no-op

    assert calls == [1, 2, 3], "every game attempted exactly once, despite game 2 raising"


# --------------------------------------------------------------------------- #
# --- _PbpSink                                                             --- #
# --------------------------------------------------------------------------- #

def _sink(tmp_path, monkeypatch, *, n_sims, gid=9):
    import simulation.stage_eval as se

    monkeypatch.setattr(se, "history_to_cleaned_frame",
                        lambda history, spec, game_id: pd.DataFrame([{"e": history[0]}]))
    chunk = [{"gid": gid, "out_dir": tmp_path / "game9", "spec": object(),
              "game": pd.DataFrame([{"game_id": gid}])}]
    return se._PbpSink(chunk, n_sims=n_sims), chunk


def test_sink_filename_width_comes_from_the_requested_sim_count(tmp_path, monkeypatch):
    """At 100 sims a two-digit width sorts sim_9 after sim_100. The finished list no longer
    exists at write time, so the width has to come from n_sims."""
    sink, chunk = _sink(tmp_path, monkeypatch, n_sims=100)
    (chunk[0]["out_dir"] / "playbyplay").mkdir(parents=True)

    sink(0, 0, ["a"], None)
    sink(0, 99, ["b"], None)

    names = sorted(q.name for q in (chunk[0]["out_dir"] / "playbyplay").glob("sim_*.csv"))
    assert names == ["sim_001_playbyplay.csv", "sim_100_playbyplay.csv"]


def test_sink_keeps_the_legacy_two_digit_width_at_stage_sims(tmp_path, monkeypatch):
    sink, chunk = _sink(tmp_path, monkeypatch, n_sims=21)
    (chunk[0]["out_dir"] / "playbyplay").mkdir(parents=True)

    sink(0, 0, ["a"], None)

    assert (chunk[0]["out_dir"] / "playbyplay" / "sim_01_playbyplay.csv").exists()


def test_sink_prepare_clears_a_previous_attempt(tmp_path, monkeypatch):
    """A run killed at sim 43 of 100 then re-run at 21 sims would otherwise leave sim_022..043
    orphaned beside a record.json claiming 21."""
    sink, chunk = _sink(tmp_path, monkeypatch, n_sims=21)
    pbp = chunk[0]["out_dir"] / "playbyplay"
    pbp.mkdir(parents=True)
    for i in (1, 22, 43):
        (pbp / f"sim_{i:03d}_playbyplay.csv").write_text("stale", encoding="utf-8")
    (chunk[0]["out_dir"] / "record.partial.json").write_text("{}", encoding="utf-8")

    sink.prepare()

    assert list(pbp.glob("sim_*_playbyplay.csv")) == [], "stale sims cleared"
    assert not (chunk[0]["out_dir"] / "record.partial.json").exists(), "stale partial cleared"
    assert (pbp / "actual_playbyplay.csv").exists()


def test_sink_write_failure_costs_one_pbp_not_the_sim(tmp_path, monkeypatch):
    import simulation.stage_eval as se
    sink, chunk = _sink(tmp_path, monkeypatch, n_sims=4)
    monkeypatch.setattr(se, "history_to_cleaned_frame",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))

    sink(0, 0, ["a"], None)         # must not raise


# --------------------------------------------------------------------------- #
# --- _write_game_folder: which filename marks a game done                 --- #
# --------------------------------------------------------------------------- #

class _FakeBox:
    home_score = 101.0
    away_score = 99.0

    def to_frame(self, side):
        return pd.DataFrame([{"Player": "TEAM", "pts": 1}])

    def render(self):
        return "box"


def _write(tmp_path, monkeypatch, record):
    """Call the REAL _write_game_folder with only the rendering stubbed out."""
    import simulation.stage_eval as se

    monkeypatch.setattr(se, "generate_box_score", lambda *a, **kw: _FakeBox())
    monkeypatch.setattr(se, "_averaged_box", lambda *a, **kw: _FakeBox())
    monkeypatch.setattr(se, "render_game_html", lambda *a, **kw: "<html></html>")

    out_dir = tmp_path / "game1"
    se._write_game_folder(out_dir, pd.DataFrame([{"game_id": 1}]), [_FakeBox()], record,
                          "PHI", "ORL")
    return out_dir


def _base_record(**kw):
    rec = {"game_id": 1, "n_sims": 100, "pred_home_score": 101.0, "pred_away_score": 99.0,
           "actual_home_score": 100, "actual_away_score": 98, "win_prob_home": 0.6}
    rec.update(kw)
    return rec


def test_a_normal_record_lands_as_the_completion_marker(tmp_path, monkeypatch):
    out_dir = _write(tmp_path, monkeypatch, _base_record())

    assert (out_dir / "record.json").exists()
    assert not (out_dir / "record.partial.json").exists()


def test_a_partial_record_never_lands_as_the_completion_marker(tmp_path, monkeypatch):
    """If this ever regressed, a salvaged game would look finished: the wave would skip it and the
    run would silently report a killed game's handful of sims as its final answer."""
    out_dir = _write(tmp_path, monkeypatch, _base_record(partial=True, n_sims=12,
                                                         requested_n_sims=100))

    assert (out_dir / "record.partial.json").exists()
    assert not (out_dir / "record.json").exists()

    # The marker the resume path and eval_pool.finished_games actually count:
    assert list(out_dir.parent.glob("*/record.json")) == []


def test_writes_leave_no_temp_files_behind(tmp_path, monkeypatch):
    """record.json is written tmp+replace because the supervisor now reads it concurrently."""
    out_dir = _write(tmp_path, monkeypatch, _base_record())

    assert list(out_dir.glob("*.tmp")) == []
    assert json.loads((out_dir / "record.json").read_text(encoding="utf-8"))["game_id"] == 1
