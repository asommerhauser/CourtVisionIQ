"""
Eval process pool: sizing, shard commands, supervision, waves.

``eval_pool`` is a supervisor -- it launches ``evaluate.py --shard i/N`` children and merges one
report. These tests exercise it with fake ``Popen`` objects, so nothing here spawns a real process,
loads TensorFlow, or touches a GPU.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import config
import eval_pool

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# --- Sizing                                                               --- #
# --------------------------------------------------------------------------- #

@pytest.fixture
def sizing(monkeypatch):
    """Pin every input to autosize_procs so each test moves exactly one of them."""
    monkeypatch.setattr(eval_pool, "usable_cores", lambda: 32)
    monkeypatch.setattr(eval_pool, "free_vram_gb", lambda: 24.0)
    monkeypatch.setattr(config, "EVAL_PROC_CORES", 2)
    monkeypatch.setattr(config, "EVAL_PROC_VRAM_GB", 4.0)
    monkeypatch.setattr(config, "EVAL_PROC_MAX", 8)


def test_autosize_is_bound_by_vram(sizing, monkeypatch):
    monkeypatch.setattr(eval_pool, "free_vram_gb", lambda: 16.0)
    n, why = eval_pool.autosize_procs(holdout_games=100)
    assert n == 4                       # 16.0 / 4.0
    assert "vram" in why


def test_autosize_is_bound_by_cores(sizing, monkeypatch):
    monkeypatch.setattr(eval_pool, "usable_cores", lambda: 4)
    n, why = eval_pool.autosize_procs(holdout_games=100)
    assert n == 2                       # 4 cores / 2 per proc
    assert "cpu 4/2=2" in why


def test_autosize_is_bound_by_the_amount_of_work(sizing):
    """A shard doing one game spends longer loading 11 heads than simulating."""
    n, _ = eval_pool.autosize_procs(holdout_games=3)
    assert n == 1                       # 3 // MIN_GAMES_PER_SHARD


def test_autosize_respects_the_ceiling(sizing, monkeypatch):
    monkeypatch.setattr(eval_pool, "usable_cores", lambda: 256)
    monkeypatch.setattr(eval_pool, "free_vram_gb", lambda: 512.0)
    n, why = eval_pool.autosize_procs(holdout_games=1000)
    assert n == config.EVAL_PROC_MAX
    assert "cap 8" in why


def test_autosize_subtracts_reserved_vram(sizing):
    """The resident shell holds a model; its children only get what is left."""
    free, _ = eval_pool.autosize_procs(holdout_games=100)
    held, why = eval_pool.autosize_procs(holdout_games=100, reserved_vram_gb=8.0)
    assert held == free - 2             # (24-8)/4 = 4 vs 24/4 = 6
    assert "24.0-8.0" in why


def test_autosize_falls_back_when_the_vram_probe_fails(sizing, monkeypatch):
    """No GPU probe must not mean no pool -- fall back to the core cap, and say so."""
    monkeypatch.setattr(eval_pool, "free_vram_gb", lambda: None)
    n, why = eval_pool.autosize_procs(holdout_games=100)
    assert n == 8                       # cpu cap 16, ceiling 8
    assert "vram unknown" in why


def test_explicit_procs_wins_over_the_heuristic(sizing):
    n, why = eval_pool.autosize_procs(holdout_games=100, requested=3)
    assert n == 3 and "explicit" in why


def test_explicit_procs_is_clamped_to_the_work_and_says_so(sizing):
    n, why = eval_pool.autosize_procs(holdout_games=2, requested=16)
    assert n == 2
    assert "clamped" in why


def test_explicit_procs_rejects_nonsense(sizing):
    with pytest.raises(ValueError):
        eval_pool.autosize_procs(holdout_games=100, requested=0)


@pytest.mark.parametrize("requested", [None, "auto", ""])
def test_auto_forms_all_take_the_heuristic(sizing, requested):
    n, why = eval_pool.autosize_procs(holdout_games=100, requested=requested)
    assert n == 6 and "cpu" in why


def test_autosize_never_returns_zero(sizing, monkeypatch):
    monkeypatch.setattr(eval_pool, "usable_cores", lambda: 1)
    monkeypatch.setattr(eval_pool, "free_vram_gb", lambda: 0.5)
    n, _ = eval_pool.autosize_procs(holdout_games=1)
    assert n == 1


def test_free_vram_probe_never_raises(monkeypatch):
    """Every probe path is best-effort: a machine with no NVML and no nvidia-smi returns None."""
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert eval_pool.free_vram_gb() is None


# --------------------------------------------------------------------------- #
# --- Shard commands                                                       --- #
# --------------------------------------------------------------------------- #

def _shards(tmp_path, n=4, **kw):
    kw.setdefault("model", "v1.0")
    kw.setdefault("run", "trial1")
    kw.setdefault("dials_path", tmp_path / "dials.json")
    kw.setdefault("state_path", tmp_path / "state.json")
    return eval_pool.shard_commands(n=n, run_dir=tmp_path, **kw)


def test_shard_commands_cover_the_holdout_exactly_once(tmp_path):
    shards = _shards(tmp_path, n=4)
    flags = [s.cmd[s.cmd.index("--shard") + 1] for s in shards]
    assert flags == ["1/4", "2/4", "3/4", "4/4"]
    assert len({tuple(s.cmd) for s in shards}) == 4, "no two children may run the same slice"


def test_shard_commands_pin_the_same_run_and_dials(tmp_path):
    """Same run dir and the SAME dial package for every child -- the anti-desync invariant."""
    shards = _shards(tmp_path, n=3)
    runs = {s.cmd[s.cmd.index("--run") + 1] for s in shards}
    dials = {s.cmd[s.cmd.index("--dials") + 1] for s in shards}
    assert runs == {"trial1"} and len(dials) == 1


def test_shard_commands_forward_the_perf_flags(tmp_path):
    shards = _shards(tmp_path, n=2, sims=21, concurrency=24)
    for s in shards:
        assert s.cmd[s.cmd.index("--monte-carlo") + 1] == "21"
        assert s.cmd[s.cmd.index("--concurrency") + 1] == "24"


def test_shard_commands_forward_the_seed(tmp_path):
    """A pooled run must reproduce like a single-process one, so the base seed has to travel."""
    for s in _shards(tmp_path, n=2, seed=7):
        assert s.cmd[s.cmd.index("--seed") + 1] == "7"
    for s in _shards(tmp_path, n=2):
        assert "--seed" not in s.cmd, "the default seed stays implicit"


def test_shard_commands_forward_the_holdout_subset(tmp_path):
    """Forwarded for the record, so a shard log shows the run's real shape. The run dir's pinned
    holdout.json is what actually governs which games a child sees."""
    for s in _shards(tmp_path, n=2, subset=20):
        assert s.cmd[s.cmd.index("--holdout") + 1] == "20"
    for s in _shards(tmp_path, n=2):
        assert "--holdout" not in s.cmd, "a full-holdout run stays implicit"


def test_shard_commands_never_forward_report_every(tmp_path):
    """An intermediate flush from a child writes exactly the files a shard must not touch."""
    for s in _shards(tmp_path, n=3):
        assert "--report-every" not in s.cmd


def test_shard_logs_are_distinct_files(tmp_path):
    shards = _shards(tmp_path, n=4)
    assert len({s.log for s in shards}) == 4
    assert all(s.log.parent.name == "logs" for s in shards)


# --------------------------------------------------------------------------- #
# --- Supervision                                                          --- #
# --------------------------------------------------------------------------- #

class FakePopen:
    """A child that writes `games` record.json files, then exits with `rc`."""

    def __init__(self, run_dir, games, rc=0, tag="g"):
        self.run_dir, self.games, self._rc, self.tag = Path(run_dir), games, rc, tag
        self._polls = 0

    def poll(self):
        self._polls += 1
        if self._polls < 2:
            return None                 # one tick alive, so the render path runs
        for i in range(self.games):
            d = self.run_dir / "games" / f"{self.tag}{i}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "record.json").write_text("{}", encoding="utf-8")
        return self._rc

    def wait(self, timeout=None):
        return self._rc

    def terminate(self):
        self._rc = -15

    def kill(self):
        self._rc = -9


def _patch_launch(monkeypatch, factory):
    def fake_launch(shard, *, cwd):
        shard._fh = open(shard.log, "w", encoding="utf-8")
        shard.proc = factory(shard)

    monkeypatch.setattr(eval_pool, "_launch", fake_launch)


def test_run_pool_waits_for_every_child(tmp_path, monkeypatch, capsys):
    _patch_launch(monkeypatch, lambda s: FakePopen(tmp_path, 2, tag=f"s{s.index}g"))
    shards = _shards(tmp_path, n=3)

    eval_pool.run_pool(shards, run_dir=tmp_path, total_games=6, poll=0.0)

    assert all(s.rc == 0 for s in shards)
    assert eval_pool.finished_games(tmp_path) == 6


def test_run_pool_surfaces_a_failed_shard_without_killing_its_siblings(tmp_path, monkeypatch,
                                                                      capsys):
    def factory(s):
        return FakePopen(tmp_path, 0 if s.index == 2 else 2, rc=1 if s.index == 2 else 0,
                         tag=f"s{s.index}g")

    _patch_launch(monkeypatch, factory)
    shards = _shards(tmp_path, n=3)
    eval_pool.run_pool(shards, run_dir=tmp_path, total_games=6, poll=0.0)

    assert [s.rc for s in shards] == [0, 1, 0]
    failed = eval_pool.report_failures(shards)
    assert [s.index for s in failed] == [2]
    out = capsys.readouterr().out
    assert "exited 1" in out and "re-run just this shard" in out


def test_finished_games_counts_records(tmp_path):
    assert eval_pool.finished_games(tmp_path) == 0
    for i in range(3):
        d = tmp_path / "games" / f"g{i}"
        d.mkdir(parents=True)
        (d / "record.json").write_text("{}", encoding="utf-8")
    (tmp_path / "games" / "g9").mkdir()          # started but unfinished -> not counted
    assert eval_pool.finished_games(tmp_path) == 3


# --------------------------------------------------------------------------- #
# --- Waves                                                                --- #
# --------------------------------------------------------------------------- #

def test_a_second_wave_covers_only_the_remainder(tmp_path, monkeypatch, capsys):
    """Wave 1 leaves games unfinished; wave 2 re-shards over exactly what is missing."""
    waves_seen = []

    def factory(s):
        # Wave 1 finishes 4 of 10; wave 2 finishes the rest.
        return FakePopen(tmp_path, 2 if s.log.name.count("-w") == 0 else 3, tag=f"{s.log.stem}g")

    _patch_launch(monkeypatch, factory)

    def build(wave, remaining):
        waves_seen.append((wave, remaining))
        return _shards(tmp_path, n=2, tag="" if wave == 1 else f"-w{wave}")

    eval_pool.run_waves(build=build, run_dir=tmp_path, total_games=10, max_waves=3)

    assert waves_seen[0] == (1, 10)
    assert waves_seen[1] == (2, 6), "wave 2 must target only the unfinished remainder"
    assert eval_pool.finished_games(tmp_path) == 10


def test_a_wave_that_makes_no_progress_stops_the_loop(tmp_path, monkeypatch, capsys):
    """Without this guard one permanently-failing game respawns a pool forever."""
    _patch_launch(monkeypatch, lambda s: FakePopen(tmp_path, 0, rc=1, tag="x"))
    calls = []

    def build(wave, remaining):
        calls.append(wave)
        return _shards(tmp_path, n=2, tag=f"-w{wave}")

    eval_pool.run_waves(build=build, run_dir=tmp_path, total_games=10, max_waves=5)

    assert calls == [1], "a zero-progress wave must not be followed by another"
    assert "zero new games" in capsys.readouterr().out


def test_waves_stop_once_everything_is_finished(tmp_path, monkeypatch):
    _patch_launch(monkeypatch, lambda s: FakePopen(tmp_path, 5, tag=f"s{s.index}g"))
    calls = []

    def build(wave, remaining):
        calls.append(wave)
        return _shards(tmp_path, n=2)

    eval_pool.run_waves(build=build, run_dir=tmp_path, total_games=10, max_waves=3)
    assert calls == [1]


def test_build_returning_nothing_ends_the_pool(tmp_path):
    """How a pool of 1 tells the caller to fall back to the in-process path."""
    assert eval_pool.run_waves(build=lambda w, r: [], run_dir=tmp_path, total_games=10) == []


# --------------------------------------------------------------------------- #
# --- Post-run integrity                                                   --- #
# --------------------------------------------------------------------------- #

def _record(run_dir, name, tuning):
    d = Path(run_dir) / "games" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text(json.dumps({"tuning": tuning}), encoding="utf-8")


def test_mixed_dials_across_games_are_flagged(tmp_path, capsys):
    """The quiet failure: a child ran different physics and the aggregate mixes both."""
    _record(tmp_path, "g1", {"DELTA_TIME_SCALE": 0.97})
    _record(tmp_path, "g2", {"DELTA_TIME_SCALE": 1.05})

    assert eval_pool.assert_one_tuning(tmp_path) is False
    out = capsys.readouterr().out
    assert "DIFFERENT dials" in out and "DELTA_TIME_SCALE" in out


def test_consistent_dials_pass_quietly(tmp_path, capsys):
    _record(tmp_path, "g1", {"DELTA_TIME_SCALE": 0.97})
    _record(tmp_path, "g2", {"DELTA_TIME_SCALE": 0.97})

    assert eval_pool.assert_one_tuning(tmp_path) is True
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# --- TF-free supervisor                                                   --- #
# --------------------------------------------------------------------------- #

def test_importing_eval_pool_does_not_pull_tensorflow():
    """A supervisor that loaded TF would take a CUDA context's VRAM from its own children."""
    code = "import eval_pool, sys; print('tensorflow' in sys.modules or 'keras' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


def test_run_procs_deferred_imports_resolve():
    """run_procs imports lazily so the module stays TF-free, which also means a wrong import path
    survives every unit test and only surfaces on the pod. It shipped that way once: model_name
    was imported from models.artifacts while it lived in training.full_run, so
    `evaluate.py --procs N` died on its first line."""
    code = ("from models.artifacts import model_name; from reporting.eval_report import "
            "pin_run_holdout, resolve_results_run_dir, subset_holdout; "
            "print(model_name('1.0'))")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "v1.0"


def test_a_failed_launch_does_not_orphan_the_children_already_running(tmp_path, monkeypatch):
    """A half-launched pool must not leave CUDA contexts running on a card nobody is watching."""
    launched = []

    def flaky_launch(shard, *, cwd):
        if shard.index == 3:
            raise OSError("could not start")
        shard._fh = open(shard.log, "w", encoding="utf-8")
        shard.proc = FakePopen(tmp_path, 0, tag=f"s{shard.index}g")
        launched.append(shard)

    monkeypatch.setattr(eval_pool, "_launch", flaky_launch)
    shards = _shards(tmp_path, n=4)

    with pytest.raises(OSError):
        eval_pool.run_pool(shards, run_dir=tmp_path, total_games=8, poll=0.0)

    assert len(launched) == 2
    assert all(s.rc is not None for s in launched), "survivors must have been terminated"
    assert all(s._fh is None for s in shards), "log handles must be closed"


# ===================================================================== #
# --- Reporting over finished games (mid-run and final)                --
# ===================================================================== #

def _game(run_dir, folder, payload, *, name="record.json"):
    d = run_dir / "games" / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(payload), encoding="utf-8")
    return d


def test_finished_records_ignores_partials(tmp_path):
    """A salvaged game is not a finished game: it must not reach the aggregate, and it must not
    drag the reported sim count down to whatever a killed process managed."""
    import eval_pool as ep

    _game(tmp_path, "g1", {"game_id": 1, "n_sims": 100})
    _game(tmp_path, "g2", {"game_id": 2, "n_sims": 12}, name="record.partial.json")

    records = ep.finished_records(tmp_path)

    assert [r["game_id"] for r in records] == [1]
    assert ep.finished_games(tmp_path) == 1


def test_partial_games_lists_only_the_ones_never_completed(tmp_path):
    import eval_pool as ep

    _game(tmp_path, "g1", {"n_sims": 12, "requested_n_sims": 100}, name="record.partial.json")
    # g2 was salvaged, then the remainder wave came back and finished it properly.
    _game(tmp_path, "g2", {"n_sims": 9, "requested_n_sims": 100}, name="record.partial.json")
    _game(tmp_path, "g2", {"game_id": 2, "n_sims": 100})

    assert ep.partial_games(tmp_path) == [("g1", 12, 100)]


def test_finished_records_skips_a_torn_file(tmp_path):
    """The supervisor reads these while shards write them; a half-written file is not fatal."""
    import eval_pool as ep

    _game(tmp_path, "g1", {"game_id": 1, "n_sims": 100})
    d = tmp_path / "games" / "g2"
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text('{"game_id": 2, "n_s', encoding="utf-8")

    assert [r["game_id"] for r in ep.finished_records(tmp_path)] == [1]


def test_build_run_report_uses_the_run_dials_not_the_live_config(tmp_path, monkeypatch):
    """build_report defaults `tuning` to the supervisor's own config -- which is stock config.py,
    i.e. the wrong physics for a run whose children were handed a dial file."""
    import eval_pool as ep

    _game(tmp_path, "g1", {"game_id": 1, "n_sims": 100})
    _game(tmp_path, "g2", {"game_id": 2, "n_sims": 100})
    # write_dial_file's real shape: scalars stay scalars, dict dials stay dicts.
    (tmp_path / "dials.json").write_text(
        json.dumps({"DELTA_TIME_SCALE": 1.23, "EVENT_BIAS": {"foul": 0.11}}), encoding="utf-8")

    seen = {}

    def fake_build_report(*, records, aggregate, n_sims, run_name, tuning=None):
        seen.update(n_sims=n_sims, run_name=run_name, tuning=tuning, n=len(records))
        return {"n_sims": n_sims}

    monkeypatch.setattr("reporting.eval_report.build_report", fake_build_report)
    monkeypatch.setattr("reporting.eval_report.write_eval_report",
                        lambda rep, **kw: seen.setdefault("written", kw.get("run_dir")))
    monkeypatch.setattr("simulation.eval_metrics._aggregate", lambda recs: {})
    monkeypatch.setattr("simulation.eval_metrics.print_summary", lambda *a, **kw: None)

    ep.build_run_report(tmp_path, model="v1.0")

    # Dict dials must arrive JSON-encoded, exactly as tuning_snapshot would give them: they land
    # in a single run_summary.parquet column, and a struct there would not concatenate with the
    # string column every other run wrote.
    assert seen["tuning"] == {"DELTA_TIME_SCALE": 1.23, "EVENT_BIAS": '{"foul": 0.11}'}
    assert seen["run_name"] == "v1.0", "matches what evaluate_stage stamps, so a merge looks the same"
    assert seen["n_sims"] == 100, "read off the records, not defaulted"
    assert seen["n"] == 2
    assert seen["written"] == tmp_path


def test_build_run_report_is_none_before_any_game_finishes(tmp_path):
    import eval_pool as ep

    assert ep.build_run_report(tmp_path, model="v1.0") is None


def test_periodic_report_waits_for_new_games_and_the_interval(tmp_path, monkeypatch):
    """Mid-run rebuilds are throttled two ways so a 36-hour pool is not rebuilding constantly."""
    import eval_pool as ep

    built = []
    monkeypatch.setattr(ep, "build_run_report",
                        lambda rd, **kw: built.append(ep.finished_games(rd)) or {"n_sims": 1})

    tick = ep._periodic_report(tmp_path, model="v1.0", every=0.0)
    tick()                                          # no new games -> nothing
    assert built == []

    _game(tmp_path, "g1", {"game_id": 1, "n_sims": 100})
    tick()
    assert built == [1]

    tick()                                          # still 1 game -> no rebuild
    assert built == [1]


def test_pool_tick_failure_does_not_kill_the_pool(tmp_path, monkeypatch):
    """A bookkeeping error has no business ending a run that is hours in."""
    import eval_pool as ep

    _patch_launch(monkeypatch, lambda s: FakePopen(tmp_path, 2, tag=f"s{s.index}g"))
    shards = _shards(tmp_path, n=2)

    def boom():
        raise RuntimeError("report exploded")

    ep.run_pool(shards, run_dir=tmp_path, total_games=4, poll=0.0, on_tick=boom)

    assert all(s.rc == 0 for s in shards), "the pool ran to completion despite the failing tick"


def test_report_tuning_matches_the_shape_a_normal_run_records(tmp_path):
    """A dial file round-trips into the same column types tuning_snapshot produces.

    write_dial_file keeps dict dials as dicts so they survive apply_dials; the report wants them
    JSON-encoded so each is one Parquet column. Reporting the file raw would give the pooled run's
    run_summary.parquet struct columns where every other run has strings.
    """
    import config

    config.write_dial_file(tmp_path / "dials.json")
    raw = json.loads((tmp_path / "dials.json").read_text(encoding="utf-8"))

    encoded = config.encode_tuning(raw)
    snapshot = config.tuning_snapshot()

    assert set(encoded) == set(snapshot)
    assert {k: type(v) for k, v in encoded.items()} == {k: type(v) for k, v in snapshot.items()}
    assert encoded == snapshot, "same dials in, same report row out"
