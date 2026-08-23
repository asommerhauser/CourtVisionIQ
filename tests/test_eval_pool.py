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
