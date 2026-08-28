"""
Play-by-play harvesting: what is safe to archive, and what must never be deleted.

``harvest.py`` deletes data next to a live pool, so every test here is about a guard rather than a
feature: a folder without ``record.json`` is untouched, a bad archive keeps its source, a prune
refuses while a shard is alive. Nothing here spawns a real eval or loads TensorFlow.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import harvest

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# --- Fixtures                                                             --- #
# --------------------------------------------------------------------------- #

def make_game(run_dir: Path, name: str, *, finished: bool, n_pbp: int = 3,
              partial: bool = False) -> Path:
    """A game folder shaped like the real one: box CSVs, an HTML, pbp, and maybe a record."""
    g = run_dir / "games" / name
    (g / "playbyplay").mkdir(parents=True, exist_ok=True)
    (g / "pred_boxscore_home.csv").write_text("player,pts\nA,10\n", encoding="utf-8")
    (g / "game.html").write_text("<html></html>", encoding="utf-8")
    (g / "playbyplay" / "actual_playbyplay.csv").write_text("time,event\n0,tip\n", encoding="utf-8")
    for i in range(n_pbp):
        (g / "playbyplay" / f"sim_{i + 1:03d}_playbyplay.csv").write_text(
            f"time,event\n{i},shot\n", encoding="utf-8")
    if finished:
        (g / "record.json").write_text(json.dumps({"game_id": 1, "n_sims": n_pbp}), encoding="utf-8")
    if partial:
        (g / "record.partial.json").write_text(json.dumps({"n_sims": 1}), encoding="utf-8")
    return g


@pytest.fixture
def run_dir(tmp_path) -> Path:
    d = tmp_path / "results" / "v1.0" / "run"
    make_game(d, "game1_2023-01-10_AatH", finished=True)
    make_game(d, "game2_2023-01-11_AatH", finished=True)
    make_game(d, "game3_2023-01-12_AatH", finished=False, partial=True)
    return d


@pytest.fixture
def no_pool(monkeypatch):
    """The prune modes ask /proc whether a shard is alive; say no."""
    monkeypatch.setattr(harvest, "live_evals", lambda *a, **k: [])


@pytest.fixture
def log():
    return harvest.Log(None)


# --------------------------------------------------------------------------- #
# --- Archiving                                                            --- #
# --------------------------------------------------------------------------- #

def test_archive_tars_finished_games_and_frees_their_pbp(run_dir, tmp_path, log):
    out = tmp_path / "harvest"
    out.mkdir()
    n, freed = harvest.cmd_archive(run_dir, out, log=log, dry_run=False, keep=0, min_free_gb=0.0)

    assert n == 2 and freed > 0
    for name in ("game1_2023-01-10_AatH", "game2_2023-01-11_AatH"):
        assert (out / f"{name}.tar.gz").is_file()
        assert not (run_dir / "games" / name / "playbyplay").exists()
        # The record survives, which is the whole point: it is the completion marker AND the report.
        assert (run_dir / "games" / name / "record.json").is_file()


def test_the_archive_is_self_contained(run_dir, tmp_path, log):
    out = tmp_path / "harvest"
    out.mkdir()
    harvest.cmd_archive(run_dir, out, log=log, dry_run=False, keep=0, min_free_gb=0.0)
    with tarfile.open(out / "game1_2023-01-10_AatH.tar.gz") as tf:
        names = tf.getnames()
    # Rooted at the game folder, so `tar -xzf ... -C games/` lands it back in place.
    assert all(n.startswith("game1_2023-01-10_AatH") for n in names)
    assert "game1_2023-01-10_AatH/record.json" in names
    assert "game1_2023-01-10_AatH/playbyplay/sim_001_playbyplay.csv" in names


def test_an_unfinished_game_is_never_touched(run_dir, tmp_path, log):
    """No ``record.json`` means a live shard owns the folder -- its streamed sims are not ours."""
    out = tmp_path / "harvest"
    out.mkdir()
    harvest.cmd_archive(run_dir, out, log=log, dry_run=False, keep=0, min_free_gb=0.0)
    unfinished = run_dir / "games" / "game3_2023-01-12_AatH"
    assert (unfinished / "playbyplay").is_dir()
    assert len(list((unfinished / "playbyplay").iterdir())) == 4
    assert not (out / "game3_2023-01-12_AatH.tar.gz").exists()


def test_dry_run_writes_nothing(run_dir, tmp_path, log):
    out = tmp_path / "harvest"
    out.mkdir()
    n, freed = harvest.cmd_archive(run_dir, out, log=log, dry_run=True, keep=0, min_free_gb=0.0)
    assert n == 2 and freed > 0
    assert list(out.iterdir()) == []
    assert (run_dir / "games" / "game1_2023-01-10_AatH" / "playbyplay").is_dir()


def test_a_second_pass_is_a_no_op(run_dir, tmp_path, log):
    out = tmp_path / "harvest"
    out.mkdir()
    harvest.cmd_archive(run_dir, out, log=log, dry_run=False, keep=0, min_free_gb=0.0)
    n, freed = harvest.cmd_archive(run_dir, out, log=log, dry_run=False, keep=0, min_free_gb=0.0)
    assert (n, freed) == (0, 0)
    assert len(harvest.read_manifest(out)) == 2


def test_keep_leaves_the_newest_game_hot(run_dir, tmp_path, log):
    out = tmp_path / "harvest"
    out.mkdir()
    newest = run_dir / "games" / "game2_2023-01-11_AatH" / "record.json"
    import os
    import time
    os.utime(newest, (time.time() + 100, time.time() + 100))
    n, _ = harvest.cmd_archive(run_dir, out, log=log, dry_run=False, keep=1, min_free_gb=0.0)
    assert n == 1
    assert (run_dir / "games" / "game2_2023-01-11_AatH" / "playbyplay").is_dir()


def test_manifest_records_one_line_per_game(run_dir, tmp_path, log):
    out = tmp_path / "harvest"
    out.mkdir()
    harvest.cmd_archive(run_dir, out, log=log, dry_run=False, keep=0, min_free_gb=0.0)
    lines = (out / harvest.MANIFEST_NAME).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    entry = json.loads(lines[0])
    assert entry["game"] == "game1_2023-01-10_AatH"
    assert entry["pbp_files"] == 4 and entry["sha256"] and entry["archive_bytes"] > 0


def test_a_bad_archive_leaves_the_source_alone(run_dir, tmp_path, log, monkeypatch):
    """Verify-then-delete: if the tar does not match the source, the pbp stays put."""
    out = tmp_path / "harvest"
    out.mkdir()
    game = run_dir / "games" / "game1_2023-01-10_AatH"

    real = harvest.file_sizes

    def lying_sizes(root: Path):
        sizes = real(root)
        if root == game:
            sizes["game1_2023-01-10_AatH/playbyplay/ghost.csv"] = 123   # a member the tar won't have
        return sizes

    monkeypatch.setattr(harvest, "file_sizes", lying_sizes)
    assert harvest.archive_game(game, out, log=log) is None
    assert (game / "playbyplay").is_dir()
    assert not (out / "game1_2023-01-10_AatH.tar.gz").exists()
    assert not (out / "game1_2023-01-10_AatH.tar.gz.tmp").exists()


# --------------------------------------------------------------------------- #
# --- Pruning                                                              --- #
# --------------------------------------------------------------------------- #

def test_prune_finished_deletes_only_the_listed_games(run_dir, tmp_path, log, no_pool):
    names = tmp_path / "already-home.txt"
    names.write_text("# already on the laptop\ngame1_2023-01-10_AatH\n", encoding="utf-8")
    n, freed = harvest.cmd_prune_finished(run_dir, names, log=log, dry_run=False, force=False)
    assert n == 1 and freed > 0
    assert not (run_dir / "games" / "game1_2023-01-10_AatH" / "playbyplay").exists()
    assert (run_dir / "games" / "game2_2023-01-11_AatH" / "playbyplay").is_dir()
    assert (run_dir / "games" / "game1_2023-01-10_AatH" / "record.json").is_file()


def test_prune_finished_refuses_a_list_that_does_not_match(run_dir, tmp_path, log, no_pool):
    """A name that is not finished here means the two machines disagree -- refuse the whole thing."""
    names = tmp_path / "already-home.txt"
    names.write_text("game1_2023-01-10_AatH\ngame3_2023-01-12_AatH\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        harvest.cmd_prune_finished(run_dir, names, log=log, dry_run=False, force=False)
    assert (run_dir / "games" / "game1_2023-01-10_AatH" / "playbyplay").is_dir()


def test_prune_unfinished_clears_part_written_folders_only(run_dir, tmp_path, log, no_pool):
    n, freed = harvest.cmd_prune_unfinished(run_dir, log=log, dry_run=False, force=False)
    assert n == 1 and freed > 0
    unfinished = run_dir / "games" / "game3_2023-01-12_AatH"
    assert not (unfinished / "playbyplay").exists()
    # The salvaged record stays; a full re-run unlinks it itself (_PbpSink.prepare).
    assert (unfinished / "record.partial.json").is_file()
    assert (run_dir / "games" / "game1_2023-01-10_AatH" / "playbyplay").is_dir()


@pytest.mark.parametrize("mode", ["finished", "unfinished"])
def test_pruning_refuses_while_a_shard_is_alive(run_dir, tmp_path, log, monkeypatch, mode):
    monkeypatch.setattr(harvest, "live_evals",
                        lambda *a, **k: [(4242, "python evaluate.py --shard 1/16")])
    names = tmp_path / "already-home.txt"
    names.write_text("game1_2023-01-10_AatH\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        if mode == "finished":
            harvest.cmd_prune_finished(run_dir, names, log=log, dry_run=False, force=False)
        else:
            harvest.cmd_prune_unfinished(run_dir, log=log, dry_run=False, force=False)
    assert (run_dir / "games" / "game1_2023-01-10_AatH" / "playbyplay").is_dir()
    assert (run_dir / "games" / "game3_2023-01-12_AatH" / "playbyplay").is_dir()


def test_force_overrides_the_live_pool_guard(run_dir, tmp_path, log, monkeypatch):
    monkeypatch.setattr(harvest, "live_evals",
                        lambda *a, **k: [(4242, "python evaluate.py --shard 1/16")])
    n, _ = harvest.cmd_prune_unfinished(run_dir, log=log, dry_run=False, force=True)
    assert n == 1


# --------------------------------------------------------------------------- #
# --- CLI + TF-free                                                        --- #
# --------------------------------------------------------------------------- #

def test_status_reports_the_split(run_dir, capsys):
    assert harvest.main(["--run", str(run_dir), "--status"]) == 0
    out = capsys.readouterr().out
    assert "2 finished, 1 unfinished" in out
    assert "finished game ids: 1 2" in out


def test_archive_without_out_is_an_error(run_dir):
    with pytest.raises(SystemExit):
        harvest.main(["--run", str(run_dir)])


def test_prune_finished_without_a_list_is_an_error(run_dir):
    with pytest.raises(SystemExit):
        harvest.main(["--run", str(run_dir), "--prune-finished"])


def test_importing_harvest_does_not_pull_tensorflow():
    """It runs beside the workers; a CUDA context here is VRAM taken from a shard."""
    code = "import harvest, sys; print('tensorflow' in sys.modules or 'keras' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"
