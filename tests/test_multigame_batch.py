"""
Tests for multi-game batched eval (simulate_games): pool several games' sims into ONE rollout.

A single game only keeps ~2 sims on the same head at once, so the GPU sat ~10% utilized. Pooling
many games' sims fills the batch — in ONE process, so there's no cross-process VRAM contention.
These tests pin the pooling contract (one batched call, correct seeds/rosters per job) and the
split-back (each game gets its own n_sims histories + the right teams) on CPU, with the batched
rollout and box-score builder stubbed — no TF, no trained model.
"""
from __future__ import annotations

import simulation.batched_rollout as br
import simulation.evaluation as ev


class _FakeSpec:
    def __init__(self, tag):
        self.home_roster = [f"{tag}_H"]
        self.away_roster = [f"{tag}_A"]
        self.season = 2003

    def season_context(self):
        return {}


def _patch(monkeypatch):
    """Stub run_jobs_batched (one fake history per job) + generate_box_score (echoes the job)."""
    seen = {}

    def fake_run_jobs_batched(sim, jobs, *, batch_size, show_progress=False):
        seen["seeds"] = [j.seed for j in jobs]
        seen["rosters"] = [j.home_roster[0] for j in jobs]
        seen["batch_size"] = batch_size
        seen["n_calls"] = seen.get("n_calls", 0) + 1
        return [[{"job_index": i}] for i in range(len(jobs))]

    # simulate_games does a local `from simulation.batched_rollout import ... run_jobs_batched`.
    monkeypatch.setattr(br, "run_jobs_batched", fake_run_jobs_batched)
    monkeypatch.setattr(ev, "generate_box_score",
                        lambda h, home_team, away_team: (h[0]["job_index"], home_team, away_team))
    return seen


def test_pools_all_games_into_one_batched_call(monkeypatch):
    seen = _patch(monkeypatch)
    games = [(_FakeSpec("g0"), None, None, "H0", "A0"),
             (_FakeSpec("g1"), None, None, "H1", "A1"),
             (_FakeSpec("g2"), None, None, "H2", "A2")]
    ev.simulate_games(sim=None, games=games, n_sims=4, seed0=100, batch_size=48)

    assert seen["n_calls"] == 1                 # one pooled rollout, not one-per-game
    assert len(seen["seeds"]) == 12             # 3 games x 4 sims
    assert seen["batch_size"] == 48
    # Each game contributes n_sims jobs seeded seed0..seed0+n_sims-1, carrying its own roster.
    assert seen["seeds"] == [100, 101, 102, 103] * 3
    assert seen["rosters"] == ["g0_H"] * 4 + ["g1_H"] * 4 + ["g2_H"] * 4


def test_splits_results_back_per_game_with_right_teams(monkeypatch):
    _patch(monkeypatch)
    games = [(_FakeSpec("g0"), None, None, "H0", "A0"),
             (_FakeSpec("g1"), None, None, "H1", "A1"),
             (_FakeSpec("g2"), None, None, "H2", "A2")]
    out = ev.simulate_games(sim=None, games=games, n_sims=4, seed0=0, batch_size=48)

    assert len(out) == 3
    # Game 0 gets jobs 0-3, game 1 gets 4-7, game 2 gets 8-11 (contiguous, in order).
    boxes0, hist0 = out[0]
    boxes2, hist2 = out[2]
    assert [b[0] for b in boxes0] == [0, 1, 2, 3]
    assert [b[0] for b in boxes2] == [8, 9, 10, 11]
    assert len(hist0) == 4 and len(hist2) == 4
    # Home/away teams routed to the correct game's box scores.
    assert boxes0[0][1] == "H0" and boxes0[0][2] == "A0"
    assert boxes2[0][1] == "H2" and boxes2[0][2] == "A2"


def test_single_game_still_works(monkeypatch):
    seen = _patch(monkeypatch)
    out = ev.simulate_games(sim=None, games=[(_FakeSpec("solo"), None, None, "H", "A")],
                            n_sims=3, seed0=7, batch_size=16)
    assert seen["seeds"] == [7, 8, 9]
    assert len(out) == 1 and len(out[0][0]) == 3
