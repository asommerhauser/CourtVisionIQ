"""
GameSimulator tests: load → seed → shape history → forward pass, plus the state rules.

These train a tiny Event/Time model on synthetic cleaned data (same pattern as
test_model_persistence.py), then exercise the simulator end to end on CPU:

  - load(): pulls the event_time model + wrapper out of ModelBundle
  - build_model_inputs(): shape/dtype parity with EventTimeModel.INPUT_KEYS
  - predict_next(): finite logits, a normalized event distribution, real-seconds Δt
  - state rules: possession flips on change-of-possession; substitution swaps the roster
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import ROSTER_SIZE, STINT_MAX_SECONDS
from encoder.encoder import Encoder
from models.event_time_model import EventTimeModel
from models.stint_length_model import StintLengthModel
from models.substitution_model import SubstitutionModel
from simulation.game_simulator import GameSimulator

TEST_SEQ_LEN = 16
HOME_FIVE = ["A", "B", "C", "D", "E"]
AWAY_FIVE = ["F", "G", "H", "I", "J"]

# Season-context columns the (post-season-enrichment) preprocess consumes.
_SEASON = {
    "rest_home": str([2.0] * 5), "rest_away": str([2.0] * 5),
    "home_games_played": 0.5, "away_games_played": 0.5,
    "home_days_rest": 2.0, "away_days_rest": 2.0,
}


def _roster(players):
    return str(list(players))


def _make_csv(path: Path) -> None:
    """Three short games framed by start/end rows, with one substitution, so the vocab
    covers every token the simulator feeds at inference time."""
    rows = []
    for gid, n in [(1, 6), (2, 5), (3, 5)]:
        # start frame
        rows.append(_row(gid, 0, "start", "start", "start", "start", "none"))
        for i in range(1, n):
            rows.append(_row(gid, i * 10, "shot", "A", "2pt", "missed", "none"))
        rows.append(_row(gid, n * 10, "end", "end", "end", "end", "none"))
    # ensure a substitution token exists in the vocab
    rows.append(_row(1, 5, "substitution", "K", "substitution", "substitution", "B"))
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(gid, time, event, player, type_, result, secondary):
    return {
        "game_id": gid,
        "roster_home": _roster(HOME_FIVE),
        "roster_away": _roster(AWAY_FIVE),
        "time": time,
        "event": event,
        "player": player,
        "type": type_,
        "result": result,
        "secondary_player": secondary,
        "season": "2003",
        **_SEASON,
    }


def _train_tiny(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_csv(data_dir / "season_clean.csv")

    vocab_dir = tmp_path / "vocabs"
    artifacts_root = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"

    inst = EventTimeModel(
        Encoder(vocab_dir=vocab_dir),
        sequence_length=TEST_SEQ_LEN,
        path=str(data_dir),
        processed_dir=str(processed_dir),
    )
    inst.preprocess(rebuild_vocabs=True, test_frac=0.34)
    inst.train(epochs=1, batch_size=2, artifacts_root=str(artifacts_root),
               mixed_precision=False, report=False)
    return artifacts_root, vocab_dir, data_dir, processed_dir


def _load_sim(tmp_path: Path) -> GameSimulator:
    artifacts_root, vocab_dir, data_dir, processed_dir = _train_tiny(tmp_path)
    return GameSimulator.load(
        artifacts_root=str(artifacts_root),
        encoder=Encoder(vocab_dir=vocab_dir),
        sequence_length=TEST_SEQ_LEN,
        path=str(data_dir),
        processed_dir=str(processed_dir),
    )


def test_build_inputs_shape_parity(tmp_path):
    """Inputs match the model's expected keys/shapes/dtypes (batch = 1)."""
    sim = _load_sim(tmp_path)
    sim.start_game(HOME_FIVE, AWAY_FIVE, season="2003")
    sim.append_event("shot", "A", "2pt", "missed")

    inputs = sim.build_model_inputs()
    assert set(inputs.keys()) == set(EventTimeModel.INPUT_KEYS)

    SEQ = TEST_SEQ_LEN
    for f in ("event", "player", "type", "result", "season", "secondary_player"):
        assert inputs[f].shape == (1, SEQ) and inputs[f].dtype == np.int32
    for f in ("home_roster", "away_roster"):
        assert inputs[f].shape == (1, SEQ, ROSTER_SIZE) and inputs[f].dtype == np.int32
    for f in ("time_abs", "delta_time"):
        assert inputs[f].shape == (1, SEQ, 1) and inputs[f].dtype == np.float32
    assert inputs["pad_mask"].shape == (1, SEQ)
    # Two real steps (start seed + one event), rest padded.
    assert inputs["pad_mask"].sum() == 2

    # The model accepts these inputs and returns the two heads at full seq length.
    out = sim.model(inputs, training=False)
    assert set(out.keys()) == {"event_output", "time_output"}


def test_predict_next_returns_raw_distribution(tmp_path):
    """Forward pass yields finite logits, a normalized event dist, and seconds Δt."""
    sim = _load_sim(tmp_path)
    sim.start_game(HOME_FIVE, AWAY_FIVE, season="2003")
    for _ in range(5):
        sim.append_event("shot", "A", "2pt", "missed", time=10)

    pred = sim.predict_next()
    n_events = sim.encoder.event_vocab.next_token
    assert pred["event_logits"].shape == (n_events,)
    assert np.isfinite(pred["event_logits"]).all()
    assert pytest.approx(1.0, abs=1e-5) == pred["event_probs"].sum()
    assert pytest.approx(1.0, abs=1e-5) == sum(pred["event_dist"].values())
    # decoded keys are real event names
    assert "start" in pred["event_dist"] and "shot" in pred["event_dist"]
    assert np.isfinite(pred["delta_seconds"])


def test_possession_flips_on_change_of_possession(tmp_path):
    sim = _load_sim(tmp_path)
    sim.start_game(HOME_FIVE, AWAY_FIVE, possession="home", season="2003")
    sim.append_event("shot", "A", "2pt", "missed")
    assert sim.possession == "home"           # a miss alone does not flip
    sim.append_event("rebound", "F", "defensive", "cop")
    assert sim.possession == "away"            # change-of-possession outcome flips it
    assert sim.history[-1]["possession"] == "away"


def test_substitution_swaps_roster(tmp_path):
    sim = _load_sim(tmp_path)
    sim.start_game(HOME_FIVE, AWAY_FIVE, season="2003")
    # Convention: player = outgoing (B, on the home five), secondary_player = incoming (K).
    sim.append_event("substitution", "B", "substitution", "substitution",
                     secondary_player="K")
    assert "B" not in sim.home_roster and "K" in sim.home_roster
    assert len(sim.home_roster) == ROSTER_SIZE
    # The post-substitution lineup is what the event row (and next inputs) carry.
    assert sim.history[-1]["roster_home"] == sim.home_roster


def _train_tiny_with_substitution(tmp_path: Path):
    """Train event_time + substitution into one artifacts dir (for the bootstrap test)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_csv(data_dir / "season_clean.csv")
    vocab_dir = tmp_path / "vocabs"
    artifacts_root = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"

    et = EventTimeModel(Encoder(vocab_dir=vocab_dir), sequence_length=TEST_SEQ_LEN,
                        path=str(data_dir), processed_dir=str(processed_dir))
    et.preprocess(rebuild_vocabs=True, test_frac=0.34)
    et.train(epochs=1, batch_size=2, artifacts_root=str(artifacts_root),
             mixed_precision=False, report=False)

    sub = SubstitutionModel(Encoder(vocab_dir=vocab_dir), sequence_length=TEST_SEQ_LEN,
                            path=str(data_dir), processed_dir=str(processed_dir))
    sub.preprocess(rebuild_vocabs=False, test_frac=0.34, holdout_frac=0.0)
    sub.train(epochs=1, batch_size=2, artifacts_root=str(artifacts_root),
              mixed_precision=False, report=False)
    return artifacts_root, vocab_dir, data_dir, processed_dir


def test_opening_bootstrap_builds_five_from_full_rosters(tmp_path):
    """start_from_full_rosters builds each team's five from its whole roster via subs."""
    artifacts_root, vocab_dir, data_dir, processed_dir = _train_tiny_with_substitution(tmp_path)
    sim = GameSimulator.load(
        artifacts_root=str(artifacts_root), encoder=Encoder(vocab_dir=vocab_dir),
        sequence_length=TEST_SEQ_LEN, path=str(data_dir), processed_dir=str(processed_dir),
    )
    assert SubstitutionModel.KEY in sim.heads

    home_full = HOME_FIVE + ["K"]  # six available; the head picks five
    sim.start_from_full_rosters(home_full, AWAY_FIVE, greedy=True)

    # Each team has a distinct five drawn only from its supplied full roster.
    assert len(sim.home_roster) == ROSTER_SIZE and len(set(sim.home_roster)) == ROSTER_SIZE
    assert set(sim.home_roster) <= set(home_full)
    assert set(sim.away_roster) == set(AWAY_FIVE)

    # Opening: empty start frame + 10 "start -> starter" subs (5 per team).
    assert sim.history[0]["event"] == "start" and sim.history[0]["roster_home"] == []
    subs = [r for r in sim.history if r["event"] == "substitution"]
    assert len(subs) == 2 * ROSTER_SIZE
    assert all(r["player"] == "start" for r in subs)
    # The incoming starters land in secondary_player.
    assert {r["secondary_player"] for r in subs} == set(sim.home_roster) | set(sim.away_roster)


def test_conditioned_inputs_carry_incoming_player(tmp_path):
    """The stint-head conditioning attaches next_secondary_player (the decided incoming player)."""
    sim = _load_sim(tmp_path)
    sim.start_game(HOME_FIVE, AWAY_FIVE, season="2003")
    sim.append_event("shot", "A", "2pt", "missed")

    inputs = sim._conditioned_inputs(next_event="substitution", delta_seconds=0.0,
                                     next_player="B", next_secondary_player="K")
    assert "next_secondary_player" in inputs
    n = min(len(sim.history), TEST_SEQ_LEN)
    enc = sim.encoder
    # The decided incoming player is encoded at the decision position (n-1).
    assert inputs["next_secondary_player"][0, n - 1] == enc.encode_secondary_player("K")
    assert inputs["next_player"][0, n - 1] == enc.encode_player("B")


def _train_tiny_with_stint(tmp_path: Path):
    """Train event_time + substitution + stint_length into one artifacts dir."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_csv(data_dir / "season_clean.csv")
    vocab_dir = tmp_path / "vocabs"
    artifacts_root = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"

    et = EventTimeModel(Encoder(vocab_dir=vocab_dir), sequence_length=TEST_SEQ_LEN,
                        path=str(data_dir), processed_dir=str(processed_dir))
    et.preprocess(rebuild_vocabs=True, test_frac=0.34)
    et.train(epochs=1, batch_size=2, artifacts_root=str(artifacts_root),
             mixed_precision=False, report=False)

    for cls in (SubstitutionModel, StintLengthModel):
        m = cls(Encoder(vocab_dir=vocab_dir), sequence_length=TEST_SEQ_LEN,
                path=str(data_dir), processed_dir=str(processed_dir))
        m.preprocess(rebuild_vocabs=False, test_frac=0.34, holdout_frac=0.0)
        m.train(epochs=1, batch_size=2, artifacts_root=str(artifacts_root),
                mixed_precision=False, report=False)
    return artifacts_root, vocab_dir, data_dir, processed_dir


def test_predict_stint_length_is_finite_and_capped(tmp_path):
    """The stint head loads, carries its own log-stint norm stats, and yields a capped, finite,
    non-negative stint length in seconds."""
    artifacts_root, vocab_dir, data_dir, processed_dir = _train_tiny_with_stint(tmp_path)
    sim = GameSimulator.load(
        artifacts_root=str(artifacts_root), encoder=Encoder(vocab_dir=vocab_dir),
        sequence_length=TEST_SEQ_LEN, path=str(data_dir), processed_dir=str(processed_dir),
    )
    assert StintLengthModel.KEY in sim.heads
    assert "stint_log_mean" in sim.stint_norm_stats and "stint_log_std" in sim.stint_norm_stats

    sim.start_game(HOME_FIVE, AWAY_FIVE, season="2003")
    length = sim.predict_stint_length("K", "B", greedy=True)  # greedy: no sampling noise
    assert np.isfinite(length)
    assert 0.0 <= length <= STINT_MAX_SECONDS


def test_constrained_sample_respects_candidates(tmp_path):
    """_constrained_sample restricts the head's full-vocab logits to the candidate set."""
    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    for p in ("A", "B", "C", "D"):
        enc.encode_player(p)

    class _Inst:
        encoder = enc
        norm_stats: dict = {}
        sequence_length = TEST_SEQ_LEN

    sim = GameSimulator(model=None, instance=_Inst())
    logits = np.zeros((enc.player_vocab.next_token,), dtype=np.float32)
    logits[enc.encode_player("B")] = 10.0   # globally preferred
    logits[enc.encode_player("A")] = 5.0

    # With B available, greedy picks B.
    assert sim._constrained_sample(logits, ["A", "B", "C"], greedy=True) == "B"
    # Excluding B, greedy falls to the next-best *candidate* (A), never the masked-out B.
    assert sim._constrained_sample(logits, ["A", "C", "D"], greedy=True) == "A"
