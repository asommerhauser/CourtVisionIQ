"""
Box-score decoder tests.

These exercise the cleaned-data event semantics that the legacy notebook got wrong, plus
minutes accrual and the shared game-split partitioning. They import the decoder directly
(no TensorFlow / trained model needed), so they run fast on CPU.
"""
from __future__ import annotations

from data_loading import split_games
from simulation.box_score import generate_box_score

HOME = ["A", "B", "C", "D", "E"]
AWAY = ["F", "G", "H", "I", "J"]


def _row(time, event, player, type_, result, secondary="none",
         home=HOME, away=AWAY):
    return {
        "event": event, "player": player, "type": type_, "result": result,
        "secondary_player": secondary, "time": time,
        "roster_home": list(home), "roster_away": list(away),
    }


def _by_name(box):
    return {pl.player: pl for pl in (*box.home, *box.away)}


def test_scoring_and_shooting_splits():
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "shot", "A", "2pt", "made"),      # +2 home
        _row(20, "shot", "A", "2pt", "missed"),    # FGA only
        _row(30, "shot", "B", "3pt", "made"),      # +3 home
        _row(40, "shot", "F", "3pt", "missed"),    # away FGA/3PA only
        _row(50, "shot", "F", "free throw", "made"),    # +1 away
        _row(60, "shot", "F", "free throw", "missed"),  # FTA only
        _row(70, "end", "end", "end", "end"),
    ]
    box = generate_box_score(events)
    p = _by_name(box)

    assert (p["A"].fgm, p["A"].fga, p["A"].pts) == (1, 2, 2)
    assert (p["B"].tpm, p["B"].tpa, p["B"].fgm, p["B"].fga, p["B"].pts) == (1, 1, 1, 1, 3)
    assert (p["F"].tpa, p["F"].tpm) == (1, 0)
    assert (p["F"].ftm, p["F"].fta, p["F"].pts) == (1, 2, 1)
    assert box.home_score == 5
    assert box.away_score == 1
    # Final score equals the sum of each side's player points.
    assert box.home_score == sum(pl.pts for pl in box.home)
    assert box.away_score == sum(pl.pts for pl in box.away)


def test_steal_double_event():
    # Cleaning emits two turnover rows for a steal: the stealer (result="steal") and the
    # player who lost the ball (result="cop"). Only the latter is a turnover.
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "turnover", "F", "steal", "steal"),  # F steals
        _row(10, "turnover", "A", "steal", "cop"),    # A loses the ball
        _row(20, "end", "end", "end", "end"),
    ]
    p = _by_name(generate_box_score(events))
    assert (p["F"].stl, p["F"].tov) == (1, 0)
    assert (p["A"].tov, p["A"].stl) == (1, 0)


def test_non_steal_turnover_counts():
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "turnover", "A", "violation", "cop"),
        _row(20, "end", "end", "end", "end"),
    ]
    p = _by_name(generate_box_score(events))
    assert (p["A"].tov, p["A"].stl) == (1, 0)


def test_blocked_shot():
    # The shooter's row carries result="blocked" (a missed FGA); the blocker gets a block.
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "shot", "A", "2pt", "blocked"),
        _row(10, "block", "F", "2pt", "block", secondary="A"),
        _row(20, "end", "end", "end", "end"),
    ]
    p = _by_name(generate_box_score(events))
    assert (p["A"].fga, p["A"].fgm, p["A"].pts) == (1, 0, 0)
    assert p["F"].blk == 1


def test_rebounds_assists_fouls():
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(10, "rebound", "A", "offensive", "null"),
        _row(20, "rebound", "B", "defensive", "cop"),
        _row(30, "assist", "C", "2pt", "score"),
        _row(40, "foul", "D", "shooting", "free throw"),
        _row(50, "foul", "E", "technical", "free throw"),  # technical: NOT a personal foul
        _row(60, "end", "end", "end", "end"),
    ]
    p = _by_name(generate_box_score(events))
    assert (p["A"].oreb, p["A"].reb) == (1, 1)
    assert (p["B"].dreb, p["B"].reb) == (1, 1)
    assert p["C"].ast == 1
    assert p["D"].pf == 1
    assert p["E"].pf == 0


def test_minutes_accrual():
    # 120s of clock with the same five on court => 2.0 minutes each.
    events = [
        _row(0, "start", "start", "start", "start"),
        _row(60, "shot", "A", "2pt", "made"),
        _row(120, "shot", "F", "2pt", "made"),
        _row(120, "end", "end", "end", "end"),
    ]
    p = _by_name(generate_box_score(events))
    assert p["A"].minutes == 2.0
    assert p["F"].minutes == 2.0


def test_split_games_disjoint_and_deterministic():
    ids = list(range(100))
    train, test, holdout = split_games(ids, seed=42, test_frac=0.2, holdout_frac=0.1)
    assert len(train) == 70 and len(test) == 20 and len(holdout) == 10
    assert train.isdisjoint(test) and train.isdisjoint(holdout) and test.isdisjoint(holdout)
    assert train | test | holdout == set(ids)
    # Same seed reproduces the exact partition (so every model + the box-score tool agree).
    assert split_games(ids, seed=42, test_frac=0.2, holdout_frac=0.1) == (train, test, holdout)
