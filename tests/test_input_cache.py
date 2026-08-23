"""
Incremental input-cache tests.

``simulation/input_cache.HistoryEncoder`` encodes each history row once, as it is appended,
instead of rebuilding the whole ``(1, SEQ, ...)`` input window on every model call. These tests
pin it to the one thing that matters: it must produce **bit-identical** arrays to
``GameSimulator._build_model_inputs_uncached``, the original builder kept in the tree as the
oracle. Everything runs on CPU with a stub instance -- ``build_model_inputs`` never touches
``self.model``, so no TF graph or trained artifact is needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from config import ROSTER_SIZE
from encoder.encoder import Encoder
from models.event_time_model import EventTimeModel
from models.substitution_model import START_TOKEN, SUB_EVENT
from simulation.game_simulator import HOME, GameSimulator

SEQ = 16                     # CAP = 32, so a few hundred rows exercise many compactions
HOME_FULL = [f"H{i}" for i in range(12)]
AWAY_FULL = [f"A{i}" for i in range(12)]

_NORM_STATS = {"max_time": 2880.0, "delta_mean": 12.5, "delta_std": 9.25,
               "rest_mean": 2.5, "rest_std": 1.75}


class _Instance:
    """The three attributes GameSimulator.__init__ reads off an EventTimeModel wrapper."""

    def __init__(self, encoder: Encoder):
        self.encoder = encoder
        self.norm_stats = dict(_NORM_STATS)
        self.sequence_length = SEQ


def _encoder(tmp_path) -> Encoder:
    """A vocab warmed with every token the script below emits, then frozen.

    Freezing matters: an unfrozen vocab *mints* ids on first sight, and the cached path meets
    tokens in a different order than the batch path (row-major vs column-major), so an unwarmed
    vocab would assign different ids and the comparison would be about vocab growth rather than
    about the cache.
    """
    enc = Encoder(vocab_dir=tmp_path / "vocabs")
    for p in (*HOME_FULL, *AWAY_FULL, START_TOKEN, "nan"):
        enc.encode_player(p)
        enc.encode_secondary_player(p)
    for e in ("start", "end", SUB_EVENT, "shot", "foul", "rebound", "turnover", "assist"):
        enc.encode_event(e)
    for t in ("start", "end", SUB_EVENT, "2pt", "3pt", "free throw", "technical",
              "offensive", "personal", "offensive-rebound", "nan"):
        enc.encode_type(t)
    for r in ("start", "end", SUB_EVENT, "made", "missed", "cop", "steal", "nan"):
        enc.encode_result(r)
    enc.encode_season("2003")
    for v in enc.vocabs.values():
        v.freeze()
    return enc


def _sim(tmp_path) -> GameSimulator:
    sim = GameSimulator(None, _Instance(_encoder(tmp_path)))
    sim.home_full, sim.away_full = list(HOME_FULL), list(AWAY_FULL)
    sim.home_roster, sim.away_roster = HOME_FULL[:5], AWAY_FULL[:5]
    sim.season = "2003"
    sim._set_season_context({
        "home_games_played": 0.42, "away_games_played": 0.61,
        "home_days_rest": 1.0, "away_days_rest": 3.0,
        "home_rest": {p: 1.0 + i * 0.5 for i, p in enumerate(HOME_FULL)},
        "away_rest": {p: 0.5 + i * 0.25 for i, p in enumerate(AWAY_FULL)},
    })
    return sim


def _assert_identical(sim: GameSimulator, where: str) -> None:
    """Every array, exactly equal -- same values, dtype, shape, and key order."""
    cached = sim.build_model_inputs()
    oracle = sim._build_model_inputs_uncached()
    assert list(cached) == list(oracle) == list(EventTimeModel.INPUT_KEYS), where
    for k in oracle:
        assert cached[k].dtype == oracle[k].dtype, f"{where}: {k} dtype"
        assert cached[k].shape == oracle[k].shape, f"{where}: {k} shape"
        np.testing.assert_array_equal(cached[k], oracle[k], err_msg=f"{where}: {k}")


def _play(sim: GameSimulator, *, event, player, type, result, secondary, time) -> None:
    sim._append_row(sim._make_row(event=event, player=player, type=type, result=result,
                                  secondary_player=secondary, time=float(time)))


def _script(sim: GameSimulator, n_rows: int = 260):
    """Drive a game that hits every branch the encoder has to reproduce.

    Yields after each append so the caller can compare at every single row.
    """
    _play(sim, event="start", player="start", type="start", result="start",
          secondary="none", time=0.0)
    yield "start frame"

    # Opening five, built as start -> starter subs (the real bootstrap shape).
    for i in range(ROSTER_SIZE):
        for roster, full in ((sim.home_roster, HOME_FULL), (sim.away_roster, AWAY_FULL)):
            _play(sim, event=SUB_EVENT, player=START_TOKEN, type=SUB_EVENT,
                  result=SUB_EVENT, secondary=full[i], time=0.0)
            yield f"starter {i}"

    t = 0.0
    bench_h, bench_a = 5, 5
    for i in range(n_rows):
        t += 14.0
        # A backwards-in-time row: the batch path clips delta-t at zero, so must this one.
        if i == 40:
            t -= 30.0

        kind = i % 12
        if kind in (0, 1, 6):                                   # made / missed field goals
            shot_type = "3pt" if kind == 6 else "2pt"
            _play(sim, event="shot", player=sim.home_roster[i % 5], type=shot_type,
                  result="made" if kind != 1 else "missed", secondary="none", time=t)
            label = f"shot {shot_type}"
        elif kind == 2:                                          # free throw (1 point)
            _play(sim, event="shot", player=sim.away_roster[i % 5], type="free throw",
                  result="made", secondary="none", time=t)
            label = "free throw"
        elif kind == 3:                                          # counts toward team fouls
            _play(sim, event="foul", player=sim.away_roster[i % 5], type="personal",
                  result="nan", secondary="none", time=t)
            label = "personal foul"
        elif kind == 4:                                          # NON_TEAM_FOUL_TYPES: skipped
            _play(sim, event="foul", player=sim.home_roster[i % 5],
                  type="technical" if i % 24 == 4 else "offensive",
                  result="nan", secondary="none", time=t)
            label = "non-team foul"
        elif kind == 5:                                          # possession-flipping turnover
            _play(sim, event="turnover", player=sim.home_roster[i % 5], type="nan",
                  result="steal", secondary=sim.away_roster[i % 5], time=t)
            label = "turnover"
        elif kind == 7 and bench_h < len(HOME_FULL):             # home sub: mutates the roster
            outgoing, incoming = sim.home_roster[0], HOME_FULL[bench_h]
            bench_h += 1
            sim.home_roster = sim.home_roster[1:] + [incoming]
            _play(sim, event=SUB_EVENT, player=outgoing, type=SUB_EVENT, result=SUB_EVENT,
                  secondary=incoming, time=t)
            label = "home sub"
        elif kind == 8 and bench_a < len(AWAY_FULL):             # away sub
            outgoing, incoming = sim.away_roster[0], AWAY_FULL[bench_a]
            bench_a += 1
            sim.away_roster = sim.away_roster[1:] + [incoming]
            _play(sim, event=SUB_EVENT, player=outgoing, type=SUB_EVENT, result=SUB_EVENT,
                  secondary=incoming, time=t)
            label = "away sub"
        elif kind == 9 and i == 93:                              # foul-out: full roster shrinks
            gone = sim.home_roster[-1]
            sim.home_full.remove(gone)
            _play(sim, event="foul", player=gone, type="personal", result="nan",
                  secondary="none", time=t)
            label = "disqualification"
        else:
            _play(sim, event="rebound", player=sim.away_roster[i % 5],
                  type="offensive-rebound", result="nan", secondary="none", time=t)
            label = "rebound"
        yield f"row {i} ({label}) t={t}"

    _play(sim, event="end", player="end", type="end", result="end",
          secondary="none", time=t)
    yield "end frame"


def test_cached_matches_uncached_over_a_full_game(tmp_path):
    """The proof: identical arrays at every row of a game that covers every branch.

    Deliberately compares at *every* append rather than sampling -- with SEQ=16 the whole game is
    cheap, and the interesting failures (window slide at SEQ, buffer compaction at CAP) are
    off-by-one bugs that a sampled check can step straight over.
    """
    sim = _sim(tmp_path)
    seen_periods = set()
    for where in _script(sim):
        _assert_identical(sim, where)
        seen_periods.add(int(sim.history[-1]["time"] // 720))

    n = len(sim.history)
    assert n > 4 * SEQ, "script must slide the window and compact the buffers repeatedly"
    assert sim._cache._k < sim._cache.CAP, "write index must stay inside the buffers"
    assert seen_periods >= {0, 1, 2, 3, 4}, "must cross into overtime (per-period foul resets)"


@pytest.mark.parametrize("rows", [SEQ - 1, SEQ, SEQ + 1, 2 * SEQ - 1, 2 * SEQ,
                                  2 * SEQ + 1, 4 * SEQ - 1, 4 * SEQ, 4 * SEQ + 1])
def test_identical_at_window_and_compaction_boundaries(tmp_path, rows):
    """The slide (n == SEQ) and compaction (write index == CAP) boundaries, pinned explicitly."""
    sim = _sim(tmp_path)
    for i, where in enumerate(_script(sim), start=1):
        if i == rows:
            _assert_identical(sim, f"{where} @ n={rows}")
            return
    pytest.fail(f"script produced fewer than {rows} rows")


def test_reset_and_restart_clears_the_cache(tmp_path):
    """A second game on the same simulator must not inherit the first game's rows."""
    sim = _sim(tmp_path)
    for _ in zip(range(40), _script(sim)):
        pass
    sim.reset()
    assert sim._cache.n == 0

    sim.home_full, sim.away_full = list(HOME_FULL), list(AWAY_FULL)
    sim.home_roster, sim.away_roster = HOME_FULL[:5], AWAY_FULL[:5]
    sim.season = "2003"
    for where in zip(range(30), _script(sim)):
        _assert_identical(sim, f"after reset: {where[1]}")


def test_direct_history_append_self_heals(tmp_path):
    """Appending outside _append_row warns and re-encodes rather than serving stale tensors."""
    sim = _sim(tmp_path)
    for _ in zip(range(12), _script(sim)):
        pass

    smuggled = sim._make_row(event="shot", player=sim.home_roster[0], type="3pt",
                             result="made", secondary_player="none", time=500.0)
    sim.history.append(smuggled)                      # bypasses the cache on purpose
    assert sim._cache.n != len(sim.history)

    with pytest.warns(RuntimeWarning, match="out of sync"):
        sim.build_model_inputs()
    _assert_identical(sim, "after self-heal")
    assert sim._cache.n == len(sim.history)


def test_base_slab_is_built_once_per_event(tmp_path):
    """The whole point: ~5 head calls per event must cost ONE input build, not five."""
    sim = _sim(tmp_path)
    for _ in zip(range(20), _script(sim)):
        pass

    before = sim._cache._builds
    for _ in range(5):
        sim.build_model_inputs()
    assert sim._cache._builds == before + 1

    _play(sim, event="shot", player=sim.home_roster[0], type="2pt", result="missed",
          secondary="none", time=900.0)
    sim.build_model_inputs()
    assert sim._cache._builds == before + 2


def test_returned_arrays_are_copies_not_buffer_views(tmp_path):
    """A later append must never mutate a dict already handed to a caller (or to TF)."""
    sim = _sim(tmp_path)
    for _ in zip(range(5), _script(sim)):
        pass

    held = {k: v.copy() for k, v in sim.build_model_inputs().items()}
    live = sim.build_model_inputs()
    _play(sim, event="shot", player=sim.home_roster[0], type="2pt", result="made",
          secondary="none", time=777.0)
    for k, v in held.items():
        np.testing.assert_array_equal(live[k], v, err_msg=f"{k} was mutated by a later append")


def test_kill_switch_selects_the_oracle(tmp_path, monkeypatch):
    """CVIQ_INPUT_CACHE=0 bypasses the cache entirely and still returns the same arrays."""
    import simulation.game_simulator as gs

    sim = _sim(tmp_path)
    for _ in zip(range(25), _script(sim)):
        pass
    cached = {k: v.copy() for k, v in sim.build_model_inputs().items()}

    monkeypatch.setattr(gs, "_INPUT_CACHE_ENABLED", False)
    builds = sim._cache._builds
    plain = sim.build_model_inputs()
    assert sim._cache._builds == builds, "kill switch must not touch the cache"
    for k, v in cached.items():
        np.testing.assert_array_equal(plain[k], v, err_msg=k)


def test_empty_history_still_raises(tmp_path):
    sim = _sim(tmp_path)
    with pytest.raises(RuntimeError, match="No history to encode"):
        sim.build_model_inputs()


# --------------------------------------------------------------------------- #
# --- _avail_mask content cache                                            --- #
# --------------------------------------------------------------------------- #

def _avail_oracle(sim) -> np.ndarray:
    """The pre-cache _avail_mask body, kept here as the thing the cache must reproduce."""
    enc = sim.encoder
    V = enc.player_vocab.next_token
    players = {*sim.home_full, *sim.away_full, *sim.home_roster, *sim.away_roster}
    if not players:
        return np.ones((1, V), dtype=np.float32)
    mask = np.zeros((1, V), dtype=np.float32)
    for p in players:
        i = enc.encode_player(p)
        if 0 <= i < V:
            mask[0, i] = 1.0
    mask[0, enc.encode_player("PAD")] = 0.0
    return mask


def test_avail_mask_is_reused_until_a_roster_changes(tmp_path):
    sim = _sim(tmp_path)
    first = sim._avail_mask()
    assert sim._avail_mask() is first, "unchanged rosters must hit the cache"

    sim.home_roster = sim.home_roster[1:] + [HOME_FULL[5]]      # a substitution
    second = sim._avail_mask()
    assert second is not first
    np.testing.assert_array_equal(second, _avail_oracle(sim))


def test_avail_mask_tracks_a_disqualification(tmp_path):
    """_disqualify shrinks *_full, which must drop that player out of the mask."""
    sim = _sim(tmp_path)
    gone = sim.home_roster[0]
    sim._avail_mask()                                            # warm the cache
    sim.home_full.remove(gone)
    sim.home_roster = sim.home_roster[1:] + [HOME_FULL[5]]

    mask = sim._avail_mask()
    np.testing.assert_array_equal(mask, _avail_oracle(sim))
    assert mask[0, sim.encoder.encode_player(gone)] == 0.0


def test_avail_mask_empty_rosters_are_all_ones(tmp_path):
    sim = _sim(tmp_path)
    sim.home_full = sim.away_full = sim.home_roster = sim.away_roster = []
    np.testing.assert_array_equal(sim._avail_mask(), _avail_oracle(sim))
