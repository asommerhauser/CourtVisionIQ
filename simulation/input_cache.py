"""
input_cache.py — encode a rollout's history once per event instead of once per model call.

``GameSimulator.build_model_inputs`` shapes the growing ``history`` into the model's fixed
``(1, SEQ, ...)`` tensors. Written the obvious way it rebuilds **everything** on every call: all 23
arrays, ~10,800 vocab lookups, and two full-history scans (delta-t and game state). The rollout
calls it ~5 times per event (once per head, via ``_conditioned_inputs``) over ~500-900 events,
which makes a single game quadratic in its own length and leaves the GPU idle behind Python (see
``simulation/profile_rollout.py``).

Nothing about that work is actually per-call. History rows are **immutable once appended** --
``GameSimulator._make_row`` snapshots both rosters with ``list(...)``, so later roster mutation
never reaches an appended row -- and every column is a pure function of its own row plus running
scalar state (previous time for delta-t; score / period / team fouls for game state). So each row
can be encoded exactly once, when it is appended, into append-only buffers that this class owns.

Layout: every column is allocated at ``CAP = 2 * SEQ`` rows and **pre-filled with that column's pad
value**, rows are written at a monotonically increasing index, and the model window is always the
contiguous slice ``buf[max(0, k - SEQ) : ...]``. That is correct in both regimes with no branch --
before ``SEQ`` rows it is real rows followed by pre-filled padding (the right-padded layout
``_build_split`` produces), after ``SEQ`` it is the trailing window. When the write index reaches
``CAP`` the tail is memmoved back to the front (~6 times per game), so no step is ever O(SEQ).

``GameSimulator._build_model_inputs_uncached`` remains in the tree verbatim as the oracle this
class is tested against, the target of the ``CVIQ_INPUT_CACHE=0`` kill switch, and the readable
spec for what these buffers must contain.
"""
from __future__ import annotations

import numpy as np

from config import ROSTER_SIZE
from models.event_time_model import CATEGORICAL_FIELDS, EventTimeModel
from models.game_state_features import (
    GAME_STATE_KEYS, GameStateScan, normalize_game_state_row,
)
from models.season_features import DEFAULT_REST_DAYS, REST_CLIP_DAYS, TEAM_SCALAR_COLS

# Roster snapshots repeat for long stretches (a roster changes ~60-80 times in a ~900-row game), so
# memoizing the encode + rest standardization per distinct five removes ~10 vocab lookups and a
# clip/divide on >90% of rows. Cached arrays are only ever copied *into* a buffer, never handed out.
_ROSTER_CACHE_MAX = 64

_REST_COL = {"home_roster": "rest_home", "away_roster": "rest_away"}


class HistoryEncoder:
    """Append-only encoded columns for one :class:`~simulation.game_simulator.GameSimulator`.

    Mirrors ``GameSimulator._build_model_inputs_uncached`` exactly, one row at a time.
    ``tests/test_input_cache.py`` asserts the two agree array-for-array, bit-for-bit, across a
    scripted game that exercises the window slide, buffer compaction, period rollovers and
    mid-game roster churn.
    """

    def __init__(self, sim) -> None:
        # Deferred import: game_simulator imports this module, so binding at module scope would be
        # circular. By the time an encoder is constructed, game_simulator is fully loaded.
        from simulation.game_simulator import _norm_cat

        self._norm_cat = _norm_cat
        self.sim = sim
        self.SEQ = int(sim.sequence_length)
        self.CAP = 2 * self.SEQ
        self._buffers: dict[str, np.ndarray] = {}
        self._builds = 0            # slabs materialized; asserted on by the tests
        self.reset()

    # ------------------------------------------------------------------ state
    @property
    def n(self) -> int:
        """Rows fed so far -- compared against ``len(sim.history)`` as the staleness check."""
        return self._n

    def reset(self) -> None:
        """Drop every row and re-pad the buffers (a new game on the same simulator)."""
        SEQ, CAP = self.SEQ, self.CAP
        sim = self.sim
        pad_player = sim.encoder.encode_player("PAD")

        self._pads: dict[str, object] = {}
        buf: dict[str, np.ndarray] = {}
        for field in CATEGORICAL_FIELDS:
            self._pads[field] = sim._pad_id(field)
            buf[field] = np.full((CAP,), self._pads[field], dtype=np.int32)
        for name in ("home_roster", "away_roster"):
            self._pads[name] = pad_player
            buf[name] = np.full((CAP, ROSTER_SIZE), pad_player, dtype=np.int32)
        # Every float column pads with 0.0 -- matching the np.zeros(...) the batch builder starts
        # from for time, rest, team scalars, game state and the attention mask alike.
        for name in ("rest_home", "rest_away"):
            self._pads[name] = 0.0
            buf[name] = np.zeros((CAP, ROSTER_SIZE), dtype=np.float32)
        for name in ("time_abs", "delta_time", *TEAM_SCALAR_COLS, *GAME_STATE_KEYS):
            self._pads[name] = 0.0
            buf[name] = np.zeros((CAP, 1), dtype=np.float32)
        self._pads["pad_mask"] = 0.0
        buf["pad_mask"] = np.zeros((CAP,), dtype=np.float32)
        self._buffers = buf

        self._n = 0                 # rows appended (== len(history))
        self._k = 0                 # next write index into the buffers
        self._prev_time = 0.0
        self._scan = GameStateScan()
        self._roster_cache: dict[tuple, tuple] = {}
        self._base: dict[str, np.ndarray] | None = None

    def on_context_change(self) -> None:
        """Season context moved, so already-encoded rest / team scalars are stale -- start over.

        In practice ``_set_season_context`` always runs before the first append, making this free;
        the hook exists so a future mid-game context change cannot silently poison those columns.
        """
        self.rebuild_from(list(self.sim.history))

    def rebuild_from(self, history) -> None:
        """Re-encode from scratch. The self-heal path when the cache has fallen out of sync."""
        self.reset()
        for row in history:
            self.append(row)

    # ------------------------------------------------------------------ append
    def append(self, row: dict) -> None:
        """Encode one just-appended history row into every column at the next write index."""
        sim = self.sim
        enc = sim.encoder
        ns = sim.norm_stats
        buf = self._buffers
        k = self._k
        first = self._n == 0

        max_time = float(ns.get("max_time", 1.0)) or 1.0
        delta_mean = float(ns.get("delta_mean", 0.0))
        delta_std = float(ns.get("delta_std", 1.0)) or 1.0
        rest_mean = float(ns.get("rest_mean", DEFAULT_REST_DAYS))
        rest_std = float(ns.get("rest_std", 1.0)) or 1.0

        norm_cat = self._norm_cat
        for field in CATEGORICAL_FIELDS:
            buf[field][k] = getattr(enc, f"encode_{field}")(norm_cat(row[field]))

        for name, col, rest_map in (("home_roster", "roster_home", sim.home_rest),
                                    ("away_roster", "roster_away", sim.away_rest)):
            names = tuple(row[col])
            key = (name, names)
            hit = self._roster_cache.get(key)
            if hit is None:
                ids = np.asarray(enc.encode_roster(row[col]), dtype=np.int32)
                raw = np.zeros((ROSTER_SIZE,), dtype=np.float32)
                for j, p in enumerate(names[:ROSTER_SIZE]):
                    raw[j] = rest_map.get(p, DEFAULT_REST_DAYS)
                hit = (ids, (np.clip(raw, 0.0, REST_CLIP_DAYS) - rest_mean) / rest_std)
                if len(self._roster_cache) < _ROSTER_CACHE_MAX:
                    self._roster_cache[key] = hit
            buf[name][k] = hit[0]
            buf[_REST_COL[name]][k] = hit[1]

        # delta-t exactly as the batch path derives it: a per-game diff (first row = 0) clipped
        # backwards to zero in float64, then standardized. np.clip, not max(), so -0.0 matches too.
        t = float(row["time"])
        delta = 0.0 if first else float(np.clip(t - self._prev_time, 0, None))
        buf["time_abs"][k, 0] = t / max_time
        buf["delta_time"][k, 0] = (delta - delta_mean) / delta_std
        self._prev_time = t

        def _std_days(days):
            return (min(float(days), REST_CLIP_DAYS) - rest_mean) / rest_std

        team_values = {
            "home_games_played": sim.home_games_played,
            "away_games_played": sim.away_games_played,
            "home_days_rest": _std_days(sim.home_days_rest),
            "away_days_rest": _std_days(sim.away_days_rest),
        }
        for name in TEAM_SCALAR_COLS:
            buf[name][k, 0] = team_values[name]

        for name, value in zip(GAME_STATE_KEYS, normalize_game_state_row(self._scan.step(row))):
            buf[name][k, 0] = value

        buf["pad_mask"][k] = 1.0

        self._k = k + 1
        self._n += 1
        self._base = None
        if self._k == self.CAP:
            self._compact()

    def _compact(self) -> None:
        """Memmove the trailing window back to the front so appends stay O(1) forever."""
        SEQ, CAP = self.SEQ, self.CAP
        for name, b in self._buffers.items():
            b[0:SEQ] = b[CAP - SEQ:CAP]
            b[SEQ:CAP] = self._pads[name]
        self._k = SEQ

    # ------------------------------------------------------------------ read
    def base_inputs(self) -> dict[str, np.ndarray]:
        """The model's batch-1 input dict over the current window, memoized per appended row.

        Returns **copies**, not views into the buffers: compaction moves data out from under a
        view, a pre-``SEQ`` append writes inside a previously returned window, and
        ``tf.convert_to_tensor`` can alias a host array zero-copy. One ~91 KiB memcpy per event is
        noise against the five full rebuilds it replaces.
        """
        if self._base is not None:
            return self._base
        a = max(0, self._k - self.SEQ)
        b = a + self.SEQ
        buf = self._buffers
        self._builds += 1
        self._base = {k: np.array(buf[k][a:b])[None, ...] for k in EventTimeModel.INPUT_KEYS}
        return self._base


__all__ = ["HistoryEncoder"]
