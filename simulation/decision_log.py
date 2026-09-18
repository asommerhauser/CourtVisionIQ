"""
What each head actually sampled, per queried position, per sim (3.2 W10).

This is the input the weighted replay pass (W11) trains on: the model's own choices become the labels,
and the advantage of the sim they came from becomes the sample weight.

**A thin index, not a tensor dump.** ``GameSimulator.build_model_inputs`` serves a dict memoised per row
and shared across the roughly five head calls at one position, and the prior columns alone run to
megabytes per position at ``SEQ = 600``. Storing context per queried position across thousands of
game-sims is not feasible, and retaining the memoised dict would alias it -- every recorded decision at
one position would point at the same object, which is then mutated as the game advances.

So a decision records only *where and what*: ``(game_id, sim_index, position, head, output, token)``.
The context is **re-derived** by replaying the sim's own play-by-play through the ordinary preprocess.
``simulation/stage_eval._PbpSink`` already writes each sim's rows in the cleaned-row schema, and
``reporting/state_probes`` already runs ``LineupScan`` and ``GameStateScan`` over those files, which is
the standing proof that the round trip works.

**Per worker, never shared.** Up to ``ROLLOUT_BATCH_SIZE`` workers drive their own ``GameController`` on
their own thread, so one log per ``_WorkerSim`` and a flush when its game finishes. A shared list would
need a lock and would lose the sim identity, which is the one field the advantage calculation cannot do
without.

**Which positions count.** ``apply_query_mask`` is called by two of the twelve heads, so its ~68.5% is
the event/time head's kept share and not a global figure -- the other ten heads' queried positions are
event-token-gated and much sparser. A decision is therefore logged where a head was actually *asked*,
which is exactly where it sampled, rather than derived from a mask.
"""
from __future__ import annotations

import json
from pathlib import Path

#: One decision. Deliberately flat and small: at ~600 events a game and a handful of heads per event,
#: a 100-sim game is on the order of a million of these.
FIELDS = ("game_id", "sim_index", "position", "head", "output", "token")

LOG_FILENAME = "decisions.jsonl"


class DecisionLog:
    """Append-only record of one sim's sampled decisions."""

    __slots__ = ("game_id", "sim_index", "rows", "enabled")

    def __init__(self, game_id=None, sim_index: int | None = None, *, enabled: bool = True):
        self.game_id = game_id
        self.sim_index = sim_index
        self.rows: list[tuple] = []
        self.enabled = bool(enabled)

    def record(self, head: str, output: str, position: int, token) -> None:
        if not self.enabled:
            return
        self.rows.append((self.game_id, self.sim_index, int(position), str(head), str(output),
                          str(token)))

    def extend(self, other: "DecisionLog") -> None:
        self.rows.extend(other.rows)

    def as_dicts(self) -> list[dict]:
        return [dict(zip(FIELDS, row)) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __bool__(self) -> bool:
        # A log with no rows is still a log -- ``if log:`` must not silently disable recording.
        return True


def write_log(rows, out_dir, *, filename: str = LOG_FILENAME) -> Path:
    """Append decisions as JSON lines beside a run's per-game folder.

    **Beside** ``playbyplay/``, never inside it: ``harvest.py`` prunes that directory on a live run, and
    a log deleted by the harvester would take the replay pass's labels with it.
    """
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            entry = row if isinstance(row, dict) else dict(zip(FIELDS, row))
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
    return path


def read_log(path) -> list[dict]:
    """Every decision in a log file, in the order it was written."""
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def group_by_sim(rows) -> dict:
    """``{(game_id, sim_index): [decision, ...]}`` -- the grain the advantage is computed at."""
    out: dict = {}
    for row in rows:
        entry = row if isinstance(row, dict) else dict(zip(FIELDS, row))
        out.setdefault((entry["game_id"], entry["sim_index"]), []).append(entry)
    return out


__all__ = ["DecisionLog", "FIELDS", "LOG_FILENAME", "group_by_sim", "read_log", "write_log"]
