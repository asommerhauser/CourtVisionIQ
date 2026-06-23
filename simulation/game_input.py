"""
Game-input extractor — scan a complete game and pull out its matchup "spec".

To test the model against a real game we need that game's *input*: the givens a generator
is conditioned on, independent of how the game actually unfolded. Per the project design
the spec is the **whole roster** of each team (every player who appeared — no starter/bench
distinction), the season, and a playoff flag:

    GameInput(home_roster, away_roster, season, playoff)

The whole roster is not a stored column; it is the union of the per-row on-court fives
(``roster_home`` / ``roster_away``) across the game — the same union ``data_cleaner.py``
already computes internally as ``home_players`` / ``away_players`` but never writes out.
``playoff`` is carried as spec metadata (the current model conditions only on ``season`` —
see docs/technical_specs.md); cleaned data stores it as 1=regular / 2=playoff, and we map
that to the 0/1 flag.

This deliberately stops at the spec. Turning a whole roster into a seeded
``GameSimulator.start_game`` (which wants an on-court five) is a separate, later step.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from config import HOLDOUT_MANIFEST_NAME
from data_loading import load_all_cleaned
from models.season_features import DEFAULT_REST_DAYS, _to_list
from simulation.box_score import _as_rows, _roster

# Roster-cell tokens that are not real players.
_NON_PLAYERS = {"", "null", "none", "PAD", "UNK", "start", "end", "nan"}
# Neutral season-progress default for hand-built specs (mid-season); real games override it.
DEFAULT_GAMES_PLAYED = 0.5


@dataclass
class GameInput:
    """The matchup spec extracted from one game: the test input for regeneration."""
    home_roster: list[str]
    away_roster: list[str]
    season: int
    playoff: int  # 1 = playoff, 0 = regular
    # --- Season context (pre-game givens; see season_context.py) ---
    home_games_played: float = DEFAULT_GAMES_PLAYED
    away_games_played: float = DEFAULT_GAMES_PLAYED
    home_days_rest: float = DEFAULT_REST_DAYS
    away_days_rest: float = DEFAULT_REST_DAYS
    home_rest: dict = field(default_factory=dict)  # player -> days since last game
    away_rest: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def season_context(self) -> dict:
        """The season-context payload the simulator consumes (see GameSimulator._set_season_context)."""
        return {
            "home_games_played": self.home_games_played,
            "away_games_played": self.away_games_played,
            "home_days_rest": self.home_days_rest,
            "away_days_rest": self.away_days_rest,
            "home_rest": dict(self.home_rest),
            "away_rest": dict(self.away_rest),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GameInput":
        return cls(
            home_roster=list(d["home_roster"]),
            away_roster=list(d["away_roster"]),
            season=int(d["season"]),
            playoff=int(d["playoff"]),
            home_games_played=float(d.get("home_games_played", DEFAULT_GAMES_PLAYED)),
            away_games_played=float(d.get("away_games_played", DEFAULT_GAMES_PLAYED)),
            home_days_rest=float(d.get("home_days_rest", DEFAULT_REST_DAYS)),
            away_days_rest=float(d.get("away_days_rest", DEFAULT_REST_DAYS)),
            home_rest=dict(d.get("home_rest", {}) or {}),
            away_rest=dict(d.get("away_rest", {}) or {}),
        )


def extract_game_input(game_rows) -> GameInput:
    """Build the :class:`GameInput` for a single game's cleaned rows.

    ``game_rows`` is one game's rows as a ``pandas.DataFrame`` or a list of dicts. Each
    roster is the sorted union of that side's per-row lineups (real players only); season
    and playoff are read from the (constant-per-game) columns, with playoff mapped to the
    0/1 flag.
    """
    rows = _as_rows(game_rows)
    if not rows:
        raise ValueError("extract_game_input: no rows provided.")

    home: set[str] = set()
    away: set[str] = set()
    # Per-player rest maps (built from the roster-parallel rest columns; bench players are
    # captured too, since they appear in the rosters once subbed in).
    home_rest: dict[str, float] = {}
    away_rest: dict[str, float] = {}
    for row in rows:
        home.update(_roster(row.get("roster_home")))
        away.update(_roster(row.get("roster_away")))
        _accumulate_rest(home_rest, row.get("roster_home"), row.get("rest_home"))
        _accumulate_rest(away_rest, row.get("roster_away"), row.get("rest_away"))

    home_roster = sorted(p for p in home if str(p).strip() not in _NON_PLAYERS)
    away_roster = sorted(p for p in away if str(p).strip() not in _NON_PLAYERS)

    season = _first_int(rows, "season")
    playoff = 1 if _first_int(rows, "playoff") == 2 else 0
    return GameInput(
        home_roster=home_roster, away_roster=away_roster, season=season, playoff=playoff,
        home_games_played=_first_float(rows, "home_games_played", DEFAULT_GAMES_PLAYED),
        away_games_played=_first_float(rows, "away_games_played", DEFAULT_GAMES_PLAYED),
        home_days_rest=_first_float(rows, "home_days_rest", DEFAULT_REST_DAYS),
        away_days_rest=_first_float(rows, "away_days_rest", DEFAULT_REST_DAYS),
        home_rest=home_rest, away_rest=away_rest,
    )


def _accumulate_rest(dest: dict, roster_cell, rest_cell) -> None:
    """Zip a roster cell with its parallel rest cell into ``dest`` (player -> days)."""
    names = _roster(roster_cell)
    rest = rest_cell if isinstance(rest_cell, list) else _to_list(rest_cell)
    for name, days in zip(names, rest):
        key = str(name).strip()
        if key in _NON_PLAYERS:
            continue
        dest.setdefault(key, float(days))  # constant per game; first occurrence wins


def game_input_for_game(game_id: int, data_dir="./data") -> GameInput:
    """Extract the game input for one real cleaned game (by globally-unique game_id)."""
    df = load_all_cleaned(data_dir, parse_rosters=True)
    game = df[df["game_id"] == int(game_id)]
    if game.empty:
        raise ValueError(f"game_id {game_id} not found in cleaned data under {data_dir!r}")
    return extract_game_input(game)


def holdout_game_inputs(data_dir="./data",
                        processed_dir="./data/processed") -> dict[int, GameInput]:
    """Extract a :class:`GameInput` for every game in the holdout manifest.

    Reads ``holdout_games.json`` (written by ``preprocess``), loads the cleaned data once,
    and returns ``{game_id: GameInput}`` for exactly those held-out games.
    """
    manifest = Path(processed_dir) / HOLDOUT_MANIFEST_NAME
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found; run preprocess() to write the holdout manifest first."
        )
    holdout_ids = [int(g) for g in json.loads(manifest.read_text(encoding="utf-8"))]

    df = load_all_cleaned(data_dir, parse_rosters=True)
    wanted = df[df["game_id"].isin(holdout_ids)]
    out: dict[int, GameInput] = {}
    for gid, game in wanted.groupby("game_id"):
        out[int(gid)] = extract_game_input(game)
    return out


def write_holdout_inputs(data_dir="./data",
                         processed_dir="./data/processed") -> Path:
    """Persist the holdout game inputs to ``data/processed/holdout_inputs.json``.

    Returns the output path. The JSON is ``{game_id: {home_roster, away_roster, season,
    playoff}}`` — ready-made test fixtures that round-trip via ``GameInput.from_dict``.
    """
    inputs = holdout_game_inputs(data_dir=data_dir, processed_dir=processed_dir)
    out_path = Path(processed_dir) / "holdout_inputs.json"
    payload = {str(gid): gi.to_dict() for gid, gi in sorted(inputs.items())}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload)} holdout game inputs -> {out_path}")
    return out_path


def _first_int(rows, key: str) -> int:
    """First parseable int value for ``key`` across rows (columns are constant per game)."""
    for row in rows:
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    raise ValueError(f"No valid integer value found for column {key!r}.")


def _first_float(rows, key: str, default: float) -> float:
    """First parseable float value for ``key`` across rows, or ``default`` if none/absent.

    Unlike ``_first_int`` this never raises — the season-context columns are absent from
    pre-enrichment data, so a missing column simply falls back to the neutral default.
    """
    for row in rows:
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


__all__ = [
    "GameInput",
    "extract_game_input",
    "game_input_for_game",
    "holdout_game_inputs",
    "write_holdout_inputs",
]
