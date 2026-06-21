"""
Box-score tool — turn a single game's play-by-play into a readable box score.

This is the bridge between *event sequences* and *human-readable basketball data*. It
consumes one game's worth of event rows and aggregates per-player and per-team stats into
a ``BoxScore`` (points, FG/3PT/FT, rebounds, assists, steals, blocks, turnovers, fouls,
minutes). It is needed for two things:

  1. **Validation** — render the real box score of a held-out game so model output can be
     compared against reality.
  2. **Prediction readout** — convert a generated game into the same box score, so model
     predictions become real, interpretable data.

It deliberately accepts the *same* row shape produced by both the cleaned data and the
:class:`~simulation.game_simulator.GameSimulator` history, so one decoder serves both:
``event, player, type, result, secondary_player, time, roster_home, roster_away``.

Decoding follows the cleaned-data semantics in ``data_cleaner.py`` (not a 1:1 copy of the
legacy notebook). The two semantics that bite:

  * **Steals** are emitted as *two* turnover rows — one for the stealer
    (``type="steal", result="steal"``) and one for the player who lost the ball
    (``result="cop"``). Only the latter is a turnover; the former is a steal.
  * **Blocked shots** carry ``result="blocked"`` on the shooter's shot row (a missed
    attempt) plus a separate ``block`` event for the blocker.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

import pandas as pd

# Foul types that do NOT count as a personal foul on the box score (technicals are
# tracked separately in real box scores; everything else — shooting, offensive, loose
# ball, flagrant, etc. — is a personal foul).
NON_PERSONAL_FOUL_TYPES = {"technical"}

# Non-play frame events that never contribute stats.
SKIP_EVENTS = {"start", "end", "none", "PAD", "UNK"}


@dataclass
class PlayerLine:
    """One player's aggregated stat line for a game."""
    player: str
    seconds: float = 0.0
    fgm: int = 0
    fga: int = 0
    tpm: int = 0
    tpa: int = 0
    ftm: int = 0
    fta: int = 0
    oreb: int = 0
    dreb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tov: int = 0
    pf: int = 0
    pts: int = 0

    @property
    def reb(self) -> int:
        return self.oreb + self.dreb

    @property
    def minutes(self) -> float:
        return self.seconds / 60.0

    def as_row(self) -> dict:
        """ESPN-style display row (made-attempted shooting splits)."""
        return {
            "Player": self.player,
            "MIN": round(self.minutes, 1),
            "FG": f"{self.fgm}-{self.fga}",
            "3PT": f"{self.tpm}-{self.tpa}",
            "FT": f"{self.ftm}-{self.fta}",
            "REB": self.reb,
            "AST": self.ast,
            "STL": self.stl,
            "BLK": self.blk,
            "TO": self.tov,
            "PF": self.pf,
            "PTS": self.pts,
        }


@dataclass
class BoxScore:
    """Both teams' stat lines plus the final score."""
    home: list[PlayerLine] = field(default_factory=list)
    away: list[PlayerLine] = field(default_factory=list)
    home_score: int = 0
    away_score: int = 0

    def to_frame(self, side: str) -> pd.DataFrame:
        """DataFrame of one side's stat lines, sorted by points descending."""
        lines = self.home if side == "home" else self.away
        rows = [pl.as_row() for pl in sorted(lines, key=lambda p: p.pts, reverse=True)]
        return pd.DataFrame(rows, columns=list(_DISPLAY_COLUMNS))

    def render(self) -> str:
        """Readable two-team box score with final score."""
        out = [
            f"HOME ({self.home_score})",
            self.to_frame("home").to_string(index=False),
            "",
            f"AWAY ({self.away_score})",
            self.to_frame("away").to_string(index=False),
            "",
            f"Final: HOME {self.home_score} - {self.away_score} AWAY",
        ]
        return "\n".join(out)

    def __str__(self) -> str:  # pragma: no cover - thin wrapper
        return self.render()


_DISPLAY_COLUMNS = ("Player", "MIN", "FG", "3PT", "FT", "REB",
                    "AST", "STL", "BLK", "TO", "PF", "PTS")


def generate_box_score(events) -> BoxScore:
    """Aggregate a game's event sequence into a :class:`BoxScore`.

    ``events`` is an iterable of dict rows (or a ``pandas.DataFrame``) carrying at least
    ``event, player, type, result, time`` and the per-row ``roster_home`` / ``roster_away``
    snapshots. Rosters may be real lists or string literals (both are accepted). Events are
    processed in the given order; for minutes, the rows are treated as time-ordered.
    """
    rows = _as_rows(events)

    lines: dict[str, PlayerLine] = {}
    home_players: set[str] = set()
    away_players: set[str] = set()
    home_score = away_score = 0

    def line(name: str) -> PlayerLine:
        if name not in lines:
            lines[name] = PlayerLine(player=name)
        return lines[name]

    prev_time = None
    prev_roster = ([], [])  # (home, away) on-court at the start of the current interval

    for row in rows:
        home_roster = _roster(row.get("roster_home"))
        away_roster = _roster(row.get("roster_away"))
        home_players.update(home_roster)
        away_players.update(away_roster)

        # --- Minutes: credit the lineup that was on court over the elapsed interval. ---
        time = _as_float(row.get("time"))
        if prev_time is not None and time is not None and time > prev_time:
            elapsed = time - prev_time
            for name in (*prev_roster[0], *prev_roster[1]):
                line(name).seconds += elapsed
        if time is not None:
            prev_time = time
            prev_roster = (home_roster, away_roster)

        # --- Stat decoding. ---
        event = _norm(row.get("event"))
        if event in SKIP_EVENTS:
            continue
        player = _norm(row.get("player"))
        if not player or player in ("null", "none", "PAD", "UNK"):
            continue
        etype = _norm(row.get("type"))
        result = _norm(row.get("result"))
        team = "home" if player in home_players else "away" if player in away_players else None
        pl = line(player)

        if event == "shot":
            made = result == "made"
            if etype == "3pt":
                pl.tpa += 1
                pl.fga += 1
                if made:
                    pl.tpm += 1
                    pl.fgm += 1
                    pl.pts += 3
            elif etype == "free throw":
                pl.fta += 1
                if made:
                    pl.ftm += 1
                    pl.pts += 1
            else:  # 2pt (or any other field-goal type)
                pl.fga += 1
                if made:
                    pl.fgm += 1
                    pl.pts += 2
            if made:
                pts = 3 if etype == "3pt" else 1 if etype == "free throw" else 2
                if team == "home":
                    home_score += pts
                elif team == "away":
                    away_score += pts

        elif event == "assist":
            pl.ast += 1
        elif event == "rebound":
            if etype == "offensive":
                pl.oreb += 1
            elif etype == "defensive":
                pl.dreb += 1
            else:  # unspecified rebound — count as a rebound without an o/d split
                pl.dreb += 1
        elif event == "block":
            pl.blk += 1
        elif event == "turnover":
            # Steal pair: the stealer's row (result="steal") is a steal; the other row
            # (result="cop") is the turnover. Non-steal turnovers also carry "cop".
            if result == "steal":
                pl.stl += 1
            else:
                pl.tov += 1
        elif event == "foul":
            if etype not in NON_PERSONAL_FOUL_TYPES:
                pl.pf += 1
        # substitution: roster mutation only (already reflected in row snapshots) — no stat.

    home = [lines[p] for p in sorted(home_players) if p in lines]
    away = [lines[p] for p in sorted(away_players) if p in lines]
    return BoxScore(home=home, away=away, home_score=home_score, away_score=away_score)


def box_score_for_game(game_id: int, data_dir="./data") -> BoxScore:
    """Build the box score for one real cleaned game (by its globally-unique game_id)."""
    from data_loading import load_all_cleaned

    df = load_all_cleaned(data_dir, parse_rosters=True)
    game = df[df["game_id"] == int(game_id)].sort_values("time")
    if game.empty:
        raise ValueError(f"game_id {game_id} not found in cleaned data under {data_dir!r}")
    return generate_box_score(game)


# --------------------------------------------------------------------------- helpers


def _as_rows(events):
    """Normalize the input into a list of dict rows."""
    if isinstance(events, pd.DataFrame):
        return events.to_dict("records")
    return list(events)


def _roster(value) -> list[str]:
    """Coerce a roster cell (list or "['A', 'B']" string) into a list of names."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def _norm(value):
    """String-normalize a categorical cell (NaN -> empty string)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _as_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


__all__ = ["PlayerLine", "BoxScore", "generate_box_score", "box_score_for_game"]
