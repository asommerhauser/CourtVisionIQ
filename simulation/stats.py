"""
stats.py — team aggregation + advanced basketball metrics from a BoxScore.

Shared, model-free math used by the evaluation harness (``simulation/evaluation.py``). It turns
the per-player :class:`~simulation.box_score.PlayerLine` rows into team totals and the standard
"four factors + pace" advanced stats, so predicted box scores and real ones are summarized with
one implementation.

Possessions follow the same estimate used elsewhere in the project
(``FGA - OREB + TOV + 0.44*FTA``); pace normalizes possessions to a 48-minute game using the
team's total minutes on the floor (so overtime games are compared on the same footing).
"""
from __future__ import annotations

import math

from simulation.box_score import PlayerLine

# Numeric counting fields we average / std across sims and sum into team totals. This is the
# numeric subset of PlayerLine (everything except the player name and derived +/-, which is not
# meaningful as a team sum). ``seconds`` is kept so we can derive minutes and pace.
BOX_STATS = (
    "seconds", "fgm", "fga", "tpm", "tpa", "ftm", "fta",
    "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "pts",
)

# Full regulation game length and on-court slots, for pace normalization.
_GAME_MINUTES = 48.0
_ON_COURT = 5.0


def player_stats(line: PlayerLine) -> dict[str, float]:
    """The numeric stat dict for one player line (BOX_STATS only)."""
    return {f: float(getattr(line, f)) for f in BOX_STATS}


def team_totals(lines: list[PlayerLine]) -> dict[str, float]:
    """Sum BOX_STATS across a side's player lines into one team row."""
    return {f: float(sum(getattr(pl, f) for pl in lines)) for f in BOX_STATS}


def possessions(t: dict[str, float]) -> float:
    """Standard possessions estimate: ``FGA - OREB + TOV + 0.44*FTA``."""
    return t["fga"] - t["oreb"] + t["tov"] + 0.44 * t["fta"]


def advanced_stats(team: dict[str, float], opp: dict[str, float]) -> dict[str, float]:
    """The four factors + pace for one team, given its and the opponent's team totals.

    Returns: ``pace`` (possessions per 48 min), ``efg`` (effective FG%), ``tov_pct``
    (turnovers per possession), ``oreb_pct`` / ``dreb_pct`` (share of available boards),
    ``ft_rate`` (FTA per FGA), plus raw ``pts``, ``poss`` and ``fg_pct`` for convenience.
    """
    poss = possessions(team)
    fga = team["fga"]
    # team minutes summed across players; a 5-on-5 regulation game ≈ 240 player-minutes.
    game_minutes = (team["seconds"] / 60.0) / _ON_COURT
    oreb_chances = team["oreb"] + opp["dreb"]
    dreb_chances = team["dreb"] + opp["oreb"]
    return {
        "pts": team["pts"],
        "poss": poss,
        "pace": (poss / game_minutes * _GAME_MINUTES) if game_minutes else 0.0,
        "efg": ((team["fgm"] + 0.5 * team["tpm"]) / fga) if fga else 0.0,
        "fg_pct": (team["fgm"] / fga) if fga else 0.0,
        "tov_pct": (team["tov"] / poss) if poss else 0.0,
        "oreb_pct": (team["oreb"] / oreb_chances) if oreb_chances else 0.0,
        "dreb_pct": (team["dreb"] / dreb_chances) if dreb_chances else 0.0,
        "ft_rate": (team["fta"] / fga) if fga else 0.0,
    }


def score_win_prob(mean_margin: float, margin_std: float) -> float:
    """Home win probability from the sims' margin distribution (normal approximation).

    Uses the *average predicted score* (mean margin) and the cross-sim spread, instead of the raw
    share of sims the home team won. A confident average (e.g. +6 every sim) isn't washed back toward
    0.5 by a few coin-flip games, so the implied probability — and the winner pick (sign of the mean
    margin) — track the score the box-score model already predicts well. With zero spread it collapses
    to a hard 1/0 by sign (0.5 on an exact tie).
    """
    if margin_std <= 1e-9:
        return 1.0 if mean_margin > 0 else 0.0 if mean_margin < 0 else 0.5
    return 0.5 * (1.0 + math.erf(mean_margin / (margin_std * math.sqrt(2.0))))


# Display order + friendly labels for the advanced block (four factors + pace).
ADVANCED_LABELS = {
    "pts": "PTS",
    "pace": "Pace (poss/48)",
    "efg": "eFG%",
    "tov_pct": "TOV%",
    "oreb_pct": "OREB%",
    "dreb_pct": "DREB%",
    "ft_rate": "FT rate",
}

__all__ = [
    "BOX_STATS",
    "player_stats",
    "team_totals",
    "possessions",
    "advanced_stats",
    "score_win_prob",
    "ADVANCED_LABELS",
]
