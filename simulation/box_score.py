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

  * **Steals** are ONE turnover row: the player who lost the ball acts, and the defender who
    took it rides in ``secondary_player`` (``type="steal", result="cop"``). The turnover is
    the actor's, the steal is the secondary player's. Before 2.0 this was a two-row pair.
  * **Offensive fouls** emit no trailing turnover row — the turnover is counted from the
    foul row itself.
  * **Blocked shots** carry ``result="blocked"`` on the shooter's shot row (a missed
    attempt) plus a separate ``block`` event for the blocker.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

import pandas as pd

from models.game_state_features import period_index, period_start

from zones import FREE_THROW, is_three, points_for_shot

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
    pm: int = 0          # plus/minus: team points − opponent points while on court
    pts: int = 0

    @property
    def reb(self) -> int:
        return self.oreb + self.dreb

    @property
    def minutes(self) -> float:
        return self.seconds / 60.0

    def as_row(self) -> dict:
        """Full NBA-style display row (made-attempted splits, percentages, +/-)."""
        return {
            "Player": self.player,
            "MIN": round(self.minutes, 1),
            "FG": f"{self.fgm}-{self.fga}",
            "FG%": _pct(self.fgm, self.fga),
            "3PT": f"{self.tpm}-{self.tpa}",
            "3P%": _pct(self.tpm, self.tpa),
            "FT": f"{self.ftm}-{self.fta}",
            "FT%": _pct(self.ftm, self.fta),
            "OREB": self.oreb,
            "DREB": self.dreb,
            "REB": self.reb,
            "AST": self.ast,
            "STL": self.stl,
            "BLK": self.blk,
            "TO": self.tov,
            "PF": self.pf,
            "+/-": f"{self.pm:+d}",
            "PTS": self.pts,
        }


@dataclass
class BoxScore:
    """Both teams' stat lines plus the final score and team labels."""
    home: list[PlayerLine] = field(default_factory=list)
    away: list[PlayerLine] = field(default_factory=list)
    home_score: int = 0
    away_score: int = 0
    home_team: str = "HOME"
    away_team: str = "AWAY"
    # Team rebounds -- nobody is credited, so they belong on the TEAM line, not a player's.
    home_team_oreb: int = 0
    home_team_dreb: int = 0
    away_team_oreb: int = 0
    away_team_dreb: int = 0

    def to_frame(self, side: str, *, totals: bool = True) -> pd.DataFrame:
        """DataFrame of one side's stat lines, sorted by points descending.

        With ``totals`` (default) a ``TEAM`` row of summed counting stats is appended — the
        bottom line of a real box score (MIN totals ≈ 240, summed FG/REB/AST/…/PTS).
        """
        lines = self.home if side == "home" else self.away
        rows = [pl.as_row() for pl in sorted(lines, key=lambda p: p.pts, reverse=True)]
        if totals and lines:
            team_oreb = self.home_team_oreb if side == "home" else self.away_team_oreb
            team_dreb = self.home_team_dreb if side == "home" else self.away_team_dreb
            rows.append(_totals_row(lines, team_oreb=team_oreb, team_dreb=team_dreb))
        return pd.DataFrame(rows, columns=list(_DISPLAY_COLUMNS))

    def render(self) -> str:
        """Readable two-team box score with final score."""
        out = [
            f"{self.home_team} ({self.home_score})",
            self.to_frame("home").to_string(index=False),
            "",
            f"{self.away_team} ({self.away_score})",
            self.to_frame("away").to_string(index=False),
            "",
            f"Final: {self.home_team} {self.home_score} - {self.away_score} {self.away_team}",
        ]
        return "\n".join(out)

    def __str__(self) -> str:  # pragma: no cover - thin wrapper
        return self.render()


_DISPLAY_COLUMNS = ("Player", "MIN", "FG", "FG%", "3PT", "3P%", "FT", "FT%",
                    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "+/-", "PTS")


def _pct(made: int, att: int) -> str:
    """Shooting percentage as a 1-decimal string ("" when there were no attempts)."""
    return f"{100.0 * made / att:.1f}" if att else ""


def _other_side(side: str) -> str:
    return "away" if side == "home" else "home"


def _totals_row(lines: list[PlayerLine], *, team_oreb: int = 0, team_dreb: int = 0) -> dict:
    """A ``TEAM`` totals row: summed counting stats (no +/- — not meaningful as a sum).

    Team rebounds are added here and nowhere else: no player earned one, so they appear only on
    the bottom line, exactly as a real box score prints them.
    """
    s = lambda attr: sum(getattr(pl, attr) for pl in lines)  # noqa: E731
    fgm, fga = s("fgm"), s("fga")
    tpm, tpa = s("tpm"), s("tpa")
    ftm, fta = s("ftm"), s("fta")
    return {
        "Player": "TEAM", "MIN": round(s("seconds") / 60.0, 1),
        "FG": f"{fgm}-{fga}", "FG%": _pct(fgm, fga),
        "3PT": f"{tpm}-{tpa}", "3P%": _pct(tpm, tpa),
        "FT": f"{ftm}-{fta}", "FT%": _pct(ftm, fta),
        "OREB": s("oreb") + team_oreb, "DREB": s("dreb") + team_dreb,
        "REB": s("oreb") + s("dreb") + team_oreb + team_dreb,
        "AST": s("ast"), "STL": s("stl"), "BLK": s("blk"), "TO": s("tov"),
        "PF": s("pf"), "+/-": "", "PTS": s("pts"),
    }


def side_membership(events) -> tuple[set, set]:
    """(home names, away names) over every roster snapshot in ``events``.

    Which side a player is on is a fact about the GAME, not about any slice of it. Deriving it
    per slice drops a player who records a stat in a period he was not on the floor for -- his
    side resolves to None and his line is never emitted, so his stats vanish from that period's
    team total. Rare but real: one game in 1500 over the cleaned corpus.
    """
    home: set = set()
    away: set = set()
    for row in _as_rows(events):
        home.update(_roster(row.get("roster_home")))
        away.update(_roster(row.get("roster_away")))
    return home, away


def generate_box_score(events, *, home_team: str = "HOME",
                       away_team: str = "AWAY", seed_row=None, sides=None) -> BoxScore:
    """Aggregate a game's event sequence into a :class:`BoxScore`.

    ``events`` is an iterable of dict rows (or a ``pandas.DataFrame``) carrying at least
    ``event, player, type, result, time`` and the per-row ``roster_home`` / ``roster_away``
    snapshots. Rosters may be real lists or string literals (both are accepted). Events are
    processed in the given order; for minutes, the rows are treated as time-ordered.

    Plus/minus is credited per scoring play to the lineups on the floor at that moment: the
    scoring team's five gets ``+pts``, the opponents' five ``−pts``.

    ``sides`` is ``(home_names, away_names)`` from :func:`side_membership`, seeding which side
    each player is on. Only :func:`period_box_scores` passes it, and only because side membership
    is a game-level fact that a single period may not witness. It never invents a stat line: a
    player with no line in this slice is still not emitted.

    ``seed_row`` establishes where the first minutes interval starts, WITHOUT contributing any
    stats of its own. It exists for :func:`period_box_scores`: a slice that begins mid-game has
    real elapsed time before its first event, and with no seed that interval is credited to
    nobody -- so per-period minutes would not sum to the game's. Pass the last row of the
    preceding slice.
    """
    rows = _as_rows(events)

    lines: dict[str, PlayerLine] = {}
    home_players: set[str] = set()
    away_players: set[str] = set()
    home_score = away_score = 0
    # Team rebounds carry no player, so they are attributed by side: "team offensive" belongs to
    # whoever last shot, "team defensive" to the other team. Tracking the last shooter's team is
    # enough, and works identically on cleaned rows and on simulator rows (neither of which
    # carries possession).
    team_reb = {"home": [0, 0], "away": [0, 0]}     # [oreb, dreb]
    last_shot_team = None

    def line(name: str) -> PlayerLine:
        if name not in lines:
            lines[name] = PlayerLine(player=name)
        return lines[name]

    if sides is not None:
        home_players.update(sides[0])
        away_players.update(sides[1])

    prev_time = None
    prev_roster = ([], [])  # (home, away) on-court at the start of the current interval
    if seed_row is not None:
        prev_time = _as_float(seed_row.get("time"))
        prev_roster = (_roster(seed_row.get("roster_home")), _roster(seed_row.get("roster_away")))
        # The seed's lineup counts as having appeared in this slice: it is on the floor for the
        # interval the seed opens. Without this, a player substituted off AT the period break is
        # credited those seconds by `line()` and then dropped from the box, because the players
        # sets are built only from rows in the slice -- and his minutes vanish from the total.
        home_players.update(prev_roster[0])
        away_players.update(prev_roster[1])

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
            # Playerless rows are otherwise ignored, but a team rebound is a real stat with no
            # owner — the only row type that has to be counted before this guard.
            etype = _norm(row.get("type"))
            if event == "rebound" and etype in ("team offensive", "team defensive")                     and last_shot_team is not None:
                offensive = etype.endswith("offensive")
                side = last_shot_team if offensive else _other_side(last_shot_team)
                team_reb[side][0 if offensive else 1] += 1
            continue
        etype = _norm(row.get("type"))
        result = _norm(row.get("result"))
        team = "home" if player in home_players else "away" if player in away_players else None
        pl = line(player)

        if event == "shot":
            if team is not None:
                last_shot_team = team
            made = result == "made"
            # A shot row is either a free throw or one of the fifteen zones — points_for_shot
            # raises on anything else rather than silently scoring it as a two, which is what
            # the old `else: # 2pt` catch-all did. models/game_state_features.py calls the same
            # function, so the box score and the trained score feature cannot drift.
            pts = points_for_shot(etype) if made else 0
            if etype == FREE_THROW:
                pl.fta += 1
                if made:
                    pl.ftm += 1
                    pl.pts += 1
            else:
                pl.fga += 1
                three = is_three(etype)
                if three:
                    pl.tpa += 1
                if made:
                    pl.fgm += 1
                    if three:
                        pl.tpm += 1
                    pl.pts += pts
            if made:
                if team == "home":
                    home_score += pts
                elif team == "away":
                    away_score += pts
                # Plus/minus: credit the lineups on the floor for this scoring play.
                if team in ("home", "away"):
                    scorers, opponents = (
                        (home_roster, away_roster) if team == "home"
                        else (away_roster, home_roster)
                    )
                    for name in scorers:
                        line(name).pm += pts
                    for name in opponents:
                        line(name).pm -= pts

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
            # One row per turnover. The acting player lost the ball; on a steal the defender who
            # took it rides in secondary_player, the way a block row carries the blocked shooter.
            pl.tov += 1
            if etype == "steal":
                stealer = _norm(row.get("secondary_player"))
                if stealer and stealer not in ("null", "none", "PAD", "UNK"):
                    line(stealer).stl += 1
        elif event == "foul":
            if etype not in NON_PERSONAL_FOUL_TYPES:
                pl.pf += 1
            if etype == "offensive":
                # An offensive foul IS a turnover; the cleaner no longer emits a paired
                # turnover row, so the box score counts it from the foul.
                pl.tov += 1
        # substitution: roster mutation only (already reflected in row snapshots) — no stat.

    home = [lines[p] for p in sorted(home_players) if p in lines]
    away = [lines[p] for p in sorted(away_players) if p in lines]
    return BoxScore(home=home, away=away, home_score=home_score, away_score=away_score,
                    home_team_oreb=team_reb["home"][0], home_team_dreb=team_reb["home"][1],
                    away_team_oreb=team_reb["away"][0], away_team_dreb=team_reb["away"][1],
                    home_team=home_team, away_team=away_team)


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


# ===================================================================== #
# Period slicing                                                       --
# ===================================================================== #
#
# Nothing in the repo split a game by quarter before 2.0: eval_metrics is game-level, and the
# report's "progression" segments a run by dial changes, not by game periods. Without this,
# "is end-game mis-modelled?" has no answer -- which is the whole reason the clutch loss
# weighting was deferred rather than guessed at (correction S).
#
# The period rule is imported from models.game_state_features rather than restated. It exists
# twice already (there and GameController._period_index, which reads self.clock instead of a
# parameter and sits on the rollout's hot path); a third copy is how corrections N and R both
# started.


def split_by_period(events):
    """Partition an ordered event sequence into per-period slices.

    Yields ``(period, rows, seed_row)`` in period order, where ``seed_row`` is the last row of
    the preceding period (``None`` for the first). Feed that straight to
    :func:`generate_box_score` so the minutes interval spanning the buzzer is credited to the
    lineup that was on the floor for it -- otherwise per-period minutes do not sum to the game's.

    Rows with no usable ``time`` inherit the current period rather than forming one of their own:
    a period is a stretch of the clock, and a row that cannot say where it sits belongs with its
    neighbours. Periods are emitted in first-appearance order, which for time-ordered rows is
    numeric order; a game with no rows yields nothing.

    **A trailing slice with no elapsed time is not a period.** ``period_index`` treats a period as
    half-open, so a clock landing exactly on a boundary opens the next one -- right for the game
    STATE (at 2880 the state really is "a new period, 300 seconds left") and wrong for an EVENT,
    because a shot at 0.0 is a buzzer-beater belonging to the quarter it ended. Every regulation
    game carries such rows: the final shot, its block, and the ``end`` sentinel all sit at exactly
    2880, and left alone they invent a fifth period in every game that never went to overtime.
    Merging a zero-duration tail back into the period it ends says that without a second copy of
    the period arithmetic, and it generalises: an OT game's own buzzer rows sit at 3180 and are
    folded the same way. A mid-game slice is never zero-duration in a real game, and one that was
    would be a data fault worth seeing, so only the tail is folded.
    """
    rows = _as_rows(events)
    slices: list[tuple[int, list, object]] = []
    current = None
    prev_row = None
    for row in rows:
        t = _as_float(row.get("time"))
        period = current if t is None else period_index(t)
        if period is None:                     # leading rows with no clock at all
            period = 0
        if period != current:
            slices.append((period, [], prev_row))
            current = period
        slices[-1][1].append(row)
        prev_row = row

    if len(slices) > 1:
        _, tail, _ = slices[-1]
        times = [t for t in (_as_float(r.get("time")) for r in tail) if t is not None]
        # Every row sitting exactly on the period's own start, and no time elapsing: that is a
        # buzzer, not a period. Testing the boundary as well as the duration matters -- a short
        # trailing run that merely happens to share one clock reading is a real (if tiny) period,
        # and folding it would hide an out-of-order game rather than report it.
        if times and max(times) == min(times) == period_start(times[0]):
            slices[-2][1].extend(tail)
            slices.pop()
    return slices


def period_box_scores(events, *, home_team: str = "HOME",
                      away_team: str = "AWAY") -> dict[int, BoxScore]:
    """Per-period :class:`BoxScore` for one game, keyed by period index (0-3, then OT).

    Every counting stat sums across the returned boxes to the whole-game box, and so do minutes
    -- that identity is the check this is verified by, over the real cleaned corpus, and it is
    the reason ``seed_row`` exists.
    """
    rows = _as_rows(events)
    sides = side_membership(rows)
    out: dict[int, BoxScore] = {}
    for period, part, seed in split_by_period(rows):
        box = generate_box_score(part, home_team=home_team, away_team=away_team,
                                 seed_row=seed, sides=sides)
        if period in out:
            # Contiguous runs, so a repeat means the clock went backwards across a period
            # boundary. Overwriting would silently lose a whole run of rows; refusing says so.
            raise ValueError(f"period {period} appears twice: the event times are out of order")
        out[period] = box
    return out


__all__ = ["PlayerLine", "BoxScore", "generate_box_score", "box_score_for_game",
           "split_by_period", "period_box_scores", "side_membership"]
