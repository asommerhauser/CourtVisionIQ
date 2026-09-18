"""
Per-player and per-team cold-start priors, computed causally from the play-by-play we already have.

At tip-off the model's context window is empty, so identity is one career-averaged embedding row per
name. The 2.0 evaluation showed exactly what that costs: regressing the model's prediction on a
player's career and season averages gives ``0.24 x career + 0.61 x season``, and the 0.24 is the
drag. It predicts who a player *was*. Rookies came out 10.9% worse than "use his season average",
and players whose role moved more than 6 ppg -- the most-improved profile -- came out 24% worse.
Desmond Bane, career 14.0 ppg and 21.4 in January 2023, carried a -8.7 ppg bias.

The fix is to hand the model the season-to-date rates as an input, so who a player is *this season*
is something it can read rather than something it has to have memorised. This module computes them.

**Causality is the whole contract.** The walk writes a game's priors from running totals and only
then folds that game in, so game N's prior contains games 1..N-1 and nothing else:

    for each game, in chronological order:
        emit priors for every rostered player    # from running totals -- BEFORE this game
        box = generate_box_score(rows)           # the vetted tally
        add box to the running totals            # AFTER

Training and inference read the same written column, so they cannot disagree, and a leak here would
make training look better and inference worse -- the one failure that flatters itself.

**No second tally.** Boxes come from :func:`simulation.box_score.generate_box_score`, which is what
``actual_boxscore.txt`` and the report already score against. A throwaway re-implementation during
the 2.0 analysis matched only 81 of 96 final scores, on free-throw and heave edge cases. Team rates
come from ``simulation.stats.advanced_stats`` over the same boxes.

**Written to a sidecar, not to the season CSVs.** ``rest_home`` sets the precedent of a
roster-parallel list column, but it holds one integer per player. These hold ten floats. Measured on
``data/season2023.csv`` -- 216 MB, ~743k rows, 291 chars a row, of which ``rest_home`` is 15 -- ten
rates across five slots and two sides is ~800 characters a row, which takes ``data/`` from 4.2 GB to
roughly 13 GB and rewrites every season file in place. The natural grain is (game, player): 26,969
games x ~22 player-lines is ~590k rows for the whole corpus, a few MB of Parquet, and the cleaned
CSVs are never touched. ``models/prior_features.py`` joins it back onto the roster slots.

    python -m player_priors                      # all cleaned seasons -> data/priors/
    python -m player_priors --seasons 2013,2023  # just these

TF-free: see ``simulation/__init__`` on why importing ``generate_box_score`` no longer drags
TensorFlow into the data pipeline.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from data_loading import cleaned_csvs, season_offsets
from simulation.box_score import generate_box_score
from simulation.stats import advanced_stats, team_totals

# The rates, in the order the model reads them. Per-36 rather than per-game wherever the stat is a
# volume, so the rate is role-independent and ``min_pg`` carries the role -- a sixth man who scores
# efficiently and a starter who scores the same way look alike here, and the minutes tell them apart.
# The thirteen SHRUNK rates. Each is an estimate of a player's true rate, so each is blended toward
# his seed by ``shrink`` in proportion to how many games it rests on.
PLAYER_RATE_KEYS = (
    "min_pg", "pts_36", "fga_36", "fg_pct", "tpa_rate",
    "fta_rate", "ast_36", "oreb_36", "dreb_36", "tov_36",
    # 3.2 W5. The model had a free-throw ATTEMPT rate and no MAKE rate, and FT% is the most stable
    # player stat there is. shot_type emits corner3_l / wing3_r / top3 as distinct tokens and
    # shot_result judged them without knowing whether the player can shoot one. And nothing said who
    # actually fouls -- though note pf_36 does NOT fix the 9.6x over-production of fourth fouls,
    # which is a benching failure rather than an attribution one.
    "ft_pct", "tp_pct", "pf_36",
)

# The four DERIVED scalars. Not estimates of a rate, so they are not shrunk: career stage is a fact
# about the calendar, and each delta is already a difference of two shrunk quantities.
#
# docs/v3_direction.md 1b's failure axis IS career games and role shift -- rookies 10.9% worse than
# their own season average, role-shifters 24% -- and these index it directly rather than hoping the
# rates imply it.
PLAYER_DERIVED_KEYS = ("career_stage", "d_pts_36", "d_min_pg", "d_fga_36")

# Order is load-bearing three ways: the parquet column order, models.prior_features._NORM's index
# loop, and ops.unstack on the last axis. tests/test_player_priors.py pins it.
PLAYER_PRIOR_KEYS = PLAYER_RATE_KEYS + PLAYER_DERIVED_KEYS

# Which rate each season-over-season delta is taken on. Volume rates only: a role change shows up as
# more minutes and more shots, not as a different true FG%.
DELTA_SOURCES = {"d_pts_36": "pts_36", "d_min_pg": "min_pg", "d_fga_36": "fga_36"}
TEAM_PRIOR_KEYS = ("net_rating", "pace", "off_rating", "def_rating")

# Shrinkage: a rate from n games is worth n/(n+k) of itself, the rest coming from the seed. A
# training-side constant, deliberately NOT a _TUNING_KEYS dial -- it is baked into the written
# column, so changing it means rebuilding the sidecar and retraining, which is not what a dial is.
# k = 10 puts a player at half weight after ten games, which is about where a season-to-date scoring
# average stops being noise.
SHRINK_K = 10.0

# Where the seed comes from when a player has no previous season either -- a rookie in his first
# game, or anyone at all in the earliest season of the corpus. League-average shapes, so his prior
# reads "an ordinary NBA player" rather than "a player who does nothing", which is what a zero would
# say and what the embedding table already effectively says.
LEAGUE_DEFAULTS = {
    "min_pg": 20.0, "pts_36": 16.0, "fga_36": 13.5, "fg_pct": 0.455, "tpa_rate": 0.28,
    "fta_rate": 0.26, "ast_36": 3.6, "oreb_36": 1.6, "dreb_36": 5.2, "tov_36": 2.2,
    # Measured on 2022-23 via generate_box_score: ft_pct 0.7825, tp_pct 0.3600, pf_36 2.9689.
    # The modern value is the right one -- the training corpus starts at MIN_TRAIN_SEASON.
    "ft_pct": 0.78, "tp_pct": 0.36, "pf_36": 2.95,
    # Measured over 2011+ player-games: mean 4.84 seasons since first appearance, median 4.
    "career_stage": 4.5,
    # A delta's neutral value is genuinely zero: "no evidence he has changed". A player with no
    # previous season gets exactly this rather than a delta against the league mean, which is what a
    # naive implementation emits for every rookie.
    "d_pts_36": 0.0, "d_min_pg": 0.0, "d_fga_36": 0.0,
}
TEAM_DEFAULTS = {"net_rating": 0.0, "pace": 98.0, "off_rating": 108.0, "def_rating": 108.0}

PRIORS_DIRNAME = "priors"
MANIFEST_NAME = "manifest.json"


def _parse_roster(cell):
    if isinstance(cell, list):
        return [str(n) for n in cell]
    if isinstance(cell, str) and cell.startswith("["):
        try:
            return [str(n) for n in ast.literal_eval(cell)]
        except (ValueError, SyntaxError):
            return []
    return []


def _safe(num, den, default=0.0):
    return (num / den) if den else default


class PlayerTotals:
    """Running per-player counting totals, and the rates derived from them.

    Totals, not rates, because a rate of rates is not a rate: averaging ten single-game per-36
    figures over-weights the night a player took four minutes and hit a three.
    """

    # PriorCarry.roll sums the league aggregate by iterating __slots__, so a stat added here joins
    # the league seed automatically.
    __slots__ = ("games", "seconds", "pts", "fga", "fgm", "tpa", "tpm", "fta", "ftm", "ast",
                 "oreb", "dreb", "tov", "pf")

    def __init__(self) -> None:
        self.games = 0
        self.seconds = self.pts = self.fga = self.fgm = 0.0
        self.tpa = self.tpm = self.fta = self.ftm = 0.0
        self.ast = self.oreb = self.dreb = self.tov = self.pf = 0.0

    def add(self, line) -> None:
        self.games += 1
        self.seconds += line.seconds
        self.pts += line.pts
        self.fga += line.fga
        self.fgm += line.fgm
        self.tpa += line.tpa
        self.tpm += line.tpm
        self.fta += line.fta
        self.ftm += line.ftm
        self.pf += line.pf
        self.ast += line.ast
        self.oreb += line.oreb
        self.dreb += line.dreb
        self.tov += line.tov

    def rates(self) -> dict[str, float]:
        """Raw (unshrunk) rates from the totals so far. Undefined on zero games."""
        minutes = self.seconds / 60.0
        per36 = 36.0 / minutes if minutes else 0.0
        return {
            "min_pg": _safe(minutes, self.games),
            "pts_36": self.pts * per36,
            "fga_36": self.fga * per36,
            "fg_pct": _safe(self.fgm, self.fga, LEAGUE_DEFAULTS["fg_pct"]),
            "tpa_rate": _safe(self.tpa, self.fga, LEAGUE_DEFAULTS["tpa_rate"]),
            "fta_rate": _safe(self.fta, self.fga, LEAGUE_DEFAULTS["fta_rate"]),
            "ast_36": self.ast * per36,
            "oreb_36": self.oreb * per36,
            "dreb_36": self.dreb * per36,
            "tov_36": self.tov * per36,
            "ft_pct": _safe(self.ftm, self.fta, LEAGUE_DEFAULTS["ft_pct"]),
            "tp_pct": _safe(self.tpm, self.tpa, LEAGUE_DEFAULTS["tp_pct"]),
            "pf_36": self.pf * per36,
        }


class TeamTotals:
    """Running per-team totals, for and against, and the four team rates."""

    __slots__ = ("games", "pts", "poss", "opp_pts", "opp_poss", "minutes")

    def __init__(self) -> None:
        self.games = 0
        self.pts = self.poss = self.opp_pts = self.opp_poss = self.minutes = 0.0

    def add(self, own: dict, opp: dict) -> None:
        adv_own = advanced_stats(own, opp)
        adv_opp = advanced_stats(opp, own)
        self.games += 1
        self.pts += adv_own["pts"]
        self.poss += adv_own["poss"]
        self.opp_pts += adv_opp["pts"]
        self.opp_poss += adv_opp["poss"]
        self.minutes += (own["seconds"] / 60.0) / 5.0

    def rates(self) -> dict[str, float]:
        off = 100.0 * _safe(self.pts, self.poss, TEAM_DEFAULTS["off_rating"] / 100.0)
        dfn = 100.0 * _safe(self.opp_pts, self.opp_poss, TEAM_DEFAULTS["def_rating"] / 100.0)
        return {
            "net_rating": off - dfn,
            "pace": _safe(self.poss, self.minutes, TEAM_DEFAULTS["pace"] / 48.0) * 48.0,
            "off_rating": off,
            "def_rating": dfn,
        }


def shrink(observed: dict[str, float], seed: dict[str, float], games: int,
           keys=PLAYER_RATE_KEYS, k: float = SHRINK_K) -> dict[str, float]:
    """``games/(games+k)`` of the observed rate, the rest from ``seed``.

    On game 1 with no previous season this is the league default outright, which is the point: a
    rookie's opening night should read "an ordinary player" and grow toward what he actually is,
    rather than reading as zeros for ten games and then jumping.
    """
    weight = games / (games + k) if games else 0.0
    return {key: weight * observed.get(key, 0.0) + (1.0 - weight) * seed.get(key, 0.0)
            for key in keys}


class PriorCarry:
    """Previous-season rates, threaded across season files in chronological order.

    Seasons must be walked ascending for this to mean anything -- a seed taken from a *later* season
    is a leak, and one that no within-season causality check would catch, since every prior would
    still be built only from earlier games of its own season.
    """

    def __init__(self) -> None:
        self.player_seed: dict[str, dict[str, float]] = {}
        self.team_seed: dict[str, dict[str, float]] = {}
        self.league_seed: dict[str, float] = dict(LEAGUE_DEFAULTS)
        # 3.2 W5: the season a name is first seen in, for career_stage. Left-censored at the first
        # season of the walk -- a player already active in 2003 reads as stage 0 that year -- which is
        # why the walk covers all 21 seasons and is never cut with the training corpus.
        self.first_season: dict[str, int] = {}

    def seed_for(self, name: str) -> dict[str, float]:
        return self.player_seed.get(name) or self.league_seed

    def has_history(self, name: str) -> bool:
        """Whether ``seed_for`` returns this player's OWN previous season rather than the league mean.

        The deltas depend on the distinction. Without it a rookie's "season-over-season change" is a
        comparison against the league average, which is a statement about how good he is, not about
        how he has changed -- and it would be indistinguishable from a real role shift.
        """
        return name in self.player_seed

    def note_appearance(self, name: str, season: int) -> None:
        """Record the first season a name is seen in. Idempotent, so the walk order does the work."""
        self.first_season.setdefault(name, int(season))

    def career_stage(self, name: str, season: int) -> float:
        """Seasons since first appearance. Zero in a player's first season."""
        first = self.first_season.get(name)
        return 0.0 if first is None else float(max(0, int(season) - first))

    def team_seed_for(self, team: str) -> dict[str, float]:
        return self.team_seed.get(team) or dict(TEAM_DEFAULTS)

    def roll(self, players: dict[str, PlayerTotals], teams: dict[str, TeamTotals]) -> None:
        """Close a season: its final rates become the next season's seeds."""
        # A player needs a real sample before he can seed himself; below that the league mean is a
        # better guess than his own eleven minutes.
        self.player_seed = {name: totals.rates() for name, totals in players.items()
                            if totals.games >= 5}
        self.team_seed = {team: totals.rates() for team, totals in teams.items()}
        league = PlayerTotals()
        for totals in players.values():
            for slot in PlayerTotals.__slots__:
                setattr(league, slot, getattr(league, slot) + getattr(totals, slot))
        if league.games:
            self.league_seed = league.rates()


def _game_order(frame: pd.DataFrame) -> list[int]:
    """Game ids in chronological order -- the same ``(date, game_id)`` key season_context uses."""
    meta = (pd.DataFrame({"game_id": frame["game_id"].to_numpy(),
                          "date": pd.to_datetime(frame["game_date"], errors="coerce").to_numpy()})
            .groupby("game_id", sort=False).first().reset_index()
            .sort_values(["date", "game_id"], kind="stable"))
    return [int(g) for g in meta["game_id"]]


def priors_for_season(frame: pd.DataFrame, carry: PriorCarry) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk one season chronologically; return its (player, team) prior tables.

    ``carry`` is read for seeds and advanced past this season on the way out, so calling this over
    the seasons in order threads the previous-season seeds without the caller holding any state.
    """
    groups = dict(list(frame.groupby("game_id", sort=False)))
    players: dict[str, PlayerTotals] = {}
    teams: dict[str, TeamTotals] = {}
    player_rows: list[dict] = []
    team_rows: list[dict] = []
    season = int(frame["season"].dropna().iloc[0]) if "season" in frame else 0

    for gid in _game_order(frame):
        rows = groups[gid]
        roster = set()
        for col in ("roster_home", "roster_away"):
            for cell in rows[col]:
                roster.update(_parse_roster(cell))
        roster.discard("")
        # Seen now, so a debutant reads career_stage 0 in his first game rather than through the
        # ``first is None`` branch. Idempotent, so it is the earliest season that sticks.
        for name in roster:
            carry.note_appearance(name, season)

        # --- WRITE: every prior here comes from games already folded in. ---
        for name in sorted(roster):
            totals = players.get(name)
            observed = totals.rates() if totals else {}
            games = totals.games if totals else 0
            seed = carry.seed_for(name)
            row = shrink(observed, seed, games)
            # The derived four are NOT shrunk. career_stage is a fact about the calendar, and each
            # delta is already a difference of two shrunk quantities: it is taken against the seed, so
            # early in a season -- when the shrunk rate still IS mostly the seed -- it reads near zero
            # and grows as evidence accumulates. That is the honest shape for "has his role changed":
            # we do not yet know.
            row["career_stage"] = carry.career_stage(name, season)
            has_history = carry.has_history(name)
            for key, source in DELTA_SOURCES.items():
                row[key] = (row[source] - seed[source]) if has_history else 0.0
            player_rows.append({"game_id": gid, "player": name, "prior_games": games, **row})

        home = str(rows["home_team"].dropna().iloc[0]) if rows["home_team"].notna().any() else ""
        away = str(rows["away_team"].dropna().iloc[0]) if rows["away_team"].notna().any() else ""
        for side, team in (("home", home), ("away", away)):
            totals = teams.get(team)
            observed = totals.rates() if totals else {}
            games = totals.games if totals else 0
            row = shrink(observed, carry.team_seed_for(team), games, keys=TEAM_PRIOR_KEYS)
            team_rows.append({"game_id": gid, "side": side, "team": team,
                              "prior_games": games, **row})

        # --- ADVANCE: fold this game in, only now. ---
        box = generate_box_score(rows)
        for line in (*box.home, *box.away):
            players.setdefault(line.player, PlayerTotals()).add(line)
        own = {"home": team_totals(box.home), "away": team_totals(box.away)}
        for side, team in (("home", home), ("away", away)):
            other = "away" if side == "home" else "home"
            teams.setdefault(team, TeamTotals()).add(own[side], own[other])

    carry.roll(players, teams)
    return pd.DataFrame(player_rows), pd.DataFrame(team_rows)


def priors_dir(data_dir: str = "./data") -> Path:
    return Path(data_dir) / PRIORS_DIRNAME


def season_label(path: Path) -> str:
    return path.stem.replace("season", "")


def build(data_dir: str = "./data", *, seasons=None, echo=print) -> Path:
    """Build the whole sidecar. Seasons are walked ascending so the seed chain is causal."""
    paths = sorted(cleaned_csvs(data_dir), key=lambda p: season_label(p))
    if seasons:
        wanted = {str(s) for s in seasons}
        paths = [p for p in paths if season_label(p) in wanted]
    if not paths:
        raise FileNotFoundError(f"no cleaned season CSVs under {Path(data_dir).resolve()}")

    out = priors_dir(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    # The sidecar is joined to training rows by game_id, so it must use the SAME numbering
    # load_all_cleaned does -- per-file ids collide across seasons and are shifted past every
    # earlier file's maximum. Reading each CSV directly and keeping its raw id is what made the
    # sidecar cover 1,277 of 26,969 games (season 2003, the only file whose offset is zero) and
    # what made merge_prior_features refuse the whole thing as stale. See data_loading.season_offsets.
    offsets = season_offsets(data_dir)
    carry = PriorCarry()
    written = []
    for path in paths:
        label = season_label(path)
        frame = pd.read_csv(path)
        frame["game_id"] = frame["game_id"].astype(int) + offsets[path]
        player_frame, team_frame = priors_for_season(frame, carry)
        player_frame.to_parquet(out / f"players_{label}.parquet", index=False)
        team_frame.to_parquet(out / f"teams_{label}.parquet", index=False)
        written.append(label)
        if echo:
            echo(f"  season {label}: {len(player_frame):,} player-games, "
                 f"{player_frame['player'].nunique():,} players, "
                 f"{len(team_frame) // 2:,} games")

    (out / MANIFEST_NAME).write_text(json.dumps({
        "player_keys": list(PLAYER_PRIOR_KEYS),
        "team_keys": list(TEAM_PRIOR_KEYS),
        "shrink_k": SHRINK_K,
        "league_defaults": LEAGUE_DEFAULTS,
        "seasons": written,
    }, indent=2), encoding="utf-8")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seasons", default=None,
                    help="comma-separated seasons to (re)build; default all. Rebuilding a subset "
                         "restarts the seed chain at the first one, so its opening games get "
                         "league-mean seeds -- fine for a check, not for a train.")
    args = ap.parse_args(argv)
    seasons = [s.strip() for s in args.seasons.split(",")] if args.seasons else None
    out = build(args.data_dir, seasons=seasons)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
