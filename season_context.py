"""
Season-context enrichment — a stateful pre-pass over the cleaned data.

The cleaner (``data_cleaner.py``) emits pure per-event rows plus three raw passthroughs:
``game_date``, ``home_team``, ``away_team``. This module walks each season's games in
chronological order, tracking per-team games-played and per-team / per-player last-played
dates (all reset at season boundaries), and writes back the season-level features the
models consume. Every value is constant within a game.

Columns added to each cleaned season CSV:
  home_games_played, away_games_played : games each team has already played / 82 (season
                                         progress; 0.0 on the opener, > 1.0 in the playoffs)
  home_days_rest,    away_days_rest    : days since that team last played (3 for openers)
  rest_home,         rest_away         : roster-parallel days-since-last-game per on-court
                                         player — a list literal aligned slot-for-slot with
                                         ``roster_home`` / ``roster_away`` (3 on a player's
                                         first appearance of the season)

First appearance (a team's or player's opener) → 3 days: treat them as fresh and ready.
See ``models/season_features.py`` for how these are normalized and fed to the models.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loading import cleaned_csvs, _parse_roster

# A team's or player's first game of the season has no prior game to rest from; treat them
# as fresh rather than "infinitely rested".
DEFAULT_REST_DAYS = 3
# Nominal regular-season length used to normalize games-played into season progress.
SEASON_GAMES = 82.0

NEW_COLUMNS = [
    "home_games_played", "away_games_played",
    "home_days_rest", "away_days_rest",
    "rest_home", "rest_away",
]


def _game_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game: its date + home/away abbreviation, ordered chronologically.

    Teams/date are stamped per row by the cleaner but a couple of synthetic rows can carry
    a stale/blank value (see data_cleaner), so the per-game value is the first non-null.
    """
    date = pd.to_datetime(df["game_date"], errors="coerce")
    meta = (
        pd.DataFrame({
            "game_id": df["game_id"].to_numpy(),
            "date": date.to_numpy(),
            "home_team": df["home_team"].to_numpy(),
            "away_team": df["away_team"].to_numpy(),
        })
        .groupby("game_id", sort=False)
        .first()
        .reset_index()
    )
    return meta.sort_values(["date", "game_id"], kind="stable").reset_index(drop=True)


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the season-context columns added (pure; no file I/O).

    Games are grouped by season and walked in chronological order; trackers reset between
    seasons. ``roster_home`` / ``roster_away`` are parsed only to align the per-player rest
    lists — the original roster columns are left untouched.
    """
    df = df.copy()
    home_lists = df["roster_home"].apply(_parse_roster)
    away_lists = df["roster_away"].apply(_parse_roster)

    # Per-game results, keyed by game_id.
    hgp: dict[int, float] = {}
    agp: dict[int, float] = {}
    hdr: dict[int, int] = {}
    adr: dict[int, int] = {}
    player_rest: dict[int, dict[str, int]] = {}

    for _, season_df in df.groupby("season", sort=False):
        meta = _game_metadata(season_df)
        team_games: dict[str, int] = {}
        team_last: dict[str, pd.Timestamp] = {}
        player_last: dict[str, pd.Timestamp] = {}

        # Roster unions per game (which players are on each side this game).
        season_mask = df["season"] == season_df["season"].iloc[0]
        union: dict[int, set[str]] = {}
        for gid, h, a in zip(df.loc[season_mask, "game_id"], home_lists[season_mask], away_lists[season_mask]):
            bucket = union.setdefault(int(gid), set())
            bucket.update(h)
            bucket.update(a)

        for row in meta.itertuples(index=False):
            gid = int(row.game_id)
            date = row.date
            home, away = row.home_team, row.away_team

            hgp[gid] = team_games.get(home, 0) / SEASON_GAMES
            agp[gid] = team_games.get(away, 0) / SEASON_GAMES
            hdr[gid] = _days_since(date, team_last.get(home))
            adr[gid] = _days_since(date, team_last.get(away))

            rest = {p: _days_since(date, player_last.get(p)) for p in union.get(gid, ())}
            player_rest[gid] = rest

            # Advance trackers AFTER computing this game's features.
            team_games[home] = team_games.get(home, 0) + 1
            team_games[away] = team_games.get(away, 0) + 1
            if pd.notna(date):
                team_last[home] = date
                team_last[away] = date
                for p in union.get(gid, ()):  # noqa: PD011
                    player_last[p] = date

    gid_col = df["game_id"]
    df["home_games_played"] = gid_col.map(hgp).astype("float32")
    df["away_games_played"] = gid_col.map(agp).astype("float32")
    df["home_days_rest"] = gid_col.map(hdr).astype("int64")
    df["away_days_rest"] = gid_col.map(adr).astype("int64")
    df["rest_home"] = [
        [player_rest[int(g)].get(p, DEFAULT_REST_DAYS) for p in roster]
        for g, roster in zip(gid_col, home_lists)
    ]
    df["rest_away"] = [
        [player_rest[int(g)].get(p, DEFAULT_REST_DAYS) for p in roster]
        for g, roster in zip(gid_col, away_lists)
    ]
    return df


def _days_since(date, last) -> int:
    """Whole days between ``last`` and ``date``; DEFAULT_REST_DAYS when there is no prior."""
    if last is None or pd.isna(last) or pd.isna(date):
        return DEFAULT_REST_DAYS
    return int((date - last).days)


def enrich_file(path: Path) -> Path:
    """Enrich one cleaned season CSV in place (read → enrich → overwrite)."""
    path = Path(path)
    df = pd.read_csv(path)
    df = enrich_df(df)
    df.to_csv(path, index=False)
    print(f"Enriched season context -> {path} ({len(df)} rows)")
    return path


def enrich(data_dir="./data") -> list[Path]:
    """Enrich every cleaned season CSV under ``data_dir`` (one file == one season)."""
    paths = cleaned_csvs(data_dir)
    if not paths:
        raise FileNotFoundError(f"No cleaned CSVs found in {Path(data_dir).resolve()}")
    return [enrich_file(p) for p in paths]


if __name__ == "__main__":
    enrich()
