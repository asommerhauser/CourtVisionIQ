import os
import re

import pandas as pd

import zones

# The raw foul label, and the two tokens it splits into. A shooting foul's free-throw count is a
# fact about the fouled *attempt* (was it a 2 or a 3), which the cleaner can read off the
# following trip's ``outof``. Before 2.0 the controller guessed it by sampling the live shot-type
# head — a head never trained to answer that question.
SHOOTING_FOUL = "shooting"
SHOOTING_2PT = "shooting 2pt"
SHOOTING_3PT = "shooting 3pt"
# Private column carrying the precomputed label from parse_file into process_row. Never emitted.
_SHOOTING_LABEL_COL = "_shooting_foul_label"
# How far to look for the trip a shooting foul produced. Measured on 2022-23: the free throw is
# the very next row 82% of the time, but a substitution or two can sit in between (observed max
# gap 7). Rows that find nothing inside the window fall back on the preceding shot.
_FT_LOOKAHEAD = 8
_SHOT_LOOKBEHIND = 6
# Raw rows that end a possession sequence — never scan a lookahead/lookbehind across one.
_BOUNDARY_EVENTS = {"start of period", "end of period"}


class DataCleaner:
    """
    Converts raw NBA play-by-play CSVs into the normalized event format consumed
    by all downstream models.

    Parameters
    ----------
    start : int
        Index to begin processing files from (after filtering). Defaults to 0.
    end : int
        Index to stop processing files at. None means process all files.
    data_path : str | None
        Directory of raw master files. Defaults to DATA_PATH.
    ignore : Iterable[str] | None
        Filename substrings to skip (e.g. sample/Truncated files). Defaults to
        IGNORE_SUBSTRINGS.
    """

    DATA_PATH = "./RawData/MasterFiles"
    # Raw files whose names contain any of these are skipped (samples/subsets that
    # would duplicate games already present in the full master file).
    IGNORE_SUBSTRINGS = ("Truncated",)

    def __init__(self, start=0, end=None, data_path=None, ignore=None):
        self.start = start
        self.end = end
        self.data_path = data_path or self.DATA_PATH
        self.ignore = tuple(ignore) if ignore is not None else self.IGNORE_SUBSTRINGS
        self.season = 0
        self.playoff = 0
        # Per-game context carried onto every event (constant within a game). Date comes
        # straight from the raw row; the team abbreviations are resolved lazily from the
        # first action rows whose actor's roster membership is known (see _update_teams).
        self.game_date = None
        self.home_team = None
        self.away_team = None
        # Per-game cumulative roster (used for end-of-game sentinel event).
        self.home_players = []
        self.away_players = []
        # Last observed on-court 5-player lineups (NaN-free).
        self.last_home_five = []
        self.last_away_five = []
        self.first = True
        # Monotonic globally-unique game id (a single raw file may hold >1 game).
        self.game_id = 0
        # Last valid cumulative time, used to stamp synthetic "end" events and
        # as a fallback when a row's time cannot be parsed.
        self.last_time = 0

        self.events = []
        self.output_columns = [
            "game_id", "roster_home", "roster_away", "time", "event",
            "player", "type", "result", "secondary_player", "home/away", "season", "playoff",
            "game_date", "home_team", "away_team",
        ]

    # ------------------- HELPER METHODS -------------------

    @staticmethod
    def _clean_five(five):
        """Return only the non-NaN entries from a raw h1..h5 / a1..a5 list."""
        return [p for p in five if pd.notna(p)]

    def convert_time(self, quarter, time_past):
        """
        Cumulative game time in seconds since tipoff.
        NBA: Q1–Q4 = 12:00 each; OT periods = 5:00 each.
        quarter : int-like (1,2,3,4,5=OT1,6=OT2,…)
        time_past: "H:MM:SS" or "MM:SS" elapsed within the period.
        """
        if pd.isna(quarter) or pd.isna(time_past):
            return None
        try:
            q = int(quarter)
            parts = str(time_past).strip().split(":")
            if len(parts) == 3:
                hh, mm, ss = map(int, parts)
            elif len(parts) == 2:
                hh = 0
                mm, ss = map(int, parts)
            else:
                return None
        except Exception:
            return None

        base = (48 * 60) + ((q - 5) * 5 * 60) if q > 4 else (q - 1) * 12 * 60
        return base + (hh * 3600) + (mm * 60) + ss

    def home_indicator(self, home_roster, player):
        """Return 1 if player is on the home team, 2 if on away team."""
        return 1 if player in home_roster else 2

    def _ctx(self):
        """Per-game context keys stamped onto each emitted event (constant per game)."""
        return {
            "game_date": self.game_date,
            "home_team": self.home_team,
            "away_team": self.away_team,
        }

    def _update_teams(self, row, clean_home, clean_away):
        """Resolve the game's home/away team abbreviations from the raw ``team`` column.

        ``home``/``away`` in the raw data are jump-ball player names and ``opponent`` is
        empty, so the only stable per-game team id is the event-team abbreviation. The
        first action row whose actor sits in the home five fixes the home abbreviation;
        the first whose actor sits in the away five fixes the away one. Idempotent once
        both are known.
        """
        if self.home_team is not None and self.away_team is not None:
            return
        player = row.get("player")
        team = row.get("team")
        if pd.isna(player) or pd.isna(team):
            return
        team = str(team).strip()
        if self.home_team is None and player in clean_home:
            self.home_team = team
        elif self.away_team is None and player in clean_away:
            self.away_team = team

    def determine_turnover_type(self, data):
        """
        Map raw turnover 'type' text to a coarse category.
        Returns 'violation', 'error', 'null', or None (skip the event entirely).
        """
        check_vio = {
            '3-second violation', 'shot clock', '8-second violation', 'lane violation',
            'offensive goaltending', 'palming', 'backcourt', '5-second violation',
            'double dribble', 'discontinue dribble', 'illegal assist',
            'jump ball violation', 'offensive foul', 'illegal screen',
            'basket from below', 'punched ball', 'too many players', 'traveling',
            'kicked ball',
        }
        check_error = {
            'lost ball', 'out of bounds lost ball', 'step out of bounds',
            'bad pass', 'inbound',
        }
        if pd.isna(data):
            return 'null'
        data = str(data).strip().lower()
        if data in ('', 'null'):
            return 'null'
        if data == 'no turnover':
            return None
        if data in check_vio:
            return 'violation'
        if data in check_error:
            return 'error'
        print(f"FLAG UNRECOGNIZED TURNOVER: {data}")
        return None

    def determine_foul_type(self, data):
        """
        Normalize raw foul 'type' text.
        'offensive charge' → 'offensive'; anything ending in 'technical' → 'technical'.
        """
        if pd.isna(data):
            return "null"
        data_str = str(data).strip()
        if not data_str:
            return "null"
        if data_str == "offensive charge":
            return "offensive"
        if data_str[-9:].lower() == "technical":
            return "technical"
        return data_str

    @staticmethod
    def _label_shooting_fouls(df):
        """Per-row ``shooting 2pt`` / ``shooting 3pt`` label for every shooting-foul row.

        Needs a whole-file pass because the answer lives *after* the foul: the trip it produced
        carries ``outof``. Two or three is read straight off it. ``outof == 1`` is an and-1 —
        24% of shooting fouls in 2022-23, and 98.6% of them sit directly behind a made field
        goal — so the label comes from what that basket was worth, which is the same question
        ("was the fouled attempt a 2 or a 3") answered from the other side.

        Falls back to ``shooting 2pt``: it is 96% of the non-and-1 population, and the rows that
        reach the fallback are the 0.17% with no trip inside the window at all.
        """
        events = df["event_type"].tolist() if "event_type" in df else []
        outof = df["outof"].tolist() if "outof" in df else [None] * len(events)
        results = df["result"].tolist() if "result" in df else [None] * len(events)
        points = df["points"].tolist() if "points" in df else [None] * len(events)
        types = df["type"].tolist() if "type" in df else [None] * len(events)

        def _int(value):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return None

        labels = [None] * len(events)
        for i, event in enumerate(events):
            if event != "foul" or str(types[i] or "").strip() != SHOOTING_FOUL:
                continue

            trip = None
            for j in range(i + 1, min(i + _FT_LOOKAHEAD, len(events))):
                if events[j] in _BOUNDARY_EVENTS:
                    break
                if events[j] == "free throw":
                    trip = _int(outof[j])
                    break
            if trip in (2, 3):
                labels[i] = SHOOTING_3PT if trip == 3 else SHOOTING_2PT
                continue

            # And-1 (trip == 1), or no trip found: read the fouled attempt off the basket itself.
            labels[i] = SHOOTING_2PT
            for k in range(i - 1, max(i - _SHOT_LOOKBEHIND, -1), -1):
                if events[k] in _BOUNDARY_EVENTS:
                    break
                if events[k] == "shot":
                    if results[k] == "made" and _int(points[k]) == 3:
                        labels[i] = SHOOTING_3PT
                    break
        return labels

    def determine_foul_result(self, foul_type):
        """Map foul type to a result token."""
        mapping = {
            "personal": "nothing",
            "null": "nothing",
            "away from play": "nothing",
            SHOOTING_2PT: "free throw",
            SHOOTING_3PT: "free throw",
            "technical": "free throw",
            "personal take": "free throw op",
            "flagrant-1": "free throw op",
            "transition take": "free throw op",
            "offensive": "cop",
            "loose ball": "op",
            "flagrant-2": "ejection",
        }
        if foul_type not in mapping:
            raise ValueError(f"Unknown foul type: {foul_type!r}")
        return mapping[foul_type]

    # -------------------------------------------------------

    def _end_event(self):
        """Build the synthetic end-of-game sentinel event."""
        return {
            "game_id": self.game_id,
            "roster_home": self.last_home_five,
            "roster_away": self.last_away_five,
            "time": self.last_time,
            "event": "end",
            "player": "end",
            "type": "end",
            "result": "end",
            "secondary_player": "none",
            "home/away": 0,
            "season": self.season,
            "playoff": 2 if self.playoff else 1,
            **self._ctx(),
        }

    def process_row(self, row):
        """
        Convert one raw CSV row into a list of 0-or-more normalized event dicts.
        """
        events = []

        # ---- GAME BOUNDARY ----
        if row['event_type'] == "start of period" and row['period'] == 1:
            self.playoff = 0 if str(row["data_set"])[-1] == "n" else 1

            if not self.first:
                events.append(self._end_event())
            else:
                self.first = False

            self.game_id += 1
            self.last_time = 0
            # New game: reset per-game context. Date is on every raw row; the team
            # abbreviations are resolved as soon as a determinable action row arrives.
            self.game_date = row["date"] if pd.notna(row.get("date")) else None
            self.home_team = None
            self.away_team = None
            self.home_players = []
            self.away_players = []
            self.last_home_five = self._clean_five(
                [row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]]
            )
            self.last_away_five = self._clean_five(
                [row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]]
            )

            events.append({
                "game_id": self.game_id,
                "roster_home": self.last_home_five,
                "roster_away": self.last_away_five,
                "time": 0,
                "event": "start",
                "player": "start",
                "type": "start",
                "result": "start",
                "secondary_player": "none",
                "home/away": 0,
                "season": self.season,
                "playoff": 2 if self.playoff else 1,
            })

        # ---- CURRENT ON-COURT LINEUPS (NaN-free) ----
        home_five = [row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]]
        away_five = [row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]]
        clean_home = self._clean_five(home_five)
        clean_away = self._clean_five(away_five)

        # Keep last-known lineups up to date so the end event is accurate.
        if clean_home:
            self.last_home_five = clean_home
        if clean_away:
            self.last_away_five = clean_away

        # Resolve home/away team abbreviations once they become determinable.
        self._update_teams(row, clean_home, clean_away)

        # Track every player who appeared on each side (for historical reference).
        for p in clean_home:
            if p not in self.home_players:
                self.home_players.append(p)
        for p in clean_away:
            if p not in self.away_players:
                self.away_players.append(p)

        home = self.home_indicator(clean_home, row["player"])

        # Resolve time; fall back to last known time if the row is unparseable.
        time_val = self.convert_time(row["period"], row["elapsed"])
        if time_val is not None:
            self.last_time = time_val
        time_safe = time_val if time_val is not None else self.last_time

        # ---- SHOT ZONE ----
        # One of the fifteen spatial tokens (zones.py), replacing the old 2pt/3pt binary. The raw
        # ``type`` text stays the authority on point value; geometry only picks the zone within
        # the 2pt or 3pt family. Computed ONLY for shot rows — the old binary ran on every raw
        # row and produced a bogus type for turnovers, fouls and substitutions alike. Assists and
        # blocks are emitted from the shot row they belong to, so they share this one value.
        shot_zone = None
        if row["event_type"] == "shot":
            shot_zone = zones.zone_for(
                row.get("converted_x"), row.get("converted_y"),
                three=zones.marker_is_three(row.get("type")),
                shot_distance=row.get("shot_distance"),
            )

        # ---- ASSIST ----
        if pd.notna(row["assist"]) and str(row["assist"]).strip():
            assist_player = str(row["assist"]).strip()
            assist_home = self.home_indicator(clean_home, assist_player)
            # Assists ride on the shot row they set up (measured: every assist-bearing row in
            # 2002-03 is event_type="shot"), so shot_zone is set. The fallback covers the case
            # only in principle, and uses zones.py's own no-coordinates default rather than
            # inventing a token.
            assist_zone = shot_zone if shot_zone is not None else zones.zone_for(
                three=zones.marker_is_three(row.get("type")))
            events.append({
                "roster_home": clean_home,
                "roster_away": clean_away,
                "time": time_safe,
                "event": "assist",
                "player": assist_player,
                "type": assist_zone,
                "result": "score",
                "secondary_player": "none",
                "home/away": assist_home,
                "season": self.season,
                "playoff": 2 if self.playoff else 1,
            })

        has_block = pd.notna(row.get("block")) and str(row.get("block")).strip()

        # ---- SHOT ----
        if row["event_type"] == "shot":
            events.append({
                "roster_home": clean_home,
                "roster_away": clean_away,
                "time": time_safe,
                "event": "shot",
                "player": row["player"] if pd.notna(row["player"]) else "null",
                "type": shot_zone,
                "result": "blocked" if has_block else (row["result"] if pd.notna(row["result"]) else "null"),
                "secondary_player": "none",
                "home/away": home,
                "season": self.season,
                "playoff": 2 if self.playoff else 1,
            })

            if has_block:
                blocker = str(row["block"]).strip()
                block_home = self.home_indicator(clean_home, blocker)
                blocked_shooter = row["player"] if pd.notna(row["player"]) else "null"
                events.append({
                    "roster_home": clean_home,
                    "roster_away": clean_away,
                    "time": time_safe,
                    "event": "block",
                    "player": blocker,
                    "type": shot_zone,
                    "result": "block",
                    "secondary_player": blocked_shooter,
                    "home/away": block_home,
                    "season": self.season,
                    "playoff": 2 if self.playoff else 1,
                })

        # ---- FREE THROW (normalized under "shot") ----
        if row["event_type"] == "free throw":
            events.append({
                "roster_home": clean_home,
                "roster_away": clean_away,
                "time": time_safe,
                "event": "shot",
                "player": row["player"] if pd.notna(row["player"]) else "null",
                "type": "free throw",
                "result": row["result"] if pd.notna(row["result"]) else "null",
                "secondary_player": "none",
                "home/away": home,
                "season": self.season,
                "playoff": 2 if self.playoff else 1,
            })

        # ---- REBOUND ----
        if row["event_type"] == "rebound":
            # A bare "team rebound" carries no side at all. 22.6% of them sit at a period
            # boundary and 71% follow a free throw, so they are mostly bookkeeping rather than a
            # live board, and nothing in the row says which way the ball went -- inferring it
            # from the next team-bearing event gives an implausible 65/12 offensive split. They
            # stay dropped, as before. The playerless *typed* rebounds below do carry a side.
            if row["type"] == "team rebound":
                return events

            rebound_type = (
                "defensive" if row["type"] == "rebound defensive"
                else "offensive" if row["type"] == "rebound offensive"
                else "null"
            )
            # No player credited: a team rebound. The raw type still says which side got the
            # ball, so it becomes its own token rather than a player row named "null" -- which
            # is what the player head used to be trained on, ~11.9k times a season.
            rebounder = row["player"] if pd.notna(row["player"]) else None
            if rebounder is None and rebound_type in ("offensive", "defensive"):
                rebound_type = f"team {rebound_type}"
            events.append({
                "roster_home": clean_home,
                "roster_away": clean_away,
                "time": time_safe,
                "event": "rebound",
                "player": rebounder if rebounder is not None else "none",
                "type": rebound_type,
                "result": "cop" if rebound_type.endswith("defensive") else "null",
                "secondary_player": "none",
                "home/away": home,
                "season": self.season,
                "playoff": 2 if self.playoff else 1,
            })

        # ---- TURNOVER ----
        if row["event_type"] == "turnover":
            turnover_player = row["player"] if pd.notna(row["player"]) else "null"
            steal_player = row.get("steal")
            has_steal = pd.notna(steal_player) and str(steal_player).strip()

            if has_steal:
                steal_player = str(steal_player).strip()
                steal_home = self.home_indicator(clean_home, steal_player)
                events.append({
                    "roster_home": clean_home,
                    "roster_away": clean_away,
                    "time": time_safe,
                    "event": "turnover",
                    "player": steal_player,
                    "type": "steal",
                    "result": "steal",
                    "secondary_player": "none",
                    "home/away": steal_home,
                    "season": self.season,
                    "playoff": 2 if self.playoff else 1,
                })
                events.append({
                    "roster_home": clean_home,
                    "roster_away": clean_away,
                    "time": time_safe,
                    "event": "turnover",
                    "player": turnover_player,
                    "type": "steal",
                    "result": "cop",
                    "secondary_player": "none",
                    "home/away": home,
                    "season": self.season,
                    "playoff": 2 if self.playoff else 1,
                })
            else:
                turnover_type = self.determine_turnover_type(row.get("type"))
                if turnover_type is None:
                    return events
                events.append({
                    "roster_home": clean_home,
                    "roster_away": clean_away,
                    "time": time_safe,
                    "event": "turnover",
                    "player": turnover_player,
                    "type": turnover_type,
                    "result": "cop",
                    "secondary_player": "none",
                    "home/away": home,
                    "season": self.season,
                    "playoff": 2 if self.playoff else 1,
                })

        # ---- FOUL ----
        if row["event_type"] == "foul":
            foul_type = self.determine_foul_type(row.get("type"))
            if foul_type == SHOOTING_FOUL:
                # Split into "shooting 2pt" / "shooting 3pt" so the foul-type head learns the
                # real share of three-shot trips in game context. The label is precomputed over
                # the whole file (needs the *following* trip's `outof`); see _label_shooting_fouls.
                foul_type = row.get(_SHOOTING_LABEL_COL) or SHOOTING_2PT
            foul_result = self.determine_foul_result(foul_type)
            # Who got fouled, from the raw `opponent` column. Populated for 100% of non-technical
            # fouls (2022-23) and empty for 100% of technicals, which have no victim. It shares
            # the player embedding (encoder.encode_secondary_player delegates to player_vocab),
            # so every head sees the fouled player through history at no architectural cost --
            # and drawing fouls stops being a skill the model cannot represent.
            fouled = row.get("opponent")
            fouled = str(fouled).strip() if pd.notna(fouled) and str(fouled).strip() else "none"
            events.append({
                "roster_home": clean_home,
                "roster_away": clean_away,
                "time": time_safe,
                "event": "foul",
                "player": row["player"] if pd.notna(row["player"]) else "null",
                "type": foul_type,
                "result": foul_result,
                "secondary_player": fouled,
                "home/away": home,
                "season": self.season,
                "playoff": 2 if self.playoff else 1,
            })

        # ---- SUBSTITUTION ----
        if row["event_type"] == "substitution":
            entered = row["entered"]  # incoming player (off the bench)
            left = row["left"]        # outgoing player (on the active five)
            if pd.isna(entered) and pd.isna(left):
                return events
            # Convention: `player` is the OUTGOING player (predicted by the Player
            # model, sampled from the active roster) and `secondary_player` is the
            # INCOMING player (predicted by the Substitution model, sampled from the
            # bench). The sub row's lineups are post-substitution, so home/away keys
            # off the INCOMING player (who is on the resulting five); the outgoing
            # player has already left it. Fall back to the outgoing if no one entered.
            sub_ref = entered if pd.notna(entered) else left
            events.append({
                "roster_home": clean_home,
                "roster_away": clean_away,
                "time": time_safe,
                "event": "substitution",
                "player": left if pd.notna(left) else "null",
                "type": "substitution",
                "result": "substitution",
                "secondary_player": entered if pd.notna(entered) else "none",
                "home/away": self.home_indicator(clean_home, sub_ref),
                "season": self.season,
                "playoff": 2 if self.playoff else 1,
            })

        return events

    def parse_file(self, csv_path):
        self.events = []

        # Some scraped master files have a handful of malformed rows (a stray comma in a
        # free-text description yields an extra field — e.g. ~402 of 603k rows in 2016-17).
        # Skip those rather than abort the whole file; they are individual play rows and
        # dropping a scattered few does not meaningfully affect a season's games.
        df = pd.read_csv(csv_path, low_memory=False, na_values=["", " "], on_bad_lines="skip")
        df.rename(columns=lambda c: c.strip(), inplace=True)
        # ``team`` is kept (the only stable per-game team id; see _update_teams); ``date``
        # is also kept and consumed at the game boundary.
        df = df.drop(columns=[
            "game_id", "away_score", "home_score", "remaining_time",
            "play_length", "play_id", "possession",
            "original_x", "original_y", "description",
        ], errors="ignore")

        # Shooting fouls need the FOLLOWING trip's `outof` to know whether the fouled attempt
        # was a 2 or a 3, so the label is computed over the whole file before the row loop.
        df[_SHOOTING_LABEL_COL] = self._label_shooting_fouls(df)

        for _, row in df.iterrows():
            new_row = self.process_row(row)
            if new_row:
                for evt in new_row:
                    evt.setdefault("game_id", self.game_id)
                    # _end_event already carries the prior game's context; everything
                    # else inherits the current game's via setdefault.
                    evt.setdefault("game_date", self.game_date)
                    evt.setdefault("home_team", self.home_team)
                    evt.setdefault("away_team", self.away_team)
                self.events.extend(new_row)

        # Synthetic end event for the last game in this file.
        self.events.append(self._end_event())

        return df, pd.DataFrame(self.events)

    def _input_files(self):
        """Resolved raw files to process: *.csv, ignore-filtered, sorted, sliced.

        Filtering happens before slicing so --clean-start/--clean-end index into
        the meaningful master files (not the skipped samples).
        """
        files = sorted(
            f for f in os.listdir(self.data_path)
            if f.endswith(".csv") and not any(s in f for s in self.ignore)
        )
        return files[self.start:self.end]

    def run(self):
        """
        Loop through the resolved master files, parse them, and write cleaned
        events to ./data/season<YYYY>.csv (one file per season).

        Idempotent within a run: the first time a season file is written this run
        it is overwritten fresh (header + mode "w"); additional master files for
        the same season append. So re-running clean regenerates the season files
        instead of duplicating onto stale output.
        """
        files = self._input_files()

        os.makedirs("./data", exist_ok=True)
        written = set()  # season output paths already (re)started this run

        for fname in files:
            fpath = os.path.join(self.data_path, fname)

            # Reset per-file state (game_id persists for global uniqueness).
            self.first = True
            self.game_date = None
            self.home_team = None
            self.away_team = None
            self.home_players = []
            self.away_players = []
            self.last_home_five = []
            self.last_away_five = []

            temp_df = pd.read_csv(fpath, nrows=1)
            dataset_val = str(temp_df.iloc[0]["data_set"])
            # The data_set label carries the season's start year, but the prefix varies:
            # "2002-03 Regular Season" vs the 2019-20 bubble's "NBA 2019-2020 Regular Season".
            # Pull the first 4-digit year out of the label rather than assuming it's at index 0.
            year_match = re.search(r"\d{4}", dataset_val)
            if not year_match:
                raise ValueError(f"no season year found in data_set {dataset_val!r} ({fname})")
            self.season = int(year_match.group()) + 1

            _, cleaned_df = self.parse_file(fpath)

            out_path = f"./data/season{self.season}.csv"
            first_write = out_path not in written
            written.add(out_path)
            cleaned_df.to_csv(
                out_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )


if __name__ == "__main__":
    DataCleaner().run()
