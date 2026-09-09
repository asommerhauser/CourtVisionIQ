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
# A bare "team rebound" carries no side. These bound the scan that recovers it: back to the shot
# it came off, forward to the next event that says who ended up with the ball.
_REBOUND_LOOKBEHIND = 6
_POSSESSION_LOOKAHEAD = 12
# Events that actually indicate possession. A foul is deliberately NOT one: it is usually
# committed by the team WITHOUT the ball, so counting it inverts the answer.
_POSSESSION_EVENTS = {"shot", "free throw", "turnover"}
# Private column carrying the recovered side from parse_file into process_row. Never emitted.
_TEAM_REBOUND_LABEL_COL = "_team_rebound_side"
# Raw rows that end a possession sequence — never scan a lookahead/lookbehind across one.
_BOUNDARY_EVENTS = {"start of period", "end of period"}

# The raw h1..h5 / a1..a5 snapshots do not agree with the substitution rows. Two distinct
# faults, both measured over 200 games of 2022-23:
#
#   * a STALE snapshot. A free-throw row that follows interleaved substitution rows carries the
#     PRE-substitution five, so the lineup appears to flip and flip back over three rows. About
#     10.5 such flickers a game -- confirmed in the raw master file itself (game 22200001, rows
#     86-91: free throw, rebound, sub, sub, sub, free throw, where the second free throw still
#     names the outgoing player). Read naively, these are ~21 phantom lineup changes a game
#     against 46.5 real substitutions.
#   * a MISSING substitution row. 7.8 lineup changes a game are real and permanent but have no
#     substitution row anywhere near them. Almost all are quarter-break changes: the raw file
#     records the new lineup as a changed snapshot on the "start of period" row and nothing else,
#     so the five the second quarter opens with is one the event stream never accounts for.
#
# _repair_fives resolves both by carrying a running five, updated in place by substitution rows
# and resynced to the snapshot only when that snapshot is STABLE -- the next raw row of the same
# game carries the same two sets. A one-row flicker is never stable, so it is ignored; a real
# change persists, so it is adopted, and the swap is emitted as a substitution row of its own.
# Residual disagreement between the running five and the snapshot falls from 156.6 rows a game
# to 10.7, and those are the stale rows themselves: the five is right and the column is wrong.
#
# Private columns carrying the repaired lineups and the recovered substitutions from parse_file
# into process_row. Never emitted.
_HOME_FIVE_COL = "_repaired_home_five"
_AWAY_FIVE_COL = "_repaired_away_five"
_DERIVED_SUBS_COL = "_derived_subs"

# The cleaned-data schema, and the ENFORCED contract: parse_file checks every emitted event
# against it and raises on a mismatch. This used to be a per-instance ``self.output_columns``
# that nothing ever read, so it drifted out of date silently -- and a stale third copy of the
# schema is worse than none. Ordering matters: it is the column order of data/season<YYYY>.csv.
OUTPUT_COLUMNS = (
    "game_id", "roster_home", "roster_away", "time", "event",
    "player", "type", "result", "secondary_player", "home/away", "season", "playoff",
    "game_date", "home_team", "away_team",
)


# Every raw column that holds a player name. Trimmed on read: see parse_file.
_PLAYER_COLUMNS = (
    "player", "assist", "block", "steal", "entered", "left", "opponent", "away", "home",
    "h1", "h2", "h3", "h4", "h5", "a1", "a2", "a3", "a4", "a5",
)


# Raw rows whose ``team`` column is NOT the team of the player named in ``player``, so they must
# never bind a side in ``_update_teams``. A jump ball credits the team that won the tip while
# naming one of the two opposing jumpers.
_TEAM_AGNOSTIC_EVENTS = {"jump ball"}


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

        **Jump balls do not bind.** On a jump-ball row the ``team`` column is the team that
        *won the tip*, not the team of the player named in ``player`` -- the two jumpers are
        opponents by definition, so the row credits one player and the other's team about half
        the time. Binding from it collapsed both sides onto one abbreviation in **~47% of games
        in every era** (615/1320 in 2022-23, 572/1277 in 2002-03, 638/1314 in 2012-13), because
        the jump ball bound one side wrongly and the next action row bound the other side to the
        same string. Excluding them resolves every game correctly, checked against the majority
        abbreviation over shot rows by home players: 0 mismatches in all three eras.
        """
        if self.home_team is not None and self.away_team is not None:
            return
        if str(row.get("event_type") or "").strip() in _TEAM_AGNOSTIC_EVENTS:
            return
        player = row.get("player")
        team = row.get("team")
        if pd.isna(player) or pd.isna(team):
            return
        team = str(team).strip()
        # The two sides cannot share an abbreviation. A guard rather than a fix -- excluding
        # jump balls is what actually resolves it -- but it makes the corrupt state
        # unrepresentable, and it is the check that would have caught this the first time.
        if self.home_team is None and player in clean_home and team != self.away_team:
            self.home_team = team
        elif self.away_team is None and player in clean_away and team != self.home_team:
            self.away_team = team

    def _side_of_team(self, team):
        """Map a raw team abbreviation to "home"/"away", or None if it cannot be resolved.

        Uses the abbreviations ``_update_teams`` resolves from the first action rows. Measured on
        2022-23: both are known before the first timeout in every game, so None is a guard, not a
        path the data actually takes.
        """
        if pd.isna(team) or not str(team).strip():
            return None
        team = str(team).strip()
        if team == self.home_team:
            return "home"
        if team == self.away_team:
            return "away"
        return None

    def determine_turnover_type(self, data):
        """
        Map raw turnover 'type' text to a coarse category.
        Returns 'violation', 'error', 'null', or None (skip the event entirely).
        """
        check_vio = {
            '3-second violation', 'shot clock', '8-second violation', 'lane violation',
            'offensive goaltending', 'palming', 'backcourt', '5-second violation',
            'double dribble', 'discontinue dribble', 'illegal assist',
            'jump ball violation', 'illegal screen',
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
        if data == 'offensive foul':
            # An offensive foul already emits a foul row (type "offensive", result "cop"), and
            # the raw data pairs the two for 100% of the 5,063 in 2022-23. Emitting both made the
            # sim reproduce a two-row grammar; the box score counts the turnover from the foul.
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
    def _repair_fives(df):
        """Repaired on-court fives per raw row, plus the substitutions the raw data omits.

        Returns ``(home_fives, away_fives, derived_subs)``, each a per-row list. ``derived_subs``
        holds ``(outgoing, incoming, side)`` triples for the rows where a real lineup change had
        no substitution row of its own; every other row carries an empty list.

        See the note above ``_HOME_FIVE_COL`` for why this exists. The rule, per row:

          * the first row of a game seeds the running five from its own snapshot;
          * a ``substitution`` row applies ``left -> entered`` **in place**, so slot order is
            stable for the whole game and the roster-parallel per-player features stay aligned
            to the same slot from tip-off;
          * any other row adopts its snapshot only when the snapshot is *stable* -- the next raw
            row of the same game carries the same two sets -- **and does not contradict a
            substitution made at this same instant**.

        That second condition is not belt-and-braces, it is the whole difficulty. A dead-ball
        substitution sequence is written as ``free throw, sub, sub, sub, free throw, free throw``,
        and *both* trailing free throws carry the pre-substitution five. Two stale rows in a row
        are "stable" by the first test alone, so stability on its own resyncs backwards, undoes
        three real substitutions, and then emits three more spurious ones when the next live row
        restores them -- six phantom substitutions from one real trip. Substitution rows are
        authoritative: a snapshot that puts a player back on the floor at the same instant a
        substitution row took them off is stale, however many rows repeat it.

        A substitution whose outgoing player is not on the running five resyncs from the snapshot
        rather than being dropped, so a single bad row cannot desync the rest of the game. It
        happens 0.01 times a game once the stability rule is in place, against 2.79 without it.
        """
        n = len(df)
        if n == 0:
            return [], [], []

        cols = {c: df[c].tolist() if c in df else [None] * n
                for c in ("event_type", "period", "elapsed", "entered", "left",
                          "h1", "h2", "h3", "h4", "h5", "a1", "a2", "a3", "a4", "a5")}

        def snapshot(i):
            home = [cols[c][i] for c in ("h1", "h2", "h3", "h4", "h5")]
            away = [cols[c][i] for c in ("a1", "a2", "a3", "a4", "a5")]
            return ([p for p in home if pd.notna(p)], [p for p in away if pd.notna(p)])

        def game_start(i):
            return (str(cols["event_type"][i] or "").strip() == "start of period"
                    and str(cols["period"][i]).strip() in ("1", "1.0"))

        def instant(i):
            return (str(cols["period"][i]), str(cols["elapsed"][i]))

        home_fives, away_fives, derived = [], [], []
        run_home, run_away = [], []
        # player -> the instant a substitution row took them off the floor. A snapshot at that
        # same instant naming them is the stale one, not the substitution.
        removed_at = {}

        def resync(five, now, side, now_at, subs):
            """Move ``five`` onto ``now``, recording every transition it takes to get there.

            Every path that changes the running five goes through here, so the five can never
            change without a substitution to explain it. That is exactly what the measurement
            pass checks by folding the substitutions forward from the opening lineup, and the
            paths that skipped it are where the last disagreements lived.

            Unequal counts are a lineup that changed SIZE, which happens when a raw substitution
            row names only one side of the swap -- four null ``entered`` and two null ``left`` a
            season. They are carried as substitutions with ``none`` standing in for the half the
            data never gave, the same sentinel the raw substitution path uses.
            """
            gone = [p for p in five if p not in now]
            arrived = [p for p in now if p not in five]
            for out, inc in zip(gone, arrived):
                five[five.index(out)] = inc
                removed_at[out] = now_at
                removed_at.pop(inc, None)
                subs.append((out, inc, side))
            for out in gone[len(arrived):]:
                five.remove(out)
                removed_at[out] = now_at
                subs.append((out, "none", side))
            for inc in arrived[len(gone):]:
                five.append(inc)
                removed_at.pop(inc, None)
                subs.append(("none", inc, side))

        snap = snapshot(0)
        for i in range(n):
            event = str(cols["event_type"][i] or "").strip()
            here = snap
            snap = snapshot(i + 1) if i + 1 < n else here
            now_at = instant(i)
            subs = []

            if game_start(i) or not (run_home or run_away):
                run_home, run_away = list(here[0]), list(here[1])
                removed_at = {}
            elif event == "substitution":
                out, inc = cols["left"][i], cols["entered"][i]
                placed = False
                # A substitution is only applicable if the outgoing player is on the floor and
                # the incoming one is not. The second half is not pedantry: 2002-03 has rows like
                # "Gerald Wallace out, Jim Jackson in" where Jim Jackson is already playing and
                # the snapshot shows the real arrival was Doug Christie -- the `entered` column
                # simply names the wrong man. Applied blindly it puts one player in two slots,
                # and every later comparison is done by membership, so the five then grows to six
                # and never recovers: 12 games in 2002-03, one of them for 148 rows.
                if pd.notna(out) and pd.notna(inc) and inc not in run_home and inc not in run_away:
                    for five in (run_home, run_away):
                        if out in five:
                            five[five.index(out)] = inc
                            removed_at[out] = now_at
                            removed_at.pop(inc, None)
                            placed = True
                            break
                if not placed:
                    # The row contradicts the floor, so its pairing cannot be trusted. Take the
                    # snapshot and describe the transition it implies; the caller drops the raw
                    # pairing in favour of these, or the same change is told twice, once wrongly.
                    resync(run_home, here[0], "home", now_at, subs)
                    resync(run_away, here[1], "away", now_at, subs)
            else:
                # Adopt the snapshot only if it survives the next row -- a flicker never does.
                # The last row of a game has no next row inside it, so it is taken as stable.
                stable = (i + 1 >= n or game_start(i + 1)
                          or (set(snap[0]) == set(here[0]) and set(snap[1]) == set(here[1])))
                if stable:
                    for five, now, side in ((run_home, here[0], "home"),
                                            (run_away, here[1], "away")):
                        if any(removed_at.get(p) == now_at for p in now if p not in five):
                            continue        # this snapshot is the stale half of a live trip
                        resync(five, now, side, now_at, subs)

            home_fives.append(list(run_home))
            away_fives.append(list(run_away))
            derived.append(subs)
        return home_fives, away_fives, derived

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

    @staticmethod
    def _label_team_rebounds(df):
        """Side for each bare ``team rebound`` row: a team token, or None meaning drop it.

        Two thirds of these rows are not rebounds. 6,336 of 2022-23's 9,374 follow a missed free
        throw that was **not the last of its trip** (6,078 are literally "missed 1 of 2"): the
        ball is dead, the shooter shoots again, and the row is bookkeeping. Emitting those as
        rebounds would inject ~6.3k phantom boards a season into the head whose entire job is
        the offensive/defensive split.

        The rest are real -- a team board off a missed field goal (2,698) or a missed last free
        throw (327) -- and the raw type simply does not record which way the ball went. It is
        recoverable: the next event that actually indicates possession (a shot, a free throw or
        a turnover -- **not** a foul, which is usually committed by the team without the ball,
        and not a substitution or timeout) names the team that ended up with it. Validated
        against the 11,884 playerless rebounds whose side IS recorded, in the same structural
        position: **99.8% correct** where decidable, 0.6% undecidable.
        """
        events = df["event_type"].tolist() if "event_type" in df else []
        types = df["type"].tolist() if "type" in df else [None] * len(events)
        teams = df["team"].tolist() if "team" in df else [None] * len(events)
        results = df["result"].tolist() if "result" in df else [None] * len(events)
        nums = df["num"].tolist() if "num" in df else [None] * len(events)
        outofs = df["outof"].tolist() if "outof" in df else [None] * len(events)

        def _blank(v):
            return v is None or pd.isna(v) or not str(v).strip()

        def _int(v):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return None

        labels = [None] * len(events)
        for i, event in enumerate(events):
            if event != "rebound" or str(types[i] or "").strip() != "team rebound":
                continue

            # 1) Was a live rebound even possible? Only after a missed field goal or a missed
            #    LAST free throw -- the same condition the controller calls pending_rebound.
            off_team = None
            for k in range(i - 1, max(i - _REBOUND_LOOKBEHIND, -1), -1):
                if events[k] in _BOUNDARY_EVENTS:
                    break
                if events[k] == "shot":
                    if results[k] == "missed" and not _blank(teams[k]):
                        off_team = teams[k]
                    break
                if events[k] == "free throw":
                    last = _int(nums[k]) == _int(outofs[k])
                    if results[k] == "missed" and last and not _blank(teams[k]):
                        off_team = teams[k]
                    break
            if off_team is None:
                continue

            # 2) Which side got it? The next event that indicates possession.
            for j in range(i + 1, min(i + _POSSESSION_LOOKAHEAD, len(events))):
                if events[j] in _BOUNDARY_EVENTS:
                    break
                if events[j] in _POSSESSION_EVENTS and not _blank(teams[j]):
                    labels[i] = ("team offensive" if teams[j] == off_team
                                 else "team defensive")
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
            # Through the repaired columns, so the start row and every row after it read the
            # same five. At a game start the two agree by construction; going through one of
            # them is what keeps that true if the seeding rule ever changes.
            seed_home = row.get(_HOME_FIVE_COL)
            seed_away = row.get(_AWAY_FIVE_COL)
            self.last_home_five = list(seed_home) if seed_home is not None else self._clean_five(
                [row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]]
            )
            self.last_away_five = list(seed_away) if seed_away is not None else self._clean_five(
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
        # Repaired by _repair_fives, not read off this row: the raw snapshot is stale on roughly
        # eight rows a game (see the note above _HOME_FIVE_COL). The raw columns remain the
        # fallback for a frame that never went through the pre-pass.
        clean_home = row.get(_HOME_FIVE_COL)
        clean_away = row.get(_AWAY_FIVE_COL)
        if clean_home is None or clean_away is None:
            clean_home = self._clean_five([row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]])
            clean_away = self._clean_five([row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]])
        clean_home = list(clean_home)
        clean_away = list(clean_away)

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

        # ---- SUBSTITUTIONS THE RAW DATA OMITS ----
        # A lineup change that persists but carries no substitution row -- 7.8 a game, almost all
        # of them at a period start, where quarter-break changes are recorded as a changed
        # snapshot and nothing else (see the note above _HOME_FIVE_COL). Emitted before this
        # row's own events, because the five they produce is the five this row is played with.
        # Without them the sequence contains a change of personnel that no event explains, which
        # the controller could never reproduce.
        #
        # Each carries its own PROGRESSIVE five, not the finished one. Three changes at a period
        # break are three rows, and a row that names one swap while its roster shows all three
        # already made is not a state any consumer can fold forward -- it is what the gate's
        # fold-the-substitutions check catches, and it disagreed on 409 rows over 82 games before
        # this. So the swaps are undone back to the five this row started from, then reapplied
        # one at a time.
        derived = list(row.get(_DERIVED_SUBS_COL) or ())
        if derived:
            step_home, step_away = list(clean_home), list(clean_away)
            for outgoing, incoming, side in reversed(derived):
                five = step_home if side == "home" else step_away
                if outgoing == "none":              # nobody left: undo by removing the arrival
                    if incoming in five:
                        five.remove(incoming)
                elif incoming == "none":            # nobody came on: undo by restoring the leaver
                    five.append(outgoing)
                elif incoming in five:
                    five[five.index(incoming)] = outgoing
            for outgoing, incoming, side in derived:
                five = step_home if side == "home" else step_away
                if outgoing == "none":
                    five.append(incoming)
                elif incoming == "none":
                    if outgoing in five:
                        five.remove(outgoing)
                elif outgoing in five:
                    five[five.index(outgoing)] = incoming
                events.append({
                    "roster_home": list(step_home),
                    "roster_away": list(step_away),
                    "time": time_safe,
                    "event": "substitution",
                    "player": outgoing,
                    "type": "substitution",
                    "result": "substitution",
                    "secondary_player": incoming,
                    "home/away": 1 if side == "home" else 2,
                    "season": self.season,
                    "playoff": 2 if self.playoff else 1,
                })

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

        # ---- TIMEOUT ----
        # A timeout is the single most common dead ball, and without it the sim has no way to
        # substitute after a made basket -- the largest reason its rotations look nothing like a
        # real game. The calling team rides in `type` (a seventh conditional type head predicts
        # it); no player is involved. The raw `team` column is populated for 100% of timeouts and
        # both abbreviations are always resolved by the time one appears, so nothing is dropped.
        if row["event_type"] == "timeout":
            side = self._side_of_team(row.get("team"))
            if side is None:
                return events
            events.append({
                "roster_home": clean_home,
                "roster_away": clean_away,
                "time": time_safe,
                "event": "timeout",
                "player": "none",
                "type": side,
                "result": "none",
                "secondary_player": "none",
                "home/away": 1 if side == "home" else 2,
                "season": self.season,
                "playoff": 2 if self.playoff else 1,
            })

        # ---- REBOUND ----
        if row["event_type"] == "rebound":
            # A bare "team rebound" carries no side. Two thirds of them are not rebounds at all
            # (the dead-ball row between free throws in a trip) and are dropped; for the rest the
            # side is recovered from what happens next. See _label_team_rebounds.
            if row["type"] == "team rebound":
                side = row.get(_TEAM_REBOUND_LABEL_COL)
                if not side or pd.isna(side):
                    return events
                events.append({
                    "roster_home": clean_home,
                    "roster_away": clean_away,
                    "time": time_safe,
                    "event": "rebound",
                    "player": "none",
                    "type": side,
                    "result": "cop" if side.endswith("defensive") else "null",
                    "secondary_player": "none",
                    "home/away": home,
                    "season": self.season,
                    "playoff": 2 if self.playoff else 1,
                })
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
                # ONE row, not two. The stealer rides in secondary_player exactly as a block row
                # carries the blocked shooter. The pair encoding made the event head learn a
                # two-row grammar the controller then had to reproduce exactly -- a whole class
                # of drift between cleaner and sim, for 19,167 duplicated rows a season.
                events.append({
                    "roster_home": clean_home,
                    "roster_away": clean_away,
                    "time": time_safe,
                    "event": "turnover",
                    "player": turnover_player,
                    "type": "steal",
                    "result": "cop",
                    "secondary_player": str(steal_player).strip(),
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

        # ---- STANDALONE TECHNICAL ----
        # The raw data files defensive three seconds, double technicals and coach technicals
        # under their own event_type, so the cleaner never saw them at all. They are ordinary
        # technical fouls; the 21% with no player named (mostly coach technicals) have no actor
        # to attribute and stay dropped.
        if row["event_type"] == "technical foul":
            if pd.isna(row["player"]) or not str(row["player"]).strip():
                return events
            events.append({
                "roster_home": clean_home,
                "roster_away": clean_away,
                "time": time_safe,
                "event": "foul",
                "player": str(row["player"]).strip(),
                "type": "technical",
                "result": self.determine_foul_result("technical"),
                "secondary_player": "none",
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
            # _repair_fives only reconstructs a substitution row's transition when the row names
            # a player who is not on the floor, and it has already emitted what actually
            # happened. Emitting the raw pairing as well would tell the same change twice, once
            # with a name the lineup does not contain.
            if derived:
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
                # "none", not "null": pandas reads "null" back as NaN with its default NA list,
                # so data_loading.py:53 destroys the sentinel before the vocabulary is built --
                # "null" appears in none of the five frozen vocabularies, in any column. Here it
                # decides who left the floor, so it has to survive.
                "player": left if pd.notna(left) else "none",
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

        # Trim every player-valued column before anything reads one. 2002-03 spells Nene as
        # "Nene " in `entered`/`left` and "Nene" in the h1..h5 snapshots, so the two never match:
        # folding the substitutions desynced the on-court five for the rest of every Denver game,
        # and the frozen 2.0 player vocabulary carried "Nene" and "Nene " as two separate players
        # with two separate embeddings. One trim here, before the pre-passes and the row loop, is
        # the only place that cannot drift -- a dozen call sites each stripping their own would.
        for col in _PLAYER_COLUMNS:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].str.strip()

        # Shooting fouls need the FOLLOWING trip's `outof` to know whether the fouled attempt
        # was a 2 or a 3, so the label is computed over the whole file before the row loop.
        df[_SHOOTING_LABEL_COL] = self._label_shooting_fouls(df)
        # Bare team rebounds record no side; recover it (or mark the row for dropping).
        df[_TEAM_REBOUND_LABEL_COL] = self._label_team_rebounds(df)
        # The raw on-court snapshots flicker and omit substitutions; repair both before the row
        # loop, since deciding whether a snapshot is real needs the row after it.
        home_fives, away_fives, derived_subs = self._repair_fives(df)
        df[_HOME_FIVE_COL] = pd.Series(home_fives, index=df.index, dtype=object)
        df[_AWAY_FIVE_COL] = pd.Series(away_fives, index=df.index, dtype=object)
        df[_DERIVED_SUBS_COL] = pd.Series(derived_subs, index=df.index, dtype=object)

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

        self._check_schema(self.events)
        return df, pd.DataFrame(self.events, columns=list(OUTPUT_COLUMNS))

    @staticmethod
    def _check_schema(events):
        """Every emitted event must carry exactly OUTPUT_COLUMNS — no more, no less.

        Cheap insurance at exactly the moment it matters: 2.0 changes the emitted schema in five
        separate branches before a single re-clean, and a missing key would otherwise surface as
        a silent all-NaN column in data/season<YYYY>.csv.
        """
        expected = set(OUTPUT_COLUMNS)
        for event in events:
            keys = set(event)
            if keys != expected:
                raise ValueError(
                    f"emitted event does not match the cleaned schema: "
                    f"missing={sorted(expected - keys)} unexpected={sorted(keys - expected)} "
                    f"in {event!r}"
                )

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
