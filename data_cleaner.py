import os
import pandas as pd

class DataCleaner:
    """
    The DataCleaner class is responsible for converting the raw NBA play-by-play data 
    into the format that the models will use to train and learn from.

    Parameters
    ----------
    start : int
        Index to begin processing files from. Defaults to 0 (start at the beginning).
    end : int
        Index to stop processing files at. If None, goes through the end.
    """

    DATA_PATH = "./RawData/MasterFiles"

    def __init__(self, start=0, end=None):
        self.start = start
        self.end = end
        self.season = 0
        self.playoff = 0
        self.home_players = []
        self.away_players = []
        self.first = True

        self.events = []
        self.output_columns = ["roster1","roster2","time","event","player","type","result","season","playoff"]

    # ----------------- HELPER FUNCTIONS -----------------

    def parse_file(self, csv_path):
        df = pd.read_csv(csv_path, low_memory=False, na_values=["", " "])

        # normalize column names
        df.rename(columns=lambda c: c.strip(), inplace=True)

        drop_cols = [
            "game_id", "away_score", "home_score", "remaining_time",
            "play_length", "play_id", "team", "outof", "possession", "shot_distance",
            "original_x", "original_y", "converted_x", "converted_y", "description"
        ]

        df = df.drop(columns=drop_cols, errors="ignore")

        for _, row in df.iterrows():
            new_row = self.process_row(row)
            if new_row:
                self.events.extend(new_row)
        
        self.events.append({
            "teammates": self.home_players,
            "opponents": self.away_players,
            "time": 0,
            "event": "end",
            "player": "end",
            "type": "end",
            "result": "end",
            "home/away": 0,
            "season": self.season,
            "playoff": self.playoff
        })

        cleaned_df = pd.DataFrame(self.events)

        return df, cleaned_df
    
    def convert_time(self, quarter, time_past):
        """
        Compute cumulative game time in seconds since tipoff.
        NBA: Q1–Q4 = 12:00 each; OT periods = 5:00 each.
        quarter: int-like (1,2,3,4,5=OT1,6=OT2,...)
        time_past: "H:MM:SS" (hours usually 0)
        """
        if pd.isna(quarter) or pd.isna(time_past):
            return None
        try:
            q = int(quarter)
            parts = str(time_past).strip().split(":")
            if len(parts) == 3:
                hh, mm, ss = map(int, parts)
            elif len(parts) == 2:  # fallback, "MM:SS"
                hh = 0
                mm, ss = map(int, parts)
            else:
                return None
        except Exception:
            return None

        # Base seconds through the start of this period
        if q > 4:
            base = (48 * 60) + ((q - 5) * 5 * 60)
        else:
            base = (q - 1) * 12 * 60

        # Add elapsed within the period
        return base + (hh * 3600) + (mm * 60) + ss
    
    def home_indicator(self, home_roster, player):
        """
        Given the home roster list and a player name,
        return 1 if the player is in the home roster,
        else return 0.
        """
        return 1 if player in home_roster else 0
    
    def parse_rosters(self, roster1, roster2, actor):
        """
        Given two rosters (roster1, roster2) and an actor (player name),
        return (teammates, opponents) where the actor is removed from
        their own roster and the rosters are returned as lists.
        """
        if actor in roster1:
            teammates = [p for p in roster1 if p != actor]
            opponents = list(roster2)
        elif actor in roster2:
            teammates = [p for p in roster2 if p != actor]
            opponents = list(roster1)
        else:
            teammates = list(roster1)
            opponents = list(roster2)
        return teammates, opponents

    def determine_turnover_type(self, data):
        """
        Map raw turnover 'type' text into a coarse turnover category.

        Returns one of:
        - 'violation'
        - 'error'
        - 'null'
        - 'unrecognized'
        """
        check_vio = [
            '3-second violation', 'shot clock', '8-second violation', 'lane violation', 'offensive goaltending',
            'palming', 'backcourt', '5-second violation', 'double dribble', 'discontinue dribble', 'illegal assist',
            'jump ball violation', 'offensive foul', 'illegal screen', 'basket from below', 'punched ball',
            'too many players', 'traveling', 'kicked ball'
        ]
        check_error = [
            'lost ball', 'out of bounds lost ball', 'step out of bounds',
            'bad pass', 'inbound'
        ]

        if pd.isna(data):
            return 'null'  # keep missing data
        data = str(data).strip().lower()

        if data == '' or data == 'null':
            return 'null'  # still keep
        elif data == 'no turnover':
            return None  # skip this one completely
        elif data in check_vio:
            return 'violation'
        elif data in check_error:
            return 'error'
        else:
            print("FLAG UNRECOGNIZED TURNOVER", data)
            return None  # skip unrecognized junk


    # -----------------------------------------------------
    
    def process_row(self, row):
        """
        Takes in a dataframe row and returns a list of one or more
        normalized event dictionaries.
        """
        events = []

        if row['event_type'] == "start of period" and row['period'] == 1:
            if str(row["data_set"])[-1] == "n":
                self.playoff = 0
            else:
                self.playoff = 1

            if not self.first:
                events.append({
                    "teammates": self.home_players,
                    "opponents": self.away_players,
                    "time": 0,
                    "event": "end",
                    "player": "end",
                    "type": "end",
                    "result": "end",
                    "home/away": 0,
                    "season": self.season,
                    "playoff": self.playoff
                })
            else:
                self.first = False

            self.home_players = []
            self.away_players = []

        home_five = [row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]]
        away_five = [row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]]

        for p in home_five:
            if pd.notna(p) and p not in self.home_players:
                self.home_players.append(p)
        for p in away_five:
            if pd.notna(p) and p not in self.away_players:
                self.away_players.append(p)

        rosters = self.parse_rosters(home_five, away_five, row["player"])

        home = self.home_indicator(home_five, row["player"])

        # Common computed values
        time_val = self.convert_time(row["period"], row["elapsed"])
        shot_type = ("3pt" if (pd.notna(row["type"]) and str(row["type"]).lower().startswith("3pt"))
                    else ("2pt" if pd.notna(row["type"]) else "null"))

        # ASSIST (only when present; implies made basket)
        if pd.notna(row["assist"]) and str(row["assist"]).strip() != "":
            # Parse rosters for the assist action
            assist_rosters = self.parse_rosters([row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]], [row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]], row["assist"])

            events.append({
                "teammates": assist_rosters[0],
                "opponents": assist_rosters[1],
                "time": time_val if time_val is not None else "null",
                "event": "assist",
                "player": row["assist"],
                "type": shot_type,      # 2pt/3pt
                "result": "score",
                "home/away": home,
                "season": self.season,
                "playoff": self.playoff
            })

        # BLOCK (when present; paired with a shot that becomes 'blocked')
        has_block = pd.notna(row.get("block")) and str(row.get("block")).strip() != ""

        if row["event_type"] == "shot":
            # SHOT (coerce result to 'blocked' if a block occurred)
            events.append({
                "teammates": rosters[0],
                "opponents": rosters[1],
                "time": time_val if time_val is not None else "null",
                "event": "shot",
                "player": row["player"] if pd.notna(row["player"]) else "null",
                "type": shot_type,  # 2pt/3pt
                "result": ("blocked" if has_block else (row["result"] if pd.notna(row["result"]) else "null")),
                "home/away": home,
                "season": self.season,
                "playoff": self.playoff
            })

            if has_block:
                block_rosters = self.parse_rosters([row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]], [row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]], row["block"])
                
                events.append({
                    "teammates": block_rosters[0],
                    "opponents": block_rosters[1],
                    "time": time_val if time_val is not None else "null",
                    "event": "block",
                    "player": row["block"],                                    # blocker
                    "type": (row["player"] if pd.notna(row["player"]) else "null"),  # victim (shooter), per old file
                    "result": "block",
                    "home/away": (0 if home == 1 else 1),
                    "season": self.season,
                    "playoff": self.playoff
                })

        # FREE THROW normalized under shot
        if row["event_type"] == "free throw":
            events.append({
                "teammates": rosters[0],
                "opponents": rosters[1],
                "time": time_val if time_val is not None else "null",
                "event": "shot",
                "player": row["player"] if pd.notna(row["player"]) else "null",
                "type": "free throw",
                "result": row["result"] if pd.notna(row["result"]) else "null",
                "home/away": home,
                "season": self.season,
                "playoff": self.playoff
            })

        # REBOUND
        if row["event_type"] == "rebound":
            # rebound type: offensive/defensive/etc.
            rebound_type = (
                "defensive" if row["type"] == "rebound defensive"
                else "offensive" if row["type"] == "rebound offensive"
                else "null"
            )

            # 'cop' flag matches old behavior: 'cop' if offensive, else 'null'
            cop = "cop" if (pd.notna(row["type"]) and str(row["type"]).lower() == "offensive") else "null"

            events.append({
                "teammates": rosters[0],
                "opponents": rosters[1],
                "time": time_val if time_val is not None else "null",
                "event": "rebound",
                "player": row["player"] if pd.notna(row["player"]) else "null",
                "type": rebound_type,   # offensive / defensive / etc.
                "result": cop,          # 'cop' or 'null' per old logic
                "home/away": home,
                "season": self.season,
                "playoff": self.playoff
            })

        # TURNOVER
        if row["event_type"] == "turnover":
            time_safe = time_val if time_val is not None else "null"

            turnover_player = row["player"] if pd.notna(row["player"]) else "null"
            steal_player = row.get("steal")
            has_steal = pd.notna(steal_player) and str(steal_player).strip() != ""

            # If a steal is credited, create a steal-side event
            if has_steal:
                steal_player = str(steal_player).strip()

                steal_rosters = self.parse_rosters(home_five, away_five, steal_player)
                steal_home = self.home_indicator(home_five, steal_player)

                # Steal perspective: same 'event' as old code (turnover),
                # but result='steal' marks this as the steal event.
                events.append({
                    "teammates": steal_rosters[0],
                    "opponents": steal_rosters[1],
                    "time": time_safe,
                    "event": "turnover",
                    "player": steal_player,
                    "type": "steal",      # keep simple; detailed cause not needed here
                    "result": "steal",    # steal marker
                    "home/away": steal_home,
                    "season": self.season,
                    "playoff": self.playoff
                })

                # Turnover perspective: ballhandler commits a turnover via steal
                events.append({
                    "teammates": rosters[0],
                    "opponents": rosters[1],
                    "time": time_safe,
                    "event": "turnover",
                    "player": turnover_player,
                    "type": "steal",      # turnover caused by a steal
                    "result": "cop",      # change of possession
                    "home/away": home,
                    "season": self.season,
                    "playoff": self.playoff
                })

            else:
                # No steal recorded -> classify as violation/error/etc
                turnover_type = self.determine_turnover_type(row.get("type"))

                if turnover_type is None:
                    return events  # skip this row completely

                events.append({
                    "teammates": rosters[0],
                    "opponents": rosters[1],
                    "time": time_safe,
                    "event": "turnover",
                    "player": turnover_player,
                    "type": turnover_type,  # 'violation', 'error', or 'null'
                    "result": "cop",
                    "home/away": home,
                    "season": self.season,
                    "playoff": self.playoff
                })

        return events

    def run(self):
        """
        Loop through files in DATA_PATH and parse them.
        """
        files = sorted(os.listdir(self.DATA_PATH))
        files = files[self.start:self.end]

        for fname in files:
            if fname.endswith(".csv"):
                # establish the path to the file
                fpath = os.path.join(self.DATA_PATH, fname)

                # reset the first flag
                self.first = True

                # peek first row to get season
                temp_df = pd.read_csv(fpath, nrows=1)
                dataset_val = str(temp_df.iloc[0]["data_set"])
                self.season = int(dataset_val[:4]) + 1

                # run the parser
                df = self.parse_file(fpath)[1]
                cols = ["teammates", "opponents", "event", "player", "type"]

                print(df[cols].head(100))   # first 10
                print(df[cols].tail(100))   # last 10


if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run()