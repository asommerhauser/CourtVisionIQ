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

        self.events = []
        self.output_columns = ["roster1","roster2","time","event","player","type","result","season","playoff"]

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

        # Common computed values
        time_val = self.convert_time(row["period"], row["elapsed"])
        shot_type = ("3pt" if (pd.notna(row["type"]) and str(row["type"]).lower().startswith("3pt"))
                    else ("2pt" if pd.notna(row["type"]) else "null"))

        # ASSIST (only when present; implies made basket)
        if pd.notna(row["assist"]) and str(row["assist"]).strip() != "":
            events.append({
                "roster1": [row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]],
                "roster2": [row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]],
                "time": time_val if time_val is not None else "null",
                "event": "assist",
                "player": row["assist"],
                "type": shot_type,      # 2pt/3pt
                "result": "score",
                "season": self.season,
                "playoff": self.playoff
            })

        # BLOCK (when present; paired with a shot that becomes 'blocked')
        has_block = pd.notna(row.get("block")) and str(row.get("block")).strip() != ""

        if row["event_type"] == "shot":
            # SHOT (coerce result to 'blocked' if a block occurred)
            events.append({
                "roster1": [row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]],
                "roster2": [row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]],
                "time": time_val if time_val is not None else "null",
                "event": "shot",
                "player": row["player"] if pd.notna(row["player"]) else "null",
                "type": shot_type,  # 2pt/3pt
                "result": ("blocked" if has_block else (row["result"] if pd.notna(row["result"]) else "null")),
                "season": self.season,
                "playoff": self.playoff
            })

            if has_block:
                events.append({
                    "roster1": [row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]],
                    "roster2": [row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]],
                    "time": time_val if time_val is not None else "null",
                    "event": "block",
                    "player": row["block"],                                    # blocker
                    "type": (row["player"] if pd.notna(row["player"]) else "null"),  # victim (shooter), per old file
                    "result": "block",
                    "season": self.season,
                    "playoff": self.playoff
                })

        # FREE THROW normalized under shot
        if row["event_type"] == "free throw":
            events.append({
                "roster1": [row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]],
                "roster2": [row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]],
                "time": time_val if time_val is not None else "null",
                "event": "shot",
                "player": row["player"] if pd.notna(row["player"]) else "null",
                "type": "free throw",
                "result": row["result"] if pd.notna(row["result"]) else "null",
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

                # peek first row to get season
                temp_df = pd.read_csv(fpath, nrows=1)
                dataset_val = str(temp_df.iloc[0]["data_set"])
                self.season = int(dataset_val[:4]) + 1

                # run the parser
                df = self.parse_file(fpath)[1]
                cols = ["time", "event", "player", "type", "result", "season", "playoff"]

                print(df[cols].head(10))   # first 10
                print(df[cols].tail(10))   # last 10


if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run()