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

        self.events = []
        self.output_columns = ["gameId","roster1","roster2","time","event","player","type","result","season","playoff"]

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
    
    def process_row(self, row):
        """
        Takes in a dataframe row and returns a list of one or more
        normalized event dictionaries.
        """

        events = []

        if pd.notna(row["assist"]) and str(row["assist"]).strip() != "":
            events.append({
                "roster1": [row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]],
                "roster2": [row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]],
                "time": "null",
                "event": "assist",
                "player": row["assist"],
                "type": str(row["points"]) if pd.notna(row["points"]) else "null",
                "result": "score",  # assist implies made basket
                "season": 1,
                "playoff": 1
            })

        if row["event_type"] == "shot":
            events.append({
                "roster1": [row["h1"], row["h2"], row["h3"], row["h4"], row["h5"]],
                "roster2": [row["a1"], row["a2"], row["a3"], row["a4"], row["a5"]],
                "time": "null",
                "event": "shot",
                "player": row["player"] if pd.notna(row["player"]) else "null",
                "type": str(row["points"]) if pd.notna(row["points"]) else "null",
                "result": row["result"] if pd.notna(row["result"]) else "null",
                "season": 1,
                "playoff": 1
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
                fpath = os.path.join(self.DATA_PATH, fname)
                df = self.parse_file(fpath)[1]
                print(df.head)
                print(df.columns)


if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run()