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

        return df

    def run(self):
        """
        Loop through files in DATA_PATH and parse them.
        """
        files = sorted(os.listdir(self.DATA_PATH))
        files = files[self.start:self.end]

        for fname in files:
            if fname.endswith(".csv"):
                fpath = os.path.join(self.DATA_PATH, fname)
                df = self.parse_file(fpath)
                print(df.columns)

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run()