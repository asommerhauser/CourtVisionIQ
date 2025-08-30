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

    DATA_PATH = "./RawData/MasterFiles"  # constant route to the data folder

    def __init__(self, start=0, end=None):
        self.start = start
        self.end = end

    def parse_file(self, csv_path):
        """
        Load CSV into a DataFrame and print the first few rows for testing.
        """
        df = pd.read_csv(csv_path)
        print(f"\n=== Preview of {os.path.basename(csv_path)} ===")
        print(df.head())   # show first 5 rows
        return df

    def run(self):
        """
        Loop through files in DATA_PATH and parse them.
        """
        files = sorted(os.listdir(self.DATA_PATH))
        files = files[self.start:self.end]  # respect start/end slice

        for fname in files:
            if fname.endswith(".csv"):
                fpath = os.path.join(self.DATA_PATH, fname)
                self.parse_file(fpath)

if __name__ == "__main__":
    cleaner = DataCleaner(start=0, end=None)  # go through all files
    cleaner.run()