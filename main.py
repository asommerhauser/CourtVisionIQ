from encoder.encoder import Encoder
from pathlib import Path
import pandas as pd


def main():
    encoder = Encoder()

    data_dir = Path("./data")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir.resolve()}")
    csv_files = list(data_dir.glob("*.csv"))

    for csv_path in csv_files:
        print(f"Processing {csv_path.name}")
        df = pd.read_csv(csv_path)

        df = df[["teammates", "opponents"]]

        df["teammates_encoded"] = df["teammates"].apply(encoder.encode_roster)
        df["opponents_encoded"] = df["opponents"].apply(encoder.encode_roster)
        print(df)

if __name__ == "__main__":
    main()