from encoder.encoder import Encoder
from pathlib import Path
from models.event_time_model import EventTimeModel
import pandas as pd


def main():
    encoder = Encoder()

    event_time_model = EventTimeModel(encoder)
    event_time_model.preprocess()

if __name__ == "__main__":
    main()