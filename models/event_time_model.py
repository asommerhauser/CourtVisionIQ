import tensorflow as tf
import pandas as pd
from keras import layers, models, Input
from config import MAX_SEQUENCE_LENGTH
from pathlib import Path

class EventTimeModel:
    """
    The event time model - more to come.
    """

    def __init__(self, encoder, max_seq_len=MAX_SEQUENCE_LENGTH, model_dim=256, num_event_classes=7, path="./data"):
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim
        self.num_event_classes = num_event_classes
        self.data_dir = Path(path)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir.resolve()}")
        self.csv_files = list(self.data_dir.glob("*.csv"))
        self.encoder = encoder

    def preprocess(self):
        """
        Shape the cleaned data into a form that the model will use for training.
        """
        for csv_path in self.csv_files:
            print(f"Processing {csv_path.name}")
            df = pd.read_csv(csv_path)

            df = df[["teammates", "opponents"]]

            df["teammates_encoded"] = df["teammates"].apply(self.encoder.encode_roster)
            df["opponents_encoded"] = df["opponents"].apply(self.encoder.encode_roster)
            print(df)

    def model(self):
        """
        Build the keras model.
        """

        # Inputs
        sequence_input = Input(
            shape=(self.max_seq_len, self.model_dim),
            name="sequence_input"
        )

        # Transformer Encoder
        x = layers.Dense(self.model_dim, activation="relu")(sequence_input)

        # --- Output Heads ---
        # Event head
        event_logits = layers.Dense(self.num_event_classes, activation="softmax", name="event_logits")(x)

        # Next-time regression head
        time_delta = layers.Dense(1, activation="linear", name="time_delta")(x)

        # Build model
        model = models.Model(inputs=sequence_input, outputs=[event_logits, time_delta], name="EventTimeModel")

        return model