import tensorflow as tf
import pandas as pd
from keras import layers, models, Input
from config import MAX_SEQUENCE_LENGTH
from pathlib import Path
from encoder.encoder import Encoder
from models.roster_set_encoder import RosterSetEncoder, RosterEncoderParams

class EventTimeModel:
    """
    The event time model - more to come.
    """

    def __init__(self, encoder: Encoder, roster_parameters: RosterEncoderParams,
                 sequence_length = MAX_SEQUENCE_LENGTH, 
                 model_dim = 256, 
                 event_classes = 7,
                 path="./data"):
        
        # Setting up hyper-parameters
        # --- Model ---
        self.sequence_length = sequence_length
        self.model_dimensions = 256
        self.event_classes = 7

        # -- Roster Enocder ---
        self.ROSTER = roster_parameters
        self.roster_set_encoder = RosterSetEncoder(self.ROSTER)
        
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

        teammates_roster_ids = Input(
            shape=(self.sequence_length, self.ROSTER.roster_size),
            dtype="int32",
            name="opponents_roster_ids"
        )

        opponents_roster_ids = Input(
            shape=(self.sequence_length, self.ROSTER.roster_size),
            dtype="int32",
            name="opponents_roster_ids"
        )

        # Roster Set Transforming
        teammates_vec = layers.TimeDistributed(
            self.roster_encoder,
            name="teammates_roster_vec"
        )(teammates_roster_ids)

        opponents_vec = layers.TimeDistributed(
            self.roster_encoder,
            name="opponents_roster_vec"
        )(opponents_roster_ids)

        x = layers.Concatenate(name="concat_tokens_rosters")(
            [sequence_input, teammates_vec, opponents_vec]
        )

        # Project to model_dim
        x = layers.Dense(self.model_dim, activation="relu", name="fusion_projection")(x)

        # Placeholder Transformer Encoder
        x = layers.Dense(self.model_dim, activation="relu", name="token_mlp")(x)

        # --- Output Heads ---
        # Event head
        event_logits = layers.Dense(self.num_event_classes, activation="softmax", name="event_logits")(x)

        # Next-time regression head
        time_delta = layers.Dense(1, activation="linear", name="time_delta")(x)

        # Build model
        model = models.Model(inputs=sequence_input, outputs=[event_logits, time_delta], name="EventTimeModel")

        return model