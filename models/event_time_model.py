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

    def __init__(self, encoder: Encoder,
                 sequence_length = MAX_SEQUENCE_LENGTH, 
                 model_dim = 256, 
                 event_classes = 7,
                 path="./data"):
        
        # Setting up hyper-parameters
        # --- Model ---
        self.sequence_length = sequence_length
        self.model_dimensions = model_dim
        self.event_classes = event_classes
        
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

            df["teammates"] = df["teammates"].apply(self.encoder.encode_roster)
            df["opponents"] = df["opponents"].apply(self.encoder.encode_roster)
            df["event"] = df["event"].apply(self.encoder.encode_event)
            df["player"] = df["player"].apply(self.encoder.encode_player)
            df["type"] = df["type"].apply(self.encoder.encode_type)
            df["result"] = df["result"].apply(self.encoder.encode_result)
            df["season"] = df["season"].apply(self.encoder.encode_season)

            df["teammates_cur"] = df["teammates"].shift(-1)
            df["opponents_cur"] = df["opponents"].shift(-1)
            df["event_output"]     = df["event"].shift(-1)

            df["delta_time"] = df["time"] - df["time"].shift(1)
            df["delta_time"] = df["delta_time"].fillna(0)

            PAD_EVENT   = self.encoder.encode_event("PAD")
            PAD_PLAYER  = self.encoder.encode_player("PAD")
            PAD_TYPE    = self.encoder.encode_type("PAD")
            PAD_RESULT  = self.encoder.encode_result("PAD")
            PAD_SEASON  = self.encoder.encode_season("PAD")
            PAD_ROSTER  = self.encoder.encode_roster([])  # empty roster = PAD

            SEQ_LEN = self.sequence_length

            rows = []
            current_len = 0

            for _, row in df.iterrows():
                rows.append(row.to_dict())
                current_len += 1

                # END TOKEN HIT — PAD OUT TO SEQUENCE_LENGTH
                if row["event"] == self.encoder.encode_event("end"):
                    pad_needed = SEQ_LEN - current_len

                    if pad_needed > 0:
                        pad_block = pd.DataFrame({
                            "teammates":        [PAD_ROSTER] * pad_needed,
                            "opponents":        [PAD_ROSTER] * pad_needed,
                            "event":            [PAD_EVENT] * pad_needed,
                            "player":           [PAD_PLAYER] * pad_needed,
                            "type":             [PAD_TYPE] * pad_needed,
                            "result":           [PAD_RESULT] * pad_needed,
                            "season":           [PAD_SEASON] * pad_needed,
                            "teammates_cur":    [PAD_ROSTER] * pad_needed,
                            "opponents_cur":    [PAD_ROSTER] * pad_needed,
                            "event_output":     [PAD_EVENT] * pad_needed,
                            "delta_time":       [0.0] * pad_needed,
                            "time":             [row["time"]] * pad_needed,
                        })
                        rows.extend(pad_block.to_dict("records"))

                    # reset counter for next game
                    current_len = 0

            df = pd.DataFrame(rows)
            out_path = csv_path.with_name(csv_path.stem + "_padded.csv")
            df.to_csv(out_path, index=False)
            print(f"Saved padded file to {out_path}")
            print(df)

        # -- Roster Encoder ---
        ROSTER = RosterEncoderParams(
            roster_size=5,
            num_players=self.encoder.player_vocab.next_token,
            roster_dim=128,
            num_sab_layers=2,
            num_heads=4,
            d_ff=256,
            dropout=0.1,
        )
        self.roster_encoder = RosterSetEncoder(ROSTER)

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