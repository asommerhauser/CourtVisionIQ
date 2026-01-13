import tensorflow as tf
from keras import layers, models, Input
from config import MAX_SEQUENCE_LENGTH

class EventTimeModel:
    """
    The event time model - more to come.
    """

    def __init__(self, max_seq_len=MAX_SEQUENCE_LENGTH, model_dim=256, num_event_classes=7):
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim
        self.num_event_classes = num_event_classes

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