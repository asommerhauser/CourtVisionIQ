from .vocab import Vocab
import ast

class Encoder:
    def __init__(self):
        self.player_vocab = Vocab(["PAD"], f"./vocabs/player_vocab.json")

    def encode_roster(self, roster):
        encoded_roster = set()
        roster = self.str_to_list(roster)
        for player in roster:
            encoded_roster.add(self.player_vocab.encode(player))

        return encoded_roster


    # ========================
    # --- Helper Functions ---
    # ========================

    def str_to_list(self, cell):
        if isinstance(cell, list):
            return cell
        if isinstance(cell, str):
            try:
                parsed = ast.literal_eval(cell)
            except Exception as e:
                raise ValueError(f"Failed to parse roster cell: {cell}") from e

            if not isinstance(parsed, list):
                raise TypeError(f"Expected list, got {type(parsed)}: {parsed}")
            return parsed

        raise TypeError(f"Unsupported roster cell type: {type(cell)}")