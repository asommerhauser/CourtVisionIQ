from .vocab import Vocab
import ast
from pathlib import Path

class Encoder:
    def __init__(self):
        self.player_vocab = Vocab(["PAD", "start", "end"], "/vocabs/player_vocab.json")
        self.event_vocab = Vocab(["PAD", "start", "end"], "/vocabs/event_vocab.json")
        self.type_vocab = Vocab(["PAD", "start", "type"], "/vocab/type_vocab")
        self.result_vocab = Vocab(["PAD", "start", "end"], "/vocabs/result_vocab")
        self.season_vocab = Vocab(["PAD"], "/vocabs/season_vocab")

    # ==========================
    # --- Encoding Functions ---
    # ==========================

    def encode_roster(self, roster):
        encoded_roster = set()
        roster = self.str_to_list(roster)
        for player in roster:
            encoded_roster.add(self.player_vocab.encode(player))
        return encoded_roster

    def encode_player(self, player):
        return self.player_vocab.encode(player)

    def encode_event(self, event):
        return self.event_vocab.encode(event)

    def encode_type(self, type_code):
        return self.type_vocab.encode(type_code)

    def encode_result(self, result):
        return self.result_vocab.encode(result)

    def encode_season(self, season):
        return self.season_vocab.encode(season)
    

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