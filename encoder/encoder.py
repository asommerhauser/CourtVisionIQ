from .vocab import Vocab

class Encoder:
    def __init__(self):
        self.player_vocab = Vocab(["PAD"], f"./vocabs/player_vocab.json")