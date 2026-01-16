class Vocab:
    def __init__(self, token_init: list[str]):
        """
        start_token:
            Allows reserving special tokens later (e.g. PAD=0, UNK=1).
        """
        self.string_to_token = {}
        self.token_to_string = {}
        self.next_token = 0
        for token in token_init:
            self.encode(token)

    def encode(self, value: str) -> int:
        """
        Convert a string to its token.
        If the string is new, add it to the vocabulary.
        """
        if value not in self.string_to_token:
            token = self.next_token
            self.string_to_token[value] = token
            self.token_to_string[token] = value
            self.next_token += 1
        return self.string_to_token[value]

    def decode(self, token: int) -> str:
        """
        Convert a token back to its string.
        Raises KeyError if token does not exist.
        """
        return self.token_to_string[token]

    def __len__(self) -> int:
        """
        Number of entries in the vocabulary.
        """
        return len(self.string_to_token)