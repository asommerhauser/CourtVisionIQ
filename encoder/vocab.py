import json
from pathlib import Path
from typing import Optional

class Vocab:
    def __init__(self, token_init: list[str], path: Optional[str | Path] = None):
        """
        start_token:
            Allows reserving special tokens later (e.g. PAD=0, UNK=1).
        """
        self.string_to_token = {}
        self.token_to_string = {}
        self.next_token = 0
        self.path: Optional[Path] = Path(path) if path is not None else None

        if self.path is not None and self.path.exists():
            self.load(self.path)
        else:
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
    
    def save(self, path: Optional[str | Path] = None) -> None:
        """
        Save vocab to JSON. If no path is provided, uses the stored default path.
        """
        if path is not None:
            self.path = Path(path)

        if self.path is None:
            raise ValueError("No save path specified for Vocab.")

        data = {
            "string_to_token": self.string_to_token,
            "token_to_string": {str(k): v for k, v in self.token_to_string.items()},
            "next_token": self.next_token,
        }

        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str | Path] = None) -> None:
        """
        Load vocab from JSON. If no path is provided, uses the stored default path.
        """
        if path is not None:
            self.path = Path(path)

        if self.path is None or not self.path.exists():
            raise ValueError("No vocab file found to load.")

        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.string_to_token = data["string_to_token"]
        self.token_to_string = {int(k): v for k, v in data["token_to_string"].items()}
        self.next_token = data["next_token"]