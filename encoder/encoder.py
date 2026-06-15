from .vocab import Vocab
import ast
import pandas as pd
from pathlib import Path

from config import VOCAB_DIR, ROSTER_SIZE

# Reserved special tokens, in the order that fixes their integer IDs.
# PAD must be id 0 (used for padding + roster slot masking).
# "none" = real event with no secondary participant (distinct from PAD = padded position).
SPECIALS = ["PAD", "UNK", "start", "end", "none"]
SEASON_SPECIALS = ["PAD", "UNK"]

# Cleaned-data column -> the vocab that encodes it.
PLAYER_FIELD = "player"
ROSTER_FIELDS = ("roster_home", "roster_away")


class Encoder:
    """
    The shared "language" every model speaks: one Vocab per categorical field,
    persisted to disk so token IDs are reusable and reproducible across models.

    Roster encoding returns a FIXED-length ordered int array (length ROSTER_SIZE,
    PAD-filled). Permutation invariance is provided by the Set Transformer, not by
    using a Python set, so a tensor-friendly fixed shape is what we emit here.
    """

    def __init__(self, vocab_dir: str | Path = VOCAB_DIR):
        self.vocab_dir = Path(vocab_dir)
        self.vocab_dir.mkdir(parents=True, exist_ok=True)

        self.player_vocab = Vocab(SPECIALS, self.vocab_dir / "player_vocab.json")
        self.event_vocab  = Vocab(SPECIALS, self.vocab_dir / "event_vocab.json")
        self.type_vocab   = Vocab(SPECIALS, self.vocab_dir / "type_vocab.json")
        self.result_vocab = Vocab(SPECIALS, self.vocab_dir / "result_vocab.json")
        self.season_vocab = Vocab(SEASON_SPECIALS, self.vocab_dir / "season_vocab.json")

    @property
    def vocabs(self) -> dict[str, Vocab]:
        return {
            "player": self.player_vocab,
            "event": self.event_vocab,
            "type": self.type_vocab,
            "result": self.result_vocab,
            "season": self.season_vocab,
        }

    # ==========================
    # --- Encoding Functions ---
    # ==========================

    def encode_roster(self, roster) -> list[int]:
        """
        Encode a roster into a fixed-length ordered list of player token IDs.
        Length is exactly ROSTER_SIZE: right-padded with PAD(0) if fewer players,
        truncated if more. Order is preserved as given.
        """
        players = self.str_to_list(roster)
        ids = [self.player_vocab.encode(p) for p in players[:ROSTER_SIZE]]
        pad_id = self.player_vocab.encode("PAD")
        if len(ids) < ROSTER_SIZE:
            ids = ids + [pad_id] * (ROSTER_SIZE - len(ids))
        return ids

    def encode_player(self, player) -> int:
        return self.player_vocab.encode(player)

    def encode_event(self, event) -> int:
        return self.event_vocab.encode(event)

    def encode_type(self, type_code) -> int:
        return self.type_vocab.encode(type_code)

    def encode_result(self, result) -> int:
        return self.result_vocab.encode(result)

    def encode_season(self, season) -> int:
        return self.season_vocab.encode(season)

    def encode_secondary_player(self, player) -> int:
        return self.player_vocab.encode(player)

    # =============================
    # --- Build / Persist / Lock ---
    # =============================

    def build_vocabs(self, csv_paths) -> "Encoder":
        """
        Canonical "build the language" step: stream cleaned CSV(s) and encode every
        field once so all tokens are registered (growing the vocabs), then save.
        Append-only: re-running over more data never renumbers existing IDs.
        """
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            for col in ROSTER_FIELDS:
                if col in df.columns:
                    df[col].apply(self.encode_roster)
            if "event" in df.columns:
                df["event"].apply(self.encode_event)
            if "player" in df.columns:
                df["player"].apply(self.encode_player)
            if "type" in df.columns:
                df["type"].apply(self.encode_type)
            if "result" in df.columns:
                df["result"].apply(self.encode_result)
            if "season" in df.columns:
                df["season"].apply(self.encode_season)
        self.save_all()
        return self

    def save_all(self) -> None:
        for v in self.vocabs.values():
            v.save()

    def load_all(self) -> "Encoder":
        for v in self.vocabs.values():
            if v.path is not None and v.path.exists():
                v.load(v.path)
        return self

    def freeze_all(self) -> "Encoder":
        for v in self.vocabs.values():
            v.freeze()
        return self

    # ========================
    # --- Helper Functions ---
    # ========================

    def str_to_list(self, cell):
        if isinstance(cell, list):
            return cell
        if cell is None or (isinstance(cell, float) and pd.isna(cell)):
            return []
        if isinstance(cell, str):
            try:
                parsed = ast.literal_eval(cell)
            except Exception as e:
                raise ValueError(f"Failed to parse roster cell: {cell}") from e

            if not isinstance(parsed, list):
                raise TypeError(f"Expected list, got {type(parsed)}: {parsed}")
            return parsed

        raise TypeError(f"Unsupported roster cell type: {type(cell)}")
