from .vocab import Vocab
import ast
import pandas as pd
from pathlib import Path

from config import VOCAB_DIR, ROSTER_SIZE
from player_floor import PROTECTED, load_aliases, save_aliases

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

        # 3.2: below-floor players are aliased to anonymous slot tokens before they ever reach the
        # player vocab. The map lives beside the vocabs because it IS part of the language -- it is
        # loaded here, snapshotted into artifacts/<name>/vocabs/ with them, and fingerprinted by the
        # manifest, so a reloaded model aliases names exactly as its train did. Empty means no floor.
        #
        # Applying it HERE rather than at the call sites is what keeps the change small: every
        # encode_roster / encode_player / encode_secondary_player caller -- six heads' preprocess, the
        # simulator, the incremental input cache -- is untouched. It is safe to hold on the encoder
        # because the map is global and stateless, not per game; see player_floor's module docstring
        # for why one global map is sufficient.
        self.aliases: dict[str, str] = load_aliases(self.vocab_dir)

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

    def encode_roster(self, roster, size: int = ROSTER_SIZE) -> list[int]:
        """
        Encode a roster into a fixed-length ordered list of player token IDs.
        Length is exactly ``size`` (ROSTER_SIZE by default, BENCH_SIZE for a bench bundle):
        right-padded with PAD(0) if fewer players, truncated if more. Order is preserved
        as given -- the set encoder is permutation-invariant, so order carries no meaning,
        but it must stay aligned with the per-player scalars for the same row.
        """
        players = self.str_to_list(roster)
        ids = [self.player_vocab.encode(self.alias(p)) for p in players[:size]]
        pad_id = self.player_vocab.encode("PAD")
        if len(ids) < size:
            ids = ids + [pad_id] * (size - len(ids))
        return ids

    def alias(self, name):
        """The name this player is encoded under: himself, or his anonymous slot below the floor.

        Reserved tokens pass through untouched -- ``player`` legitimately holds ``"start"`` on a
        period-opening row, and aliasing one would move a fixed vocabulary id.
        """
        if not self.aliases or not isinstance(name, str) or name in PROTECTED:
            return name
        return self.aliases.get(name, name)

    def set_aliases(self, aliases: dict[str, str]) -> "Encoder":
        """Install an alias map (the vocab-rebuild path, before any name is encoded)."""
        self.aliases = dict(aliases or {})
        return self

    def encode_player(self, player) -> int:
        return self.player_vocab.encode(self.alias(player))

    def encode_event(self, event) -> int:
        return self.event_vocab.encode(event)

    def encode_type(self, type_code) -> int:
        return self.type_vocab.encode(type_code)

    def encode_result(self, result) -> int:
        return self.result_vocab.encode(result)

    def encode_season(self, season) -> int:
        return self.season_vocab.encode(season)

    def encode_secondary_player(self, player) -> int:
        # Shares the player vocab (the embedding is weight-tied), so it shares the aliasing too --
        # otherwise a below-floor assister would encode as himself here and as his slot in the roster.
        return self.player_vocab.encode(self.alias(player))

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
        # The alias map is part of the language, so it is written and snapshotted with the vocabs.
        # Writing it on every save_all (rather than only on the rebuild) is what makes
        # manifest.snapshot_vocabs pick it up for whichever head finishes first.
        if self.aliases:
            save_aliases(self.vocab_dir, self.aliases)

    def load_all(self) -> "Encoder":
        for v in self.vocabs.values():
            if v.path is not None and v.path.exists():
                v.load(v.path)
        # Reload the aliases too: a head constructed before the vocab rebuild wrote them would
        # otherwise hold an empty map and encode below-floor players under their own names, which is
        # the one way the heads could silently disagree about what a player id means.
        self.aliases = load_aliases(self.vocab_dir)
        return self

    def freeze_all(self) -> "Encoder":
        self.assert_aliases_absent()
        for v in self.vocabs.values():
            v.freeze()
        return self

    def assert_aliases_absent(self) -> None:
        """No aliased player may hold a row in the player vocab. Checked as the language is frozen.

        ``Vocab`` is append-only by design -- re-running a build over narrower data never renumbers or
        removes an id -- so a rebuild on top of a vocab written BEFORE the floor existed keeps every
        below-floor name, and the floor then does nothing at all. Nothing else would look wrong: the
        train runs, the table stays large, and the one symptom is an embedding table that did not
        shrink.

        The same check catches raising the floor without rebuilding, which is the likelier mistake.
        """
        if not self.aliases:
            return
        present = sorted(n for n in self.aliases if n in self.player_vocab.string_to_token)
        if present:
            raise ValueError(
                f"{len(present)} aliased players still hold their own rows in "
                f"{self.vocab_dir / 'player_vocab.json'} (e.g. {present[:3]}). The vocab is "
                f"append-only, so it predates this floor. Delete the vocabs and rebuild:\n"
                f"  rm {self.vocab_dir}/*.json && python train.py --full --name <name> "
                f"--rebuild-vocabs")

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
