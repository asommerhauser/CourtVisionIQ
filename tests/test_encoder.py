import pytest
import pandas as pd
from encoder.encoder import Encoder


def test_encode_roster_list_input_returns_set_of_ints(tmp_path):
    enc = Encoder(player_vocab_path=tmp_path / "player_vocab.json")

    out = enc.encode_roster(["A", "B", "C"])

    assert isinstance(out, set)
    assert len(out) == 3
    assert all(isinstance(x, int) for x in out)


def test_encode_roster_string_cell_parses_and_encodes(tmp_path):
    enc = Encoder(player_vocab_path=tmp_path / "player_vocab.json")

    out = enc.encode_roster("['A', 'B', 'C']")

    assert isinstance(out, set)
    assert len(out) == 3
    assert all(isinstance(x, int) for x in out)


def test_str_to_list_raises_on_bad_string(tmp_path):
    enc = Encoder(player_vocab_path=tmp_path / "player_vocab.json")

    with pytest.raises(ValueError):
        enc.encode_roster("not a list")


def test_pandas_apply_works(tmp_path):
    enc = Encoder(player_vocab_path=tmp_path / "player_vocab.json")

    df = pd.DataFrame({
        "teammates": ["['A','B','C']", "['A','D','E']"],
        "opponents": [["F","G","H"], ["I","J","K"]],
    })

    df["teammates_encoded"] = df["teammates"].apply(enc.encode_roster)
    df["opponents_encoded"] = df["opponents"].apply(enc.encode_roster)

    assert df["teammates_encoded"].apply(lambda s: isinstance(s, set)).all()
    assert df["opponents_encoded"].apply(lambda s: isinstance(s, set)).all()