import pytest
import pandas as pd
from encoder.encoder import Encoder
from config import ROSTER_SIZE


def test_encode_roster_list_input_returns_fixed_length_list(tmp_path):
    enc = Encoder(vocab_dir=tmp_path)

    out = enc.encode_roster(["A", "B", "C"])

    assert isinstance(out, list)
    assert len(out) == ROSTER_SIZE
    assert all(isinstance(x, int) for x in out)
    # 3 real players + PAD(0) fill
    pad = enc.encode_player("PAD")
    assert out[3:] == [pad] * (ROSTER_SIZE - 3)


def test_encode_roster_string_cell_parses_and_encodes(tmp_path):
    enc = Encoder(vocab_dir=tmp_path)

    out = enc.encode_roster("['A', 'B', 'C', 'D', 'E']")

    assert isinstance(out, list)
    assert len(out) == ROSTER_SIZE
    assert all(isinstance(x, int) for x in out)
    assert enc.encode_player("PAD") not in out  # full roster, no padding


def test_encode_roster_truncates_oversized(tmp_path):
    enc = Encoder(vocab_dir=tmp_path)
    out = enc.encode_roster(["A", "B", "C", "D", "E", "F"])
    assert len(out) == ROSTER_SIZE


def test_encode_roster_empty_is_all_pad(tmp_path):
    enc = Encoder(vocab_dir=tmp_path)
    out = enc.encode_roster([])
    assert out == [enc.encode_player("PAD")] * ROSTER_SIZE


def test_str_to_list_raises_on_bad_string(tmp_path):
    enc = Encoder(vocab_dir=tmp_path)

    with pytest.raises(ValueError):
        enc.encode_roster("not a list")


def test_pandas_apply_works(tmp_path):
    enc = Encoder(vocab_dir=tmp_path)

    df = pd.DataFrame({
        "roster_home": ["['A','B','C']", "['A','D','E']"],
        "roster_away": [["F","G","H"], ["I","J","K"]],
    })

    df["home_encoded"] = df["roster_home"].apply(enc.encode_roster)
    df["away_encoded"] = df["roster_away"].apply(enc.encode_roster)

    assert df["home_encoded"].apply(lambda s: isinstance(s, list) and len(s) == ROSTER_SIZE).all()
    assert df["away_encoded"].apply(lambda s: isinstance(s, list) and len(s) == ROSTER_SIZE).all()


def test_freeze_maps_unseen_to_unk(tmp_path):
    enc = Encoder(vocab_dir=tmp_path)
    known = enc.encode_player("A")
    enc.freeze_all()
    assert enc.encode_player("NEVER_SEEN") == enc.player_vocab.string_to_token["UNK"]
    assert enc.encode_player("A") == known  # known tokens still resolve


def test_save_and_load_roundtrip(tmp_path):
    enc = Encoder(vocab_dir=tmp_path)
    a = enc.encode_player("A")
    enc.save_all()

    enc2 = Encoder(vocab_dir=tmp_path)  # autoloads from disk
    assert enc2.encode_player("A") == a
