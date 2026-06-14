import json
import pytest

from encoder.vocab import Vocab


def test_init_with_token_init_sets_reserved_tokens():
    v = Vocab(["PAD", "UNK"])
    assert v.encode("PAD") == 0
    assert v.encode("UNK") == 1
    assert len(v) == 2


def test_encode_assigns_new_tokens_sequentially():
    v = Vocab(["PAD"])
    a = v.encode("A")
    b = v.encode("B")
    c = v.encode("C")

    assert a == 1
    assert b == 2
    assert c == 3
    assert len(v) == 4  # PAD + A + B + C


def test_encode_is_idempotent_for_existing_value():
    v = Vocab(["PAD"])
    t1 = v.encode("A")
    t2 = v.encode("A")
    assert t1 == t2
    assert len(v) == 2


def test_decode_roundtrip():
    v = Vocab(["PAD"])
    tok = v.encode("LeBron James")
    assert v.decode(tok) == "LeBron James"


def test_decode_missing_token_raises_keyerror():
    v = Vocab(["PAD"])
    with pytest.raises(KeyError):
        v.decode(9999)


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "vocab.json"

    v1 = Vocab(["PAD"], path=path)
    t_a = v1.encode("A")
    t_b = v1.encode("B")
    v1.save()

    v2 = Vocab(["PAD"], path=path)  # should autoload
    assert v2.encode("A") == t_a
    assert v2.encode("B") == t_b
    assert v2.decode(t_a) == "A"
    assert v2.next_token == v1.next_token


def test_init_prefers_loading_over_token_init(tmp_path):
    path = tmp_path / "vocab.json"

    v1 = Vocab(["PAD"], path=path)
    v1.encode("A")
    v1.save()

    # token_init here should be ignored because file exists
    v2 = Vocab(["PAD", "SHOULD_NOT_APPEAR"], path=path)

    assert "A" in v2.string_to_token
    assert "SHOULD_NOT_APPEAR" not in v2.string_to_token


def test_save_requires_path_if_not_provided():
    v = Vocab(["PAD"], path=None)
    with pytest.raises(ValueError):
        v.save()


def test_load_requires_existing_file(tmp_path):
    v = Vocab(["PAD"], path=None)
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError):
        v.load(missing)


def test_freeze_maps_unseen_to_unk():
    v = Vocab(["PAD", "UNK"])
    a = v.encode("A")
    v.freeze()
    assert v.encode("NEVER_SEEN") == v.encode("UNK")
    assert v.encode("A") == a  # known values still resolve
    assert "NEVER_SEEN" not in v.string_to_token  # frozen vocab does not grow


def test_freeze_without_unk_raises():
    v = Vocab(["PAD"])
    with pytest.raises(ValueError):
        v.freeze()


def test_frozen_flag_persists_through_save_load(tmp_path):
    path = tmp_path / "vocab.json"
    v1 = Vocab(["PAD", "UNK"], path=path)
    v1.encode("A")
    v1.freeze()
    v1.save()

    v2 = Vocab(["PAD", "UNK"], path=path)
    assert v2.frozen is True
    assert v2.encode("NEW") == v2.encode("UNK")  # stays frozen after load


def test_encode_coerces_non_string_values():
    v = Vocab(["PAD"])
    t = v.encode(2003)
    assert v.encode("2003") == t  # int and its str form share one token


def test_save_writes_expected_json_shape(tmp_path):
    path = tmp_path / "vocab.json"

    v = Vocab(["PAD"], path=path)
    v.encode("A")
    v.save()

    data = json.loads(path.read_text(encoding="utf-8"))

    assert "string_to_token" in data
    assert "token_to_string" in data
    assert "next_token" in data

    # token_to_string keys should be strings in file
    assert all(isinstance(k, str) for k in data["token_to_string"].keys())