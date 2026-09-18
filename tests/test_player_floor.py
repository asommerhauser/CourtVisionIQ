"""
The player vocabulary floor and the anonymous slots below it.

The load-bearing property is in :func:`test_two_players_who_share_a_floor_never_share_a_slot`. Everything
else guards a way the floor could be configured and then silently do nothing, which is the failure mode
this whole area has: the train runs, the embedding table is simply the size it always was, and the only
symptom arrives hours later.
"""
import json
import tempfile
from pathlib import Path

import pytest

from encoder.encoder import Encoder
from player_floor import (
    ANON_FILENAME,
    build_alias_map,
    colour_anon_slots,
    is_anon,
    kept_players,
    load_aliases,
    n_slots,
    require_player_floor,
    save_aliases,
)


def _counts(**kwargs):
    return dict(kwargs)


# --------------------------------------------------------------------------- who clears the floor

def test_the_floor_keeps_players_at_or_above_it():
    counts = _counts(star=200, rotation=20, fringe=19, cameo=1)
    keep = kept_players(counts, 20)
    assert keep == {"star", "rotation"}, "at the floor counts as clearing it"


def test_no_floor_keeps_everyone():
    counts = _counts(star=200, cameo=1)
    assert kept_players(counts, None) == {"star", "cameo"}
    assert kept_players(counts, 0) == {"star", "cameo"}
    assert build_alias_map(counts, {}, None) == {}


# --------------------------------------------------------------------------- the slot assignment

def test_two_players_who_share_a_floor_never_share_a_slot():
    """**The property the design exists for.**

    A single shared ``UNK`` cannot represent two anonymous players in one game -- the player and
    substitution heads cannot tell them apart, and a sampled token does not resolve to one man.
    Measured on the real corpus that is 28.7% of games. Colouring the co-occurrence graph fixes it
    globally, which is why no per-game bookkeeping is needed anywhere.
    """
    game_players = {
        1: {"star", "a", "b"},          # a and b share a floor
        2: {"star", "b", "c"},          # so do b and c
        3: {"a", "c"},                  # and a and c
    }
    counts = _counts(star=200, a=1, b=1, c=1)
    aliases = build_alias_map(counts, game_players, 20)

    assert set(aliases) == {"a", "b", "c"}
    for names in game_players.values():
        slots = [aliases[n] for n in names if n in aliases]
        assert len(slots) == len(set(slots)), f"collision among {names}"
    assert n_slots(aliases) == 3, "a triangle needs three colours and must not use more"


def test_players_who_never_meet_share_a_slot():
    """The point of colouring rather than one slot each: slots are reused wherever it is safe.

    Without reuse the table would gain a row per below-floor player, which is the memorisation surface
    the floor exists to remove.
    """
    game_players = {1: {"a"}, 2: {"b"}, 3: {"c"}}
    aliases = build_alias_map(_counts(a=1, b=1, c=1), game_players, 20)
    assert n_slots(aliases) == 1, "no two ever appear together, so one slot serves all three"


def test_the_assignment_is_deterministic():
    """It is persisted and the whole vocabulary depends on it, so ties must not break by dict order."""
    game_players = {i: {"star", f"p{i}", f"p{i + 1}"} for i in range(8)}
    counts = {"star": 200, **{f"p{i}": 1 for i in range(10)}}
    first = build_alias_map(counts, game_players, 20)
    again = build_alias_map(dict(reversed(list(counts.items()))), game_players, 20)
    assert first == again


def test_a_below_floor_player_in_no_recorded_game_still_gets_a_slot():
    """He shares a floor with nobody, so slot 0 is correct rather than a missing key later."""
    aliases = build_alias_map(_counts(ghost=1), {}, 20)
    assert aliases == {"ghost": "ANON_0"}


def test_every_alias_is_an_anon_token_and_no_kept_player_is_aliased():
    counts = _counts(star=200, rotation=50, fringe=2)
    aliases = build_alias_map(counts, {1: {"star", "rotation", "fringe"}}, 20)
    assert all(is_anon(v) for v in aliases.values())
    assert set(aliases).isdisjoint(kept_players(counts, 20))


def test_needing_more_slots_than_the_cap_raises_rather_than_truncating():
    """Truncating would put two players on the same floor under one token, invisibly."""
    clique = {f"p{i}" for i in range(6)}
    with pytest.raises(ValueError, match="exceed ANON_SLOTS_MAX"):
        colour_anon_slots({1: clique}, clique, max_slots=3)


# --------------------------------------------------------------------------- persistence

def test_the_map_round_trips_beside_the_vocabs(tmp_path):
    save_aliases(tmp_path, {"a": "ANON_0", "b": "ANON_1"}, floor=20)
    assert (tmp_path / ANON_FILENAME).is_file()
    assert load_aliases(tmp_path) == {"a": "ANON_0", "b": "ANON_1"}

    payload = json.loads((tmp_path / ANON_FILENAME).read_text(encoding="utf-8"))
    assert payload["floor"] == 20 and payload["n_slots"] == 2 and payload["n_aliased"] == 2


def test_no_map_reads_as_no_floor_rather_than_as_an_error(tmp_path):
    """A weights-only machine and a synthetic fixture both legitimately have none."""
    assert load_aliases(tmp_path) == {}


# --------------------------------------------------------------------------- the encoder applies it

def test_a_below_floor_player_encodes_as_his_slot(tmp_path):
    enc = Encoder(vocab_dir=tmp_path).set_aliases({"fringe": "ANON_3"})
    assert enc.encode_player("fringe") == enc.encode_player("ANON_3")
    assert "fringe" not in enc.player_vocab.string_to_token, "he never gets his own row"


def test_the_roster_and_the_actor_columns_agree(tmp_path):
    """They must, or a below-floor scorer would be a different id from himself on the floor.

    ``secondary_player`` shares the player vocab because the embedding is weight-tied, so it shares the
    aliasing too.
    """
    enc = Encoder(vocab_dir=tmp_path).set_aliases({"fringe": "ANON_1"})
    roster = enc.encode_roster(["star", "fringe"])
    assert enc.encode_player("fringe") in roster
    assert enc.encode_secondary_player("fringe") == enc.encode_player("fringe")


def test_aliasing_preserves_roster_order_and_length(tmp_path):
    """``rest_home`` / ``rest_away`` and the prior planes are roster-parallel, slot for slot."""
    from config import ROSTER_SIZE
    enc = Encoder(vocab_dir=tmp_path).set_aliases({"b": "ANON_0", "d": "ANON_1"})
    ids = enc.encode_roster(["a", "b", "c", "d"])
    assert len(ids) == ROSTER_SIZE
    assert ids[1] == enc.encode_player("ANON_0") and ids[3] == enc.encode_player("ANON_1")
    assert ids[0] == enc.encode_player("a") and ids[2] == enc.encode_player("c")


def test_reserved_tokens_are_never_aliased(tmp_path):
    """``player`` legitimately holds ``"start"`` on a period-opening row."""
    enc = Encoder(vocab_dir=tmp_path).set_aliases({"start": "ANON_0", "PAD": "ANON_1"})
    assert enc.encode_player("start") == enc.player_vocab.string_to_token["start"]
    assert enc.encode_player("PAD") == enc.player_vocab.string_to_token["PAD"]


def test_a_stale_vocab_is_refused_at_freeze(tmp_path):
    """``Vocab`` is append-only, so a rebuild over a pre-floor vocab keeps every below-floor name.

    Nothing else would look wrong: the train runs and the embedding table simply does not shrink.
    """
    enc = Encoder(vocab_dir=tmp_path)
    enc.encode_player("fringe")          # registered before the floor existed
    enc.set_aliases({"fringe": "ANON_0"})
    with pytest.raises(ValueError, match="aliased players still hold their own rows"):
        enc.freeze_all()


def test_load_all_picks_up_a_map_written_after_construction(tmp_path):
    """A head built before the vocab rebuild would otherwise hold an empty map.

    That is the one way two heads could disagree about what a player id means.
    """
    enc = Encoder(vocab_dir=tmp_path)
    assert enc.aliases == {}
    save_aliases(tmp_path, {"fringe": "ANON_0"}, floor=20)
    enc.load_all()
    assert enc.aliases == {"fringe": "ANON_0"}


def test_save_all_carries_the_map_so_it_is_snapshotted_with_the_weights(tmp_path):
    """``manifest.snapshot_vocabs`` copies ``*.json``, so writing it here is what pins it to weights."""
    enc = Encoder(vocab_dir=tmp_path).set_aliases({"fringe": "ANON_0"})
    enc.save_all()
    assert load_aliases(tmp_path) == {"fringe": "ANON_0"}


# --------------------------------------------------------------------------- the pre-flight

def test_a_configured_floor_with_no_map_refuses_to_train(tmp_path):
    """Same argument as ``require_priors``: configured but not materialised is invisible otherwise."""
    with pytest.raises(SystemExit, match="no anon_slots.json"):
        require_player_floor(tmp_path, 20)


def test_the_pre_flight_is_a_no_op_without_a_floor(tmp_path):
    assert require_player_floor(tmp_path, None) == 0
    assert require_player_floor(tmp_path, 0) == 0


def test_the_pre_flight_reports_how_many_are_aliased(tmp_path):
    save_aliases(tmp_path, {"a": "ANON_0", "b": "ANON_0"}, floor=20)
    assert require_player_floor(tmp_path, 20) == 2


def test_a_rebuild_picks_up_a_map_written_after_the_encoder_was_constructed(tmp_path):
    """**The ordering gap the pre-flight cannot see.**

    A head's ``Encoder`` is built when the head object is created, which is *before*
    ``full_run.train`` extracts the subset -- and the extract is what writes ``anon_slots.json``. So at
    construction the map is legitimately empty. A rebuild that trusted that in-memory copy would
    register every below-floor player under his own name and the floor would do nothing.

    Nothing else catches it: ``require_player_floor`` checks the FILE, which exists;
    ``assert_aliases_absent`` returns early on an empty map; and the only symptom is an embedding table
    that did not shrink. Without ``prepare_for_rebuild`` this test fails.
    """
    enc = Encoder(vocab_dir=tmp_path)
    assert enc.aliases == {}, "nothing on disk yet, which is the normal case at construction"

    save_aliases(tmp_path, {"fringe": "ANON_0"}, floor=20)   # the extract runs
    enc.prepare_for_rebuild()                                # what the rebuild branch now does

    enc.encode_roster(["star", "fringe"])
    enc.freeze_all()
    assert "fringe" not in enc.player_vocab.string_to_token, (
        "the floor must apply to the vocab this rebuild writes")
    assert enc.encode_player("fringe") == enc.encode_player("ANON_0")


def test_every_rebuild_branch_refreshes_the_alias_map():
    """All six heads own a rebuild branch, and each must refresh -- not just the vocab owner.

    ``event_time`` is the only head that runs with ``rebuild_vocabs=True`` in a full train today, but
    that is a property of ``models.pipeline``'s routing rather than of these call sites, and a single
    head left out would be a silent floor rather than a failure.
    """
    import pathlib
    heads = ["event_time_model", "player_model", "conditional_time_model",
             "conditional_type_model", "substitution_model", "sub_decision_model"]
    for head in heads:
        src = pathlib.Path("models") / f"{head}.py"
        text = src.read_text(encoding="utf-8")
        i = text.index("if rebuild_vocabs:")
        window = text[i:i + 700]
        assert "prepare_for_rebuild()" in window, f"{head} rebuilds without refreshing the aliases"
