"""
The player vocabulary floor, and the anonymous slots below it.

A player who appears in fewer than ``MIN_PLAYER_SUBSET_GAMES`` games *inside the training subset* does
not get his own embedding row. He is **aliased** to an anonymous slot token instead: he keeps his
fourteen (3.2: twenty-one) season-to-date prior scalars, which enter the roster set encoder additively
before attention, and loses only the identity row -- which for a deep-bench player was mostly noise.
``docs/v3_direction.md`` 1f names that embedding table as where the memorisation lives, so this is the
one capacity change in 3.2 and it goes downward.

**Why a single global map is enough.** The obvious design is a per-game map: two below-floor players in
the same game must not share a token, or the player and substitution heads cannot tell them apart and
a sampled token does not resolve to one man. Measured at the 2011 cut with a floor of 20, that is a
real problem -- two or more anonymous players are rostered in **28.7%** of games, up to twelve in one
game -- so a single shared ``UNK`` is not an option.

But per-game maps are not the only way to fix it. Assign slots by **colouring the co-occurrence
graph**: an edge between two below-floor players who ever appear in the same game, then a colour per
player such that no edge joins two of the same colour. Measured on the real corpus, that needs
**36 slots for zero collisions**. So one global, stateless name -> token map gives exactly the
guarantee a per-game map would, with none of the bookkeeping: no grouping by game in six heads'
preprocess, no second parse of every roster cell, and nothing to thread through
``simulation/input_cache.py`` or ``simulation/game_input.py``.

For comparison, the naive ``rank mod slots`` assignment collides in 7.4% of games at 16 slots and 2.2%
at 64. The colouring collides in none, at 36.

**What a slot's embedding means.** Not "a different person every game" -- it is a fixed partition of
below-floor players into buckets, each averaging roughly eighteen men, and the colouring guarantees no
two bucket-mates ever share a floor. So the row learns "generic deep-bench player", which is what we
want it to learn, and everything that distinguishes one from another arrives through the priors.

This module is deliberately TF-free and lives at the repo root next to ``player_priors.py`` and
``season_context.py``, so the simulator side can import it without pulling TensorFlow.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ANON_PREFIX = "ANON"

#: Names that are never aliased. The encoder's reserved tokens plus the substitution head's
#: start-of-stint marker: these appear in the ``player`` / ``secondary_player`` columns as real values
#: (a period-opening row has ``player == "start"``), and aliasing one would corrupt the vocabulary's
#: fixed ids.
PROTECTED = frozenset({"PAD", "UNK", "start", "end", "none", ""})

ANON_FILENAME = "anon_slots.json"


def anon_token(slot: int) -> str:
    """The name a slot is registered under. ``ANON_0``, ``ANON_1``, ..."""
    return f"{ANON_PREFIX}_{int(slot)}"


def is_anon(name) -> bool:
    return isinstance(name, str) and name.startswith(ANON_PREFIX + "_")


def kept_players(player_games: dict, floor: int | None) -> set[str]:
    """Names clearing ``floor`` games. ``floor`` of ``None`` or ``<= 0`` keeps everyone."""
    if not floor or floor <= 0:
        return set(player_games)
    return {str(n) for n, c in player_games.items() if int(c) >= int(floor)}


def colour_anon_slots(game_players, below: set[str], *, max_slots: int | None = None) -> dict[str, int]:
    """Assign each below-floor player a slot such that co-occurring players never share one.

    ``game_players`` maps a game id to the set of names appearing in it. Greedy colouring, highest
    degree first -- which is the standard Welsh-Powell ordering and, measured on this corpus, lands on
    36 colours against a maximum clique of at least 12.

    **Deterministic**, because the result is persisted and the whole vocabulary depends on it: ties in
    degree break by name, and the colour chosen is always the lowest free one.
    """
    below = {str(n) for n in below}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for names in game_players.values():
        present = sorted(n for n in (str(x) for x in names) if n in below)
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                adjacency[a].add(b)
                adjacency[b].add(a)

    # Highest degree first; name breaks the tie so the assignment is reproducible.
    order = sorted(below, key=lambda n: (-len(adjacency[n]), n))
    slots: dict[str, int] = {}
    for name in order:
        taken = {slots[m] for m in adjacency[name] if m in slots}
        slot = 0
        while slot in taken:
            slot += 1
        slots[name] = slot

    used = (max(slots.values()) + 1) if slots else 0
    if max_slots is not None and used > max_slots:
        raise ValueError(
            f"anonymous slots needed ({used}) exceed ANON_SLOTS_MAX ({max_slots}). That means the "
            f"co-occurrence graph got much denser -- raise the cap deliberately after looking at why, "
            f"rather than truncating, which would silently make two players on the same floor share "
            f"one token.")
    return slots


def alias_map(player_games: dict, floor: int | None, *, max_slots: int | None = None) -> dict[str, str]:
    """``{below_floor_name: anon_token}`` from per-player subset game counts.

    Needs the co-occurrence structure, so callers pass ``game_players`` to
    :func:`build_alias_map` instead when they have it. This form exists for the degenerate case of no
    co-occurrence information, and puts every below-floor player in his own slot.
    """
    below = set(player_games) - kept_players(player_games, floor)
    ordered = sorted(below)
    if max_slots is not None and len(ordered) > max_slots:
        raise ValueError(f"{len(ordered)} below-floor players exceed ANON_SLOTS_MAX ({max_slots})")
    return {name: anon_token(i) for i, name in enumerate(ordered)}


def build_alias_map(player_games: dict, game_players, floor: int | None, *,
                    max_slots: int | None = None) -> dict[str, str]:
    """``{below_floor_name: anon_token}``, slots assigned by colouring the co-occurrence graph."""
    below = set(player_games) - kept_players(player_games, floor)
    if not below:
        return {}
    slots = colour_anon_slots(game_players, below, max_slots=max_slots)
    # A below-floor player who never appears in ``game_players`` gets no edges and so colour 0, which
    # is correct: he shares a floor with nobody.
    return {name: anon_token(slots.get(name, 0)) for name in sorted(below)}


def n_slots(aliases: dict[str, str]) -> int:
    """How many distinct anonymous tokens an alias map uses."""
    return len({v for v in aliases.values()})


def save_aliases(vocab_dir, aliases: dict[str, str], *, floor: int | None = None) -> Path:
    """Persist the alias map beside the vocabs, so it travels with them.

    It belongs here rather than in ``training/subset_games.json`` because it is part of *the language*:
    ``Encoder`` loads it with the vocabs, ``full_run`` snapshots the directory into
    ``artifacts/<name>/vocabs/``, and the manifest fingerprints it. A model reloaded from those weights
    then aliases names exactly as its train did, which is the property that has to hold.
    """
    path = Path(vocab_dir) / ANON_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "floor": floor,
        "n_slots": n_slots(aliases),
        "n_aliased": len(aliases),
        "aliases": dict(sorted(aliases.items())),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return path


def load_aliases(vocab_dir) -> dict[str, str]:
    """The persisted alias map, or ``{}`` when there is none (no floor applied)."""
    path = Path(vocab_dir) / ANON_FILENAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in (payload.get("aliases") or {}).items()}


def require_player_floor(vocab_dir, floor: int | None) -> int:
    """Pre-flight for a real train: a configured floor must have an alias map on disk.

    The same argument ``prior_features.require_priors`` makes. If ``MIN_PLAYER_SUBSET_GAMES`` is set
    but ``anon_slots.json`` was never written -- the subset was extracted before the floor existed, or
    a fresh clone carries no manifest -- then every player keeps his own embedding row and the floor
    does nothing. Nothing looks wrong: the train runs and the table is simply the size it always was.

    Returns how many players are aliased. Zero with a floor set is an error, not an answer.
    """
    if not floor or floor <= 0:
        return 0
    aliases = load_aliases(vocab_dir)
    if not aliases:
        raise SystemExit(
            f"MIN_PLAYER_SUBSET_GAMES is {floor} but there is no {ANON_FILENAME} in "
            f"{Path(vocab_dir).resolve()}, so every player would keep his own embedding row and the "
            f"floor would do nothing.\nExtract the subset first (it writes the map):\n"
            f"  python -m training.subset extract")
    return len(aliases)


def player_counts_from_subset(path) -> dict[str, int]:
    """The per-player subset game counts ``training.subset.extract`` writes, or ``{}``."""
    p = Path(path)
    if not p.is_file():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in (payload.get("players") or {}).items()}


__all__ = [
    "ANON_FILENAME", "ANON_PREFIX", "PROTECTED",
    "alias_map", "anon_token", "build_alias_map", "colour_anon_slots", "is_anon",
    "kept_players", "load_aliases", "n_slots", "player_counts_from_subset",
    "require_player_floor", "save_aliases",
]
