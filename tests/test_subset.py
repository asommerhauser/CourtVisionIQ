"""
The representative training subset.

This file is new in 3.2 and it starts from zero: before it, nothing in the suite referenced
``build_subset``, ``season_sample_rates``, ``load_subset_games``, ``SUBSET_MODEL_KEYS`` or
``subset_train_games``. The sampler decides which games eleven of the twelve heads ever see, and it
was completely untested -- which is also why retiring coverage-completeness would have produced no
failure in either direction.
"""
import json

import pytest

import config
from models.registry import STAGE_MODEL_KEYS
from training.subset import (
    build_subset,
    load_subset_games,
    season_sample_rates,
)


def _corpus(per_season, players_per_game=("A", "B", "C")):
    """``(train_ids, game_players, game_season)`` for ``{season: n_games}``, ids counting up."""
    game_players, game_season = {}, {}
    gid = 0
    for season, n in sorted(per_season.items()):
        for _ in range(n):
            gid += 1
            game_players[gid] = set(players_per_game)
            game_season[gid] = season
    return sorted(game_players), game_players, game_season


# --------------------------------------------------------------------------- per-season rates

def test_the_newest_season_is_sampled_at_the_first_listed_rate():
    rates = season_sample_rates([2019, 2020, 2021, 2022], (1.0, 0.70, 0.50), 5.0)
    assert rates[2022] == 1.0
    assert rates[2021] == 0.70
    assert rates[2020] == 0.50


def test_seasons_past_the_listed_block_decay_by_halving():
    """The tail decays from the LAST listed rate, halving every ``halflife`` seasons."""
    rates = season_sample_rates(range(2010, 2023), (1.0, 0.70, 0.50), 5.0)
    anchor = rates[2020]                      # rank 2, the last listed rate
    assert anchor == 0.50
    assert rates[2015] == pytest.approx(anchor * 0.5, rel=1e-6)   # five seasons further back
    assert rates[2010] == pytest.approx(anchor * 0.25, rel=1e-6)  # ten


def test_a_single_season_corpus_is_taken_whole():
    assert season_sample_rates([2023], (1.0, 0.70, 0.50), 5.0) == {2023: 1.0}


# --------------------------------------------------------------------------- selection

def test_each_season_is_filled_to_its_rate():
    train, gp, gs = _corpus({2022: 10, 2023: 10})
    out, stats = build_subset(train, gp, gs, recent_rates=(1.0, 0.50), halflife=5.0, seed=1)

    chosen_2023 = sum(1 for g in out if gs[g] == 2023)
    chosen_2022 = sum(1 for g in out if gs[g] == 2022)
    assert chosen_2023 == 10, "the newest season enters at rate 1.0"
    assert chosen_2022 == 5, "the season before it at 0.50"
    assert stats["n_subset"] == len(out) == 15
    assert stats["by_season"]["2022"] == {"chosen": 5, "total": 10, "rate": 0.5}


def test_the_subset_is_a_subset():
    train, gp, gs = _corpus({2021: 6, 2022: 6, 2023: 6})
    out, _ = build_subset(train, gp, gs, seed=3)
    assert set(out) <= set(train)
    assert out == sorted(set(out)), "sorted and deduplicated"


def test_selection_is_deterministic_for_a_seed_and_moves_with_it():
    """``SUBSET_SEED`` is what makes a train reproducible; a partial rate is what makes it visible."""
    train, gp, gs = _corpus({2020: 40, 2023: 10})
    kw = dict(recent_rates=(1.0,), halflife=5.0)
    first, _ = build_subset(train, gp, gs, seed=7, **kw)
    again, _ = build_subset(train, gp, gs, seed=7, **kw)
    other, _ = build_subset(train, gp, gs, seed=8, **kw)

    assert first == again
    assert first != other, "a different seed draws a different sample from the older season"


def test_an_empty_train_pool_is_empty_rather_than_an_error():
    out, stats = build_subset([], {}, {}, seed=1)
    assert out == []
    assert stats["n_subset"] == 0 and stats["players"] == {}


# ------------------------------------------------------- coverage-completeness, deliberately gone

def test_a_player_who_only_appears_in_a_zero_rate_season_is_not_rescued():
    """**The 3.2 behaviour change, pinned so it is a decision rather than a regression.**

    The sampler used to run a coverage phase first: walk players rarest-first and, for anyone not yet
    covered, add one game containing them, so every player in the train pool was guaranteed at least
    one game. Under W4's minimum-games floor that guarantee is inert -- a player rescued with a single
    game falls below the floor anyway and maps to an anonymous slot -- and it dragged old games into a
    deliberately modern-heavy sample to do it.

    Here ``old_timer`` appears only in a season whose target rounds to zero. He must simply be absent.
    """
    train, gp, gs = _corpus({2023: 6})
    gid = max(train) + 1
    gp[gid] = {"old_timer", "A"}
    gs[gid] = 2005
    train = sorted(gp)

    out, stats = build_subset(train, gp, gs, recent_rates=(1.0,), halflife=1.0, seed=1)

    assert gs.get(out[0]) == 2023 and all(gs[g] == 2023 for g in out), (
        "the 2005 season's target rounds to zero, so none of it is selected")
    assert "old_timer" not in stats["players"], (
        "no coverage phase means no game is added on his behalf")


def test_the_per_player_counts_are_inside_the_subset_not_the_train_pool():
    """W4's floor is defined against subset games, so this is the number it must be counting."""
    train, gp, gs = _corpus({2020: 40, 2023: 10})
    out, stats = build_subset(train, gp, gs, recent_rates=(1.0,), halflife=5.0, seed=5)

    assert stats["players"]["A"] == len(out) < len(train), (
        "A is in every game, so his count is the subset size -- not the 50-game train pool")
    assert stats["n_players"] == len(stats["players"])


def test_a_player_in_only_some_games_is_counted_only_for_those():
    train, gp, gs = _corpus({2023: 8})
    gp[train[0]] = {"A", "cameo"}
    out, stats = build_subset(train, gp, gs, recent_rates=(1.0,), halflife=5.0, seed=1)

    assert len(out) == 8, "rate 1.0 takes the whole season, so the count is exact"
    assert stats["players"]["cameo"] == 1
    assert stats["players"]["A"] == 8


# --------------------------------------------------------------------------- persistence

def test_a_missing_manifest_reads_as_none_rather_than_as_the_full_corpus(tmp_path):
    """The distinction full_run depends on: absent means "extract one", not "use everything"."""
    assert load_subset_games(str(tmp_path / "nope.json")) is None


def test_the_manifest_round_trips_to_a_set_of_ints(tmp_path):
    path = tmp_path / "subset_games.json"
    path.write_text(json.dumps({"subset_game_ids": [3, 1, 2, 2]}), encoding="utf-8")
    assert load_subset_games(str(path)) == {1, 2, 3}


# --------------------------------------------------------------------------- the routing itself

def test_every_trained_head_is_on_the_subset():
    """3.2 put all twelve on it, and this is the guard that keeps the two lists from drifting.

    The list was wrong before: it named six of the seven conditional heads, which was true of nothing
    -- ``timeout_team`` already trained on subset rows because ``run_stage`` builds the shared
    ``cond_*.npz`` once from ``cond_keys[0]``. A membership test rather than a count, so the failure
    names the head.
    """
    assert set(config.SUBSET_MODEL_KEYS) == set(STAGE_MODEL_KEYS)
    assert len(config.SUBSET_MODEL_KEYS) == len(set(config.SUBSET_MODEL_KEYS)), "no duplicates"


def test_the_conditional_heads_move_as_one_group():
    """They share one preprocessed file, so a partial listing desynchronises tensors from graphs.

    ``models.pipeline.run_stage`` raises on a partial listing; putting all twelve on the subset
    satisfies that by definition, and this records why the group cannot be split.
    """
    from models.conditional_type_model import CONDITIONAL_MODEL_CLASSES
    cond = {cls.KEY for cls in CONDITIONAL_MODEL_CLASSES}
    assert cond <= set(config.SUBSET_MODEL_KEYS)
    assert len(cond) == 7
