"""
Season-to-date priors: causality first, then the cold-start cases.

The 2.0 evaluation found the model predicting who a player *was* -- regressing its prediction on
career and season averages gave ``0.24 x career + 0.61 x season``, rookies came out 10.9% worse than
"use his season average", and the most-improved profile 24% worse. The priors exist to put the
season-to-date rates where the model can read them.

**The causality test is the one that matters.** A prior that can see the game it is attached to
would make training look better and inference worse -- the single failure mode that flatters itself
all the way to a finished run. Everything else here is a cold-start case: the states where a naive
implementation quietly emits zeros, and a zero prior is not "no information", it is the confident
claim that a player does nothing.

Pure pandas -- ``generate_box_score`` is TF-free (see ``simulation/__init__``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from player_priors import (
    LEAGUE_DEFAULTS,
    PLAYER_PRIOR_KEYS,
    SHRINK_K,
    TEAM_PRIOR_KEYS,
    PriorCarry,
    PlayerTotals,
    priors_for_season,
    shrink,
)

HOME = ["H1", "H2", "H3", "H4", "H5"]
AWAY = ["A1", "A2", "A3", "A4", "A5"]


def _game(gid, date, *, scorer="H1", baskets=5, home=None, away=None, home_team="HOM",
          away_team="AWY"):
    """One minimal game: a tip, some made rim shots by ``scorer``, and a closing row."""
    home = home if home is not None else HOME
    away = away if away is not None else AWAY

    def row(t, event, player, type_, result):
        return {"game_id": gid, "game_date": date, "home_team": home_team,
                "away_team": away_team, "time": t, "event": event, "player": player,
                "type": type_, "result": result, "secondary_player": "",
                "roster_home": list(home), "roster_away": list(away),
                "home/away": "home", "season": 2023, "playoff": 0}

    rows = [row(0, "start", "start", "start", "start")]
    rows += [row(100 + 10 * i, "shot", scorer, "rim", "made") for i in range(baskets)]
    rows.append(row(2879, "shot", away[0], "rim", "missed"))
    return rows


def _season(games):
    return pd.DataFrame([r for g in games for r in g])


# --------------------------------------------------------------------------- causality

def test_a_games_prior_is_built_only_from_earlier_games():
    """The leakage guard. Game 1's prior cannot know about game 1.

    H1 scores 10 in game 1 and nothing afterwards. If the prior for game 1 already reflected that
    game, its scoring rate would be non-default on the very first row.
    """
    frame = _season([
        _game(1, "2023-01-01", scorer="H1", baskets=5),
        _game(2, "2023-01-03", scorer="H2", baskets=5),
        _game(3, "2023-01-05", scorer="H2", baskets=5),
    ])
    players, _teams = priors_for_season(frame, PriorCarry())
    first = players[(players.game_id == 1) & (players.player == "H1")].iloc[0]
    assert first.prior_games == 0
    assert first.pts_36 == pytest.approx(LEAGUE_DEFAULTS["pts_36"])

    second = players[(players.game_id == 2) & (players.player == "H1")].iloc[0]
    assert second.prior_games == 1
    # One game in, shrunk 1/(1+k) toward the league mean -- so it has MOVED off the default
    # without having jumped all the way to the observed rate.
    assert second.pts_36 != pytest.approx(LEAGUE_DEFAULTS["pts_36"])
    assert second.pts_36 > LEAGUE_DEFAULTS["pts_36"] * 0.5


def test_the_prior_moves_monotonically_toward_the_observed_rate():
    """Shrinkage is what makes a rookie's first ten games usable instead of noise: the prior walks
    from the league mean toward what he actually does, and never overshoots it.

    Minutes are the clean case here -- these players are on the floor for a full 48-minute game
    against a 20 mpg league default, so the prior has to climb, game after game, without ever
    reaching the observed value.
    """
    frame = _season([_game(i, f"2023-01-{i:02d}", scorer="H1", baskets=6) for i in range(1, 12)])
    players, _ = priors_for_season(frame, PriorCarry())
    h1 = players[players.player == "H1"].sort_values("game_id")
    minutes = h1.min_pg.to_numpy()
    assert minutes[0] == pytest.approx(LEAGUE_DEFAULTS["min_pg"]), "game 1 is the league mean"
    assert (np.diff(minutes) > 0).all(), "each game of evidence must move it further"
    observed = h1.min_pg.iloc[-1]
    assert observed < 48.0, "shrinkage must never let it reach the raw observed rate"


def test_seasons_must_be_walked_in_order_for_the_seed_chain_to_be_causal():
    """A seed taken from a LATER season is a leak that no within-season check would catch -- every
    prior would still be built only from earlier games of its own season. The ordering lives in
    ``build``; this pins the carry's half of the contract."""
    def _totals(pts, n=10):
        totals = PlayerTotals()
        for _ in range(n):
            totals.add(type("L", (), {"seconds": 1800.0, "pts": pts, "fga": 15, "fgm": 8,
                                      "tpa": 5, "fta": 4, "ast": 5, "oreb": 1, "dreb": 4,
                                      "tov": 2})())
        return totals

    carry = PriorCarry()
    star, filler = _totals(30), _totals(6)
    # Before the season closes, nobody has a personal seed.
    assert carry.seed_for("Star") == carry.league_seed
    carry.roll({"Star": star, "Filler": filler}, {})
    # After it closes, each player seeds himself, and neither one is the league mean -- two
    # different players must not collapse to the same seed.
    assert carry.seed_for("Star")["pts_36"] == pytest.approx(star.rates()["pts_36"])
    assert carry.seed_for("Filler")["pts_36"] == pytest.approx(filler.rates()["pts_36"])
    assert carry.seed_for("Star")["pts_36"] > carry.league_seed["pts_36"]
    assert carry.seed_for("Filler")["pts_36"] < carry.league_seed["pts_36"]
    # And a player nobody has seen still falls back to the league.
    assert carry.seed_for("Newcomer") == carry.league_seed


# --------------------------------------------------------------------------- cold start

def test_a_player_with_no_history_gets_the_league_mean_not_zero():
    """A zero vector claims a player does nothing. The league mean says nothing is known."""
    frame = _season([_game(1, "2023-01-01")])
    players, _ = priors_for_season(frame, PriorCarry())
    rookie = players[players.prior_games == 0]
    assert len(rookie) == 10
    for key in PLAYER_PRIOR_KEYS:
        assert rookie[key].iloc[0] == pytest.approx(LEAGUE_DEFAULTS[key])
    assert (rookie[list(PLAYER_PRIOR_KEYS)].to_numpy() > 0).all()


def test_a_player_on_the_roster_who_never_touches_the_ball_still_accrues_minutes():
    """He is on the floor, so he played. A shot-row tally would give him zero and make his next
    prior read as a player with no minutes -- which is why generate_box_score is mandatory here:
    it credits minutes from the roster snapshots, not from box events."""
    frame = _season([_game(1, "2023-01-01", scorer="H1"),
                     _game(2, "2023-01-03", scorer="H1")])
    players, _ = priors_for_season(frame, PriorCarry())
    quiet = players[(players.game_id == 2) & (players.player == "H5")].iloc[0]
    assert quiet.prior_games == 1
    assert quiet.min_pg > 0, "a roster-only player must not read as zero minutes"


def test_every_rostered_player_gets_a_row_for_every_game():
    frame = _season([_game(1, "2023-01-01"), _game(2, "2023-01-03")])
    players, _ = priors_for_season(frame, PriorCarry())
    assert len(players) == 20
    assert set(players[players.game_id == 1].player) == set(HOME) | set(AWAY)


# --------------------------------------------------------------------------- shrinkage

def test_shrinkage_is_the_stated_weighting():
    observed = {"pts_36": 30.0}
    seed = {"pts_36": 10.0}
    got = shrink(observed, seed, games=SHRINK_K, keys=("pts_36",))
    assert got["pts_36"] == pytest.approx(0.5 * 30.0 + 0.5 * 10.0)


def test_zero_games_is_the_seed_outright():
    got = shrink({"pts_36": 30.0}, {"pts_36": 10.0}, games=0, keys=("pts_36",))
    assert got["pts_36"] == pytest.approx(10.0)


# --------------------------------------------------------------------------- team priors

def test_team_priors_are_emitted_per_side_per_game():
    frame = _season([_game(1, "2023-01-01"), _game(2, "2023-01-03")])
    _players, teams = priors_for_season(frame, PriorCarry())
    assert len(teams) == 4
    assert set(teams.side) == {"home", "away"}
    for key in TEAM_PRIOR_KEYS:
        assert key in teams.columns


def test_a_teams_first_game_carries_no_history_either():
    frame = _season([_game(1, "2023-01-01")])
    _players, teams = priors_for_season(frame, PriorCarry())
    assert (teams.prior_games == 0).all()


# --------------------------------------------------------------------------- the model side

def test_the_normalized_league_mean_is_what_an_unknown_name_and_a_pad_slot_both_read():
    from models.prior_features import _DEFAULT_PLAYER, pad_priors

    planes = pad_priors(["Known", "Unknown"], {"Known": np.ones(len(PLAYER_PRIOR_KEYS),
                                                               dtype=np.float32)})
    assert planes.shape == (5, len(PLAYER_PRIOR_KEYS))
    np.testing.assert_array_equal(planes[0], np.ones(len(PLAYER_PRIOR_KEYS), dtype=np.float32))
    np.testing.assert_allclose(planes[1], _DEFAULT_PLAYER)   # name not in the map
    np.testing.assert_allclose(planes[4], _DEFAULT_PLAYER)   # PAD slot


def test_normalization_puts_an_average_player_near_one():
    """The whole point of fixed constants over a fitted z-score: the scale is legible, and it is
    the same scale on the day the sidecar is rebuilt as it was on the day the model trained."""
    from models.prior_features import _DEFAULT_PLAYER

    assert (_DEFAULT_PLAYER > 0.4).all() and (_DEFAULT_PLAYER < 1.6).all()


def test_the_roster_encoder_is_built_for_rest_plus_rotation_plus_the_priors():
    """NUM_ROSTER_SCALARS is baked into scalar_proj's kernel shape, so a mismatch fails at
    load_weights rather than loading quietly and meaning something else."""
    from models.prior_features import N_PLAYER_PRIORS
    from models.rotation_features import NUM_ROSTER_SCALARS

    assert NUM_ROSTER_SCALARS == 4 + N_PLAYER_PRIORS


def test_a_prior_vector_is_ordered_the_way_the_model_unstacks_it():
    """side_prior_scalars unstacks the last axis positionally; if the column order in the sidecar
    and PLAYER_PRIOR_KEYS ever disagree, every rate silently becomes a different rate."""
    from models.prior_features import N_PLAYER_PRIORS

    assert N_PLAYER_PRIORS == len(PLAYER_PRIOR_KEYS)
    frame = _season([_game(1, "2023-01-01")])
    players, _ = priors_for_season(frame, PriorCarry())
    assert list(players.columns[-N_PLAYER_PRIORS:]) == list(PLAYER_PRIOR_KEYS)
