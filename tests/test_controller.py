"""
Controller rule-engine tests.

These verify the hard basketball rules the Controller enforces, using a lightweight ``FakeSim``
that stands in for :class:`~simulation.game_simulator.GameSimulator`: it reproduces the real
``append_event`` roster-snapshot + substitution behavior and returns *scripted* head outputs, so
the rules can be exercised on CPU with **no trained models**. Each test drives a single play
handler and asserts the emitted rows + game context (score, possession, fouls, pending rebound).
"""
from __future__ import annotations

import numpy as np
import pytest

# Dials are read as ``config.<DIAL>`` at assert time, matching how the Controller reads them.
# Binding them here would make these tests pass vacuously once a dial is overridden at runtime.
import config
from config import ROSTER_SIZE
from simulation.controller import (
    GameController, OPEN_PLAY_EVENTS, REGULATION, OT_LENGTH, PERIOD_LENGTH,
)
from simulation.controller import SHOOTING_2PT, SHOOTING_3PT, SHOOTING_FOUL_TYPES
from simulation.game_simulator import HOME, AWAY
from models.game_state_features import GameStateScan
from zones import ZONE_TOKENS

HOME_FIVE = ["A", "B", "C", "D", "E"]
AWAY_FIVE = ["F", "G", "H", "I", "J"]
REQUIRED_HEADS = config.REQUIRED_HEADS   # one list, so a new head cannot drift


class FakeSim:
    """Scripted stand-in for GameSimulator — no TF graph, no artifacts."""

    def __init__(self, sub_count: int = 0, timeouts: bool = False):
        self.home_roster = list(HOME_FIVE)
        self.away_roster = list(AWAY_FIVE)
        self.home_full = list(HOME_FIVE)
        self.away_full = list(AWAY_FIVE)
        self.history: list[dict] = []
        self.heads = {k: object() for k in REQUIRED_HEADS}
        if timeouts:                        # opt-in to the timeout_team head
            self.heads["timeout_team"] = object()
        self.sub_count = sub_count          # fixed count returned by predict_sub_count
        self.rng = np.random.default_rng(0)
        self.calls: list[tuple] = []
        self._q: dict[str, list] = {"player": [], "type": [], "result": [], "incoming": [],
                                    "delta": []}

    # --- scripting ---
    def script(self, **queues):
        for k, v in queues.items():
            self._q[k] = list(v)
        return self

    def _pop(self, kind):
        return self._q[kind].pop(0)

    # --- head stand-ins (record args, return scripted values) ---
    def predict_player(self, next_event, candidates, *, delta_seconds=0.0, greedy=False,
                       temperature=1.0):
        self.calls.append(("player", next_event, list(candidates), temperature))
        return self._pop("player")

    def predict_type(self, key, next_event, next_player, allowed, *, delta_seconds=0.0, greedy=False):
        self.calls.append(("type", key, next_player, list(allowed)))
        return self._pop("type")

    def predict_result(self, next_player, next_type, allowed, *, delta_seconds=0.0, greedy=False,
                       bias=None):
        self.calls.append(("result", next_player, next_type, list(allowed)))
        return self._pop("result")

    def predict_incoming(self, outgoing, candidates, *, delta_seconds=0.0, greedy=False):
        self.calls.append(("incoming", outgoing, list(candidates)))
        return self._pop("incoming")

    def predict_delta(self, next_event, next_player, *, delta_seconds=0.0):
        self.calls.append(("delta", next_event, next_player))
        return self._pop("delta") if self._q["delta"] else 12.0

    def sample_substitution(self, *, team=None, delta_seconds=0.0, greedy=False,
                            outgoing_bias=None):
        self.calls.append(("sub", team, outgoing_bias))
        return self._pop("player"), self._pop("incoming")

    def predict_sub_count(self, team, *, greedy=False):
        self.calls.append(("sub_count", team))
        return self.sub_count

    def start_alternating(self, home_full, away_full, *, season="2003",
                          tipoff_time=0.0, greedy=False, greedy_starters=False,
                          season_context=None):
        self.calls.append(("start_alternating", greedy, greedy_starters))

    def start_with_starters(self, home_full, away_full, home_starters, away_starters,
                            *, season="2003", tipoff_time=0.0,
                            season_context=None):
        self.calls.append(("start_with_starters", list(home_starters), list(away_starters)))
        # The real simulator copies the full rosters here too (game_simulator.py:777) — keep
        # that faithful, or a bench player is missing from anything keyed off home_full.
        self.home_full = list(home_full)
        self.away_full = list(away_full)
        self.home_roster = list(home_starters)
        self.away_roster = list(away_starters)

    # --- faithful append_event (roster mutation on subs + snapshot) ---
    def append_event(self, event, player, type, result, secondary_player="none", time=None):
        if event == "substitution":
            for roster in (self.home_roster, self.away_roster):
                if player in roster:
                    roster[roster.index(player)] = secondary_player
                    break
        row = {"event": event, "player": player, "type": type, "result": result,
               "secondary_player": secondary_player, "time": time,
               "roster_home": list(self.home_roster), "roster_away": list(self.away_roster)}
        self.history.append(row)
        return row


def make_controller(possession=HOME):
    ctrl = GameController(FakeSim(), seed=0)
    ctrl.possession = possession
    return ctrl


def rows(ctrl):
    return ctrl.sim.history


# ===================================================================== #
# Assist → made shot
# ===================================================================== #

def test_assist_forces_made_shot_by_different_teammate():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A", "C"], type=["paint"])  # assister A, shooter C
    ctrl._do_assist(delta=5.0)

    assist, shot = rows(ctrl)
    assert (assist["event"], assist["player"], assist["type"], assist["result"]) == \
        ("assist", "A", "paint", "score")
    assert (shot["event"], shot["player"], shot["type"], shot["result"]) == \
        ("shot", "C", "paint", "made")
    # The shooter pool excluded the assister.
    shooter_call = [c for c in ctrl.sim.calls if c[0] == "player" and c[1] == "shot"][0]
    assert "A" not in shooter_call[2]
    assert ctrl.score[HOME] == 2
    assert ctrl.possession == AWAY     # made FG flips possession


# ===================================================================== #
# Block → missed FGA + paired block + rebound
# ===================================================================== #

def test_blocked_shot_emits_paired_block_and_awaits_rebound():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A", "F"], type=["paint"], result=["blocked"])  # shooter A, blocker F
    ctrl._do_shot(delta=5.0)

    shot, block = rows(ctrl)
    assert (shot["event"], shot["player"], shot["result"]) == ("shot", "A", "blocked")
    assert (block["event"], block["player"], block["result"], block["secondary_player"]) == \
        ("block", "F", "block", "A")
    # The blocker was sampled from the defense five.
    block_call = [c for c in ctrl.sim.calls if c[0] == "player" and c[1] == "block"][0]
    assert block_call[2] == AWAY_FIVE
    assert ctrl.pending_rebound is True
    assert ctrl.score[HOME] == 0


def test_missed_shot_awaits_rebound_no_score():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A"], type=["paint"], result=["missed"])
    ctrl._do_shot(delta=5.0)
    assert ctrl.pending_rebound is True
    assert ctrl.possession == HOME     # no change until the rebound resolves
    assert ctrl.score[HOME] == 0


# ===================================================================== #
# Rebounds
# ===================================================================== #

def test_defensive_rebound_flips_possession():
    ctrl = make_controller(HOME)            # home just missed
    ctrl.sim.script(type=["defensive"], player=["F"])  # type head: defensive; rebounder F
    ctrl._do_rebound(delta=2.0)
    reb = rows(ctrl)[-1]
    assert (reb["event"], reb["type"], reb["result"]) == ("rebound", "defensive", "cop")
    assert ctrl.possession == AWAY
    # Rebounder was sampled from the defending (away) five, not all ten.
    reb_call = [c for c in ctrl.sim.calls if c[0] == "player" and c[1] == "rebound"][0]
    assert reb_call[2] == AWAY_FIVE


def test_offensive_rebound_retains_possession():
    ctrl = make_controller(HOME)
    ctrl.sim.script(type=["offensive"], player=["B"])  # type head: offensive; rebounder B
    ctrl._do_rebound(delta=2.0)
    reb = rows(ctrl)[-1]
    assert (reb["event"], reb["type"], reb["result"]) == ("rebound", "offensive", "null")
    assert ctrl.possession == HOME
    # Rebounder was sampled from the offense (home) five, not all ten.
    reb_call = [c for c in ctrl.sim.calls if c[0] == "player" and c[1] == "rebound"][0]
    assert reb_call[2] == HOME_FIVE


def test_rebound_type_head_decides_split_before_player():
    ctrl = make_controller(HOME)
    ctrl.sim.script(type=["offensive"], player=["B"])
    ctrl._do_rebound(delta=2.0)
    # The off/def split comes from the rebound_type head, masked to the two live types.
    type_call = [c for c in ctrl.sim.calls if c[0] == "type"][0]
    assert type_call[1] == "rebound_type"
    assert type_call[3] == ["offensive", "defensive", "team offensive", "team defensive"]


# ===================================================================== #
# Turnovers / steals
# ===================================================================== #

def test_a_steal_is_one_row_naming_the_stealer():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A", "F"], type=["steal"])  # committer A (home), stealer F (away)
    ctrl._do_turnover(delta=5.0)

    (row,) = rows(ctrl)
    assert (row["player"], row["type"], row["result"]) == ("A", "steal", "cop")
    assert row["secondary_player"] == "F"
    assert ctrl.possession == AWAY


def test_nonsteal_turnover_single_row():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A"], type=["violation"])
    ctrl._do_turnover(delta=5.0)
    (tov,) = rows(ctrl)
    assert (tov["player"], tov["type"], tov["result"]) == ("A", "violation", "cop")
    assert ctrl.possession == AWAY


# ===================================================================== #
# Fouls → free throws + NBA bonus
# ===================================================================== #

def test_shooting_foul_on_2pt_yields_two_free_throws():
    ctrl = make_controller(HOME)            # home has the ball; away fouls
    # fouler F, then the fouled shooter A; the intended attempt is a 2pt → 2 FTs.
    ctrl.sim.script(player=["F", "A"], type=[SHOOTING_2PT], result=["made", "made"])
    ctrl._do_foul(delta=5.0)

    foul = rows(ctrl)[0]
    fts = rows(ctrl)[1:]
    assert (foul["event"], foul["type"], foul["result"]) == ("foul", SHOOTING_2PT, "free throw")
    assert len(fts) == 2 and all(r["type"] == "free throw" for r in fts)
    assert ctrl.score[HOME] == 2
    assert ctrl.possession == AWAY         # made last FT → other team inbounds


def test_shooting_foul_on_3pt_yields_three_free_throws():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F", "A"], type=[SHOOTING_3PT],
                    result=["made", "made", "made"])
    ctrl._do_foul(delta=5.0)
    fts = rows(ctrl)[1:]
    assert len(fts) == 3                    # a 3pt shooting foul is three free throws
    assert ctrl.score[HOME] == 3


def test_and_one_keeps_basket_and_adds_one_free_throw():
    # The time head says "no gap" for this foul (delta 0.0) -> P(and-1) = 1: this foul IS the and-1.
    ctrl = make_controller(AWAY)            # made FG already flipped possession to AWAY
    ctrl.sim.append_event("shot", "A", "paint", "made", time=0)   # A (home) just scored
    ctrl.score[HOME] = 2                     # the basket counted
    # An away player fouls on the made basket → and-1: A shoots a single FT.
    ctrl.sim.script(player=["G"], type=[SHOOTING_2PT], result=["made"], delta=[0.0])
    ctrl._do_foul(delta=5.0)

    fts = [r for r in rows(ctrl) if r["type"] == "free throw"]
    assert len(fts) == 1 and fts[0]["player"] == "A"
    assert ctrl.score[HOME] == 3            # 2 (basket) + 1 (and-1 FT)
    # The fouled attempt was a made FG, so it is the only field-goal attempt logged (no phantom).
    assert sum(1 for r in rows(ctrl) if r["type"] in ZONE_TOKENS) == 1


def test_a_foul_after_a_basket_is_an_ordinary_foul_when_the_head_says_a_long_gap():
    """The other 66%: the head's gap is at or past the scale -> P(and-1) = 0. The ball changed
    hands, so it is a foul on the NEW possession -- re-drawn from that possession's side and
    paid at the token's count -- and its gap is floored at the later-foul scale."""
    ctrl = make_controller(AWAY)            # the made FG flipped possession to AWAY
    ctrl.sim.append_event("shot", "A", "paint", "made", time=0)
    ctrl.score[HOME] = 2
    # Probe: F (away, the defender on the old possession) with a 20s gap -> not on the shot.
    # Then the ordinary foul: home is defending now, B fouls G in the act -> two FTs for G.
    ctrl.sim.script(player=["F", "B", "G"], type=[SHOOTING_2PT], result=["made", "made"],
                    delta=[20.0, 3.0])
    ctrl._do_foul(delta=5.0)

    foul = [r for r in rows(ctrl) if r["event"] == "foul"][0]
    fts = [r for r in rows(ctrl) if r["type"] == "free throw"]
    assert foul["player"] == "B" and foul["secondary_player"] == "G"   # the new possession's draw
    assert len(fts) == 2 and all(r["player"] == "G" for r in fts)
    assert ctrl.score == {HOME: 2, AWAY: 2}
    assert [c[2] for c in ctrl.sim.calls if c[0] == "delta"] == ["F", "B"]   # probe, then the play
    assert ctrl.clock == config.AND_ONE_GAP_SCALE   # 3.0 floored: the long branch of the mixture


def test_an_and_one_sits_at_the_baskets_clock():
    """No time elapses between the basket and the whistle: the foul row carries the shot's clock.
    The head is asked once (the probe) and its answer is the branch, not a clock advance."""
    ctrl = make_controller(AWAY)
    ctrl.clock = 100.0
    ctrl.sim.append_event("shot", "A", "paint", "made", time=100.0)
    ctrl.sim.script(player=["G"], type=[SHOOTING_2PT], result=["made"], delta=[0.0])
    ctrl._do_foul(delta=5.0)

    foul = [r for r in rows(ctrl) if r["event"] == "foul"][0]
    assert foul["time"] == 100.0 and ctrl.clock == 100.0
    assert [c[2] for c in ctrl.sim.calls if c[0] == "delta"] == ["G"]
    type_call = [c for c in ctrl.sim.calls if c[0] == "type"][0]
    assert set(type_call[3]) == set(SHOOTING_FOUL_TYPES)   # an and-1 is a shooting foul


def test_the_and_one_is_read_off_the_time_heads_gap():
    """P(and-1) = 1 - gap / AND_ONE_GAP_SCALE: the head's mean gap is a mixture mean, and the
    dial is only the scale that turns it back into a probability."""
    assert "AND_ONE_GAP_SCALE" in config._TUNING_KEYS
    scale = config.AND_ONE_GAP_SCALE
    p = GameController._and_one_prob
    assert p(0.0) == 1.0 and p(scale) == 0.0 and p(scale * 2) == 0.0
    assert abs(p(scale / 2) - 0.5) < 1e-9
    assert p(2.0) > p(5.0) > p(8.0)                      # a shorter gap is more and-1
    ctrl = make_controller(AWAY)
    n, gap = 20000, scale * (1 - 0.338)                  # the gap whose p is the 2023 level
    hits = sum(ctrl._draw_and_one(gap) for _ in range(n))
    assert abs(hits / n - 0.338) < 0.015


def test_greedy_takes_the_modal_and_one_branch():
    """Deterministic path: p >= 0.5 is the and-1, otherwise the ordinary foul."""
    scale = config.AND_ONE_GAP_SCALE
    ctrl = GameController(FakeSim(), seed=0, greedy=True)
    ctrl.possession = AWAY
    ctrl.sim.append_event("shot", "A", "paint", "made", time=0)
    ctrl.sim.script(player=["F", "B", "G"], type=[SHOOTING_2PT], result=["made", "made"],
                    delta=[scale * 0.8, 12.0])           # p = 0.2 -> the ordinary foul
    ctrl._do_foul(delta=5.0)
    assert len([r for r in rows(ctrl) if r["type"] == "free throw"]) == 2

    ctrl = GameController(FakeSim(), seed=0, greedy=True)
    ctrl.possession = AWAY
    ctrl.sim.append_event("shot", "A", "paint", "made", time=0)
    ctrl.sim.script(player=["G"], type=[SHOOTING_2PT], result=["made"],
                    delta=[scale * 0.2])                 # p = 0.8 -> the and-1
    ctrl._do_foul(delta=5.0)
    assert len([r for r in rows(ctrl) if r["type"] == "free throw"]) == 1


def test_rebounding_foul_is_masked_to_common_types():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F", "A"], type=["personal"])   # fouler F, then his victim
    ctrl._do_foul(delta=5.0, rebounding=True)   # a foul during a rebound
    type_call = [c for c in ctrl.sim.calls if c[0] == "type"][0]
    assert not any(t in type_call[3] for t in SHOOTING_FOUL_TYPES)   # never on a rebound
    foul = rows(ctrl)[-1]
    assert (foul["event"], foul["type"]) == ("foul", "personal")


def test_offensive_foul_is_a_turnover_no_fts():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A", "F"], type=["offensive"])  # fouler A on offense, victim F
    ctrl._do_foul(delta=5.0)
    (foul,) = rows(ctrl)
    assert (foul["type"], foul["result"]) == ("offensive", "cop")
    assert ctrl.possession == AWAY


def test_common_foul_nothing_when_not_in_bonus():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F", "A"], type=["personal"])   # away defender, not in penalty
    ctrl._do_foul(delta=5.0)
    (foul,) = rows(ctrl)
    assert (foul["type"], foul["result"]) == ("personal", "nothing")
    assert ctrl.possession == HOME          # offense keeps the ball
    assert ctrl.team_fouls[AWAY] == 1


def test_bonus_common_foul_awards_two_free_throws():
    ctrl = make_controller(HOME)
    ctrl.team_fouls[AWAY] = 4                # this common foul is the 5th → penalty
    ctrl.sim.script(player=["F", "A"], type=["personal"], result=["missed", "made"])
    ctrl._do_foul(delta=5.0)

    foul = rows(ctrl)[0]
    fts = rows(ctrl)[1:]
    assert foul["result"] == "free throw"
    assert len(fts) == 2
    assert ctrl.score[HOME] == 1            # one made FT
    assert ctrl.possession == AWAY          # made last FT → flip


def test_flagrant2_ejects_fouler_and_keeps_possession():
    ctrl = make_controller(HOME)
    # fouler G (away); replacement from bench is none here (full == on-court), so no sub row.
    ctrl.sim.script(player=["G", "A"], type=["flagrant-2"], result=["made", "made"])
    ctrl._do_foul(delta=5.0)
    assert "G" in ctrl.ejected
    assert "G" not in ctrl.sim.away_full
    assert ctrl.score[HOME] == 2
    assert ctrl.possession == HOME          # flagrant: fouled team retains the ball


# ===================================================================== #
# Side-aware fouls — the fouler's side is resolved before the type is sampled
# ===================================================================== #

def test_offense_side_fouler_is_masked_to_offensive_side_types():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A", "F"], type=["offensive"])  # A is on the offense; victim F
    ctrl._do_foul(delta=5.0)

    type_call = [c for c in ctrl.sim.calls if c[0] == "type"][0]
    allowed = type_call[3]
    # An offensive player cannot commit a shooting, personal, take or away-from-play foul.
    assert not any(t in allowed for t in SHOOTING_FOUL_TYPES)
    assert "personal" not in allowed
    assert "away from play" not in allowed
    assert "personal take" not in allowed and "transition take" not in allowed
    assert set(allowed) == {"offensive", "loose ball", "technical", "flagrant-1", "flagrant-2"}


def test_defense_side_fouler_is_masked_to_everything_but_offensive():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F", "A"], type=[SHOOTING_2PT], result=["made", "made"])
    ctrl._do_foul(delta=5.0)

    allowed = [c for c in ctrl.sim.calls if c[0] == "type"][0][3]
    assert "offensive" not in allowed       # a defender cannot commit an offensive foul
    assert SHOOTING_2PT in allowed


def test_foul_side_is_resolved_before_the_type_is_sampled():
    """The fouler must be picked first — the mask depends on which side he turns out to be on."""
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A", "F"], type=["offensive"])
    ctrl._do_foul(delta=5.0)

    kinds = [c[0] for c in ctrl.sim.calls if c[0] in ("player", "type")]
    # fouler, then type, then the victim -- the mask depends on the fouler's side.
    assert kinds[:2] == ["player", "type"]


def test_offense_side_technical_sends_free_throws_to_the_defense():
    ctrl = make_controller(HOME)                 # home has the ball
    # A (home, on offense) picks up a technical: the AWAY team shoots it, not home.
    ctrl.sim.script(player=["A", "F"], type=["technical"], result=["made"])
    ctrl._do_foul(delta=5.0)

    fts = [r for r in rows(ctrl) if r["type"] == "free throw"]
    assert len(fts) == 1 and fts[0]["player"] == "F"
    assert ctrl.score[AWAY] == 1 and ctrl.score[HOME] == 0
    assert ctrl.possession == HOME               # a technical does not change possession
    assert ctrl.team_fouls[HOME] == 0            # technicals never count toward the penalty


def test_offense_side_flagrant_sends_free_throws_and_the_ball_to_the_defense():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A", "F"], type=["flagrant-1"], result=["made", "made"])
    ctrl._do_foul(delta=5.0)

    assert ctrl.score[AWAY] == 2 and ctrl.score[HOME] == 0
    assert ctrl.possession == AWAY               # flagrant: the fouled team gets the ball
    assert ctrl.team_fouls[HOME] == 1            # charged to the fouling team, offense or not


def test_offense_side_loose_ball_foul_counts_and_keeps_possession():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A", "F"], type=["loose ball"])
    ctrl._do_foul(delta=5.0)

    (foul,) = rows(ctrl)
    assert (foul["player"], foul["type"], foul["result"]) == ("A", "loose ball", "nothing")
    assert ctrl.possession == HOME               # possession unchanged
    assert ctrl.team_fouls[HOME] == 1            # still a team foul against the offense


def test_team_of_resolves_a_bench_player_not_on_the_floor():
    """Reading the on-court five resolved every bench player to AWAY."""
    ctrl = GameController(FakeSim(), seed=0)
    ctrl.start(HOME_FIVE + ["K"], AWAY_FIVE + ["L"],
               home_starters=HOME_FIVE, away_starters=AWAY_FIVE)

    assert "K" not in ctrl.sim.home_roster and "L" not in ctrl.sim.away_roster   # both benched
    assert ctrl._team_of("K") == HOME
    assert ctrl._team_of("L") == AWAY
    assert ctrl._team_of("A") == HOME and ctrl._team_of("F") == AWAY
    assert ctrl._team_of("start") is None        # sentinels are not players


def test_foul_by_a_subbed_off_player_still_resolves_to_his_own_team():
    ctrl = GameController(FakeSim(), seed=0)
    ctrl.start(HOME_FIVE + ["K"], AWAY_FIVE, home_starters=HOME_FIVE, away_starters=AWAY_FIVE)
    ctrl._apply_sub("A", "K")                    # A leaves the floor for K
    assert "A" not in ctrl.sim.home_roster
    # A is off the floor but still a home player — previously he resolved to AWAY.
    assert ctrl._team_of("A") == HOME


def test_and_one_survives_the_possession_flip_on_the_made_basket():
    """A made FG flips possession, so the and-1 foul must not read as an offensive-side foul."""
    ctrl = make_controller(AWAY)                 # made FG already flipped possession to AWAY
    ctrl.sim.append_event("shot", "A", "paint", "made", time=0)   # A (home) just scored
    ctrl.sim.script(player=["G"], type=[SHOOTING_2PT], result=["made"], delta=[0.0])
    ctrl._do_foul(delta=5.0)

    allowed = [c for c in ctrl.sim.calls if c[0] == "type"][0][3]
    assert SHOOTING_2PT in allowed               # G is still the defender on that possession


# ===================================================================== #
# Clock / period structure
# ===================================================================== #

def test_team_fouls_reset_on_period_boundary():
    ctrl = make_controller(HOME)
    ctrl.team_fouls = {HOME: 3, AWAY: 4}
    ctrl.team_fouls_window = {HOME: 1, AWAY: 2}
    ctrl.clock = PERIOD_LENGTH + 1          # into Q2
    ctrl._check_period()
    assert ctrl.team_fouls == {HOME: 0, AWAY: 0}
    assert ctrl.team_fouls_window == {HOME: 0, AWAY: 0}


# ===================================================================== #
# The last-two-minutes penalty
# ===================================================================== #

def test_last_two_minutes_penalty_needs_a_second_foul_inside_the_window():
    ctrl = make_controller(HOME)
    ctrl.clock = PERIOD_LENGTH - 90         # 1:30 left in Q1 — inside the window
    ctrl.team_fouls[AWAY] = 2               # two period fouls, but none yet in the window
    # The old rule read the period total here and called this the penalty already.
    assert ctrl._in_bonus(AWAY) is False    # still one free foul in here

    ctrl._count_team_foul(AWAY)             # first foul in the window
    assert ctrl.team_fouls_window[AWAY] == 1
    assert ctrl._in_bonus(AWAY) is False

    ctrl._count_team_foul(AWAY)             # second foul in the window → penalty
    assert ctrl.team_fouls[AWAY] == 4       # still under 5, so this is the window rule firing
    assert ctrl._in_bonus(AWAY) is True


def test_fifth_period_foul_is_the_penalty_regardless_of_the_window():
    ctrl = make_controller(HOME)
    ctrl.clock = 60.0                       # early in Q1, nowhere near the window
    ctrl.team_fouls[AWAY] = 5
    assert ctrl._in_bonus(AWAY) is True


def test_fouls_before_the_window_do_not_count_toward_it():
    ctrl = make_controller(HOME)
    ctrl.clock = 60.0                       # early in the period
    ctrl._count_team_foul(AWAY)
    ctrl._count_team_foul(AWAY)
    assert ctrl.team_fouls[AWAY] == 2
    assert ctrl.team_fouls_window[AWAY] == 0     # neither was inside the last 2:00

    ctrl.clock = PERIOD_LENGTH - 30              # now inside the window
    assert ctrl._in_bonus(AWAY) is False         # two period fouls, none in the window


# ===================================================================== #
# Timeouts
# ===================================================================== #

def _timeout_ctrl(possession=HOME):
    ctrl = GameController(FakeSim(timeouts=True), seed=0)
    ctrl.possession = possession
    return ctrl


def test_a_timeout_is_only_offered_at_a_dead_ball():
    ctrl = _timeout_ctrl()
    ctrl.ball_dead = False
    assert "timeout" not in ctrl._event_menu(post_miss=False)
    ctrl.ball_dead = True
    assert "timeout" in ctrl._event_menu(post_miss=False)
    assert "timeout" not in ctrl.open_play_events        # never in the base menu itself


def test_a_timeout_is_offered_after_a_made_basket():
    """The ball is live for rebound purposes after a basket, but the team inbounding may call
    time -- and 60% of real timeouts are called exactly there. Gating on the dead ball alone
    never put that context on the menu."""
    ctrl = _timeout_ctrl()
    ctrl.ball_dead = False
    assert "timeout" not in ctrl._event_menu(post_miss=False)
    ctrl.sim.append_event("shot", "A", "paint", "made", time=0)
    assert "timeout" in ctrl._event_menu(post_miss=False)
    ctrl.sim.append_event("shot", "A", "paint", "missed", time=5)
    assert "timeout" not in ctrl._event_menu(post_miss=True)
    ctrl.sim.append_event("shot", "A", "free throw", "made", time=9)   # a made FT is not a FG
    assert "timeout" not in ctrl._event_menu(post_miss=False)


def test_a_team_out_of_timeouts_cannot_be_offered_one():
    ctrl = _timeout_ctrl()
    ctrl.ball_dead = True
    ctrl.timeouts_left = {HOME: 0, AWAY: 0}
    assert "timeout" not in ctrl._event_menu(post_miss=False)


def test_a_timeout_emits_a_row_and_spends_the_budget():
    ctrl = _timeout_ctrl()
    ctrl.sim.script(type=["home"])
    ctrl._do_timeout(delta=5.0)

    (row,) = rows(ctrl)
    assert (row["event"], row["player"], row["type"]) == ("timeout", "none", "home")
    assert ctrl.timeouts_left[HOME] == 6 and ctrl.timeouts_left[AWAY] == 7
    assert ctrl.ball_dead is True          # the point: a substitution opportunity


def test_the_budget_is_seven_a_game():
    ctrl = _timeout_ctrl()
    assert ctrl.timeouts_left == {HOME: 7, AWAY: 7}
    for _ in range(7):
        ctrl.sim.script(type=["home"])
        ctrl._do_timeout(delta=1.0)
    assert ctrl.timeouts_left[HOME] == 0
    assert HOME not in ctrl._timeout_teams()            # masked out of the head's choices
    assert AWAY in ctrl._timeout_teams()


def test_the_fourth_quarter_caps_what_is_still_usable():
    """A team that hoarded all seven cannot spend them all in the fourth."""
    ctrl = _timeout_ctrl()
    ctrl.clock = 60.0                                   # Q1
    assert ctrl._timeouts_available(HOME) == 7
    ctrl.clock = REGULATION - 300                        # Q4, 5:00 left
    assert ctrl._timeouts_available(HOME) == 4
    ctrl.clock = REGULATION - 100                        # Q4, inside the final 3:00
    assert ctrl._timeouts_available(HOME) == 2


def test_overtime_grants_two_more_each():
    ctrl = _timeout_ctrl()
    ctrl.timeouts_left = {HOME: 1, AWAY: 0}
    ctrl.score = {HOME: 100, AWAY: 100}                  # tied, so the game opens an OT
    ctrl.clock = REGULATION + 1
    ctrl._check_period()
    assert ctrl.timeouts_left == {HOME: 3, AWAY: 2}


def test_a_bundle_without_the_head_never_calls_a_timeout():
    """Weights trained before 2.0 have no timeout_team head and no "timeout" event token."""
    ctrl = make_controller(HOME)                        # FakeSim without the timeout head
    assert ctrl.use_timeouts is False
    ctrl.ball_dead = True
    assert "timeout" not in ctrl._event_menu(post_miss=False)


# ===================================================================== #
# Team rebounds
# ===================================================================== #

def test_a_team_rebound_emits_a_row_and_picks_no_rebounder():
    """DEADBALL_REBOUND_PROB flipped possession silently and emitted NO row at all."""
    ctrl = make_controller(HOME)
    ctrl.sim.script(type=["team defensive"])
    ctrl._do_rebound(delta=2.0)

    (reb,) = rows(ctrl)
    assert (reb["event"], reb["player"], reb["type"]) == ("rebound", "none", "team defensive")
    assert not [c for c in ctrl.sim.calls if c[0] == "player"]   # nobody is credited
    assert ctrl.possession == AWAY                                # defensive board flips it


def test_a_team_offensive_rebound_keeps_possession():
    ctrl = make_controller(HOME)
    ctrl.sim.script(type=["team offensive"])
    ctrl._do_rebound(delta=2.0)

    (reb,) = rows(ctrl)
    assert (reb["type"], reb["result"]) == ("team offensive", "null")
    assert ctrl.possession == HOME
    assert ctrl.ball_dead is True          # out of bounds: inbounded, not live off the rim


def test_the_rebound_head_sees_all_four_tokens():
    ctrl = make_controller(HOME)
    ctrl.sim.script(type=["offensive"], player=["B"])
    ctrl._do_rebound(delta=2.0)
    allowed = [c for c in ctrl.sim.calls if c[0] == "type"][0][3]
    assert allowed == ["offensive", "defensive", "team offensive", "team defensive"]


def test_the_deadball_rebound_dial_is_gone():
    """It was a coin flip standing in for a distribution the head can now learn."""
    assert not hasattr(config, "DEADBALL_REBOUND_PROB")
    assert "DEADBALL_REBOUND_PROB" not in config._TUNING_KEYS


# ===================================================================== #
# The fouled player
# ===================================================================== #

def test_a_foul_row_names_who_was_fouled_and_he_shoots():
    """One player, drawn once: the foul row's victim IS the free-throw shooter."""
    ctrl = make_controller(HOME)
    ctrl.team_fouls[AWAY] = 4                    # put the defense in the penalty -> 2 FTs
    ctrl.sim.script(player=["F", "B"], type=["personal"], result=["made", "made"])
    ctrl._do_foul(delta=5.0)

    foul = rows(ctrl)[0]
    fts = [r for r in rows(ctrl) if r["type"] == "free throw"]
    assert foul["secondary_player"] == "B"       # the fouled player rides in the row
    assert len(fts) == 2 and all(r["player"] == "B" for r in fts)
    # Exactly one player draw for the victim, not one for the row and another for the shooter.
    assert len([c for c in ctrl.sim.calls if c[0] == "player" and c[1] == "shot"]) == 1


def test_the_fouled_player_comes_from_the_fouled_team():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F", "C"], type=["personal"])
    ctrl._do_foul(delta=5.0)
    victim_call = [c for c in ctrl.sim.calls if c[0] == "player" and c[1] == "shot"][0]
    assert victim_call[2] == HOME_FIVE           # F (away) fouled someone on home


def test_a_common_foul_with_no_free_throws_still_names_the_victim():
    """Drawing a foul is a skill whether or not it produced a trip to the line."""
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F", "D"], type=["personal"])
    ctrl._do_foul(delta=5.0)
    (foul,) = rows(ctrl)
    assert (foul["result"], foul["secondary_player"]) == ("nothing", "D")


def test_a_technical_names_nobody():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F", "A"], type=["technical"], result=["made"])
    ctrl._do_foul(delta=5.0)
    foul = rows(ctrl)[0]
    assert foul["secondary_player"] == "none"    # no victim; the raw data agrees
    fts = [r for r in rows(ctrl) if r["type"] == "free throw"]
    assert len(fts) == 1 and fts[0]["player"] == "A"   # an independent draw, by design


def test_a_shooting_foul_names_the_fouled_shooter():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F", "C"], type=[SHOOTING_3PT], result=["made"] * 3)
    ctrl._do_foul(delta=5.0)

    foul = rows(ctrl)[0]
    fts = [r for r in rows(ctrl) if r["type"] == "free throw"]
    assert foul["secondary_player"] == "C"
    assert len(fts) == 3 and all(r["player"] == "C" for r in fts)


def test_an_and_one_names_the_scorer_as_the_fouled_player():
    ctrl = make_controller(AWAY)                 # made FG already flipped possession
    ctrl.sim.append_event("shot", "A", "paint", "made", time=0)
    ctrl.score[HOME] = 2
    ctrl.sim.script(player=["G"], type=[SHOOTING_2PT], result=["made"], delta=[0.0])
    ctrl._do_foul(delta=5.0)

    foul = [r for r in rows(ctrl) if r["event"] == "foul"][0]
    assert foul["secondary_player"] == "A"       # the scorer was the one fouled
    # No extra draw: the and-1 victim is known structurally.
    assert not [c for c in ctrl.sim.calls if c[0] == "player" and c[1] == "shot"]


def test_an_offensive_foul_names_the_defender_who_drew_it():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A", "G"], type=["offensive"])
    ctrl._do_foul(delta=5.0)
    (foul,) = rows(ctrl)
    assert (foul["type"], foul["secondary_player"]) == ("offensive", "G")
    assert ctrl.possession == AWAY


# ===================================================================== #
# Shot zones
# ===================================================================== #

def test_the_shot_type_head_is_masked_to_the_fifteen_zones():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A"], type=["rim"], result=["made"])
    ctrl._do_shot(delta=5.0)
    allowed = [c for c in ctrl.sim.calls if c[0] == "type" and c[1] == "shot_type"][0][3]
    assert allowed == list(ZONE_TOKENS)
    assert "2pt" not in allowed and "3pt" not in allowed


def test_each_zone_scores_its_own_point_value():
    for token, want in (("rim", 2), ("paint", 2), ("mid_top", 2),
                        ("corner3_l", 3), ("top3", 3), ("heave", 3)):
        ctrl = make_controller(HOME)
        ctrl.sim.script(player=["A"], type=[token], result=["made"])
        ctrl._do_shot(delta=5.0)
        assert ctrl.score[HOME] == want, token


def test_the_per_zone_result_bias_layers_on_top_of_the_global():
    """v1.0 had one make-rate dial for every shot; fifteen zones give it a hook."""
    ctrl = make_controller(HOME)
    config.SHOT_RESULT_BIAS = {"made": 0.40, "blocked": -0.15}
    config.SHOT_RESULT_BIAS_BY_ZONE = {"rim": {"made": 0.9}}

    rim = ctrl._shot_result_bias(HOME, "rim")
    assert rim["made"] == pytest.approx(0.9 + ctrl.home_court_bias)
    assert rim["blocked"] == pytest.approx(-0.15)          # untouched keys fall through

    # A zone with no entry gets the global value.
    top = ctrl._shot_result_bias(HOME, "top3")
    assert top["made"] == pytest.approx(0.40 + ctrl.home_court_bias)


def test_the_foul_token_decides_the_free_throw_count():
    """Not a sampled shot type: the cleaner read the count off the real trip's `outof`."""
    for token, want in ((SHOOTING_2PT, 2), (SHOOTING_3PT, 3)):
        ctrl = make_controller(HOME)
        ctrl.sim.script(player=["F", "A"], type=[token], result=["made"] * want)
        ctrl._do_foul(delta=5.0)
        assert len([r for r in rows(ctrl) if r["type"] == "free throw"]) == want, token


def test_a_shooting_foul_never_samples_the_shot_type_head():
    """The phantom sample: a head trained on TAKEN shots asked about an attempt never logged."""
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F", "A"], type=[SHOOTING_3PT], result=["made", "made", "made"])
    ctrl._do_foul(delta=5.0)
    assert not [c for c in ctrl.sim.calls if c[0] == "type" and c[1] == "shot_type"]
    # The only type call was the foul-type pick itself.
    assert [c[1] for c in ctrl.sim.calls if c[0] == "type"] == ["foul_type"]


def test_an_and_one_is_one_free_throw_whatever_the_token_says():
    """The made basket already counted, so the and-1 branch overrides the token's count."""
    ctrl = make_controller(AWAY)                 # made FG already flipped possession
    ctrl.sim.append_event("shot", "A", "top3", "made", time=0)
    ctrl.score[HOME] = 3
    ctrl.sim.script(player=["G"], type=[SHOOTING_3PT], result=["made"], delta=[0.0])
    ctrl._do_foul(delta=5.0)

    fts = [r for r in rows(ctrl) if r["type"] == "free throw"]
    assert len(fts) == 1 and fts[0]["player"] == "A"
    assert ctrl.score[HOME] == 4                 # 3 (basket) + 1 (and-1 FT)


# ===================================================================== #
# Dead-ball state — what actually stops play (and so allows substitutions)
# ===================================================================== #

def test_a_foul_kills_the_ball():
    ctrl = make_controller(HOME)
    ctrl.ball_dead = False
    ctrl.sim.script(player=["F", "A"], type=["personal"])   # defensive common foul, no FTs
    ctrl._do_foul(delta=5.0)
    assert ctrl.ball_dead is True


def test_a_steal_stays_live_but_a_plain_turnover_kills_the_ball():
    live = make_controller(HOME)
    live.ball_dead = False
    live.sim.script(player=["A", "F"], type=["steal"])
    live._do_turnover(delta=5.0)
    assert live.ball_dead is False              # the defense is already going the other way

    dead = make_controller(HOME)
    dead.ball_dead = False
    dead.sim.script(player=["A"], type=["error"])
    dead._do_turnover(delta=5.0)
    assert dead.ball_dead is True               # whistle: out of bounds, travel


def test_a_live_rebound_keeps_the_ball_live():
    ctrl = make_controller(HOME)
    ctrl.ball_dead = True
    ctrl.sim.script(type=["defensive"], player=["F"])
    ctrl._do_rebound(delta=2.0)
    assert ctrl.ball_dead is False


def test_a_team_rebound_kills_the_ball():
    ctrl = make_controller(HOME)
    ctrl.ball_dead = False
    ctrl.sim.script(type=["team defensive"])   # no player pick on a team board
    ctrl._do_rebound(delta=2.0)
    assert ctrl.ball_dead is True


def test_a_made_basket_only_stops_the_clock_late_in_the_period():
    # Q1 with 1:30 left — the clock keeps running, so this is not a substitution opportunity.
    early = make_controller(HOME)
    early.clock = PERIOD_LENGTH - 90
    early.sim.script(player=["A"], type=["paint"], result=["made"])
    early._do_shot(delta=5.0)
    assert early.ball_dead is False

    # Same 1:30 left, but in Q4 — the window is two minutes there, so the ball is dead.
    late = make_controller(HOME)
    late.clock = REGULATION - 90
    late.sim.script(player=["A"], type=["paint"], result=["made"])
    late._do_shot(delta=5.0)
    assert late.ball_dead is True


def test_a_missed_shot_leaves_the_ball_live():
    ctrl = make_controller(HOME)
    ctrl.ball_dead = True
    ctrl.sim.script(player=["A"], type=["paint"], result=["missed"])
    ctrl._do_shot(delta=5.0)
    assert ctrl.ball_dead is False and ctrl.pending_rebound is True


def test_the_last_free_throw_decides_whether_the_ball_is_live():
    missed = make_controller(HOME)
    missed.sim.script(result=["made", "missed"])
    missed._free_throws("A", HOME, 2, live_last=True, retain=False)
    assert missed.ball_dead is False            # missed last FT → live rebound
    assert missed.pending_rebound is True

    made = make_controller(HOME)
    made.sim.script(result=["missed", "made"])
    made._free_throws("A", HOME, 2, live_last=True, retain=False)
    assert made.ball_dead is True               # made last FT → the other team inbounds


def test_a_period_boundary_kills_the_ball():
    ctrl = make_controller(HOME)
    ctrl.ball_dead = False
    ctrl.clock = PERIOD_LENGTH + 1              # into Q2
    ctrl._check_period()
    assert ctrl.ball_dead is True


# ===================================================================== #
# No play straddles a period boundary
# ===================================================================== #

def test_a_sampled_gap_is_clamped_at_the_buzzer():
    ctrl = make_controller(HOME)
    ctrl.clock = PERIOD_LENGTH - 10             # 10s left in Q1
    ctrl._advance_clock(600.0)                  # a gap that would run deep into Q2
    assert ctrl.clock == PERIOD_LENGTH          # stopped exactly at the buzzer, not past it


def test_the_clamp_does_not_stall_the_clock_at_a_boundary():
    """Sitting exactly on a boundary must advance into the next period, not deadlock."""
    ctrl = make_controller(HOME)
    ctrl.clock = float(PERIOD_LENGTH)
    ctrl._advance_clock(10.0)
    assert ctrl.clock > PERIOD_LENGTH


def test_substitutions_wait_for_a_legal_opportunity():
    ctrl = GameController(FakeSim(sub_count=1), seed=0)
    ctrl.sim.home_full = HOME_FIVE + ["K"]

    ctrl.can_sub = False
    ctrl._run_substitutions()
    assert ctrl.sim.home_roster == HOME_FIVE    # no opportunity: nobody moves

    ctrl.can_sub = True
    ctrl.sim.script(player=["A"], incoming=["K"])
    ctrl._run_substitutions()
    assert "K" in ctrl.sim.home_roster          # the next opportunity catches it


def test_game_ends_at_regulation_when_not_tied():
    ctrl = make_controller(HOME)
    ctrl.score = {HOME: 100, AWAY: 98}
    ctrl.clock = REGULATION
    ctrl._check_period()
    assert ctrl.finished is True


def test_tie_at_regulation_opens_overtime():
    ctrl = make_controller(HOME)
    ctrl.score = {HOME: 100, AWAY: 100}
    ctrl.clock = REGULATION
    ctrl._check_period()
    assert ctrl.finished is False
    assert ctrl.period_end == REGULATION + OT_LENGTH


def test_missing_heads_raises():
    sim = FakeSim()
    sim.heads = {"player": object()}        # missing the rest
    with pytest.raises(RuntimeError):
        GameController(sim, seed=0)


# ===================================================================== #
# Sampling temperature / minutes / substitution bookkeeping
# ===================================================================== #

def test_player_temperature_passed_to_actor_picks():
    ctrl = GameController(FakeSim(), seed=0, player_temp=1.7)
    ctrl.possession = HOME
    ctrl.sim.script(player=["A"], type=["paint"], result=["missed"])
    ctrl._do_shot(delta=5.0)
    shooter_call = [c for c in ctrl.sim.calls if c[0] == "player" and c[1] == "shot"][0]
    assert shooter_call[3] == 1.7           # temperature threaded through to the player head


def test_start_takes_greedy_starters_independent_of_rollout_greedy():
    ctrl = GameController(FakeSim(), seed=0)   # in-game greedy off (default)
    ctrl.start(HOME_FIVE, AWAY_FIVE)
    call = [c for c in ctrl.sim.calls if c[0] == "start_alternating"][0]
    assert call[2] is True                  # greedy_starters always argmax
    assert call[1] is False                 # decoupled from the in-game greedy flag


def test_start_with_given_starters_skips_the_substitution_model():
    ctrl = GameController(FakeSim(), seed=0)
    ctrl.start(HOME_FIVE + ["K"], AWAY_FIVE + ["L"],
               home_starters=HOME_FIVE, away_starters=AWAY_FIVE)
    # Routed to the no-model seeding path; the model-driven path was never touched.
    seed_call = [c for c in ctrl.sim.calls if c[0] == "start_with_starters"][0]
    assert seed_call[1] == HOME_FIVE and seed_call[2] == AWAY_FIVE
    assert not [c for c in ctrl.sim.calls if c[0] == "start_alternating"]
    assert not [c for c in ctrl.sim.calls if c[0] == "incoming"]
    assert ctrl.sim.home_roster == HOME_FIVE and ctrl.sim.away_roster == AWAY_FIVE


# ===================================================================== #
# Foul-out (personal-foul disqualification)
# ===================================================================== #

def test_sixth_personal_foul_disqualifies_and_replaces():
    ctrl = make_controller(HOME)
    ctrl.sim.away_full = AWAY_FIVE + ["K"]          # a bench player to replace the DQ'd one
    ctrl.sim.script(incoming=["K"])
    for _ in range(config.FOUL_OUT_LIMIT):
        ctrl._charge_foul("F", "personal")
    assert "F" in ctrl.fouled_out
    assert "F" not in ctrl.sim.away_full            # removed for good — no sub can bring him back
    assert "K" in ctrl.sim.away_roster and "F" not in ctrl.sim.away_roster


def test_fifth_personal_foul_does_not_disqualify():
    ctrl = make_controller(HOME)
    for _ in range(config.FOUL_OUT_LIMIT - 1):
        ctrl._charge_foul("F", "personal")
    assert "F" not in ctrl.fouled_out
    assert "F" in ctrl.sim.away_roster
    assert ctrl.player_fouls["F"] == config.FOUL_OUT_LIMIT - 1


def test_technical_fouls_never_count_toward_foul_out():
    ctrl = make_controller(HOME)
    for _ in range(config.FOUL_OUT_LIMIT + 4):
        ctrl._charge_foul("F", "technical")
    assert ctrl.player_fouls.get("F", 0) == 0       # technicals are not personal fouls
    assert "F" not in ctrl.fouled_out


def test_do_foul_charges_the_fouler():
    ctrl = make_controller(AWAY)                     # F (away) is on offense → clean "nothing" foul
    # "loose ball" is the common foul an offensive player can legally commit (a "personal" is
    # masked out on that side now), and it still resolves to "nothing" outside the bonus.
    ctrl.sim.script(player=["F", "A"], type=["loose ball"])
    ctrl._do_foul(delta=5.0)
    assert ctrl.player_fouls["F"] == 1


# ===================================================================== #
# Temperature defaults
# ===================================================================== #

def test_default_player_temperature_matches_config():
    # The actor head is intentionally FLATTENED (>1) — the full-corpus retrain converges to an
    # over-concentrated head, so PLAYER_TEMPERATURE=2.0 spreads usage back to a realistic shot share
    # (see config.py). The controller must adopt the config default.
    ctrl = GameController(FakeSim(), seed=0)
    assert ctrl.player_temp == config.PLAYER_TEMPERATURE
    assert config.PLAYER_TEMPERATURE > 1.0                 # flatten over-concentration, not sharpen


def test_rebounder_uses_player_temperature():
    """The within-team rebounder pick uses the actor temperature (the off/def split is the
    type head's job, so there is no separate rebound dial)."""
    ctrl = GameController(FakeSim(), seed=0, player_temp=1.7)
    ctrl.possession = HOME
    ctrl.sim.script(type=["offensive"], player=["B"])
    ctrl._do_rebound(delta=2.0)
    reb_call = [c for c in ctrl.sim.calls if c[0] == "player" and c[1] == "rebound"][0]
    assert reb_call[3] == 1.7               # rebounder flattened like any other actor pick


def test_conditional_time_head_drives_clock_when_loaded():
    # With the conditional time head loaded, the clock advances by Δt(event, actor) — conditioned on
    # the sampled play — not by the event head's marginal Δt passed into the handler.
    sim = FakeSim()
    sim.heads["event_time_cond"] = object()
    ctrl = GameController(sim, seed=0)
    assert ctrl.use_condtime
    sim.script(player=["A"], type=["paint"], result=["missed"], delta=[18.0])
    ctrl._do_shot(delta=5.0)                 # marginal 5.0 conditions the actor pick; clock uses 18.0
    assert ctrl.player_seconds["A"] == pytest.approx(18.0 * config.DELTA_TIME_SCALE)
    assert ("delta", "shot", "A") in sim.calls


def test_marginal_delta_drives_clock_without_conditional_head():
    # Back-compat: no conditional time head → fall back to the event head's marginal Δt.
    ctrl = make_controller(HOME)
    assert not ctrl.use_condtime
    ctrl.sim.script(player=["A"], type=["paint"], result=["missed"])
    ctrl._do_shot(delta=7.0)
    assert ctrl.player_seconds["A"] == pytest.approx(7.0 * config.DELTA_TIME_SCALE)


def test_advance_clock_accrues_on_court_minutes():
    ctrl = GameController(FakeSim(), seed=0)
    ctrl._advance_clock(30.0)
    # Both on-court fives get the elapsed seconds (scaled by DELTA_TIME_SCALE); the clamp keeps a
    # huge delta bounded.
    tick = 30.0 * config.DELTA_TIME_SCALE
    assert all(ctrl.player_seconds[p] == pytest.approx(tick) for p in (*HOME_FIVE, *AWAY_FIVE))
    ctrl._advance_clock(10_000.0)
    assert ctrl.player_seconds["A"] == pytest.approx(tick + config.MAX_DELTA)  # second tick clamped to MAX_DELTA


def test_do_substitution_updates_the_stint_and_cadence_tracking():
    ctrl = GameController(FakeSim(), seed=0)
    ctrl.sim.home_full = HOME_FIVE + ["K"]
    ctrl.clock = 300.0
    ctrl.stint_start = {p: 0.0 for p in (*HOME_FIVE, *AWAY_FIVE)}
    ctrl.sim.script(player=["A"], incoming=["K"])
    ctrl._do_substitution(delta=0.0)

    # No outgoing_bias any more: SUB_FATIGUE_WEIGHT is retired, and the stint seconds it
    # approximated are an input the roster encoder reads directly.
    sub_call = [c for c in ctrl.sim.calls if c[0] == "sub"][0]
    assert sub_call[2] is None
    assert "K" in ctrl.sim.home_roster and "A" not in ctrl.sim.home_roster
    assert ctrl.stint_start["K"] == 300.0           # incoming starts a fresh stint
    assert "A" not in ctrl.stint_start              # outgoing's stint cleared
    assert ctrl.last_sub_clock[HOME] == 300.0


def test_force_sub_fires_when_team_starved():
    ctrl = GameController(FakeSim(), seed=0, sub_max_gap=300.0)
    ctrl.sim.home_full = HOME_FIVE + ["K"]
    ctrl.clock = 400.0
    ctrl.last_sub_clock = {HOME: 0.0, AWAY: 400.0}   # only HOME is overdue
    ctrl.stint_start = {p: 0.0 for p in (*HOME_FIVE, *AWAY_FIVE)}
    ctrl.sim.script(player=["A"], incoming=["K"])
    ctrl._maybe_force_sub()

    assert "K" in ctrl.sim.home_roster and "A" not in ctrl.sim.home_roster
    assert ctrl.last_sub_clock[HOME] == 400.0
    # AWAY was within the gap and has no bench — it must not have subbed.
    assert ctrl.sim.away_roster == AWAY_FIVE


def test_force_sub_skips_where_the_rules_do_not_permit_a_substitution():
    ctrl = GameController(FakeSim(), seed=0, sub_max_gap=300.0)
    ctrl.sim.home_full = HOME_FIVE + ["K"]
    ctrl.clock = 400.0
    ctrl.last_sub_clock = {HOME: 0.0, AWAY: 0.0}
    ctrl.can_sub = False                     # no legal window: even the backstop waits
    ctrl._maybe_force_sub()
    assert ctrl.sim.home_roster == HOME_FIVE
    # ...and the same team gets its sub at the next opportunity.
    ctrl.can_sub = True
    ctrl.sim.script(player=["A"], incoming=["K"])
    ctrl._maybe_force_sub()
    assert "K" in ctrl.sim.home_roster

# ===================================================================== #
# Rotation: the sub-decision head
# ===================================================================== #

def test_substitution_is_never_in_the_event_menu():
    """The event head is not trained to emit substitution, so it is never a sampled event.
    Rotation is owned by the sub-decision head and the cadence backstop, not the event stream."""
    ctrl = GameController(FakeSim(), seed=0)
    assert "substitution" not in ctrl.open_play_events
    assert "substitution" not in OPEN_PLAY_EVENTS


def test_the_head_is_asked_once_per_side_at_an_opportunity():
    ctrl = GameController(FakeSim(sub_count=0), seed=0)
    ctrl.can_sub = True
    ctrl._run_substitutions()
    asked = [c for c in ctrl.sim.calls if c[0] == "sub_count"]
    assert [c[1] for c in asked] == [HOME, AWAY]


def test_no_substitution_happens_where_the_rules_do_not_permit_one():
    """can_sub, not ball_dead: a made field goal is a dead ball and never an opportunity."""
    ctrl = GameController(FakeSim(sub_count=3), seed=0)
    ctrl.sim.home_full = HOME_FIVE + ["K"]
    ctrl.can_sub = False
    ctrl.ball_dead = True
    ctrl._run_substitutions()
    assert ctrl.sim.home_roster == HOME_FIVE
    assert not [c for c in ctrl.sim.calls if c[0] == "sub_count"]


def test_the_head_decides_how_many_come_off():
    ctrl = GameController(FakeSim(sub_count=2), seed=0)
    # Rosters set directly rather than through start(): FakeSim.start_alternating is a stub
    # that records the call and sets nothing, so a bench established before it would vanish.
    ctrl.sim.home_full = HOME_FIVE + ["K", "L"]
    ctrl.sim.away_full = AWAY_FIVE + ["M", "N"]
    ctrl.stint_start = {p: 0.0 for p in (*HOME_FIVE, *AWAY_FIVE)}
    ctrl.can_sub = True
    ctrl.sim.script(player=["A", "B", "F", "G"], incoming=["K", "L", "M", "N"])
    before = len([r for r in ctrl.sim.history if r["event"] == "substitution"])
    ctrl._run_substitutions()
    after = [r for r in ctrl.sim.history if r["event"] == "substitution"]
    assert len(after) - before == 4          # two a side


def test_a_side_with_no_bench_is_skipped_rather_than_forced():
    ctrl = GameController(FakeSim(sub_count=2), seed=0)
    ctrl.sim.home_full = list(HOME_FIVE)     # nobody available
    ctrl.can_sub = True
    ctrl._run_substitutions()
    assert ctrl.sim.home_roster == HOME_FIVE


def test_the_retired_stint_dials_are_gone():
    """The scheduler they served is gone; a dial left behind is a second opinion on the
    rotation that nothing reconciles. Follows the DEADBALL_REBOUND_PROB precedent."""
    for name in ("SUB_FATIGUE_WEIGHT", "STINT_SAMPLE_SIGMA",
                 "STINT_LENGTH_SCALE", "STINT_MAX_SECONDS"):
        assert not hasattr(config, name), name
        assert name not in config._TUNING_KEYS, name


# ===================================================================== #
# Play expansion vs. the loss mask — workstream 12's independent check
# ===================================================================== #
#
# The loss mask exists to stop the event and time heads training at positions the controller
# never queries. Which positions those are is decided by GameStateScan._continuation_of, reading
# cleaned rows; which positions they actually are is decided here, by the controller's own
# _append calls. Those are two independent routes to one number, and §10 requires they cannot
# drift, so this asserts they agree EXACTLY — not within a tolerance.
#
# The ledger form: a play that appends k rows contributes exactly k-1 continuations, and the
# first row of every play is never one. Equivalently, the set of indices the predicate calls
# "not a continuation" is exactly the set of play starts.


def _continuation_flags(ctrl):
    """Run the production scan over the controller's own emitted rows."""
    scan = GameStateScan()
    flags = []
    for row in rows(ctrl):
        scan.step(row)
        flags.append(scan.is_continuation)
    return flags


def _drive_every_play_shape():
    """One controller, driven through every expansion shape; returns it + the play-start indices."""
    ctrl = make_controller(HOME)
    starts = []

    def play(fn, **script):
        starts.append(len(rows(ctrl)))
        ctrl.sim.script(**script)
        fn()

    # 1 row: an unassisted miss.
    play(lambda: ctrl._do_shot(delta=5.0),
         player=["A"], type=["paint"], result=["missed"])
    # 1 row: the defensive board that follows it.
    play(lambda: ctrl._do_rebound(delta=2.0),
         type=["defensive"], player=["F"])
    # 2 rows: assist -> the assisted basket.
    ctrl.possession = AWAY
    play(lambda: ctrl._do_assist(delta=5.0),
         player=["F", "H"], type=["rim"])
    # 2 rows: blocked shot -> the block.
    ctrl.possession = HOME
    play(lambda: ctrl._do_shot(delta=5.0),
         player=["A", "F"], type=["paint"], result=["blocked"])
    # 1 row: a steal collapses to one turnover row.
    play(lambda: ctrl._do_turnover(delta=4.0),
         player=["A", "F"], type=["steal"])
    # 3 rows: a shooting foul -> two free throws.
    ctrl.possession = HOME
    play(lambda: ctrl._do_foul(delta=3.0),
         player=["F", "A"], type=[SHOOTING_2PT], result=["made", "made"])
    # 1 row: a controller-forced substitution (never a continuation — it is masked by the
    # separate event_target rule, and must not be double-counted here).
    ctrl.sim.home_full = HOME_FIVE + ["K"]
    ctrl.stint_start = {p: 0.0 for p in (*HOME_FIVE, *AWAY_FIVE)}
    play(lambda: ctrl._do_substitution(delta=1.0),
         player=["A"], incoming=["K"])
    return ctrl, starts


def test_continuation_rule_matches_the_controllers_own_expansion():
    ctrl, starts = _drive_every_play_shape()
    flags = _continuation_flags(ctrl)

    emitted = len(rows(ctrl))
    assert emitted > len(starts), "no play expanded — the fixture is not exercising the rule"

    # Every play start is NOT a continuation; every other row IS.
    got_starts = [i for i, f in enumerate(flags) if not f]
    assert got_starts == starts, (
        f"predicate and expansion disagree: predicate says plays start at {got_starts}, "
        f"the controller's _append calls say {starts}")

    # The same statement as a ledger, which is the form that survives a new play handler:
    # rows appended minus plays sampled is exactly the continuation count.
    assert sum(flags) == emitted - len(starts)


def test_free_throw_trip_is_continuation_all_the_way_down():
    """A three-shot trip is one query and two continuations, not three queries."""
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F", "A"], type=[SHOOTING_3PT], result=["made", "made", "made"])
    ctrl._do_foul(delta=3.0)

    flags = _continuation_flags(ctrl)
    assert [r["event"] for r in rows(ctrl)] == ["foul", "shot", "shot", "shot"]
    assert flags == [False, True, True, True]


def test_a_new_play_after_a_trip_is_not_a_continuation():
    """The trip closes: the next sampled play is a question the controller really does ask."""
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F", "A"], type=[SHOOTING_2PT], result=["made", "made"])
    ctrl._do_foul(delta=3.0)
    n = len(rows(ctrl))
    ctrl.sim.script(player=["F"], type=["rim"], result=["missed"])
    ctrl._do_shot(delta=6.0)

    flags = _continuation_flags(ctrl)
    assert flags[n] is False, "the shot after a free-throw trip must be a real query"
