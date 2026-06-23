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

from config import FOUL_OUT_LIMIT, PLAYER_TEMPERATURE
from simulation.controller import GameController, REGULATION, OT_LENGTH, PERIOD_LENGTH
from simulation.game_simulator import HOME, AWAY

HOME_FIVE = ["A", "B", "C", "D", "E"]
AWAY_FIVE = ["F", "G", "H", "I", "J"]
REQUIRED_HEADS = {"player", "substitution", "shot_type", "shot_result",
                  "assist_type", "turnover_type", "foul_type", "rebound_type"}


class FakeSim:
    """Scripted stand-in for GameSimulator — no TF graph, no artifacts."""

    def __init__(self):
        self.home_roster = list(HOME_FIVE)
        self.away_roster = list(AWAY_FIVE)
        self.home_full = list(HOME_FIVE)
        self.away_full = list(AWAY_FIVE)
        self.history: list[dict] = []
        self.heads = {k: object() for k in REQUIRED_HEADS}
        self.rng = np.random.default_rng(0)
        self.calls: list[tuple] = []
        self._q: dict[str, list] = {"player": [], "type": [], "result": [], "incoming": []}

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

    def predict_result(self, next_player, next_type, allowed, *, delta_seconds=0.0, greedy=False):
        self.calls.append(("result", next_player, next_type, list(allowed)))
        return self._pop("result")

    def predict_incoming(self, outgoing, candidates, *, delta_seconds=0.0, greedy=False):
        self.calls.append(("incoming", outgoing, list(candidates)))
        return self._pop("incoming")

    def sample_substitution(self, *, team=None, delta_seconds=0.0, greedy=False,
                            outgoing_bias=None):
        self.calls.append(("sub", team, outgoing_bias))
        return self._pop("player"), self._pop("incoming")

    def start_alternating(self, home_full, away_full, *, possession=HOME, season="2003",
                          tipoff_time=0.0, greedy=False, greedy_starters=False):
        self.calls.append(("start_alternating", greedy, greedy_starters))

    def start_with_starters(self, home_full, away_full, home_starters, away_starters,
                            *, possession=HOME, season="2003", tipoff_time=0.0):
        self.calls.append(("start_with_starters", list(home_starters), list(away_starters)))
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
    ctrl.sim.script(player=["A", "C"], type=["2pt"])  # assister A, shooter C
    ctrl._do_assist(delta=5.0)

    assist, shot = rows(ctrl)
    assert (assist["event"], assist["player"], assist["type"], assist["result"]) == \
        ("assist", "A", "2pt", "score")
    assert (shot["event"], shot["player"], shot["type"], shot["result"]) == \
        ("shot", "C", "2pt", "made")
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
    ctrl.sim.script(player=["A", "F"], type=["2pt"], result=["blocked"])  # shooter A, blocker F
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
    ctrl.sim.script(player=["A"], type=["2pt"], result=["missed"])
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
    assert type_call[3] == ["offensive", "defensive"]


# ===================================================================== #
# Turnovers / steals
# ===================================================================== #

def test_steal_emits_two_rows_and_flips_possession():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A", "F"], type=["steal"])  # committer A (home), stealer F (away)
    ctrl._do_turnover(delta=5.0)

    stealer_row, loser_row = rows(ctrl)
    assert (stealer_row["player"], stealer_row["type"], stealer_row["result"]) == \
        ("F", "steal", "steal")
    assert (loser_row["player"], loser_row["type"], loser_row["result"]) == \
        ("A", "steal", "cop")
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
    ctrl.sim.script(player=["F", "A"], type=["shooting", "2pt"], result=["made", "made"])
    ctrl._do_foul(delta=5.0)

    foul = rows(ctrl)[0]
    fts = rows(ctrl)[1:]
    assert (foul["event"], foul["type"], foul["result"]) == ("foul", "shooting", "free throw")
    assert len(fts) == 2 and all(r["type"] == "free throw" for r in fts)
    assert ctrl.score[HOME] == 2
    assert ctrl.possession == AWAY         # made last FT → other team inbounds


def test_shooting_foul_on_3pt_yields_three_free_throws():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F", "A"], type=["shooting", "3pt"],
                    result=["made", "made", "made"])
    ctrl._do_foul(delta=5.0)
    fts = rows(ctrl)[1:]
    assert len(fts) == 3                    # a 3pt shooting foul is three free throws
    assert ctrl.score[HOME] == 3


def test_and_one_keeps_basket_and_adds_one_free_throw():
    ctrl = make_controller(AWAY)            # made FG already flipped possession to AWAY
    ctrl.sim.append_event("shot", "A", "2pt", "made", time=0)   # A (home) just scored
    ctrl.score[HOME] = 2                     # the basket counted
    # An away player fouls on the made basket → and-1: A shoots a single FT.
    ctrl.sim.script(player=["G"], type=["shooting"], result=["made"])
    ctrl._do_foul(delta=5.0)

    fts = [r for r in rows(ctrl) if r["type"] == "free throw"]
    assert len(fts) == 1 and fts[0]["player"] == "A"
    assert ctrl.score[HOME] == 3            # 2 (basket) + 1 (and-1 FT)
    # The fouled attempt was a made FG, so it is the only field-goal attempt logged (no phantom).
    assert sum(1 for r in rows(ctrl) if r["type"] in ("2pt", "3pt")) == 1


def test_rebounding_foul_is_masked_to_common_types():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F"], type=["personal"])
    ctrl._do_foul(delta=5.0, rebounding=True)   # a foul during a rebound
    type_call = [c for c in ctrl.sim.calls if c[0] == "type"][0]
    assert "shooting" not in type_call[3]        # never a shooting foul on a rebound
    foul = rows(ctrl)[-1]
    assert (foul["event"], foul["type"]) == ("foul", "personal")


def test_offensive_foul_is_a_turnover_no_fts():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["A"], type=["offensive"])  # fouler A is on offense
    ctrl._do_foul(delta=5.0)
    (foul,) = rows(ctrl)
    assert (foul["type"], foul["result"]) == ("offensive", "cop")
    assert ctrl.possession == AWAY


def test_common_foul_nothing_when_not_in_bonus():
    ctrl = make_controller(HOME)
    ctrl.sim.script(player=["F"], type=["personal"])   # away defender, team not in penalty
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
# Clock / period structure
# ===================================================================== #

def test_team_fouls_reset_on_period_boundary():
    ctrl = make_controller(HOME)
    ctrl.team_fouls = {HOME: 3, AWAY: 4}
    ctrl.clock = PERIOD_LENGTH + 1          # into Q2
    ctrl._check_period()
    assert ctrl.team_fouls == {HOME: 0, AWAY: 0}


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
# Sampling temperature / minutes / fatigue-driven substitutions
# ===================================================================== #

def test_player_temperature_passed_to_actor_picks():
    ctrl = GameController(FakeSim(), seed=0, player_temp=1.7)
    ctrl.possession = HOME
    ctrl.sim.script(player=["A"], type=["2pt"], result=["missed"])
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
    for _ in range(FOUL_OUT_LIMIT):
        ctrl._charge_foul("F", "personal")
    assert "F" in ctrl.fouled_out
    assert "F" not in ctrl.sim.away_full            # removed for good — no sub can bring him back
    assert "K" in ctrl.sim.away_roster and "F" not in ctrl.sim.away_roster


def test_fifth_personal_foul_does_not_disqualify():
    ctrl = make_controller(HOME)
    for _ in range(FOUL_OUT_LIMIT - 1):
        ctrl._charge_foul("F", "personal")
    assert "F" not in ctrl.fouled_out
    assert "F" in ctrl.sim.away_roster
    assert ctrl.player_fouls["F"] == FOUL_OUT_LIMIT - 1


def test_technical_fouls_never_count_toward_foul_out():
    ctrl = make_controller(HOME)
    for _ in range(FOUL_OUT_LIMIT + 4):
        ctrl._charge_foul("F", "technical")
    assert ctrl.player_fouls.get("F", 0) == 0       # technicals are not personal fouls
    assert "F" not in ctrl.fouled_out


def test_do_foul_charges_the_fouler():
    ctrl = make_controller(AWAY)                     # F (away) is on offense → clean "nothing" foul
    ctrl.sim.script(player=["F"], type=["personal"])
    ctrl._do_foul(delta=5.0)
    assert ctrl.player_fouls["F"] == 1


# ===================================================================== #
# Temperature defaults
# ===================================================================== #

def test_default_player_temperature_is_sharpening():
    ctrl = GameController(FakeSim(), seed=0)
    assert ctrl.player_temp == PLAYER_TEMPERATURE
    assert PLAYER_TEMPERATURE < 1.0                 # sharpen, don't flatten, the actor head


def test_rebounder_uses_player_temperature():
    """The within-team rebounder pick uses the actor temperature (the off/def split is the
    type head's job, so there is no separate rebound dial)."""
    ctrl = GameController(FakeSim(), seed=0, player_temp=1.7)
    ctrl.possession = HOME
    ctrl.sim.script(type=["offensive"], player=["B"])
    ctrl._do_rebound(delta=2.0)
    reb_call = [c for c in ctrl.sim.calls if c[0] == "player" and c[1] == "rebound"][0]
    assert reb_call[3] == 1.7               # rebounder flattened like any other actor pick


def test_advance_clock_accrues_on_court_minutes():
    ctrl = GameController(FakeSim(), seed=0)
    ctrl._advance_clock(30.0)
    # Both on-court fives get the elapsed seconds; the clamp keeps a huge delta bounded.
    assert all(ctrl.player_seconds[p] == 30.0 for p in (*HOME_FIVE, *AWAY_FIVE))
    ctrl._advance_clock(10_000.0)
    assert ctrl.player_seconds["A"] == 30.0 + 40.0   # second tick clamped to MAX_DELTA


def test_fatigue_bias_weights_long_stints():
    ctrl = GameController(FakeSim(), seed=0, sub_fatigue_weight=0.1)
    ctrl.clock = 600.0
    ctrl.stint_start = {p: 0.0 for p in (*HOME_FIVE, *AWAY_FIVE)}
    ctrl.stint_start["C"] = 540.0           # C just checked in (short stint)
    bias = ctrl._fatigue_bias()
    assert bias["A"] == 0.1 * 600.0         # long stint → big nudge toward coming off
    assert bias["C"] == 0.1 * 60.0          # short stint → small nudge
    assert bias["A"] > bias["C"]


def test_do_substitution_applies_bias_and_updates_tracking():
    ctrl = GameController(FakeSim(), seed=0, sub_fatigue_weight=0.1)
    ctrl.sim.home_full = HOME_FIVE + ["K"]
    ctrl.clock = 300.0
    ctrl.stint_start = {p: 0.0 for p in (*HOME_FIVE, *AWAY_FIVE)}
    ctrl.sim.script(player=["A"], incoming=["K"])
    ctrl._do_substitution(delta=0.0)

    sub_call = [c for c in ctrl.sim.calls if c[0] == "sub"][0]
    assert sub_call[2]["A"] == 0.1 * 300.0          # bias passed for the outgoing pick
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


def test_force_sub_skips_during_pending_rebound():
    ctrl = GameController(FakeSim(), seed=0, sub_max_gap=300.0)
    ctrl.sim.home_full = HOME_FIVE + ["K"]
    ctrl.clock = 400.0
    ctrl.last_sub_clock = {HOME: 0.0, AWAY: 0.0}
    ctrl.pending_rebound = True              # mid-play: no subbing at a live ball
    ctrl._maybe_force_sub()
    assert ctrl.sim.home_roster == HOME_FIVE
