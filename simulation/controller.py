"""
GameController — the rule engine + sampling-rollout loop (the "Controller").

This is the piece ``docs/technical_specs.md`` calls "not yet built": it wraps a loaded
:class:`~simulation.game_simulator.GameSimulator` and actually *plays one game to the final
whistle*. The simulator owns the trained models and the per-step plumbing (shape history →
tensors, run a head, constrained sampling); the Controller owns everything the models do **not**
see — the clock, the score, possession, per-period team fouls / NBA bonus, ejections — and the
**hard rules** that keep every generated step a legal basketball state.

Design: the models only ever choose *among legal options*. The Controller masks the event head
to the events that are legal in the current context, samples the actor / type / result from the
conditional heads (again masked to legal tokens), then expands forced consequences exactly as the
cleaned data encodes them (an assist is followed by a made shot; a blocked shot is a missed FGA
plus a block; a foul yields the right free throws; a steal is two turnover rows). Possession,
score, fouls and the clock are bookkept here, never by the model.

Rule references in the docstrings below point at ``data_cleaner.py`` — the cleaned-data semantics
are the source of truth, so a generated game is in the same distribution the models trained on.
"""
from __future__ import annotations

import numpy as np

# Rollout dials are read as ``config.<DIAL>`` at call time, never bound at import: the shell
# (and config.dials()) rebind them on the module between runs, and a ``from config import X``
# alias here would silently freeze the value from startup. See config._TUNING_KEYS.
import config
from models.conditional_time_model import ConditionalTimeModel
from models.game_state_features import NON_TEAM_FOUL_TYPES
from models.stint_length_model import StintLengthModel
from models.substitution_model import START_TOKEN
from simulation.game_simulator import GameSimulator, HOME, AWAY
from data_cleaner import SHOOTING_2PT, SHOOTING_3PT
from zones import ZONE_TOKENS, points_for_shot

# --- Game structure (NBA) ---
PERIOD_LENGTH = 720          # 12:00 regulation quarter (seconds)
OT_LENGTH = 300              # 5:00 overtime period
REGULATION = 4 * PERIOD_LENGTH  # 2880s (48:00)
MAX_EVENTS = 4000            # hard safety cap on rollout length (≈ 8× a real game)
# A made basket stops the clock only late in a period: the last minute of Q1–Q3, the last two
# minutes of Q4 and of every overtime. Earlier than that the ball is inbounded live and play
# continues, which is why a made basket is not by itself a substitution opportunity.
LATE_CLOCK_STOP = 60.0       # Q1–Q3
LATE_CLOCK_STOP_FINAL = 120.0  # Q4 and OT
# Timeout budget (NBA): seven a game, at most four still available in the fourth quarter, at
# most two inside the final three minutes, and two more granted per overtime.
TIMEOUTS_PER_GAME = 7
TIMEOUTS_MAX_Q4 = 4
TIMEOUTS_MAX_LATE = 2
TIMEOUTS_PER_OT = 2
TIMEOUT_LATE_SECONDS = 180.0

# Legal next-events the event head is masked to, per context. Substitution is intentionally NOT
# here: subs are owned by the rotation scheduler (injected at stint expiry), and the retrained
# event head is not trained to emit them (its loss masks substitution targets — see
# EventTimeModel._make_dataset), so there is no substitution mass to renormalize away.
OPEN_PLAY_EVENTS = ["shot", "assist", "turnover", "foul"]
POST_MISS_EVENTS = ["rebound", "foul"]   # a rebound is only legal right after a miss
# A timeout is legal only while the ball is dead and the calling team still has one. It is
# appended to whichever mask applies rather than living in either, so the gate is explicit.
TIMEOUT_EVENT = "timeout"

# Conditional-head token whitelists (intentional sampling / masking).
SHOT_TYPES = list(ZONE_TOKENS)           # a live field goal is one of the fifteen court zones
                                         # (FTs never come from this head -- they come from fouls)
LIVE_SHOT_RESULTS = ["made", "missed", "blocked"]
FT_RESULTS = ["made", "missed"]
TURNOVER_TYPES = ["steal", "violation", "error"]
# A shooting foul carries the fouled attempt's point value: the foul-type head learns the real
# share of three-shot trips in game context, rather than the controller guessing it by sampling
# a live shot-type head that was never trained to answer the question.
SHOOTING_FOUL_TYPES = (SHOOTING_2PT, SHOOTING_3PT)
FOUL_TYPES = ["personal", SHOOTING_2PT, SHOOTING_3PT, "offensive", "loose ball",
              "technical", "flagrant-1", "flagrant-2", "away from play",
              "personal take", "transition take"]
# A foul drawn while a missed shot is in the air to be rebounded is a loose-ball / common foul,
# never a shooting foul (the shot already happened and was logged) — masking shooting out here
# keeps us from double-counting a real missed FGA *and* awarding shooting-foul free throws.
REBOUNDING_FOUL_TYPES = ["personal", "loose ball", "away from play"]
FIELD_GOAL_TYPES = ZONE_TOKENS
# A rebound is offensive (shooting team keeps the ball) or defensive (possession flips), and
# either can be a TEAM rebound -- nobody credited, the ball out of bounds off someone. The head
# learns the team share from ~11.9k real examples a season instead of it being a coin flip on
# DEADBALL_REBOUND_PROB, which emitted no row at all and so taught the model nothing.
TEAM_REBOUND_TYPES = ("team offensive", "team defensive")
REBOUND_TYPES = ["offensive", "defensive", *TEAM_REBOUND_TYPES]

# Common fouls that can trigger bonus free throws when the defense is in the penalty.
COMMON_FOULS = {"personal", "loose ball", "away from play"}
# Take fouls (in the vocab and cleaned data as "personal take" / "transition take"): a deliberate
# common foul to stop the ball, awarded 1 FT + the fouled team retains (the cleaner maps both to
# "free throw op"). Not in COMMON_FOULS — their outcome is fixed, not bonus-dependent.
TAKE_FOULS = {"personal take", "transition take"}
# Which foul types each side can legally commit. The foul-type head is conditioned on the fouler
# but has no notion of which side he is on, so unmasked it will charge an offensive foul to a
# defender or type an offensive player's foul as a shooting foul — and then the outcome resolves
# for the wrong team. The fouler's side is resolved first and the head masked to that side's set.
OFFENSIVE_SIDE_FOUL_TYPES = ["offensive", "loose ball", "technical", "flagrant-1", "flagrant-2"]
DEFENSIVE_SIDE_FOUL_TYPES = [t for t in FOUL_TYPES if t != "offensive"]

# Fouls that count toward a team's per-period foul total (the penalty count) have ONE definition,
# imported from the game-state features (models/game_state_features.py:NON_TEAM_FOUL_TYPES) so the
# trained feature and the sim cannot drift: every foul except technicals and offensive fouls,
# charged to the fouling team whichever side he is on. The feature's scan resolves the team by
# roster membership and cannot see offense/defense, so the sim must not gate on it either.
def counts_as_team_foul(ftype: str) -> bool:
    """Does ``ftype`` add to the fouling team's per-period penalty count?"""
    return ftype not in NON_TEAM_FOUL_TYPES


class GameController:
    """Drive a full single-game rollout off a loaded :class:`GameSimulator`, enforcing rules."""

    def __init__(self, sim: GameSimulator, *, seed: int | None = None, greedy: bool = False,
                 player_temp: float | None = None,
                 sub_fatigue_weight: float | None = None,
                 sub_max_gap: float | None = None,
                 home_court_bias: float | None = None):
        self.sim = sim
        self.greedy = greedy
        # Sampling/rotation dials (config defaults, overridable per run/test). The rebounder is
        # sampled with the same actor temperature as every other player pick — the off/def split
        # is owned by the rebound-type head, so there is no separate rebound dial.
        self.player_temp = config.PLAYER_TEMPERATURE if player_temp is None else player_temp
        self.sub_fatigue_weight = (config.SUB_FATIGUE_WEIGHT if sub_fatigue_weight is None
                                   else sub_fatigue_weight)
        self.sub_max_gap = config.SUB_MAX_GAP_SECONDS if sub_max_gap is None else sub_max_gap
        # Logit nudge to the home offense's made-shot outcome (away gets the negation): the rollout's
        # one source of home/away asymmetry, so win prediction isn't a coin flip. See config.
        self.home_court_bias = (config.HOME_COURT_SHOT_BIAS if home_court_bias is None
                                else home_court_bias)
        if seed is not None:
            self.sim.rng = np.random.default_rng(seed)
        self.rng = self.sim.rng

        required = {"player", "substitution", "shot_type", "shot_result",
                    "assist_type", "turnover_type", "foul_type", "rebound_type"}
        missing = required - set(self.sim.heads)
        if missing:
            raise RuntimeError(
                f"GameController needs these heads loaded but they're missing: {sorted(missing)}. "
                f"Train them and load via GameSimulator.load()."
            )

        # --- Game context the models do not see ---
        self.clock: float = 0.0
        self.period_end: float = REGULATION   # grows by OT_LENGTH while tied at a boundary
        self._last_period: int = 0
        self.score = {HOME: 0, AWAY: 0}
        self.possession: str = HOME
        self.team_fouls = {HOME: 0, AWAY: 0}
        # Team fouls committed inside the final 2:00 of the current period, tracked separately
        # because the last-two-minutes penalty triggers on the second foul *in the window*, not
        # on the second of the period (a team with 4 period fouls still gets one free one).
        self.team_fouls_window = {HOME: 0, AWAY: 0}
        self.timeouts_left = {HOME: TIMEOUTS_PER_GAME, AWAY: TIMEOUTS_PER_GAME}
        # Name -> team, built once from the FULL rosters (see _build_team_map). Never read the
        # on-court five for this: a bench or subbed-off player is not in it.
        self.player_team: dict[str, str] = {}
        self._build_team_map()
        self.ejected: set[str] = set()
        # Per-player personal-foul tally and the set already disqualified (6-foul DQ + ejections),
        # so a fouled-out/ejected player is pulled and can never be subbed back in.
        self.player_fouls: dict[str, int] = {}
        self.fouled_out: set[str] = set()
        self.pending_rebound: bool = False
        # Is the ball dead right now? Internal to the controller — never written to a row, never
        # a model input. Substitutions are only legal while it is True. Before 2.0 the dead-ball
        # test was "no rebound pending", which is not the same thing at all: it treats an inbound
        # after a made basket, a live steal and a live offensive rebound as equally substitutable,
        # which is why rotations landed at the wrong moments. Starts True (pre-tip).
        self.ball_dead: bool = True
        self.finished: bool = False

        # --- Minutes / rotation bookkeeping (the model never sees these) ---
        # Accumulated on-court seconds per player, the clock each player's current stint began,
        # and the last sub time per team (drives the fatigue nudge + per-team cadence safety net).
        self.player_seconds: dict[str, float] = {}
        self.stint_start: dict[str, float] = {}
        self.last_sub_clock: dict[str, float] = {HOME: 0.0, AWAY: 0.0}

        # --- Stint-length scheduler ---
        # Substitutions are never sampled from the event head (it isn't trained to emit them);
        # rotation is owned here. When the stint-length head is loaded, each entering player is
        # committed to a stint and scheduled to exit at this clock, then pulled at the next dead
        # ball once reached (see _schedule_stint / _process_scheduled_subs). When the head is
        # absent we fall back to the cadence backstop (_maybe_force_sub) alone so a bundle without
        # the stint head still rotates (just without committed stint lengths).
        self.use_scheduler: bool = StintLengthModel.KEY in self.sim.heads
        self.stint_target_clock: dict[str, float] = {}
        self.open_play_events: list[str] = list(OPEN_PLAY_EVENTS)

        # --- Conditional time head (event→player→Δt) ---
        # When loaded, the authoritative clock advance comes from ConditionalTimeModel — Δt
        # conditioned on the sampled event + actor — instead of the EventTimeModel's marginal time
        # head. Absent (older bundle / minimal test): fall back to the marginal Δt so the rollout
        # still runs. The event head's marginal Δt is still read each step and used to condition the
        # actor pick (PlayerModel's unchanged next_delta_time contract).
        self.use_condtime: bool = ConditionalTimeModel.KEY in self.sim.heads

        # --- Timeouts ---
        # Optional like the stint and conditional-time heads: a bundle trained before 2.0 has no
        # timeout_team head and no "timeout" event token, so it simply never calls one rather
        # than asking the event head for a token it has never seen.
        self.use_timeouts: bool = "timeout_team" in self.sim.heads

    # ===================================================================== #
    # --- Setup + main loop                                                --
    # ===================================================================== #

    def start(self, home_full: list[str], away_full: list[str], *,
              possession: str = HOME, season: str = "2003",
              home_starters: list[str] | None = None,
              away_starters: list[str] | None = None,
              season_context: dict | None = None) -> "GameController":
        """Build both starting fives and set possession.

        When ``home_starters`` / ``away_starters`` are given (e.g. a real game's actual tip-off
        five), those exact starters are seeded with no model calls. Otherwise both fives are
        built via alternating H,A substitutions, with starters taken at the substitution head's
        argmax (``greedy_starters=True``, the most-likely opening five) regardless of the
        in-game ``greedy`` flag. ``season_context`` carries the pre-game rest / games-played
        givens, applied before the opening five is built (the sub head consumes rest).
        """
        if home_starters is not None and away_starters is not None:
            self.sim.start_with_starters(home_full, away_full, home_starters, away_starters,
                                         season=season, tipoff_time=0.0,
                                         season_context=season_context)
        else:
            self.sim.start_alternating(home_full, away_full,
                                       season=season, tipoff_time=0.0, greedy=self.greedy,
                                       greedy_starters=True, season_context=season_context)
        self.possession = possession
        self.ball_dead = False              # the opening tip puts the ball in play
        # Rebuild from this game's rosters, now that the simulator has been seeded.
        self._build_team_map(home_full, away_full)
        # Every starter begins a stint at tip-off (clock 0); used by the fatigue nudge.
        for player in self._all_ten():
            self.stint_start[player] = 0.0
        self.last_sub_clock = {HOME: 0.0, AWAY: 0.0}
        # Commit each starter to an opening stint (outgoing = the "start" token, as trained).
        if self.use_scheduler:
            for player in self._all_ten():
                self._schedule_stint(player, START_TOKEN)
        return self

    def run(self) -> list[dict]:
        """Play to the final whistle; return the full event history (simulator rows)."""
        while not self.finished and len(self.sim.history) < MAX_EVENTS:
            self._step()
        self._append("end", "end", "end", "end")
        return self.sim.history

    def _step(self) -> None:
        """Sample and resolve one top-level event (a "play"), updating all game context.

        Order is event → actor → Δt: the event head picks the play and a *marginal* Δt (used only to
        condition the actor pick); each handler samples its primary actor, then advances the clock by
        the authoritative Δt from the conditional time head (``_advance_for``) before resolving the
        rest of the play. The clock is advanced exactly once per step, inside the handler.
        """
        post_miss = self.pending_rebound
        self.pending_rebound = False
        event, marginal = self._sample_event(self._event_menu(post_miss))

        if event == "shot":
            self._do_shot(marginal)
        elif event == "assist":
            self._do_assist(marginal)
        elif event == "turnover":
            self._do_turnover(marginal)
        elif event == "foul":
            self._do_foul(marginal, rebounding=post_miss)
        elif event == "rebound":
            self._do_rebound(marginal)
        elif event == "substitution":
            self._do_substitution(marginal)
        elif event == TIMEOUT_EVENT:
            self._do_timeout(marginal)

        self._check_period()
        if self.use_scheduler:
            self._process_scheduled_subs()
        self._maybe_force_sub()   # cadence backstop (and the legacy in-game sub path's safety net)

    def _event_menu(self, post_miss: bool) -> list[str]:
        """The events the event head may be sampled from right now.

        A timeout is the one context-gated entry: legal only while the ball is dead, only when
        some team still has one under the budget, and only when the timeout head is loaded at
        all. Everything else is the fixed open-play / post-miss mask.
        """
        allowed = POST_MISS_EVENTS if post_miss else self.open_play_events
        if self.use_timeouts and self.ball_dead and self._timeout_teams():
            return [*allowed, TIMEOUT_EVENT]
        return list(allowed)

    def _sample_event(self, allowed: list[str]) -> tuple[str, float]:
        """Run the event/time head and pick the next event from ``allowed`` (masked).

        Returns the event and the **marginal** Δt (the EventTimeModel time head's average gap). The
        authoritative clock advance is computed per play from the conditional time head once the
        actor is known (see :meth:`_advance_for`); the marginal is the actor head's Δt conditioning.
        """
        pred = self.sim.predict_next()
        # EVENT_BIAS: per-event logit calibration dial (config.py) — the event-head sibling of
        # SHOT_RESULT_BIAS, for pulling the event mix (fouls/assists/turnovers) to real rates.
        event = self.sim._masked_sample(pred["event_logits"], allowed,
                                        self.sim.encoder.encode_event, greedy=self.greedy,
                                        temperature=config.EVENT_TEMPERATURE, bias=config.EVENT_BIAS)
        return event, pred["delta_seconds"]

    def _advance_for(self, event: str, actor: str | None, marginal: float) -> float:
        """Advance the clock by the play's Δt and return it (real seconds, pre-clamp).

        When the conditional time head is loaded and we have an actor, Δt follows the sampled play
        (``predict_delta``); otherwise we fall back to the event head's marginal Δt. The returned
        value conditions the play's type/result heads. ``_advance_clock`` applies DELTA_TIME_SCALE
        and the MAX_DELTA clamp.
        """
        if self.use_condtime and actor is not None:
            delta = self.sim.predict_delta(event, actor)
        else:
            delta = marginal
        self._advance_clock(delta)
        return delta

    # ===================================================================== #
    # --- Play handlers (each emits 1+ rows and updates context)           --
    # ===================================================================== #

    def _do_shot(self, delta: float) -> None:
        """Unassisted shot: sample shooter, advance Δt(shot, shooter), then type/result + consequences.

        ``delta`` enters as the event head's marginal Δt (conditions the shooter pick), then is
        reassigned to the authoritative Δt from the conditional time head (conditions type/result).
        """
        offense = self.possession
        shooter = self.sim.predict_player("shot", self._offense_five(),
                                          delta_seconds=delta, greedy=self.greedy,
                                          temperature=self.player_temp)
        delta = self._advance_for("shot", shooter, delta)
        stype = self.sim.predict_type("shot_type", "shot", shooter, SHOT_TYPES,
                                      delta_seconds=delta, greedy=self.greedy)
        result = self.sim.predict_result(shooter, stype, LIVE_SHOT_RESULTS,
                                         delta_seconds=delta, greedy=self.greedy,
                                         bias=self._shot_result_bias(offense, stype))
        self._append("shot", shooter, stype, result)

        if result == "made":
            self._score(offense, points_for_shot(stype))
            self.possession = self._other(offense)      # made FG → other team inbounds
            self.ball_dead = self._made_basket_stops_clock()
        elif result == "blocked":
            # Block → the shot is a missed FGA; the blocker is an opposing on-court player and
            # the block row carries the blocked shooter as secondary_player (data_cleaner.py:285).
            blocker = self.sim.predict_player("block", self._defense_five(),
                                              delta_seconds=0.0, greedy=self.greedy,
                                              temperature=self.player_temp)
            self._append("block", blocker, stype, "block", secondary=shooter)
            self.pending_rebound = True
            self.ball_dead = False          # the ball is live for the rebound
        else:  # missed
            self.pending_rebound = True
            self.ball_dead = False

    def _shot_result_bias(self, offense: str, zone: str | None = None) -> dict[str, float] | None:
        """Per-shot result-logit bias: SHOT_RESULT_BIAS, the zone override, and the home nudge.

        ``SHOT_RESULT_BIAS_BY_ZONE[zone]`` is merged on top of the global, so a zone with no entry
        just gets the global value. That hook is new in 2.0: with one make-rate dial for every
        shot, rim finishing and long-mid frequency were competing for a single number.

        The home offense then gets ``+home_court_bias`` on "made", the away offense
        ``-home_court_bias`` (symmetric, so the pooled make rate is preserved while the home/away
        split is tilted). Returns ``None`` when nothing applies so ``predict_result`` takes its
        raw path.
        """
        bias = dict(config.SHOT_RESULT_BIAS)
        if zone is not None:
            bias.update(config.SHOT_RESULT_BIAS_BY_ZONE.get(zone, {}))
        if self.home_court_bias:
            nudge = self.home_court_bias if offense == HOME else -self.home_court_bias
            bias["made"] = bias.get("made", 0.0) + nudge
        return bias or None

    def _do_assist(self, delta: float) -> None:
        """Assist → a *made* shot of the same type by a different teammate (data_cleaner.py:250).

        The assist row precedes the made shot in the cleaned data; the shooter is still sampled
        (the player head), constrained to the assister's team minus the assister. ``delta`` enters
        as the marginal Δt and is reassigned to the authoritative Δt after the assister is chosen.
        """
        offense = self.possession
        assister = self.sim.predict_player("assist", self._offense_five(),
                                           delta_seconds=delta, greedy=self.greedy,
                                           temperature=self.player_temp)
        delta = self._advance_for("assist", assister, delta)
        atype = self.sim.predict_type("assist_type", "assist", assister, SHOT_TYPES,
                                      delta_seconds=delta, greedy=self.greedy)
        self._append("assist", assister, atype, "score")

        teammates = [p for p in self._offense_five() if p != assister] or self._offense_five()
        shooter = self.sim.predict_player("shot", teammates, delta_seconds=0.0, greedy=self.greedy,
                                          temperature=self.player_temp)
        self._append("shot", shooter, atype, "made")
        self._score(offense, points_for_shot(atype))
        self.possession = self._other(offense)
        self.ball_dead = self._made_basket_stops_clock()

    def _do_turnover(self, delta: float) -> None:
        """Turnover by the offense; a steal is encoded as two rows (data_cleaner.py:343)."""
        offense = self.possession
        committer = self.sim.predict_player("turnover", self._offense_five(),
                                            delta_seconds=delta, greedy=self.greedy,
                                            temperature=self.player_temp)
        delta = self._advance_for("turnover", committer, delta)
        ttype = self.sim.predict_type("turnover_type", "turnover", committer, TURNOVER_TYPES,
                                      delta_seconds=delta, greedy=self.greedy)
        if ttype == "steal":
            stealer = self.sim.predict_player("turnover", self._defense_five(),
                                              delta_seconds=0.0, greedy=self.greedy,
                                              temperature=self.player_temp)
            self._append("turnover", stealer, "steal", "steal")   # the stealer (defender)
            self._append("turnover", committer, "steal", "cop")   # the ball-loser (offense)
            self.ball_dead = False       # a steal is live — the defense is already going
        else:
            self._append("turnover", committer, ttype, "cop")
            self.ball_dead = True        # whistle: out of bounds, travel, offensive foul
        self.possession = self._other(offense)

    def _timeouts_available(self, team: str) -> int:
        """How many timeouts ``team`` may still call right now, budget caps applied.

        Seven a game, but at most four still usable once the fourth quarter starts and at most
        two inside its final three minutes — the caps bite regardless of how many are banked, so
        a team that hoarded all seven cannot spend them in the last minute.
        """
        left = self.timeouts_left[team]
        if self._period_index() < 3:                   # Q1–Q3: no cap beyond the total
            return left
        if self._current_period_end() - self.clock <= TIMEOUT_LATE_SECONDS:
            return min(left, TIMEOUTS_MAX_LATE)
        return min(left, TIMEOUTS_MAX_Q4)

    def _timeout_teams(self) -> list[str]:
        """The teams that could call a timeout right now — the type head's mask."""
        return [t for t in (HOME, AWAY) if self._timeouts_available(t) > 0]

    def _do_timeout(self, delta: float) -> None:
        """A timeout: the seventh conditional type head picks which side called it.

        No player is involved, so the row is ``timeout / none / <home|away>`` and the head is
        masked to the teams that still have one. The ball stays dead afterwards, which is the
        whole point — this is the substitution opportunity the sim never had after a made basket.
        """
        teams = self._timeout_teams()
        if not teams:                                  # both budgets spent (mask should prevent)
            return
        team = self.sim.predict_type("timeout_team", TIMEOUT_EVENT, None, teams,
                                     delta_seconds=delta, greedy=self.greedy)
        self._advance_clock(delta)
        self._append(TIMEOUT_EVENT, "none", team, "none")
        self.timeouts_left[team] -= 1
        self.ball_dead = True

    def _do_rebound(self, delta: float) -> None:
        """Resolve a rebound after a miss: pick the off/def type, then the rebounder on that team.

        The rebound-type head decides offensive vs defensive from the game state (it learns the
        real ~25% offensive share), *then* the player head samples the rebounder from the team
        that type implies — the offense's five for an offensive rebound (keeps possession,
        ``result="null"``), the defense's five for a defensive one (flips possession,
        ``result="cop"``; data_cleaner.py:319). Rarely there is no individual rebound — a dropped
        team rebound / out-of-bounds — and the ball simply changes hands with no row.
        """
        offense = self.possession  # team that just missed
        # Off/def and team-or-not both come from the type head on the marginal Δt; the
        # authoritative Δt is then conditioned on the decided rebounder (a team rebound has
        # none, so it times on the marginal gap).
        rtype = self.sim.predict_type("rebound_type", "rebound", None, REBOUND_TYPES,
                                      delta_seconds=delta, greedy=self.greedy)
        offensive = rtype.endswith("offensive")
        team_rebound = rtype in TEAM_REBOUND_TYPES

        if team_rebound:
            rebounder = "none"                         # nobody is credited; skip the pick
            self._advance_clock(delta)
            self.ball_dead = True                      # out of bounds: the ball is inbounded
        else:
            five = self._offense_five() if offensive else self._defense_five()
            rebounder = self.sim.predict_player("rebound", five,
                                                delta_seconds=delta, greedy=self.greedy,
                                                temperature=self.player_temp)
            self._advance_for("rebound", rebounder, delta)
            self.ball_dead = False                     # a live rebound: play continues

        self._append("rebound", rebounder, rtype, "null" if offensive else "cop")
        if not offensive:                              # defensive board — possession flips
            self.possession = self._other(offense)

    def _do_substitution(self, delta: float) -> None:
        """One in-game substitution (outgoing from the floor, incoming from the bench).

        Legacy path — the event head is not trained to emit substitutions (rotation is owned by the
        scheduler), so this is effectively unreachable; kept for completeness. Advances the clock by
        the marginal Δt so the per-step clock invariant holds if it is ever reached.
        """
        self._advance_clock(delta)
        outgoing, incoming = self.sim.sample_substitution(
            delta_seconds=delta, greedy=self.greedy, outgoing_bias=self._fatigue_bias())
        self._apply_sub(outgoing, incoming)

    def _fatigue_bias(self, team: str | None = None) -> dict[str, float]:
        """Per-player outgoing-sub bonus: ``weight × current stint seconds`` for on-court players.

        Restricted to ``team``'s five when given (the safety net subs one team at a time)."""
        if not self.sub_fatigue_weight:
            return {}
        pool = self._five_of(team) if team is not None else self._all_ten()
        return {p: self.sub_fatigue_weight * (self.clock - self.stint_start.get(p, self.clock))
                for p in pool}

    def _apply_sub(self, outgoing: str, incoming: str) -> None:
        """Emit the substitution row and update minutes/stint/last-sub bookkeeping."""
        team = self._team_of(outgoing)
        self._append("substitution", outgoing, "substitution", "substitution", secondary=incoming)
        self.stint_start.pop(outgoing, None)
        self.stint_start[incoming] = self.clock
        self.last_sub_clock[team] = self.clock
        # Commit the incoming player to a fresh stint; the outgoing player's schedule is done.
        if self.use_scheduler:
            self.stint_target_clock.pop(outgoing, None)
            self._schedule_stint(incoming, outgoing)

    def _schedule_stint(self, incoming: str, outgoing: str) -> None:
        """Sample ``incoming``'s stint length and schedule their exit at ``clock + length``.

        ``outgoing`` is the player they replace (the literal ``"start"`` token for an opener),
        passed through as the stint head's outgoing conditioning.
        """
        length = self.sim.predict_stint_length(incoming, outgoing, greedy=self.greedy)
        self.stint_target_clock[incoming] = self.clock + length

    def _process_scheduled_subs(self) -> None:
        """Scheduler trigger: at a dead ball, pull each team's most-overdue committed player.

        A player is due when the clock reaches their scheduled exit; we sub out the single
        most-overdue player per team per dead ball (the next dead ball catches the next one) and
        let the substitution head pick the bench replacement. Foul-outs/ejections still pull
        players immediately elsewhere; this only governs ordinary rotation timing.

        The dead-ball test is ``self.ball_dead``, not "no rebound pending" — the old proxy let
        subs land after a live steal or an inbound following a made basket.
        """
        if not self.ball_dead or self.finished:
            return
        for team in (HOME, AWAY):
            five = self._five_of(team)
            overdue = [(self.clock - self.stint_target_clock[p], p) for p in five
                       if p in self.stint_target_clock
                       and self.clock >= self.stint_target_clock[p]]
            if not overdue:
                continue
            overdue.sort(reverse=True)          # most overdue first
            outgoing = overdue[0][1]
            bench = [p for p in (self.sim.home_full if team == HOME else self.sim.away_full)
                     if p not in five]
            if not bench:                       # nobody to bring in — push the target out, move on
                self.stint_target_clock[outgoing] = self.clock + self.sub_max_gap
                continue
            incoming = self.sim.predict_incoming(outgoing, bench, greedy=self.greedy)
            self._apply_sub(outgoing, incoming)

    def _maybe_force_sub(self) -> None:
        """Cadence safety net: force a sub for any team starved past ``sub_max_gap``.

        The event head never targets a team, so without this a team can play five men all game.
        Fires only at a real dead ball and reuses the model's sub sampling — a starved team
        therefore waits for the next whistle rather than being subbed mid-play.
        """
        if not self.ball_dead or self.finished:
            return
        for team in (HOME, AWAY):
            if self.clock - self.last_sub_clock[team] <= self.sub_max_gap:
                continue
            bench = [p for p in (self.sim.home_full if team == HOME else self.sim.away_full)
                     if p not in self._five_of(team)]
            if not bench:                       # nobody to bring in — reset the timer, move on
                self.last_sub_clock[team] = self.clock
                continue
            outgoing, incoming = self.sim.sample_substitution(
                team=team, greedy=self.greedy, outgoing_bias=self._fatigue_bias(team))
            self._apply_sub(outgoing, incoming)

    def _do_foul(self, delta: float, *, rebounding: bool = False) -> None:
        """Foul: derive result from foul type (data_cleaner.py:137) + NBA bonus, expand FTs.

        Order is fouler → side → type: the fouler is sampled from all ten, his side resolved off
        the full-roster team map, and only then is the foul-type head masked — to that side's
        legal types, intersected with the rebounding mask when a missed shot is in the air. A
        foul drawn during a rebound is masked to common (non-shooting) types. A shooting foul is
        handled specially for free-throw *count*: a 3pt shooting foul is 3 FTs, a 2pt is 2, and a
        foul on a basket that just went in is an **and-1** (the basket counts, plus 1 FT).
        ``delta`` enters as the marginal Δt and is reassigned to the authoritative Δt after the
        fouler is chosen.
        """
        fouler = self.sim.predict_player("foul", self._all_ten(),
                                         delta_seconds=delta, greedy=self.greedy,
                                         temperature=self.player_temp)
        delta = self._advance_for("foul", fouler, delta)

        # Resolve the side BEFORE typing the foul, then mask the head to what that side can
        # commit. Typing first and asking about the side afterwards is what let an offensive
        # player's foul resolve as a defensive one (and vice versa).
        fouler_team = self._team_of(fouler)
        on_defense = fouler_team == self._other(self._foul_offense())
        side_types = DEFENSIVE_SIDE_FOUL_TYPES if on_defense else OFFENSIVE_SIDE_FOUL_TYPES
        base_types = REBOUNDING_FOUL_TYPES if rebounding else FOUL_TYPES
        allowed_types = [t for t in base_types if t in side_types]

        ftype = self.sim.predict_type("foul_type", "foul", fouler, allowed_types,
                                      delta_seconds=delta, greedy=self.greedy)

        if ftype in SHOOTING_FOUL_TYPES:
            self._do_shooting_foul(fouler, fouler_team, ftype, delta)
            return

        # Free throws always go to the fouler's OPPONENT, in every branch. Reading possession
        # here instead sent an offensive player's technical or flagrant to his own team.
        ft_team = self._other(fouler_team)

        # Count it toward the fouling team's per-period total (the bonus/penalty count), on
        # whichever side he is on — see counts_as_team_foul.
        if counts_as_team_foul(ftype):
            self._count_team_foul(fouler_team)

        result, n_ft, retain = self._foul_outcome(ftype, fouler_team, on_defense)
        # Who got fouled. Sampled BEFORE the row is emitted so it can ride in secondary_player,
        # and reused as the free-throw shooter -- one player, drawn once, instead of a foul row
        # that names nobody plus an unrelated draw for the shooter. A technical has no victim
        # (the raw `opponent` column is empty for 100% of them), so it stays "none".
        fouled = "none" if ftype == "technical" else self._pick_fouled_player(ft_team)
        self._append("foul", fouler, ftype, result, secondary=fouled)
        self._charge_foul(fouler, ftype)
        self.ball_dead = True          # every foul is a whistle; _free_throws may revive it

        if ftype == "offensive":
            # Offensive foul = turnover: the offense loses the ball (no FTs). Only reachable from
            # the offense now, so the ball goes to the defense by construction.
            self.possession = ft_team
            return
        if ftype == "flagrant-2":
            self._eject(fouler)
        if ftype == "technical":
            # One technical FT to the other side, then play resumes with whoever had the ball —
            # a technical does not change possession. Dead ball, so no rebound on a miss. The
            # shooter is an independent draw here precisely because nobody was fouled.
            held = self.possession
            self._free_throws(self._pick_shooter(ft_team), ft_team, 1,
                              live_last=False, retain=True)
            self.possession = held
            return
        if n_ft > 0:
            self._free_throws(fouled, ft_team, n_ft,
                              live_last=not retain, retain=retain)
        # else "nothing" → common foul, no FTs, possession unchanged.

    def _foul_offense(self) -> str:
        """Which team counts as the offense for a foul sampled right now.

        Normally whoever has the ball. The exception is the **and-1**: a made basket flips
        possession the instant it drops (``_do_shot``), so a foul sampled as the very next play
        would classify the defender who fouled on the shot as an offensive player — masking
        ``shooting`` away and making and-1s unreachable. While the previous row is a made field
        goal, the possession that just ended is still the one this foul belongs to, so the
        scoring team is the offense and the team scored on is the defense.
        """
        prev = self.sim.history[-1] if self.sim.history else None
        if (prev is not None and prev.get("event") == "shot" and prev.get("result") == "made"
                and prev.get("type") in FIELD_GOAL_TYPES):
            scorer_team = self._team_of(prev.get("player"))
            if scorer_team is not None:
                return scorer_team
        return self.possession

    def _do_shooting_foul(self, fouler: str, fouler_team: str, ftype: str,
                          delta: float) -> None:
        """A shooting foul: and-1 if a basket just went in, else the count ``ftype`` carries.

        The free-throw count now comes from the **foul token itself** — ``shooting 3pt`` is three
        attempts, ``shooting 2pt`` is two. Before 2.0 the controller sampled the live shot-type
        head here to guess whether the fouled attempt was a 2 or a 3, which asked a head trained
        on *taken* shots a question about an attempt that was never logged. The cleaner reads the
        answer off the real trip's ``outof`` instead, so the foul-type head learns the true share
        of three-shot trips in context: who is fouling, who is shooting, where in the game.

        And-1 — the previous row is a made field goal — still overrides the count structurally:
        the basket already counted, so it is one attempt whatever the token says. That path is
        properly reachable now (see :meth:`_foul_offense`).

        ``fouler_team`` is resolved by the caller before the type is sampled, so "a shooting foul
        is defensive by definition" is true by construction: neither shooting token is in the
        offensive side's mask, so this is only ever reached for a defender.
        """
        shooting_team = self._other(fouler_team)

        # The and-1 check reads the row BEFORE the foul, so it must run before the foul row is
        # appended -- the fouled player has to be known to ride in secondary_player.
        prev = self.sim.history[-1] if self.sim.history else None
        and_one = (prev is not None and prev.get("event") == "shot"
                   and prev.get("result") == "made" and prev.get("type") in FIELD_GOAL_TYPES
                   and self._team_of(prev.get("player")) == shooting_team)

        if and_one:
            fouled = prev["player"]                    # the player who made the basket
            n_ft = 1                                   # the basket already counted
        else:
            fouled = self._pick_fouled_player(shooting_team)
            n_ft = 3 if ftype == SHOOTING_3PT else 2   # straight off the sampled foul token

        self._append("foul", fouler, ftype, "free throw", secondary=fouled)
        self._charge_foul(fouler, ftype)
        self.ball_dead = True          # whistle; _free_throws decides the state after the trip
        self._count_team_foul(fouler_team)             # always a defensive team foul
        self._free_throws(fouled, shooting_team, n_ft, live_last=True, retain=False)

    def _foul_outcome(self, ftype: str, fouler_team: str, on_defense: bool) -> tuple[str, int, bool]:
        """Map a foul to (result token, number of FTs, retain-possession) — bonus-aware.

        Mirrors ``data_cleaner.determine_foul_result`` and layers the NBA bonus on top: a common
        defensive foul that normally yields ``nothing`` instead awards 2 FTs once the defense is
        in the penalty. ``retain`` marks fouls where the fouled team keeps the ball after the FTs
        (flagrant "free throw op"/"ejection") rather than the normal made-last-FT flip.
        Shooting fouls are handled separately (see :meth:`_do_shooting_foul`).
        """
        if ftype == "technical":
            return ("free throw", 1, True)
        if ftype == "offensive":
            return ("cop", 0, False)
        if ftype in TAKE_FOULS:
            # Take foul (personal take / transition take): 1 FT, fouled team keeps the ball —
            # mirrors the cleaner's "free throw op" for both types. Only coherent from the
            # defense; a sampled offensive-side take degrades to a no-FT common foul.
            if on_defense:
                return ("free throw op", 1, True)
            return ("nothing", 0, False)
        if ftype == "flagrant-1":
            return ("free throw op", 2, True)
        if ftype == "flagrant-2":
            return ("ejection", 2, True)
        # Common foul (personal / loose ball / away from play).
        if ftype in COMMON_FOULS and on_defense and self._in_bonus(fouler_team):
            return ("free throw", 2, False)
        return ("nothing", 0, False)

    def _pick_fouled_player(self, team: str) -> str:
        """Sample which player on ``team`` was fouled — and therefore shoots any free throws.

        Drawing fouls is a skill (rim pressure, shooting motion, being the player in the bonus)
        and the model could not represent it: every foul row carried ``secondary_player="none"``,
        so there was no ground truth for who drew a foul anywhere in the corpus. The raw
        ``opponent`` column has it for 100% of non-technical fouls, and matches the actual
        free-throw shooter 99.5% of the time, so one draw serves both roles.
        """
        return self.sim.predict_player("shot", self._five_of(team),
                                       delta_seconds=0.0, greedy=self.greedy,
                                       temperature=self.player_temp)

    def _pick_shooter(self, team: str) -> str:
        """Sample a free-throw shooter with no fouled player to inherit — technicals only."""
        return self.sim.predict_player("shot", self._five_of(team),
                                       delta_seconds=0.0, greedy=self.greedy,
                                       temperature=self.player_temp)

    def _free_throws(self, shooter: str, shooting_team: str, n: int, *,
                     live_last: bool, retain: bool) -> None:
        """Emit ``n`` free throws by ``shooter``; resolve possession off the last attempt."""
        last_made = False
        for _ in range(n):
            res = self.sim.predict_result(shooter, "free throw", FT_RESULTS,
                                          delta_seconds=0.0, greedy=self.greedy)
            self._append("shot", shooter, "free throw", res)
            last_made = res == "made"
            if last_made:
                self._score(shooting_team, 1)
        if retain:
            self.possession = shooting_team           # flagrant/technical: keep the ball
            self.ball_dead = True                     # inbounded, not live off the rim
        elif live_last and not last_made:
            self.possession = shooting_team           # missed last FT → live rebound for offense
            self.pending_rebound = True
            self.ball_dead = False                    # the miss is live
        else:
            self.possession = self._other(shooting_team)  # made last FT → other team inbounds
            self.ball_dead = True

    # ===================================================================== #
    # --- Clock / period / bonus bookkeeping                               --
    # ===================================================================== #

    def _advance_clock(self, delta: float) -> None:
        # DELTA_TIME_SCALE calibrates pace (>1 slows the clock → fewer possessions); MAX_DELTA clamps
        # the rare blown gap. Scale first, then clamp.
        inc = max(0.0, min(float(delta) * config.DELTA_TIME_SCALE, config.MAX_DELTA))
        # No event straddles a period boundary. A sampled gap that would run past the buzzer stops
        # there instead, and the play resolves at the buzzer; _check_period then closes the period
        # (fouls reset, ball dead) and the next step samples inside the new one. Without this a
        # play sampled at 11:58 of a quarter could land in the middle of the next.
        inc = min(inc, max(0.0, self._current_period_end() - self.clock))
        # Credit the lineup on the floor over this interval (mirrors box_score minutes accounting:
        # the pre-resolution rosters are who played the elapsed seconds). Subs this step happen
        # afterwards at the advanced clock, so their stints start clean.
        if inc:
            for player in self._all_ten():
                self.player_seconds[player] = self.player_seconds.get(player, 0.0) + inc
        self.clock += inc

    def _check_period(self) -> None:
        """Reset team fouls at each period boundary; end the game per the clock rule."""
        period = self._period_index()
        if period != self._last_period:
            self.team_fouls = {HOME: 0, AWAY: 0}
            self.team_fouls_window = {HOME: 0, AWAY: 0}
            self.ball_dead = True           # a period break is a dead ball
            if period >= 4:                 # entering an overtime: two more timeouts each
                for team in (HOME, AWAY):
                    self.timeouts_left[team] += TIMEOUTS_PER_OT
            self._last_period = period
        # End at a period boundary only when the score is not tied; otherwise open an OT.
        while self.clock >= self.period_end:
            if self.score[HOME] != self.score[AWAY]:
                self.finished = True
                return
            self.period_end += OT_LENGTH

    def _period_index(self) -> int:
        """Monotonic period id (0–3 regulation, then one per OT) — used for foul resets."""
        if self.clock < REGULATION:
            return int(self.clock // PERIOD_LENGTH)
        return 4 + int((self.clock - REGULATION) // OT_LENGTH)

    def _current_period_end(self) -> float:
        if self.clock < REGULATION:
            return (int(self.clock // PERIOD_LENGTH) + 1) * PERIOD_LENGTH
        return REGULATION + (int((self.clock - REGULATION) // OT_LENGTH) + 1) * OT_LENGTH

    def _in_last_two_minutes(self) -> bool:
        return (self._current_period_end() - self.clock) <= 120.0

    def _made_basket_stops_clock(self) -> bool:
        """Does a made basket right now stop the clock (and so allow substitutions)?

        Only late: the last minute of Q1–Q3, the last two minutes of Q4 and of every overtime.
        Earlier the ball is inbounded live and play continues, which is exactly why a made
        basket is not by itself a substitution opportunity for most of a game.
        """
        cutoff = LATE_CLOCK_STOP if self._period_index() < 3 else LATE_CLOCK_STOP_FINAL
        return (self._current_period_end() - self.clock) <= cutoff

    def _count_team_foul(self, team: str) -> None:
        """Add one to ``team``'s per-period penalty count, and to the last-2:00 count in window.

        One entry point so the period total and the window total cannot fall out of step, and so
        both handlers (:meth:`_do_foul`, :meth:`_do_shooting_foul`) count identically.
        """
        self.team_fouls[team] += 1
        if self._in_last_two_minutes():
            self.team_fouls_window[team] += 1

    def _in_bonus(self, team: str) -> bool:
        """NBA penalty: 5th team foul in a period, or the 2nd committed inside the final 2:00.

        The window clause counts fouls *in the window*, not the period total — a team that
        reaches the final 2:00 with 2 to 4 period fouls still gets one free foul there, and only
        the second one inside the window puts them in the penalty.
        """
        if self.team_fouls[team] >= 5:
            return True
        return self._in_last_two_minutes() and self.team_fouls_window[team] >= 2

    # ===================================================================== #
    # --- Roster / possession / scoring helpers                            --
    # ===================================================================== #

    def _append(self, event: str, player: str, type: str, result: str,
                secondary: str = "none") -> dict:
        return self.sim.append_event(event, player, type, result,
                                     secondary_player=secondary, time=self.clock)

    def _score(self, team: str, pts: int) -> None:
        self.score[team] += pts

    def _five_of(self, team: str) -> list[str]:
        return self.sim.home_roster if team == HOME else self.sim.away_roster

    def _offense_five(self) -> list[str]:
        return self._five_of(self.possession)

    def _defense_five(self) -> list[str]:
        return self._five_of(self._other(self.possession))

    def _all_ten(self) -> list[str]:
        return self.sim.home_roster + self.sim.away_roster

    def _build_team_map(self, home_full: list[str] | None = None,
                        away_full: list[str] | None = None) -> None:
        """Map every player on either full roster to his team, once, at tip-off.

        Built from the FULL rosters rather than the on-court fives, and never rebuilt mid-game: a
        player's team does not change, and :meth:`_disqualify` *removes* names from ``full``, so a
        later rebuild would lose everyone who fouled out. :meth:`start` passes its own arguments
        (the game spec's rosters); the no-argument form falls back to whatever the simulator is
        holding, which is what a controller driven straight into a handler gets.
        """
        home = self.sim.home_full if home_full is None else home_full
        away = self.sim.away_full if away_full is None else away_full
        self.player_team = {p: HOME for p in home}
        self.player_team.update({p: AWAY for p in away})

    def _team_of(self, player: str) -> str | None:
        """The player's team, or ``None`` for a non-player (``start``/``end``, an unknown name).

        Reading ``sim.home_roster`` here — the on-court **five**, mutated in place by every
        substitution — resolved every bench player, every subbed-off player and every sentinel
        token to AWAY, which is how a foul-out mid-resolution could flip whose free throws they
        were. Callers that may pass a sentinel (the and-1 check) rely on the ``None``.
        """
        return self.player_team.get(player)

    @staticmethod
    def _other(team: str) -> str:
        return AWAY if team == HOME else HOME

    def _charge_foul(self, fouler: str, ftype: str) -> None:
        """Tally a personal foul against ``fouler`` and disqualify him at the 6-foul limit.

        Technicals are team/bench fouls and do not count toward the personal-foul DQ; flagrant-2
        still tallies but is ejected separately by :meth:`_do_foul`, so the ``_gone`` guard skips
        the foul-out path for an already-removed player.
        """
        if ftype == "technical":
            return
        self.player_fouls[fouler] = self.player_fouls.get(fouler, 0) + 1
        if ftype == "flagrant-2":
            return   # ejected separately by _do_foul
        if self.player_fouls[fouler] >= config.FOUL_OUT_LIMIT and fouler not in self._gone():
            self._foul_out(fouler)

    def _gone(self) -> set[str]:
        """Players removed for the rest of the game (fouled out or ejected)."""
        return self.ejected | self.fouled_out

    def _eject(self, player: str) -> None:
        """Eject a player for the rest of the game; replace immediately if on the floor."""
        self.ejected.add(player)
        self._disqualify(player)

    def _foul_out(self, player: str) -> None:
        """Disqualify a player who reached the personal-foul limit; replace if on the floor."""
        self.fouled_out.add(player)
        self._disqualify(player)

    def _disqualify(self, player: str) -> None:
        """Remove ``player`` from the game (full roster too, so no sub can bring him back) and,
        if he was on the floor, sub in the model's best available bench replacement."""
        team = self._team_of(player)
        if team is None:                      # not on either roster — nothing to remove
            return
        full = self.sim.home_full if team == HOME else self.sim.away_full
        if player in full:
            full.remove(player)
        five = self._five_of(team)
        if player in five:
            bench = [p for p in full if p not in five]
            if bench:
                incoming = self.sim.predict_incoming(player, bench, delta_seconds=0.0,
                                                     greedy=self.greedy)
                self._apply_sub(player, incoming)
