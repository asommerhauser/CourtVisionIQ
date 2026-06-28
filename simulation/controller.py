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

from config import (
    DEADBALL_REBOUND_PROB, DELTA_TIME_SCALE, EVENT_TEMPERATURE, FOUL_OUT_LIMIT,
    HOME_COURT_SHOT_BIAS, MAX_DELTA, PLAYER_TEMPERATURE, SHOT_RESULT_BIAS,
    SUB_FATIGUE_WEIGHT, SUB_MAX_GAP_SECONDS,
)
from models.conditional_time_model import ConditionalTimeModel
from models.stint_length_model import StintLengthModel
from models.substitution_model import START_TOKEN
from simulation.game_simulator import GameSimulator, HOME, AWAY

# --- Game structure (NBA) ---
PERIOD_LENGTH = 720          # 12:00 regulation quarter (seconds)
OT_LENGTH = 300              # 5:00 overtime period
REGULATION = 4 * PERIOD_LENGTH  # 2880s (48:00)
MAX_EVENTS = 4000            # hard safety cap on rollout length (≈ 8× a real game)

# Legal next-events the event head is masked to, per context. Substitution is intentionally NOT
# here: subs are owned by the rotation scheduler (injected at stint expiry), and the retrained
# event head is not trained to emit them (its loss masks substitution targets — see
# EventTimeModel._make_dataset), so there is no substitution mass to renormalize away.
OPEN_PLAY_EVENTS = ["shot", "assist", "turnover", "foul"]
POST_MISS_EVENTS = ["rebound", "foul"]   # a rebound is only legal right after a miss

# Conditional-head token whitelists (intentional sampling / masking).
SHOT_TYPES = ["2pt", "3pt"]              # a live field goal is a 2 or a 3 (FTs come from fouls)
LIVE_SHOT_RESULTS = ["made", "missed", "blocked"]
FT_RESULTS = ["made", "missed"]
TURNOVER_TYPES = ["steal", "violation", "error"]
FOUL_TYPES = ["personal", "shooting", "offensive", "loose ball",
              "technical", "flagrant-1", "flagrant-2", "away from play"]
# A foul drawn while a missed shot is in the air to be rebounded is a loose-ball / common foul,
# never a shooting foul (the shot already happened and was logged) — masking shooting out here
# keeps us from double-counting a real missed FGA *and* awarding shooting-foul free throws.
REBOUNDING_FOUL_TYPES = ["personal", "loose ball", "away from play"]
FIELD_GOAL_TYPES = ("2pt", "3pt")
# A live rebound is offensive (shooting team keeps the ball) or defensive (possession flips).
# The rebound-type head is masked to these two; the rare "null"/team rebound is modeled
# separately by DEADBALL_REBOUND_PROB below (the ball just changes hands with no row).
REBOUND_TYPES = ["offensive", "defensive"]

# Common fouls that can trigger bonus free throws when the defense is in the penalty.
COMMON_FOULS = {"personal", "loose ball", "away from play"}
# Fouls that count toward a team's per-period foul total (the penalty count).
TEAM_FOUL_TYPES = {"shooting", "personal", "loose ball", "away from play",
                   "flagrant-1", "flagrant-2"}

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
        self.player_temp = PLAYER_TEMPERATURE if player_temp is None else player_temp
        self.sub_fatigue_weight = (SUB_FATIGUE_WEIGHT if sub_fatigue_weight is None
                                   else sub_fatigue_weight)
        self.sub_max_gap = SUB_MAX_GAP_SECONDS if sub_max_gap is None else sub_max_gap
        # Logit nudge to the home offense's made-shot outcome (away gets the negation): the rollout's
        # one source of home/away asymmetry, so win prediction isn't a coin flip. See config.
        self.home_court_bias = (HOME_COURT_SHOT_BIAS if home_court_bias is None
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
        self.ejected: set[str] = set()
        # Per-player personal-foul tally and the set already disqualified (6-foul DQ + ejections),
        # so a fouled-out/ejected player is pulled and can never be subbed back in.
        self.player_fouls: dict[str, int] = {}
        self.fouled_out: set[str] = set()
        self.pending_rebound: bool = False
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
                                         possession=possession, season=season, tipoff_time=0.0,
                                         season_context=season_context)
        else:
            self.sim.start_alternating(home_full, away_full, possession=possession,
                                       season=season, tipoff_time=0.0, greedy=self.greedy,
                                       greedy_starters=True, season_context=season_context)
        self.possession = possession
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
        allowed = POST_MISS_EVENTS if post_miss else self.open_play_events

        event, marginal = self._sample_event(allowed)

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

        self._check_period()
        if self.use_scheduler:
            self._process_scheduled_subs()
        self._maybe_force_sub()   # cadence backstop (and the legacy in-game sub path's safety net)

    def _sample_event(self, allowed: list[str]) -> tuple[str, float]:
        """Run the event/time head and pick the next event from ``allowed`` (masked).

        Returns the event and the **marginal** Δt (the EventTimeModel time head's average gap). The
        authoritative clock advance is computed per play from the conditional time head once the
        actor is known (see :meth:`_advance_for`); the marginal is the actor head's Δt conditioning.
        """
        pred = self.sim.predict_next()
        event = self.sim._masked_sample(pred["event_logits"], allowed,
                                        self.sim.encoder.encode_event, greedy=self.greedy,
                                        temperature=EVENT_TEMPERATURE)
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
                                         bias=self._shot_result_bias(offense))
        self._append("shot", shooter, stype, result)

        if result == "made":
            self._score(offense, 3 if stype == "3pt" else 2)
            self.possession = self._other(offense)      # made FG → other team inbounds
        elif result == "blocked":
            # Block → the shot is a missed FGA; the blocker is an opposing on-court player and
            # the block row carries the blocked shooter as secondary_player (data_cleaner.py:285).
            blocker = self.sim.predict_player("block", self._defense_five(),
                                              delta_seconds=0.0, greedy=self.greedy,
                                              temperature=self.player_temp)
            self._append("block", blocker, stype, "block", secondary=shooter)
            self.pending_rebound = True
        else:  # missed
            self.pending_rebound = True

    def _shot_result_bias(self, offense: str) -> dict[str, float] | None:
        """Per-shot result-logit bias: the global SHOT_RESULT_BIAS plus the home-court made nudge.

        The home offense gets ``+home_court_bias`` on "made", the away offense ``-home_court_bias``
        (symmetric, so the pooled make rate is preserved while the home/away split is tilted). Returns
        ``None`` when nothing applies so ``predict_result`` takes its raw path.
        """
        bias = dict(SHOT_RESULT_BIAS)
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
        self._score(offense, 3 if atype == "3pt" else 2)
        self.possession = self._other(offense)

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
        else:
            self._append("turnover", committer, ttype, "cop")
        self.possession = self._other(offense)

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
        if self.rng.random() < DEADBALL_REBOUND_PROB:
            self._advance_clock(delta)                 # no rebounder to time on → marginal gap
            self.possession = self._other(offense)     # out of bounds → other team
            return
        # Off/def type then the rebounder are sampled on the marginal Δt; the authoritative Δt for the
        # clock is then conditioned on the decided rebounder.
        rtype = self.sim.predict_type("rebound_type", "rebound", None, REBOUND_TYPES,
                                      delta_seconds=delta, greedy=self.greedy)
        five = self._offense_five() if rtype == "offensive" else self._defense_five()
        rebounder = self.sim.predict_player("rebound", five,
                                            delta_seconds=delta, greedy=self.greedy,
                                            temperature=self.player_temp)
        self._advance_for("rebound", rebounder, delta)
        if rtype == "offensive":                       # offensive rebound — offense retains
            self._append("rebound", rebounder, "offensive", "null")
        else:                                          # defensive rebound — possession flips
            self._append("rebound", rebounder, "defensive", "cop")
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
        """
        if self.pending_rebound or self.finished:
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
        Fires only at a dead ball (no rebound pending) and reuses the model's sub sampling.
        """
        if self.pending_rebound or self.finished:
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

        A foul drawn during a rebound is masked to common (non-shooting) types. A shooting foul
        is handled specially for free-throw *count*: a 3pt shooting foul is 3 FTs, a 2pt is 2,
        and a foul on a basket that just went in is an **and-1** (the basket counts, plus 1 FT).
        ``delta`` enters as the marginal Δt and is reassigned to the authoritative Δt after the
        fouler is chosen.
        """
        allowed_types = REBOUNDING_FOUL_TYPES if rebounding else FOUL_TYPES
        fouler = self.sim.predict_player("foul", self._all_ten(),
                                         delta_seconds=delta, greedy=self.greedy,
                                         temperature=self.player_temp)
        delta = self._advance_for("foul", fouler, delta)
        ftype = self.sim.predict_type("foul_type", "foul", fouler, allowed_types,
                                      delta_seconds=delta, greedy=self.greedy)

        if ftype == "shooting":
            self._do_shooting_foul(fouler, delta)
            return

        fouler_team = self._team_of(fouler)
        on_defense = fouler_team == self._other(self.possession)
        offense = self.possession

        # Count it toward the fouling team's per-period total (the bonus/penalty count).
        if ftype in TEAM_FOUL_TYPES and on_defense:
            self.team_fouls[fouler_team] += 1

        result, n_ft, retain = self._foul_outcome(ftype, fouler_team, on_defense)
        self._append("foul", fouler, ftype, result)
        self._charge_foul(fouler, ftype)

        if ftype == "offensive":
            # Offensive foul = turnover: the offense loses the ball (no FTs).
            self.possession = self._other(fouler_team)
            return
        if ftype == "flagrant-2":
            self._eject(fouler)
        if ftype == "technical":
            # One technical FT, possession unchanged, dead ball (no rebound on a miss).
            self._free_throws(self._pick_shooter(offense), offense, 1,
                              live_last=False, retain=True)
            return
        if n_ft > 0:
            self._free_throws(self._pick_shooter(offense), offense, n_ft,
                              live_last=not retain, retain=retain)
        # else "nothing" → defensive foul, offense retains possession, no FTs.

    def _do_shooting_foul(self, fouler: str, delta: float) -> None:
        """A shooting foul: and-1 if a basket just went in, else 2 FTs (2pt) or 3 FTs (3pt).

        And-1 — the previous row is a made field goal — keeps the basket (already scored) and
        awards a single free throw to that shooter. Otherwise the fouled attempt is *not* logged
        as a field-goal attempt (NBA scoring); we sample the intended shot type only to decide
        whether it was a 2 (2 FTs) or a 3 (3 FTs), with the fouled offensive player shooting.
        """
        self._append("foul", fouler, "shooting", "free throw")
        self._charge_foul(fouler, "shooting")
        # A shooting foul is defensive by definition: the fouled team is the fouler's opponent.
        fouler_team = self._team_of(fouler)
        shooting_team = self._other(fouler_team)

        prev = self.sim.history[-1 - 1] if len(self.sim.history) >= 2 else None  # row before foul
        and_one = (prev is not None and prev.get("event") == "shot"
                   and prev.get("result") == "made" and prev.get("type") in FIELD_GOAL_TYPES
                   and self._team_of(prev.get("player")) == shooting_team)

        if and_one:
            shooter = prev["player"]                   # the player who made the basket
            n_ft = 1                                   # the basket already counted
        else:
            shooter = self._pick_shooter(shooting_team)
            stype = self.sim.predict_type("shot_type", "shot", shooter, SHOT_TYPES,
                                          delta_seconds=0.0, greedy=self.greedy)
            n_ft = 3 if stype == "3pt" else 2          # a 3pt shooting foul is three FTs

        self.team_fouls[fouler_team] += 1              # always a defensive team foul
        self._free_throws(shooter, shooting_team, n_ft, live_last=True, retain=False)

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
        if ftype == "flagrant-1":
            return ("free throw op", 2, True)
        if ftype == "flagrant-2":
            return ("ejection", 2, True)
        # Common foul (personal / loose ball / away from play).
        if ftype in COMMON_FOULS and on_defense and self._in_bonus(fouler_team):
            return ("free throw", 2, False)
        return ("nothing", 0, False)

    def _pick_shooter(self, team: str) -> str:
        """Sample which player on ``team`` takes the awarded free throws (the fouled player)."""
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
        elif live_last and not last_made:
            self.possession = shooting_team           # missed last FT → live rebound for offense
            self.pending_rebound = True
        else:
            self.possession = self._other(shooting_team)  # made last FT → other team inbounds

    # ===================================================================== #
    # --- Clock / period / bonus bookkeeping                               --
    # ===================================================================== #

    def _advance_clock(self, delta: float) -> None:
        # DELTA_TIME_SCALE calibrates pace (>1 slows the clock → fewer possessions); MAX_DELTA clamps
        # the rare blown gap. Scale first, then clamp.
        inc = max(0.0, min(float(delta) * DELTA_TIME_SCALE, MAX_DELTA))
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

    def _in_bonus(self, team: str) -> bool:
        """NBA penalty: 5th team foul in a period, or 2nd in the final 2:00."""
        fouls = self.team_fouls[team]
        last_two_min = (self._current_period_end() - self.clock) <= 120.0
        return fouls >= 5 or (last_two_min and fouls >= 2)

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

    def _team_of(self, player: str) -> str:
        return HOME if player in self.sim.home_roster else AWAY

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
        if self.player_fouls[fouler] >= FOUL_OUT_LIMIT and fouler not in self._gone():
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
