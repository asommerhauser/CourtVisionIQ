"""
GameSimulator — the generation / orchestration driver.

This is the inference-time counterpart to training: where ``EventTimeModel`` learns
``p(next event, Δt | history)``, ``GameSimulator`` *uses* that distribution to drive a
game forward one event at a time. It is the transformer-era rebuild of the legacy
pre-transformer loop (an LSTM/GRU fed a growing prefix of events and emitted an event
head + a time head, then consumed its own output) — but here the plumbing sits on top of
the trained transformer, the shared frozen ``Encoder``, and ``ModelBundle``.

What it does (this step — see docs/technical_specs.md → "Sampling-rollout … not yet built"):

  * holds the model(s) and loads them by key (``GameSimulator.load``), via ``ModelBundle``
    so future downstream heads come online for free;
  * tracks **roster + possession** state *outside* the model (the model consumes rosters
    per timestep but does not see possession — we keep that book ourselves);
  * **seeds** the opening input in the exact shape the model trained on (a ``start`` row);
  * shapes the growing event ``history`` into the model's fixed ``(1, SEQ, …)`` tensors
    (``build_model_inputs``) and runs a forward pass that returns the **raw** next-step
    distribution (``predict_next``) — reading the prediction at the last real timestep.

Substitutions are wired here: it builds each team's opening five from their whole roster
via ``start -> starter`` subs (``start_from_full_rosters``, the inference twin of the
synthesized openers in ``SubstitutionModel`` preprocessing) and generates in-game subs
(``sample_substitution``) — the outgoing player from the Player head over the active five,
the incoming player from the Substitution head over the bench, with legality applied as a
candidate mask in ``_constrained_sample``.

What it deliberately does NOT do yet (future steps): sample the event/time stream
(temperature / top-p), run to game end, enforce the rest of basketball legality (the
Controller), or aggregate Monte-Carlo rollouts. The caller samples ``predict_next()`` for now.

The per-row encode + Δt/time normalization mirrors ``EventTimeModel._build_split`` /
``preprocess`` exactly so the simulator speaks the model's language; that method is the
source of truth if the two ever need to change together.
"""
from __future__ import annotations

import numpy as np

from config import (
    MAX_SEQUENCE_LENGTH, RESULT_TEMPERATURE, ROSTER_SIZE, STINT_MAX_SECONDS,
    STINT_SAMPLE_SIGMA, SUB_INCOMING_TEMPERATURE, SUB_TEMPERATURE, TYPE_BIAS, TYPE_TEMPERATURE,
)
from models.conditional_time_model import ConditionalTimeModel
from encoder.encoder import Encoder
from models.artifacts import DEFAULT_ARTIFACTS_ROOT
from models.event_time_model import CATEGORICAL_FIELDS, EventTimeModel
from models.model_bundle import ModelBundle
from models.player_model import PlayerModel
from models.season_features import (
    DEFAULT_REST_DAYS, REST_CLIP_DAYS, TEAM_SCALAR_COLS,
)
from models.game_state_features import (
    GAME_STATE_KEYS, derive_game_state, normalize_game_state,
)
from models.stint_length_model import StintLengthModel
from models.substitution_model import START_TOKEN, SUB_EVENT, SubstitutionModel

# Neutral default for a hand-built matchup with no schedule given: treat both teams as
# mid-season (real-game predictions read the actual value from the GameInput).
DEFAULT_GAMES_PLAYED = 0.5

# result outcomes that hand the ball to the other team. The model does not consume
# possession; we track it for downstream use (and the future Controller will own the
# richer clock/score/foul bookkeeping). A made field goal also flips possession, but in
# the data that is realized through the *following* inbound/rebound events, so we key off
# the explicit change-of-possession outcomes here and leave the rest to the Controller.
POSSESSION_FLIP_RESULTS = {"cop", "steal"}

HOME, AWAY = "home", "away"

# Cleaned CSV uses the literal string "null" (e.g. an offensive rebound's result, a missing
# player). pandas' default CSV read coerces "null" -> NaN, so at TRAIN time those cells were
# encoded as str(NaN) == "nan" (a real token in the type/result vocabs; UNK in player/event).
# The simulator builds rows in memory and never goes through pandas, so we reproduce that one
# coercion here to keep generated rows on the model's training distribution.
_NULL_STRINGS = {"null"}


def _norm_cat(value):
    """Mirror pandas' CSV null coercion for a single categorical value before encoding."""
    if value is None:
        return "nan"
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    if isinstance(value, str) and value.strip().lower() in _NULL_STRINGS:
        return "nan"
    return value


class GameSimulator:
    """Drive game generation off the trained Event/Time model (and future heads)."""

    def __init__(self, model, instance: EventTimeModel):
        self.model = model                       # loaded keras model (event_time)
        self.instance = instance                 # EventTimeModel wrapper
        self.encoder: Encoder = instance.encoder
        self.norm_stats: dict = instance.norm_stats or {}
        self.sequence_length: int = instance.sequence_length

        # Extra model heads (Player / Substitution / …), keyed by KEY.
        self.heads: dict = {}
        # Stint-length head's normalization stats (its own log-stint mean/std), populated at
        # load() from that head's wrapper. Empty -> predict_stint_length reads raw (mean 0/std 1).
        self.stint_norm_stats: dict = {}
        # Conditional-time head's Δt normalization stats (its own delta mean/std), populated at
        # load(). Empty -> predict_delta falls back to the shared (event_time) norm stats.
        self.condtime_norm_stats: dict = {}
        # RNG for constrained sampling (seedable for reproducible rollouts/tests).
        self.rng = np.random.default_rng()

        # --- Game state, tracked outside the model ---
        self.home_roster: list[str] = []
        self.away_roster: list[str] = []
        # Each team's full roster (everyone available), used to mask substitution
        # candidates to the bench. Set when seeding from a GameInput spec.
        self.home_full: list[str] = []
        self.away_full: list[str] = []
        self.possession: str = HOME
        self.season: str = ""
        # --- Season context (pre-game givens; constant across the game) ---
        # Per-team season progress (games played / 82) and days of rest, plus per-player
        # days-since-last-game maps. Set from the GameInput at start; defaulted otherwise.
        self.home_games_played: float = DEFAULT_GAMES_PLAYED
        self.away_games_played: float = DEFAULT_GAMES_PLAYED
        self.home_days_rest: float = DEFAULT_REST_DAYS
        self.away_days_rest: float = DEFAULT_REST_DAYS
        self.home_rest: dict[str, float] = {}
        self.away_rest: dict[str, float] = {}
        # Growing sequence of event rows; each carries its own post-update roster
        # snapshot + absolute time. This list IS the model's input context.
        self.history: list[dict] = []

    # ===================================================================== #
    # --- Loading                                                          --
    # ===================================================================== #

    @classmethod
    def load(cls, artifacts_root: str = DEFAULT_ARTIFACTS_ROOT,
             encoder: Encoder | None = None, **kwargs) -> "GameSimulator":
        """
        Load the trained model(s) and return a ready simulator.

        Uses ``ModelBundle.load`` so every registered model with artifacts on disk is
        reloaded the same way; the Event/Time model is required (it is the game skeleton).
        Extra kwargs (e.g. ``path=``, ``sequence_length=``) flow through to each model's
        ``from_artifacts`` — handy for tests that point at a tiny trained artifact.
        """
        bundle = ModelBundle.load(root=artifacts_root, encoder=encoder, **kwargs)
        if EventTimeModel.KEY not in bundle:
            raise FileNotFoundError(
                f"No '{EventTimeModel.KEY}' artifacts under {artifacts_root!r}; "
                f"train the Event/Time model first."
            )
        sim = cls(bundle.models[EventTimeModel.KEY], bundle.instances[EventTimeModel.KEY])
        # Stash any other loaded heads for future use (none consumed yet).
        sim.heads = {k: m for k, m in bundle.models.items() if k != EventTimeModel.KEY}
        # The stint-length head carries its own log-stint norm stats (separate from the shared
        # time/rest stats); keep them on hand to denormalize its predictions.
        stint_inst = bundle.instances.get(StintLengthModel.KEY)
        if stint_inst is not None:
            sim.stint_norm_stats = dict(stint_inst.norm_stats or {})
        # The conditional-time head carries its own Δt norm stats (delta_mean/std on the raw stream).
        ct_inst = bundle.instances.get(ConditionalTimeModel.KEY)
        if ct_inst is not None:
            sim.condtime_norm_stats = dict(ct_inst.norm_stats or {})
        return sim

    # ===================================================================== #
    # --- Game setup / state                                               --
    # ===================================================================== #

    def reset(self) -> None:
        """Clear all per-game state (rosters, possession, season context, history)."""
        self.home_roster, self.away_roster = [], []
        self.home_full, self.away_full = [], []
        self.possession = HOME
        self.season = ""
        self.home_games_played = DEFAULT_GAMES_PLAYED
        self.away_games_played = DEFAULT_GAMES_PLAYED
        self.home_days_rest = DEFAULT_REST_DAYS
        self.away_days_rest = DEFAULT_REST_DAYS
        self.home_rest, self.away_rest = {}, {}
        self.history = []

    def _set_season_context(self, ctx: dict | None) -> None:
        """Apply a pre-game season-context dict (from GameInput.season_context()).

        Keys (all optional; defaults when absent): ``home_games_played`` /
        ``away_games_played`` (fraction of an 82-game season), ``home_days_rest`` /
        ``away_days_rest`` (team rest in days), and ``home_rest`` / ``away_rest`` (per-player
        days-since-last-game maps, covering bench players who may be subbed in).
        """
        if not ctx:
            return
        self.home_games_played = float(ctx.get("home_games_played", DEFAULT_GAMES_PLAYED))
        self.away_games_played = float(ctx.get("away_games_played", DEFAULT_GAMES_PLAYED))
        self.home_days_rest = float(ctx.get("home_days_rest", DEFAULT_REST_DAYS))
        self.away_days_rest = float(ctx.get("away_days_rest", DEFAULT_REST_DAYS))
        self.home_rest = dict(ctx.get("home_rest", {}) or {})
        self.away_rest = dict(ctx.get("away_rest", {}) or {})

    def start_game(self, home_roster: list[str], away_roster: list[str],
                   possession: str = HOME, season: str = "2003",
                   seed_event: dict | None = None, tipoff_time: float = 0.0,
                   season_context: dict | None = None) -> "GameSimulator":
        """
        Initialize game state and seed the input with the opening row.

        The cleaned data frames every game with an explicit ``start`` event row (all
        categorical fields = ``"start"``, ``secondary_player="none"``, time 0) carrying
        the tip-off rosters, so the default seed reproduces exactly that in-distribution
        first timestep. Pass ``seed_event`` to override (e.g. to resume from a real
        partial game); it is a dict of the categorical fields below. ``season_context``
        carries the pre-game rest / games-played givens (see ``_set_season_context``).
        """
        self.reset()
        self.home_roster = list(home_roster)
        self.away_roster = list(away_roster)
        self.possession = possession
        self.season = str(season)
        self._set_season_context(season_context)

        seed = {"event": "start", "player": "start", "type": "start",
                "result": "start", "secondary_player": "none"}
        if seed_event:
            seed.update(seed_event)
        # Seed bypasses state-rule updates (it is the frame, not a play) and snapshots
        # the tip-off rosters directly.
        self.history.append(self._make_row(time=float(tipoff_time), **seed))
        return self

    def append_event(self, event: str, player: str, type: str, result: str,
                     secondary_player: str = "none", time: float | None = None) -> dict:
        """
        Append one event to the history, applying the concrete state rules first.

        Order matters: substitutions mutate the roster and the event row must then carry
        the **post**-substitution lineup (matching how the cleaned data is laid out);
        possession is flipped on change-of-possession outcomes. The post-update rosters +
        possession are snapshotted into the returned/stored row.

        ``time`` is the absolute game clock (seconds). If omitted it carries the previous
        row's time (a zero-Δt event), keeping the sequence monotonic.
        """
        if not self.history:
            raise RuntimeError("start_game() must be called before append_event().")
        if time is None:
            time = self.history[-1]["time"]

        # 1) Roster mutation (substitution) — apply before snapshotting. Convention:
        # `player` is the outgoing player (on the active five), `secondary_player` the
        # incoming player (off the bench).
        if event == "substitution":
            self._apply_substitution(incoming=secondary_player, outgoing=player)
        # 2) Possession bookkeeping (not a model input; tracked for downstream use).
        if result in POSSESSION_FLIP_RESULTS:
            self._flip_possession()

        row = self._make_row(event=event, player=player, type=type, result=result,
                             secondary_player=secondary_player, time=float(time))
        self.history.append(row)
        return row

    def _make_row(self, *, event, player, type, result, secondary_player, time) -> dict:
        """Build a history row with the current post-update state snapshot."""
        return {
            "event": event, "player": player, "type": type, "result": result,
            "secondary_player": secondary_player, "season": self.season, "time": time,
            "roster_home": list(self.home_roster), "roster_away": list(self.away_roster),
            "possession": self.possession,
        }

    def _flip_possession(self) -> None:
        self.possession = AWAY if self.possession == HOME else HOME

    def _team_of(self, player: str) -> str | None:
        if player in self.home_roster:
            return HOME
        if player in self.away_roster:
            return AWAY
        return None

    def _apply_substitution(self, incoming: str, outgoing: str) -> None:
        """Swap ``outgoing`` for ``incoming`` on whichever roster holds the outgoing player."""
        team = self._team_of(outgoing)
        if team is None:
            # Outgoing player not on either lineup (inconsistent input) — skip rather
            # than corrupt state; the Controller will enforce legality later.
            return
        roster = self.home_roster if team == HOME else self.away_roster
        roster[roster.index(outgoing)] = incoming

    # ===================================================================== #
    # --- Constrained sampling of substitution players                     --
    # ===================================================================== #

    def _masked_sample(self, logits: np.ndarray, candidates: list[str], encode,
                       *, greedy: bool = False, temperature: float = 1.0,
                       bias: dict[str, float] | None = None) -> str:
        """Pick a token from ``candidates`` by restricting ``logits`` to their vocab ids.

        A head emits a distribution over its whole vocab; legality (which tokens are actually
        eligible — the bench/active five for players, ``{2pt, 3pt}`` for a live shot type,
        made/missed for a free throw) is a sampling-time constraint applied here. ``encode``
        maps a candidate string to its vocab id. Renormalizes over the candidate ids and either
        samples (default) or takes the argmax (``greedy``, for deterministic tests).

        ``temperature`` (>1 flattens, <1 sharpens) scales the logits before the softmax; 1.0 is
        the raw model. ``bias`` adds a per-candidate logit offset before sampling (e.g. a fatigue
        nudge on the outgoing-sub pick). Both are ignored under ``greedy`` argmax except that the
        bias still applies (it can change which candidate is the max).
        """
        if not candidates:
            raise ValueError("no candidates to sample from")
        ids = [encode(c) for c in candidates]
        sub = np.asarray(logits)[ids].astype(np.float64)
        if bias:
            sub = sub + np.array([bias.get(c, 0.0) for c in candidates], dtype=np.float64)
        if greedy:
            choice = int(np.argmax(sub))
        else:
            probs = _softmax(sub, temperature=temperature)
            choice = int(self.rng.choice(len(ids), p=probs))
        return candidates[choice]

    def _constrained_sample(self, logits: np.ndarray, candidates: list[str],
                            *, greedy: bool = False, temperature: float = 1.0,
                            bias: dict[str, float] | None = None) -> str:
        """Player-vocab specialization of :meth:`_masked_sample` (the bench/active five)."""
        return self._masked_sample(logits, candidates, self.encoder.encode_player,
                                   greedy=greedy, temperature=temperature, bias=bias)

    def _avail_mask(self) -> np.ndarray:
        """(1, V) availability over the player vocab for the player-picking heads.

        Mirrors the per-game training mask (``game_available_mask``): 1.0 for every player
        on either full roster (everyone available this game) — plus the current on-court
        five as a safety net — and 0.0 elsewhere, so the Player / Substitution heads' masked
        softmax matches train time. If no rosters are known yet (availability unknown), fall
        back to an all-ones mask (no masking), preserving the pre-mask behavior. The other
        heads ignore this key. PAD is always 0.
        """
        enc = self.encoder
        V = enc.player_vocab.next_token
        players = {*self.home_full, *self.away_full, *self.home_roster, *self.away_roster}
        if not players:
            return np.ones((1, V), dtype=np.float32)
        mask = np.zeros((1, V), dtype=np.float32)
        for p in players:
            i = enc.encode_player(p)
            if 0 <= i < V:
                mask[0, i] = 1.0
        mask[0, enc.encode_player("PAD")] = 0.0
        return mask

    def _conditioned_inputs(self, *, next_event: str, delta_seconds: float,
                            next_player: str | None = None,
                            next_type: str | None = None,
                            next_secondary_player: str | None = None) -> dict:
        """Base history inputs + the next-step conditioning a downstream head reads.

        Every head conditions on ``next_event`` / ``next_delta_time`` at the decision position
        (the last real step ``n-1``); the player head needs nothing more, the type heads add
        ``next_player`` (the decided actor), the result head also adds ``next_type`` (the type
        the shot is about to be), and the stint-length head adds ``next_secondary_player`` (the
        decided incoming player, whose stint it predicts). Only the requested conditioning is
        attached, so a head is never handed an input it doesn't define — Keras functional models
        reject a dict with unknown keys, so the player-vocab ``avail_mask`` is attached only on
        the incoming-substitution path (:meth:`_next_step_inputs`), not here. The Player head's
        on-court candidate mask is built in-graph from the roster inputs already in ``base``.
        """
        base = self.build_model_inputs()
        SEQ = self.sequence_length
        n = min(len(self.history), SEQ)
        enc = self.encoder

        def _col(encode, value) -> np.ndarray:
            arr = np.full((1, SEQ), encode("PAD"), dtype=np.int32)
            arr[0, n - 1] = encode(_norm_cat(value))
            return arr

        delta_mean = float(self.norm_stats.get("delta_mean", 0.0))
        delta_std = float(self.norm_stats.get("delta_std", 1.0)) or 1.0
        next_delta = np.zeros((1, SEQ, 1), dtype=np.float32)
        next_delta[0, n - 1, 0] = (delta_seconds - delta_mean) / delta_std

        inputs = {**base,
                  "next_event": _col(enc.encode_event, next_event),
                  "next_delta_time": next_delta}
        if next_player is not None:
            inputs["next_player"] = _col(enc.encode_player, next_player)
        if next_type is not None:
            inputs["next_type"] = _col(enc.encode_type, next_type)
        if next_secondary_player is not None:
            inputs["next_secondary_player"] = _col(enc.encode_secondary_player,
                                                   next_secondary_player)
        return inputs

    def _next_step_inputs(self, *, outgoing: str | None, delta_seconds: float) -> dict:
        """Substitution-path conditioning (``next_event`` = ``substitution`` + the outgoing
        player). Thin wrapper over :meth:`_conditioned_inputs` kept for the sub helpers.
        Adds the player-vocab ``avail_mask`` only when the incoming (Substitution) head is the
        consumer — it is the only head that defines that input (its OnCourtCandidateMask
        subtracts the on-court ten from it in-graph, matching the legal bench at train time).
        The Player head's mask is built entirely from the roster inputs already present."""
        inputs = self._conditioned_inputs(next_event=SUB_EVENT, delta_seconds=delta_seconds,
                                          next_player=outgoing)
        if outgoing is not None:  # incoming pick (SubstitutionModel) — the avail-defined head
            inputs["avail_mask"] = self._avail_mask()
        return inputs

    def _infer(self, model_key: str, inputs: dict) -> dict:
        """Run one head's forward pass and return its outputs as numpy (batch dim kept).

        The single seam every prediction routes through (``predict_next`` for the event/time head,
        ``_head_logits`` for the conditional heads). ``model_key`` is ``EventTimeModel.KEY`` for the
        core model, else a key into ``self.heads``. Subclassing/overriding **just this method** lets
        the batched rollout coordinator pool many games' forward passes into one GPU call without
        touching any rule or sampling logic (see ``simulation/batched_rollout.py``).
        """
        model = self.model if model_key == EventTimeModel.KEY else self.heads.get(model_key)
        if model is None:
            raise RuntimeError(
                f"'{model_key}' head not loaded; train it and load via GameSimulator.load()."
            )
        out = model(inputs, training=False)
        return {k: np.asarray(v) for k, v in out.items()}

    def _head_logits(self, key: str, output_name: str, inputs: dict) -> np.ndarray:
        """Run a loaded head and return its logits at the decision position (n-1)."""
        n = min(len(self.history), self.sequence_length)
        out = self._infer(key, inputs)
        return out[output_name][0, n - 1]

    def predict_outgoing(self, candidates: list[str], *, delta_seconds: float = 0.0,
                         greedy: bool = False, temperature: float = SUB_TEMPERATURE,
                         outgoing_bias: dict[str, float] | None = None) -> str:
        """Sample the outgoing player from ``candidates`` (the active roster) via PlayerModel.

        ``outgoing_bias`` adds a per-candidate logit offset (the Controller's fatigue nudge, so a
        long-stint player is more likely to be the one who comes off) on top of the model's
        learned "who usually gets subbed" distribution.
        """
        inputs = self._next_step_inputs(outgoing=None, delta_seconds=delta_seconds)
        logits = self._head_logits(PlayerModel.KEY, "player_output", inputs)
        return self._constrained_sample(logits, candidates, greedy=greedy,
                                        temperature=temperature, bias=outgoing_bias)

    def predict_incoming(self, outgoing: str, candidates: list[str], *,
                         delta_seconds: float = 0.0, greedy: bool = False,
                         temperature: float = SUB_INCOMING_TEMPERATURE) -> str:
        """Sample the incoming player from ``candidates`` (the bench) via SubstitutionModel,
        conditioned on the decided ``outgoing`` player.

        Defaults to the sharpened ``SUB_INCOMING_TEMPERATURE`` (like the actor head) so the bench
        pick follows the model's real preference instead of spreading near-uniformly across the
        bench — the deep bench should check in rarely, not as often as a rotation regular."""
        inputs = self._next_step_inputs(outgoing=outgoing, delta_seconds=delta_seconds)
        logits = self._head_logits(SubstitutionModel.KEY, "secondary_player_output", inputs)
        return self._constrained_sample(logits, candidates, greedy=greedy, temperature=temperature)

    def sample_substitution(self, *, team: str | None = None, delta_seconds: float = 0.0,
                            greedy: bool = False,
                            outgoing_bias: dict[str, float] | None = None) -> tuple[str, str]:
        """Generate one in-game substitution: (outgoing, incoming).

        The outgoing player is drawn from the active on-court roster (a given ``team`` or
        all ten); the incoming player from that team's bench (full roster minus on-court).
        ``outgoing_bias`` is the Controller's per-player fatigue nudge for the outgoing pick.
        Does not mutate state — call ``append_event('substitution', outgoing, …,
        secondary_player=incoming)`` to apply it.
        """
        if team is not None:
            active = self.home_roster if team == HOME else self.away_roster
        else:
            active = self.home_roster + self.away_roster
        outgoing = self.predict_outgoing(active, delta_seconds=delta_seconds, greedy=greedy,
                                         outgoing_bias=outgoing_bias)

        sub_team = self._team_of(outgoing)
        full = self.home_full if sub_team == HOME else self.away_full
        on_court = self.home_roster if sub_team == HOME else self.away_roster
        bench = [p for p in full if p not in on_court]
        incoming = self.predict_incoming(outgoing, bench, delta_seconds=delta_seconds,
                                          greedy=greedy)
        return outgoing, incoming

    def predict_stint_length(self, incoming: str, outgoing: str, *,
                             delta_seconds: float = 0.0, greedy: bool = False,
                             sigma: float | None = None) -> float:
        """Predict how long ``incoming`` will stay on the floor (game-seconds), via StintLengthModel.

        Conditions on the fully decided substitution — ``next_player`` (outgoing, ``"start"``
        for an opener) and ``next_secondary_player`` (incoming) — and regresses standardized
        log-stint. Denormalizes with the head's own ``stint_log_mean`` / ``stint_log_std``, then
        (unless ``greedy``) adds multiplicative log-space noise (``STINT_SAMPLE_SIGMA``) for
        rotation variety. Capped at ``STINT_MAX_SECONDS``; there is **no** lower bound — a short
        specialist stint is legitimate.
        """
        inputs = self._conditioned_inputs(
            next_event=SUB_EVENT, delta_seconds=delta_seconds,
            next_player=outgoing, next_secondary_player=incoming,
        )
        pred = self._head_logits(StintLengthModel.KEY, "stint_output", inputs)
        log_norm = float(np.ravel(pred)[0])  # (1,) regression scalar at position n-1

        mean = float(self.stint_norm_stats.get("stint_log_mean", 0.0))
        std = float(self.stint_norm_stats.get("stint_log_std", 1.0)) or 1.0
        log_stint = log_norm * std + mean

        s = STINT_SAMPLE_SIGMA if sigma is None else sigma
        if not greedy and s > 0:
            log_stint += float(self.rng.normal(0.0, s))

        seconds = float(np.expm1(log_stint))
        return max(0.0, min(seconds, STINT_MAX_SECONDS))

    # ===================================================================== #
    # --- Conditional heads (player / type / result) for the rollout       --
    # ===================================================================== #

    def predict_player(self, next_event: str, candidates: list[str], *,
                       delta_seconds: float = 0.0, greedy: bool = False,
                       temperature: float = 1.0) -> str:
        """Sample the actor of ``next_event`` from ``candidates`` via the Player head.

        ``candidates`` is the legal pool (e.g. the on-court five of the team with the ball, or
        a team's five minus a just-credited assister) — the Controller owns that legality.
        ``temperature`` (>1) flattens the head so one star doesn't take nearly every possession.
        """
        inputs = self._conditioned_inputs(next_event=next_event, delta_seconds=delta_seconds)
        logits = self._head_logits(PlayerModel.KEY, "player_output", inputs)
        return self._constrained_sample(logits, candidates, greedy=greedy, temperature=temperature)

    def predict_type(self, key: str, next_event: str, next_player: str, allowed: list[str], *,
                     delta_seconds: float = 0.0, greedy: bool = False,
                     temperature: float = TYPE_TEMPERATURE,
                     bias: dict[str, float] | None = None) -> str:
        """Sample an event's ``type`` from ``allowed`` via a conditional type head.

        ``key`` is the head (``shot_type`` / ``assist_type`` / ``turnover_type`` /
        ``foul_type``); ``allowed`` is the legal token set (e.g. ``{"2pt", "3pt"}`` for a live
        field goal). The head outputs ``type_output`` and conditions on the decided actor.

        ``bias`` adds a per-token logit offset on top of the head's ``TYPE_BIAS`` calibration
        entry (config.py, keyed by ``key``) — e.g. ``{"turnover_type": {"steal": -0.2}}`` pulls
        steal-type turnovers down without moving the overall turnover rate.
        """
        inputs = self._conditioned_inputs(next_event=next_event, delta_seconds=delta_seconds,
                                          next_player=next_player)
        logits = self._head_logits(key, "type_output", inputs)
        eff_bias = {**TYPE_BIAS.get(key, {}), **(bias or {})} or None
        return self._masked_sample(logits, allowed, self.encoder.encode_type, greedy=greedy,
                                   temperature=temperature, bias=eff_bias)

    def predict_result(self, next_player: str, next_type: str, allowed: list[str], *,
                       delta_seconds: float = 0.0, greedy: bool = False,
                       temperature: float = RESULT_TEMPERATURE,
                       bias: dict[str, float] | None = None) -> str:
        """Sample a shot's ``result`` from ``allowed`` via the ``shot_result`` head.

        Conditions on the decided actor and the decided ``next_type`` (the type the shot is
        about to be). ``allowed`` is the legal outcome set — made/missed/blocked for a live FG,
        made/missed for a free throw. ``bias`` (e.g. the ``SHOT_RESULT_BIAS`` calibration on a live
        shot) adds a per-outcome logit offset before sampling — see :meth:`_masked_sample`.
        """
        inputs = self._conditioned_inputs(next_event="shot", delta_seconds=delta_seconds,
                                          next_player=next_player, next_type=next_type)
        logits = self._head_logits("shot_result", "result_output", inputs)
        return self._masked_sample(logits, allowed, self.encoder.encode_result, greedy=greedy,
                                   temperature=temperature, bias=bias)

    def predict_delta(self, next_event: str, next_player: str, *,
                      delta_seconds: float = 0.0) -> float:
        """Predict the inter-event Δt (seconds) before ``next_event`` via the ConditionalTimeModel.

        Conditions on the decided event and its actor (``next_player``); regresses standardized Δt
        and denormalizes with the conditional-time head's own ``delta_mean`` / ``delta_std`` (falling
        back to the shared event/time stats if the head carries none). This is the *authoritative*
        clock advance — Δt follows the play, replacing the marginal time head's average.
        """
        inputs = self._conditioned_inputs(next_event=next_event, delta_seconds=delta_seconds,
                                           next_player=next_player)
        # The conditional-time graph does not declare next_delta_time (Δt is its target), so drop the
        # key _conditioned_inputs always attaches — the dict must match the model's inputs exactly.
        inputs.pop("next_delta_time", None)
        pred = self._head_logits(ConditionalTimeModel.KEY, "time_output", inputs)
        norm = float(np.ravel(pred)[0])
        stats = self.condtime_norm_stats or self.norm_stats
        mean = float(stats.get("delta_mean", 0.0))
        std = float(stats.get("delta_std", 1.0)) or 1.0
        return norm * std + mean

    # ===================================================================== #
    # --- Opening-lineup bootstrap (build the starting five via subs)      --
    # ===================================================================== #

    def start_from_full_rosters(self, home_full: list[str], away_full: list[str],
                                possession: str = HOME, season: str = "2003",
                                tipoff_time: float = 0.0, greedy: bool = False,
                                greedy_starters: bool = True,
                                season_context: dict | None = None) -> "GameSimulator":
        """Seed an empty ``start`` frame and build each team's starting five via subs.

        This is the inference counterpart of the synthesized opening subs in
        ``SubstitutionModel`` preprocessing: the outgoing slot is the ``"start"`` token and
        the substitution head picks each incoming starter (constrained to that team's
        not-yet-placed roster), with the on-court five filling 0->5. Takes each team's whole
        roster (e.g. ``GameInput.home_roster`` / ``away_roster``).

        ``greedy_starters`` (default True) takes the substitution head's argmax for each
        starter — the most-likely opening five — independent of the in-game ``greedy`` flag.
        """
        self.reset()
        self.home_full = list(home_full)
        self.away_full = list(away_full)
        self.possession = possession
        self.season = str(season)
        # Empty start frame — the lineup is not yet built (mirrors preprocessing).
        self.history.append(self._make_row(
            event="start", player="start", type="start", result="start",
            secondary_player="none", time=float(tipoff_time),
        ))
        self._build_team_opening(HOME, home_full, tipoff_time, greedy=greedy_starters)
        self._build_team_opening(AWAY, away_full, tipoff_time, greedy=greedy_starters)
        return self

    def start_alternating(self, home_full: list[str], away_full: list[str],
                          possession: str = HOME, season: str = "2003",
                          tipoff_time: float = 0.0, greedy: bool = False,
                          greedy_starters: bool = True,
                          season_context: dict | None = None) -> "GameSimulator":
        """Seed an empty ``start`` frame and build both starting fives **alternating** H, A.

        Same synthesized ``start -> starter`` subs as :meth:`start_from_full_rosters`, but the
        picks alternate home/away (home starter 1, away starter 1, home 2, …) so each incoming
        starter is conditioned on every starter placed so far on *both* teams — the lineups
        react to each other instead of being built one team in isolation.

        ``greedy_starters`` (default True) takes the substitution head's argmax for each
        starter — the most-likely opening five — independent of the in-game ``greedy`` flag.
        """
        self.reset()
        self.home_full = list(home_full)
        self.away_full = list(away_full)
        self.possession = possession
        self.season = str(season)
        # Set before building the opening five — the substitution head now consumes rest.
        self._set_season_context(season_context)
        self.history.append(self._make_row(
            event="start", player="start", type="start", result="start",
            secondary_player="none", time=float(tipoff_time),
        ))
        home_avail, away_avail = list(home_full), list(away_full)
        for _ in range(ROSTER_SIZE):
            for team, avail in ((HOME, home_avail), (AWAY, away_avail)):
                if avail:
                    self._place_starter(team, avail, tipoff_time, greedy=greedy_starters)
        return self

    def start_with_starters(self, home_full: list[str], away_full: list[str],
                            home_starters: list[str], away_starters: list[str],
                            possession: str = HOME, season: str = "2003",
                            tipoff_time: float = 0.0,
                            season_context: dict | None = None) -> "GameSimulator":
        """Seed an empty ``start`` frame and place the **given** starting fives, no model calls.

        The real-starters counterpart of :meth:`start_alternating`: instead of asking the
        SubstitutionModel who opens, the actual starters (e.g. a real game's tip-off five) are
        placed directly. The history shape is identical — an empty ``start`` frame followed by
        alternating home/away ``start -> starter`` subs filling the on-court fives 0->5 — so the
        downstream rollout stays in-distribution. ``home_full`` / ``away_full`` remain the whole
        rosters (for later bench substitutions); the starters must be a subset of them.
        """
        self.reset()
        self.home_full = list(home_full)
        self.away_full = list(away_full)
        self.possession = possession
        self.season = str(season)
        # Set before building the opening five — the substitution head now consumes rest.
        self._set_season_context(season_context)
        self.history.append(self._make_row(
            event="start", player="start", type="start", result="start",
            secondary_player="none", time=float(tipoff_time),
        ))
        home_q, away_q = list(home_starters), list(away_starters)
        for i in range(ROSTER_SIZE):
            for team, q in ((HOME, home_q), (AWAY, away_q)):
                if i < len(q):
                    self._place_known_starter(team, q[i], tipoff_time)
        return self

    def _place_known_starter(self, team: str, incoming: str, tipoff_time: float) -> None:
        """Log one ``start -> starter`` sub for a **predetermined** ``incoming`` starter.

        The no-model sibling of :meth:`_place_starter`: same post-add roster snapshot and
        synthesized sub row, but the incoming player is given rather than sampled.
        """
        roster = self.home_roster if team == HOME else self.away_roster
        roster.append(incoming)  # post-add snapshot, like the synthesized openers
        self.history.append(self._make_row(
            event=SUB_EVENT, player=START_TOKEN, type=SUB_EVENT, result=SUB_EVENT,
            secondary_player=incoming, time=float(tipoff_time),
        ))

    def _place_starter(self, team: str, available: list[str], tipoff_time: float,
                       *, greedy: bool = False) -> None:
        """Pick one ``start -> starter`` sub for ``team`` from ``available`` and log it.

        ``greedy`` here is the starter-selection flag (argmax of the substitution head over the
        not-yet-placed roster), not the in-game rollout flag — they are decoupled on purpose.
        """
        roster = self.home_roster if team == HOME else self.away_roster
        incoming = self.predict_incoming(START_TOKEN, available,
                                         delta_seconds=0.0, greedy=greedy)
        available.remove(incoming)
        roster.append(incoming)  # post-add snapshot, like the synthesized openers
        self.history.append(self._make_row(
            event=SUB_EVENT, player=START_TOKEN, type=SUB_EVENT, result=SUB_EVENT,
            secondary_player=incoming, time=float(tipoff_time),
        ))

    def _build_team_opening(self, team: str, full_roster: list[str], tipoff_time: float,
                            *, greedy: bool = False) -> None:
        """Fill ``team``'s on-court five with ``start -> starter`` subs, one starter at a time.

        ``greedy`` is the starter-selection flag (see :meth:`_place_starter`)."""
        available = list(full_roster)
        for _ in range(min(ROSTER_SIZE, len(available))):
            self._place_starter(team, available, tipoff_time, greedy=greedy)

    # ===================================================================== #
    # --- Data shaping (history -> model tensors)                          --
    # ===================================================================== #

    def build_model_inputs(self) -> dict[str, np.ndarray]:
        """
        Shape the current ``history`` into the model's fixed input dict (batch = 1).

        Mirrors ``EventTimeModel._build_split``: right-pad each field to ``sequence_length``,
        encode categoricals + rosters to token ids, normalize ``time`` (absolute and Δt),
        and emit a ``pad_mask`` (1 = real step). When the game exceeds ``sequence_length``
        the window slides to the most recent ``SEQ`` events; Δt is computed over the full
        history first so the windowed deltas stay correct relative to the dropped events.
        """
        if not self.history:
            raise RuntimeError("No history to encode; call start_game() first.")

        SEQ = self.sequence_length
        enc = self.encoder

        # Δt over the FULL history (per-game diff, first=0, backwards clipped to 0),
        # then take the trailing window — identical transform to preprocess().
        times = np.array([r["time"] for r in self.history], dtype=np.float64)
        deltas = np.diff(times, prepend=times[:1])
        np.clip(deltas, 0, None, out=deltas)

        window = self.history[-SEQ:]
        deltas = deltas[-SEQ:]
        n = len(window)

        max_time = float(self.norm_stats.get("max_time", 1.0)) or 1.0
        delta_mean = float(self.norm_stats.get("delta_mean", 0.0))
        delta_std = float(self.norm_stats.get("delta_std", 1.0)) or 1.0

        inputs: dict[str, np.ndarray] = {}

        # Categorical fields -> int32, right-padded with each field's PAD id.
        for field in CATEGORICAL_FIELDS:
            encode = getattr(enc, f"encode_{field}")
            buf = np.full((SEQ,), self._pad_id(field), dtype=np.int32)
            for i, row in enumerate(window):
                buf[i] = encode(_norm_cat(row[field]))
            inputs[field] = buf

        # Rosters -> fixed-5 token arrays, PAD-row padded.
        pad_player = enc.encode_player("PAD")
        for name, col in (("home_roster", "roster_home"), ("away_roster", "roster_away")):
            buf = np.full((SEQ, ROSTER_SIZE), pad_player, dtype=np.int32)
            for i, row in enumerate(window):
                buf[i] = enc.encode_roster(row[col])
            inputs[name] = buf

        # Continuous: normalized absolute clock and standardized Δt.
        time_abs = np.zeros((SEQ, 1), dtype=np.float32)
        delta_time = np.zeros((SEQ, 1), dtype=np.float32)
        for i, row in enumerate(window):
            time_abs[i, 0] = row["time"] / max_time
            delta_time[i, 0] = (deltas[i] - delta_mean) / delta_std
        inputs["time_abs"] = time_abs
        inputs["delta_time"] = delta_time

        # Season context — mirrors season_features exactly so it matches _build_split:
        # per-player rest is clipped+standardized over the full 5 slots (PAD slots fall out
        # of (0 - mean)/std and are masked by the roster encoder); team days-rest is
        # standardized the same way; games-played is fed raw.
        rest_mean = float(self.norm_stats.get("rest_mean", DEFAULT_REST_DAYS))
        rest_std = float(self.norm_stats.get("rest_std", 1.0)) or 1.0

        def _std_days(days):
            return (min(float(days), REST_CLIP_DAYS) - rest_mean) / rest_std

        rest_home = np.zeros((SEQ, ROSTER_SIZE), dtype=np.float32)
        rest_away = np.zeros((SEQ, ROSTER_SIZE), dtype=np.float32)
        for i, row in enumerate(window):
            for buf, col, rest_map in (
                (rest_home, "roster_home", self.home_rest),
                (rest_away, "roster_away", self.away_rest),
            ):
                raw = np.zeros((ROSTER_SIZE,), dtype=np.float32)
                for j, p in enumerate(row[col][:ROSTER_SIZE]):
                    raw[j] = rest_map.get(p, DEFAULT_REST_DAYS)
                buf[i] = (np.clip(raw, 0.0, REST_CLIP_DAYS) - rest_mean) / rest_std
        inputs["rest_home"] = rest_home
        inputs["rest_away"] = rest_away

        team_values = {
            "home_games_played": self.home_games_played,
            "away_games_played": self.away_games_played,
            "home_days_rest": _std_days(self.home_days_rest),
            "away_days_rest": _std_days(self.away_days_rest),
        }
        for name in TEAM_SCALAR_COLS:
            buf = np.zeros((SEQ, 1), dtype=np.float32)
            buf[:n, 0] = team_values[name]
            inputs[name] = buf

        # Game state — running score / period-clock / per-period team fouls. Derived over the
        # FULL history (inclusive running sums) with the SAME derive_game_state scan used in
        # preprocess, then windowed to the trailing SEQ — so training and inference see identical
        # values at each row (the live controller score matches the inclusive per-row sum).
        gs_raw = derive_game_state(self.history)
        gs_norm = normalize_game_state(gs_raw)
        for name in GAME_STATE_KEYS:
            buf = np.zeros((SEQ, 1), dtype=np.float32)
            buf[:n, 0] = gs_norm[name][-SEQ:]
            inputs[name] = buf

        # 1 for real steps, 0 for padding (attention key-padding mask).
        pad_mask = np.zeros((SEQ,), dtype=np.float32)
        pad_mask[:n] = 1.0
        inputs["pad_mask"] = pad_mask

        # Add the leading batch axis (batch = 1), preserving INPUT_KEYS ordering.
        return {k: inputs[k][None, ...] for k in EventTimeModel.INPUT_KEYS}

    def _pad_id(self, field: str) -> int:
        """PAD token id for a categorical field (secondary_player uses the player vocab)."""
        return getattr(self.encoder, f"encode_{field}")("PAD")

    # ===================================================================== #
    # --- Forward pass (raw distribution; no sampling yet)                 --
    # ===================================================================== #

    def predict_next(self) -> dict:
        """
        Run a forward pass and return the raw next-step distribution.

        The causal model emits a prediction at every timestep; the next event after the
        last real row lives at position ``n-1``. Returns the event logits / probabilities
        (with a decoded ``{event_name: prob}`` map) and the time head in both normalized
        and real-seconds form. No sampling is done here — the caller draws from this
        distribution (temperature / top-p / Controller) in a later step.
        """
        inputs = self.build_model_inputs()
        n = min(len(self.history), self.sequence_length)
        out = self._infer(EventTimeModel.KEY, inputs)

        event_logits = out["event_output"][0, n - 1]   # (event_vocab,)
        event_probs = _softmax(event_logits)
        event_vocab = self.encoder.vocabs["event"]
        event_dist = {event_vocab.decode(i): float(p) for i, p in enumerate(event_probs)}

        delta_norm = float(out["time_output"][0, n - 1, 0])
        delta_mean = float(self.norm_stats.get("delta_mean", 0.0))
        delta_std = float(self.norm_stats.get("delta_std", 1.0)) or 1.0
        delta_seconds = delta_norm * delta_std + delta_mean

        return {
            "event_logits": event_logits,
            "event_probs": event_probs,
            "event_dist": event_dist,
            "delta_time_norm": delta_norm,
            "delta_seconds": delta_seconds,
        }


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with an optional temperature (>1 flattens, <1 sharpens); 1.0 is the raw model."""
    scaled = np.asarray(logits, dtype=np.float64)
    if temperature != 1.0:
        scaled = scaled / temperature
    z = scaled - np.max(scaled)
    e = np.exp(z)
    return e / np.sum(e)
