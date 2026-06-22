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

from config import MAX_SEQUENCE_LENGTH, ROSTER_SIZE
from encoder.encoder import Encoder
from models.artifacts import DEFAULT_ARTIFACTS_ROOT
from models.event_time_model import CATEGORICAL_FIELDS, EventTimeModel
from models.model_bundle import ModelBundle
from models.player_model import PlayerModel
from models.substitution_model import START_TOKEN, SUB_EVENT, SubstitutionModel

# result outcomes that hand the ball to the other team. The model does not consume
# possession; we track it for downstream use (and the future Controller will own the
# richer clock/score/foul bookkeeping). A made field goal also flips possession, but in
# the data that is realized through the *following* inbound/rebound events, so we key off
# the explicit change-of-possession outcomes here and leave the rest to the Controller.
POSSESSION_FLIP_RESULTS = {"cop", "steal"}

HOME, AWAY = "home", "away"


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
        return sim

    # ===================================================================== #
    # --- Game setup / state                                               --
    # ===================================================================== #

    def reset(self) -> None:
        """Clear all per-game state (rosters, possession, history)."""
        self.home_roster, self.away_roster = [], []
        self.home_full, self.away_full = [], []
        self.possession = HOME
        self.season = ""
        self.history = []

    def start_game(self, home_roster: list[str], away_roster: list[str],
                   possession: str = HOME, season: str = "2003",
                   seed_event: dict | None = None, tipoff_time: float = 0.0) -> "GameSimulator":
        """
        Initialize game state and seed the input with the opening row.

        The cleaned data frames every game with an explicit ``start`` event row (all
        categorical fields = ``"start"``, ``secondary_player="none"``, time 0) carrying
        the tip-off rosters, so the default seed reproduces exactly that in-distribution
        first timestep. Pass ``seed_event`` to override (e.g. to resume from a real
        partial game); it is a dict of the categorical fields below.
        """
        self.reset()
        self.home_roster = list(home_roster)
        self.away_roster = list(away_roster)
        self.possession = possession
        self.season = str(season)

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

    def _constrained_sample(self, logits: np.ndarray, candidates: list[str],
                            *, greedy: bool = False) -> str:
        """Pick a player from ``candidates`` by restricting ``logits`` to their token ids.

        The heads emit a distribution over the whole player vocab; legality (which players
        are actually eligible — the bench, the active five, the remaining starters) is a
        sampling-time constraint applied here. Renormalizes over the candidate ids and
        either samples (default) or takes the argmax (``greedy``, for deterministic tests).
        """
        if not candidates:
            raise ValueError("no candidate players to sample from")
        ids = [self.encoder.encode_player(p) for p in candidates]
        sub = np.asarray(logits)[ids]
        if greedy:
            choice = int(np.argmax(sub))
        else:
            probs = _softmax(sub)
            choice = int(self.rng.choice(len(ids), p=probs))
        return candidates[choice]

    def _next_step_inputs(self, *, outgoing: str | None, delta_seconds: float) -> dict:
        """Base history inputs + the next-step conditioning a downstream head reads.

        Adds ``next_event`` (= ``substitution``) / ``next_delta_time`` at the decision
        position (the last real step), plus ``next_player`` (the decided outgoing player)
        when ``outgoing`` is given (the substitution head). Mirrors the per-head INPUT_KEYS.
        """
        base = self.build_model_inputs()
        SEQ = self.sequence_length
        n = min(len(self.history), SEQ)
        enc = self.encoder

        next_event = np.full((1, SEQ), enc.encode_event("PAD"), dtype=np.int32)
        next_event[0, n - 1] = enc.encode_event(SUB_EVENT)
        delta_mean = float(self.norm_stats.get("delta_mean", 0.0))
        delta_std = float(self.norm_stats.get("delta_std", 1.0)) or 1.0
        next_delta = np.zeros((1, SEQ, 1), dtype=np.float32)
        next_delta[0, n - 1, 0] = (delta_seconds - delta_mean) / delta_std

        inputs = {**base, "next_event": next_event, "next_delta_time": next_delta}
        if outgoing is not None:
            next_player = np.full((1, SEQ), enc.encode_player("PAD"), dtype=np.int32)
            next_player[0, n - 1] = enc.encode_player(outgoing)
            inputs["next_player"] = next_player
        return inputs

    def _head_logits(self, key: str, output_name: str, inputs: dict) -> np.ndarray:
        """Run a loaded head and return its logits at the decision position (n-1)."""
        head = self.heads.get(key)
        if head is None:
            raise RuntimeError(
                f"'{key}' head not loaded; train it and load via GameSimulator.load()."
            )
        n = min(len(self.history), self.sequence_length)
        out = head(inputs, training=False)
        return np.asarray(out[output_name])[0, n - 1]

    def predict_outgoing(self, candidates: list[str], *, delta_seconds: float = 0.0,
                         greedy: bool = False) -> str:
        """Sample the outgoing player from ``candidates`` (the active roster) via PlayerModel."""
        inputs = self._next_step_inputs(outgoing=None, delta_seconds=delta_seconds)
        logits = self._head_logits(PlayerModel.KEY, "player_output", inputs)
        return self._constrained_sample(logits, candidates, greedy=greedy)

    def predict_incoming(self, outgoing: str, candidates: list[str], *,
                         delta_seconds: float = 0.0, greedy: bool = False) -> str:
        """Sample the incoming player from ``candidates`` (the bench) via SubstitutionModel,
        conditioned on the decided ``outgoing`` player."""
        inputs = self._next_step_inputs(outgoing=outgoing, delta_seconds=delta_seconds)
        logits = self._head_logits(SubstitutionModel.KEY, "secondary_player_output", inputs)
        return self._constrained_sample(logits, candidates, greedy=greedy)

    def sample_substitution(self, *, team: str | None = None, delta_seconds: float = 0.0,
                            greedy: bool = False) -> tuple[str, str]:
        """Generate one in-game substitution: (outgoing, incoming).

        The outgoing player is drawn from the active on-court roster (a given ``team`` or
        all ten); the incoming player from that team's bench (full roster minus on-court).
        Does not mutate state — call ``append_event('substitution', outgoing, …,
        secondary_player=incoming)`` to apply it.
        """
        if team is not None:
            active = self.home_roster if team == HOME else self.away_roster
        else:
            active = self.home_roster + self.away_roster
        outgoing = self.predict_outgoing(active, delta_seconds=delta_seconds, greedy=greedy)

        sub_team = self._team_of(outgoing)
        full = self.home_full if sub_team == HOME else self.away_full
        on_court = self.home_roster if sub_team == HOME else self.away_roster
        bench = [p for p in full if p not in on_court]
        incoming = self.predict_incoming(outgoing, bench, delta_seconds=delta_seconds,
                                          greedy=greedy)
        return outgoing, incoming

    # ===================================================================== #
    # --- Opening-lineup bootstrap (build the starting five via subs)      --
    # ===================================================================== #

    def start_from_full_rosters(self, home_full: list[str], away_full: list[str],
                                possession: str = HOME, season: str = "2003",
                                tipoff_time: float = 0.0, greedy: bool = False) -> "GameSimulator":
        """Seed an empty ``start`` frame and build each team's starting five via subs.

        This is the inference counterpart of the synthesized opening subs in
        ``SubstitutionModel`` preprocessing: the outgoing slot is the ``"start"`` token and
        the substitution head picks each incoming starter (constrained to that team's
        not-yet-placed roster), with the on-court five filling 0->5. Takes each team's whole
        roster (e.g. ``GameInput.home_roster`` / ``away_roster``).
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
        self._build_team_opening(HOME, home_full, tipoff_time, greedy=greedy)
        self._build_team_opening(AWAY, away_full, tipoff_time, greedy=greedy)
        return self

    def _build_team_opening(self, team: str, full_roster: list[str], tipoff_time: float,
                            *, greedy: bool = False) -> None:
        """Fill ``team``'s on-court five with ``start -> starter`` subs, one starter at a time."""
        roster = self.home_roster if team == HOME else self.away_roster
        available = list(full_roster)
        for _ in range(min(ROSTER_SIZE, len(available))):
            incoming = self.predict_incoming(START_TOKEN, available,
                                             delta_seconds=0.0, greedy=greedy)
            available.remove(incoming)
            roster.append(incoming)  # post-add snapshot, like the synthesized openers
            self.history.append(self._make_row(
                event=SUB_EVENT, player=START_TOKEN, type=SUB_EVENT, result=SUB_EVENT,
                secondary_player=incoming, time=float(tipoff_time),
            ))

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
                buf[i] = encode(row[field])
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
        out = self.model(inputs, training=False)

        event_logits = np.asarray(out["event_output"])[0, n - 1]   # (event_vocab,)
        event_probs = _softmax(event_logits)
        event_vocab = self.encoder.vocabs["event"]
        event_dist = {event_vocab.decode(i): float(p) for i, p in enumerate(event_probs)}

        delta_norm = float(np.asarray(out["time_output"])[0, n - 1, 0])
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


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)
