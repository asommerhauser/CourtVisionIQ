# Global configuration for CourtVisionIQ models

from pathlib import Path

# Project root (this file lives at the repo root)
ROOT_DIR = Path(__file__).resolve().parent

# Max sequence length for a game (right-padded; covers OT/overflow, truncate beyond)
MAX_SEQUENCE_LENGTH = 600

# Fixed number of on-court player slots per roster (PAD-filled below this)
ROSTER_SIZE = 5

# --- Model capacity (shared backbone dims; one place so train + reload always agree) ---
# Every head's __init__/model()/train() default to these, and from_artifacts rebuilds with them,
# so changing a value here re-sizes the whole chain consistently (requires a fresh train — old
# weights are shaped to the old values). Train 2 bumps these from the original 256/4/8/1024/2 for
# more learning headroom; the per-field embedding dims live in models/event_time_model.EMBED_DIMS.
MODEL_DIM = 384            # transformer width (was 256)
NUM_LAYERS = 6             # causal transformer blocks per head (was 4)
NUM_HEADS = 8              # attention heads (key_dim = MODEL_DIM // NUM_HEADS = 48)
FF_DIM = 1536              # feed-forward inner dim per block (was 1024)
ROSTER_SAB_LAYERS = 3      # Set-Attention blocks in the roster set-encoder (was 2)

# --- Rollout sampling (GameController / GameSimulator) ---
# Per-head softmax temperature for the rollout: <1 sharpens (emphasizes the head's preference),
# >1 flattens toward uniform, 1.0 is the raw model. Every categorical pick routes through
# _masked_sample / _constrained_sample, which apply these.
#
# The player/actor head (shooter, rebounder, assister, fouler, FT shooter) is FLATTENED above 1.
# The full-corpus, recency-weighted retrain converges to a much more confident head than the old
# curriculum stages: restricted to the on-court five it puts ~0.55-0.85 of the mass on a single
# player (measured via a probe over real holdout lineups), so at the old 0.8 (a *sharpening*) one
# star vacuumed points+rebounds+assists at once (the shared head also picks the rebounder/assister)
# — e.g. a 50/15/14 line — while a star the head under-ranks got starved. 2.0 lands the alpha's
# shot share back in the realistic ~0.33-0.41 band while preserving the ranking. Temperature only
# flattens over-concentration; it cannot fix a genuine per-player mis-ranking (a training issue).
# The rest default to 1.0 (raw) and are exposed as knobs for future tuning.
PLAYER_TEMPERATURE = 2.0
EVENT_TEMPERATURE = 1.0    # next-event head (shot / foul / turnover / … mix)
# Multiplicative calibration on the predicted inter-event Δt before it advances the game clock
# (GameController._advance_clock). 1.0 = raw model. Pace ≈ possessions/48 ≈ FGA-driven, and the
# clock filling 48 min sets how many possessions fit: >1 slows the clock (fewer possessions →
# lower pace), <1 speeds it up. Ideal value = real_Δt_mean/sim_Δt_mean from
# `python -m simulation.diagnostics` (no retrain needed).
#
# HISTORY: 1.06 was fit to Train 1, whose smaller time head ran pace HIGH (107.8 vs 101.3 real,
# +6.4%), so the clock was slowed x1.06. Train 2's larger conditional-time head no longer
# under-predicts the gap, but the +6% slowdown stayed on top of it and pace COLLAPSED the other way
# — v1.0's full holdout ran 92.1 vs 100.7 real poss/48 (bias -8.6), the mirror image of Train 1.
# Pace scales ~inversely with seconds/event, so 1.06 x (92.1/100.7) ≈ 0.97 targets real pace as a
# first cut without a retrain. Re-confirm against sim vs real Δt with `simulation.diagnostics` and
# set precisely = real_Δt_mean/sim_Δt_mean. (Secondary pace lever if this still lags: MAX_DELTA.)
DELTA_TIME_SCALE = 0.97
SUB_TEMPERATURE = 1.0      # outgoing substitution pick (legacy path) / generic sub sampling
# Incoming-sub pick temperature. The substitution head emits over the *player* vocab, so — like
# the actor head — its small, real preferences (which bench player actually checks in) should
# drive the pick, not be smoothed toward a uniform bench. <1 sharpens; push toward 0 to approach
# argmax (the single most-likely sub every time, at the cost of all rotation variety). Sharper
# than the actor head by default since a coach's bench order is more concentrated than shot usage.
# Lowered from 0.7: the stage eval over-played the deep bench (rank 15+ by +4..+14 min) and
# under-played starters (~9 min); sharpening concentrates check-ins on the real 8–9 man rotation.
SUB_INCOMING_TEMPERATURE = 0.45
TYPE_TEMPERATURE = 1.0     # shot_type / assist_type / turnover_type / foul_type / rebound_type
RESULT_TEMPERATURE = 1.0   # shot_result (made / missed / blocked)
# Per-outcome logit offset applied to the live-shot result sample (made / missed / blocked) via the
# existing `bias` arg of GameSimulator._masked_sample. Default {} = raw model. Fit to v1.0 trial1
# (100-game holdout): the unassisted make rate ran 21.8% vs 27.1% real (2P% .510 vs .548, 3P% .322
# vs .374 — ~-7.8 pts/team) and blocks were over-produced (+27%). Values are the log-odds-ratio
# corrections from the pooled live-shot table (per-type ideal: made +0.22 on 2pt / +0.35 on 3pt;
# global weighted below). FTs do NOT route through this bias (FT% was already accurate).
# RE-MEASURE after any retrain of shot_result — a modern-heavier train should need less of this.
# v1.0 full1 (100-game holdout, this bias already applied): eFG% still -2.17% / FGM -1.04 vs real
# -- the +0.27 correction closed most but not all of the original make-rate gap. Bumped further.
SHOT_RESULT_BIAS: dict[str, float] = {"made": 0.40, "blocked": -0.15}
# Per-event-token logit offset applied to the next-event pick (GameController._sample_event), the
# event-head sibling of SHOT_RESULT_BIAS. Default {} = raw model. Fit to v1.0 trial1: fouls ran
# 36.1/game vs 40.5 real (-11%, the biggest driver of the FTA -4.7/team deficit) and turnovers
# 44.9 vs 41.6 (+8%). Re-measure after any retrain — the event mix moves with the event head.
# v1.0 full1 (this bias already applied): FTA over-produced +3.845 (16% high) instead of under --
# the foul-volume push overshot. Roughly halved.
# v1.0 full2 (this bias + the foul_type.shooting cut below, applied together): FTA improved to
# +1.44 but total fouls overshot the other way -- PF flipped from +0.41 to -0.71 (team level).
# Restoring some volume here; foul_type.shooting is cut further below to keep FTA falling without
# re-inflating PF.
EVENT_BIAS: dict[str, float] = {"foul": 0.11, "turnover": -0.08}
# Per-head per-token logit offset on the conditional type heads (GameSimulator.predict_type),
# keyed by head then token, e.g. {"turnover_type": {"steal": -0.2}} to pull steal-type turnovers
# down without moving the overall turnover rate. Default {} = raw model. Fit to v1.0 trial1:
#   foul_type    — shooting-foul share ran 40.5% vs 52.8% real (the other half of the FTA deficit),
#                  personal over-picked (16.5 vs 10.6/game), loose-ball fouls near-absent (0.06 vs
#                  2.70/game — hence the outsized +2.5 on a near-zero-mass token; iterate on it),
#                  technicals 0.12 vs 0.63. Offensive fouls were already over (4.62 vs 3.83).
#   turnover_type — steal share slightly high (56.8% vs 54.8% of TOs).
#   assist_type  — assisted-3 share 38.5% vs 41.0% (drives the residual TPM gap).
#   rebound_type — offensive share of rebounds 23.4% vs 27.3%.
# v1.0 full1 (all values below already applied): FTA over-produced +3.845 (16% high) -- "loose
# ball" is named directly as a likely culprit (a very large offset on a near-zero-mass token);
# roughly halved rather than zeroed, since the original trial1 gap it corrected was real. OREB% is
# now over +2.48% (was under at trial1, hence the original +0.20) -- halved since it's overshooting.
# v1.0 full2 (all values below already applied): FTA still over +1.44 (team level) even with PF now
# under -0.71 -- cut shooting-foul share further so FTA keeps falling as EVENT_BIAS.foul is restored
# above. OREB landed close (+0.46) while DREB is still over (+1.17); bumped the offensive split back
# up a touch to protect OREB's share while DEADBALL_REBOUND_PROB (below) pulls more total volume,
# mostly from DREB, on the next pass. "loose ball" left untouched to isolate its effect this round.
TYPE_BIAS: dict[str, dict[str, float]] = {
    "foul_type": {"shooting": 0.15, "personal": -0.45, "offensive": -0.30,
                  "loose ball": 1.2, "technical": 1.5},
    "turnover_type": {"steal": -0.12},
    "assist_type": {"3pt": 0.10},
    "rebound_type": {"offensive": 0.13},
}
# Home-court edge. The rollout is otherwise home/away symmetric (HOME just inbounds first), so the
# sim can't separate winners and win-pick accuracy sits near a coin flip. This adds a logit nudge to
# the live-shot "made" outcome: +HOME_COURT_SHOT_BIAS for the home offense, -HOME_COURT_SHOT_BIAS for
# the away offense. Symmetric on purpose — it tilts the home/away split (driving win prediction)
# without moving the pooled eFG/FG% the four-factors table already gets ~right. ~0.10 lifts home eFG
# ~+1pt / drops away ~-1pt, roughly a ~2.5-pt home edge (real NBA ~2.5-3.0). 0 = off; tune against
# the win-prediction calibration + spread bias in the eval report. Applied in GameController._do_shot.
# Trimmed 0.10 -> 0.07: v1.0 trial1 predicted the home margin at +4.05 vs +2.58 actual (spread bias
# +1.47), so the edge was ~55% too strong; scaled proportionally.
# Trimmed further 0.07 -> 0.047: v1.0 full1 (100-game holdout) still predicted +3.87 vs +2.58
# actual (spread bias +1.29) -- same proportional scaling: 0.07 * (2.58/3.87).
# Raised 0.047 -> 0.055: v1.0 full2 overshot the other way, spread bias flipping from +1.35 (full1)
# to -0.69 -- win-pick accuracy also dropped 66% -> 62%, though with only 21 sims x 100 games some
# of that swing is noise. Split the difference via linear interpolation between the two known
# (dial, bias) points, targeting bias ~0.
HOME_COURT_SHOT_BIAS = 0.055
# Logit bonus per second of a player's current on-court stint, added to the outgoing-sub pick so
# a long-tenured player (a star included) is *nudged* — not forced — toward coming off. 0 = off.
# Lowered from 0.15 so starters are pulled for tenure less aggressively (the stage eval under-played
# the top of the rotation); lets stars hold longer stints.
SUB_FATIGUE_WEIGHT = 0.08
# Max game-seconds a team may go without a substitution before the Controller forces one (the
# event head never targets a team, so this safety net keeps a team from playing five men 48 min).
# Raised 420 -> 600 alongside STINT_LENGTH_SCALE: real teams average one sub every ~2 min, so the
# backstop should stay rare; at 420 it would re-create the churn the longer stints remove.
SUB_MAX_GAP_SECONDS = 600.0

# Number of independent game-sims the batched rollout runs concurrently, pooling their per-event
# forward passes into one batched GPU call (simulation/batched_rollout.py). >1 enables batching; 1 is
# the original one-at-a-time path. The win comes from amortizing batch-1 kernel-launch overhead, so
# size it to how many concurrent sims fit in VRAM (the heads are small — dozens are fine). It does NOT
# affect results (pure scheduling), so it's a perf knob, not a tuning dial. 48 keeps the batch full
# across the pooled games (see EVAL_GAMES_PER_BATCH) so the GPU isn't starved by a single game's ~2-wide
# effective batch (its sims desync across heads). Lower it if concurrent workers pressure the GPU.
ROLLOUT_BATCH_SIZE = 48
# Eval pools this many holdout games' sims into ONE batched rollout so the GPU sees a full batch
# (one game alone only keeps ~2 sims on the same head at a time -> the card sat ~10% utilized). With
# STAGE_SIMS sims each, the pool is EVAL_GAMES_PER_BATCH*STAGE_SIMS concurrent sims, run in cohorts of
# ROLLOUT_BATCH_SIZE. Pure scheduling — results are unchanged (each sim keeps its own seed). Runs in a
# single process (no cross-process VRAM contention). Lower it if system RAM/thread pressure is high.
EVAL_GAMES_PER_BATCH = 6
# ...but only up to EVAL_POOL_JOBS concurrent sims -- see it, below STAGE_SIMS.

# --- Eval process pool (`run --procs N` / `evaluate.py --procs N`) ---
# Batching fills the GPU, but everything around the forward pass -- input building, sampling,
# the rule engine -- is Python, so one process is capped by the GIL at roughly one core no
# matter how many the box has. Separate processes on disjoint holdout slices are what turn the
# rest of the cores into throughput. These size that pool; like ROLLOUT_BATCH_SIZE they are perf
# knobs, NOT tuning dials -- they change how long a run takes, never what it predicts.
# Re-measure them per machine with the scaling curve in README (--procs 1 / 2 / 4).
EVAL_PROC_CORES = 2        # cores to budget per eval process (GIL-bound Python + TF intra-op)
EVAL_PROC_VRAM_GB = 4.0    # VRAM a loaded eval process costs (~3-4 GB for the 11 heads)
EVAL_PROC_MAX = 8          # ceiling, so a 128-core pod does not fork a pathological pool

# --- Stint-length scheduler (StintLengthModel + GameController hybrid scheduler) ---
# When the stint-length head is loaded, the Controller commits each entering player to a stint:
# it samples a length (game-seconds on the floor) and schedules the player's exit at
# clock + length; at each dead ball a player past their scheduled exit is subbed out. The model
# regresses log-stint, so we sample with multiplicative log-space noise for rotation variety.
# STINT_SAMPLE_SIGMA is the std of that log-space noise (0 = deterministic / point estimate).
STINT_SAMPLE_SIGMA = 0.25
# Multiplicative calibration on the predicted stint length (applied in predict_stint_length before
# the cap) — the rotation sibling of DELTA_TIME_SCALE. The head regresses LOG-stint, so its point
# estimate is the geometric mean, which systematically under-predicts the arithmetic mean of a
# right-skewed duration distribution. Measured on v1.0 trial1: sim stints averaged 366s vs 474s
# real (ratio 1.30), producing 79 subs/game vs 46 real and over-playing the 9th-13th men by 2-3x
# while starters ran ~5 min short. Tune to match subs/game (~46) in the eval report; 1.0 = raw.
STINT_LENGTH_SCALE = 1.30
# Numerical cap on a sampled stint (game-seconds). There is intentionally NO lower bound — a
# short specialist stint (a one-possession 3pt shooter / rebounder) is legitimate basketball.
# Raised 900 -> 2400: real stints run to ~2600s (p90 886s), so the old cap truncated the real
# tail right where long starter stints live and clipped the sim's max stint to ~938s.
STINT_MAX_SECONDS = 2400.0
# Personal fouls that disqualify a player for the rest of the game (NBA standard: 6). Offensive
# fouls count toward this; technicals do not.
FOUL_OUT_LIMIT = 6

# Clamp on a single predicted inter-EVENT Δt before it advances the game clock (after
# DELTA_TIME_SCALE), so one bad gap can't blow up the clock. NOT the 24s shot clock — this is the
# gap between two consecutive events, which is usually well under 24s but can legitimately exceed it
# (a dead-ball / timeout stretch, or the gap spanning a quarter/half break; free throws are logged
# at Δt=0). Lowering it trims that long tail and nudges pace UP, so it's the secondary pace lever
# after DELTA_TIME_SCALE (leave at 60 while tuning the scale; try ~45 only if pace still lags).
MAX_DELTA = 60.0
# Probability a missed shot yields no individual rebound (an out-of-bounds / dropped team rebound):
# the ball just changes hands with no row. The off/def split of real rebounds is the rebound-type
# head's job; this is only the rare no-rebounder case. The controller imports this.
# Raised 0.06 -> 0.09: v1.0 full1 (100-game holdout) had OREB *and* DREB both over by the same
# +2.22 -- total individually-attributed rebound volume is inflated on both sides, not just the
# off/def split (that's TYPE_BIAS.rebound_type's job, tuned separately above). More dead-ball
# rebounds pulls both counts down together without touching the split.
# Raised further 0.09 -> 0.10: v1.0 full2 landed OREB close (+0.46) but DREB is still over (+1.17)
# -- the rebound_type.offensive cut (also made in full2) pulled OREB down twice as hard as DREB.
# Pulling a bit more total volume here, offset by bumping rebound_type.offensive back up above, so
# the extra cut lands mostly on DREB. This should also trim the pace bias regression full2 saw
# (+0.50 -> +0.95): pace ~= FGA - OREB + TOV + 0.44*FTA, so full2's OREB drop mechanically pushed
# the pace estimate up; restoring some FTA (via EVENT_BIAS.foul above) and reeling in rebounds
# further both pull pace back down as a side effect, not a direct target.
DEADBALL_REBOUND_PROB = 0.10

# Post-hoc linear calibration on predicted point margin, applied only when aggregating spread
# metrics for the eval report (simulation/eval_metrics.py) -- never to the raw per-game record, so
# record.json / games.parquet / the margin scatter plot keep the literal model output and this can
# be refit without touching sim data. NOT a rollout dial (nothing at sim time reads it, so it is
# deliberately absent from _TUNING_KEYS below). Regressing actual margin on predicted margin across
# v1.0's full1 holdout: actual = -0.31 + 0.745*predicted -- predicted margins run ~25% too extreme
# for their information content. slope=1.0/intercept=0.0 = off.
MARGIN_CALIBRATION_SLOPE = 0.745
MARGIN_CALIBRATION_INTERCEPT = -0.31

# Rollout dials captured into each evaluation report (reporting/eval_report.py) so tuning settings
# are recorded alongside results for cross-run analysis. Order is the display order in the report.
_TUNING_KEYS = (
    "DELTA_TIME_SCALE", "MAX_DELTA", "DEADBALL_REBOUND_PROB",
    "PLAYER_TEMPERATURE", "EVENT_TEMPERATURE", "TYPE_TEMPERATURE", "RESULT_TEMPERATURE",
    "SUB_TEMPERATURE", "SUB_INCOMING_TEMPERATURE", "SUB_FATIGUE_WEIGHT", "SUB_MAX_GAP_SECONDS",
    "STINT_SAMPLE_SIGMA", "STINT_LENGTH_SCALE", "STINT_MAX_SECONDS", "FOUL_OUT_LIMIT",
    "SHOT_RESULT_BIAS", "EVENT_BIAS", "TYPE_BIAS", "HOME_COURT_SHOT_BIAS",
)


def tuning_snapshot() -> dict:
    """The live values of every rollout dial (read from this module at call time).

    Reading the module globals means edits to this file between runs are reflected accurately, so
    each evaluation report records exactly the tuning that produced it. Dict-valued dials
    (``SHOT_RESULT_BIAS`` / ``EVENT_BIAS`` / ``TYPE_BIAS``) are JSON-encoded to a compact string
    so each sits cleanly in a single Parquet column.
    """
    import json as _json
    g = globals()
    snap: dict = {}
    for k in _TUNING_KEYS:
        v = g[k]
        snap[k] = _json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
    return snap


def get_dials() -> dict:
    """Live values of every rollout dial, in native Python types.

    Unlike ``tuning_snapshot`` (which JSON-encodes dict dials for Parquet), this returns the
    dicts as dicts, so the result round-trips back through ``apply_dials``. Deep-copied so a
    caller holding the result cannot mutate module state by editing a nested dict.
    """
    import copy as _copy
    g = globals()
    return {k: _copy.deepcopy(g[k]) for k in _TUNING_KEYS}


def set_dial(name: str, value):
    """Set one rollout dial, coercing ``value`` to the type the dial already holds.

    Validating against ``_TUNING_KEYS`` means a typo raises instead of silently minting a new
    module global that nothing reads. Strings are coerced from the shell: ``json.loads`` for the
    dict-valued dials, otherwise ``int``/``float``/``bool`` to match the current value. Returns
    the coerced value that was stored.
    """
    import json as _json
    if name not in _TUNING_KEYS:
        raise KeyError(f"unknown dial '{name}'; expected one of: {', '.join(_TUNING_KEYS)}")
    current = globals()[name]
    if isinstance(current, dict):
        if isinstance(value, str):
            value = _json.loads(value)
        if not isinstance(value, dict):
            raise TypeError(f"{name} is a dict dial; got {type(value).__name__}")
        value = dict(value)
    elif isinstance(current, bool):
        value = value.strip().lower() in ("1", "true", "yes", "on") if isinstance(value, str) else bool(value)
    elif isinstance(current, int):
        value = int(float(value))
    elif isinstance(current, float):
        value = float(value)
    globals()[name] = value
    return value


def apply_dials(values: dict) -> dict:
    """Set many dials at once (a "dial package"). Returns the coerced values actually stored."""
    return {k: set_dial(k, v) for k, v in values.items()}


def apply_dial_file(path) -> dict:
    """Apply a dial package from a JSON file (an object of DIAL -> value); return what was stored.

    Lives here rather than in the shell so the eval CLI can use it without importing anything that
    pulls TensorFlow: a sharded or pooled eval hands each child process the parent's dial package
    this way, which is the only thing keeping N processes from silently running different physics.
    Raises ValueError for anything malformed; callers wanting their own error type wrap it.
    """
    import json as _json
    from pathlib import Path as _Path

    p = _Path(path)
    if not p.is_file():
        raise ValueError(f"no dial file at {p}")
    try:
        values = _json.loads(p.read_text(encoding="utf-8"))
    except _json.JSONDecodeError as e:
        raise ValueError(f"{p} is not valid JSON: {e}") from None
    if not isinstance(values, dict):
        raise ValueError(f"{p} must contain a JSON object of DIAL -> value")
    try:
        return apply_dials(values)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{p}: {e}") from None


def write_dial_file(path) -> dict:
    """Write the live dial values to ``path`` as JSON and return them. Inverse of the above."""
    import json as _json
    from pathlib import Path as _Path

    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    values = get_dials()
    p.write_text(_json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")
    return values


from contextlib import contextmanager as _contextmanager


@_contextmanager
def dials(**overrides):
    """Temporarily apply dial overrides, restoring the previous values on exit.

    Convenience for tests. This works only because every consumer reads ``config.<DIAL>`` at call
    time rather than binding the name at import; see the note on ``_TUNING_KEYS`` above.
    """
    import copy as _copy
    g = globals()
    saved = {k: _copy.deepcopy(g[k]) for k in overrides}
    try:
        apply_dials(overrides)
        yield
    finally:
        g.update(saved)

# Full-run state (holdout ids + the train/holdout cut). Machine-local, never committed.
# Defined here rather than in training.full_run so the eval CLI and the process-pool
# supervisor can read it without importing anything that pulls TensorFlow.
FULL_RUN_STATE_PATH = "./training/full_run_state.json"

# Where the shared vocab "language" files live
VOCAB_DIR = ROOT_DIR / "encoder" / "vocabs"

# Where pipeline-level normalization stats (time) are persisted alongside vocabs
NORM_STATS_PATH = VOCAB_DIR / "norm_stats.json"

# --- Game splitting (shared by every model's preprocess + the box-score validation) ---
# Deterministic seed so the train/val/holdout partition is reproducible across models.
SEED = 42
# Fraction of games used as the early-stopping validation ("test") split.
TEST_FRAC = 0.2
# Fraction of games fully reserved as a holdout: never trained on AND never used for early
# stopping, so it can serve as an unbiased batch of real games to test models against.
HOLDOUT_FRAC = 0.1
# Filename of the holdout game-id manifest, written under each model's processed_dir.
HOLDOUT_MANIFEST_NAME = "holdout_games.json"

# --- Chronological schedule helpers (training/chronology.py) ---
# Utilities for contiguous, cumulative training slices + sequential (next-N) holdouts. Retained as
# building blocks (build_schedule / sequential_partition); the single full train (full_run) is the
# active path — see FINAL_HOLDOUT_GAMES below.
# Default number of sequential games held out after a training boundary (schedule helper default).
HOLDOUT_GAMES = 10
# Predictions run per holdout game when scoring a stage (the simulator is stochastic; we average).
# Bumped 11 -> 21: the eval averages the per-game sims before scoring, so more sims tighten the
# box-score means (cuts sampling-noise MAE) and halve the win-vote quantization (1/11 -> 1/21),
# which flattered the Brier score. eval-all cost scales ~linearly (the batched rollout absorbs it).
STAGE_SIMS = 21
# The eval pool's real unit is CONCURRENT SIMS, not games: run_jobs_batched holds every
# finished history in memory until the pool drains, so the peak is games x sims. At the
# default 6 x 21 that is 126; a 100-sim run pooling 6 games would be 600 -- ~5x the host RAM
# for no gain, since 100 sims of a SINGLE game already over-fill a ROLLOUT_BATCH_SIZE cohort
# on their own. evaluate_stage pools ceil-down to this many jobs, which is exactly
# EVAL_GAMES_PER_BATCH games at STAGE_SIMS (default shape unchanged) and 1 game at 100 sims.
EVAL_POOL_JOBS = EVAL_GAMES_PER_BATCH * STAGE_SIMS
# Seasons of training added between stops. A stop is placed every SEASONS_PER_STAGE seasons
# (the first stop after the first SEASONS_PER_STAGE seasons), and the stop POINT cycles through
# BOUNDARY_CYCLE across those stops. So with 3: train ~3 seasons -> stop 25% in -> +3 seasons ->
# stop 50% in -> +3 seasons -> stop pre-playoffs -> repeat. Keeps the run to ~7 stages over the
# full corpus rather than 3 stops every single season.
SEASONS_PER_STAGE = 3
# The repeating stop-point cycle (one entry consumed per stop, in order):
#   "frac:f"      -> stop at the game f-of-the-way through that season's regular games.
#   "pre_playoffs"-> stop at the last regular-season game (holdout = first HOLDOUT_GAMES playoffs).
BOUNDARY_CYCLE = ("frac:0.25", "frac:0.50", "pre_playoffs")

# --- Recency weighting (single full train: older seasons contribute less to the loss) ---
# Every game still trains, but its loss weight decays with age so the modern game dominates the
# gradient. Newest season = 1.0; weight halves every RECENCY_HALFLIFE_SEASONS seasons, floored at
# RECENCY_FLOOR (so old-player embeddings keep getting a little gradient). See season_features.
RECENCY_WEIGHTING = True
# Halved 6 -> 3 for v1.1: v1.0's subset heads anchored near a corpus-era average — eFG ran .500 vs
# .554 real (2P% -3.8pp, 3P% -5.2pp) and the per-shooter eFG gradient was compressed, the classic
# signature of older, lower-efficiency eras diluting the modern game despite the season embedding.
# A shorter halflife makes the modern era dominate the gradient harder.
RECENCY_HALFLIFE_SEASONS = 3.0
RECENCY_FLOOR = 0.05

# --- Single full-train + batched holdout eval (full_train.py / training/full_run.py) ---
# Stop training partway through the most recent season, hold out the next FINAL_HOLDOUT_GAMES real
# games, and predict them EVAL_BATCH at a time (pausing between batches). Full-train weights go to
# their own root so the curriculum's ./artifacts is never clobbered.
FINAL_SEASON_FRACTION = 0.5
FINAL_HOLDOUT_GAMES = 100
EVAL_BATCH = 10
# Models live one-per-dir under ./artifacts/<name>/ (see models.artifacts.model_root).
# Names are free-form slugs -- "v1.0", "endgame-feats" -- and a name IS the train identity:
# retraining picks a NEW name rather than overwriting, so weights and the runs evaluated
# against them never drift apart.
#
# DEFAULT_MODEL is what the shell loads with a bare `load`, and the default target for a
# single-head retrain. FULL_ARTIFACTS_ROOT is derived from it so the two cannot disagree
# (tests/test_full_run.py asserts they match). A full train chooses its own name on the CLI:
# train.py --full --name <name>.
DEFAULT_MODEL = "v1.0"
FULL_ARTIFACTS_ROOT = f"./artifacts/{DEFAULT_MODEL}"
# Deprecated alias. Callers that still say "version" get the same value; note it is the full
# name ("v1.0"), not the bare "1.0" this constant used to hold.
DEFAULT_VERSION = DEFAULT_MODEL

# --- Representative subset for the small heads (training/subset.py) ---
# The small categorical/regression heads (event/type/result/conditional-time) saturate long before
# they see the whole corpus and start to overfit, so they train on a compact, *representative*
# slice instead of every game. The slice is selected by a per-season sample RATE that is heavy on
# the modern game (so current players are well-learned) and decays gently for older seasons, but it
# stays coverage-complete: every player who appears in the train pool is guaranteed at least one
# game, so no embedding goes starved. The big player-vocab heads (player / substitution /
# stint_length) keep the full corpus — they actually need the data.
#
# Per-season sample rate for the most recent seasons, NEWEST FIRST. Raised for v1.1
# ((0.70, 0.40, 0.25) -> (1.0, 0.70, 0.50)): the subset heads (shot_result especially) under-fit
# the modern game's efficiency (see RECENCY_HALFLIFE_SEASONS note), and shot_result early-stopped
# at epoch 7 on the old 3,239-game subset — it has headroom for more modern data. The newest
# season is itself already truncated at FINAL_SEASON_FRACTION (we cut partway through it), so
# 100% of that is a modest absolute count.
SUBSET_RECENT_SEASON_RATES = (1.0, 0.70, 0.50)
# Seasons older than the recent block decay from the last recent rate, halving every this-many
# seasons — a gentle exponential tail (tightened 8 -> 5 for v1.1, same rationale). Coverage still
# guarantees every player a game, so old-only players pull in the older games they need
# regardless of the rate.
SUBSET_RECENCY_HALFLIFE_SEASONS = 5.0
SUBSET_SEED = 42                     # deterministic subset selection
SUBSET_GAMES_PATH = "./training/subset_games.json"  # persisted subset manifest (one extract step)
# Heads trained on the representative subset rather than the full corpus. All six conditional
# type/result heads share one preprocess file, so they move as a group. Everything NOT listed here
# (event_time, player, substitution, stint_length) trains on the full corpus.
SUBSET_MODEL_KEYS = (
    "event_time_cond",
    "shot_type", "shot_result", "assist_type", "turnover_type", "foul_type", "rebound_type",
)