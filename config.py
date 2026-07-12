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
# existing `bias` arg of GameSimulator._masked_sample. Default {} = raw model. Blocks were
# over-produced (+53% in the stage eval), which drags eFG/FG% down; if eFG stays low after the
# pace fix, push "blocked" negative (and/or "made" positive) here to pull make/block rates to real.
SHOT_RESULT_BIAS: dict[str, float] = {}
# Per-event-token logit offset applied to the next-event pick (GameController._sample_event), the
# event-head sibling of SHOT_RESULT_BIAS. Default {} = raw model. Tune post-train from the eval
# report's team-bias table: e.g. the Train-1 holdout under-produced fouls (PF -3.6 / FTA -3.9 per
# game — the model can't see bonus/clutch contexts) and over-produced assists (+3.9) / turnovers
# (+2.25), which would suggest something like {"foul": +0.2, "assist": -0.15, "turnover": -0.1}.
# Re-measure before touching: the game-state features change the whole event mix.
EVENT_BIAS: dict[str, float] = {}
# Per-head per-token logit offset on the conditional type heads (GameSimulator.predict_type),
# keyed by head then token, e.g. {"turnover_type": {"steal": -0.2}} to pull steal-type turnovers
# down without moving the overall turnover rate. Default {} = raw model.
TYPE_BIAS: dict[str, dict[str, float]] = {}
# Home-court edge. The rollout is otherwise home/away symmetric (HOME just inbounds first), so the
# sim can't separate winners and win-pick accuracy sits near a coin flip. This adds a logit nudge to
# the live-shot "made" outcome: +HOME_COURT_SHOT_BIAS for the home offense, -HOME_COURT_SHOT_BIAS for
# the away offense. Symmetric on purpose — it tilts the home/away split (driving win prediction)
# without moving the pooled eFG/FG% the four-factors table already gets ~right. ~0.10 lifts home eFG
# ~+1pt / drops away ~-1pt, roughly a ~2.5-pt home edge (real NBA ~2.5-3.0). 0 = off; tune against
# the win-prediction calibration + spread bias in the eval report. Applied in GameController._do_shot.
HOME_COURT_SHOT_BIAS = 0.10
# Logit bonus per second of a player's current on-court stint, added to the outgoing-sub pick so
# a long-tenured player (a star included) is *nudged* — not forced — toward coming off. 0 = off.
# Lowered from 0.15 so starters are pulled for tenure less aggressively (the stage eval under-played
# the top of the rotation); lets stars hold longer stints.
SUB_FATIGUE_WEIGHT = 0.08
# Max game-seconds a team may go without a substitution before the Controller forces one (the
# event head never targets a team, so this safety net keeps a team from playing five men 48 min).
SUB_MAX_GAP_SECONDS = 420.0

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

# --- Stint-length scheduler (StintLengthModel + GameController hybrid scheduler) ---
# When the stint-length head is loaded, the Controller commits each entering player to a stint:
# it samples a length (game-seconds on the floor) and schedules the player's exit at
# clock + length; at each dead ball a player past their scheduled exit is subbed out. The model
# regresses log-stint, so we sample with multiplicative log-space noise for rotation variety.
# STINT_SAMPLE_SIGMA is the std of that log-space noise (0 = deterministic / point estimate).
STINT_SAMPLE_SIGMA = 0.25
# Numerical cap on a sampled stint (game-seconds). There is intentionally NO lower bound — a
# short specialist stint (a one-possession 3pt shooter / rebounder) is legitimate basketball.
STINT_MAX_SECONDS = 900.0
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
DEADBALL_REBOUND_PROB = 0.06

# Rollout dials captured into each evaluation report (reporting/eval_report.py) so tuning settings
# are recorded alongside results for cross-run analysis. Order is the display order in the report.
_TUNING_KEYS = (
    "DELTA_TIME_SCALE", "MAX_DELTA", "DEADBALL_REBOUND_PROB",
    "PLAYER_TEMPERATURE", "EVENT_TEMPERATURE", "TYPE_TEMPERATURE", "RESULT_TEMPERATURE",
    "SUB_TEMPERATURE", "SUB_INCOMING_TEMPERATURE", "SUB_FATIGUE_WEIGHT", "SUB_MAX_GAP_SECONDS",
    "STINT_SAMPLE_SIGMA", "STINT_MAX_SECONDS", "FOUL_OUT_LIMIT", "SHOT_RESULT_BIAS",
    "EVENT_BIAS", "TYPE_BIAS", "HOME_COURT_SHOT_BIAS",
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
RECENCY_HALFLIFE_SEASONS = 6.0
RECENCY_FLOOR = 0.05

# --- Single full-train + batched holdout eval (full_train.py / training/full_run.py) ---
# Stop training partway through the most recent season, hold out the next FINAL_HOLDOUT_GAMES real
# games, and predict them EVAL_BATCH at a time (pausing between batches). Full-train weights go to
# their own root so the curriculum's ./artifacts is never clobbered.
FINAL_SEASON_FRACTION = 0.5
FINAL_HOLDOUT_GAMES = 100
EVAL_BATCH = 10
# Model versions live one-per-dir under ./artifacts/v<MAJOR.MINOR>/ (see models.artifacts.
# version_root / latest_version). The current weights (formerly ./artifacts_full2, "train 2.5") are
# christened v1.0. FULL_ARTIFACTS_ROOT is the default/latest root; a full train writes a new version
# dir chosen on the CLI (train.py --full --version X.Y). Keep this string equal to
# version_root(DEFAULT_VERSION) — tests/test_full_run.py asserts it.
DEFAULT_VERSION = "1.0"
FULL_ARTIFACTS_ROOT = "./artifacts/v1.0"

# --- Representative subset for the small heads (training/subset.py) ---
# The small categorical/regression heads (event/type/result/conditional-time) saturate long before
# they see the whole corpus and start to overfit, so they train on a compact, *representative*
# slice instead of every game. The slice is selected by a per-season sample RATE that is heavy on
# the modern game (so current players are well-learned) and decays gently for older seasons, but it
# stays coverage-complete: every player who appears in the train pool is guaranteed at least one
# game, so no embedding goes starved. The big player-vocab heads (player / substitution /
# stint_length) keep the full corpus — they actually need the data.
#
# Per-season sample rate for the most recent seasons, NEWEST FIRST: the newest season gets 70% of
# its games, the next 40%, the third 25%. The newest season is itself already truncated at
# FINAL_SEASON_FRACTION (we cut partway through it), so 70% of that is a modest absolute count.
SUBSET_RECENT_SEASON_RATES = (0.70, 0.40, 0.25)
# Seasons older than the recent block decay from the last recent rate (0.25), halving every
# this-many seasons — a gentle exponential tail. Coverage still guarantees every player a game, so
# old-only players pull in the older games they need regardless of the rate.
SUBSET_RECENCY_HALFLIFE_SEASONS = 8.0
SUBSET_SEED = 42                     # deterministic subset selection
SUBSET_GAMES_PATH = "./training/subset_games.json"  # persisted subset manifest (one extract step)
# Heads trained on the representative subset rather than the full corpus. All six conditional
# type/result heads share one preprocess file, so they move as a group. Everything NOT listed here
# (event_time, player, substitution, stint_length) trains on the full corpus.
SUBSET_MODEL_KEYS = (
    "event_time_cond",
    "shot_type", "shot_result", "assist_type", "turnover_type", "foul_type", "rebound_type",
)