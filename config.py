# Global configuration for CourtVisionIQ models

from pathlib import Path

# Project root (this file lives at the repo root)
ROOT_DIR = Path(__file__).resolve().parent

# Max sequence length for a game (right-padded; covers OT/overflow, truncate beyond)
MAX_SEQUENCE_LENGTH = 600

# Fixed number of on-court player slots per roster (PAD-filled below this)
ROSTER_SIZE = 5

# --- Rollout sampling (GameController / GameSimulator) ---
# Per-head softmax temperature for the rollout: <1 sharpens (emphasizes the head's preference),
# >1 flattens toward uniform, 1.0 is the raw model. Every categorical pick routes through
# _masked_sample / _constrained_sample, which apply these.
#
# The player/actor head (shooter, rebounder, assister, fouler, FT shooter) is sharpened below 1
# so usage concentrates on the better players instead of spreading evenly across the on-court
# five — the small, real differences in the head's logits should drive who shoots, not be smoothed
# away. The rest default to 1.0 (raw) and are exposed as knobs for future tuning.
PLAYER_TEMPERATURE = 0.8
EVENT_TEMPERATURE = 1.0    # next-event head (shot / foul / turnover / … mix)
SUB_TEMPERATURE = 1.0      # outgoing substitution pick (legacy path) / generic sub sampling
# Incoming-sub pick temperature. The substitution head emits over the *player* vocab, so — like
# the actor head — its small, real preferences (which bench player actually checks in) should
# drive the pick, not be smoothed toward a uniform bench. <1 sharpens; push toward 0 to approach
# argmax (the single most-likely sub every time, at the cost of all rotation variety). Sharper
# than the actor head by default since a coach's bench order is more concentrated than shot usage.
SUB_INCOMING_TEMPERATURE = 0.7
TYPE_TEMPERATURE = 1.0     # shot_type / assist_type / turnover_type / foul_type / rebound_type
RESULT_TEMPERATURE = 1.0   # shot_result (made / missed / blocked)
# Logit bonus per second of a player's current on-court stint, added to the outgoing-sub pick so
# a long-tenured player (a star included) is *nudged* — not forced — toward coming off. 0 = off.
SUB_FATIGUE_WEIGHT = 0.15
# Max game-seconds a team may go without a substitution before the Controller forces one (the
# event head never targets a team, so this safety net keeps a team from playing five men 48 min).
SUB_MAX_GAP_SECONDS = 420.0

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

# --- Curriculum training (training/curriculum.py + training/chronology.py) ---
# The full corpus is trained in contiguous, cumulative stages. After each stage we predict the
# next block of real games as a sequential holdout (no random split) and score the simulator.
# Number of sequential games held out for evaluation after each stage's training boundary.
HOLDOUT_GAMES = 10
# Predictions run per holdout game when scoring a stage (the simulator is stochastic; we average).
STAGE_SIMS = 11
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