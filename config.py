# Global configuration for CourtVisionIQ models

from pathlib import Path

# Project root (this file lives at the repo root)
ROOT_DIR = Path(__file__).resolve().parent

# Max sequence length for a game (right-padded; covers OT/overflow, truncate beyond)
MAX_SEQUENCE_LENGTH = 600

# Fixed number of on-court player slots per roster (PAD-filled below this)
ROSTER_SIZE = 5

# Fixed number of bench slots per side (PAD-filled below this, truncated above). The bench is
# everyone who plays in the game minus the five on the floor, so it runs 4-8 in practice against
# a rotation of 9-13; ten is headroom, not a limit that binds. An ARCH_KEY: a set's size changes
# the mask shapes but no weight shape, so a graph rebuilt with a different value would load
# quietly and pool over the wrong number of slots -- the same failure mode as correction K.
BENCH_SIZE = 10

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

# --- Local attention (models/backbone.py) ---
# Attention averages each row over every earlier row, and nothing pushes any head toward the last
# few -- but basketball is overwhelmingly local. These restrict the FIRST N of the NUM_HEADS heads
# in every block to a trailing window, by a banded mask ANDed into the key-padding mask; the
# remaining heads stay global. No new weights and no custom kernel, so the split is free.
#
# ARCHITECTURE, not a rollout dial: they are in models/manifest.ARCH_KEYS and deliberately NOT in
# _TUNING_KEYS. Weights trained with local heads have adapted to the restriction, so reloading
# them into an all-global graph is silently wrong rather than an error -- the manifest check is
# what catches it. A/B-ing the setting therefore means a retrain, not a re-run.
#
# --- W3: the per-game regime latent (models/regime.py) ---------------------------------------
# One free vector per training game, concatenated into the fusion, sampled once per rollout at
# inference. It exists because the simulator's two teams do not share a game: corr(home pts, away
# pts) across the sims of one game is 0.02 against a real 0.35, which makes the margin sd 16.5
# against a real 13.7 AND the total sd 16.7 against a real 19.7 -- one missing shared term, seen
# from both sides. A shrinkage dial cannot add a term that is not there.
#
# Four dimensions: enough for tempo, shooting and whistle to separate, small enough that the L2
# keeps it from memorising a game outright. These are ARCHITECTURE, not rollout dials -- they
# change weight shapes, so they belong in the manifest's arch snapshot and not in _TUNING_KEYS.
REGIME_ENABLED = True
REGIME_DIM = 4
REGIME_L2 = 1e-3

# LOCAL_ATTENTION_HEADS = 0 disables the mechanism entirely and rebuilds the pre-2.0 graph
# unchanged, which is what makes that A/B a clean comparison.
LOCAL_ATTENTION_HEADS = 2   # heads per block restricted to the window (0 = all global)
LOCAL_ATTENTION_WINDOW = 8  # rows a local head can see, inclusive of its own row

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
# v2-run1: pace ran 102.57 vs 100.58 real (+2.0), which would fit at 1.0198 -- HELD AT NEUTRAL
# ANYWAY. Pace is FGA-driven and run 1 was missing 40% of its free throws (the fouler-side bug,
# see FOUL_OFFENSE_SIDE_PROB), so possessions were counted under a regime that no longer exists
# and that +2.0 is not a clean fit. Fixing pace and the foul path in the same run would also make
# neither readable. This is run 3's first knob, once run 2 re-measures pace honestly.
DELTA_TIME_SCALE = 1.0
SUB_TEMPERATURE = 1.0      # outgoing substitution pick (legacy path) / generic sub sampling
# Incoming-sub pick temperature. The substitution head emits over the *player* vocab, so — like
# the actor head — its small, real preferences (which bench player actually checks in) should
# drive the pick, not be smoothed toward a uniform bench. <1 sharpens; push toward 0 to approach
# argmax (the single most-likely sub every time, at the cost of all rotation variety). Sharper
# than the actor head by default since a coach's bench order is more concentrated than shot usage.
# Lowered from 0.7: the stage eval over-played the deep bench (rank 15+ by +4..+14 min) and
# under-played starters (~9 min); sharpening concentrates check-ins on the real 8–9 man rotation.
# 0.45 was fit to deep-bench over-play under the STINT SCHEDULER, which workstream 11 deleted.
# v2-run1 ran this neutral and player-minutes MAE came in at 5.552 against v1.0's 5.664 with the
# sharpening on, so the sub_decision head does not need it. Stays neutral.
SUB_INCOMING_TEMPERATURE = 1.0
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
# 2.0 REFIT (v2-run1): made fits at -0.022, i.e. ZERO. This is the single clearest sign the
# workstreams reached the weights -- v1.0 needed +0.40 here and the refit needs nothing, so the
# make-rate defect this dial existed to paper over is fixed at the source rather than dialled
# around. eFG bias went -0.017 -> -0.008 with the dial OFF. Left at 0 rather than at -0.022:
# that is noise, and points are still short until the free throws come back.
# `blocked` does need it -- 11.00 blocks/game vs 9.55 real, fitted from the 3-way live-shot table.
SHOT_RESULT_BIAS: dict[str, float] = {"blocked": -0.134}
# Per-ZONE override of the above, keyed zone then outcome, e.g. {"rim": {"made": 0.1}}. Merged on
# top of SHOT_RESULT_BIAS for the zone actually sampled, so an absent zone just gets the global.
# Default {} = global only, which is what v1.0 had to live with: the fitted per-type ideal was
# already known to differ (+0.22 on 2pt, +0.35 on 3pt) with no hook to express it, so rim make
# rate and long-mid frequency competed for one number. Fifteen zones give that hook. Fit from
# zero after the 2.0 train -- every value carried over from full1/full2 is against a vocabulary
# that no longer exists.
SHOT_RESULT_BIAS_BY_ZONE: dict[str, dict[str, float]] = {}
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
# 2.0 REFIT (v2-run1), from the open-play mix (shot / assist / turnover / foul per game, FTs and
# rebounds excluded since neither comes from this head):
#   assist    54.34 sim vs 48.55 real  -> -0.087, applied.
#   foul      44.42 sim vs 41.05 real  -> would fit at -0.053, NOT applied. The foul-type refit
#             already takes the total to 40.93 on its own (technicals 4.63 -> 1.05 is most of it),
#             so applying this too would double-correct -- the same trap the run 1 package was
#             designed to avoid.
#   turnover  22.04 sim vs 22.80 real  -> would fit at +0.060, NOT applied. Box turnovers count
#             offensive fouls, and once those land at 3.82/game the box number is 12.93 against
#             12.92 real. The event count is mildly under; the number that matters is exact.
EVENT_BIAS: dict[str, float] = {"assist": -0.087}
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
# ------------------------------------------------------------------------------------------------
# 2.0 REFIT, from v2-run1 (the first 2.0 eval, every dial at neutral -- that was the point of it).
# Every v1.0 note above is history now; the live values below are fitted against 2.0's heads.
# Method: pool the sim play-by-play (20 games x 5 sims) and the real 2023 cleaned season, and take
# log(real_share / sim_share) per token, shifted so the modal token sits at 0.
#
#   foul_type    -- NOT a plain log-ratio, because the head is masked per side and the two masks
#                   overlap on only four tokens. Solved jointly with FOUL_OFFENSE_SIDE_PROB as a
#                   two-sided masked multinomial (least squares on the per-token shares). That
#                   coupling is the whole point: `offensive` ran 13.34/game vs 3.82 real, but
#                   almost all of that excess was the side draw, not the head -- fitting the
#                   marginal alone would have put ~-1.25 on it and double-corrected hard once the
#                   side was fixed. The joint solve puts -0.386 on it. It reproduces every token:
#                   shooting 2pt 20.23/game (real 20.23), personal 11.53 (11.53), offensive 3.82
#                   (3.82), loose ball 2.41 (2.41), technical 1.05 (1.05), total 40.93 (40.99).
#                   `shooting 3pt` at +0.774 is the one value to watch -- a large offset on a
#                   small-mass token is the exact shape of v1.0's `loose ball` +1.2 mistake -- but
#                   it is worth only ~2 FTA if it overshoots, against the ~21 being recovered.
#   rebound_type -- a clean log-ratio: all four tokens are always allowed and the head is sampled
#                   with next_player=None, so no mask distorts it. The two TEAM tokens workstream
#                   7 added are over-produced 3.2x (31.7/game vs 9.25 real). A team rebound
#                   credits no player, so it leaves the box score entirely -- that is the whole of
#                   dreb -10.45/team, and it is also why oreb_pct read .345 vs .242: the
#                   denominator collapsed, not the numerator. Rebound EVENTS were already right
#                   (97.4/game vs 98.35), which is what kept this invisible until the split was
#                   counted directly. Another one that no test would have found.
#
# Left at {} deliberately, with the measurement, so run 2 stays readable:
#   turnover_type -- steal lands exactly (14.53/game vs 14.52) and steals are what the box score
#                    measures. error/violation are under (4.57/2.94 vs 5.40/4.26) and would fit at
#                    +0.167/+0.371, but applying them renormalizes mass OFF steal and breaks the
#                    one number here that is already right.
#   assist_type   -- mid-range zones are mildly inflated (mid_base_r 1.37/game vs 0.40), but eFG
#                    is -0.008 and tpa/tpm are the best they have ever been (+0.60 / -0.30, from
#                    +2.31 / -0.60 in v1.0). Do not tune what 2.0 just fixed.
# ------------------------------------------------------------------------------------------------
TYPE_BIAS: dict[str, dict[str, float]] = {
    # Relative to "personal" = 0. Read together with FOUL_OFFENSE_SIDE_PROB -- they were solved
    # as one system and neither is meaningful on its own. "personal take" / "transition take" /
    # "away from play" were left free at 0 and land within 0.03/game of real anyway.
    "foul_type": {"shooting 2pt": 0.268, "shooting 3pt": 0.774,
                  "offensive": -0.386, "loose ball": -0.761,
                  "technical": -1.188, "flagrant-1": -1.491, "flagrant-2": -1.912},
    "turnover_type": {},
    # assist_type was {"3pt": 0.10}. That key can no longer match anything -- assist rows now
    # carry a zone token, not the 2pt/3pt binary -- so it is removed rather than left as config
    # that silently does nothing. Re-key per zone when the dials are fitted from zero after the
    # 2.0 train (the mechanism needs no change: TYPE_BIAS is already head -> token).
    "assist_type": {},
    # Relative to "defensive" = 0, the token the head under-produces (.433 of rebounds vs .684).
    "rebound_type": {"offensive": -0.564,
                     "team offensive": -1.617, "team defensive": -1.653},
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
# Probability that the player who commits a foul is on the OFFENSE (GameController._do_foul).
# Like HOME_COURT_SHOT_BIAS, this supplies information the model structurally lacks rather than
# calibrating something it knows: the player head is conditioned on the sequence, not on which
# side has the ball, so asked for a fouler out of all ten it splits ~50/50 by side — and
# PLAYER_TEMPERATURE=2.0 flattens away whatever weak signal it does carry. The controller now
# draws the side from this dial FIRST and samples the fouler from that side's five, so the head
# still decides who fouls while the rate is pinned here.
#
# 0.1286 is a joint least-squares solve of the two-sided masked foul-type multinomial (this dial
# plus TYPE_BIAS["foul_type"]) against the real 2023 per-game marginals, fitted to v2-run1. It
# agrees with an independent read of the same file: offensive fouls are offense-side by
# definition (3.82/game) and roughly half of loose-ball + technical + flagrant are
# (~1.8/game), i.e. ~13.7% of 40.99 fouls/game.
#
# RE-MEASURE after any retrain of the player head, and note the two are coupled — a lower
# PLAYER_TEMPERATURE lets more of the head's own (weak) side preference through, which would
# show up here as an over-correction. Run 1 held PLAYER_TEMPERATURE at 2.0 deliberately, so this
# fit is against that value.
FOUL_OFFENSE_SIDE_PROB = 0.1286
# The and-1 read off the conditional time head. For a foul sampled as the very next row after a
# made field goal, the controller asks the head for its gap and takes P(and-1) = 1 - gap / SCALE,
# clipped to [0, 1], BEFORE the side and the type are drawn: a hit is a defensive shooting foul at
# the basket's clock with the scorer shooting one; a miss is an ordinary foul on the NEXT
# possession with its gap floored at SCALE (the long branch of the mixture).
#
# Why a gap, not a rate: the real gap after a basket is 0 on an and-1 (35% of these fouls, 2023)
# and ~13s otherwise, and the head regresses the MEAN -- so its output is (1 - p) * later_gap and
# p is recoverable from it. Run 3's recorded gaps show the head learned the structure: rim 5.4s
# vs corner three 12.0s (real and-1 rates .44 vs .04; rank corr -0.94 across 14 zones), Giannis
# 4.5s vs Buddy Hield 10.5s (Spearman -0.61 across 250 scorers). Inverting reproduces the zone
# rates to 0.033 MAE where a flat 0.338 is off by 0.142. The variation is the model's; this dial
# sets only the level.
#
# 10.10s is the value at which the implied mean over run 3's 55,794 fouls-after-a-basket equals
# the real 0.338 (5.24 of 15.50/game over 1320 games). The physical later-foul gap is 13.2s; the
# head's mean runs short of the mixture mean (7.4s vs 8.7s), and the fit absorbs that. RE-FIT
# after any retrain of the conditional time head. See dials/README.md, `v2-run4.json`.
AND_ONE_GAP_SCALE = 10.10
# SUB_FATIGUE_WEIGHT is GONE (2.0, workstream 11). It was a logit bonus per second of a player's
# on-court stint, nudging the outgoing pick toward whoever had been on longest -- a hand-written
# stand-in for exactly what the roster encoder now sees directly, since every head reads stint
# seconds, minutes played and fouls per player. Do not bring it back to 'help' the rotation: the
# model has the input, and a dial on top of it is a second opinion that nothing reconciles.
# Max game-seconds a team may go without a substitution before the Controller forces one (the
# event head never targets a team, so this safety net keeps a team from playing five men 48 min).
# Raised 420 -> 600 when the stint scheduler still set the cadence: real teams average one sub
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

# --- Eval durability (long pooled runs) ---
# Also perf/safety knobs, NOT tuning dials: keep them out of _TUNING_KEYS or a benign difference
# here would make assert_one_tuning claim the run mixed physics.
# A CUDA OOM usually poisons TensorFlow for the life of the process, so a shard that fails this
# many chunks in a row stops rather than burning GPU hours failing identically on every remaining
# game; the pool's remainder wave respawns it with a clean context.
EVAL_MAX_CONSECUTIVE_GAME_FAILURES = 3
# How often the supervisor rebuilds report.html + data/*.parquet from the finished per-game
# records while the shards are still running, so a 36-hour run is queryable long before it ends.
EVAL_REPORT_EVERY_SEC = 300

# STINT_SAMPLE_SIGMA / STINT_LENGTH_SCALE / STINT_MAX_SECONDS are GONE (2.0, workstream 11),
# with the stint-length head and the scheduler that consumed it. The Controller used to commit
# each entering player to a sampled stint and pull him when the clock reached it; rotation is
# now a decision the sub-decision head makes wherever Rule 3 permits a substitution.
#
# STINT_LENGTH_SCALE is the one worth remembering. It existed because the head regressed LOG
# stint, so its point estimate was a geometric mean that under-predicted a right-skewed
# duration by ~30%, and the dial multiplied the gap away. A dial correcting a distributional
# artefact of the target is a sign the target is wrong, and it was: the question was never how
# long a player will stay on, it was whether anyone comes off here.
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
# DEADBALL_REBOUND_PROB is gone. It flipped possession on a coin toss and emitted NO ROW,
# so the model never saw a team rebound and could not learn its share. The rebound-type
# head now carries "team offensive"/"team defensive" tokens and learns it from the data.
# Its fitted history (0.09 -> 0.10, chasing the OREB/DREB split and the pace bias) is in
# git; do not resurrect the value -- refit rebound_type from zero after the 2.0 train.

# Post-hoc linear calibration on predicted point margin, applied only when aggregating spread
# metrics for the eval report (simulation/eval_metrics.py) -- never to the raw per-game record, so
# record.json / games.parquet / the margin scatter plot keep the literal model output and this can
# be refit without touching sim data. NOT a rollout dial (nothing at sim time reads it, so it is
# deliberately absent from _TUNING_KEYS below). Regressing actual margin on predicted margin across
# v1.0's full1 holdout: actual = -0.31 + 0.745*predicted -- predicted margins run ~25% too extreme
# for their information content. slope=1.0/intercept=0.0 = off.
# Neutralised for the 2.0 train: 0.745 / -0.31 were regressed on v1.0's full1 holdout, so
# leaving them on would report 2.0's spread and win metrics through v1.0's calibration and
# quietly flatter (or flatten) them. This is refit from games.parquet without re-simulating
# anything -- the raw per-game record never passes through it -- so measure 2.0 raw first.
MARGIN_CALIBRATION_SLOPE = 1.0
MARGIN_CALIBRATION_INTERCEPT = 0.0

# Rollout dials captured into each evaluation report (reporting/eval_report.py) so tuning settings
# are recorded alongside results for cross-run analysis. Order is the display order in the report.
_TUNING_KEYS = (
    "DELTA_TIME_SCALE", "MAX_DELTA",
    "PLAYER_TEMPERATURE", "EVENT_TEMPERATURE", "TYPE_TEMPERATURE", "RESULT_TEMPERATURE",
    "SUB_TEMPERATURE", "SUB_INCOMING_TEMPERATURE", "SUB_MAX_GAP_SECONDS", "FOUL_OUT_LIMIT",
    "SHOT_RESULT_BIAS", "SHOT_RESULT_BIAS_BY_ZONE", "EVENT_BIAS", "TYPE_BIAS",
    "HOME_COURT_SHOT_BIAS", "FOUL_OFFENSE_SIDE_PROB", "AND_ONE_GAP_SCALE",
)


def encode_tuning(values: dict) -> dict:
    """Put dial values in the shape a report stores: dict-valued dials JSON-encoded to a compact
    string, so each sits cleanly in a single Parquet column rather than a struct.

    Shared by :func:`tuning_snapshot` (live module values) and any caller reporting a dial file
    written by :func:`write_dial_file`, which keeps dicts as dicts so they round-trip through
    ``apply_dials``. Both must land in run_summary.parquet identically typed or the cross-run
    knobs-to-results table stops concatenating.
    """
    import json as _json
    return {k: _json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
            for k, v in values.items()}


def tuning_snapshot() -> dict:
    """The live values of every rollout dial (read from this module at call time).

    Reading the module globals means edits to this file between runs are reflected accurately, so
    each evaluation report records exactly the tuning that produced it. Dict-valued dials
    (``SHOT_RESULT_BIAS`` / ``EVENT_BIAS`` / ``TYPE_BIAS``) are JSON-encoded to a compact string
    so each sits cleanly in a single Parquet column.
    """
    g = globals()
    return encode_tuning({k: g[k] for k in _TUNING_KEYS})


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

# The heads GameController needs to play a game. Checked at LOAD (shell/actions.py) so a
# missing head fails immediately rather than mid-rollout, and again in the controller's own
# constructor. Defined HERE because three copies of it existed -- the shell's, the
# controller's, and the FakeSim in tests/test_controller.py -- and adding sub_decision to two
# of them broke 87 tests that had nothing to do with rotation. config is the one module all
# three can import without pulling in TensorFlow.
REQUIRED_HEADS = ("player", "substitution", "sub_decision", "shot_type", "shot_result",
                  "assist_type", "turnover_type", "foul_type", "rebound_type")

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

# --- Loss masking (event + time heads: train only where the sim actually asks) ---
# The controller expands ONE sampled play into several emitted rows -- the shot after an
# assist, the block after a blocked shot, every free throw of a trip -- and never asks the
# event head 'what next' at the intermediate ones. Training there teaches a question that is
# never asked at inference. Substitutions are masked already (they are injected by the
# rotation scheduler, not sampled); this covers the rest. Measured over the cleaned corpus,
# the masked share of event-head training positions goes 10.6% -> 31.5% on 2022-23, and
# 9.0% -> 29.8% / 9.9% -> 30.2% on 2002-03 / 2012-13:
#
#     python -m models.game_state_features --seasons 2003,2013,2023
#
# A TRAINING knob, not a rollout dial and not architecture: it changes no weight shapes and
# nothing at sim time reads it, so it is in neither _TUNING_KEYS nor ARCH_KEYS. The mask is
# built at preprocess and stored in the npz either way -- this switches whether the dataset
# APPLIES it, so the A/B is two trains against the same preprocess, no re-clean.
MASK_CONTINUATION_ROWS = True
# The time head additionally skips the last row of each period: that gap spans a buzzer, and
# the controller clamps at the boundary rather than sampling across one. The event head is
# still asked what opens the next period, so this is time-only.
MASK_PERIOD_BREAK_TIME = True

# --- Single full-train + batched holdout eval (full_train.py / training/full_run.py) ---
# Stop training partway through the most recent season, hold out the next FINAL_HOLDOUT_GAMES real
# games, and predict them EVAL_BATCH at a time (pausing between batches). Full-train weights go to
# their own root so the curriculum's ./artifacts is never clobbered.
FINAL_SEASON_FRACTION = 0.5
# Raised 100 -> 300 after v2-run2. 100 games cannot resolve a dial change on the win/spread
# metrics, which is the only reason those metrics kept reading as flat across v1.0 full1..full4
# and v2-run1..run2. Paired over the same 100 game ids, run 2 vs v1.0 full4-s100 came in at
# Brier +0.0128 (SE 0.0140), score-view Brier +0.0029 (SE 0.0133), |margin error| +0.48 (SE
# 0.58) -- every comparison inside one sigma. The per-game paired Brier sd is 0.133, so 80%
# power at two-sided 5% needs ~350 games to see a 0.02 Brier move and ~1,400 to see 0.01. 300
# puts a 0.02 move (about half the whole span from an always-pick-home baseline at 0.245 to a
# sportsbook at ~0.20) just inside reach; the box-score biases were always resolvable, since
# they are scored over 200 team-games and 2,071 player-games rather than 100 win/loss bits.
#
# For metric RESOLUTION more games beats more sims: the paired SE falls as 1/sqrt(games) while
# sims only remove the per-game Monte-Carlo term (which at 20 sims inflates the sim-count Brier
# by ~0.010 and attenuates spread corr from a signal 0.420 down to 0.375). Both matter, so the
# recommended shape is 300 games x --monte-carlo 50: expected spread corr ~0.400, Brier MC
# inflation ~0.004, paired Brier SE ~0.008. That is 15,000 sims against run 2's 2,000 and v1.0
# full4-s100's 10,000. STAGE_SIMS deliberately STAYS at 21 -- it feeds EVAL_POOL_JOBS, whose
# comment explains why a high default there is a RAM trap; pass the sim count on the CLI.
#
# The window only ever extends FORWARD from the stored boundary_idx, so the train/holdout cut
# cannot move and the first 100 ids stay the first 100 -- v2-run1 and v2-run2 remain directly
# comparable, and their own results/<run>/holdout.json pins keep them at 100 games regardless.
# Room to grow: the corpus holds 26,969 games against a boundary at 26,267, so 702 are available.
# On an ALREADY-TRAINED model this constant is not read again (full_run.setup consumes it, and
# re-running setup would reset status/trained_models) -- use `python train.py --extend-holdout`.
# 3.0: this is the POOL, not one run's holdout. Runs rotate through HOLDOUT_WINDOWS disjoint
# windows of HOLDOUT_WINDOW_GAMES games each, so six runs over a season cover 600 DISTINCT games
# instead of re-scoring the same 100 six times (docs/v3_direction.md §4). 700 = 100 × 7 against the
# 702 games available after the cut.
#
# Widening the pool passes `extend_holdout`'s prefix guard untouched, which is the point: the
# existing 100 ids ARE the first 100 of the 700, so window 0 is byte-identical to the games
# v2-run1..4 were scored on and no invariant is weakened to get there.
FINAL_HOLDOUT_GAMES = 700
# One run's holdout. Kept separate from the pool so raising one never silently changes the other.
HOLDOUT_WINDOW_GAMES = 100
HOLDOUT_WINDOWS = FINAL_HOLDOUT_GAMES // HOLDOUT_WINDOW_GAMES
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
# sub_decision) keep the full corpus — they actually need the data.
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
# Heads trained on the representative subset rather than the full corpus. All SEVEN conditional
# type/result heads share one cond_*.npz, so they move as a group — listing six of them was true
# of nothing: timeout_team already trained on subset rows, because run_stage builds that shared file
# once from cond_keys[0] (shot_type, which is listed). The list was wrong, not the behaviour, and
# models.pipeline.run_stage now raises rather than let the two drift again. Everything NOT listed
# here (event_time, player, substitution, sub_decision) trains on the full corpus.
SUBSET_MODEL_KEYS = (
    "event_time_cond",
    "shot_type", "shot_result", "assist_type", "turnover_type", "foul_type", "rebound_type",
    "timeout_team",
)