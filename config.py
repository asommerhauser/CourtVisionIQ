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
# --- W4 rung 1: scheduled sampling (models/scheduled_sampling.py) ------------------------------
# With probability p the previous event's token in a training sequence is the model's own sample
# rather than the real one, so it learns to keep going after its own mistakes. Training-side knobs,
# NOT rollout dials: they change no weight shapes and nothing at sim time reads them, so they belong
# in neither _TUNING_KEYS nor the arch snapshot -- the same classification MASK_CONTINUATION_ROWS
# already has, and for the same reason.
#
# p is capped low and ramped late. The model has to fit the next-step distribution before its own
# samples are worth anything, and the mixed rows carry a real game state against a sampled token
# (see the module docstring on why the derived columns cannot be re-derived in-graph), so a high p
# would spend most of the run on contexts that cannot occur.
# --- W4 rung 2: checkpoint selection on rollout metrics (models/rollout_selection.py) ---------
# Every head early-stops on val_loss -- one-step NLL against the real history -- and is then judged
# on a 600-step self-fed rollout. Whether the epoch that minimises NLL is the epoch that ROLLS OUT
# best is an open question, and W1's probes make it pressing: the simulator benches a player in
# foul trouble 24% of the time against a real 78%, with every input it needs already in the weights.
# No next-step loss can see that, because each individual prediction is roughly right and it is the
# composition over hundreds of steps that is wrong.
#
# OFF by default: rung 2 is a SECOND PASS. Scoring a rollout needs all twelve heads, and during a
# from-scratch train the first head has no bundle to roll out. Finish the train, then retrain the
# head under test warm-started off the finished bundle -- which also makes the A/B clean.
#
# Cost, from the run-4 logs: 20 games x 10 sims is 200 game-sims, ~7.5 GPU-minutes per evaluation;
# every third epoch over a 30-epoch stage is ~1.3 GPU-hours on the train.
# 3.2 W8: on. The bridge that constructs the score function now exists (models/rollout_bridge.py),
# setup() records the train-tail games eval_game_ids samples from, and models.pipeline has a channel to
# pass the callable to the event/time head. Before all three, turning this on would have selected on a
# score computed over an empty game set.
# 2026-09-23: OFF. The first full rung-2 pass (event_time, 14 evaluations at 20 games x 3 sims) scored
# 17.6-24.3 with no trend while val NLL fell monotonically -- noise at that size, at ~85% of wall time.
# 3.2 runs rung 3 (the replay pass) unconditionally, so this gate has no decision left to make.
ROLLOUT_SELECTION = False
ROLLOUT_EVAL_EVERY = 3
ROLLOUT_EVAL_GAMES = 20
# 10 -> 3 on 2026-09-23, sized to the measured rate rather than to an estimate. One process runs
# ~3.2 game-sims/min however the sims are scheduled (see ROLLOUT_EVAL_BATCH_SIZE), so 200 sims was
# 63 and 68 min on two runs and 60 is ~19 min, inside ROLLOUT_EVAL_BUDGET_MIN. Games are kept at 20
# and sims cut, because the behaviour probes pool over games and the per-game box noise is what
# averaging across twenty games is for. Raise it only with a measured rate that pays for it.
ROLLOUT_EVAL_SIMS = 3
# Games before the train cut to sample the eval set from. NEVER a holdout window: selecting a
# checkpoint against the holdout turns the report into a training metric, and nothing downstream
# would look wrong.
ROLLOUT_EVAL_TAIL = 500
# Score weights. Dispersion is scaled into points so it is commensurable with box MAE; the
# behaviour weight makes a fully-absent game-state behaviour cost about as much as a point of MAE,
# so a checkpoint cannot win by fixing the box while still never benching anyone.
ROLLOUT_SCORE_DISPERSION_WEIGHT = 2.0
ROLLOUT_SCORE_BEHAVIOUR_WEIGHT = 1.0
# Rung 2's rollout runs INSIDE the training process, which is the one path where the eager
# inference default is fatal: one scored evaluation is ~200 game-sims of tiny forward passes, and
# eager dispatch leaves the card idle while a single core does Python op-dispatch (see the
# "Compiled inference" block in simulation/game_simulator.py). Measured 2026-09-22 on the first
# real invocation: 9.5 hours without finishing ONE evaluation, 122% CPU, 1% GPU. So this path opts
# IN to the compiled forward regardless of CVIQ_TF_INFER, which stays off elsewhere until it is
# measured there too. Set False to reproduce the eager behaviour.
ROLLOUT_COMPILED_INFERENCE = True
# Wall-clock ceiling for ONE evaluation, in minutes; exceeding it aborts the train with the measured
# number instead of paying the same cost at every ROLLOUT_EVAL_EVERY epoch for the rest of the run.
# 25 is ~3x the 7.5-minute estimate above -- wide enough to absorb the one-time simulator load the
# first evaluation pays, narrow enough to fail inside a single epoch rather than overnight.
# That estimate was wrong by ~8x (measured 63-68 min for 200 sims); ROLLOUT_EVAL_SIMS is now sized
# to the measured rate, and the budget is left where it was so the guard still means something.
ROLLOUT_EVAL_BUDGET_MIN = 25.0

SCHEDULED_SAMPLING = True
SCHEDULED_SAMPLING_MAX_P = 0.25
SCHEDULED_SAMPLING_WARMUP_EPOCHS = 3
SCHEDULED_SAMPLING_RAMP_EPOCHS = 10

REGIME_ENABLED = True
REGIME_DIM = 4
REGIME_L2 = 1e-3

# --- W4 rung 3: the weighted replay pass (models/replay.py, training/replay_pass.py) ------------
# One pass, AFTER the main train, over one game in ten of the training subset, ten sims each. The
# sibling set is what makes the leave-one-out baseline work, so it is ten sims of the SAME game
# rather than one sim of ten games. Never a holdout window: fine-tuning on one would turn the
# report into a training metric.
REPLAY_GAME_FRACTION = 0.1
REPLAY_SIMS_PER_GAME = 10
# One pass means one epoch. The same pass every epoch of a thirty-epoch stage is ~96 GPU-hours,
# which is worse than the naive form this design exists to avoid (W11).
REPLAY_EPOCHS = 1
# A tenth of the train LR. NOT in the spec, chosen here: one pass over ~5,200 sim-games at 3e-4
# moves the weights about as far as several ordinary epochs, and W11's gate says the pass must not
# worsen margin dispersion. A fine-tune free to overwrite the bundle cannot be judged against that.
REPLAY_LR = 3e-5
# Sim ids live above every real id so the two can never collide, and carry the real id in their
# digits so an advantage finds its way back to the rows it belongs to (training/replay_corpus.py).
REPLAY_ID_BASE = 1_000_000_000
# The pass writes a NEW bundle rather than overwriting the one it started from -- arm 3 is compared
# against arm 2, so arm 2 has to still exist when it finishes.
REPLAY_ARTIFACTS_SUFFIX = "-kpi"

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
# Rung 2 sizes its own: None means one slot per sim, so a whole evaluation
# (ROLLOUT_EVAL_GAMES x ROLLOUT_EVAL_SIMS) is in flight at once and never steps in waves.
# Why, and why NOT a big fixed width: slots are concurrent, and game-sims of similar length finish
# together, so at 48 slots 200 sims completed in waves 15 min apart (63 and 68 min, 2026-09-23).
# 8f85ee0 read that as "width is free" and set 200. It is not: the coordinator is a barrier, every
# slot's controller step runs in Python under ONE GIL, so a round should cost ~slots x the per-sim
# Python work, making a 200-wide wave ~4x a 48-wide one (inferred, not yet measured at 200). The
# repo's own full-batch figure (~3.5 sims/min per process, run-4 logs) is the rate rung 2 measured
# -- the process was already at its ceiling, and only fewer sims or more processes change the
# total. The rollout heartbeat now prints fwd/s and avg batch, which settles it. Matching slots to
# sims removes the one real waste (a last partial wave), which is all width can buy here.
ROLLOUT_EVAL_BATCH_SIZE = None
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

# --- Training corpus floor (3.2 W2: seasons before this leave the TRAINING pool) ---
# Seasons 2003-2010 are dropped from training (moved 2008 -> 2011 by decision, 2026-09-17). Not as a fix for the static-identity problem --
# docs/v3_direction.md 1f is right that it is not one -- but as the simplification that makes the
# vocabulary floor below possible: the vocabulary is built from what clears a games threshold inside
# the subset, and old-era players are exactly the rows that would otherwise sit in the embedding
# table under-trained.
#
# Expect it to be NEUTRAL on every metric. Old seasons are already discounted twice, once by
# SUBSET_RECENCY_HALFLIFE_SEASONS in sampling and once by RECENCY_HALFLIFE_SEASONS / RECENCY_FLOOR in
# the loss. Every season this removes is already pinned at the RECENCY_FLOOR of 0.05 -- the halflife
# of 3.0 reaches the floor at about seven seasons back, so 2016 and older are all there -- and their
# subset sampling rates run 0.11 (2010) down to 0.0021 (2003). Multiplying the two, a 2010 game
# carries about 0.5% of the gradient a current game does, and 2003-2010 together come to roughly 37
# current-game equivalents: one to two percent of the total.
#
# The cost the direction names still stands: rare tokens get rarer, and the rarest foul and rebound
# sub-types are where it would show. Note also that 2011 removes three more seasons than the 2008 the
# direction proposed, and MEASURED it is the *milder* cut for the vocabulary floor -- it drops 183
# players (1,797 -> 1,614) who were almost all low-exposure old-era names, so the median player's
# subset exposure RISES from 31 games to 34. See docs/v3_2_progress.md measurement 2.
#
# WHERE this is applied is load-bearing. game_id is POSITIONAL -- data_loading.season_offsets shifts
# each season file's ids past every earlier file's maximum, and the raw per-season ranges are not
# even ordered by season (2016 holds 1313-2628, 2019 holds 1-1312). Filtering the FILE LIST would
# therefore renumber every surviving game, invalidating training/full_run_state.json,
# training/subset_games.json and every results/<run>/holdout.json, and breaking the property that
# holdout window 0 is byte-identical to the games v2-run1..4 scored. So the floor is applied to ROWS,
# after the offset walk, by data_loading.load_training_corpus -- every id survives and only `pos` and
# `boundary_idx` renumber.
#
# It is also why cleaned_csvs stays unfiltered, which makes the direction's own warning -- "cut the
# training games, never the sidecar", the thing it calls the single easiest mistake to make -- true by
# construction: player_priors and season_context walk cleaned_csvs directly, so the priors chain
# still seeds 2008 from real 2007 production.
#
# None means no floor. The test suite sets None, because 18 test modules build 2003 fixtures.
MIN_TRAIN_SEASON = 2011

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

# --- Cross-roster attention (3.2 W7) ---
# The two rosters pass through ONE weight-tied encoder independently and meet only at the fusion concat
# (docs/v3_2_direction.md 5.1), so nothing in the graph can represent one lineup AGAINST another: a
# switch-heavy defence and a rim-protecting one are the same input to the offence's representation.
#
# With this on, each roster's per-slot representations attend over the other's before pooling, through
# ONE shared attention block used in both directions -- so it learns "how a lineup reads an opponent"
# rather than which side of the ledger a team sits on.
#
# ARCHITECTURE, not a dial: off produces a graph with different layers, which load_weights would match
# by name and silently skip. It is in models.manifest.ARCH_KEYS for that reason.
CROSS_ROSTER_ENABLED = True

# --- Context modulation, FiLM (3.2 W6) ---
# Season, the team priors and the regime latent all enter ONCE today, as columns in a wide concat
# projected to MODEL_DIM, and then have to survive six residual blocks on their own
# (models/backbone.py fusion_concat -> fusion_projection). Season is not under-weighted, it is
# UNDER-PLUMBED: nothing downstream can condition its computation on which season or which night it
# is, only on a few of the 384 dimensions it was compressed into.
#
# FiLM fixes the plumbing rather than the weighting. One game-context vector drives a per-block scale
# and shift on the residual stream, so every layer's computation is modulated by the context instead
# of the context being one column at the bottom.
#
# Deliberately NOT extra width. docs/v3_direction.md 5.4: heads reach the base rate in ~5 epochs and
# then memorise, so the inputs do not contain the answer and capacity is not the binding limit --
# widening MODEL_DIM in the same cycle the corpus shrinks fivefold would make it worse. FILM_DIM is a
# bottleneck for exactly that reason: the per-block projections read a 64-wide summary, not the raw
# ~160-wide context, which keeps this at roughly 600k parameters a head against the backbone's ~10.6M.
#
# The scale and shift are ZERO-INITIALISED and applied as h * (1 + gamma) + beta, so at initialisation
# the modulation is the identity and the graph starts out numerically the same as one built without it.
# That is what makes the A/B honest: FiLM has to earn its effect from zero rather than perturbing the
# stream before training begins.
FILM_ENABLED = True
FILM_DIM = 64

# --- Player vocabulary floor (3.2 W4) ---
# A player needs this many games INSIDE the subset to get his own embedding row. Everyone below the
# floor is aliased to an anonymous slot token, keeping his season-to-date priors and losing his
# identity row.
#
# Why: docs/v3_direction.md 1f names the 2,153 x 192 player embedding as where the memorisation
# lives. This is the one capacity change in 3.2 and it goes DOWNWARD -- long-retired players leave
# with the corpus cut, and the thin rows the floor catches leave here. Measured at the 2011 cut, a
# floor of 20 takes the table to 959 rows and makes 4.0% of all minutes anonymous (2.0% of the 2021+
# minutes the scored holdout is drawn from).
#
# Why anonymity is affordable: the per-player prior scalars enter the roster set encoder ADDITIVELY,
# before the SAB layers (models/roster_set_encoder.py:137). An anonymous player therefore still
# arrives carrying his production profile -- and for a deep-bench player the embedding row was mostly
# noise. See docs/v3_2_progress.md measurement 2 for the floor table it was chosen from.
#
# Set to None to keep every player (the test suite does this: its fixtures have no subset manifest).
MIN_PLAYER_SUBSET_GAMES = 20

# Anonymous slots are NOT a fixed count: the assignment colours the co-occurrence graph of below-floor
# players, so two who ever appear in the same game never share a slot, and the number of slots is
# whatever that needs. Measured on the real corpus at floor 20: 36 slots for zero collisions, against
# 12 anonymous players in the busiest single game.
#
# That is the whole reason a GLOBAL name -> slot map is enough, and no per-game bookkeeping is needed.
# A naive `rank mod slots` assignment collides in 7.4% of games at 16 slots and 2.2% at 64; the
# colouring collides in none, at 36. This is only a sanity bound -- if a re-clean ever pushed the
# requirement past it, that is a finding, not something to silently truncate into.
ANON_SLOTS_MAX = 128

# --- Representative training subset (training/subset.py) ---
# Every head trains on a compact, representative slice of the corpus rather than all of it. The slice
# is selected by a per-season sample RATE that is heavy on the modern game and decays gently for older
# seasons.
#
# 3.2 moved the FOUR remaining full-corpus heads (event_time, player, substitution, sub_decision) onto
# the subset too, so all twelve are on it. Two reasons, and one stated risk.
#
# Consistency and compute: the four big heads drop from 26,267 games to roughly 5,200, taking their
# epochs from ~420 s to ~84 s, which roughly pays for 3.2's two new architecture layers.
#
# THE RISK, stated rather than discovered later: thinning every player embedding lands on the
# identity axis, which docs/v3_2_direction.md calls the larger of the programme's two failures. Three
# things make it acceptable rather than reckless. The newest season enters the subset at rate 1.0, so
# players who appear in the scored holdout keep most of their recent games -- what thins is older-era
# players who rarely appear in what is scored. The per-player prior scalars enter the set encoder
# additively BEFORE attention, so a player whose embedding is thin still arrives carrying his
# production profile. And MIN_PLAYER_SUBSET_GAMES removes the players who would have had the thinnest
# embeddings outright, rather than leaving them in the table under-trained. If the rookie /
# role-shifter numbers come back WORSE rather than better, this is the first suspect.
#
# Coverage-completeness is retired with it -- see training/subset.build_subset for why a
# minimum-games floor makes that guarantee inert.
#
# Per-season sample rate for the most recent seasons, NEWEST FIRST. Raised for v1.1
# ((0.70, 0.40, 0.25) -> (1.0, 0.70, 0.50)): the subset heads (shot_result especially) under-fit
# the modern game's efficiency (see RECENCY_HALFLIFE_SEASONS note), and shot_result early-stopped
# at epoch 7 on the old 3,239-game subset. The newest season is itself already truncated at
# FINAL_SEASON_FRACTION (we cut partway through it), so 100% of that is a modest absolute count.
SUBSET_RECENT_SEASON_RATES = (1.0, 0.70, 0.50)
# Seasons older than the recent block decay from the last recent rate, halving every this-many
# seasons -- a gentle exponential tail (tightened 8 -> 5 for v1.1, same rationale).
SUBSET_RECENCY_HALFLIFE_SEASONS = 5.0
SUBSET_SEED = 42                     # deterministic subset selection
SUBSET_GAMES_PATH = "./training/subset_games.json"  # persisted subset manifest (one extract step)
# Heads trained on the subset. As of 3.2 that is ALL TWELVE, so models.pipeline.run_stage's
# all-or-nothing guard over the seven conditional heads (they share one cond_*.npz, so a partial
# listing would make the stored tensors and the graphs disagree) is satisfied by definition.
SUBSET_MODEL_KEYS = (
    "event_time", "event_time_cond", "player",
    "shot_type", "shot_result", "assist_type", "turnover_type", "foul_type", "rebound_type",
    "timeout_team",
    "substitution", "sub_decision",
)