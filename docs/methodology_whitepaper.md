# CourtVisionIQ: Generative Simulation of NBA Games as a Test of the Momentum Hypothesis

> **Working draft** — current as of 2026-08-29, describing model `v1.0` (21 seasons, eleven heads)
> and the `full4-s100` holdout evaluation. Numbers move with each retrain and dial pass; the
> results in §5 name the run they come from.

---

## Abstract

CourtVisionIQ is a generative simulator for NBA play-by-play. Rather than predicting a game's
outcome from pre-game statistics, it **generates** a game one event at a time with a stack of
causal transformers, **samples** each step from the models' predicted distributions, constrains
every step to a legal basketball state, and **Monte-Carlo simulates** a matchup (21–100 runs) to
produce a *distribution* of outcomes — final score, pace, box-score lines, and win probability. The
project is organized around a single hypothesis: **games predict themselves.** Outcomes are
path-dependent and momentum-driven, so predictive signal lives in the unfolding sequence of plays,
not in season averages. The autoregressive transformer is deliberately chosen as an *instrument*
for this hypothesis: self-attention over prior plays plus output-feedback is mechanically a
momentum model, and the simulation aggregate *measures* how much momentum the data actually
contains. Consequently, system quality is judged by **calibration and distributional fidelity**,
not by top-1 next-event accuracy. On a 100-game held-out slice of 2022-23 the simulator picks
winners 63% of the time at a Brier score of 0.217, and its simulated team box scores land within
about half a point of real scoring and half a possession of real pace.

---

## 1. Motivation & Hypothesis

Conventional sports prediction maps pre-game features (team ratings, rest, injuries) to an outcome.
This implicitly assumes the game is a deterministic-ish function of prior state. CourtVisionIQ
rejects that framing for basketball:

- **Path dependence.** A player gets rattled or gets hot; a defensive stretch swings the building.
  These are properties of the *trajectory*, not the starting conditions.
- **Momentum as the load-bearing assumption.** Recent events shift the probabilities of the next
  ones. This is a *testable* claim, not an assumed constant — the "hot hand" has been debated for
  decades (see [References](#references-informal)).

The design commitment that follows: **model the conditional next-step distribution
`p(event_{t+1}, Δt_{t+1} | history)`** and generate forward from it. If momentum is real and
learnable, a sequence model that conditions on prior plays will capture it; if it is weak, long
rollouts will regress toward base rates. Either way we get a *measurement*.

---

## 2. System Overview

An ensemble with a clear division of labor:

```
Event/Time Transformer  →  game skeleton (what happens next, and when)
        ↓
Actor head              →  who does it (masked to the on-court ten)
        ↓
Conditional-time head   →  re-times the step once event + actor are decided
        ↓
Six detail heads        →  shot type / shot result / assist / turnover / foul / rebound type
        ↓
Rotation heads          →  substitution (who checks in) + stint length (for how long)
        ↓
Controller              →  hard rules: clock, score, possession, fouls, bonus, foul-outs
        ↓
Sampling + Monte-Carlo  →  distribution of game outcomes
```

Eleven trained heads in total, ~140M parameters combined, all sharing one frozen token language
and one set-transformer roster encoder design. Each head is a full causal transformer over the
whole game history; they differ in what they predict and what decided values they condition on.

---

## 3. Data

| | |
|---|---|
| **Scope** | 21 seasons, 2002-03 through 2022-23 |
| **Games** | ≈26,400 → **21,014 train / 5,253 validation** (80/20 by game) + a **100-game sequential holdout** |
| **Players** | 2,153 |
| **Raw** | `RawData/MasterFiles/` |
| **Cleaned** | `data/season<YYYY>.csv` (grouped by `game_id`, one game = one sequence) |

Each event row carries `game_id`, `roster_home`/`roster_away` (5 on-court players each), `time`,
`event`, `player`, `secondary_player`, `type`, `result`, `season`. Sequences are chronologically
ordered and framed by `start`/`end` tokens.

**The holdout is sequential, not random.** Training stops halfway through the most recent season
and the next 100 real games are held out — never trained on, never used for early stopping. This is
the honest analogue of the deployment case (predict games that have not happened yet), and it is
strictly harder than a random split, which would let the model see later games from the same season.

**Older seasons are down-weighted, not dropped.** Every game trains, but its loss weight halves
every 3 seasons of age, floored at 0.05. The modern game therefore dominates the gradient while
older-player embeddings still receive some. A 21-season corpus at a uniform weight visibly
compressed shooting efficiency toward a cross-era average — the 3-point revolution makes an
unweighted corpus actively misleading about the current game.

---

## 4. Methods

### 4.1 Encoding & normalization

Shared, frozen integer vocabularies for `event` (12), `type` (24), `result` (19), `season` (23) and
`player` (2,153), each reserving `PAD=0`, `UNK=1`; `secondary_player` shares the `player` table
(same entity → same embedding). Time features are normalized using **train-split-only** statistics:

```
time_abs   = time / max_time                  # max_time = 4080 s
delta_time = (Δt − delta_mean) / delta_std    # delta_mean = 5.89 s, delta_std = 7.30 s
rest       = (clip(days_rest, 0, 30) − 2.45) / 2.29
```

Beyond the base features, each timestep carries **six season-context inputs** (per-player days rest
into the roster encoder; team games-played and days-rest scalars into the fusion) and **six
game-state inputs** (`score_diff`, `score_total`, `period_idx`, `period_time_left`, and both teams'
period foul counts). The game-state plumbing is built and tested but `v1.0`'s weights predate its
activation — the next full train is the first whose weights consume it.

### 4.2 Backbone

A causal transformer encoder that emits a prediction at **every** timestep (no pooling), shared by
all eleven heads:

| Hyperparameter | Value |
|---|---|
| `model_dim` | 384 |
| layers / heads | 6 / 8 |
| `ff_dim` | 1536 |
| dropout | 0.15 |
| sequence length | 600 |
| embed dims | event 32, player 192, type 32, result 16, season 16 |
| roster set-attention blocks | 3 |
| roster_dim | 128 |
| **Params per head** | **12.5–13.4M** (≈140M across the stack) |

**On the parameter count.** Transformer size scales as ≈ `layers × 12 × model_dim²`; the 6 blocks
dominate, and a ~2,200-token vocabulary keeps the embedding tables modest. Billion-parameter
language models are large because of `model_dim` (2k–12k), depth (24–96), and 50k–150k-token
vocabularies — none of which apply to basketball, where a few hundred things can happen. Capacity
scales *with* data: the backbone was 256/4/1024 on the original single-season baseline and was
raised to 384/6/1536 once the corpus reached 21 seasons.

### 4.3 Roster encoding (Set Transformer)

Each 5-player lineup is encoded to a fixed vector by a **permutation-invariant** Set Transformer
(3 Set-Attention blocks + pooling-by-attention), shared and weight-tied across home and away. Order
of players carries no meaning, so the encoder is invariant to it by construction. Per-player rest
rides alongside player identity into the encoder, so every head — including actor selection — sees
freshness.

### 4.4 Training

- **Loss:** per head — `1.0 · CE(event) + 0.5 · MAE(Δt)` for the Event/Time head, masked sparse CE
  for the categorical heads, masked MAE for the two regression heads. Every head trains with a
  per-row `sample_weight` mask, so PAD rows, no-next-step rows and rows the head does not own
  contribute zero.
- **Optimizer:** AdamW (`weight_decay=1e-4`, `clipnorm=1.0`), `lr = 3e-4`.
- **LR schedule:** linear warmup (2 epochs) → cosine decay to `0.05 · lr`.
- **Regularization:** dropout 0.15 across embedding/attention/FF/roster layers.
- **Early stopping:** `patience=15`, best weights restored, ≤50 epochs. Mixed precision on GPU.
- **Availability masking:** the player-vocab heads are masked to each game's available player set,
  so no probability mass can land on someone not dressed.
- **Representative subset:** the seven small conditional heads train on a coverage-complete,
  recency-weighted slice rather than all 21,014 games — they saturate long before the full corpus
  and start to overfit. The four heads over the player vocab keep everything.
- **Reporting:** every run emits an HTML report + queryable Parquet capturing all hyperparameters,
  the trainable-parameter count, per-epoch curves, and final test metrics.

### 4.5 Rollout

Generation runs ~5 forward passes per event over ~500–900 events per game. Three mechanisms make
that affordable without changing what is predicted: an **incremental input cache** (each history
row is encoded once, not once per head call), a **batched rollout** that runs 48 independent
game-sims concurrently and pools their per-event forward passes into one batched GPU call per head,
and a **process pool** that shards the holdout across cores and GPUs — because everything around
the forward pass is Python, one process is GIL-bound to roughly one core.

### 4.6 Inference dials

A set of rollout constants in `config.py`, read at call time, calibrate the sampled distribution
without retraining: a pace multiplier on predicted Δt, per-token logit offsets on the event mix and
the shot result, an actor-head temperature, a stint-length multiplier, a home-court shot bias, and
a dead-ball rebound rate. Each is fit against a specific diagnostic in the eval report, and every
report records the exact dial package that produced it.

**Dials are calibrations against a specific set of weights, not physics.** Carrying one across a
retrain unexamined caused the largest single regression in this project's history: a +6% pace
correction fit to train 1's under-predicting time head stayed in place through train 2's larger
conditional-time head, and pace collapsed 8.6 possessions the other way.

---

## 5. Results

### 5.1 Training metrics (model `v1.0`)

| Head | Best val loss | Final validation metric |
|---|---|---|
| `event_time` | 0.2229 | event accuracy **75.3%** (12-class), Δt MAE **2.37 s** |
| `player` | 0.4893 | actor accuracy **37.5%** (2,153-way, masked to the on-court ten) |
| `event_time_cond` | 0.0802 | Δt MAE **2.31 s** |
| `shot_type` | 0.0410 | **80.1%** |
| `shot_result` | 0.0725 | **76.6%** |
| `assist_type` | 0.0184 | **65.4%** |
| `turnover_type` | 0.0131 | **80.6%** |
| `foul_type` | 0.0257 | **58.7%** |
| `rebound_type` | 0.0338 | **73.9%** |
| `substitution` | 0.0514 | **47.7%** |
| `stint_length` | 0.0149 | stint MAE **187 s** |

**These are not the project's scoreboard** — see §6. They measure the fit of the next-step
distribution, which is a means, not the product.

### 5.2 Simulation results — `full4-s100`

100 held-out games from 2022-23, each simulated 100 times.

| Metric | Value | Reference |
|---|---|---|
| Win pick accuracy | **63%** | coin flip 50% |
| **Brier score** | **0.2173** | coin flip 0.250; ESPN pregame ≈0.219; inpredictable ≈0.216; betting market ≈0.19 |
| Log loss | 0.622 | |
| Point spread MAE | 9.49 | spread bias −0.42, correlation 0.47, 56% within 10 points |
| Team points | 115.41 predicted vs 115.93 actual | **bias −0.52**, MAE 9.26 |
| Pace | 101.15 vs 100.68 | bias +0.47 |
| eFG% | .538 vs .554 | bias −0.017 |
| Team minutes | 240.83 vs 241.00 | bias −0.17 |
| Player minutes | 23.26 vs 23.27 | **MAE 5.66 min** |
| Player points | 11.15 vs 11.20 | MAE 4.83 |

Win-probability calibration is close across the populated bins: predicted .329 → observed .333;
.507 → .488; .682 → .684. Only the sparse ≥0.8 bin (n=6) is off.

### 5.3 How it got here

| Run | Sims | Pick acc | Brier | Team pts bias | Pace bias |
|---|---|---|---|---|---|
| `trial1` | 28 | 61% | 0.2273 | **−11.55** | +0.04 |
| `full1` | 21 | 66% | 0.2176 | +0.43 | +0.50 |
| `full2` | 21 | 62% | 0.2250 | +0.62 | +0.95 |
| `full3` | 21 | 65% | 0.2178 | +0.66 | +0.85 |
| `full4-s100` | 100 | 63% | 0.2173 | −0.52 | +0.47 |

The dominant defect at `trial1` was that simulated teams scored ~11.5 points too few per game: the
shape of the game was right, but finishing rates ran low and the sim drew too few free throws.
Three rounds of dial fitting against the eval diagnostics — a make-rate correction on the shot
result head, an event-mix correction on fouls and turnovers, rebound-volume and home-court trims —
closed it without a retrain. Win-pick accuracy moves several points run to run at this sample size,
so the 66%/62%/65%/63% sequence is best read as one number near the mid-60s, not a trend.

### 5.4 The baseline that matters

A low per-player MAE means little on its own, because NBA box scores are heavily mean-reverting:
"predict each player's season-to-date average every game" is a strong predictor. Scored against it
at `full1`, the model **lost**: player points MAE 4.92 vs 4.56, about 8% worse, winning the paired
per-player-game comparison 46.8% of the time.

That is the single most important open number in the project, and it is moving in the right
direction — the same comparison in June was ~44% worse at a 35% win rate. `full4-s100` improved
player points MAE to 4.83 but has not yet been scored against the baseline.

Two things suggest where the remaining gap lives. First, per-player **minutes** MAE is 5.66 — a
player's minutes are the multiplier on every counting stat they produce, so a rotation error is a
box-score error everywhere at once. Second, the season-average baseline is *structurally* advantaged
on mean-reverting stats and cannot be beaten by matching means; it is beaten only by getting the
game-specific deviations right, which is exactly the momentum claim under test.

---

## 6. Evaluation Strategy

Training metrics fit the next-step distribution; they do not measure simulation quality. The two
are kept separate — `reports/` for training, `results/` for evaluation.

| Dimension | Method | Status |
|---|---|---|
| **Win prediction** | Pick accuracy, Brier, log loss, reliability bins — scored both by majority vote across sims and by the sign of the mean margin | Built |
| **Distributional fidelity** | Simulated vs actual team and per-player box lines, plus four factors and pace | Built |
| **Timing realism** | Δt MAE; simulated game length and possession counts | Built |
| **Rotation realism** | Player-minutes MAE, within-2/5/10-minute rates, signed-error distribution | Built |
| **Vs. the naive baseline** | Model vs season-to-date player averages, paired per player-game | Built |
| **Calibration of the event head** | Reliability curves / ECE on the next-event distribution | Not built |
| **Context sensitivity** | Does `p(next \| on a run)` differ from `p(next \| flat game)`? | Not built |

The last two are the direct tests of the momentum hypothesis, and they are the notable gap: the
current scoreboard measures whether simulated games *look* like real ones, not whether the model's
conditional distribution actually *moves with context*. A model that had collapsed to base rates
could still post plausible aggregate box scores. Distinguishing those two cases requires the
context-sensitivity probe.

---

## 7. Discussion

**Why accuracy is the wrong scoreboard.** For simulation you do not want the single most-likely
event; you want to sample from the predicted distribution. A well-*calibrated* 75% event head
produces better simulations than a miscalibrated 85% one. Variance is the mechanism, not the enemy
— it is what lets 100 rollouts tell 100 plausible stories.

**Every product is a query against one engine.** Because the simulator generates whole games rather
than a single number, matchup win probability, projected box scores and player evaluation from
simulated usage and efficiency are all reads off the same rollout. Roster is a live input, so
swapping a player for a league-average replacement and re-simulating reads wins-over-replacement
directly from the change in outcomes — an experiment rather than a regression coefficient. That is
the strongest claim the architecture supports and it is not available to a ratings model.

**What the dials mean, epistemically.** A dial is an admission that the trained distribution is
mis-calibrated in a specific, measured way, plus a correction that does not require six hours of
GPU time to test. They are honest as calibration and dishonest as physics: a dial fit on one set of
weights is not a fact about basketball, and every one of them is a candidate to be retired by
better training. The shot-zone expansion in train 3 exists precisely to attack the make-rate bias at
its source rather than through `SHOT_RESULT_BIAS`.

---

## 8. Limitations & Risks

- **Holdout size.** 100 games is directional, not settled. Win-pick accuracy swings several points
  between runs on the same weights. The reason it is not 1,000 is GPU hours, not capability — the
  same harness runs the larger holdout unchanged.
- **The naive baseline is not yet beaten** on per-player box MAE (§5.4).
- **No score/clock conditioning in `v1.0`.** The game-state inputs are plumbed but untrained, so
  end-of-game behavior — leads sat on, trailing teams fouling, garbage time — is not modeled. Close
  games are exactly where win prediction is decided.
- **No live inputs.** Injury reports and confirmed lineups land minutes before tip and are priced
  by the market; this model ingests none of them. That is a data integration, not a modeling gap,
  but it is a real part of the distance to a 0.19 Brier.
- **Exposure bias / compounding drift.** A ~600-step self-fed rollout can wander off the manifold
  of real games; the Controller mitigates this, and aggregate sim stats must be validated across
  the *full* game, not just early steps.
- **Momentum effect size is empirical.** The model captures only as much momentum as the data
  contains, and the probes that would isolate it (§6) are not built yet.

---

## 9. Reproducibility

TensorFlow dropped native-Windows GPU support after 2.10, so GPU work runs under WSL2 or on a
rented cloud GPU (RunPod; a single 24 GB consumer card is sufficient — the stack is
overhead-bound rather than compute-bound). Full setup, transfer and run procedure is in `README.md`.

```bash
# train every head under a new name
python train.py --full --name v1.1 --batch-size 64 --clean --rebuild-vocabs

# evaluate the frozen holdout, sharded across cores/GPUs
python evaluate.py --model v1.1 --run full1 --monte-carlo 100 --procs auto

# score it against the season-average baseline
python -m reporting.baseline_comparison results/v1.1/full1
```

Every run is captured by the reporting layer: training under `reports/<head>/<run_id>/`, evaluation
under `results/<model>/<run>/`, both as HTML + Parquet. A full train plus a full evaluation costs a
few dollars of rented GPU time.

---

## 10. Future Work

**Version 2** (decided change set in `docs/v2_planned_changes.md`; nothing built yet): activate the
game-state features; weight the loss toward close-and-late rows; expand `shot_type` from
`{2pt, 3pt}` to seven court zones derived from the raw shot coordinates already on disk, so
per-player-per-zone make rates are learned instead of dialed; add player age and coach (rolling
team style priors plus a coach embedding) for year-to-year generalization.

A second cluster of theories moves rules the simulator currently hard-codes into the data itself:
free-throw counts (the rollout awards three free throws roughly twelve times too often, because the
count is decided by a branch reading a head trained not to answer that question), the fouled player
(present in the raw data, dropped by the cleaner), and a loss mask that would stop the event head
from training on the ~22% of positions the rollout never visits.

**Beyond that:**

1. **A rotation/minutes model** — predict stints and on-court minutes directly, with seeded
   starters, instead of deriving them from sampled substitution events. Player minutes are the
   multiplier on every per-player stat and currently miss by 5.7 minutes.
2. **The momentum probes** — event-head calibration curves and the context-sensitivity test, which
   are the direct measurements the whole thesis is built to make (§6).
3. **A relative offense/defense encoding** instead of home/away, evaluable only with a full retrain.
4. **Tracking data** — turning shot location into something the model generates against the
   specific defense on the floor, rather than a token it looks up. Needs a license.
5. **Live inputs** — injuries and confirmed lineups.

---

## References (informal)

- Gilovich, T., Vallone, R., & Tversky, A. (1985). *The hot hand in basketball: On the
  misperception of random sequences.* — the original "hot hand is a myth" study.
- Miller, J. B., & Sanjurjo, A. (2018). *Surprised by the Hot Hand Fallacy?* — identifies a
  selection bias in the 1985 analysis; finds a real (modest) hot-hand effect.

---

*CourtVisionIQ — simulate the NBA from scratch.*
