# CourtVisionIQ: Generative Simulation of NBA Games as a Test of the Momentum Hypothesis

> **⚠️ ROUGH DRAFT** — working methodology write-up, current as of the 2002-03 single-season
> baseline. Numbers and scope will change; sections marked *(planned)* are not yet implemented.

---

## Abstract

CourtVisionIQ is a generative simulator for NBA play-by-play. Rather than predicting a game's
outcome from pre-game statistics, it **generates** a game one event at a time with a causal
transformer, **samples** each step from the model's predicted distribution, constrains every step
to a legal basketball state, and **Monte-Carlo simulates** a matchup (~100 runs) to produce a
*distribution* of outcomes — final score, pace, box-score lines, and win probability. The project
is organized around a single hypothesis: **games predict themselves.** Outcomes are
path-dependent and momentum-driven, so predictive signal lives in the unfolding sequence of plays,
not in season averages. The autoregressive transformer is deliberately chosen as an *instrument*
for this hypothesis: self-attention over prior plays plus output-feedback is mechanically a
momentum model, and the simulation aggregate *measures* how much momentum the data actually
contains. Consequently, system quality is judged by **calibration and distributional fidelity**,
not by top-1 next-event accuracy.

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
Detail heads (planned: Shot / Player / Points)  →  granular outcomes
        ↓
Controller (hard constraints)  →  legal game state every step
        ↓
Sampling + Monte-Carlo  →  distribution of game outcomes
```

- **Event/Time Transformer** — the spine; generates the event sequence and its timing.
- **Detail heads** *(planned)* — fill in which player, shot location, points, etc., conditioned on
  the unfolding sequence.
- **Controller** — deterministic bookkeeping/rules layer; both realism and drift insurance.

---

## 3. Data

| | |
|---|---|
| **Current scope** | One season, 2002-03 |
| **Games** | 1,277 total → **1,022 train / 255 test** (80/20 by game) |
| **Raw** | `RawData/MasterFiles/` |
| **Cleaned** | `data/season<YYYY>.csv` (grouped by `game_id`, one game = one sequence) |

Each event row carries `game_id`, `roster_home`/`roster_away` (5 on-court players each), `time`,
`event`, `player`, `secondary_player`, `type`, `result`. Sequences are chronologically ordered and
framed by `start`/`end` tokens. **Multi-season ingestion is future work** and is the prerequisite
for scaling model capacity (§10).

---

## 4. Methods

### 4.1 Encoding & normalization

Shared, frozen integer vocabularies for `event`, `player`, `type`, `result`, `season`
(`PAD=0`, `UNK=1`); `secondary_player` shares the `player` table (same entity → same embedding).
Time features are normalized using **train-split-only** statistics (no leakage):

```
time_abs   = time / max_time                  # max_time ≈ 3480 s
delta_time = (Δt − delta_mean) / delta_std    # delta_mean ≈ 5.98 s, delta_std ≈ 7.35 s
```

where Δt is the per-game gap between consecutive events.

### 4.2 Event/Time Transformer

A causal transformer encoder that emits a prediction at **every** timestep (no pooling).

| Hyperparameter | Value |
|---|---|
| `model_dim` | 256 |
| layers / heads | 4 / 8 |
| `ff_dim` | 1024 |
| dropout | 0.2 |
| sequence length | 600 |
| embed dims | event 32, player 128, type 32, result 16, season 16 |
| roster_dim | 128 |
| **Total params** | **≈ 4.06M** |

**On the parameter count.** ~4M is the *correct* size for this configuration, not a defect.
Transformer size scales as ≈ `layers × 12 × model_dim²`; here the 4 blocks account for ~3.15M, and
the small (~500-token) vocabularies keep embedding tables tiny (~0.1M). Approximate breakdown:

| Component | Params |
|---|---|
| 4 transformer blocks | ~3.15M |
| Roster set-encoder | ~0.40M |
| Positional embedding (600×256) | ~0.15M |
| Fusion projection | ~0.16M |
| Token embeddings | ~0.11M |
| Output heads | ~0.01M |

Billion-parameter language models are large because of `model_dim` (2k–12k), depth (24–96), and
huge vocabularies (50k–150k) — none of which apply to a ~500-token basketball vocabulary on a
single season. **Capacity must scale with data**, so a much larger model is gated on multi-season
ingestion, not adopted speculatively.

### 4.3 Roster encoding (Set Transformer)

Each 5-player lineup is encoded to a fixed vector by a **permutation-invariant** Set Transformer
(SAB ×2 + pooling-by-attention), shared/weight-tied across home and away. Order of players carries
no meaning, so the encoder is invariant to it by construction.

### 4.4 Training

- **Loss:** `L = 1.0 · CE(event) + 0.25 · MAE(Δt)`, both masked so PAD / no-next-step positions
  contribute zero (sparse CE `from_logits`; MAE on normalized Δt).
- **Optimizer:** AdamW (`weight_decay=1e-4`, `clipnorm=1.0`).
- **LR schedule:** linear warmup → cosine decay (floor = `lr_alpha · lr`), replacing an earlier
  `ReduceLROnPlateau` that collapsed the LR at the first plateau and stalled learning.
- **Regularization:** dropout (0.2) as a single knob across embedding/attention/FF/roster layers.
- **Early stopping:** `patience=10`, best weights restored. Mixed precision on GPU.
- **Reporting:** every run emits an HTML report + queryable Parquet (`run.parquet`,
  `epochs.parquet`) capturing all hyperparameters, the trainable-parameter count, per-epoch curves,
  and final test metrics.

---

## 5. Current Results (baseline)

Run `20260614-230954-782760` — 2002-03 season, 50 epochs (no early stop), ~20 min on one GPU
(TF 2.20 / Keras 3.12):

| Metric | Value |
|---|---|
| Best val loss (epoch 47) | **0.749** |
| Event accuracy (val) | **~55%** (12-class) |
| Time MAE | **≈ 3.1 s** (`mae_sec`; ≈ 0.42 normalized × 7.35) |
| Parameters | 4,057,747 (≈4.06M) |

**Diagnosis — plateau, not overfitting.** Train and validation loss track each other almost
exactly (final gap ≈ +0.007) and both flatten by epoch ~5–10. That is an *optimization/representation
plateau*, not a generalization gap. Two implications:

1. More regularization alone would not help (there is no gap to close) — addressed instead with the
   warmup→cosine LR schedule (the prior schedule decayed the LR away mid-plateau).
2. The 4M model is **not** capacity-starved for one season; a much larger model would memorize.

Crucially, **0.749 / 55% is not the project's scoreboard** — see §6.

---

## 6. Evaluation Strategy

Training metrics fit the next-step distribution; they do not measure simulation quality. The two
are kept separate.

| Dimension | Method |
|---|---|
| **Calibration** | Reliability curves / ECE on the event distribution |
| **Context sensitivity** | Does `p(next | on a run)` differ from `p(next | flat game)`? (Is momentum captured, or does it collapse to base rates?) |
| **Distributional fidelity** | Simulated vs. actual distributions of final score, pace/possessions, box lines |
| **Timing realism** | `mae_sec`; simulated game length & possession counts in believable ranges |
| **Predictive power** | Win-probability **calibration** across many games |

**Headline test:** simulate a held-out game ~100× (sampling + Controller) and check whether the
*actual* outcome falls sensibly inside the simulated distribution — repeated across games to assess
win-probability calibration. *(Harness planned, §10.)*

---

## 7. Discussion

**Why accuracy is the wrong scoreboard.** For simulation you do not want the single most-likely
event; you want to sample from the predicted distribution. A well-*calibrated* 55% model produces
better simulations than a miscalibrated 65% one. Variance is the mechanism, not the enemy — it is
what lets 100 rollouts tell 100 plausible stories.

**Momentum as a measurable property.** Because the architecture conditions on prior plays, momentum
(if present and learnable) shows up as the conditional distribution shifting with context. This is
directly testable (§6) and converts "is momentum real?" from a debate into a measurement on this
dataset.

---

## 8. Limitations & Risks

- **Data scale.** One season caps both the achievable model size and the diversity of situations.
- **Exposure bias / compounding drift.** A 600-step self-fed rollout can wander off the manifold of
  real games; the Controller mitigates this, and aggregate sim stats must be validated across the
  *full* game, not just early steps.
- **Noise floor.** Next-event prediction has irreducible entropy; point-accuracy gains are bounded —
  which is exactly why the *distribution* is the product.
- **Momentum effect size is empirical.** The model captures only as much momentum as the data
  contains; the aggregate will reveal whether that is large or modest.

---

## 9. Reproducibility

GPU training runs on WSL2 + CUDA (TF has no native-Windows GPU path):

```bash
wsl -d Ubuntu-22.04
source ~/cviq-venv/bin/activate
cd /mnt/c/Projects/CourtVisionIQ
python main.py --rebuild-vocabs --epochs 50 --batch-size 64 --train
```

Every run is captured by the reporting layer under `reports/event_time/<run_id>/`
(HTML + Parquet). See `README.md` and `docs/gpu_wsl2_setup.md`.

---

## 10. Future Work

1. **Sampling + Monte-Carlo harness** — the generation rollout (temperature/top-p) wrapped by the
   Controller, plus aggregation over ~100 runs. *Highest leverage; this is what turns the model into
   a simulator.*
2. **Calibration + distributional-fidelity evaluation** — implement the §6 scoreboard.
3. **Scale data and model together** — multi-season ingestion, then a proportionally larger
   `model_dim`/depth/vocabulary (scaling-laws rationale): data and parameters move together or the
   model either underfits or memorizes.
4. **Detail heads** — Player / Shot / Points models conditioned on the generated skeleton.
5. **Rotation / minutes model** — a dedicated head that predicts each player's stints and
   on-court minutes (who is on the floor over time), replacing the current event-head-driven
   substitutions. Today the event head samples `substitution` events and a blunt forced-cadence
   safety net keeps a team from playing five men all game, with no notion of starters or target
   minutes — so a real starter can fail to start or log far too few minutes. A rotation model
   that predicts lineups/minutes directly would fix both starter selection and realistic minute
   distributions.

---

## References (informal)

- Gilovich, T., Vallone, R., & Tversky, A. (1985). *The hot hand in basketball: On the
  misperception of random sequences.* — the original "hot hand is a myth" study.
- Miller, J. B., & Sanjurjo, A. (2018). *Surprised by the Hot Hand Fallacy?* — identifies a
  selection bias in the 1985 analysis; finds a real (modest) hot-hand effect.

---

*CourtVisionIQ — simulate the NBA from scratch.*
