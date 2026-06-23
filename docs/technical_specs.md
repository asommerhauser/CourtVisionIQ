# CourtVisionIQ — Technical Specification

> *"Learn the rhythm and structure of basketball games first, then layer detail on top."*

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy — Simulation Thesis](#design-philosophy--simulation-thesis)
3. [Data](#data)
4. [Data Cleaning](#data-cleaning)
5. [Preprocessing](#preprocessing)
6. [Models](#models)
7. [Model Training](#model-training)
8. [Evaluation](#evaluation)
9. [Evaluation Strategy (Simulation)](#evaluation-strategy-simulation)
10. [System Integration](#system-integration)
11. [TBD / Open Questions](#tbd--open-questions)

---

## Overview

CourtVisionIQ is a multi-model basketball **simulation** system. It does not score a game from pre-game statistics; it **generates** a game one event at a time and reads predictive signal from the *distribution* of many simulated games.

The system is structured as an **ensemble of models**:

- The **Event/Time Transformer** generates the game skeleton (what happens next, and when)
- **Downstream models** (e.g., Shot Generator, future Player/Points heads) fill in granular details
- A **Controller** enforces basketball rules so every generated step is a legal game state

At inference the models are **sampled** (not argmaxed), so each run produces a different but plausible game. A matchup is **Monte-Carlo simulated** (run ~100×) and the outcome distribution — final score, pace, box lines, win probability — is the product. See [Design Philosophy](#design-philosophy--simulation-thesis).

### Pipeline

```
Raw CSV → Cleaning → Encoding → Model Training → Reporting
                                      ↓
                        Sampling + Controller → Monte-Carlo Simulation → Outcome Distribution
```

---

## Design Philosophy — Simulation Thesis

> **Core hypothesis: games predict themselves.** You cannot reliably predict what happens in a basketball game from season averages alone. Outcomes are **path-dependent** — a player gets rattled or gets hot, a run swings the building — and that story only exists *inside the unfolding sequence of plays*, not in the box score before tip-off.

### Momentum is the load-bearing assumption

The system is built on the premise that **momentum is real**: recent events shift the probabilities of the next ones. This is *why* prior plays are fed back into the model during generation — the play sequence carries the momentum state, the model conditions on it, and in simulation it consumes its own output, so a run the model starts can perpetuate itself the way a real run does.

**The architecture *is* the hypothesis.** A causal transformer that attends over prior plays is, mechanically, a momentum instrument: self-attention weights recent context, and autoregressive feedback lets that context compound. The model does **not assume** a fixed momentum effect — it learns whatever autocorrelation actually exists in the data and lets the Monte-Carlo aggregate *measure* it. (Momentum/"hot-hand" is an empirically contested claim; this design treats it as testable, not given.)

### Generation by sampling, not classification

- The model's value is its **conditional distribution** `p(next event, Δt | history)`, not its single top-1 guess.
- Inference **samples** from that distribution (temperature / top-p), so variance is a *feature*: it is what lets 100 rollouts tell 100 different believable stories.
- A diagnostic that matters more than loss: does the predicted distribution actually **move with context** — e.g. is `p(event | team on a 10-2 run)` meaningfully different from `p(event | flat game)`? If it collapses to base rates, the momentum signal is not being captured.

### Constraints keep rollouts on the manifold

A long (~600-step) self-fed rollout risks **compounding drift** (exposure bias): small per-step biases snowball into unrealistic games. The hard-coded Controller clamps each step back to a legal basketball state (clock, score, possession, fouls, substitutions, quarter structure). It is therefore both a **realism layer** and **drift insurance**.

### What "good" means here

Success is **not** top-1 next-event accuracy. It is **calibration** (do predicted probabilities match observed frequencies?) and **aggregate distributional fidelity** (do the statistics of simulated games match real ones?). The headline test: *simulate a held-out game 100× and check whether the actual outcome falls sensibly inside the simulated distribution, and whether win-probability calls are calibrated across many games.* See [Evaluation Strategy](#evaluation-strategy-simulation).

---

## Data

### Data Source

- **Raw** play-by-play master files: `./RawData/MasterFiles/` (e.g. the 2002-03 season,
  `[10-29-2002]-[06-15-2003]-combined-stats.csv`).
- **Cleaned, model-ready** season files: `./data/season<YYYY>.csv` (e.g. `data/season2003.csv`).
  The preprocessor consumes **every** `*.csv` in `./data` that has `game_id` + roster columns;
  anything parked in `data/_excluded/` is ignored.

> **Current scale:** training is presently on a **single season (2002-03)** — 1,277 games total.
> Multi-season ingestion is future work (see [TBD](#tbd--open-questions)) and is the prerequisite
> for scaling the model up.

### Data Structure (Per Event)

Each row represents a single game event:

| Field | Description |
|---|---|
| `game_id` | Game the event belongs to (sequences are grouped by this) |
| `roster_home` | 5 players — home lineup on court |
| `roster_away` | 5 players — away lineup on court |
| `time` | Game clock time (seconds) |
| `event` | Event type (see [event classes](#eventtime-model-core)) |
| `player` | Primary player involved |
| `secondary_player` | Secondary player (e.g. assister, blocker); shares the player vocab |
| `type` | Subtype of event |
| `result` | Outcome |

Prior-play history is **not** a stored column — it is supplied implicitly: each game is one
chronologically-ordered sequence and the causal transformer attends over all prior timesteps.

### Sequence Structure

- Data is grouped into **full game sequences** by `game_id`
- Each sequence is **chronologically ordered**

---

## Data Cleaning

### Pipeline

```
RawData/MasterFiles/*.csv → clean → data/season<YYYY>.csv
```

### Steps

1. Import raw master play-by-play CSV(s) from `./RawData/MasterFiles/`.
2. Normalize/clean into model-ready rows with `game_id`, `roster_home`, `roster_away`,
   and the event fields above.
3. Write cleaned per-season CSV(s) to `./data/` (triggered by `--clean` in `main.py`).

### Notes

Cleaned data must be:
- Chronologically ordered
- Grouped by `game_id` (one game = one sequence)

Encoding (integer tokenization) happens in **preprocessing**, not cleaning — the cleaned
files stay human-readable CSV.

---

## Preprocessing

### Vocabulary Construction

Shared, frozen "language" vocabularies are extracted for:

- `event`, `player`, `type`, `result`, `season`

Stored at:

```
encoder/vocabs/*.json
```

Each vocab reserves `PAD=0`, `UNK=1`. `player` is shared by both `player` and
`secondary_player` (same entity → same embedding).

### Encoding

- All categorical values → integer tokens (used by embedding layers).
- Rosters are encoded as 5 player-id slots (PAD-filled), order-agnostic.

### Feature Normalization

Stats are computed on the **train split only** (no test leakage) and persisted to
`encoder/vocabs/norm_stats.json` / `artifacts/event_time/norm_stats.json`:

```
time_abs   = time / max_time                       # max_time ≈ 3480 (58 min)
delta_time = (Δt - delta_mean) / delta_std         # delta_mean ≈ 5.98s, delta_std ≈ 7.35s
```

Δt is the per-game gap between consecutive events (`groupby(game_id).time.diff()`, first
event → 0, backwards clips → 0).

### Sequence Construction

- One game = one sequence, framed by `start` / `end` event tokens.
- Right-padded to a fixed length (`MAX_SEQUENCE_LENGTH`, `config.py`):

```
sequence_length = 600     # covers OT/overflow; truncate beyond
```

### Input Features (Per Timestep)

| Feature | Notes |
|---|---|
| `event` | Categorical (embedding) |
| `player` | Categorical (embedding) |
| `secondary_player` | Categorical; **weight-tied** to `player` |
| `type` | Categorical (embedding) |
| `result` | Categorical (embedding) |
| `season` | Categorical (embedding) — game context |
| `home_roster` | 5-player lineup → Set-Transformer vector |
| `away_roster` | 5-player lineup → Set-Transformer vector |
| `time_abs` | Normalized absolute clock |
| `delta_time` | Normalized Δt since previous event |
| `pad_mask` | 1 = real step, 0 = padding (for attention masking) |

> Note: earlier drafts listed `home_flag` and `playoff` inputs — these are **not** in the
> current model. Home/away is carried by the separate `home_roster` / `away_roster` inputs;
> only `season` is currently used as explicit game context.

---

## Models

### Event/Time Model (Core)

**Purpose:** Predict the next event and time until that event.

```
p(e_{t+1}, Δt_{t+1} | history)
```

**Architecture:**

- Causal Transformer encoder, learned positional embeddings
- Set Transformer for roster encoding (shared/weight-tied across home & away)
- Concrete dimensions (current config):

| Hyperparameter | Value |
|---|---|
| `model_dim` | 256 |
| `num_layers` | 4 |
| `num_heads` | 8 |
| `ff_dim` | 1024 |
| `dropout` | 0.2 |
| `sequence_length` | 600 |
| embed dims | event 32, player 128, type 32, result 16, season 16 |
| `roster_dim` | 128 |
| **Total params** | **≈ 4.06M** |

> **On model size:** ~4M parameters is **correct** for this configuration, not under-built.
> A transformer's size scales as ≈ `num_layers × 12 × model_dim²`; with `model_dim=256` and
> 4 layers the blocks dominate at ~3.15M, and the small (~500-token) vocabularies keep the
> embedding tables tiny. Reaching the hundreds-of-millions/billions seen in LLMs would require
> a much larger `model_dim`, more layers, and a far larger vocabulary — and would overfit a
> single season. Capacity scales *with* data (see [Future Work / TBD](#tbd--open-questions)).

**Inputs:** the per-timestep features in [Input Features](#input-features-per-timestep)
(categorical embeddings + two roster vectors + `time_abs`/`delta_time`), concatenated, projected
to `model_dim`, layer-normed, plus learned positional embeddings.

**Outputs:**

| Output | Type |
|---|---|
| `event_output` | Classification over the event vocab (logits; softmax via `from_logits` loss) |
| `time_output` | Regression — next-step normalized `delta_time` (scalar) |

**Event Vocab (12 tokens):**

- Control: `PAD`, `UNK`, `start`, `end`
- Classes: `none`, `shot`, `rebound`, `assist`, `turnover`, `block`, `foul`, `substitution`

---

### Shot Generator Model

- Separate model trained **after** the Event/Time Model
- Predicts shot outcomes and details
- Operates as a downstream detail layer

---

### Transformer Implementation

**Layer flow:**

1. Per-field **embedding** layers (categoricals) + **Set-Transformer** roster vectors + Dense
   projections of `time_abs` / `delta_time`.
2. **Concatenate → Dense(`model_dim`) → LayerNorm**, add learned positional embeddings, dropout.
3. **N causal transformer blocks** (pre-LN: MHA with causal + key-padding mask → residual;
   GELU feed-forward → residual). **No global pooling** — the model is autoregressive and
   emits a prediction at **every** timestep.
4. Two float32 output heads: `event_output` (logits) and `time_output` (scalar).

---

### Set Transformer (Roster Encoding)

- Encodes 5-player lineups
- **Permutation invariant** — order of players doesn't matter
- Based on attention over sets

---

### Controller (Post-Model)

Hard-coded constraint layer that forces each **sampled** step into a legal basketball state.
It is both the realism layer and the **drift insurance** for long autoregressive rollouts
(see [Design Philosophy](#design-philosophy--simulation-thesis)).

- Event sequencing logic (e.g. assists/blocks follow shots; foul logic)
- Bookkeeping the model must not violate: clock, score, possession, fouls, substitutions,
  quarter/OT structure
- Keeps each rollout on the manifold of real games so per-step biases can't snowball

---

## Model Training

### Training Paradigm

**Autoregressive (GPT-style)** — predict the next timestep given all prior timesteps.

### Loss Function

```
L = λ_event · CE + λ_time · MAE
```

| Term | Value / Description |
|---|---|
| `λ_event · CE` | Sparse categorical cross-entropy (`from_logits`) on the event head, weight **1.0** |
| `λ_time · MAE` | Mean absolute error on the normalized `delta_time`, weight **0.25** |

Both terms are **masked** by a per-step `loss_mask` (applied as sample weights), so PAD steps
and the final step (no next-step target) contribute zero loss.

### Optimization

- Optimizer: **AdamW** (`weight_decay=1e-4`, `clipnorm=1.0`)
- **LR schedule:** linear **warmup → cosine decay** (peak = `lr`, floor = `lr_alpha · lr`),
  replacing the earlier `ReduceLROnPlateau` (which collapsed the LR at the first plateau and
  stalled learning)
- **Regularization:** `dropout` (default 0.2) is the single tunable knob — it routes through the
  embedding, attention, feed-forward, **and** roster-encoder layers
- **Mixed precision** (`mixed_float16`) on GPU; output heads forced to float32

### Training Setup

- Train/test split: **80% / 20% by game** (split before computing norm stats → no leakage)
- Early stopping (`patience=10`, restores best weights)
- Standardized **reporting layer** emits a per-run HTML report + queryable Parquet
  (`run.parquet`, `epochs.parquet`) capturing all hyperparameters, the
  **trainable-parameter count**, per-epoch curves, and final test metrics (`reporting/`)

### Sequence Handling

- Right-padding to `sequence_length`
- Causal mask + key-padding mask applied during attention

---

## Evaluation

> These are **training-time** metrics for fitting the next-step distribution. They are *not*
> the measure of system quality — see [Evaluation Strategy (Simulation)](#evaluation-strategy-simulation).

### Metrics (per-epoch, train + val)

| Prediction Task | Metric |
|---|---|
| Event prediction | Cross-entropy loss, accuracy |
| Time prediction | MAE (normalized) **and `mae_sec`** (denormalized, in real seconds = normalized MAE × `delta_std`) |

`mae_sec` makes the time head's error human-readable — you can watch the prediction converge
toward the true Δt in seconds rather than in unitless normalized space.

### Tracking & Visualization

Captured automatically by the **reporting layer** (`reporting/`): per-epoch loss/metric curves,
epoch durations, learning rate, final held-out test metrics, and a model-size table (incl.
trainable params) — rendered to a self-contained HTML report and written as queryable Parquet.

---

## Evaluation Strategy (Simulation)

The real test of the system is **not** how well it predicts the next event, but how well its
**simulated games** match reality in aggregate. Training metrics and simulation evaluation are
deliberately separate.

### What we measure

| Dimension | How |
|---|---|
| **Calibration** | Reliability curves / ECE on the event distribution — when the model says 12%, does it happen ~12%? |
| **Context sensitivity** | Does `p(next | recent run)` shift vs. `p(next | flat game)`? (Is momentum actually captured, or does it collapse to base rates?) |
| **Distributional fidelity** | Simulate held-out games and compare distributions of final score, pace/possession count, and box-score lines against the real outcomes |
| **Timing realism** | `mae_sec` + simulated game length / possession counts land in believable ranges |
| **Predictive power** | Win-probability **calibration** across many games |

### The headline test

> **Simulate a held-out game ~100×** (sampling + Controller), then check whether the *actual*
> outcome falls sensibly inside the simulated outcome distribution — repeated across many games
> to assess win-probability calibration.

### Risks this guards against

- **Compounding drift / exposure bias** — a 600-step self-fed rollout wandering off-manifold
  (mitigated by the Controller; validated by checking aggregate sim stats stay stable across the
  *full* game, not just early).
- **Base-rate collapse** — a model with fine average loss that ignores momentum would produce
  statistically plausible but lifeless games; the context-sensitivity check catches this.
- **Noise floor** — next-event prediction has irreducible entropy; gains are bounded, which is
  exactly why the distribution (not point accuracy) is the product.

---

## System Integration

### Generation Loop

```
1. Input initial game state
2. Predict p(next event, Δt | history)
3. SAMPLE an event + Δt from that distribution (temperature / top-p)
4. Controller validates / corrects to a legal state; append to sequence
5. Repeat until game end (feeding the model its own output)
```

### Monte-Carlo Simulation

```
Run the generation loop N times (~100) for a matchup
        ↓
Aggregate outcomes → distribution over final score, pace, box lines, win probability
```

### Ensemble Flow

```
Event/Time Model  →  Game skeleton (events + timing)
        ↓
Shot / Player / Foul Models  →  Granular details
        ↓
Controller  →  Rule enforcement & validation
```

---

## TBD / Open Questions

| Area | Open Question |
|---|---|
| **Data** | Multi-season ingestion pipeline (currently one season, 2002-03) |
| **Cleaning** | Handling of missing / null values |
| **Models** | Capacity scaling once more data is available (scale model *with* data) |
| **Models** | Player / Points / Shot detail-head specs |
| **Inference** | Sampling-rollout + Monte-Carlo harness (not yet built) |
| **Inference** | Rotation/minutes model — predict player stints & minutes directly instead of event-head-driven subs (fixes starter selection + realistic minutes) |
| **Evaluation** | Calibration + distributional-fidelity eval implementation |

*Resolved since the original draft:* transformer layers = 4; embed dims = event 32 / player 128
/ type 32 / result 16 / season 16; sequence length = 600; batch size 32–64; epochs ≤ 50 with
early stopping; split = 80/20 **by game**.

---

## Next Steps

- [ ] Build the sampling + Controller **rollout** loop and the **Monte-Carlo** harness
- [ ] Implement **calibration** + **distributional-fidelity** evaluation (the real scoreboard)
- [ ] Multi-season data ingestion → then scale model capacity alongside it
- [ ] Define downstream detail-head specs (Player / Shot / Points)

---

*CourtVisionIQ — simulate the NBA from scratch.*