# CourtVisionIQ — Technical Specification

> *"Learn the rhythm and structure of basketball games first, then layer detail on top."*

**Status: current as of 2026-08-29 (model `v1.0`, eval run `full4-s100`).** Everything described
here is built and running unless a line says otherwise; forward-looking work is confined to
[Open work](#open-work) and `docs/v2_theories.md`.

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy — Simulation Thesis](#design-philosophy--simulation-thesis)
3. [Data](#data)
4. [Data Cleaning](#data-cleaning)
5. [Preprocessing](#preprocessing)
6. [Models](#models)
7. [Model Training](#model-training)
8. [Rollout & Controller](#rollout--controller)
9. [Evaluation](#evaluation)
10. [Inference Dials](#inference-dials)
11. [Reporting](#reporting)
12. [Repository Map](#repository-map)
13. [Open work](#open-work)

---

## Overview

CourtVisionIQ is a multi-model basketball **simulation** system. It does not score a game from
pre-game statistics; it **generates** a game one event at a time and reads predictive signal from
the *distribution* of many simulated games.

The system is an **ensemble of eleven transformer heads** over one shared token language:

- The **Event/Time Transformer** generates the game skeleton (what happens next, and when).
- A **player/actor head** picks who does it, and a **conditional-time head** re-times the step
  once the event and actor are decided.
- Six **conditional type/result heads** fill in the detail (shot type, shot result, assist type,
  turnover type, foul type, rebound type).
- A **substitution head** and a **stint-length head** drive the rotation.
- A hard-coded **Controller** enforces basketball rules so every generated step is a legal game
  state, and owns everything the models do not see (clock, score, possession, team fouls, bonus,
  foul-outs, ejections).

At inference the heads are **sampled** (not argmaxed), so each run produces a different but
plausible game. A matchup is **Monte-Carlo simulated** (21–100 runs) and the outcome distribution
— final score, pace, box lines, win probability — is the product.

### Pipeline

```
RawData/MasterFiles/*.csv
   └─ data_cleaner ──> data/season<YYYY>.csv             (21 seasons, human-readable)
        └─ encoder + preprocess ──> data/processed/*.npz  (+ frozen vocabs, norm stats)
             └─ training/full_run ──> artifacts/<name>/   (11 heads + manifest + vocab snapshot)
                  └─ GameSimulator + GameController ──> one simulated game
                       └─ batched rollout x N sims x M games ──> results/<name>/<run>/
                            └─ reporting ──> report.html + data/*.parquet
```

---

## Design Philosophy — Simulation Thesis

> **Core hypothesis: games predict themselves.** You cannot reliably predict what happens in a
> basketball game from season averages alone. Outcomes are **path-dependent** — a player gets
> rattled or gets hot, a run swings the building — and that story only exists *inside the
> unfolding sequence of plays*, not in the box score before tip-off.

### Momentum is the load-bearing assumption

The system is built on the premise that **momentum is real**: recent events shift the
probabilities of the next ones. This is *why* prior plays are fed back into the model during
generation — the play sequence carries the momentum state, the model conditions on it, and in
simulation it consumes its own output, so a run the model starts can perpetuate itself the way a
real run does.

**The architecture *is* the hypothesis.** A causal transformer that attends over prior plays is,
mechanically, a momentum instrument: self-attention weights recent context, and autoregressive
feedback lets that context compound. The model does **not assume** a fixed momentum effect — it
learns whatever autocorrelation actually exists in the data and lets the Monte-Carlo aggregate
*measure* it. (Momentum/"hot-hand" is an empirically contested claim; this design treats it as
testable, not given.)

### Generation by sampling, not classification

- The model's value is its **conditional distribution** `p(next event, Δt | history)`, not its
  single top-1 guess.
- Inference **samples** from that distribution (per-head temperature, plus logit biases), so
  variance is a *feature*: it is what lets 100 rollouts tell 100 different believable stories.
- Success is **not** top-1 next-event accuracy. It is **calibration** and **aggregate
  distributional fidelity** — do the statistics of simulated games match real ones?

### Constraints keep rollouts on the manifold

A long (~500–900 step) self-fed rollout risks **compounding drift** (exposure bias): small
per-step biases snowball into unrealistic games. The Controller clamps each step back to a legal
basketball state. It is therefore both a **realism layer** and **drift insurance**.

---

## Data

### Data Source

- **Raw** play-by-play master files: `RawData/MasterFiles/` (one combined-stats CSV per season).
- **Cleaned, model-ready** season files: `data/season<YYYY>.csv`, one per season. The preprocessor
  consumes **every** `*.csv` in `data/` that has `game_id` + roster columns; anything parked in
  `data/_excluded/` is ignored.

### Current scale

| | |
|---|---|
| Seasons | **21** — `season2003.csv` … `season2023.csv` (2002-03 through 2022-23) |
| Games | **≈26,400** total |
| Train / validation split | **21,014 / 5,253** games (80/20 by game, split before norm stats) |
| Final holdout | **100** sequential real games after the training cut, never trained on and never used for early stopping |
| Player vocabulary | **2,153** players |
| Cleaned CSV on disk | ~4.3 GB (≈133 MB packed) |

The cut is chronological: training stops `FINAL_SEASON_FRACTION` (0.5) of the way through the most
recent season and the next `FINAL_HOLDOUT_GAMES` (100) real games become the holdout. Older seasons
are **recency-weighted** in the loss rather than dropped (see [Training](#model-training)).

### Data Structure (Per Event)

Each row represents a single game event:

| Field | Description |
|---|---|
| `game_id` | Game the event belongs to (sequences are grouped by this) |
| `roster_home` | 5 players — home lineup on court |
| `roster_away` | 5 players — away lineup on court |
| `time` | Game clock time (seconds elapsed) |
| `event` | Event type (see [vocabularies](#vocabularies)) |
| `player` | Primary player involved |
| `secondary_player` | Secondary player (e.g. assister, blocker); shares the player vocab |
| `type` | Subtype of event |
| `result` | Outcome |
| `season` | Season token |
| `home`/`away`, `playoff` | Game context carried through the cleaner |

Prior-play history is **not** a stored column — it is supplied implicitly: each game is one
chronologically-ordered sequence and the causal transformer attends over all prior timesteps.

Season context (per-player rest, team rest, games played) is derived by `season_context.py` and
merged in at preprocess time; game state (score, period, clock, team fouls) is derived by
`models/game_state_features.py`.

---

## Data Cleaning

```
RawData/MasterFiles/*.csv → clean → data/season<YYYY>.csv
```

1. Import the raw master play-by-play CSV(s).
2. Normalize into model-ready rows with `game_id`, `roster_home`, `roster_away`, and the event
   fields above; roll multi-row NBA events into the project's single-row-per-event encoding.
3. Write cleaned per-season CSVs to `data/`.

```bash
python main.py --clean --rebuild-vocabs --model event_time
```

Cleaning is CPU-only data processing — no GPU, and normally done locally rather than on a rented
pod. Cleaned data must stay chronologically ordered and grouped by `game_id` (one game = one
sequence). Encoding (integer tokenization) happens in **preprocessing**, not cleaning, so the
cleaned files stay human-readable CSV. Commit `encoder/vocabs/*.json` after a re-clean.

---

## Preprocessing

### Vocabularies

Shared, frozen "language" vocabularies live in `encoder/vocabs/*.json`, each reserving `PAD=0`,
`UNK=1`. `player` is shared by both `player` and `secondary_player` (same entity → same embedding).

| Vocab | Size | Tokens |
|---|---|---|
| `event` | 12 | `PAD UNK start end none shot rebound assist turnover block foul substitution` |
| `type` | 24 | `2pt 3pt free throw defensive offensive error personal steal shooting loose ball violation technical flagrant-1 flagrant-2 away from play personal take transition take substitution …` |
| `result` | 19 | `missed made blocked block score cop op steal free throw free throw op nothing substitution ejection …` |
| `season` | 23 | `2003` … `2023` |
| `player` | 2,153 | one token per player across all 21 seasons |

**Every head's `save_artifacts()` writes the shared `encoder/vocabs/`**, so each trained model also
carries a private snapshot at `artifacts/<name>/vocabs/` — otherwise training a new model would
invalidate an older model's embedding tables.

### Feature Normalization

Stats are computed on the **train split only** (no test leakage) and persisted to
`encoder/vocabs/norm_stats.json`:

```json
{"max_time": 4080.0, "delta_mean": 5.894, "delta_std": 7.301,
 "rest_mean": 2.453, "rest_std": 2.293}
```

```
time_abs   = time / max_time                       # max_time = 4080 s (68 min, covers OT)
delta_time = (Δt - delta_mean) / delta_std
rest       = (clip(days_rest, 0, 30) - rest_mean) / rest_std
```

Δt is the per-game gap between consecutive events (`groupby(game_id).time.diff()`, first event → 0,
backwards clips → 0). Game-state scalars use **fixed** clip/divisor constants rather than fitted
stats, so they add no `norm_stats` keys and cannot drift between train and inference.

### Sequence Construction

- One game = one sequence, framed by `start` / `end` event tokens.
- Right-padded to `MAX_SEQUENCE_LENGTH = 600` (covers OT/overflow; truncate beyond).

### Input Features (Per Timestep)

23 arrays in total: 10 base, 6 season-context, 6 game-state, plus the pad mask.

| Group | Feature | Notes |
|---|---|---|
| **Base** | `event`, `player`, `secondary_player`, `type`, `result`, `season` | Categorical embeddings; `secondary_player` is **weight-tied** to `player` |
| | `home_roster`, `away_roster` | 5-player lineups → shared Set-Transformer vector |
| | `time_abs`, `delta_time` | Normalized clock + normalized Δt since previous event |
| **Season context** | `rest_home`, `rest_away` | Per-player days rest, `(SEQ, 5)`, fed **into the roster set-encoder** so player selection sees freshness |
| | `home_games_played`, `away_games_played`, `home_days_rest`, `away_days_rest` | `(SEQ, 1)` team scalars, Dense-projected into the fusion |
| **Game state** | `score_diff`, `score_total`, `period_idx`, `period_time_left`, `team_fouls_home`, `team_fouls_away` | `(SEQ, 1)` scalars, Dense(16)-projected into the fusion. **Plumbed and tested, but `v1.0`'s weights were trained before activation** — train 3 is the first run whose weights consume them |
| | `pad_mask` | 1 = real step, 0 = padding (attention masking) |

Train/inference parity for the game-state features is by construction: one `GameStateScan` class is
shared by the preprocessor and the live simulator.

### Representative subset

The small categorical/regression heads saturate long before they see the whole corpus and start to
overfit, so seven of them train on a compact, *representative* slice (`training/subset.py`,
persisted to `training/subset_games.json`) instead of every game:

- Per-season sample rate, newest first: `(1.0, 0.70, 0.50)`, then decaying with a 5-season halflife.
- **Coverage-complete**: every player in the train pool is guaranteed at least one game, so no
  embedding goes starved.
- Subset heads: `event_time_cond`, `shot_type`, `shot_result`, `assist_type`, `turnover_type`,
  `foul_type`, `rebound_type`. The big player-vocab heads (`event_time`, `player`, `substitution`,
  `stint_length`) keep the **full corpus** — they actually need the data.

---

## Models

### Shared backbone

Every head is a causal transformer encoder over the same 23 inputs, emitting a prediction at
**every** timestep (no pooling). One place — `config.py` — sets the capacity so train and reload
cannot disagree:

| Hyperparameter | Value |
|---|---|
| `MODEL_DIM` | 384 |
| `NUM_LAYERS` | 6 |
| `NUM_HEADS` | 8 (key_dim 48) |
| `FF_DIM` | 1536 |
| `ROSTER_SAB_LAYERS` | 3 |
| `MAX_SEQUENCE_LENGTH` | 600 |
| dropout | 0.15 |
| embed dims | event 32, player 192, type 32, result 16, season 16 |
| `ROSTER_DIM` | 128 |

**Layer flow.** Per-field embeddings (categoricals) + Set-Transformer roster vectors + Dense
projections of the continuous features → concatenate → `Dense(MODEL_DIM)` → LayerNorm → add
learned positional embeddings → dropout → 6 pre-LN causal blocks (MHA with causal + key-padding
mask → residual; GELU feed-forward → residual) → float32 output head(s).

> **On model size.** ~12–13M parameters per head is the *correct* size for this configuration.
> Transformer size scales as ≈ `layers × 12 × model_dim²`; with `model_dim=384` and 6 layers the
> blocks dominate, and the small (~2,200-token) vocabularies keep the embedding tables modest.
> Capacity was raised from the original 256/4/1024/2 backbone in train 2, and scales *with* data.

### Set Transformer (roster encoding)

Each 5-player lineup is encoded to a fixed vector by a **permutation-invariant** Set Transformer
(3 Set-Attention blocks + pooling-by-attention), shared and weight-tied across home and away. Order
of players carries no meaning, so the encoder is invariant to it by construction. Per-player rest
rides alongside the player ids into the encoder.

### The eleven heads

Trained in dependency order (`models/registry.STAGE_MODEL_KEYS`); the rollout calls them in the
same order per event.

| # | Key | Predicts | Corpus | Params |
|---|---|---|---|---|
| 1 | `event_time` | next event token **and** next Δt (two output heads) | full | 12.53M |
| 2 | `player` | the actor, over the player vocab, masked to the on-court ten | full | 13.37M |
| 3 | `event_time_cond` | Δt again, now conditioned on the decided event + actor | subset | 12.61M |
| 4 | `shot_type` | `2pt` / `3pt` for a live field goal | subset | 12.62M |
| 5 | `shot_result` | `made` / `missed` / `blocked`, conditioned on shooter **and** shot type | subset | 12.64M |
| 6 | `assist_type` | assisted-shot type | subset | 12.62M |
| 7 | `turnover_type` | turnover subtype (steal, error, violation …) | subset | 12.62M |
| 8 | `foul_type` | foul subtype (shooting, personal, loose ball, offensive, technical …) | subset | 12.62M |
| 9 | `rebound_type` | offensive / defensive — decided from history **before** the rebounder is picked | subset | 12.55M |
| 10 | `substitution` | the **incoming** player of a substitution, over the player vocab | full | 13.44M |
| 11 | `stint_length` | log-seconds an entering player stays on the floor | full | 12.69M |

Total ≈ **140M** parameters across the stack; a loaded eval process costs ~3–4 GB of VRAM.

**Availability masking** (from train 2): the player-vocab heads are masked to each game's actually
available player set, so a head can never spend probability mass on someone not dressed.

`shot_type` learns **live field goals only** (`{2pt, 3pt}`): in the cleaned data a free throw is a
shot row, but the simulator never asks `shot_type` to choose one — FTs are emitted directly from
fouls by the Controller — so training on them would only confuse a binary head.

---

## Model Training

### Paradigm

**Autoregressive (GPT-style)** — predict the next timestep given all prior timesteps. Every head
trains with a per-row `sample_weight` mask, so PAD steps, the final step (no next-step target) and
rows the head does not own contribute zero loss.

### Loss

| Head | Loss |
|---|---|
| `event_time` | `1.0 · CE(event) + 0.5 · MAE(Δt_norm)`, both masked |
| `player`, `substitution`, the type/result heads | masked sparse categorical cross-entropy (`from_logits`) |
| `event_time_cond`, `stint_length` | masked MAE on the normalized / log target |

### Optimization

- Optimizer **AdamW** (`weight_decay=1e-4`, `clipnorm=1.0`), `lr = 3e-4`.
- **LR schedule:** linear warmup (2 epochs) → cosine decay to `lr_alpha · lr` (α = 0.05).
- **Regularization:** dropout 0.15, routed through embedding, attention, feed-forward **and**
  roster-encoder layers.
- **Early stopping** `patience=15`, best weights restored; ≤50 epochs.
- **Mixed precision** (`mixed_float16`) on GPU; output heads forced to float32.
- **Recency weighting:** every game trains, but its loss weight halves every
  `RECENCY_HALFLIFE_SEASONS = 3.0` seasons of age, floored at `RECENCY_FLOOR = 0.05`, so the modern
  game dominates the gradient while old-player embeddings keep getting some.

### Train batch capacity

`--batch-size` sets the train batch for every head; the three player-vocab heads are additionally
capped by `LARGE_OUTPUT_BATCH` (`models/pipeline.py`) because they emit logits over the 2,153-token
player vocab — that logits/gradients bulge is the actual OOM cause on a tight card.

### Measured results — `v1.0` (full train 2, 2026-07-06/07)

Per-head, from the training reports under `reports/<head>/<run>/`:

| Head | Best val loss | Final val metric | Best epoch / run | Wall clock |
|---|---|---|---|---|
| `event_time` | 0.2229 | event acc **75.3%**, Δt MAE **2.37 s** | 47/50 | 5.4 h |
| `player` | 0.4893 | actor acc **37.5%** (2,153-way, masked to the on-court ten) | 13/29 | 3.4 h |
| `event_time_cond` | 0.0802 | Δt MAE **2.31 s** | 48/50 | 1.0 h |
| `shot_type` | 0.0410 | acc **80.1%** | 6/22 | 27 min |
| `shot_result` | 0.0725 | acc **76.6%** | 7/23 | 28 min |
| `assist_type` | 0.0184 | acc **65.4%** | 4/20 | 25 min |
| `turnover_type` | 0.0131 | acc **80.6%** | 5/21 | 26 min |
| `foul_type` | 0.0257 | acc **58.7%** | 6/22 | 27 min |
| `rebound_type` | 0.0338 | acc **73.9%** | 13/29 | 35 min |
| `substitution` | 0.0514 | acc **47.7%** | 13/29 | 3.5 h |
| `stint_length` | 0.0149 | stint MAE **187 s** | 47/50 | 5.5 h |

**None of these are the project's scoreboard** — see [Evaluation](#evaluation).

Several subset heads early-stop in single-digit epochs (`shot_type` 6, `assist_type` 4), which is
why the subset sample rates were raised afterwards: they have headroom for more modern data.

---

## Rollout & Controller

### Generation loop

```
1. Build the model inputs from the game history so far (incrementally cached).
2. event_time      -> sample the next event token (masked to what is legal here) + a Δt.
3. player          -> sample the actor, masked to the on-court five of the acting team.
4. event_time_cond -> re-time the step now that event + actor are decided.
5. type/result     -> sample the detail heads for that event class.
6. Controller      -> expand forced consequences, apply the rules, advance clock/score/fouls.
7. Append to history; repeat until the final whistle.
```

Roughly **5 forward passes per event over ~500–900 events** per game.

### Controller (`simulation/controller.py`)

The rule engine. The models only ever choose *among legal options*: the Controller masks the event
head to what is legal in the current context, samples the actor/type/result from the conditional
heads (again masked), then expands forced consequences exactly as the cleaned data encodes them (an
assist is followed by a made shot; a blocked shot is a missed FGA; a shooting foul yields the right
number of free throws). It owns what the models never see:

- clock, score, possession, period/OT structure
- per-period team fouls and the NBA bonus
- foul-outs (`FOUL_OUT_LIMIT = 6`) and ejections
- the substitution scheduler: each entering player is committed to a sampled stint length and
  scheduled off at the next dead ball past their exit, with `SUB_MAX_GAP_SECONDS` as a backstop so
  a team cannot play five men for 48 minutes
- dead-ball rebounds (`DEADBALL_REBOUND_PROB`) — the rare no-individual-rebounder case

### Throughput

Three independent mechanisms, none of which change what is predicted:

| Mechanism | What it does |
|---|---|
| **Input cache** (`simulation/input_cache.py`) | History rows are immutable once appended, so each row is encoded **once** rather than rebuilt on all ~5 head calls per event. Bit-identical to the naive path, which remains as the test oracle (`CVIQ_INPUT_CACHE=0`). |
| **Batched rollout** (`simulation/batched_rollout.py`) | Runs `ROLLOUT_BATCH_SIZE` (48) independent game-sims concurrently on worker threads, pooling their per-event forward passes into **one batched GPU call per head**. Slots backfill, so a slot that finishes a short game immediately pulls the next one. |
| **Process pool** (`eval_pool.py`, `--procs N`) | Everything around the forward pass is Python, so one process is GIL-bound to ~one core. `--procs` runs N `--shard i/N` children over disjoint holdout slices and merges one report. On a multi-GPU box shards are dealt round-robin across cards. |

An **opt-in** compiled-inference path (`CVIQ_TF_INFER=1`) wraps the head calls in `tf.function`; it
falls back to eager per-signature on any incompatibility, so results are unchanged. It needs
on-hardware measurement before being trusted.

**Durability.** Long pooled runs write each sim to disk as it finishes; one failed sim does not
take the slot, and one failed game does not take the run
(`EVAL_MAX_CONSECUTIVE_GAME_FAILURES = 3` stops a shard that a CUDA OOM has poisoned, and the
pool's remainder wave respawns it with a clean context). The supervisor rebuilds the report from
finished records every `EVAL_REPORT_EVERY_SEC` (300 s), so a 36-hour run is queryable long before
it ends. `harvest.py` archives finished games' play-by-play off the volume on a timer, which is
what keeps a long run's disk usage flat.

---

## Evaluation

The real test of the system is **not** next-event accuracy but whether **simulated games** match
reality. Training metrics and simulation evaluation are deliberately separate: `reports/` holds
training reports, `results/` holds evaluation runs.

### What is measured

| Dimension | How |
|---|---|
| **Win prediction** | Pick accuracy + **Brier score** + log loss + a reliability table, scored two ways: majority vote across sims, and the sign of the mean predicted margin |
| **Point spread** | MAE / bias / RMSE / correlation and within-3/6/10 hit rates on the predicted margin |
| **Team box accuracy** | Per-stat predicted mean vs actual mean, MAE and bias over 200 team-games (minutes, pts, fga/fgm, tpa/tpm, fta/ftm, oreb/dreb, ast, stl, blk, tov, pf) |
| **Advanced** | Four factors + pace (eFG%, TOV%, OREB%/DREB%, FT rate) |
| **Per-player box accuracy** | The same stat table over every real player-game |
| **Player minutes** | Pooled MAE, within-2/5/10-minute hit rates, predicted-vs-actual scatter and a signed-error histogram |
| **Tuning progression** | The run segmented by where a dial changed, so a dial's effect is readable in place |

### The headline test

> Simulate each of 100 held-out games 100× and check whether the actual outcome falls sensibly
> inside the simulated distribution, then assess win-probability calibration across the games.

### Measured results — `v1.0`

Each run is the same 100-game 2022-23 holdout; `full4-s100` is the current headline (100 sims per
game rather than 21).

| Run | Sims | Pick acc | Brier | Team pts MAE | Spread MAE | Pts bias | Pace bias |
|---|---|---|---|---|---|---|---|
| `trial1` | 28 | 61% | 0.2273 | 13.05 | 9.64 | **−11.55** | +0.04 |
| `full1` | 21 | 66% | 0.2176 | 9.54 | 9.48 | +0.43 | +0.50 |
| `full2` | 21 | 62% | 0.2250 | 9.52 | 9.58 | +0.62 | +0.95 |
| `full3` | 21 | 65% | 0.2178 | 9.42 | 9.06 | +0.66 | +0.85 |
| **`full4-s100`** | **100** | **63%** | **0.2173** | **9.26** | **9.49** | **−0.52** | **+0.47** |

The scoring shortfall that dominated `trial1` (simulated teams scoring ~11.5 points too few) was
closed by the dial packages fitted across `full1`–`full3`: a shot-result make-rate correction, a
foul/turnover event-mix correction, and rebound-volume and home-court trims. `full4-s100` lands at
115.4 predicted vs 115.9 actual points per team.

**`full4-s100` detail** (200 team-games, 2,071 player-games):

| | Predicted | Actual | MAE | Bias |
|---|---|---|---|---|
| Points | 115.41 | 115.93 | 9.26 | −0.52 |
| Pace | 101.15 | 100.68 | 4.24 | +0.47 |
| eFG% | .538 | .554 | .050 | −0.017 |
| FGA | 89.27 | 88.10 | 5.35 | +1.17 |
| 3PA | 36.27 | 33.96 | 5.13 | +2.31 |
| FTA | 25.32 | 23.88 | 5.68 | +1.44 |
| AST | 23.36 | 25.32 | 4.04 | −1.95 |
| Team minutes | 240.83 | 241.00 | 1.77 | −0.17 |
| **Player minutes** | 23.26 | 23.27 | **5.66** | −0.02 |
| Player points | 11.15 | 11.20 | 4.83 | −0.05 |

Win-probability calibration is close across the populated bins (predicted .329 → observed .333;
.507 → .488; .682 → .684), with only the small ≥0.8 bin (n=6) off.

**Remaining known gaps at `full4-s100`:** eFG still runs ~1.7 points low, 3PA ~2.3 attempts high,
assists ~2 low, and player-minutes MAE of 5.7 minutes is the largest single lever on per-player box
accuracy. Win-pick accuracy (63%) moves by several points between runs at these sample sizes, so
treat single-run differences of that size as noise, not signal.

### The baseline that matters

`reporting/baseline_comparison.py` answers the question a low MAE alone cannot: **does the model
beat "predict the player's season-to-date average"?** NBA box scores are heavily mean-reverting, so
that baseline is strong. It is read-only and runs no sims — it reuses the per-game records already
in a run's `report.json`.

```bash
python -m reporting.baseline_comparison results/v1.0/full4-s100
```

At `full1` the model **did not** beat it: player points MAE 4.92 (model) vs 4.56 (season-to-date)
— about 8% worse, winning the paired comparison on 46.8% of player-games. That gap has narrowed
from ~44% worse / 35% win rate in June, and `full4-s100`'s player points MAE improved to 4.83, but
the comparison has not been re-run against it. **This is the open headline item.**

---

## Inference Dials

Rollout behavior is shaped by dials in `config.py`, read at call time — **no retrain**. Change them
with `set` in the `cviq` shell (no reload), or edit `config.py` for a new default. Every eval report
records the dials that produced it, and the progression table segments a run by where they changed.

| Dial | Effect |
|---|---|
| `DELTA_TIME_SCALE` (0.97) | Multiplies predicted inter-event Δt → **pace**. Re-fit from `python -m simulation.diagnostics` (= real Δt mean / sim Δt mean) **after every retrain**. |
| `MAX_DELTA` (60 s) | Clamp on a single inter-event gap. Lower trims long dead-ball gaps → nudges pace up. Secondary pace lever. |
| `EVENT_BIAS` | Additive logit offsets on the event mix (currently `foul +0.11`, `turnover −0.08`). |
| `TYPE_BIAS` | Per-head, per-token offsets on the conditional type heads (foul mix, steal share, assisted-3 share, offensive-rebound share). |
| `SHOT_RESULT_BIAS` | Made/missed/blocked offsets → eFG / FG% (currently `made +0.40`, `blocked −0.15`). |
| `PLAYER_TEMPERATURE` (2.0) | Flattens the actor head. Above 1 on purpose: the full-corpus head is confident enough that at 0.8 one star vacuumed points, rebounds **and** assists at once. |
| `SUB_INCOMING_TEMPERATURE` (0.45) | Sharpens the incoming-sub pick onto the real 8–9 man rotation. |
| `STINT_LENGTH_SCALE` (1.30) | Multiplies predicted stint lengths → substitution rate / minutes concentration. The log-stint head's point estimate is a geometric mean and under-shoots the arithmetic mean; tune to real ~46 subs/game. Re-fit **after every retrain**. |
| `HOME_COURT_SHOT_BIAS` (0.055) | Symmetric made-shot logit nudge for the home offense; the rollout is otherwise home/away symmetric, so this is what separates winners. Tune against spread bias. |
| `DEADBALL_REBOUND_PROB` (0.10) | Share of misses yielding no individual rebound — the total-rebound-volume lever (the off/def split is `TYPE_BIAS.rebound_type`'s job). |
| `MARGIN_CALIBRATION_SLOPE/INTERCEPT` | Post-hoc linear calibration on predicted margin, applied **only** when aggregating spread metrics — never to the raw per-game record, so it can be refit without touching sim data. Not a rollout dial. |

Dials marked "re-fit after every retrain" are calibrations against a specific set of weights;
carrying one across a retrain unexamined is how `trial1`'s pace correction became `full1`'s pace
collapse in the other direction.

`ROLLOUT_BATCH_SIZE`, `EVAL_GAMES_PER_BATCH`, `EVAL_PROC_*` and the durability constants are
**perf knobs, not dials** — deliberately excluded from `_TUNING_KEYS` so a benign difference cannot
make a merged report claim the run mixed physics.

---

## Reporting

One standardized layer (`reporting/`) that every model and every eval run routes through. Each run
emits a self-contained **HTML report** plus **queryable Parquet**, so cross-run analysis is a
`pandas.read_parquet` away rather than a re-parse of HTML.

**Training** (`reports/<head>/<run>/`): `report.html`, `report.json`, `run.parquet`,
`epochs.parquet` — hyperparameters, the trainable-parameter count, per-epoch loss/metric curves,
epoch durations, learning rate, and final held-out test metrics.

**Evaluation** (`results/<model>/<run>/`):

```
report.html  report.json          aggregate: win/spread/box/advanced/minutes + the dials that produced it
holdout.json                      the game ids this run covers
dials.json                        the dial package every shard was handed
data/                             games, box_players, summary, run_summary, progression (Parquet)
games/<matchup>/                  one folder per holdout game:
  game.html                       predicted (mean) / actual (raw) / variance box scores
  *_boxscore_*.csv  *.txt         box scores as CSV/text
  playbyplay/                     actual + every simulated play-by-play (CSV)
  record.json  run.json
```

`record.json` is load-bearing: it is the completion marker the resume path counts, and the only
input the report builder reads. That is what lets `harvest.py` archive a finished game's
play-by-play off the volume without breaking either resume or the final merge.

Regenerate a report from `report.json` with no new sims:

```bash
python -m reporting.update_eval_report results/v1.0/full4-s100
```

---

## Repository Map

```
data/                     cleaned, model-ready season CSVs (season2003.csv …); not in git
  processed/              preprocessed tensors (regenerated by training)
encoder/                  Encoder + frozen token vocabularies (vocabs/ is committed)
layers/                   transformer building blocks (MAB, SAB, PMA, row feed-forward)
models/                   the 11 head wrappers + registry, artifacts/manifest, pipeline,
                          season_features, game_state_features, roster_set_encoder
simulation/               rollout engine, Controller, box scores, evaluation, diagnostics,
                          batched_rollout, input_cache, stage_eval, profile_rollout
reporting/                training + evaluation reports (HTML + Parquet), baseline_comparison
training/                 full-train orchestration, chronological slicing, subset selection
  full_run_state.json     holdout ids + data paths, written by a full train; not in git
artifacts/<name>/         one trained model per dir (weights + manifest.json + vocabs/); not in git
reports/<head>/<run>/     per-head TRAINING reports
results/<model>/<run>/    EVALUATION runs; not in git
shell/  cviq.py           the interactive TRAIN / LOAD / RUN shell
train.py  evaluate.py     non-interactive CLIs
eval_pool.py  harvest.py  process-pool supervisor; play-by-play archiver (both TF-free by rule)
main.py  data_cleaner.py  cleaning + single-model entry points
config.py                 capacity, dials, split/seed, perf knobs — one source of truth
tests/                    38 test modules; model IO coverage is parametrized over the registry
```

**Model names.** Weights live one directory per model under `artifacts/<name>/`, where the name is
free-form (`v1.0`, `endgame-feats`). The name **is** the train identity: a retrain takes a new name
rather than overwriting, so weights and the runs evaluated against them never drift apart. Each
model carries a `manifest.json` (arch, seed, epochs, git commit, data cut, holdout, recommended
dials) and a `vocabs/` snapshot; `cviq> adopt <name>` back-fills both for weights trained before
either existed, with no retrain.

---

## Open work

### Under consideration — v2 theories (see `docs/v2_theories.md`)

**Nothing below is agreed or built.** The theory doc carries the reasoning and the measurements
behind each one. Ids are its own: `S` = schema (what the event stream should carry), `M` = model
and training (how it consumes that stream).

| # | Theory | Note |
|---|---|---|
| S1 | Shot zones — expand `shot_type` from `{2pt, 3pt}` to 7 court zones | Changes the shared vocab, so it forces a full retrain |
| S2 | Free-throw count as a learned outcome (`shooting 2pt` / `shooting 3pt`) | The sim currently awards 3 FTs ~12x too often |
| S3 | Fouled player as `secondary_player` on foul rows | Raw `opponent` is 98.1% populated and currently dropped |
| S4 | Collapse the steal pair into one turnover row | Matches the pattern `block` already uses |
| S5 | FT index (`num`/`outof`) | Cleaner change; also the label that makes S2 possible |
| S6 | Player age (external roster table) | The name-matching join is the real work |
| S7 | Coach (rolling team style priors + coach embedding) | External table |
| M1 | Game-state features (score / period / clock / team fouls) | Plumbing built and tested; needs a train that consumes it |
| M2 | Clutch loss weighting | Rides the existing `sample_weight` masks |
| M3 | Recency / local-sequence bias in the backbone | Touches all 11 heads |
| M4 | Play-boundary loss masking | 21.9% of event-head training positions never occur at inference |

### Parked

| Area | Question |
|---|---|
| **Rotation / minutes model** | A dedicated head that predicts stints and on-court minutes directly, with seeded starters, replacing the current event-head-driven substitutions. Player-minutes MAE of 5.7 min is the biggest single lever on per-player box accuracy. |
| **Baseline** | Beating season-to-date player averages on per-player box MAE — still ~8% behind at `full1`, and the check has not been re-run since. |
| **Relative encoding** | Offense/defense frame instead of home/away; evaluable only with a full retrain. |
| **Calibration auxiliary losses** | Aggregate-consistency terms per head; design exists in notes only. |
| **Tracking data** | Shot location as something the model *generates* against the specific defense on the floor, rather than a token it looks up. Needs a license. |
| **Live inputs** | Injury reports and confirmed lineups, which the betting market prices and this model does not ingest. |

---

*CourtVisionIQ — simulate the NBA from scratch.*
