# CourtVisionIQ — Technical Specification

> *"Learn the rhythm and structure of basketball games first, then layer detail on top."*

---

## Table of Contents

1. [Overview](#overview)
2. [Data](#data)
3. [Data Cleaning](#data-cleaning)
4. [Preprocessing](#preprocessing)
5. [Models](#models)
6. [Model Training](#model-training)
7. [Evaluation](#evaluation)
8. [System Integration](#system-integration)
9. [TBD / Open Questions](#tbd--open-questions)

---

## Overview

CourtVisionIQ is a multi-model basketball simulation and prediction system designed to generate realistic play-by-play NBA game sequences.

The system is structured as an **ensemble of models**:

- The **Event/Time Transformer** generates the game skeleton (events + timing)
- **Downstream models** (e.g., Shot Generator) fill in granular details

### Pipeline

```
Raw CSV Data → Cleaning → JSON → Encoding → Model Training → Evaluation → Controller → Simulation
```

---

## Data

### Data Source

Raw data is stored as `.csv` files inside a zip archive:

```
/NBAdata/NBAdata-dirty.zip
```

### Data Structure (Per Event)

Each row represents a single game event:

| Field | Description |
|---|---|
| `roster1` | 5 players — Team 1 |
| `roster2` | 5 players — Team 2 |
| `time` | Game clock time |
| `event` | Event type |
| `player` | Player involved |
| `type` | Subtype of event |
| `result` | Outcome |
| `prior_plays` | Nested history of prior events |

### Sequence Structure

- Data is grouped into **full game sequences**
- Each sequence is **chronologically ordered**

---

## Data Cleaning

### Pipeline

```
CSV → Nested DataFrame → JSON
```

### Steps

1. Import raw CSV files
2. Convert to structured format (nested per game)
3. Export to JSON:

```
/content/data/NBAdata
```

### Output Variants

| Variant | Description |
|---|---|
| Plaintext JSON | For inspection and feature engineering |
| Encoded JSON | Integer-tokenized, ready for training |

### Notes

Data must be:
- Chronologically ordered
- Grouped by game
- Nested structure preserves full play history

---

## Preprocessing

### Vocabulary Construction

Vocabularies are extracted for:

- `players`
- `events`
- `types`
- `results`
- `rosters`

Stored at:

```
/content/data/languages/*.json
```

### Encoding

- All categorical values → integer tokens
- Used for embedding layers
- Encoded prior plays → `/NBAdata-encoded`

### Feature Normalization

```
time = time / max_time
```

Δtime is normalized at the model level.

### Sequence Construction

- Data is split into sequences by **game start token**
- Fixed sequence length:

```
game_length = 800
```

### Input Features (Per Timestep)

| Feature | Notes |
|---|---|
| `event` | Categorical |
| `player` | Categorical |
| `type` | Categorical |
| `result` | Categorical |
| `roster1` | 5-player lineup |
| `roster2` | 5-player lineup |
| `time` | Absolute + delta |
| Game context | Season, playoff flag, home flag |

---

## Models

### Event/Time Model (Core)

**Purpose:** Predict the next event and time until that event.

```
p(e_{t+1}, Δt_{t+1} | history)
```

**Architecture:**

- Transformer Encoder (causal)
- Multi-head attention
- Positional encoding
- Set Transformer for roster encoding

**Inputs:** Concatenated embeddings:

```
[event, player, type, result,
 roster_team1, roster_team2,
 time_abs, time_delta,
 home_flag, season, playoff]
```

**Outputs:**

| Output | Type |
|---|---|
| `event_output` | Classification (softmax) |
| `time_output` | Regression (scalar) |

**Event Classes:**

- `shot`
- `assist`
- `rebound`
- `block`
- `turnover`
- `foul`
- `substitution`

---

### Shot Generator Model

- Separate model trained **after** the Event/Time Model
- Predicts shot outcomes and details
- Operates as a downstream detail layer

---

### Transformer Implementation

**Inputs:**

```
game_id, roster1, roster2,
time, event, player, type, result
```

**Layers:**

1. Embedding layers (categorical features)
2. Dense layers (feature alignment)
3. Transformer block
4. Global pooling

**Outputs:**

- `event_output` → softmax classification
- `time_output` → regression scalar

---

### Set Transformer (Roster Encoding)

- Encodes 5-player lineups
- **Permutation invariant** — order of players doesn't matter
- Based on attention over sets

---

### Controller (Post-Model)

Enforces basketball rules on generated sequences:

- Assists must follow shots
- Blocks must follow shots
- Foul logic and sequencing

---

## Model Training

### Training Paradigm

**Autoregressive (GPT-style)** — predict the next timestep given all prior timesteps.

### Loss Function

```
L = λ_event · CE + λ_time · MAE
```

| Term | Description |
|---|---|
| `λ_event · CE` | Cross-entropy for event classification |
| `λ_time · MAE` | Mean absolute error for time regression |

### Optimization

- Optimizer: `Nadam` / `AdamW`
- Gradient clipping applied

### Training Setup

- Train/test split: **80% / 20%**
- Batch training
- Early stopping
- Learning rate reduction on plateau

### Sequence Handling

- Right-padding applied
- Padding masking used during attention

---

## Evaluation

### Metrics

| Prediction Task | Metric |
|---|---|
| Event prediction | Accuracy, Cross-entropy loss |
| Time prediction | Mean Absolute Error (MAE) |

### Tracking

- Batch accuracy over time
- Median accuracy
- Loss curves

### Visualization

- Accuracy vs. batch
- Loss vs. batch

---

## System Integration

### Generation Loop

```
1. Input initial game state
2. Predict next event + Δtime
3. Append to sequence
4. Repeat until game end
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
| **Data** | Exact source provider (NBA API, scraping, etc.) |
| **Cleaning** | Handling of missing / null values |
| **Preprocessing** | Padding strategy specifics (assumption: right-padding) |
| **Models** | Number of transformer layers |
| **Models** | Embedding dimensions (partially defined: 128) |
| **Training** | Epoch count (variable) |
| **Training** | Batch size (~100 observed) |
| **Evaluation** | Validation split strategy beyond 80/20 |

---

## Next Steps

- [ ] Define Player Generator model spec
- [ ] Finalize embedding dimensions across all models
- [ ] Specify data source and ingestion pipeline
- [ ] Define null/missing value handling strategy
- [ ] Map spec to repo folder structure

---

*CourtVisionIQ — simulate the NBA from scratch.*