# CourtVisionIQ 2.0 — Theories

> **Status: these are ideas we are weighing for a version 2. Nothing here is agreed, and nothing
> here is built.** The doc exists to hold the reasoning and the measurements behind each idea so
> they can be argued with, priced against each other, and either promoted or dropped on evidence
> rather than memory. Started as the "train 3 spec" (scope sketched 2026-08-25); reframed
> 2026-08-29, because none of it was ever built and all of it is really the same activity —
> figuring out what would actually improve the model next.

## The thesis

Several of these theories look unrelated and are not. **The cleaned event schema is a
transcription format, not a modeling format.** It mirrors NBA play-by-play row-for-row, and the
simulator papers over everything the schema omits with hard-coded rules in
`simulation/controller.py`: free-throw counts, and-1 detection, steal-pair expansion, multi-row
play expansion.

Every one of those rules is a place where the model is not allowed to learn something it could
learn, and where the sim can be confidently wrong with no gradient to correct it. The 3-FT bug
below is exactly that failure, caught in the wild.

So the organizing idea for a v2 is: **move what the controller hard-codes into the data and the
model.** Theories S2, S3, S4, S5 and M4 are all instances of it. The rest (S1, S6, S7, M1, M2, M3)
are the independent question of giving the model richer signal to learn from.

---

## Evidence

Four measurements the theories below lean on. Each names how it was produced so it can be
re-derived without the conversation that found it.

### Where the model actually stands (`full4-s100`, 100 games x 100 sims)

- Team points bias **-0.52**, pace bias **+0.47** — the scoring shortfall earlier versions of this
  doc were partly aimed at is now closed by dials.
- What remains on the shooting side is *composition*: eFG runs **1.7 points low** while 3PA runs
  **2.3 attempts high**. That strengthens the case for shot zones (S1) and weakens the case for
  further blunt eFG dialing.
- Per-player minutes are now scored and miss by **5.66 minutes** (MAE). Minutes multiply every
  per-player counting stat, so this is plausibly the largest single term in per-player box MAE —
  and **nothing in this doc addresses it.** The rotation-minutes model stays parked below, but
  that parking is a measured trade rather than an assumed one.

### The 3-FT bug — the sim awards three free throws ~12x too often

`GameController._do_shooting_foul` (`simulation/controller.py:516-545`) decides the free-throw
count like this:

```python
stype = self.sim.predict_type("shot_type", "shot", shooter, SHOT_TYPES, delta_seconds=0.0, ...)
n_ft = 3 if stype == "3pt" else 2          # controller.py:541-542
```

The fouled attempt is never logged as a field-goal attempt (NBA scoring), so there is nothing to
*read* — the shot type is **sampled fresh from the live-field-goal `shot_type` head**, whose prior
is the league-wide 2pt/3pt split of *all* attempts (~35-40% threes). Four compounding faults:

- **Wrong distribution.** `shot_type` trains on real FGAs only
  (`models/conditional_type_model.py:115`, `target_tokens=("2pt","3pt")`). Its own docstring says
  the simulator should not ask it to choose here.
- **Out-of-distribution context.** The `foul / shooting / free throw` row is appended at
  `controller.py:524` *before* the sample, so the head is conditioned on "a shot follows a
  shooting-foul row" — a state absent from training.
- **`delta_seconds=0.0` hardcoded** (`controller.py:541`) normalizes to an out-of-range z-score.
- **And-1 under-triggers structurally.** `_do_shot` flips possession the instant a field goal goes
  in (`controller.py:280`), so a later shooting foul falls in the *next* possession and the guard
  at `controller.py:533` fails. Almost everything funnels into the 2-or-3 sample.

There is no `TYPE_BIAS` entry for `shot_type` (`config.py:106-112`), so this sample is entirely
uncalibrated. Three rounds of dial tuning (`config.py:76-105`) chased shooting-foul *volume* and
team FTA; none ever touched the per-trip 2-vs-3 split.

Ground truth, from the raw `outof` column
(`RawData/MasterFiles/[10-18-2022]-[06-12-2023]-combined-stats.csv`, first ~400k rows):

| FT trip (`outof`) | Count | Share of trips |
|---|---|---|
| 2 FTs | 16,500 | 72.1% |
| 1 FT | 5,824 | 25.5% |
| **3 FTs** | **548** | **2.4%** |

Against 18,010 shooting fouls in the same slice: ~3.0% are 3-FT trips, ~25% are and-1s, ~72% are
2-FT. The sim produces 3-FT trips roughly **12x too often**, and and-1s far too rarely.

> **Caveat for anyone reading existing diagnostics:** every rollout produced before this is fixed
> carries the ~12x 3-FT inflation. Treat FTA and PF numbers from those runs accordingly.

### The fouled player is already in the raw data

The raw `opponent` column is populated on **34,535 of 35,208 foul rows (98.1%)** and appears on
*no other event type*. It is exactly the fouled player — e.g. `shooting / Joel Embiid /
Jayson Tatum`. The 673 empty rows are technicals (607 + 5 + 1) plus 60 blank-type fouls: the cases
where there genuinely is no fouled individual.

`data_cleaner.parse_file` drops the column (`data_cleaner.py:501-505`). Meanwhile
`secondary_player` already exists in the cleaned schema, already shares the player vocab
weight-tied to `player` (`docs/technical_specs.md:227`), and `"none"` is already a vocab special
(`encoder/encoder.py:11`).

### The steal pair — not a double-count, but it exposes a real mismatch

Steals appear as two `turnover` rows: the stealer (`result="steal"`) and the ball-loser
(`result="cop"`). This reads like a double-count and is not one:

- **The stats are correct.** `simulation/box_score.py:263-269` splits on `result` — `steal` counts
  a steal, `cop` counts a turnover. Neither row is counted twice.
- **The sim is faithful to the data.** `data_cleaner.py:396-424` emits the identical pair; the
  same pattern is in the real cleaned data (`data/season2023.csv:29-30`). Steals are 19,167 of
  36,985 real 2022-23 turnovers, so turnover *rows* are inflated **1.518x** over turnover *events*.

The real problem is broader. The event head is trained to predict the next **row** at every
position, but the simulator only queries it at **play-start** positions and then expands a play
into all of its rows atomically. Measured over `data/season2023.csv`: **147,761 of 673,873 rows
(21.9%) are mid-play continuations the sim never asks about.**

| Continuation kind | Rows |
|---|---|
| shot after assist | 66,559 |
| free throw after foul / free throw | 49,710 |
| steal `cop` after steal | 19,167 |
| block after blocked shot | 12,325 |

So ~22% of the event head's training signal sits on positions that never occur at inference, and
its learned marginals are computed over the wrong denominator. This is a plausible root cause for
why `EVENT_BIAS` / `TYPE_BIAS` need hand-tuned corrections that have to be re-tuned after every
train.

---

## Group 1 — Schema: what the event stream should carry

| # | Theory | New data source? | New model? |
|---|---|---|---|
| S1 | Shot zones (expand `shot_type`'s vocabulary) | no — raw x/y already on disk | no — vocab change to an existing head |
| S2 | Free-throw count as a learned outcome | no — raw `outof` already on disk | no — vocab change to an existing head |
| S3 | Fouled player as `secondary_player` | no — raw `opponent` already on disk | no for S3a; new plumbing for S3b |
| S4 | Collapse the steal pair into one row | no | no |
| S5 | Free-throw index (`num` / `outof`) | no — already on disk | no |
| S6 | Player age | **yes** — external roster table | no |
| S7 | Coach | **yes** — external coach table | no — embedding + style priors |

### S1. Shot zones — expand `shot_type`'s vocabulary, fix the scoring consumers

**No new head.** The zone implies 2pt/3pt, so the existing `shot_type` conditional head would swap
its 2-token vocab for a zone vocab. `shot_result` then conditions on zone through the exact pathway
it conditions on type today (`predict_result(shooter, <type token>, ...)`).

**Zone vocabulary (7 tokens, deliberately coarse):** `restricted_area`, `paint` (non-RA),
`short_mid`, `long_mid`, `corner3`, `abovebreak3`, `heave`. Coarse on purpose — per-player-per-zone
make rates need volume, and a bench player's single-season corner-3 sample is already thin. Do not
go finer without evidence.

**Derivation (cleaner).**

- From raw `converted_x/y` (~100% populated in every sampled era: 2002-03, 2012-13, 2022-23), with
  a **fixed geometric rule** (distance bands from the hoop + the corner-3 x-threshold), *not*
  `shot_distance` alone — one definition across all 21 seasons.
- The raw 2pt/3pt marker stays the authority on point value where geometry is ambiguous at the
  arc: if the raw row says 3pt but the derived zone is a 2pt zone (or vice versa), trust the marker
  and snap to the nearest consistent zone. Log the disagreement rate; investigate if >1%.
- Missing/garbage coordinates (rare, <=0.6% in 2022-23): fall back on `shot_distance` + the
  2pt/3pt marker; if both unusable, `short_mid` for 2s / `abovebreak3` for 3s.
- Validation gate before any training: per-zone league-average make rates by season must look like
  known basketball (RA ~62-67%, long_mid ~38-42%, corner3 > abovebreak3, 3PA share rising across
  eras). One diagnostic table, eyeballed once.

**Cleaned schema.** Shot rows: `type` holds the zone token (was `2pt`/`3pt`); `free throw` rows
unchanged. Assist rows: **keep coarse `2pt`/`3pt`** (derived from the assisted shot's zone) — the
assist head does not need zone granularity and its dial (`assist_type: {"3pt": ...}`) stays keyed
as-is. Block/steal/rebound rows unchanged; they see zones through history like everyone else.

**Static lookup replaces string checks.** Add `ZONE_POINTS` / `is_three(zone)` (e.g. in
`simulation/stats.py` or a small `zones.py`) and route every current `"3pt"`/`"2pt"` consumer
through it:

| Consumer | Today | Change |
|---|---|---|
| `simulation/box_score.py:218,236` | `etype == "3pt"` for 3PA/3PM and points | zone lookup |
| `simulation/controller.py:48,59` (`SHOT_TYPES`, `FIELD_GOAL_TYPES`) | `("2pt","3pt")` | zone token tuple |
| `simulation/controller.py:279,325` | score 3 if `"3pt"` | zone lookup |
| `simulation/controller.py:542` | shooting foul -> 3 FTs if `"3pt"` | superseded by S2 |
| `models/game_state_features.py:137` | running score | zone lookup |
| `models/conditional_type_model.py:115` (`shot_type` spec vocab) | `("2pt","3pt")` | zone vocab |
| `simulation/game_simulator.py:641` (allowed-token sets) | `{"2pt","3pt"}` | zone token set |
| `data_cleaner.py:292` | emits `2pt`/`3pt` | emits zone (+ coarse type on assist rows) |
| eFG / shot-mix dials (`config.py`) | keyed 2pt/3pt | re-key per zone |

Tests touching `"3pt"` literals (box_score, controller, input_cache, model_persistence,
predict_game, game_state_features) update alongside.

**Dials.** Re-keying the shot dials per zone is an upgrade, not a cost: zone-frequency and per-zone
make-rate dials replace the blunt eFG knobs, matching how the full1/full2 biases actually
decomposed (rim make rate vs. long-mid frequency are separate problems today squeezed through one
eFG dial). Keep names stable-ish: `SHOT_ZONE_MIX` (dict dial); `SHOT_RESULT_BIAS` gains per-zone
keys.

**Why we think it pays.** `shot_result` learns per-player-per-zone make rates, attacking the
make-rate bias at the source instead of via post-hoc dials. The rebound head sees the zone of the
miss in history (long rebounds off 3s vs. rim misses). Era drift (the 3-point revolution) becomes
learnable as a spatial fact instead of being absorbed into player embeddings. Rollout throughput is
unchanged — same number of sampling steps, a 7-way softmax instead of a 2-way. And it is the direct
answer to the measured `full4-s100` composition problem (eFG 1.7 low, 3PA 2.3 high).

### S2. Free-throw count as a learned outcome

**The problem** is the 3-FT bug documented in the evidence section: the count is decided by a
hard-coded branch reading a sample from a head that was explicitly trained not to answer that
question, and it is wrong by ~12x.

**The theory.** Split the `shooting` foul-type token into **`shooting 2pt` / `shooting 3pt`**,
labelled directly from the following free-throw trip's raw `outof`. The `foul_type` head then
learns the real ~3% 3-FT rate *in real game context* — who is fouling, who is shooting, where in
the game — rather than inheriting the league-wide 3PA share. `controller.py:541-542` stops sampling
`shot_type` and simply reads the decided token.

And-1 stays a structural check on history (`controller.py:530-533`) rather than becoming a token,
so that a four-point play takes its point value from the actual preceding made shot instead of
needing `shooting and1 2pt` / `shooting and1 3pt` tokens for a rare case. The cost of that choice
is that the token split fixes the 3-vs-2 error but *not* the and-1 under-trigger, which is a
separate controller-ordering problem (below).

**Interaction with S1.** If zones land, `shooting 2pt` / `shooting 3pt` should route through
`is_three(zone)` for consistency rather than keeping their own string checks. Carrying the *zone*
of the fouled attempt on the foul token was considered and rejected — 7 zone tokens crossed with
foul types is too sparse, and the fouled attempt is not a logged FGA anyway.

**Stopgaps we know about and are not taking.** Both are inference-side, need no retrain, and are
recorded here so the option is not lost:

1. **And-1 reachability** — defer the made-FG possession flip at `controller.py:280` by one step,
   or let the and-1 guard look back past it, so and-1s can fire at a realistic rate. Worth noting
   this touches the shot/possession flow that every other rate is calibrated against.
2. **A ~0.03 Bernoulli 3-FT dial** replacing the `shot_type` sample at `controller.py:541`,
   matching the existing inference-dial pattern.

### S3. The fouled player, as `secondary_player` on foul rows

**The problem.** Nothing in the cleaned data records *who got fouled*. The sim compensates by
sampling an FT shooter from the offense's five with the generic player head (`_pick_shooter`,
`controller.py:576`) — an independent draw with no connection to the foul that caused it. Drawing
fouls is a real skill (rim pressure, shooting motion, being the guy in the bonus) and the model
currently cannot represent it at all.

The data is already there: raw `opponent`, 98.1% populated on foul rows, empty only on technicals
(see evidence above). Two stages, priced separately because the first is nearly free and the second
is not.

**S3a — carry it through (cheap).** The cleaner reads `row["opponent"]` on foul rows into
`secondary_player`, writing `"none"` where it is empty. Because `secondary_player` is already an
input feature and already weight-tied to `player`, **every head immediately sees the fouled player
through history at zero schema, vocab, or architecture cost.** On the sim side the controller
writes the fouled player into the foul row and reuses that same player as the FT shooter, which
also *removes* the independent `_pick_shooter` draw — for a shooting foul or a bonus foul, the
fouled player and the FT shooter are the same person.

The honest limit: this is passive history context. Nothing yet *conditions* on it within the play,
so the fouled player is still chosen by a generic player-head sample; it just becomes visible to
everything downstream.

**S3b — condition on it (expensive, depends on S3a).** Make `foul_type` condition on the fouled
player, which fixes the causal order: **fouler -> fouled player -> foul type.** Who you foul is a
large part of what makes it a shooting foul rather than a reach-in. This needs
`TypeGenSpec.condition_fields` (`models/conditional_type_model.py:88-121`) to support a
secondary-player condition — it currently understands only `player` and `type` — plus a controller
reorder so the fouled player is sampled before the type. That plumbing is the real cost of the
idea and should be weighed on its own, not smuggled in with S3a.

### S4. Collapse the steal pair into one row

Emit one `turnover` row for a steal, carrying `secondary_player = stealer`, instead of the current
two rows. This is exactly the pattern `block` already uses (`data_cleaner.py:332-348`: the shot row
carries the result, a separate row names the other participant), and it makes `secondary_player`
semantics uniform across block / steal / foul once S3a lands — in every case, *the other player
involved in this play*.

Payoff: removes ~19k phantom turnover rows per season, makes the play-by-play readable (the pair is
genuinely confusing to read even though the box score handles it correctly), and removes 2.8
percentage points of the play-continuation mismatch quantified in M4.

Cost: `simulation/box_score.py:263-269` must count STL from `secondary_player` rather than from a
separate row, and `controller._do_turnover` (`controller.py:328-345`) emits one row instead of two.

Note honestly that this is the *smaller* half of the continuation problem — see M4, which carries
the weight.

### S5. Free-throw index (`num` / `outof`)

Carry the raw `num` / `outof` columns through the cleaner as two integer columns on free-throw
rows; the encoder ingests them as small scalar features. This gives the model first-of-two vs.
second-of-two make rates. (Precisely: `outof` is explicitly dropped at `data_cleaner.py:503`, while
`num` survives the drop but is never read into `output_columns` — `data_cleaner.py` mentions it
nowhere. Both need adding to the emitted free-throw row.)

**This has been promoted.** It used to be the "free rider — first thing to cut if the schema change
gets crowded." It is not optional any more: **raw `outof` is the label that makes S2 possible.**
Without it there is no supervision for `shooting 2pt` vs. `shooting 3pt`. The conditioning value
(FT-sequence make rates) is still expected to be small on its own; the reason to do it is S2.

### S6. Player age

- **Source:** external roster table (Basketball-Reference season rosters: player, team, birth date,
  height, position). One-time scrape/join, cached under `RawData/` as a plain CSV.
- **Feature:** age at `game_date`, one scalar per player per game, normalized by fixed constants
  (clip [18, 45], center ~26). Plumbed exactly like the season-context features
  (`models/season_features.py` pattern: rest / games-played), attached to each player in the
  roster-set encoder so every head sees it wherever player identity is consumed.
- Height/position ride along **if** the join is clean — same table, near-zero marginal cost, helps
  rebound/substitution heads and cold-start players. Drop them without ceremony if they complicate
  the join; age is the item that matters.
- **Name matching is the real work.** Raw data keys players by display name with format drift
  across 21 seasons. Build the join as its own audited step with an exceptions file (manual
  overrides) and a coverage report — target >99% of player-minutes matched; unmatched players get
  the neutral (mean-age) value, never a crash.

### S7. Coach

- **Source:** external coach table (Basketball-Reference): team, season, coach, tenure dates —
  mid-season changes included (row per stint, not per season).
- **Features, two layers:**
  1. **Rolling team style priors** (no external dependency): pace, 3PA rate, FTA rate over each
     team's trailing N games (N~20, from the cleaned data itself, computed in the same pass as
     season context). These carry most of the "system" variance and update mid-season.
  2. **Coach ID embedding** (small, ~8-16 dims), conditioning the rotation-adjacent heads:
     `substitution`, `stint_length`, `event_time`. Rare coaches (<~50 games in-corpus) collapse to
     an UNK-coach token to avoid one-game embeddings.
- The embedding's job is to make the style priors **transfer across a coaching change** — the
  upcoming-season case where the roster carries over but the system flips. Confounding with
  team-era is real; the style priors are the control. If diagnostics show the embedding is just
  memorizing team-season, keep the priors and drop the embedding at inference (it is an input, not
  a head — cheap to ablate).
- Upcoming-season inference needs a way to supply the coach for a matchup — add it to the game
  spec (`extract_game_input`) with a lookup default from the table.

---

## Group 2 — Model & training: how it consumes that stream

| # | Theory | New data source? | New model? |
|---|---|---|---|
| M1 | Game-state features (score / period / clock / team fouls) | no — derived | no — **already plumbed**, needs a train that uses it |
| M2 | Clutch loss weighting | no | no — rides existing `sample_weight` masks |
| M3 | Recency / local-sequence bias | no | backbone change, all 11 heads |
| M4 | Play-boundary loss masking | no | no — rides existing `sample_weight` masks |

### M1. Game-state features — activation, not construction

`models/game_state_features.py` already derives six per-row scalars with train/inference parity by
construction (`GameStateScan` is shared by preprocessing and the simulator):

`score_diff`, `score_total`, `period_idx`, `period_time_left`, `team_fouls_home`, `team_fouls_away`

Fixed-constant normalization (no new `norm_stats` keys), `Dense(16)` projections into each head's
fusion, wiring covered by `test_game_state_features.py` / `test_game_state_wiring.py`. **No new
code.** The only thing missing is a training run whose weights actually consume these inputs —
`v1.0` was trained before activation.

One code touch if S1 lands: the points arithmetic inside `GameStateScan.step`
(`game_state_features.py:137`) reads `etype == "3pt"` and must route through the zone->points
lookup.

Expected effect: end-game behavior (leads sat on, trailing teams fouling, garbage time),
bonus-aware foul value, and better win% in close games — the metric where we still trail the
baseline (47%).

### M2. Clutch loss weighting

Every head already trains with a per-row `sample_weight` mask (PAD / no-next rows zeroed). Clutch
weighting multiplies that mask by a weight >= 1 on rows that are **close and late**:

- Definition (initial): `period_idx >= 3` (Q4/OT) AND `period_time_left <= 300` AND
  `abs(score_diff) <= 8` -> weight `CLUTCH_LOSS_WEIGHT` (config, default ~2.0). All other rows 1.0.
- Implemented once in a shared helper (the raw pre-normalization game-state arrays are available at
  preprocess time), applied in each head's dataset builder where the mask is built.
- `CLUTCH_LOSS_WEIGHT = 1.0` disables it — keep it a config constant so an A/B against the same
  preprocess is trivial.

**Risk:** upweighting late-game rows shifts every head's base rates slightly toward late-game
basketball. Start at 2.0, not higher, and validate that early-game pace/eFG calibration does not
drift (the eval report's per-quarter splits will show it).

### M3. Recency — giving the model a reason to weight the last few plays

**State the situation accurately first:** the backbone is already a 6-layer causal transformer over
up to 600 steps with learned positional embeddings (`docs/technical_specs.md:257-276`). It *can
already see* the last five plays; there is no missing information. What is missing is an
**inductive bias** toward them. With ~600 positions, nothing pushes attention to concentrate
locally, and seven of the heads train on a subset corpus where they may not have the volume to
discover that on their own.

Basketball is strongly locally sequential — a miss is followed by a rebound, a steal by a
transition, a hard foul by a technical — so a recency prior is the kind of structure worth building
in rather than hoping to learn. Options, cheapest first:

1. **ALiBi-style linear distance penalty** on attention logits, with a different slope per head.
   Zero new parameters, a few lines in the attention block. Steep-slope heads become local, gentle
   ones stay global, and the split is learned rather than chosen. Best value for cost.
2. **Dedicate ~2 of 8 attention heads to a hard local window** (e.g. last 8 steps) via the existing
   causal + key-padding mask machinery. Cheap, explicit, and directly encodes "some decisions
   should mostly look at what just happened."
3. **An explicit last-N-events feature block** — concatenate the previous ~5 events'
   event/type/result/player embeddings as an extra Dense-projected input per step. Simple and
   interpretable, but adds parameters and duplicates work attention should be doing.
4. **Play-boundary segment embeddings** — mark each row with its index within the current play and
   the play's index, making "immediately around" explicit rather than positional. Pairs naturally
   with M4.

Recommend trying (1) and (2): both are logit/mask-level changes to the shared backbone with no
schema change. Note that this touches **all 11 heads** and is the one theory here that changes the
architecture rather than the data.

### M4. Play-boundary loss masking

**The problem** is the train/inference mismatch quantified in the evidence section: the event head
is trained to predict the next *row* at every position, but the simulator only ever queries it at
*play-start* positions and expands each play atomically. **21.9% of training positions never occur
at inference**, so the head's learned marginals are computed over the wrong denominator.

**The theory.** Zero the event head's `sample_weight` on continuation rows — any row whose
predecessor belongs to the same play — so it trains on exactly the positions the sim queries. This
rides the same per-row mask machinery M2 uses; it is a dataset-builder change, not a model change.

Determining "same play" is a small, explicit rule set matching what the controller expands
atomically: FT after foul/FT, shot after assist, block after blocked shot, steal `cop` after steal.
It should be derived from the same definition the controller uses, so the two cannot drift.

**Why this may matter more than it looks.** If the event head's marginals are systematically off
because of the denominator, then some of the hand-tuned `EVENT_BIAS` / `TYPE_BIAS` values in
`config.py:76-112` are compensating for a masking artifact rather than a real rate error — which
would explain why they need re-tuning after every train and why fixes in one direction keep
overshooting in the other (the documented full1 -> full2 FTA/PF oscillation). **If this lands, the
dials must be re-measured from zero, not carried forward.**

S4 (collapsing the steal pair) removes 2.8 points of the 21.9%; this removes the rest. They are
complementary, and this is the half that carries the weight.

**The bigger question underneath.** If ~22% of rows exist only to continue a play the model never
gets asked about, that is evidence the row is the wrong unit. The v2-scale version of this theory
is: **should the event stream be one row per *play* rather than one row per transcript row**, with
participants and outcomes as fields instead of as follow-on rows? That would dissolve S4, most of
M4, and much of the controller's expansion logic at once — and it is a far larger change than
anything else in this doc, touching the cleaner, the vocabs, every head, and the box-score decoder.
Recorded as the open question it is, not as a proposal.

---

## If these land: ordering and gotchas

1. **Cleaner changes first**, together — S1 (zone column, assist coarse type), S2 (foul token
   split), S3a (`opponent` -> `secondary_player`), S4 (steal collapse), S5 (`num`/`outof`) — then
   re-clean, then `enrich`, then the external joins (S6 age, S7 coach) landing as additional
   season-context-style columns. These all change the cleaned schema, so they should land in one
   pass or not at all; a re-clean is the expensive step, not the individual edits.
2. **Anything that changes the token language forces a full retrain with `--rebuild-vocabs`** —
   S1 and S2 both do. Every head retrains. Train 2's availability masking and capacity bump carry
   forward unchanged.
3. **Check `data/processed` norm stats before training.** pytest overwrites the committed
   `encoder/vocabs/norm_stats.json` (known pollution issue); a full train with `--rebuild-vocabs`
   refreezes them anyway, but verify the freeze happens from real data rather than test residue.
4. **Training is user-run on the WSL/CUDA side**, as always — e.g.
   `python train.py --full --name <run> --batch-size 64 --clean --rebuild-vocabs`.
5. **Post-train:** dial re-key (per-zone dials), diagnostics pass, then the standard dial-package
   cycle. If M4 landed, re-measure the dials from zero rather than starting from the current
   values.

## Parked

- **Rotation-minutes model / seeded starters** (separate future model). Worth restating that
  per-player minutes miss by 5.66 MAE and nothing in this doc addresses it — this is the largest
  known gap we are choosing not to attack yet.
- **Calibration auxiliary losses** (aggregate-consistency terms per head) — design exists in
  conversation notes only.
- **Relative offense/defense encoding** instead of home/away — needs its own retrain to evaluate.
- **RL / policy-gradient fine-tuning against game MAE** — last resort; reward-hacking risk.

## Open questions

1. **One row per play instead of one row per transcript row** (M4) — the largest question here, and
   the one that would subsume several other theories. Needs its own sketch before it can be priced.
2. **Does S3b earn its plumbing?** S3a is nearly free; S3b needs a new condition type in
   `TypeGenSpec` plus a controller reorder. Is conditioning `foul_type` on the fouled player worth
   that, or does passive history context (S3a) capture most of it?
3. **Which recency mechanism** (M3) — ALiBi slopes, dedicated local heads, both, or neither. Wants
   a cheap ablation rather than an argument.
4. **Exact zone geometry thresholds** (S1) — distance bands, corner-3 x cutoff. Fix during cleaner
   work, gate on the era make-rate validation table.
5. **Where age/height attach in the roster-set encoder** (S6) — per-player scalar concat vs.
   embedding-side projection. Decide at implementation, follow the `season_features` precedent.
6. **`CLUTCH_LOSS_WEIGHT` value** (M2, start 2.0) and whether the clutch window definition should
   be a dial-style config constant for A/B.
7. **How much of the current dial package is compensating for M4's masking artifact** rather than
   for real rate errors. Unknown until M4 is tried; it changes how much of `config.py:76-112`
   should be trusted.
8. Coach table licensing/scrape etiquette for Basketball-Reference (S7) — a manual one-time pull is
   fine at this scale (21 seasons x 30 teams).
