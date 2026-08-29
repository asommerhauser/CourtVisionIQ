# Train 3 Specification

Scope agreed 2026-08-25. Four items, one free rider. Everything here targets the two headline
metrics — per-player box MAE and score MAE — plus year-to-year generalization for upcoming-season
prediction (age, coach).

> **Status 2026-08-29: nothing in this spec has been built yet.** Item 1's plumbing predates the
> spec and is still the only thing in place; there is no `CLUTCH_LOSS_WEIGHT`, no zone vocabulary,
> and no external age/coach tables in the tree. The spec below stands as written.
>
> Two things measured since it was agreed are worth carrying into the work:
>
> - **The score-MAE target moved.** `full4-s100` (100 games × 100 sims) put team points bias at
>   −0.52 and pace bias at +0.47 — the scoring shortfall this spec was partly aimed at is now
>   closed by dials. What remains on the shooting side is the *composition* problem item 3 attacks
>   directly: eFG runs 1.7 points low while 3PA runs 2.3 attempts high. That strengthens the case
>   for shot zones and weakens the case for further blunt eFG dialing.
> - **Per-player minutes are now scored**, and they miss by **5.66 minutes** (MAE, `full4-s100`).
>   Minutes multiply every per-player counting stat, so this is plausibly the largest single term
>   in the per-player box MAE this train is targeting — and nothing in this spec addresses it. The
>   rotation-minutes model stays parked below, but that parking is now a measured trade rather
>   than an assumed one.

| # | Item | New data source? | New model? | Status |
|---|------|------------------|-----------|--------|
| 1 | Game-state features (score / period / clock / team fouls) | no (derived) | no | **Plumbing already built** — activates with this train |
| 2 | Clutch loss weighting | no | no | To build (small; rides on existing `sample_weight` masks) |
| 3 | Shot zones (shot_type vocab expansion) | no (raw x/y already on disk) | no — vocab change to the existing shot_type head | To build |
| 4 | Player age | **yes** (external roster table) | no | To build |
| 5 | Coach | **yes** (external coach table) | no (embedding + style priors) | To build |
| — | FT index (`num`/`outof`) — free rider | no (raw already on disk) | no | Carry through cleaner only |

This is a **full retrain with a vocab rebuild** (`--rebuild-vocabs`): item 3 changes the shared
token language, so every head retrains. Train 2's availability masking and capacity bump carry
forward unchanged.

---

## 1. Game-state features — activation only

`models/game_state_features.py` already derives six per-row scalars with train/inference parity
by construction (`GameStateScan` is shared by preprocessing and the simulator):

`score_diff`, `score_total`, `period_idx`, `period_time_left`, `team_fouls_home`, `team_fouls_away`

Fixed-constant normalization (no new `norm_stats` keys), `Dense(16)` projections into each head's
fusion, wiring covered by `test_game_state_features.py` / `test_game_state_wiring.py`. **No new
code.** Train 3 is simply the first run where trained weights consume these inputs.

One code touch: the points arithmetic inside `GameStateScan.step`
(`game_state_features.py:137`) reads `etype == "3pt"` and must route through the zone→points
lookup from item 3 (see the consumer list there).

Expected effect: end-game behavior (leads sat on, trailing fouls, garbage time), bonus-aware foul
value, and better win% in close games — the metric where we still trail the baseline (47%).

## 2. Clutch loss weighting

Every head already trains with a per-row `sample_weight` mask (PAD/no-next rows zeroed). Clutch
weighting multiplies that mask by a weight ≥ 1 on rows that are **close and late**:

- Definition (initial): `period_idx >= 3` (Q4/OT) AND `period_time_left <= 300` AND
  `|score_diff| <= 8` → weight `CLUTCH_LOSS_WEIGHT` (config, default ~2.0). All other rows 1.0.
- Implemented once in a shared helper (the raw pre-normalization game-state arrays are available
  at preprocess time), applied in each head's dataset builder where the mask is built.
- `CLUTCH_LOSS_WEIGHT = 1.0` disables it — keep it a config constant so an A/B against the same
  preprocess is trivial.

Risk note: upweighting late-game rows shifts every head's base rates slightly toward late-game
basketball. Start at 2.0, not higher; validate that early-game pace/eFG calibration doesn't drift
(the eval report's per-quarter splits will show it).

## 3. Shot zones — expand shot_type's vocabulary, fix the scoring consumers

**No new head.** The zone implies 2pt/3pt, so the existing shot_type conditional head swaps its
2-token vocab for a zone vocab. shot_result then conditions on zone through the exact pathway it
conditions on type today (`predict_result(shooter, <type token>, ...)`).

### Zone vocabulary (7 tokens, deliberately coarse)

`restricted_area`, `paint` (non-RA), `short_mid`, `long_mid`, `corner3`, `abovebreak3`, `heave`

Coarse on purpose: per-player-per-zone make rates need volume; a bench player's single-season
corner-3 sample is already thin. Do not go finer without evidence.

### Derivation (cleaner)

- From raw `converted_x/y` (~100% populated in every sampled era: 2002-03, 2012-13, 2022-23),
  with a **fixed geometric rule** (distance bands from the hoop + the corner-3 x-threshold), NOT
  `shot_distance` alone — one definition across all 21 seasons.
- The raw 2pt/3pt marker stays the authority on point value where geometry is ambiguous at the
  arc: if the raw row says 3pt but the derived zone is a 2pt zone (or vice versa), trust the raw
  marker and snap to the nearest consistent zone. Log the disagreement rate; investigate if >1%.
- Missing/garbage coordinates (rare, ≤0.6% in 2022-23): fall back on `shot_distance` + the
  2pt/3pt marker; if both unusable, `short_mid` for 2s / `abovebreak3` for 3s.
- Validation gate before training: per-zone league-average make rates by season must look like
  known basketball (RA ~62-67%, long_mid ~38-42%, corner3 > abovebreak3, 3PA share rising across
  eras). One diagnostic table, eyeballed once.

### Cleaned schema

- Shot rows: `type` holds the zone token (was `2pt`/`3pt`). `free throw` rows unchanged.
- Assist rows: **keep coarse `2pt`/`3pt`** (derived from the assisted shot's zone). The assist
  head doesn't need zone granularity and its dial (`assist_type: {"3pt": ...}`) stays keyed as-is.
- Block/steal/rebound rows: unchanged (they see zones through history like everyone else).

### Static lookup replaces string checks

Add `ZONE_POINTS` / `is_three(zone)` (e.g. in `simulation/stats.py` or a small `zones.py`) and
route every current `"3pt"`/`"2pt"` consumer through it:

| Consumer | Today | Change |
|---|---|---|
| `simulation/box_score.py:218,236` | `etype == "3pt"` for 3PA/3PM and points | zone lookup |
| `simulation/controller.py:48,59` (`SHOT_TYPES`, `FIELD_GOAL_TYPES`) | `("2pt","3pt")` | zone token tuple |
| `simulation/controller.py:279,325` | score 3 if `"3pt"` | zone lookup |
| `simulation/controller.py:542` | shooting foul → 3 FTs if `"3pt"` | `is_three(zone)` |
| `models/game_state_features.py:137` | running score | zone lookup |
| `models/conditional_type_model.py:115` (shot_type spec vocab) | `("2pt","3pt")` | zone vocab |
| `simulation/game_simulator.py:641` (allowed-token sets) | `{"2pt","3pt"}` | zone token set |
| `data_cleaner.py:292` | emits `2pt`/`3pt` | emits zone (+ coarse type on assist rows) |
| eFG / shot-mix dials (`config.py`) | keyed 2pt/3pt | re-key per zone (see below) |

Tests touching `"3pt"` literals (box_score, controller, input_cache, model_persistence,
predict_game, game_state_features) update alongside.

### Dials

Re-keying the shot dials per zone is an upgrade, not a cost: zone-frequency and per-zone
make-rate dials replace the blunt eFG knobs, matching how the full1/full2 biases actually
decomposed (rim make rate vs. long-mid frequency are separate problems today squeezed through
one eFG dial). Keep the dial names stable-ish: `SHOT_ZONE_MIX` (dict dial), `SHOT_RESULT_BIAS`
gains per-zone keys.

### Payoffs

- shot_result learns per-player-per-zone make rates — attacks the make-rate bias at the source
  instead of via post-hoc dials.
- Rebound head sees the zone of the miss in history (long rebounds off 3s vs. rim misses).
- Era drift (3-point revolution) becomes learnable as a spatial fact instead of being absorbed
  into player embeddings.
- Rollout throughput unchanged: same number of sampling steps, 7-way softmax instead of 2-way.

## 4. Player age

- **Source**: external roster table (Basketball-Reference season rosters: player, team, birth
  date, height, position). One-time scrape/join step, cached under `RawData/` as a plain CSV.
- **Feature**: age at `game_date`, one scalar per player per game, normalized by fixed constants
  (e.g. clip [18, 45], center ~26). Plumbed exactly like the season-context features
  (`models/season_features.py` pattern: rest / games-played), attached to each player in the
  roster-set encoder so every head sees it wherever player identity is consumed.
- Height/position ride along **if** the join is clean — same table, near-zero marginal cost,
  helps rebound/substitution heads and cold-start players. Cut them without ceremony if they
  complicate the join; age is the committed item.
- **Name matching is the real work**: raw data keys players by display name with format drift
  across 21 seasons. Build the join as its own audited step with an exceptions file
  (manual overrides), and a coverage report — target >99% of player-minutes matched; unmatched
  players get the neutral (mean-age) value, never a crash.

## 5. Coach

- **Source**: external coach table (Basketball-Reference): team, season, coach, tenure dates —
  mid-season changes included (row per stint, not per season).
- **Features**, two layers:
  1. **Rolling team style priors** (no external dependency): pace, 3PA rate, FTA rate over each
     team's trailing N games (N≈20, from the cleaned data itself, computed in the same pass as
     season context). These carry most of the "system" variance and update mid-season.
  2. **Coach ID embedding** (small, ~8-16 dims), conditioning the rotation-adjacent heads:
     substitution, stint_length, event_time. Rare coaches (< ~50 games in-corpus) collapse to an
     UNK-coach token to avoid one-game embeddings.
- The embedding's job is to make the style priors **transfer across a coaching change** — the
  upcoming-season case where the roster carries over but the system flips. Confounding with
  team-era is real; the style priors are the control. If diagnostics show the embedding is just
  memorizing team-season, keep the priors and drop the embedding at inference (it's an input,
  not a head — cheap to ablate).
- Upcoming-season inference needs a way to supply the coach for a matchup — add it to the game
  spec (`extract_game_input`) with a lookup default from the table.

## FT index (free rider)

Cleaner carries raw `num`/`outof` through as two integer columns on free-throw rows (currently
dropped at `data_cleaner.py:503`). Encoder ingests them as small scalar features. The sim's
controller already enforces FT structure (`_resolve_foul` / `_free_throws` hard-code counts and
last-FT possession), so this is *conditioning only* — first-of-two vs. second-of-two make rates.
Zero new logic beyond the two columns; expected gain small. If the schema change gets crowded,
this is the first thing to cut.

## Run plan

1. Cleaner changes (zone column, assist coarse type, FT num/outof) → re-clean → `enrich` →
   external joins (age, coach) land as additional season-context-style columns.
2. **Check `data/processed` norm stats before training** — pytest overwrites the committed
   encoder/vocabs/norm_stats.json (known pollution issue); a full train with `--rebuild-vocabs`
   refreezes them anyway, but verify the freeze happens from real data, not test residue.
3. Full train: `python train.py --full --name full_train_3 --batch-size 64 --clean
   --rebuild-vocabs` on the WSL/CUDA side (user-run, as always). All heads retrain (new vocab
   language). Availability masking + capacity settings from train 2 carry forward.
4. Post-train: dial re-key (zone dials), diagnostics pass, then the standard dial-package cycle —
   or the automated dial optimizer if it exists by then.

## Explicitly out of scope (parked)

- Calibration auxiliary losses (aggregate-consistency terms per head) — candidate for this train
  if capacity allows, but not committed; design exists in conversation notes only.
- Rotation-minutes model / seeded starters (separate future model).
- Relative offense/defense encoding (needs its own retrain evaluation).
- RL / policy-gradient fine-tuning against game MAE (last resort; reward-hacking risk).

## Open questions

1. Exact zone geometry thresholds (distance bands, corner-3 x cutoff) — fix during cleaner work,
   gate on the era make-rate validation table.
2. Where age/height attach in the roster-set encoder (per-player scalar concat vs. embedding-side
   projection) — decide at implementation, follow the season_features precedent.
3. Coach table licensing/scrape etiquette for Basketball-Reference — manual one-time pull is
   fine at this scale (21 seasons × 30 teams).
4. `CLUTCH_LOSS_WEIGHT` value (start 2.0) and whether the clutch window definition should be a
   dial-style config constant for A/B.
