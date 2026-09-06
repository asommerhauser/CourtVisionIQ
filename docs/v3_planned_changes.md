# CourtVisionIQ 3.0 — Parked Features

> Things deliberately cut from 2.0, with the reasoning, so the design is not lost and the decision
> is not re-litigated from scratch. Nothing here is scheduled. See
> [`v2_planned_changes.md`](v2_planned_changes.md) for what is actually being built.

Each item below was scoped during 2.0 planning and cut for a stated reason. Two of the three need
an external data source, which is the main thing separating them from everything in 2.0 — every
2.0 feature is derivable from data already on disk.

---

## 1. Free-throw index — **pending review, not committed**

**What it would be.** The raw `num` / `outof` columns carried onto free-throw rows as two small
integer features the encoder ingests, so the model conditions on first-of-two versus second-of-two.

**Why it is parked.** Free-throw outcomes within a trip are sequential: the second attempt should
depend on what happened on the first, not on a flat "out of 2" label. If the local attention heads
in [2.0 §9](v2_planned_changes.md#9-local-context) work as intended, the model should pick this up
from the event sequence itself — the previous free throw is one row back, well inside the eight-row
band. Adding an explicit feature first would mask whether local attention is doing its job.

**Review trigger.** After the 2.0 train, check free-throw make rates split by position in the trip
(1-of-2, 2-of-2, 1-of-3, and so on) against real rates. If the model is flat across positions where
the real data is not, local attention did not capture it and this feature earns its place.

**Cost if it comes back.** Small. Two integer columns in the cleaner and two scalar inputs in the
encoder. The sim's controller already enforces free-throw structure (`_resolve_foul` / `_free_throws`
hard-code the counts and last-attempt possession), so this is conditioning only — no new sim logic.

**Do not confuse this with 2.0 §4.** [Learned free-throw counts](v2_planned_changes.md#4-learned-free-throw-counts)
also reads `outof`, but in the **cleaner**, to label each shooting foul as `shooting 2pt` /
`shooting 3pt`. That is a labeling step feeding the foul-type head, not a model input, and it **is**
in 2.0. Only the per-row feature is parked.

---

## 2. Player age

**What it would be.** Age at `game_date`, one scalar per player per game, normalized by fixed
constants (clip roughly [18, 45], centered near 26), attached to each player in the roster-set
encoder so every head sees it wherever player identity is consumed.

**Why it is parked.** It needs an external roster table, and the join is the real work — not the
feature. Nothing in the repo does an external join today.

**Source.** Basketball-Reference season rosters (player, team, birth date). A one-time manual pull
is fine at this scale — 21 seasons × 30 teams — cached under `RawData/` as a plain CSV. Do not
scrape without deciding that explicitly.

**The hard part: name matching.** The pipeline has **no player IDs anywhere**. Players are keyed by
display-name string end to end: the raw `player` / `assist` / `block` / `steal` / `h1..h5` /
`a1..a5` columns, the cleaned `player` and `roster_*` columns, and `encoder/encoder.py:33`'s single
name-keyed `player_vocab`. So the join key is `(display_name, season)` or
`(display_name, team, season)`, against 21 seasons of format drift.

Build it as its own audited step: a manual-overrides exceptions file, and a coverage report
targeting >99% of player-minutes matched. Unmatched players get the neutral mean-age value and
**never** a crash.

**Where it attaches.** Follow the `rest_home` / `rest_away` precedent exactly — a roster-parallel
list column produced in `season_context.enrich_df`, plumbed by `models/season_features.py`
(`REST_LIST_COLS` at `:28`), fused into the player embedding in `models/roster_set_encoder.py:120`
via a `Dense(roster_dim)` projection alongside `rest_proj`. Note `RosterEncoderParams` is a frozen
dataclass with a `get_config` / `from_config` round-trip (`:131-179`) — any new param must be added
to both or weight reload breaks. Inference side: `simulation/game_input.py` needs `home_age` /
`away_age` maps mirroring `home_rest` / `away_rest`, and `simulation/input_cache.py:142-156`
picks them up.

**Explicitly rejected riders.** Player **height** and **position** come free with the same join and
were considered. Both are **dropped permanently** — not in 2.0 and not planned here. Position would
also need its own vocab and an `Embedding` rather than a `Dense` projection, and position labels are
noisy across 21 seasons.

---

## 3. Coach

**What it would be.** Two layers: rolling team style priors, and a coach ID embedding.

**Why it is parked.** The embedding is confounded with team-era, and — the larger problem — the
model has **no team identity at all** today. `GameInput` (`simulation/game_input.py:40-53`) carries
rosters, season, playoff flag and season context; no team ID, no team name, no game date. Team
labels like `home_team="HOME"` are display-only strings that never reach the model. A coach
embedding would be the first team-level categorical input the architecture has ever had: new vocab,
new encoder field, new `GameInput` field, new simulator context plumbing, and cache invalidation.

**The cheap half, if this is ever revisited.** Layer 1 — rolling team style priors (pace, 3PA rate,
FTA rate over each team's trailing ~20 games) — needs **no external data at all**. It is computed
from the cleaned data in the same pass as season context, and slots straight into the existing
`TEAM_SCALAR_COLS` pattern (`models/season_features.py:30`, projected by `season_team_projections`
at `:198`). These priors carry most of the "system" variance and update mid-season. If coach work
ever restarts, start here and measure before touching the embedding.

**The expensive half.** Layer 2 — a small coach ID embedding (~8-16 dims) conditioning the
rotation-adjacent heads (substitution, stint_length, event_time), with rare coaches (<~50 games
in-corpus) collapsed to an UNK-coach token. Source would be a Basketball-Reference coach table with
tenure dates, one row per stint so mid-season changes are represented.

The embedding's only real job is to make the style priors **transfer across a coaching change** —
the upcoming-season case where the roster carries over but the system flips. The priors are the
control: if diagnostics show the embedding is just memorizing team-season, keep the priors and drop
it. It is an input, not a head, so it is cheap to ablate at inference.

**Also required.** Upcoming-season inference needs a way to supply the coach for a matchup — a field
on the game spec with a lookup default from the table.

---

## Also parked (unchanged from 2.0)

| Item | Reason |
|---|---|
| One row per play | Would subsume 2.0's pair collapse and loss masking, but touches every head and the decoder |
| Conditioning foul type on the fouled player | Needs a new condition type in the spec machinery; decide after 2.0's passive version is measured |
| ALiBi recency | Needs a custom attention layer; 2.0's masked heads get the bias without one |
| A dedicated minutes model | 2.0's rotation model is the first real attempt at the minutes gap; decide after |
| Calibration auxiliary losses | Aggregate-consistency terms per head; design exists in conversation notes only |
| Relative offense/defense encoding | Offense/defense frame instead of home/away; needs its own full-retrain evaluation |
| RL / policy-gradient fine-tuning against game MAE | Last resort; reward-hacking risk |
