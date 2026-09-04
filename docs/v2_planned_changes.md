# CourtVisionIQ 2.0 — Feature Plan

> What we are building for version 2.0, organized by workstream. Each workstream states its
> goal, the design we settled on, the work items, and what "done" means. Sequencing is in
> [Milestones](#milestones). Nothing here is built yet.

---

## Overview

Version 2.0 is one re-clean of the data, one full retrain, and a rebuilt rule engine. The theme
is the same across every workstream: **the model learns what the controller currently
hard-codes, and the controller enforces the basketball rules it currently does not know.**

| # | Workstream | Delivers | Needs the retrain? |
|---|---|---|---|
| 1 | [Shot location](#1-shot-location) | Seven court zones instead of 2pt / 3pt; per-zone make rates and dials | yes |
| 2 | [Fouls and free throws](#2-fouls-and-free-throws) | Learned 2-vs-3 free-throw count, the fouled player, side-aware foul typing, correct FT attribution, reachable and-1s | yes |
| 3 | [Ball state and possession](#3-ball-state-and-possession) | Dead-ball state, one possession tracker, team rebounds, timeouts, clean period boundaries | partly |
| 4 | [Rotation](#4-rotation) | Substitutions only at dead balls, decided by a model that sees stint time, minutes, fouls and bench rest | yes |
| 5 | [Local context](#5-local-context) | Attention heads focused on the last few plays; a shot-clock proxy | yes |
| 6 | [Training](#6-training) | Game-state features finally trained, clutch weighting, play-boundary loss masking | yes |
| 7 | [Schema cleanup](#7-schema-cleanup) | One row per play for steals and offensive fouls; standalone technicals; raw columns kept | yes |
| 8 | [Rule audit](#8-rule-audit) | A checker that gates every other workstream | no |

Ids used below: **R** = a controller rule (hard limit on sampling), **D** = a data / schema
change, **F** = a model or training change, **T** = tooling.

---

## 1. Shot location

**Goal.** Make shot selection and shot efficiency spatial, so per-player-per-zone make rates are
learned instead of squeezed through one eFG dial, and so the rebound head can see where a miss
came from.

**Design.**

- `shot_type`'s vocabulary becomes seven coarse zones: `restricted_area`, `paint`, `short_mid`,
  `long_mid`, `corner3`, `abovebreak3`, `heave`. Coarse on purpose — a bench player's single-season
  corner-3 sample is already thin.
- Zones are derived in the cleaner from the raw shot coordinates with one fixed geometric rule
  (distance bands from the hoop plus the corner-3 x-threshold), the same across all 21 seasons.
  The raw 2/3 marker wins where geometry is ambiguous at the arc; missing coordinates fall back on
  `shot_distance` plus the marker.
- Assist rows keep coarse `2pt` / `3pt`. The forced made shot after an assist samples a zone
  masked to the assist's class.
- A static lookup (`ZONE_POINTS`, `is_three`) replaces every `"3pt"` string check in the code.
- Shot dials re-key per zone: a zone-mix dial and per-zone make-rate biases replace the blunt eFG
  knobs.

**Work.**

- [ ] D1 · Cleaner: keep `converted_x/y` and `shot_distance` at load; derive the zone; emit it on
      shot rows — `data_cleaner.py`
- [ ] D1 · New `simulation/zones.py` with the lookup; route consumers through it —
      `simulation/box_score.py`, `simulation/controller.py`, `models/game_state_features.py`,
      `models/conditional_type_model.py`, `simulation/game_simulator.py`
- [ ] D1 · Assist-then-shot: zone sample masked to the assist's 2/3 class — `simulation/controller.py`
- [ ] D1 · Re-key `SHOT_RESULT_BIAS` per zone; add `SHOT_ZONE_MIX` — `config.py`
- [ ] D1 · Tests touching `"3pt"` literals updated alongside

**Done when.** A per-zone, per-era make-rate table from the cleaned data looks like known
basketball (restricted area high-60s, long mid ~40%, corner 3 above break 3, 3PA share rising
across eras), and every consumer of shot type goes through the lookup.

---

## 2. Fouls and free throws

**Goal.** Every foul resolves to a legal, complete outcome — the right type for the fouler's
side, the right number of free throws, shot by the right player on the right team — and the
model learns the parts of that which are learnable.

**Design.**

- *Learned free-throw count.* The `shooting` foul token splits into `shooting 2pt` /
  `shooting 3pt`, labelled from the raw free-throw index (`num` / `outof`) carried onto FT rows.
  The controller reads the token; it stops sampling a shot type to guess the count.
- *The fouled player.* Raw `opponent` (populated on nearly every foul) becomes the foul row's
  `secondary_player`. The controller writes it and uses that player as the free-throw shooter,
  removing the independent shooter draw. Conditioning foul type on the fouled player is a later
  step, not 2.0.
- *Side-aware foul typing.* After the fouler is sampled, the controller knows which side they
  are on and masks the type head:

  | Fouler on | Allowed types | Resolves to |
  |---|---|---|
  | offense | `offensive`, `loose ball`, `technical`, `flagrant-1`, `flagrant-2` | `offensive` → turnover; `loose ball` → possession unchanged, team foul; flagrant / technical → FTs to the defense |
  | defense | everything except `offensive` | common foul → offense retains, or 2 FTs in the penalty; shooting / take / flagrant / technical as today |

- *Free throws go to the fouler's opponent* in every branch — shooting, bonus, take, flagrant,
  technical. Team is resolved from a fixed map built at tip-off, never from who is currently on
  the floor, and it is resolved *before* the foul is charged.
- *And-1 becomes reachable.* The controller remembers the scorer of the last made basket; a
  foul sampled as the very next play by the scored-on team is typed against the pre-flip side
  and, if shooting, resolves as an and-1: one free throw to the scorer at zero gap.
- *Bonus rules.* Team fouls use the single definition already in the feature scan (all fouls but
  technical and offensive, either side). The last-two-minutes penalty triggers on the second foul
  committed inside the window, tracked per team and reset per period.

**Work.**

- [ ] D4 · Cleaner: keep `num` / `outof`; emit them on FT rows; split the shooting token —
      `data_cleaner.py`
- [ ] D5 · Cleaner: `opponent` → `secondary_player` on foul rows (`none` when empty) — `data_cleaner.py`
- [ ] R4 · Side-aware type mask and the offense-side resolver branch — `simulation/controller.py`
- [ ] R5 · Fixed team map; FT team = fouler's opponent everywhere; reorder the shooting-foul
      handler — `simulation/controller.py`
- [ ] R6 · And-1 window; last-two-minutes rule; import the shared team-foul set —
      `simulation/controller.py`, `models/game_state_features.py`
- [ ] D4 · Read the FT count from the token; delete the shot-type guess — `simulation/controller.py`
- [ ] D5 · Fouled player as FT shooter; delete the independent shooter draw — `simulation/controller.py`
- [ ] · Rename `TYPE_BIAS` foul keys with the split — `config.py`

**Done when.** No `offensive` foul by a defender; no `personal`, `shooting` or take foul by the
offense; no foul resolves to nothing while the fouling team is in the penalty; the free-throw
shooting team is never the fouling team; three-free-throw trips run near their real ~3% share.

---

## 3. Ball state and possession

**Goal.** The controller knows when the ball is dead and who has it, with one authoritative
tracker, and every change of possession in the stream is visible to the model.

**Design.**

- *Dead-ball state.* A controller-internal flag, never written to the data, set by every play:

  | Dead after | Live after |
  |---|---|
  | any foul; a non-steal turnover; the last free throw if made; a team rebound; a timeout; a period boundary; a made basket in the last minute of Q1–Q3 or last two minutes of Q4 / OT | a made basket otherwise; a missed or blocked shot; a live rebound; a steal; the last free throw if missed |

- *One possession tracker.* The simulator's second copy and its flip-on-result rule are
  deleted; the controller's tracker is the only one. Possession is a sampling guide, not a data
  column and not a model input. Opening possession is a coin flip; Q2 and Q3 go to the tip loser,
  Q4 to the winner, overtime re-flips.
- *Team rebounds as rows.* `rebound / none / <offensive|defensive>`, team from the raw `team`
  column. The rebound-type head grows to four tokens and the controller skips the rebounder pick
  on a team token. The dead-ball-rebound dial goes away; the head learns the share.
- *Timeouts as a sampled event.* `timeout / none / <home|away>`. The calling team rides in the
  `type` field, predicted by a seventh conditional type head — one line in the existing spec
  table. Timeouts join the event menu, masked to dead balls or the team in possession, under a
  per-team budget (7 per game, 4 in Q4, 2 in the last three minutes, +2 per overtime). This is
  the real path to a substitution after a made basket.
- *Periods are real cuts.* If a sampled gap would cross the period end, the clock stops at the
  boundary, team fouls reset, the ball is dead, and the play is re-sampled in the new period.
  No event straddles a boundary. Nothing is logged.

**Work.**

- [ ] R1 · Dead-ball flag set by every handler — `simulation/controller.py`
- [ ] R6 · Delete the simulator's possession copy; coin-flip tip and the period rule —
      `simulation/game_simulator.py`, `simulation/controller.py`
- [ ] D2 · Cleaner keeps team rebounds; four-token rebound head; controller team-rebound path;
      team line in the box score — `data_cleaner.py`, `models/conditional_type_model.py`,
      `simulation/controller.py`, `simulation/box_score.py`
- [ ] D3 · Cleaner keeps timeouts; `timeout_type` spec head; event-menu mask and budget —
      `data_cleaner.py`, `models/conditional_type_model.py`, `simulation/controller.py`, `config.py`
- [ ] R7 · Boundary cut at every clock-advance site; clamp the minutes credit — `simulation/controller.py`
- [ ] R6 · Loose-ball result `op`; offensive foul counted as a turnover — `simulation/controller.py`,
      `simulation/box_score.py`

**Done when.** Every substitution follows a dead-ball row; every missed shot is followed by
exactly one rebound row or a possession-ending row; a forward possession scan of any sim is
consistent at every row; no event straddles a period boundary; timeouts stay within budget.

---

## 4. Rotation

**Goal.** Substitutions happen only when they legally can, at the rate and with the players a
real rotation produces, decided by a model that can see fatigue, foul trouble and bench rest.

**Design.**

- *Per-player live state, bundled like rest days.* The roster encoder already takes one per-slot
  scalar (days of rest). Add `stint_seconds`, `minutes_played` and `personal_fouls` per on-court
  slot, derived by the same scan that builds game state. Every head sees them wherever it
  consumes the lineup — a five-foul player fouls less, a tired one shoots worse.
- *A bench bundle.* A second set input of up to ten available bench players with
  `bench_seconds` (since exit, or since tip-off if unused), `minutes_played`, `personal_fouls`
  and `has_played`, encoded by the same set encoder. The incoming pick keeps its vocab-sized
  output and availability mask, plus a per-candidate additive score computed from those slot
  scalars by a small shared MLP. "He sat down nine seconds ago" becomes a learned penalty.
- *A `sub_decision` head, asked only at dead balls.* At each dead ball — computed identically on
  the cleaned stream — the target is, per team, how many substitutions follow before the next
  live-ball row (`0 / 1 / 2 / 3+`). Trained on dead-ball positions only, so it is never asked
  anywhere it did not learn. At a dead ball the controller samples the count per team, then
  for each sub picks the outgoing player (player head over the five) and the incoming player
  (substitution head over the bench).
- *Retired.* The stint-length head and scheduler, the fatigue nudge, and the stint dials. The
  per-team maximum gap stays as the single cadence backstop. A hard minimum bench time exists as
  a dial but ships **off**; it is a fallback if diagnostics still show churn.
- *Fallback.* If the decision head is judged too large for 2.0: keep the stint scheduler with
  dead-ball gating and land the two bundles alone. Bench rest is still learned.

**Work.**

- [ ] F5a · Live-state derivation in the scan; roster-bundle plumbing —
      `models/game_state_features.py`, `models/season_features.py`, `models/roster_set_encoder.py`
- [ ] F5b · Bench slot construction from the available set; bench encoder; per-candidate score —
      `simulation/game_simulator.py`, `models/substitution_model.py`, `models/roster_set_encoder.py`
- [ ] F5c · New `sub_decision` head with dead-ball targets; registry entry — `models/`,
      `models/registry.py`
- [ ] R2 · Controller rotation loop at dead balls; delete the scheduler and the legacy sub path —
      `simulation/controller.py`
- [ ] R3 · `SUB_MIN_BENCH_SECONDS` (default 0) and dial cleanup — `config.py`

**Done when.** Subs per game, re-entry bench time and stint length match the real distributions
in the eval report, no substitution occurs on a live ball, and per-player minutes MAE moves off
its current 5.66.

---

## 5. Local context

**Goal.** Give the model a built-in reason to weight the last few plays, and one signal it
cannot easily compute for itself.

**Design.**

- *Local attention heads.* In every backbone block, two of the eight heads are restricted to the
  last eight rows by a banded boolean mask (the same mechanism as the padding mask; the framework
  ANDs it with the causal mask). Six heads stay global; outputs concatenate to the same width.
  No new inputs, no custom kernel. `LOCAL_ATTN_HEADS = 0` disables it for an A/B. The backbone
  loop is duplicated across six model files, so it is extracted into one shared builder first.
- *A shot-clock proxy.* One per-row scalar: seconds since the current possession started, reset
  on a change of possession or an offensive rebound, clipped and scaled to the 24-second clock.
  Derived inside the game-state scan (the scan tracks possession internally to compute it;
  possession itself is not exposed). Shot-clock pressure drives shot selection, shot-clock
  turnovers and the timing of the next event, and it is hard for attention to reconstruct. It is
  one key to remove if diagnostics say it does nothing.
- *Not chosen.* An explicit last-five-events input block is the fallback if the local heads show
  nothing; ALiBi needs a custom attention layer and waits.

**Work.**

- [ ] F1 · Extract `backbone_blocks()`; add the banded mask layer and the head split;
      `LOCAL_ATTN_HEADS` / `LOCAL_ATTN_WINDOW` — `models/*_model.py`, `config.py`
- [ ] F2 · `poss_seconds` in the scan, its normalization constant, and the feature test —
      `models/game_state_features.py`, `tests/test_game_state_features.py`

**Done when.** The head split reproduces the old graph at zero local heads, an ablation shows a
measurable difference, and the scan's incremental path is bit-identical to preprocessing.

---

## 6. Training

**Goal.** Train the heads on the positions the simulator actually queries, with the signal it
actually has, and weight the moments that decide games.

**Design.**

- *Game-state activation.* Score difference, score total, period, time left and team fouls are
  already plumbed and tested; 2.0 is the first train whose weights consume them.
- *Clutch weighting.* Multiply each head's loss mask by `CLUTCH_LOSS_WEIGHT = 2.0` on rows that
  are Q4 or overtime, under five minutes left, within eight points. Start at 2.0 and watch the
  per-quarter splits for drift.
- *Play-boundary and forced-row masking.* Zero the event and conditional-time heads' loss on
  rows the simulator never asks about: continuation rows (free throw after foul or free throw,
  shot after assist, block after blocked shot), controller-forced rows (substitutions), and for
  the time head the last row before a period boundary. The continuation rule is one function
  shared with the controller's play expansion so the two cannot drift.
- *Dials from zero.* After this train the calibration dials are re-measured from scratch, not
  carried forward — some of the current values are compensating for the masking artifact.

**Work.**

- [ ] F3 · Clutch helper beside the recency weighting; apply in each dataset builder —
      `models/season_features.py`, `models/*_model.py`
- [ ] F4 · Shared continuation rule; masks in the event and time heads — `models/event_time_model.py`,
      `models/conditional_time_model.py`, `simulation/controller.py`
- [ ] · Fresh dial package after the first 2.0 eval — `config.py`

**Done when.** Late-game behavior shows in the eval report (leads held, trailing teams fouling),
close-game win% improves on the baseline, and the event-mix dials needed after the train are
smaller than today's.

---

## 7. Schema cleanup

**Goal.** One row per play where the data currently has two, and no raw information dropped that
2.0 needs.

**Design.**

- Steals become one turnover row carrying the stealer as `secondary_player`; offensive fouls
  become one foul row with no trailing turnover row. The box score counts steals from
  `secondary_player` and turnovers from both. Matches the pattern blocks already use.
- The separate raw technical-foul event (defensive three seconds, double technical, coach
  technical) becomes a normal technical foul row where a player is named.
- Not kept: jump balls (a coin flip plus the period rule covers possession), period-end rows
  (dead-ball state stays in the controller), possession as a column, raw ejection and violation
  rows (already represented by flagrant-2 and turnover rows). The playoff flag is correct as is.

**Work.**

- [ ] D6 · Steal and offensive-foul collapse; box-score counting — `data_cleaner.py`,
      `simulation/box_score.py`, `simulation/controller.py`
- [ ] D7 · Technical-foul rows — `data_cleaner.py`
- [ ] · Stop dropping `outof`, `converted_x/y`, `shot_distance` at load; side for `none`-player
      rows from the raw team column — `data_cleaner.py`
- [ ] · Cleaner tests for every new row type and both collapses — `tests/test_data_cleaner.py`

**Done when.** The cleaned 2022-23 season has no two-row steals or offensive fouls, every new row
type appears at its real rate, and the cleaner run is idempotent.

---

## 8. Rule audit

**Goal.** A single checker that replays any play-by-play — simulator history, an exported sim
CSV, or a real cleaned game — and reports every rule violation, so each workstream above has a
pass/fail gate and the things that already work cannot regress.

**Invariants.**

| # | Invariant | Owner |
|---|---|---|
| 1 | Five a side, no duplicate players, exact player-minutes | baseline |
| 2 | No events by a fouled-out or ejected player | baseline |
| 3 | Every assist precedes a made basket by a teammate of the same class | baseline |
| 4 | Every substitution follows a dead-ball row | 3, 4 |
| 5 | No re-entry under the minimum bench time when set | 4 |
| 6 | No offensive foul by the defense; no personal, shooting or take foul by the offense | 2 |
| 7 | No foul resolves to nothing while the fouling team is in the penalty | 2 |
| 8 | The free-throw shooting team is never the fouling team | 2 |
| 9 | Every missed shot is followed by one rebound row or a possession-ending row | 3 |
| 10 | A forward possession scan is consistent at every row | 3 |
| 11 | No event straddles a period boundary; team fouls reset there | 3 |
| 12 | Per-team timeouts within budget | 3 |

**Work.**

- [ ] T1 · `simulation/rule_audit.py` with a CLI over a run directory; per-sim violation table in
      the reporting layer's Parquet conventions — `simulation/rule_audit.py`, `reporting/`
- [ ] T1 · Tests that pin the current violation counts on a committed sim as the "before", and
      zero on a scripted full game through the controller test double —
      `tests/test_rule_audit.py`, `tests/test_controller.py`
- [ ] T1 · Run on real cleaned games; any invariant real data breaks is a cleaner bug or a wrong
      invariant, fixed before it gates the sim

**Done when.** The audit runs in the eval pipeline and 2.0 rollouts report zero violations.

---

## Milestones

| Milestone | Scope | Gate |
|---|---|---|
| **A · Audit** | Workstream 8 | Reproduces the known violation counts on a committed v1.0 sim; clean on real games |
| **B · Controller** | Workstreams 2 and 3 rules (R1–R7), minus the parts that need new tokens | Audit passes on a v1.0 rollout for everything except team rebounds and timeouts. Optional: a 100-sim eval on v1.0 weights to measure what the rules alone are worth |
| **C · Re-clean** | Workstreams 1, 2, 3, 7 data changes (D1–D7) | Cleaner tests pass; era make-rate table looks right; audit clean on the new cleaned data |
| **D · Model** | Workstreams 4, 5, 6 (F1–F5) | Wiring and parity tests pass; head split reproduces the old graph at zero local heads |
| **E · Train 2.0** | Delete `encoder/vocabs/*.json`, check norm stats, `train.py --full --name v2.0 --clean --rebuild-vocabs` | User-run on WSL / CUDA |
| **F · Calibrate** | Audit on 2.0 rollouts, diagnostics, dial package from zero, re-keyed per zone | Standard eval report; zero audit violations |

**Things that will bite.**

| Gotcha | What to do |
|---|---|
| Vocabs are append-only and load from disk | Delete `encoder/vocabs/*.json` before the rebuild or stale tokens survive into every output layer |
| pytest overwrites the committed norm stats | Check `encoder/vocabs/norm_stats.json` before the train |
| Docs and some sources mix CRLF and LF per line | Edit by line, keep each line's own ending |
| The cleaner's home indicator returns "away" for anyone not on the home five | Team-rebound and timeout rows take their side from the raw `team` column |
| The backbone loop lives in six files | Extract the shared builder before touching attention |
| Tests pinning old behavior | The simulator's second possession tracker, the controller test double's `append_event`, the two-row steal, the dial-pinning test, the input-cache vocab warm-up |

---

## Out of scope for 2.0

| Item | Reason |
|---|---|
| Player age | External roster join with its own name-matching audit; adds later as a season-context column |
| Coach | External table; confounded with team-era; same |
| One row per play | Would subsume the pair collapse and the loss masking but touches every head and the decoder |
| Conditioning foul type on the fouled player | Needs a new condition type in the spec machinery; priced after the passive version is measured |
| ALiBi recency | Needs a custom attention layer; the masked heads get the bias without one |
| A dedicated minutes model | The rotation workstream is the first real attempt at the minutes gap; decide after |
| Possession as a feature, period-end rows, mid-game jump balls | Inferable, unlogged, or too rare |
| Calibration auxiliary losses, relative offense/defense encoding, RL fine-tuning | Parked as before |
