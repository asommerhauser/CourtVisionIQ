# CourtVisionIQ — Model 2: Planned Changes

> **Status (2026-09-03): decided, not built.** This document replaces `docs/v2_theories.md`.
> Every theory in that doc is either promoted into a numbered change below or listed under
> [Not in v2](#7-not-in-v2) with a reason. The measurements that motivated the theories are kept
> in [Appendix A](#appendix-a--evidence-carried-forward) so nothing already measured is lost.

---

## At a glance

**What v2 is.** One re-clean of the data, one full retrain, and a rewrite of the controller's
rule engine. The organizing idea is unchanged from the theories doc: *move what the controller
hard-codes into the data and the model*, and *make the controller enforce basketball rules it
currently does not know about* — who has the ball, when the ball is dead, who shoots free throws.

**The five things it fixes.**

| Problem today | What v2 does | Where |
|---|---|---|
| Substitutions happen during live play, and players re-enter seconds after sitting | Subs only at dead balls; timing, outgoing and incoming picks come from a rotation model that sees stint time, minutes and bench rest | [R1](#r1--dead-ball-state), [R2](#r2--substitutions-only-at-dead-balls), [F5](#5-rotation) |
| Shooting fouls award 3 free throws ~12x too often; and-1s almost never fire | The 2-vs-3 count becomes a learned foul token; and-1 becomes reachable | [D4](#d4--shooting-foul-ft-count-as-a-learned-token), [R6](#r6--data-parity-rules) |
| Fouls are typed without knowing which side the fouler is on, so bonus FTs are skipped and offensive fouls are charged to defenders | Foul type is masked by the fouler's side; free throws always go to the fouler's opponent | [R4](#r4--foul-type-is-masked-by-the-foulers-side), [R5](#r5--one-team-map-free-throws-go-to-the-foulers-opponent) |
| Shot type is only 2 / 3; efficiency runs low while 3PA runs high | Seven court zones from the raw shot coordinates | [D1](#d1--shot-zones) |
| The model has no bias toward the last few plays | Two of eight attention heads restricted to the last eight rows; a shot-clock proxy scalar | [F1](#f1--local-attention-heads), [F2](#f2--poss_seconds-a-shot-clock-proxy) |

**The retrain gate.** Every item in [Event schema](#3-event-schema) changes the token vocabulary.
They land together in one re-clean, then `--rebuild-vocabs` and a full train of every head.
Train 2's availability masking and capacity carry forward unchanged.

**How to read the ids.** `R` = controller rule (hard limit on sampling), `D` = data / cleaned
schema, `F` = model features and training. [Appendix B](#appendix-b--theory-disposition) maps the
old `S`/`M` theory ids onto these.

---

## 1. What the sim-015 audit found, verified

A rule-and-state audit of one simulated game (game 298324, POR vs ORL, `full4-s100` sim 015,
503 events) reported eight deterministic findings. Each was checked against the code.

| # | Claim | Verdict | What actually happens |
|---|---|---|---|
| B1 | 31 of 45 subs occur during live play | **Confirmed** | The controller's only "dead ball" test is *no rebound pending*. Subs fire after made baskets, live defensive rebounds and steals. Timeouts, period ends and jump balls do not exist in the cleaned data at all — the cleaner silently drops them. |
| B2 | Fouls with the team over the limit resolve to nothing | **Confirmed — same cause as B3** | Bonus logic exists, but the fouler is sampled from all ten players with no notion of side. A foul by an *offensive* player is typed "personal" and resolved as a defensive foul, so the bonus branch never runs. |
| B3 | 18 personal fouls with no consequence | **Confirmed** | Same root cause. The 7 defense-side cases are actually legal (offense keeps the ball); they only look wrong because nothing marks the dead ball. |
| B4 | 25 possession contradictions | **Confirmed — mostly invisible transitions** | The controller's possession is authoritative, but the stream hides several flips: the dead-ball-rebound dial flips possession with no row, offense-side fouls resolve wrongly, and "offensive" can be sampled for a defender. Separately, the simulator keeps a *second* possession copy that double-flips on steals and never flips on made baskets. |
| B5 | 5 misses with no rebound | **Confirmed — by design, mirrors the data** | The cleaner drops team rebounds, so real cleaned games show the same gap. Real 2022-23: 6.9% of rebounds are team rebounds. |
| B6 | The fouling team shoots its own free throws (2 cases) | **Confirmed — two different bugs** | One: a shooting foul that is the fouler's sixth. The foul is charged *before* the team is resolved, the disqualification sub pulls him from the floor, and the team lookup (by on-court membership) then returns the wrong team. Two: non-shooting free throws (flagrant, technical, bonus) are awarded to the team *in possession* rather than the fouled team. |
| B7 | Inter-event time is hard-capped at 20s | **Rejected** | The clamp is 60s. Across the 100 sims of that game the maximum gap ranges 19–23s; sim 015 hit 20 by chance. The tail is slightly short, not clamped — the missing tail is the timeouts and period breaks the cleaner drops. |
| B8 | `playoff=1` is mislabeled | **Rejected** | In cleaned data 1 = regular season, 2 = playoff. Not a model input. |

**Found while verifying (not in the report).**

| Finding | Detail |
|---|---|
| Offensive fouls lose their turnover | An offensive foul is a turnover in a real box score, and raw data logs it as a foul row *plus* a turnover row. The sim emits only the foul row, so its box score is short ~3.4 TOV per game. |
| Loose-ball fouls emit a token the data never has | The cleaner writes result `op`; the controller writes `nothing`. |
| Team fouls are counted differently at train and inference | The feature scan counts every non-technical, non-offensive foul by either side; the controller counts only defense-side fouls from its own whitelist. The `team_fouls` features would be off-distribution the moment a train uses them. |
| Re-entry churn has no constraint | Sim 015: 6 of 35 re-entries under 60s (9s, 10s, 14s…). Real 2022-23: 5.6% under 60s. |

**Real-data measurements the design leans on** (2022-23 raw, first 400k rows, 853 games).

| Fact | Value |
|---|---|
| Event preceding a real substitution | timeout 31% · free throw 21% · foul 21% · rebound 12% (40% of those are team rebounds) · turnover 12% · made shot 1.6% |
| Timeouts | 11.0 per game; 5.4 regular per team-game; **58% follow a made shot**; 70% are called by the team about to have the ball; the 135 "unknown" rows are coach challenges |
| Jump balls | 1.68 per game: 912 at a period start, 521 mid-period (0.61/game). The raw `possession` column names the player the tip went to, not a jumper |
| Team rebounds | 6.9% of all rebounds; team populated 100% |
| Separate "technical foul" rows | 431 per season: defensive 3-seconds 279, double technical 81, coach technical 71; player named 77% of the time |
| Shot coordinates | 99.8% of shots have `converted_x/y` |
| Real inter-event gap | p99 = 24s, max 74s, 4.4% over 20s |
| Real bench time before re-entry | p5 = 49s, median 394s |

---

## 2. Controller rules (hard limits)

The controller lets the models choose only among legal options. These are the constraints it
enforces regardless of what any head says. Each is testable as a hard assertion
([§8](#8-verification-the-rule-audit)).

### R1 — Dead-ball state

**Change.** The controller keeps a `ball_dead` flag, set by every play handler. It is internal
bookkeeping and is **never written to the data**.

| Ball is dead after | Ball is live after |
|---|---|
| any foul | a made field goal (except the clock-stop windows below) |
| a non-steal turnover | a missed or blocked field goal (rebound pending) |
| the last free throw of a trip, if made | a live rebound |
| a team rebound | a steal |
| a timeout | the last free throw of a trip, if missed |
| a period boundary | |
| a made field goal in the last 60s of Q1–Q3 or the last 120s of Q4/OT | |

**Why.** Every illegal substitution in the audit traces to the absence of this state. The same
rule, applied to the cleaned stream, also defines the positions the rotation model trains on
([F5c](#f5c--a-sub_decision-head-asked-only-at-dead-balls)) — every input it needs is a v2 row.

**Touches.** `simulation/controller.py`.

**Done when.** Audit assertion 4 holds: every substitution follows a dead-ball row.

### R2 — Substitutions only at dead balls

**Change.** The rotation model is consulted only when the ball is dead. Foul-out and ejection
replacements already happen at a dead ball. The unreachable event-driven substitution path and
the stint scheduler's live-ball firing are deleted.

**Why.** Real subs after a made basket go through a timeout (58% of timeouts follow a made
shot). With timeouts in the stream ([D3](#d3--timeouts-as-a-sampled-event)) the sim gets that
same path instead of subbing on a live ball.

**Touches.** `simulation/controller.py`.

### R3 — Re-entry safety net

**Change.** Bench rest is *learned* ([F5b](#f5b--a-bench-bundle)), so the rule is a backstop
only: a player is not a re-entry candidate until `SUB_MIN_BENCH_SECONDS` after exiting, unless
the bench is otherwise empty. Default **0 (off)**; set after the first v2 diagnostics if churn
persists. `SUB_MAX_GAP_SECONDS` stays as the one cadence backstop.

**Touches.** `simulation/controller.py`, `config.py`.

### R4 — Foul type is masked by the fouler's side

**Change.** After the fouler is sampled, the controller knows whether they are on offense or
defense and masks the foul-type head accordingly.

| Fouler is on | Allowed types | Resolution |
|---|---|---|
| offense | `offensive`, `loose ball`, `technical`, `flagrant-1`, `flagrant-2` | `offensive` → turnover. `loose ball` → result `op`, possession unchanged, counts as a team foul. Flagrant / technical → free throws to the *defense*. |
| defense | everything except `offensive` | Common foul → offense retains (dead ball), or 2 FTs if in the penalty. Take / flagrant / technical / shooting unchanged. |

The one exception is the and-1: a foul sampled immediately after a made basket, by the team that
was scored on, is typed against the *pre-flip* side ([R6](#r6--data-parity-rules)).

**Why.** B2 and B3. The bonus code was correct; it was simply unreachable when the sampled
fouler was on offense.

**Touches.** `simulation/controller.py`.

**Done when.** No `offensive` foul by a defender; no `personal`, `shooting` or take foul by
the offense; no foul resolves to nothing while the fouling team is in the penalty.

### R5 — One team map; free throws go to the fouler's opponent

**Change.** A `team_by_player` map is built once at tip-off from the full rosters and never
mutated. Every team lookup uses it — the current lookup by on-court membership goes away. Every
free-throw-awarding branch (shooting, bonus, take, flagrant, technical) shoots with the
fouler's opponent, and the shooting-foul handler resolves team and shooter *before* charging
the foul.

**Why.** Both B6 cases. A fouled-out player is pulled from the floor before his team is looked
up; bonus, flagrant and technical free throws are awarded to whoever has possession.

**Touches.** `simulation/controller.py`.

**Done when.** The free-throw shooting team is never the fouling team.

### R6 — Data-parity rules

Small rules where the sim and the cleaned data disagree today.

| Rule | Today | v2 |
|---|---|---|
| Offensive foul in the box score | foul only | foul **and** turnover |
| Loose-ball foul result token | `nothing` | `op` (the cleaner's token) |
| Team-foul definition | controller whitelist, defense side only | the single definition already in `models/game_state_features.py` (all but technical and offensive, either side); the controller imports it |
| Bonus in the last 2:00 | cumulative second team foul | second foul **committed inside the window**, tracked per team, reset per period |
| And-1 | guard is almost unreachable because possession flips the instant a basket goes in | the controller remembers the scorer; a foul as the very next play by the scored-on team is typed pre-flip and, if shooting, resolves as an and-1 (1 FT, dt = 0). Real: ~25% of shooting fouls are and-1s |
| Possession trackers | two (controller + simulator), and they disagree | one — the simulator's copy and its flip-on-result rule are deleted. Possession is **not** written to the data |
| Opening possession | always home | coin flip; Q2/Q3 to the tip loser, Q4 to the winner, OT re-flips |

**Touches.** `simulation/controller.py`, `simulation/game_simulator.py`, `simulation/box_score.py`.

### R7 — Periods are real cuts

**Change.** If a sampled gap would cross the period end, the clock stops at the boundary, team
fouls reset, the ball is dead, and the sampled play is **discarded** and re-sampled in the new
period. No event straddles a boundary. Nothing is logged.

**Why.** Today events spill across boundaries and the reset happens a play late.

**Touches.** every clock-advance site in `simulation/controller.py` (seven of them); the
minutes credit is clamped to the boundary.

### R8 — Timeouts obey a budget

**Change.** 7 per team per game, at most 4 in Q4, at most 2 in the last 3:00; +2 per overtime.
The event head is masked from `timeout` for a team that is out.

**Why.** Modern rules are applied to every era. The head learns era timing from the season
embedding and the clock features; the budget only caps the tail.

**Touches.** `simulation/controller.py`, `config.py`.

---

## 3. Event schema

All of these change the token vocabulary. They land in one re-clean.

### D1 — Shot zones

**Change.** `shot_type`'s vocabulary goes from `{2pt, 3pt}` to seven coarse zones:
`restricted_area`, `paint`, `short_mid`, `long_mid`, `corner3`, `abovebreak3`, `heave`.
Derived in the cleaner from the raw `converted_x/y` (99.8% populated) with a fixed geometric
rule; the raw 2/3 marker wins at the arc; missing coordinates fall back on `shot_distance` and
the marker. Assist rows keep coarse `2pt`/`3pt`; the forced made shot after an assist samples
a zone masked to the assist's 2/3 class.

**Why.** Where the model stands (`full4-s100`): points bias is closed, but eFG runs 1.7 low and
3PA 2.3 high. That is a *composition* problem the eFG dial cannot fix. Per-player-per-zone make
rates attack it at the source, and the rebound head gets to see where the miss came from.

**Touches.** `data_cleaner.py`; a new `simulation/zones.py` (`ZONE_POINTS`, `is_three`) that
every current `"3pt"` string check routes through: `simulation/box_score.py`,
`simulation/controller.py`, `models/game_state_features.py`, `models/conditional_type_model.py`,
`simulation/game_simulator.py`; the eFG dials re-key per zone.

**Done when.** A per-zone, per-era make-rate table looks like known basketball (restricted area
~62–67%, long mid ~38–42%, corner 3 above break 3, 3PA share rising across eras).

### D2 — Team rebounds as rows

**Change.** Team rebounds are kept: `rebound / none / <offensive|defensive> / <null|cop>`, side
from the raw `team` column, offensive/defensive from the team vs. the shooter's team. The
`rebound_type` head grows to four tokens (`offensive`, `defensive`, `team offensive`,
`team defensive`); the controller skips the rebounder pick on a team token. The
`DEADBALL_REBOUND_PROB` dial is deleted — the head learns the 6.9% share.

**Why.** Team rebounds are part of the rebound stat, and this removes the last possession flip
that happens without a row (most of B4 and all of B5).

**Touches.** `data_cleaner.py`, `models/conditional_type_model.py`, `simulation/controller.py`,
`simulation/box_score.py` (team line, not a player's).

### D3 — Timeouts as a sampled event

**Change.** `timeout / none / <home|away> / none`. The calling team rides in `type` because
`home/away` is not a model input; it is predicted by a seventh spec entry alongside the six
existing conditional type heads (one line in the spec table, same shape as `rebound_type`).
Both raw types (regular and coach challenge) map to one event. `timeout` joins the event menu,
masked to dead balls or the team in possession.

**Why.** 31% of real subs follow a timeout — the rotation model needs to see that window. The
time head gets the long-gap tail it is missing (B7). The event head can learn "timeout after a
run" from the score features ([F3](#f3--game-state-activation--clutch-weighting)).

**Touches.** `data_cleaner.py`, `models/conditional_type_model.py`, `simulation/controller.py`.

### D4 — Shooting-foul FT count as a learned token

**Change.** Carry the raw `num`/`outof` free-throw index on FT rows, and split the `shooting`
foul token into `shooting 2pt` / `shooting 3pt`, labelled from the trip's `outof`. The
controller reads the token; the fresh `shot_type` sample it uses today is deleted. And-1 stays
a structural check ([R6](#r6--data-parity-rules)).

**Why.** The 3-FT bug ([Appendix A](#appendix-a--evidence-carried-forward)): the count is
decided by sampling a head that was trained not to answer that question, and it is wrong by
~12x.

**Touches.** `data_cleaner.py`, `simulation/controller.py`, `config.py` (`TYPE_BIAS` keys).

### D5 — Fouled player as `secondary_player` on foul rows

**Change.** The raw `opponent` column (98.1% populated on fouls, empty only on technicals) is
carried into `secondary_player`, which already exists, already shares the player embedding, and
already has a `none` token. The controller writes it and uses that player as the free-throw
shooter, removing today's independent shooter draw.

**Why.** Drawing fouls is a skill the model currently cannot represent. This is the cheap half;
conditioning foul type on the fouled player (S3b) stays parked.

**Touches.** `data_cleaner.py`, `simulation/controller.py`.

### D6 — Collapse continuation pairs

**Change.** One row per steal (`turnover / steal / cop` with `secondary_player` = the stealer)
and one row per offensive foul (`foul / offensive / cop`, no trailing turnover row). The box
score counts STL from `secondary_player` and TOV from both.

**Why.** Matches the pattern `block` already uses, removes ~22k phantom rows per season, and
takes 2.8 points off the 21.9% continuation mismatch
([Appendix A](#appendix-a--evidence-carried-forward)).

**Touches.** `data_cleaner.py`, `simulation/box_score.py`, `simulation/controller.py`.

### D7 — Standalone technical-foul rows

**Change.** The raw `technical foul` event (defensive 3-seconds, double technical, coach
technical) becomes `foul / <player> / technical / free throw` where a player is named (77%);
the rest stay dropped.

**Touches.** `data_cleaner.py`.

### Explored and not adopted

| Item | Why not |
|---|---|
| Jump balls | Raw shape is "two jumpers, tip to a third player", once per game plus 0.6 held balls. Nothing in v2 consumes it; a coin flip plus the period rule covers possession. |
| Period-end rows | Dead-ball state stays in the controller and is not logged. The cross-boundary gap is handled in the loss instead ([F4](#f4--play-boundary-and-forced-row-loss-masking)). |
| Possession as a data column | Possession is a sampling guide, not a fact the model has to be told — once team rebounds are rows it is inferable from the last event. |
| Raw `ejection` / `violation` rows | Flagrant-2 carries the ejection; kicked-ball and similar already arrive as turnover rows. |
| Playoff flag | Correct as is (1 = regular, 2 = playoff). Document the encoding where the sim CSV is written. |

---

## 4. Model features and training

### F1 — Local attention heads

**Change.** In every backbone block, 2 of the 8 attention heads become *local*: their weights
are forced to zero outside the last `LOCAL_ATTN_WINDOW = 8` rows by a banded boolean mask, using
the same mechanism as the existing padding mask. The other 6 stay global. Outputs are
concatenated so the width is unchanged. `LOCAL_ATTN_HEADS = 0` disables it for an A/B.

**What it is.** An attention head builds each row's representation as a weighted average over
all earlier rows. Nothing pushes any head to concentrate on the last few. A local head can only
summarize the immediate play — the cheap version of "hand the model the last five events", with
no new inputs and no custom kernel.

**Why.** Basketball is strongly locally sequential (miss → rebound, steal → transition, hard
foul → technical). The 600-position backbone *can* see the last five plays; it has no reason
to prefer them.

**Touches.** The backbone loop is duplicated in six model files (`models/event_time_model.py`,
`models/player_model.py`, `models/substitution_model.py`, `models/stint_length_model.py`,
`models/conditional_type_model.py`, `models/conditional_time_model.py`) — extract one shared
builder first. Layer shapes change, so this is a same-code A/B, not a v1.0 weight reload.

**Fallback.** An explicit last-5-events input block if the ablation shows nothing.

### F2 — `poss_seconds`, a shot-clock proxy

**Change.** One new per-row scalar: seconds since the current possession started (reset on a
change of possession or an offensive rebound), clipped 0–35 and scaled by 24. Derived inside the
existing game-state scan, so training and the simulator execute the same code. The scan tracks
possession internally to compute it; possession itself is *not* a feature.

**Why.** Shot-clock pressure is a first-order driver of shot selection, of shot-clock
turnovers, and of the next gap (an event *must* come within 24s), and it is hard for attention
to compute — it means finding the last possession change and summing every gap since.

**Marked cuttable.** If it does not earn its place in diagnostics, it is one key to remove.

**Touches.** `models/game_state_features.py` (single seam: every head, the preprocessing merge
and the input cache all key off `GAME_STATE_KEYS`), `tests/test_game_state_features.py`.

### F3 — Game-state activation + clutch weighting

**Change.** No new code for the six existing game-state scalars (score diff, score total,
period, time left, team fouls) — v2 is simply the first train whose weights consume them.
Clutch weighting multiplies each head's loss mask by `CLUTCH_LOSS_WEIGHT = 2.0` on rows that are
Q4/OT, under 5:00 left, within 8 points. One shared helper next to the recency weighting.

**Why.** End-game behavior (leads sat on, trailing teams fouling), bonus-aware foul value, and
win% in close games — the metric where the model still trails the baseline.

**Risk.** Upweighting late rows shifts base rates slightly toward late-game basketball. Start
at 2.0 and watch the per-quarter splits.

**Touches.** `models/season_features.py`, each head's dataset builder.

### F4 — Play-boundary and forced-row loss masking

**Change.** Zero the event and conditional-time heads' loss on rows the simulator never asks
about: continuation rows (free throw after foul or free throw, shot after assist, block after
blocked shot), controller-forced rows (substitutions), and — for the time head only — the last
row before a period boundary, whose gap spans a break the sim never samples. The continuation
rule lives in one function shared with the controller's expansion logic so they cannot drift.

**Why.** 21.9% of the event head's training positions are mid-play continuations it is never
queried at, so its learned marginals use the wrong denominator. This is the most plausible
reason the event and type dials need re-tuning after every train.

**Consequence.** After this train, **the dials are re-measured from zero**, not carried forward.

**Touches.** the mask builders in all six model files.

---

## 5. Rotation

Today, *when* a player comes off is a stint-length regression plus a fatigue nudge, *who* comes
in is a substitution head with no knowledge of who just sat, and the whole thing fires on live
balls. v2 rebuilds this inside the model.

### F5a — Per-player live state, bundled like rest days

**Change.** The roster encoder already takes one per-slot scalar (days of rest). Add three more
for each on-court slot: `stint_seconds` (since this entry), `minutes_played` (game total),
`personal_fouls`. Derived by the same scan that builds game state, from substitution rows and
roster snapshots.

**Why.** Every head sees them wherever it consumes the lineup — a 5-foul player fouls less, a
tired one shoots worse — not just the rotation heads.

**Touches.** `models/season_features.py` (bundle plumbing), `models/game_state_features.py`
(derivation), `models/roster_set_encoder.py`.

### F5b — A bench bundle

**Change.** A second set input: up to `BENCH_SLOTS = 10` available bench players with
`bench_seconds` (since exit, or since tip-off if not yet played), `minutes_played`,
`personal_fouls`, `has_played`. Encoded by the *same* set encoder into a bench vector. The
incoming pick keeps its vocab-sized output and availability mask, plus a per-candidate additive
score computed from those slot scalars by a tiny shared MLP and added onto that candidate's
logit.

**Why.** This is how "he sat down 9 seconds ago" becomes a learned penalty instead of a dial.

**Touches.** `models/substitution_model.py`, `models/roster_set_encoder.py`,
`simulation/game_simulator.py` (bench slot construction from the available set).

### F5c — A `sub_decision` head, asked only at dead balls

**Change.** At each dead ball ([R1](#r1--dead-ball-state), computed identically on the cleaned
stream) the target is, per team, how many substitutions follow before the next live-ball row:
`0 / 1 / 2 / 3+`. Trained on the full corpus, on dead-ball positions only.

Controller loop at a dead ball: sample the count per team; for each sub, pick the outgoing
player (player head over the five, now with [F5a](#f5a--per-player-live-state-bundled-like-rest-days))
then the incoming player (substitution head over the bench, now with
[F5b](#f5b--a-bench-bundle)).

**Why.** The head is never asked anywhere it did not learn, so the train/inference denominator
is right by construction. The worry that gating alone would leave subs "the same per-row chance,
but only called at dead balls" does not arise. A "dead ball coming" input flag was considered
and is unnecessary: the head is evaluated *at* the dead ball, with that row already in history.

**Retired.** The stint-length head and scheduler, `SUB_FATIGUE_WEIGHT`, the `STINT_*` dials.
`SUB_MAX_GAP_SECONDS` stays as the single backstop.

**Fallback.** If F5c is judged too large: keep the stint scheduler with dead-ball gating and
land F5a/F5b alone — bench rest is still learned.

**Touches.** a new head under `models/`, `models/registry.py`, `simulation/controller.py`.

---

## 6. Order of operations

1. **Rule audit first** ([§8](#8-verification-the-rule-audit)). It must reproduce the sim-015
   counts exactly (31 live-ball subs, 2 inverted free-throw trips, 5 rebound-less misses) and
   run clean on real cleaned games. Any invariant real data violates is a cleaner bug or a wrong
   invariant, and gets fixed before it gates the sim.
2. **Controller rules** R1–R8. Everything except R2/R8 (which need timeouts and team-rebound
   tokens) can be evaluated on v1.0 weights in isolation. Whether to do that first is a
   sequencing choice, not a dependency.
3. **Cleaner changes** D1–D7 together, then re-clean, then `enrich`. Check the era make-rate
   table (D1).
4. **Features and heads** F1–F5, with the wiring tests extended.
5. **Controller consumers of the new tokens** (zones, `shooting 2pt/3pt`, team rebounds,
   timeouts, fouled player, the rotation loop).
6. **Train** (user-run, WSL/CUDA):
   `python train.py --full --name v2.0 --batch-size 64 --clean --rebuild-vocabs`
7. **Post-train:** audit → diagnostics → dial package **from zero** (F4), re-keyed per zone.

**Gotchas.**

| Gotcha | Detail |
|---|---|
| Vocabs are append-only | `Vocab` loads the on-disk JSON and never renumbers, so `--rebuild-vocabs` keeps stale `2pt`/`3pt` and steal-pair tokens in every head's output layer. **Delete `encoder/vocabs/*.json` before the v2 rebuild.** |
| Norm-stats pollution | pytest overwrites `encoder/vocabs/norm_stats.json`; verify the freeze happens from real data. |
| Raw columns are dropped at load | `outof`, `converted_x/y`, `shot_distance` are dropped in `data_cleaner.parse_file`; un-drop them there. |
| Side for `player = none` rows | The cleaner's home indicator returns "away" for anyone not in the home five, so team-rebound and timeout rows take their side from the raw `team` column. Team names resolve lazily from the first action row, so a timeout before any action needs a default. |
| `none` as a player is already safe | It is a vocab special; the player head zeros its loss on such rows, the box score skips them, and the availability mask already tolerates it. |
| Dial keys rename | `TYPE_BIAS` keys for `foul_type.shooting` and `rebound_type.offensive` split with their tokens. |
| Tests that pin old behavior | the simulator's second possession tracker, the controller test double's `append_event` signature, the two-row steal in the cleaner tests, the dial-pinning test, the input-cache vocab warm-up. |

---

## 7. Not in v2

| Item | Reason |
|---|---|
| S6 player age | External roster join with its own name-matching audit; not bug-driven. Adds later as a season-context column with no vocab change. |
| S7 coach | Same; external table plus a confounded embedding. |
| One row per play | Would subsume D6 and F4 but touches every head and the decoder. D6 + F4 get most of the value; revisit after v2 diagnostics. |
| S3b (condition foul type on the fouled player) | Needs a new condition type in the spec machinery plus a controller reorder. Priced separately once D5's passive context is measured. |
| ALiBi recency | Needs a custom attention layer; F1's masked heads get the inductive bias without one. |
| Rotation-minutes model / seeded starters | F5 is the first real attempt at the 5.66-minute per-player minutes MAE. If it moves, a dedicated minutes model may not be needed. |
| Mid-game jump balls, possession as a feature, period-end rows | See [Explored and not adopted](#explored-and-not-adopted). |
| Calibration auxiliary losses, relative offense/defense encoding, RL fine-tuning | Parked as before. |

---

## 8. Verification: the rule audit

A `simulation/rule_audit.py` replays any play-by-play frame — simulator history, an exported
sim CSV, or a real cleaned game — and reports every violation of:

| # | Invariant | Enforced by | Status today |
|---|---|---|---|
| 1 | 5-a-side, no duplicate players, 480 player-minutes (+300 per OT) | — | holds |
| 2 | No events by a fouled-out or ejected player | — | holds |
| 3 | Every assist immediately precedes a made basket by a teammate of the same 2/3 class | — | holds |
| 4 | Every substitution follows a dead-ball row | R1, R2 | **31 violations** in sim 015 |
| 5 | No re-entry under `SUB_MIN_BENCH_SECONDS` when set | R3 | 6 under 60s |
| 6 | No `offensive` foul by the defense; no `personal`/`shooting`/take foul by the offense | R4 | 14 |
| 7 | No foul resolves to nothing while the fouling team is in the penalty | R4, R6 | 12 |
| 8 | The free-throw shooting team is never the fouling team | R5 | 2 |
| 9 | Every missed field goal is followed by exactly one rebound row or a possession-ending row | D2 | 5 |
| 10 | A forward possession scan is consistent: every shot, assist and turnover actor is on the scanned possession team | R6, D2 | 25 |
| 11 | No event straddles a period boundary; team fouls reset there | R7 | — |
| 12 | Per-team timeouts within budget | R8 | n/a (no timeouts yet) |

Items 1–3 are the regression baseline: they hold today and must keep holding.

---

## Appendix A — Evidence carried forward

Measurements from the theories doc that motivated the changes above. Each names how it was
produced so it can be re-derived.

### Where `v1.0` stands (`full4-s100`, 100 games x 100 sims)

| Metric | Value | Reads as |
|---|---|---|
| Team points bias | -0.52 | scoring shortfall closed by dials |
| Pace bias | +0.47 | fine |
| eFG | 1.7 points low | composition, not volume → D1 |
| 3PA | 2.3 attempts high | same |
| Per-player minutes MAE | 5.66 min | the largest single term in per-player box error → F5 |
| Win% in close games | trails the season-average baseline (47%) | → F3 |

### The 3-FT bug

The shooting-foul handler decides the free-throw count by sampling the live-field-goal
`shot_type` head — a head trained on real attempts only, with a league-wide ~35–40% three share
— from a context that never occurs in training (a shooting-foul row already appended), with a
hard-coded zero gap. And-1s under-trigger structurally because possession flips the instant a
basket goes in, so the guard looks in the wrong possession.

Ground truth from the raw `outof` column (2022-23, first ~400k rows):

| FT trip | Count | Share |
|---|---|---|
| 2 FTs | 16,500 | 72.1% |
| 1 FT | 5,824 | 25.5% |
| 3 FTs | 548 | 2.4% |

Against 18,010 shooting fouls: ~3% are 3-FT trips, ~25% and-1s, ~72% 2-FT. The sim produced 3-FT
trips roughly **12x too often**. Every rollout before D4 lands carries this inflation in its FTA
and PF numbers.

### The fouled player is already in the raw data

Raw `opponent` is populated on 34,535 of 35,208 foul rows (98.1%) and on no other event type.
The 673 empty rows are technicals plus 60 blank-type fouls — the cases with no fouled individual.

### The steal pair and the continuation mismatch

Steals are two rows (stealer `steal`, ball-loser `cop`). The stats are correct and the sim is
faithful — but the event head trains to predict the next *row* at every position while the sim
only queries it at *play-start* positions:

| Continuation kind | Rows (2022-23) |
|---|---|
| shot after assist | 66,559 |
| free throw after foul / free throw | 49,710 |
| steal `cop` after steal | 19,167 |
| block after blocked shot | 12,325 |
| **Total** | **147,761 of 673,873 (21.9%)** |

D6 removes the steal row; F4 masks the rest.

---

## Appendix B — Theory disposition

| Theory | Was | Became |
|---|---|---|
| S1 shot zones | proposed | **D1** |
| S2 FT count as a learned outcome | proposed | **D4** |
| S3a fouled player as `secondary_player` | proposed | **D5** |
| S3b condition foul type on the fouled player | open question | not in v2 |
| S4 collapse the steal pair | proposed | **D6**, extended to offensive fouls |
| S5 FT index `num`/`outof` | promoted (label for S2) | **D4** |
| S6 player age | proposed | not in v2 |
| S7 coach | proposed | not in v2 |
| M1 game-state features | plumbed, untrained | **F3** |
| M2 clutch loss weighting | proposed | **F3** |
| M3 recency bias | options 1–4 | **F1** (option 2); explicit last-N block is the fallback; ALiBi not in v2 |
| M4 play-boundary loss masking | proposed | **F4**, plus forced-row and period-boundary masking |
| — (new) | — | **R1–R8** controller rules, **D2** team rebounds, **D3** timeouts, **D7** technical rows, **F2** shot-clock proxy, **F5** rotation |
