# CourtVisionIQ 2.0 — Development Guide

> What 2.0 adds, why each piece exists, and the order we build it in. Written to be worked from:
> every feature ends with **Next steps** naming the concrete entry points. Nothing here is built
> yet.

## Status

Branch `feature/version2`, level with `main`. Nothing in this document is implemented.

Version 1 closed most of the gap to the season-average baseline (+44% MAE / 35% win-pick in June →
+8% / 47% with full1) but still loses to it. The remaining error is not one problem. Roughly half
is **rules the controller doesn't enforce** — fouls resolving to the wrong side, free throws going
to the wrong team, no concept of a dead ball, substitutions on a timer instead of at stoppages. The
other half is **information the model can't see** — where a shot came from, who got fouled, the
shot clock, the last few plays.

2.0 fixes both. All data changes land in **one re-clean and one full retrain with a vocab rebuild**.

## The eleven workstreams

Ordered as we build them, not by importance.

| # | Feature | Data | Model | Engine |
|---|---|---|---|---|
| **Phase 1 — controller rules, no new tokens, measurable on today's weights** ||||
| [1](#1-side-aware-fouls) | Side-aware fouls | — | — | Heavy |
| [2](#2-dead-balls-one-possession-tracker-clean-period-boundaries) | Dead balls, one possession tracker, clean period boundaries | — | — | Heavy |
| **Phase 2 — the re-clean, new tokens** ||||
| [3](#3-shot-zones) | Shot zones | Heavy | Light | Heavy |
| [4](#4-learned-free-throw-counts) | Learned free-throw counts | Moderate | Light | Moderate |
| [5](#5-the-fouled-player) | The fouled player | Light | — | Moderate |
| [6](#6-timeouts-and-team-rebounds) | Timeouts and team rebounds | Moderate | Light | Heavy |
| [7](#7-schema-cleanup) | Schema cleanup | Heavy | — | Moderate |
| **Phase 3 — model** ||||
| [8](#8-a-rotation-model) | A rotation model | — | Heavy | Heavy |
| [9](#9-local-context) | Local context | — | Heavy | Light |
| [10](#10-training-changes) | Training changes | — | Moderate | Config |
| **Phase 4 — measurement** ||||
| [11](#11-per-quarter-eval-splits) | Per-quarter eval splits | — | — | Moderate |

Phase 1 is the only clean measurement point in the whole programme: it changes no tokens, so it can
be evaluated on the current weights and its effect isolated from everything downstream. Take that
eval.

---

# Phase 1 — Controller rules

## 1. Side-aware fouls

**What it is.** A rule the controller enforces: after the fouler is sampled, the foul-type head is
masked to the types that side can commit, and the outcome resolves for that side.

| Fouler on | Can commit | Outcome |
|---|---|---|
| offense | `offensive`, `loose ball`, `technical`, `flagrant` | offensive → turnover; loose ball → possession unchanged, team foul; flagrant or technical → free throws to the defense |
| defense | everything except `offensive` | common foul → offense keeps the ball, or two free throws in the penalty; shooting, take, flagrant and technical as today |

**Why.** The foul-type head has no idea which side the fouler is on, so it can type an offensive
player's foul as a defensive one, or charge an offensive foul to a defender. Masking by side makes
every foul resolve to a legal outcome, and it is what lets the bonus fire when it should.

**How it works.** Three concrete defects, all in `simulation/controller.py`:

1. **`_team_of` reads the floor, not the roster.** `controller.py:671` is
   `HOME if player in self.sim.home_roster else AWAY`, and `home_roster` is the **on-court five**,
   not the full roster. Any player not currently on the floor resolves to `AWAY`. A foul-out
   substitution mid-resolution can therefore flip whose free throws they are. Replace with a team
   map built once at tip-off from `home_full` / `away_full`, resolved *before* the foul is charged.
2. **Foul types are not masked by side.** `controller.py:476` picks `allowed_types` before the
   fouler is even sampled, and `on_defense` (`:489`) is computed *after* the type. Sample the
   fouler, resolve their side, then mask.
3. **`_do_shooting_foul` assumes a defender.** `controller.py:528` hard-codes
   `shooting_team = self._other(fouler_team)`. With (2) in place this becomes true by construction
   rather than assumed.

Also here: free throws go to the fouler's opponent in **every** branch — shooting, bonus, take,
flagrant, technical. Team fouls use one definition, shared with the game-state features (every foul
except technical and offensive, either side), so the trained feature and the sim cannot drift. The
last-two-minutes penalty triggers on the second qualifying foul inside the window.

**Next steps.**
- `simulation/controller.py:671` — team map from `home_full`/`away_full` at tip-off.
- `simulation/controller.py:467-514` — reorder `_do_foul` to sample, resolve side, then mask.
- `simulation/controller.py:516-545` — `_do_shooting_foul` reads the resolved side.
- `simulation/controller.py:641` — `_in_bonus` last-two-minutes rule.
- Share `TEAM_FOUL_TYPES` (`controller.py:72`) with `models/game_state_features.py:42`.
- `tests/test_controller.py` — add cases for an offensive-side foul and a foul on a bench player.

## 2. Dead balls, one possession tracker, clean period boundaries

**What it is.** The controller learns whether the ball is dead, stops keeping two copies of
possession, and stops letting plays straddle a period boundary.

**Why.** Substitutions currently key off `pending_rebound` as a dead-ball proxy, which is why they
land at the wrong moments. Two possession trackers can disagree. And a play sampled near the end of
a quarter can run past the buzzer.

**How it works.**

*Dead-ball state.* Internal to the controller, never written to the data.

| Dead after | Live after |
|---|---|
| any foul; a non-steal turnover; a made last free throw; a team rebound; a timeout; a period boundary; a made basket in the last minute of Q1–Q3 or the last two minutes of Q4 / OT | a made basket otherwise; a missed or blocked shot; a live rebound; a steal; a missed last free throw |

Substitutions happen only when the ball is dead (see [§8](#8-a-rotation-model)).

*One possession tracker.* There are two today — the controller's (`controller.py:114`) and the
simulator's (`game_simulator.py:174`, flipped at `:346`, written into every row at `:343`). The
simulator's goes. Possession stays a sampling guide: not a data column, not a model input. Opening
possession is a coin flip; Q2 and Q3 go to the tip loser, Q4 to the winner, overtime re-flips.

*Clean period boundaries.* `_check_period` (`controller.py:617`) runs *after* the play resolves.
`_current_period_end` (`:636`) already exists but is never used to clamp. If a sampled gap would
cross the boundary, stop the clock there, reset team fouls, mark the ball dead, and re-sample in the
new period. No event straddles a boundary.

**Next steps.**
- `simulation/controller.py` — add `self.ball_dead`, set it in each `_do_*` handler.
- `simulation/controller.py:428,453` — replace the `pending_rebound` dead-ball proxy.
- `simulation/game_simulator.py:174,343,346` — delete the second possession tracker and the
  `possession` row field.
- `simulation/controller.py:605-628` — clamp `_advance_clock` at `_current_period_end`.
- `tests/test_controller.py` — a play sampled across a buzzer must not straddle it.

---

# Phase 2 — The re-clean

Everything in this phase lands in **one** re-clean and rides the same retrain.

## 3. Shot zones

**What it is.** The `shot_type` head predicts one of **fifteen court zones** instead of `2pt` /
`3pt`. Assist and block rows carry the same zone token.

**Why.** Shot selection and shot efficiency are spatial. The model currently learns one make rate
per player per point value, so a rim finisher and a long-two shooter look alike. Per zone, the
rebound head sees where a miss came from, the three-point revolution becomes a learnable spatial
fact instead of something absorbed into player embeddings, and rim make rate stops competing with
long-mid frequency for a single eFG dial — the exact failure the full1/full2 biases decomposed into.

**How it works.**

Both baskets fold onto one half court **oriented as the offense attacks**, so a team gets the same
token regardless of which way it is going. Raw frame is full court: `converted_x` 0–50,
`converted_y` 0–94 feet, hoops at `(25, 5.25)` and `(25, 88.75)`.

```
end_a = y < 47
ny    = (y - 5.25) if end_a else (88.75 - y)     # depth from the hoop
nx    = (x - 25.0) if end_a else (25.0 - x)      # + is the shooter's left
dist  = hypot(nx, ny)
phi   = degrees(atan2(nx, max(ny, 1e-6)))        # 0 = straight on
```

The `nx` flip at the far basket is what makes left/right a real signal (handedness, shooter court
preference) rather than noise.

Fifteen tokens — four center zones, five mirrored pairs, plus `heave`. Names align radially, so
each mid-range zone pairs with the arc zone directly behind it:

| Token | Rule |
|---|---|
| `rim` | `dist <= 4` |
| `paint` | not rim, `abs(nx) <= 8`, `ny <= 14` (the lane) |
| `mid_base_l` / `mid_base_r` | 2pt, outside the paint, `dist < 16`, `abs(phi) > 25` |
| `mid_wing_l` / `mid_wing_r` | 2pt, `dist >= 16`, `25 < abs(phi) <= 55` |
| `mid_corner_l` / `mid_corner_r` | 2pt, `dist >= 16`, `abs(phi) > 55` |
| `mid_top` | 2pt, outside the paint, `abs(phi) <= 25` |
| `corner3_l` / `corner3_r` | 3pt, `ny <= 8.75` (the straight-line portion of the arc) |
| `wing3_l` / `wing3_r` | 3pt, not corner, `abs(phi) > 25` |
| `top3` | 3pt, not corner, `abs(phi) <= 25` |
| `heave` | 3pt, `dist >= 32` |

The raw `type` text (the `3pt` prefix) stays the **authority** on point value; geometry only picks
the zone within the 2pt or 3pt family. Measured disagreement between derived geometry and the raw
marker is **0.42%** in 2002-03 and **0.14%** in 2022-23. Coordinate coverage is **≥98.4% in every
one of the 21 seasons** (most under 0.1% null). Fallback for missing coordinates: `shot_distance`
plus the marker; if both are unusable, `mid_base_l` for 2s and `wing3_l` for 3s.

`heave` is its own token rather than folded into `top3` because heaves are 0.3–0.4% of shots at
5–15% — an order of magnitude worse than a real three — and would otherwise drag every
above-the-break make rate down by about a point. It only makes sense as a token because the
game-state features ([§10](#10-training-changes)) give the model the clock this train.

**Validation gate.** Run before training; this table is the regression check.

| Zone | 2002-03 | 2012-13 | 2022-23 |
|---|---|---|---|
| `rim` | 30.5% @ 58.9% | 33.2% @ 60.1% | 30.3% @ **66.1%** |
| `paint` | 14.7% @ 39.1% | 14.3% @ 38.6% | 19.3% @ 44.4% |
| `mid_corner_l` | 5.4% @ 38.9% | 3.4% @ 40.0% | **0.6%** @ 40.4% |
| `corner3_l` | 2.5% @ 36.6% | 3.5% @ 38.4% | 5.2% @ **38.5%** |
| `wing3_l` | 4.6% @ 35.1% | 5.9% @ 34.9% | 9.6% @ 35.9% |
| `top3` | 4.0% @ 34.5% | 5.5% @ 34.2% | **10.3%** @ 34.8% |
| `heave` | 0.3% @ 5.7% | 0.3% @ 4.6% | 0.4% @ 15.0% |

Gates: rim make rate climbs across eras; `corner3 > wing3 > top3` in **every** era; 3PA share rises
18.1% → 38.4%; the deep corner two dies off; left/right volumes near-symmetric. Any zone missing
from any era, or a broken make-rate ordering, means the geometry is wrong — stop and fix it.

**Dials.** Re-keying per zone is an upgrade, not a cost. `SHOT_RESULT_BIAS` (`config.py:75`) is
global-only today even though the fitted per-type ideal was already known to differ (+0.22 on 2pt,
+0.35 on 3pt) with no hook to express it — it gains per-zone keys. `TYPE_BIAS` (`config.py:106`)
needs **no mechanism change**: it is already keyed head → token, so `"shot_type": {...}` works as
is; its `"assist_type": {"3pt": 0.10}` entry is re-keyed per zone.

**Next steps.**
- New root-level **`zones.py`** — the fold, the rule, `ZONE_TOKENS`, `ZONE_POINTS`, `is_three()`.
  Root level because both the cleaner and the TF-free sim layer import it. It cannot live in
  `simulation/stats.py`: `stats.py` already imports from `box_score.py`, the biggest consumer, so
  that would cycle.
- `data_cleaner.py:501-505` — stop dropping `converted_x`, `converted_y`, `shot_distance`.
- `data_cleaner.py:291-294` — emit the zone. Feeds the assist (`:306`), shot (`:324`) and block
  (`:342`) rows, all three now zoned. Gate it on shot rows: today it runs on every raw row and
  computes a bogus type for non-shots.
- `models/conditional_type_model.py:115-117` — `shot_type` and `assist_type` `target_tokens` become
  the zone tuple. This is a **loss-mask restriction only**; the head already emits over the full
  shared `type` vocab.
- `models/game_state_features.py:137` — points arithmetic through `ZONE_POINTS`. Must stay
  bit-identical to `box_score.py` or the trained score feature desyncs from the box score;
  `tests/test_game_state_features.py:176` pins this.
- `simulation/box_score.py:218,236` — 3PA/3PM and points through the lookup. `:230`'s silent
  `else: # 2pt` fallback becomes explicit; an unknown token must not quietly score 2.
- `simulation/controller.py:48,59` — `SHOT_TYPES`, `FIELD_GOAL_TYPES`. `:279`, `:325` — scoring.
- `config.py:236` — any new dial name must be added to `_TUNING_KEYS` or `set_dial` raises.
  Obey the call-time-read contract (`config.<DIAL>`, never `from config import`), enforced by
  `tests/test_dials.py:110`.
- Tests carrying `"3pt"` literals: `test_controller.py:258-280`, `test_input_cache.py:53,120,224`,
  `test_game_state_features.py:46,137,161,182`, `test_box_score.py:35-36`,
  `test_model_persistence.py:210,217`, `test_predict_game.py:70,85`.
- New `tests/test_zones.py` — both baskets map to the same token, each boundary, raw-marker-wins.

## 4. Learned free-throw counts

**What it is.** The `shooting` foul token splits into `shooting 2pt` and `shooting 3pt`. **Two
tokens, not fifteen** — fouls do not get zone granularity.

**Why.** Today the controller decides two or three free throws by sampling the live-shot type head,
which was never trained to answer that question. The foul-type head should learn the real share of
three-shot trips in real game context: who is fouling, who is shooting, where in the game.

**How it works.** The cleaner labels each shooting foul from the following trip's `outof`. The
controller reads the token and awards that many free throws, **deleting the phantom shot-type
sample** at `controller.py:540`. And-1s stay a structural check on the previous made basket
(`controller.py:530-533`) and become properly reachable: the controller remembers the scorer, and a
foul sampled as the very next play by the scored-on team resolves as one free throw to that scorer.

Note this is a **cleaner labeling step, not a model input**. Carrying `num`/`outof` onto free-throw
rows as features is a separate idea, deferred — see [Not in 2.0](#not-in-20).

**Next steps.**
- `data_cleaner.py:501-505` — stop dropping `outof`.
- `data_cleaner.py:443-459` — label shooting fouls from the following trip's `outof`.
- `simulation/controller.py:516-545` — read the token; delete the `predict_type("shot_type", ...)`
  call at `:540`.
- `simulation/controller.py:52` — `FOUL_TYPES` gains the two tokens, loses `shooting`.

## 5. The fouled player

**What it is.** Foul rows carry the fouled player in `secondary_player`, from the raw `opponent`
column.

**Why.** Drawing fouls is a skill — rim pressure, shooting motion, being the player in the bonus —
and the model cannot currently represent it.

**How it works.** `secondary_player` already exists, already shares the player embedding
(`encoder/encoder.py:81`), and already has a `none` token, so every head sees the fouled player
through history at **zero architectural cost**. The controller writes the fouled player into the
foul row and uses that player as the free-throw shooter, replacing today's independent draw.

**Next steps.**
- `data_cleaner.py:443-459` — write `opponent` into `secondary_player` (`none` for technicals).
- `simulation/controller.py:576` — `_pick_shooter` gives way to the fouled player from the row.

## 6. Timeouts and team rebounds

**What it is.** The two events that most often make the ball dead become real events.

**Why.** Without timeouts the sim has no way to substitute after a made basket — the single biggest
reason its rotations look nothing like a real game. And the dead-ball rebound is currently a coin
flip on a dial rather than something the model learns.

**How it works.**

*Timeouts.* A new sampled event, `timeout / none / <home|away>`. The calling team rides in the
`type` field, predicted by a **seventh conditional type head** — one line in the existing spec
table. Masked to dead balls or the team in possession, under a per-team budget: 7 per game, 4 in
the fourth quarter, 2 in the last three minutes, +2 per overtime.

*Team rebounds.* Kept in the data as `rebound / none / <offensive|defensive>`. The rebound-type
head grows to four tokens (`offensive`, `defensive`, `team offensive`, `team defensive`) and the
controller skips the rebounder pick on a team token. **`DEADBALL_REBOUND_PROB` goes away** — the
head learns the share. Team rebounds count on the team line of the box score.

**Next steps.**
- `models/conditional_type_model.py:114` — add the `timeout_team` spec.
- `simulation/controller.py:44` — `timeout` joins the legal event set, gated on dead ball + budget.
- `simulation/controller.py:63` — `REBOUND_TYPES` grows to four.
- `simulation/controller.py:358` and `config.py:222` — delete `DEADBALL_REBOUND_PROB` and its use.
- `config.py:236` — drop it from `_TUNING_KEYS` too.
- `simulation/box_score.py` — team rebounds on the team line.

## 7. Schema cleanup

**What it is.** One row per play where we currently emit two, and rows we should never have kept.

**Why.** The pair encodings make the event head learn a two-row grammar that the controller then
has to reproduce exactly; collapsing them removes a whole class of drift between cleaner and sim.

**How it works.**

- **Steals** become one turnover row with the stealer as `secondary_player`, instead of two.
- **Offensive fouls** become one foul row with no trailing turnover row. The box score counts the
  turnover from the foul.
- **Standalone technicals** (defensive three seconds, double technicals, coach technicals) become
  normal technical-foul rows where a player is named.
- The raw columns 2.0 needs — `outof`, the shot coordinates, `shot_distance` — stop being dropped.
- **Not kept:** jump balls (a coin flip plus the period rule covers possession), period-end rows
  (dead-ball state lives in the controller), `possession` as a column, raw ejection and violation
  rows (already represented by flagrant-2 and turnover rows).

**Next steps.**
- `data_cleaner.py:396-420` — collapse the steal pair; `simulation/controller.py:341-342` must match.
- `data_cleaner.py:443-459` — offensive foul emits one row.
- `data_cleaner.py:501-505` — the drop list.
- `data_cleaner.py:57-61` — `self.output_columns` is dead code (never read) and already stale.
  Delete it, or make it the enforced schema contract. Do not leave a third stale copy.

---

# Phase 3 — Model

## 8. A rotation model

**What it is.** Substitutions move inside the model. Three pieces replace the stint-length
scheduler and the fatigue nudge.

**Why.** Player minutes are the single largest remaining box-score error, and they are currently
produced by a timer with a dial on it.

**How it works.**

*Per-player live state.* The roster encoder already takes one number per on-court player (days of
rest), projected and added to the player embedding at `roster_set_encoder.py:120`. 2.0 adds three
more: seconds in the current stint, minutes played so far, and personal fouls. Every head sees them
wherever it consumes the lineup — a five-foul player fouls less, a tired one shoots worse — not
just the rotation heads.

*A bench bundle.* A second set input: up to ten available bench players, each with seconds since
they sat down (or since tip-off if unused), minutes played, personal fouls, and whether they have
played. It goes through the same set encoder. The incoming-player pick keeps its current output and
availability mask, plus a per-candidate score computed from those numbers — so "he sat down nine
seconds ago" is a learned penalty, not a dial.

*A `sub_decision` head.* Asked only at dead balls. At each one it predicts, per team, how many
substitutions follow before the ball is live again (`0 / 1 / 2 / 3+`). Trained on dead-ball
positions only, so it is never asked anywhere it did not learn. The controller samples the count per
team, then for each substitution picks who comes off (the player head over the five, now seeing
stint time and fouls) and who comes on (the substitution head over the bench, now seeing bench rest).

*What goes away.* The stint-length head, the scheduler, the fatigue nudge and the stint dials. The
per-team maximum gap without a sub stays as the single backstop. A hard minimum bench time exists as
a dial but ships off.

*Smaller version, if needed.* Keep the stint scheduler with dead-ball gating and ship only the two
bundles. Bench rest is still learned. This does not block anything else.

**Next steps.**
- `models/roster_set_encoder.py:71,102,120` — three more per-player scalars alongside `rest_proj`.
- `models/roster_set_encoder.py:131-179` — `RosterEncoderParams` is a frozen dataclass with a
  `get_config`/`from_config` round-trip. **Every new param must be added to both** or weight reload
  breaks.
- New `models/sub_decision_model.py`; register in `models/registry.py:23,41`.
- `simulation/controller.py:411,420,447` — `_schedule_stint`, `_process_scheduled_subs` and
  `_fatigue_bias` go; `_maybe_force_sub` stays as the backstop.
- `models/stint_length_model.py` — retire.
- `config.py` — `STINT_*` and `SUB_FATIGUE_WEIGHT` dials out of `_TUNING_KEYS` (`:236`).

## 9. Local context

**What it is.** Two ways of giving the model the play at hand.

**Why.** An attention head builds each row's representation as a weighted average over every earlier
row, and nothing pushes any head toward the last few. Basketball is overwhelmingly local.

**How it works.**

*Local attention heads.* Two of the eight heads in every block are restricted to the last eight rows
by a banded mask — the same mechanism as the padding mask. Six heads stay global. No new inputs, no
custom kernel, and a config switch turns it off for an A/B. **The backbone loop is currently
duplicated across the six model files, so it is extracted into one shared builder first** — that
refactor is a prerequisite, not a nice-to-have, and should land as its own commit with tests green
so a regression in it stays separable.

*A shot-clock proxy.* One new per-row number: seconds since the current possession started, reset on
a change of possession or an offensive rebound, scaled to the 24-second clock. Derived by the same
scan that builds the game-state features (`GameStateScan`, `game_state_features.py:101`), so
training and the simulator compute it identically by construction. Shot-clock pressure drives shot
selection, shot-clock turnovers and the timing of the next event, and it is hard for attention to
reconstruct on its own. If diagnostics say it does nothing, it is one key to remove.

*Not chosen.* An explicit last-five-events input block is the fallback if the local heads show
nothing. ALiBi needs a custom attention layer and waits.

**Next steps.**
- Extract the shared backbone builder from the six model files. Own commit, tests green.
- Banded mask + `LOCAL_ATTENTION_HEADS` config switch.
- `models/game_state_features.py:101-153` — `GameStateScan` gains the possession clock;
  `GAME_STATE_KEYS` (`:47`) and `_NORM` (`:54`) gain the key.
- `simulation/input_cache.py:178` — the incremental path picks it up for free via `GameStateScan`.

## 10. Training changes

**Game-state features, trained.** Score difference, score total, period, time left and team fouls
are already plumbed and tested; 2.0 is the first train whose weights consume them. This is where
end-game behavior comes from: leads held, trailing teams fouling, bonus-aware foul value. The only
code touch is the `ZONE_POINTS` fix in [§3](#3-shot-zones).

> **SUPERSEDED 2026-09-09 — the paragraph below was NOT built.** Clutch weighting was started
> and dropped: the features it keys on are already model inputs, no train has ever consumed them,
> so there is no evidence the model under-fits end-game — and the programme is scored on
> box-score accuracy over whole games, where the great majority of every box score comes from
> non-clutch rows. See `v2_progress.md` correction S and workstream 12. The rest of §10, the loss
> masking, IS being built. Text kept as the record of what was planned.

**Clutch weighting.** Rows that are close and late — `period_idx >= 3` and
`period_time_left <= 300` and `abs(score_diff) <= 8` — count double in every head's loss. Start at
2.0, not higher; `CLUTCH_LOSS_WEIGHT = 1.0` disables it, which makes an A/B against the same
preprocess trivial. Watch the per-quarter splits ([§11](#11-per-quarter-eval-splits)) for
early-game drift.

**Loss masked to the positions the sim queries.** The event and time heads stop training on rows the
simulator never asks about: continuation rows (a free throw after a foul, the shot after an assist,
the block after a blocked shot), controller-forced rows (substitutions), and for the time head the
last row before a period break. The continuation rule is **one function shared with the controller's
play expansion** so the two cannot drift.

**Dials from zero.** After the 2.0 train the calibration dials are re-measured from scratch rather
than carried forward. Some of today's values are compensating for the masking artifact.

**Next steps.**

> The first four bullets are the clutch weighting and are **superseded** — see the banner above.
> The last one (the test gap) still applies to the masking work.

- New `apply_clutch(mask, split)` in `models/game_state_features.py`, as a sibling of
  `apply_recency` (`models/season_features.py:162`) — the single funnel all six heads already call.
- Call it alongside `apply_recency` at the six `_make_dataset` sites: `event_time_model.py:685`,
  `player_model.py:463`, `conditional_type_model.py:551`, `substitution_model.py:595`,
  `stint_length_model.py:491`, `conditional_time_model.py:375`.
- **Two shape traps.** The recency weight is per-*game*, `(N,)` reshaped to `(-1,1)`; the clutch
  weight is per-*row*, `(N, SEQ)`. The game-state columns in `split` are `(N, SEQ, 1)` and need a
  reshape. Normalization is fixed constants (`_NORM`, `:54`), so thresholds convert to normalized
  units exactly — there is no need to carry raw arrays alongside.
- `config.py` — `CLUTCH_LOSS_WEIGHT` and the window bounds. These are **training** knobs, so they do
  **not** go in `_TUNING_KEYS`.
- `tests/test_game_state_wiring.py` — covers event_time, conditional_type, conditional_time and
  stint_length but **not** `player` or `substitution`. Close that gap.

---

# Phase 4 — Measurement

## 11. Per-quarter eval splits

**What it is.** Period-sliced box scores in the eval record and the report, plus a per-zone shot-mix
diagnostic.

**Why.** Every eval record is whole-game, with zero period-aware metrics anywhere in
`simulation/`, `reporting/` or `evaluate.py` — nothing splits by quarter. Build this **before**
the train, so the train is measurable.

> **Updated 2026-09-09.** This section originally justified itself by §10's clutch weighting — watch
> the splits for Q1 drift. That weighting was dropped, and the real reason is stronger: per-quarter
> splits are the only way to see whether the model gets end-game basketball right at all, which is
> also the evidence that would justify revisiting the weighting. The per-zone shot-mix half was
> never about clutch.

**How it works.** The per-game record carries period-sliced boxes alongside the whole-game one,
using the same period constants `GameStateScan` uses. The report gains a per-quarter section that
reuses the existing stat-registry-driven table builder, so per-quarter pace / eFG / points come
nearly free.

**Next steps.**
- `simulation/evaluation.py:296-330` — period-sliced boxes on the per-game record.
- `reporting/eval_report.py:591-613` — a `_quarter_section` in the sections list, reusing
  `_accuracy_section` (`:412`).
- `reporting/eval_report.py:639` — `box_quarters.parquet` alongside the existing per-run frames.
- `simulation/diagnostics.py:144` — per-zone shot-mix histogram in `compare_holdout`. It is a
  distribution comparison, not a per-player accuracy stat, so that is the right home.

---

## The path forward

1. **Phase 1 — controller rules.** No new tokens, so evaluate on the current weights and **take
   that eval**. It is the only uncontaminated measurement in the programme. Expect movement in FTA,
   PF, and the offensive/defensive foul split.
2. **`zones.py` + `tests/test_zones.py`.** Geometry alone, no consumers.
3. **Phase 2 — one re-clean.** Zones, the shooting-foul split, the fouled player, team rebounds,
   timeouts, technicals, the steal and offensive-foul collapse, and the kept columns. **Delete
   `encoder/vocabs/*.json` before rebuilding** — the vocabs are append-only and would otherwise keep
   the dead tokens. Run the [§3](#3-shot-zones) validation table against the re-clean.
4. **Phase 3 — model.** Shared backbone builder first, then local heads, the shot-clock proxy, the
   live-state and bench bundles, the `sub_decision` head, and the loss masks (boundary and
   continuation; the clutch mask was dropped — see the banner in §10).
5. **Controller consumers of the new tokens.** Zones, the free-throw token, team rebounds, timeouts
   and their budget, the fouled player as shooter, the rotation loop at dead balls.
6. **Phase 4 — per-quarter eval.** Before the train.
7. **Train 2.0.**

```bash
python train.py --full --name full_train_3 --batch-size 64 --clean --rebuild-vocabs
```

Train 2's availability masking and capacity settings carry forward unchanged.

**Pre-train checklist.**
- `pytest tests/` green.
- **Check `encoder/vocabs/norm_stats.json` and the vocab files before training.** pytest overwrites
  the committed encoder artifacts; confirm the freeze happened from a real clean, not test residue.
- Zone validation table matches [§3](#3-shot-zones) in all three sampled eras.
- Derived-vs-raw 3pt disagreement under 1% per season.

**Post-train.** Per-zone make rates against the era table; the per-quarter section read for the
end-game behaviour the model actually produces; then the dial package from zero, re-keyed per
zone. (The clutch A/B that stood here is gone with the weighting.)

## Risks

- **Eleven workstreams land in one retrain and their effects confound.** Phase 1 is the one clean
  measurement point. If the post-train result is ambiguous, the local-attention config switch is
  the **only** cheap ablation available without a re-preprocess — dropping clutch weighting removed
  the other one, which makes this risk larger, not smaller.
- **[§8](#8-a-rotation-model) and [§9](#9-local-context) are each large enough to be their own
  train.** Both have documented smaller versions; cutting either does not block the rest.
- **Every dial fitted against full1/full2 is invalidated** by the vocab rebuild. Expect the first
  eval to look worse than full2 before the dial cycle runs. Do not read raw post-train numbers as a
  verdict on any single feature.
- **`mid_corner_l/r` is thin in the modern era** — 0.6% of shots by 2022-23. It degrades gracefully,
  since `shot_result` is a softmax conditioned on the player embedding rather than a per-player
  lookup table, but merging it into `mid_wing` is a cheap retreat if the first eval shows noise.
- **`heave` could be sampled at absurd times** if the game-state features underperform. Watch heave
  frequency by period in the new per-quarter diagnostic.

## Not in 2.0

| Item | Reason |
|---|---|
| Free-throw index (`num`/`outof` as row features) | Deferred pending review: local attention ([§9](#9-local-context)) should learn first-of-two vs second-of-two from the sequence without an explicit feature. Distinct from the [§4](#4-learned-free-throw-counts) cleaner labeling step, which **is** in 2.0 |
| Player age | External roster join with its own name-matching work; see `docs/v3_planned_changes.md` |
| Coach | External table; confounded with team-era; see `docs/v3_planned_changes.md` |
| Player height, position | Dropped permanently — not in 2.0 and not planned for 3.0 |
| One row per play | Would subsume the pair collapse and the loss masking, but touches every head and the decoder |
| Conditioning foul type on the fouled player | Needs a new condition type in the spec machinery; decide after the passive version is measured |
| ALiBi recency | Needs a custom attention layer; the masked heads get the bias without one |
| A dedicated minutes model | The rotation model is the first real attempt at the minutes gap; decide after |
| Calibration auxiliary losses, relative offense/defense encoding, RL fine-tuning | Parked as before |
