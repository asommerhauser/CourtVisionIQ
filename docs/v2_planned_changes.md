# CourtVisionIQ 2.0 — New Features

> The path forward for model 2.0: what the new features are, how each one works, and the order
> we build them in. Nothing here is built yet.

---

## What 2.0 adds

| Feature | In one line |
|---|---|
| [Shot zones](#1-shot-zones) | Shots are located on the court in seven zones instead of just 2pt / 3pt |
| [Learned free-throw counts](#2-learned-free-throw-counts) | Whether a shooting foul is worth two or three free throws is a learned token, not a guess |
| [The fouled player](#3-the-fouled-player) | Foul rows name who was fouled, and that player shoots the free throws |
| [Side-aware fouls](#4-side-aware-fouls) | Foul types are sampled knowing whether the fouler is on offense or defense, and free throws always go to the fouler's opponent |
| [Dead balls, timeouts, team rebounds](#5-dead-balls-timeouts-and-team-rebounds) | The controller knows when the ball is dead; timeouts and team rebounds are real events |
| [A rotation model](#6-a-rotation-model) | Substitutions are decided at dead balls by a head that sees stint time, minutes, fouls and bench rest |
| [Local context](#7-local-context) | Attention heads that focus on the last few plays, plus a shot-clock proxy |
| [Training changes](#8-training-changes) | Game-state features trained, clutch moments weighted, loss masked to the positions the sim actually queries |
| [Schema cleanup](#9-schema-cleanup) | One row per play for steals and offensive fouls; standalone technicals |

All of the data changes land in one re-clean and one full retrain.

---

## 1. Shot zones

**What it is.** The `shot_type` head predicts one of seven court zones instead of `2pt` / `3pt`:
`restricted_area`, `paint`, `short_mid`, `long_mid`, `corner3`, `abovebreak3`, `heave`.

**Why.** Shot selection and shot efficiency are spatial. The model currently learns one make
rate per player per point value; 2.0 learns it per zone, so a rim finisher and a long-two
shooter stop looking alike, the rebound head can see where a miss came from, and the three-point
revolution becomes a learnable spatial fact rather than something absorbed into player
embeddings.

**How it works.**

- The cleaner derives the zone from the raw shot coordinates with one fixed geometric rule,
  the same across all 21 seasons. The raw 2/3 marker wins where geometry is ambiguous at the arc.
- The zone implies the point value, so a small lookup replaces every `"3pt"` check in the code.
- Assist rows keep a coarse `2pt` / `3pt`; the made shot that follows an assist samples a zone in
  that class.
- Shot dials re-key per zone: a zone-mix dial and per-zone make-rate biases replace the single
  eFG knob.

---

## 2. Learned free-throw counts

**What it is.** The `shooting` foul token splits into `shooting 2pt` and `shooting 3pt`. The
free-throw index (`num` / `outof`) is carried onto free-throw rows.

**Why.** Today the controller decides two or three free throws by sampling the live-shot type
head, which was never trained to answer that question. In 2.0 the foul-type head learns the real
share of three-free-throw trips in real game context — who is fouling, who is shooting, where in
the game.

**How it works.**

- The cleaner labels each shooting foul from the following trip's `outof`.
- The controller reads the token and awards that many free throws.
- And-1s stay a structural check on the previous made basket, and become reachable: the
  controller remembers the scorer, and a foul sampled as the very next play by the scored-on
  team resolves as one free throw to the scorer.

---

## 3. The fouled player

**What it is.** Foul rows carry the fouled player in `secondary_player`, from the raw `opponent`
column.

**Why.** Drawing fouls is a skill — rim pressure, shooting motion, being the player in the
bonus — and the model cannot currently represent it. `secondary_player` already exists, already
shares the player embedding, and already has a `none` token, so every head sees the fouled
player through history at no architectural cost.

**How it works.**

- The cleaner writes `opponent` into `secondary_player` on foul rows (`none` for technicals).
- The controller writes the fouled player into the foul row and uses that player as the
  free-throw shooter, replacing today's independent shooter draw.

---

## 4. Side-aware fouls

**What it is.** A rule the controller enforces: after the fouler is sampled, the foul-type head
is masked to the types that side can commit, and the outcome resolves for that side.

| Fouler is on | Can commit | Outcome |
|---|---|---|
| offense | `offensive`, `loose ball`, `technical`, `flagrant` | offensive foul → turnover; loose ball → possession unchanged, team foul; flagrant or technical → free throws to the defense |
| defense | everything except `offensive` | common foul → offense keeps the ball, or two free throws in the penalty; shooting, take, flagrant and technical fouls as today |

**Why.** The foul-type head has no idea which side the fouler is on, so it can type an offensive
player's foul as a defensive one, or charge an offensive foul to a defender. Masking by side makes
every foul resolve to a legal outcome, and it is what lets the bonus fire when it should.

**Also part of this.**

- Free throws always go to the fouler's opponent, in every branch: shooting, bonus, take,
  flagrant, technical.
- A player's team comes from a fixed map built at tip-off, never from who is currently on the
  floor, and it is resolved before the foul is charged — so a foul-out substitution can no longer
  change whose free throws they are.
- Team fouls use one definition, shared with the game-state features: every foul except
  technical and offensive, on either side. The last-two-minutes penalty triggers on the second
  foul committed inside the window.

---

## 5. Dead balls, timeouts, and team rebounds

**What it is.** The controller tracks whether the ball is dead, and the stream gains the two
events that most often make it dead.

**Dead-ball state.** Internal to the controller, never written to the data.

| Dead after | Live after |
|---|---|
| any foul; a non-steal turnover; the last free throw if made; a team rebound; a timeout; a period boundary; a made basket in the last minute of Q1–Q3 or the last two minutes of Q4 / OT | a made basket otherwise; a missed or blocked shot; a live rebound; a steal; the last free throw if missed |

Substitutions happen only when the ball is dead ([§6](#6-a-rotation-model)).

**Timeouts.** A new sampled event, `timeout / none / <home|away>`. The calling team rides in the
`type` field, predicted by a seventh conditional type head — one line in the existing spec
table. Timeouts are masked to dead balls or the team in possession, under a per-team budget
(7 per game, 4 in the fourth quarter, 2 in the last three minutes, +2 per overtime). This is how
the sim gets a substitution after a made basket: the team about to inbound calls time.

**Team rebounds.** Kept in the data as `rebound / none / <offensive|defensive>`. The
rebound-type head grows to four tokens (`offensive`, `defensive`, `team offensive`,
`team defensive`) and the controller skips the rebounder pick on a team token. The dead-ball
rebound dial goes away; the head learns the share. Team rebounds count on the team line of the
box score.

**One possession tracker.** The controller's possession is the only one; the simulator's second
copy is removed. Possession stays a sampling guide — it is not a data column and not a model
input. Opening possession is a coin flip; Q2 and Q3 go to the tip loser, Q4 to the winner,
overtime re-flips.

**Clean period boundaries.** If a sampled gap would cross the period end, the clock stops at
the boundary, team fouls reset, the ball is dead, and the play is re-sampled in the new period.
No event straddles a boundary.

---

## 6. A rotation model

**What it is.** Substitutions move inside the model. Three pieces replace the stint-length
scheduler and the fatigue nudge.

**Per-player live state.** The roster encoder already takes one number per on-court player
(days of rest). 2.0 adds three more: seconds in the current stint, minutes played so far, and
personal fouls. Every head sees them wherever it consumes the lineup — a five-foul player fouls
less, a tired one shoots worse — not just the rotation heads.

**A bench bundle.** A second set input: up to ten available bench players, each with seconds
since they sat down (or since tip-off if unused), minutes played, personal fouls, and whether
they have played. It goes through the same set encoder. The incoming-player pick keeps its
current output and availability mask, plus a per-candidate score computed from those numbers —
so "he sat down nine seconds ago" is a learned penalty, not a dial.

**A `sub_decision` head.** Asked only at dead balls. At each one, the head predicts, per team,
how many substitutions follow before the ball is live again (`0 / 1 / 2 / 3+`). It is trained on
dead-ball positions only, so it is never asked anywhere it did not learn. The controller samples
the count for each team, then for each substitution picks who comes off (the player head over
the five, now seeing stint time and fouls) and who comes on (the substitution head over the
bench, now seeing bench rest).

**What goes away.** The stint-length head, the scheduler, the fatigue nudge and the stint dials.
The per-team maximum gap without a sub stays as the single backstop. A hard minimum bench time
exists as a dial but ships off.

**Smaller version, if needed.** Keep the stint scheduler with dead-ball gating and ship only the
two bundles. Bench rest is still learned.

---

## 7. Local context

**What it is.** Two ways of giving the model the play at hand.

**Local attention heads.** An attention head builds each row's representation as a weighted
average over every earlier row, and nothing pushes any head toward the last few. In 2.0, two of
the eight heads in every block are restricted to the last eight rows by a banded mask — the same
mechanism as the padding mask. Six heads stay global. No new inputs, no custom kernel, and a
config switch turns it off for an A/B. The backbone loop is currently duplicated across the six
model files, so it is extracted into one shared builder first.

**A shot-clock proxy.** One new per-row number: seconds since the current possession started,
reset on a change of possession or an offensive rebound, scaled to the 24-second clock. It is
derived by the same scan that builds the game-state features, so training and the simulator
compute it identically. Shot-clock pressure drives shot selection, shot-clock turnovers and the
timing of the next event, and it is hard for attention to reconstruct on its own. If diagnostics
say it does nothing, it is one key to remove.

**Not chosen.** An explicit last-five-events input block is the fallback if the local heads show
nothing. ALiBi needs a custom attention layer and waits.

---

## 8. Training changes

**Game-state features, trained.** Score difference, score total, period, time left and team
fouls are already plumbed and tested; 2.0 is the first train whose weights consume them. This is
where end-game behavior comes from: leads held, trailing teams fouling, bonus-aware foul value.

**Clutch weighting.** Rows that are close and late — fourth quarter or overtime, under five
minutes, within eight points — count double in every head's loss. Start at 2.0 and watch the
per-quarter splits for drift.

**Loss masked to the positions the sim queries.** The event and time heads stop training on
rows the simulator never asks about: continuation rows (a free throw after a foul, the shot after
an assist, the block after a blocked shot), controller-forced rows (substitutions), and for the
time head the last row before a period break. The continuation rule is one function shared with
the controller's play expansion so the two cannot drift.

**Dials from zero.** After the 2.0 train the calibration dials are re-measured from scratch
rather than carried forward. Some of today's values are compensating for the masking artifact.

---

## 9. Schema cleanup

- **Steals** become one turnover row, with the stealer as `secondary_player`, instead of two.
- **Offensive fouls** become one foul row, with no trailing turnover row. The box score counts
  the turnover from the foul.
- **Standalone technical fouls** (defensive three seconds, double technicals, coach technicals)
  become normal technical-foul rows where a player is named.
- The raw columns 2.0 needs — `outof`, the shot coordinates, `shot_distance` — stop being dropped
  at load.
- **Not kept:** jump balls (a coin flip plus the period rule covers possession), period-end rows
  (dead-ball state lives in the controller), possession as a column, raw ejection and violation
  rows (already represented by flagrant-2 and turnover rows).

---

## The path forward

1. **Controller rules first.** Side-aware fouls, free-throw attribution, the fixed team map,
   dead-ball state, one possession tracker, clean period boundaries. None of these need new
   tokens, so they can be evaluated on the current weights before anything else moves.
2. **One re-clean.** Shot zones, the shooting-foul split, the free-throw index, the fouled
   player, team rebounds, timeouts, technicals, the steal and offensive-foul collapse. Delete
   `encoder/vocabs/*.json` before rebuilding — the vocabs are append-only and would keep the old
   tokens.
3. **Model changes.** The shared backbone builder and local heads, the shot-clock proxy, the
   live-state and bench bundles, the `sub_decision` head, the clutch and boundary masks.
4. **Controller consumers of the new tokens.** Zones, the free-throw token, team rebounds,
   timeouts and their budget, the fouled player as shooter, the rotation loop at dead balls.
5. **Train 2.0** with `--clean --rebuild-vocabs`, then diagnostics and a dial package from zero,
   re-keyed per zone.

---

## Not in 2.0

| Item | Reason |
|---|---|
| Player age | External roster join with its own name-matching work; adds later as a season-context column |
| Coach | External table; confounded with team-era |
| One row per play | Would subsume the pair collapse and the loss masking, but touches every head and the decoder |
| Conditioning foul type on the fouled player | Needs a new condition type in the spec machinery; decide after the passive version is measured |
| ALiBi recency | Needs a custom attention layer; the masked heads get the bias without one |
| A dedicated minutes model | The rotation model is the first real attempt at the minutes gap; decide after |
| Calibration auxiliary losses, relative offense/defense encoding, RL fine-tuning | Parked as before |
