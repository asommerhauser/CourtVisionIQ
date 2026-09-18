# CourtVisionIQ 3.2 — Direction

**Status: proposal, 2026-09-18.** Written from the 3.0 direction document, the 3.0+ addendum
(`v3_direction.md` §9), the v2-run4 evaluation, and the run-4 training logs. Every cost in §6 is
derived from a measured rate in this repository — `train-version2.clean.log` for epoch wall times,
`v3_direction.md` §3 W4 for the batched-rollout throughput — and every "measured now" figure in §7
is from the 2.0 evaluation. Estimates are labelled as estimates.

This document supersedes nothing. `v3_direction.md` stays the standing direction; 3.2 is the next
build against it, and it **reverses two of that document's decisions with stated reasons** (§2).

**Decisions taken (2026-09-18).** All twelve heads train on the recency-weighted subset. The corpus
is cut at 2008. The player vocabulary is rebuilt from what remains, behind a minimum-games floor.
W4 rung 3 is built, in the cheap form specified in §4. The season signal moves into the roster set
encoder. Architecture work is split across 3.2 and 3.3 on an attribution argument, not a cost one
(§5).

---

## 1. What this cycle is for

3.0 answered its first gate and the answer reprioritised everything after it. Measured on v2-run4's
5,357 sim play-by-plays against the real 2022-23 season:

| behaviour | sim | real | ratio |
|---|---|---|---|
| 3rd foul before half → off the floor within 60 s | 0.238 | 0.776 | 0.31x |
| 4th foul before half → off within 60 s | 0.280 | 0.961 | 0.29x |
| 4th-foul events per game | 0.371 | 0.039 | 9.51x |
| Q4 starter seconds, blowout ÷ close | 0.838 | 0.525 | 1.60x |
| trailing-team fouls per 100 s, last 2:00 down 4–9 | 1.62 | 1.99 | 0.81x |

The model has the inputs. `court_fouls_*`, `score_diff` and the period clock all reached the weights
in 2.0 and produced none of the behaviour. **No next-step loss can see this**, because each
individual prediction is roughly right and it is the composition over hundreds of steps that is
wrong.

That splits the programme's remaining error into two kinds, and 3.2 attacks them with different
tools:

- **Composition.** The simulator does not behave like a basketball game over hundreds of steps.
  Fixed by changing what the training objective optimises (§4) and by giving the heads the scales
  that behaviour lives at (§5).
- **Identity.** The model predicts who a player *was*. §1b of the 3.0 direction: rookies 10.9%
  worse than "use his season average", role-shifters 24% worse, Bane at −8.7 ppg bias. Fixed by
  inputs and by vocabulary surgery (§3). **The rollout objective cannot touch this** — no amount of
  reweighting a sampling distribution teaches a model a fact it was never given.

Keeping those two straight is the single most important thing in this document. Most of the
expected-improvement table in §7 is a statement about which of the two a given item addresses.

---

## 2. Two reversals of 3.0's decisions

Both are recorded here rather than edited into `v3_direction.md`, so the original reasoning stays
readable and the change is attributable.

### 2.1 Data restriction was rejected; 3.2 adopts it

**3.0's reasoning** (§1f): restricting the corpus to 2010+ "removes data the loss weight already
floors at 0.05 and makes every rare token rarer; it does not touch the static-identity problem."

**Both clauses are correct.** The reversal is that 3.0 evaluated the cutoff as a *fix* for identity,
which it is not, and did not evaluate it as a *simplification* enabling vocabulary surgery, which it
is. Old seasons are already discounted twice — once by `SUBSET_RECENCY_HALFLIFE_SEASONS = 5.0` in
sampling, once by `RECENCY_HALFLIFE_SEASONS = 3.0` / `RECENCY_FLOOR = 0.05` in the loss:

| season, relative to newest | subset sampling | loss weight | combined |
|---|---|---|---|
| newest | 1.000 | 1.000 | 1.000 |
| 7 back | 0.250 | 0.200 | 0.050 |
| 12 back | 0.125 | 0.063 | 0.008 |
| 17 back | 0.063 | 0.050 | 0.003 |

A 2006 game contributes roughly three thousandths of what a current game does. Across 2003–2007 the
total is on the order of one percent of the gradient. **Expect the cutoff to be neutral on every
metric.** It is adopted for what it unlocks in §3.2, not for what it does on its own.

**The cost 3.0 named still stands:** rare tokens get rarer. At a 0.05 weight those instances carried
little gradient, but zero is not small, and the rarest foul and rebound sub-types are where it would
show.

### 2.2 The subset was for the small heads only; 3.2 puts all twelve on it

**3.0's reasoning** (`training/subset.py`): "The big player-vocab heads (player / substitution /
sub_decision) want every game." The small categorical heads saturate and then overfit on the full
corpus; the big heads need volume for their embeddings.

**The reversal is a decision, not a measurement.** It is taken for consistency and compute, and the
risk is real and stated: moving `player`, `substitution` and `sub_decision` from 26,267 games to
roughly 5,200 thins every player embedding, and that lands on the identity axis which is already the
larger of the two failures.

**Three things mitigate it, and they are the reason it is acceptable rather than reckless.**

1. The newest season enters the subset at rate 1.0, so players who appear in the scored holdout keep
   most of their recent games. What thins is older-era players who rarely appear in what is scored.
2. The player priors (3.0 W2.1) carry fourteen per-player season-to-date scalars into the set
   encoder additively, before attention. A player whose embedding is thin still arrives carrying his
   production profile.
3. §3.2's vocabulary floor removes the players who would have had the thinnest embeddings, rather
   than leaving them in the table under-trained.

**Consequence for coverage-completeness.** The subset sampler currently guarantees every player at
least one game so no embedding trains on zero rows. Under a minimum-games floor that guarantee is
inert — anyone it rescues with a single game falls below the floor anyway — and it drags old games
into the sample for players who will map to `UNK` regardless. **Coverage-completeness is retired**
(§3.2).

---

## 3. Data and inputs

Everything in this section is **free before the retrain and a retrain afterwards**. It all lands
first, and the retrain happens once.

### 3.1 Corpus cut at 2008

`data_loading` / `training.chronology` gain a minimum-season bound. Seasons 2003–2007 leave the
training pool. Expect roughly 300 games to leave the *subset*, since those seasons were already
sampled at four to six percent — the subset goes from about 5,500 to about 5,200. Both figures are
estimates; `training/subset_games.json` is generated at extract time and is not in the repository.

**The priors sidecar is NOT cut.** `player_priors.py` walks all 21 seasons and seeds each player
from his previous season (`SHRINK_K = 10.0`, falling back to `LEAGUE_DEFAULTS`). Cut the walk at
2008 and every 2008 game seeds from league averages instead of from real 2007 production, and the
career-stage scalar in §3.4 loses its history. **Cut the training games, never the sidecar.** This
is the single easiest thing in 3.2 to get wrong, and it fails silently — the model trains, and every
2008–2009 prior is quietly wrong.

### 3.2 Player vocabulary rebuilt behind a floor

Order of operations, which matters:

1. Cut the corpus at 2008.
2. Carve the subset from what remains, with the existing recency-weighted sampler.
3. Build the player vocabulary from players clearing **a minimum-games floor within the subset**.
   Start at 20–30 games.
4. Everyone below the floor maps to `UNK`.

The embedding table is 2,153 × 192 today. §1f names it as where the memorisation lives. This shrinks
it by removing long-retired players outright and by removing the thin rows the floor catches, which
is the one capacity change in 3.2 and it goes *downward*.

**Why `UNK` is affordable now and was not in 2.0.** The fourteen prior scalars enter at
`models/roster_set_encoder.py:137` as `x = emb + scalar_proj(stacked)` — additively, before the SAB
layers. An `UNK` player loses his identity row and keeps his season-to-date production. For a
deep-bench player that is the right trade; the embedding row was mostly noise anyway.

### 3.3 Three new prior stats

`v3_direction.md` §9.5(a) prices these as free only until the retrain starts. `pf`, `ftm`, `tpm`,
`stl` and `blk` are in `BOX_STATS` and in every box score but are not accumulated.

| stat | why | head it serves |
|---|---|---|
| `ft_pct` (`ftm/fta`) | the model has an FT *attempt* rate and no *make* rate; FT% is the most stable player stat there is | free throws |
| `tp_pct` (`tpm/tpa`) | `shot_type` emits `corner3_l` / `wing3_r` / `top3` as distinct tokens and `shot_result` judges them without knowing whether the player can shoot one | `shot_result` |
| `pf_36` | who actually fouls | `foul_type` |

`stl_36` / `blk_36` stay out: rare enough that the prior is mostly noise. `fg_pct` is already in
`PLAYER_PRIOR_KEYS` and mixes shot difficulty with shooting skill; noted, not changed.

Each is one accumulator line in `player_priors.py`, one `_NORM` entry, and a sidecar rebuild of
roughly ten minutes.

**Note that `pf_36` does not fix the 9.6× over-production of fourth fouls.** That is a benching
failure, not an attribution one, and §4 is what addresses it.

### 3.4 Season signal into the set encoder

The current season plumbing is thin by design: a 16-dim `season` embedding and four team scalars,
all entering the fusion concat once, then projected to 384 and asked to survive six residual blocks.
Four per-player scalars are added at the same seam as the priors:

- **Career stage** — seasons since first appearance, clipped and normalized.
- **Season-over-season deltas** on the volume rates — this season's `pts_36`, `min_pg`, `fga_36`
  minus last season's.

§1b's failure axis *is* career games and role shift. These index it directly. The previous-season
rates are already loaded for the shrinkage seed, so the deltas are close to free.

`NUM_ROSTER_SCALARS` goes 14 → 21: three prior stats (§3.3), career stage, and three deltas. Zero fusion width is added; only the eight
*team* priors widen the backbone concat, and that already happened in 3.0.

**Two things deliberately not done.** Player-by-season embeddings would represent "who he is this
season" literally, but each slice trains on at most 80 games and that is the memorisation problem in
a new hat. Broadcasting the existing 16-wide season token into the roster slots is cheap and
harmless and carries information the backbone already has.

### 3.5 Unchanged

`RECENCY_WEIGHTING = True`, halflife 3.0, floor 0.05. The double discount in §2.1 is the reason the
cutoff is neutral; changing the weighting at the same time would confound that.

---

## 4. W4 rung 3 — training on what the simulator produces

3.0 priced rung 3 at ~160 GPU-hours per attempt and made it conditional on
`checkpoint_selection.epochs_disagree`. **3.2 builds it unconditionally, in a form that costs 3.2
GPU-hours**, because the §1 measurement makes composition the headline failure and rung 3 is the
only item in the programme that puts a gradient on it.

### 4.1 Why the gradient cannot flow through a rollout

`simulation/controller.GameController` is pure-Python rules on worker threads, and every sample goes
through numpy at `simulation/game_simulator.py:413` and `:658`. The TensorFlow graph is discarded at
each step. There is no reparameterisation trick available and no backprop through the game. Only
score-function estimators exist, which is exactly why 3.0's costing was what it was.

### 4.2 The cheap form: weighted replay

Not a new training loop. The same estimator with the cost moved.

1. **Log during the rollout.** At each queried position, keep the context tensors the head already
   built and the action it sampled. `apply_query_mask` already marks these positions — roughly 68.5%
   of rows.
2. **Score the finished game once**, from the existing `simulation/eval_metrics` aggregate.
3. **Advantage per sim** = mean score of the other nine sims of the same game, minus this one. Same
   matchup, same date, so the baseline is tightly matched and most of the variance cancels.
4. **One weighted pass.** The model's own sampled action is the label; the advantage is the sample
   weight. Positive advantage reinforces those choices, negative makes them less likely. It reuses
   `models/train_steps.build_trainer` rather than adding a second training path.

The replay pass is smaller than one normal epoch. **The sims are the cost, and they are shared: one
batch of rollouts updates all twelve heads**, each filtering to its own decisions.

### 4.3 Per-head metrics

A shared scalar punishes a head that did its job for another head's failure. That mis-assignment is
why the generic estimator needs hundreds of steps. Matching each head to the metric closest to its
own output is the largest available variance reduction and it costs nothing.

**`event_time` is scored on the histogram of its own output vocabulary.** The head emits eight real
tokens (`encoder/vocabs/event_vocab.json`), and its metric is the per-team count of each:

| token | box stat |
|---|---|
| shot | `fga` |
| rebound | `oreb`, `dreb` |
| assist | `ast` |
| turnover | `tov` |
| block | `blk` |
| foul | `pf` |
| substitution | count of substitution events |
| timeout | count of timeouts |

Six are already in `BOX_STATS`; substitution and timeout counts need a small counter off the
play-by-play, on both the sim and real side.

Three exclusions follow from that table and are deliberate. **Free throws drop out** — there is no
free-throw token; attempts are produced by the rules engine downstream of a shooting foul, so
scoring the head on them charges it for a rule it does not control. **Steals stay with
`turnover_type`** — the event head emits `turnover`, and whether it is credited as a steal is the
sub-type split. **Substitutions are counted, not converted to minutes** — the head controls how
often an opportunity fires, not who goes in or how many.

The rest:

| head | metric |
|---|---|
| `player` | the same eight counts as each player's **share** of his team's total, weighted by real minutes |
| `event_time_cond` | pace, game length |
| `shot_type` | three-point attempt rate, attempts by zone |
| `shot_result` | effective field goal percentage |
| `assist_type` | assists |
| `turnover_type` | turnovers, steals |
| `foul_type` | fouls, plus the three game-state probes |
| `rebound_type` | offensive rebound share |
| `timeout_team` | timeout counts |
| `substitution`, `sub_decision` | player minutes |

Every head also carries a **small shared term** for Brier and spread error, since the winner is a
joint product of all twelve.

**Rate-normalisation is the principle behind the two least obvious rows.** `shot_result` gets
effective field goal percentage rather than team points because it divides by attempts: a simulator
playing too fast produces wrong points, and eFG% does not blame the shooting head for it.
`shot_type` must *not* get eFG% — it controls the mix, and the cheapest way to win an efficiency
metric by changing the mix is to shoot more threes, because eFG counts them at 1.5. Same reason the
`player` head gets shares rather than raw counts.

### 4.4 Three implementation rules that decide whether this works

1. **Win probability comes from the Gaussian margin approximation**
   (`simulation.stats.score_win_prob`), never from counting which side won. From one sim the winner
   is 0/1 and the Brier term is a coin flip; the margin is continuous.
2. **Normalize every stat by its cross-game standard deviation**, frozen as constants in the style
   of the existing `_NORM` blocks. Raw averaging is dominated by `seconds` (~14,400 per team-game)
   and then by points; blocks at ~5 contribute nothing. **Drop `seconds` from the team aggregate
   entirely** — a team always plays ~240 player-minutes, so its error is information-free.
   `eval_metrics` already says this in its headline block.
3. **The dispersion guard stays on.** Mean absolute error is minimised by collapsing the spread, and
   a head that fixes the box by killing variance destroys the joint structure the whole thesis rests
   on. `rollout_score`'s `ROLLOUT_SCORE_DISPERSION_WEIGHT` term is that guard and it is not
   optional under a KPI objective.

### 4.5 Cadence and sampling

**One pass, after the main train, over one in ten of the subset, ten sims per game.** Roughly 520
games, 5,200 game-sims, ~520 update opportunities.

Three reasons for each part of that:

- **After, not during.** Mid-train, the other eleven heads are still moving, so a scored rollout came
  from a bundle that stops existing a few epochs later. Credit is assigned to a lineup that gets
  substituted before the next play.
- **Ten sims of the *same* game**, not one sim of ten games. The sibling set is what makes the
  leave-one-out baseline work.
- **Sampled from the subset, not on a flat stride over the corpus.** The subset is deliberately
  modern-heavy; a flat stride is era-neutral and would fine-tune `shot_result` toward the old game,
  partially undoing the v1.1 rate increase that exists because it was under-fitting modern
  efficiency.

**Never from a holdout window.** Selecting or fine-tuning against the holdout turns the report into a
training metric, and nothing downstream would look wrong. A stride over the training subset respects
this without anyone having to remember it.

**Do not repeat the pass per epoch.** One pass is 3.2 GPU-hours; the same pass every epoch of a
thirty-epoch stage is ~96, which is worse than the naive form this design exists to avoid.

### 4.6 Rung 2's missing bridge

`models/rollout_selection.py` is written and tested — the policy, the callback, the scoring formula,
the EarlyStopping ordering, the `epochs_disagree` record. Two things stop it running:
`ROLLOUT_SELECTION = False`, and **nothing in the repository ever constructs the `rollout_score_fn`**
that `models/event_time_model.py:865` expects. That bridge is one day of work and it is the cheapest
item in 3.2.

`rollout_score` currently reads only team points MAE, margin dispersion and the probe gaps. The
aggregate it draws from **already carries Brier, score-Brier and per-player MAE for every stat**.
Adding the §4.3 targets is a weighting change, not new machinery.

---

## 5. Architecture — what lands in 3.2, what waits for 3.3

### 5.1 The diagnosis

Two facts read off the code:

- The two rosters pass through one weight-tied encoder **independently** and meet only at the fusion
  concat (`models/backbone.py:167`). There is no structural way to represent one lineup *against*
  another.
- Season, priors, rest and the regime latent all enter **once**, as columns in a wide concat
  projected to `MODEL_DIM = 384`, and then have to survive six residual blocks on their own.

**Season is not under-weighted, it is under-plumbed.** That distinction decides §5.2.

### 5.2 In 3.2

**Context modulation (FiLM).** Build one game-context vector from the season embedding, the team
priors and the regime latent; a small network emits a per-block scale and shift applied to the
residual stream. Season and the night then shape every layer's computation instead of being one
column at the bottom. Two vectors of 384 per block — negligible parameters, no memorisation surface.
**1 day.**

**Cross-roster attention.** One attention block where each roster attends over the other before
pooling. `layers/mab.py` already has the block. Shared weights, so it generalises rather than
memorises. **1.5 days.**

### 5.3 Held for 3.3 — multi-scale time

**The design, since it is better than the one `v3_direction.md` W5 parked.** W5's cost was that
possession and stint summaries must be computed in the data pipeline *and* reproduced one event at a
time in `simulation/input_cache.py`, whose contract is that every column is a pure function of its
own row plus running scalar state. A second grain breaks that, and the project forbids computing a
feature two ways.

**Build the hierarchy inside the model graph instead.** The only new inputs are integer segment
labels per event row — possession index, stint index, period index. All three are running scalar
state, so the cache picks them up with no new incremental code. Then, after the third backbone
block:

1. Pool event hidden states by possession, causally.
2. Pool possession vectors by stint, and stint vectors by period.
3. The game level already exists: regime latent, team priors, season embedding.
4. Each event row attends over its current possession, stint, quarter and the game vector; the
   result is added back into the residual stream.

Batch and rollout run the same code by construction, and the cache contract is untouched.

**Why it waits.** Not cost — 4 to 6 days and ~1–2M parameters per head. **Attribution.** The rollout
objective is the largest unknown in the programme and deserves an A/B with nothing standing in front
of it. If the Q4 rotation probe moves under §4 and then plateaus, that is the evidence multi-scale
time is the limit, and 3.3 gets a clean before-and-after.

**What it would not fix.** The foul-benching gap. The model already has personal fouls, period and
clock at the event level and produces 0.28 against a real 0.96. That is an objective failure, not a
scale failure.

### 5.4 Not adding parameters

§1f: heads reach the base rate in ~5 epochs and then memorise; `shot_result` peaks at epoch 16 and
rises. "This is not a data-volume or capacity limit; the inputs do not contain the answer."

Widening `MODEL_DIM` / `NUM_LAYERS` now would add capacity to a model that already overfits, in the
same cycle the corpus shrinks roughly fivefold. The rollout objective makes it worse: one scalar
advantage per game is a far weaker signal than 500 labelled rows and wants *less* capacity to chew
through. §3.2's vocabulary pruning removes parameters from precisely the table §1f blames; spending
that back on width would undo it.

**Can six blocks at width 384 express "a man on his fourth foul comes off"?** Comfortably. Nothing
about the gap reads as insufficient capacity. It reads as no gradient ever having asked.

---

## 6. Cost

All rates are measured in this repository. Epoch wall times from `train-version2.clean.log`: 105 s
for subset heads, ~420 s for the four full-corpus heads, 172 epochs across the v2 run, ≈10 GPU-h
total. Rollout throughput from `v3_direction.md` §3 W4: **37 GPU-min per 1,000 batched game-sims**.

### 6.1 Producing one model

| item | GPU-h |
|---|---|
| main train, 12 heads on the subset, with §5.2 layers | 5.5 |
| rung 2 checkpoint selection | 1.3 |
| KPI rollout phase (§4.5) | 3.2 |
| **subtotal** | **10.0** |

The main train gets *cheaper* than v2 despite the new layers: the four big heads drop from 26,267
games to ~5,200, taking their epochs from ~420 s to ~84 s. That saving roughly pays for §5.2.

### 6.2 Simulation, itemised

| stage | game-sims | GPU-h |
|---|---|---|
| KPI rollout phase, 520 games × 10 | 5,200 | 3.2 |
| rung 2 checkpoint selection, ~10 evals × 200 | 2,000 | 1.3 |
| development checks, 4 × 200 games × 50 | 40,000 | 24.6 |
| final paired evaluation, 700 games × 50, two arms | 70,000 | 43.2 |
| **total simulation** | **117,200** | **72** |

**Evaluation dominates, and it is not close.** The KPI phase — the substance of §4 — is 5,200 of
117,200 sims. Trimming it saves nothing. Every lever that matters is on the evaluation side:

- **Reuse the 3.0 baseline arm** if it has already been scored on all seven windows: 21.6 GPU-h not
  spent again. Score it fresh only if weights or windows changed.
- **Two windows, not seven, during development.** 200 games is enough to see the probes move.
- **Sim count is linear and is the main dial.** 50 is the floor. The ≥200 the doc requires for
  confidence buckets takes the final line from 43 to 172 GPU-h. Do that once, at the end.

### 6.3 Realistic total

Nobody lands a twelve-head retrain with two architecture changes on the first attempt.

| item | GPU-h |
|---|---|
| three attempts at the model stack (§6.1) | 30 |
| development checks | 25 |
| one final paired evaluation at 700 games | 43 |
| **total** | **~98** |

At typical single-GPU cloud rates that is roughly 150–350 dollars. The compute is not the
constraint.

### 6.4 Build effort — estimate

| block | days |
|---|---|
| 2008 cutoff in loading / chronology; subset re-extract | 0.5 |
| vocabulary floor, retire coverage-completeness, rebuild | 0.5 |
| three prior stats, career stage, season deltas, sidecar rebuild | 1.5 |
| scalar-count bump, manifest, `ARCH_KEYS` | 0.5 |
| `rollout_score_fn` bridge (§4.6) | 1 |
| per-head metrics, frozen normalizers | 2 |
| substitution / timeout counters | 0.5 |
| decision logging during rollout | 2 |
| weighted replay step | 2–3 |
| A/B harness, run-state records | 1 |
| context modulation | 1 |
| cross-roster attention | 1.5 |
| **total** | **14–16** |

Multi-scale time (§5.3) is a further 4–6 days in 3.3.

---

## 7. Expected improvement

"Measured now" is from the 2.0 evaluation. Everything else is an estimate with a stated confidence.

| metric | measured now | expected | confidence | driven by |
|---|---|---|---|---|
| 4th-foul benching rate | 0.28 vs 0.96 real | large | 75% | §4 |
| 4th-foul event rate | 9.5× too high | large | 75% | §4 |
| Q4 blowout rotation | 1.60× too flat | moderate | 65% | §4 |
| player minutes MAE | not quoted in 3.0 | moderate | 60% | §4 |
| margin dispersion | 16.5 vs 13.7 | moderate | 55% | §4, regime latent |
| player points MAE, rookies / role-shifters | +10.9% / +24.0% vs season avg | moderate | 50% | §3.2, §3.3, §3.4 |
| team points MAE | 9.98 vs 9.62 baseline | small | 40% | §4 |
| corr(home, away) | 0.022 vs 0.351 | small | 30% | regime latent (3.0) |
| Brier | 0.2334 vs 0.2266 baseline | small | 30% | dispersion, indirectly |
| pick accuracy | 0.635 vs 0.646 baseline | flat | 20% | nothing in 3.2 |
| season-influence / per-game variety | not quantified | moderate | 55% | §5.2 |

### 7.1 Brier will move least, and that is structural

It is the metric the project most wants and the one this cycle helps least. The winner is mostly
decided by pre-game team strength, which lives in the priors, not in how faithfully the fourth
quarter composes. **The one real mechanism is indirect:** margin is over-dispersed at 16.5 against a
real 13.7, and over-dispersion flattens win probabilities toward 0.5. Tighten it and the same picks
score better. Worth perhaps 0.005–0.015.

### 7.2 Read it at 700 games, paired

Per-game Brier sd measured on run4 is **0.174**, not the 0.133 §8 assumed
(`v3_direction.md` §9.2). Paired run3-vs-run4 at n = 64 was ±0.0109 at one standard error.

| comparison | 2 SE detection threshold |
|---|---|
| unpaired, n = 100 | 0.036 |
| paired, n = 100 | 0.017 |
| paired, n = 700 | 0.007 |

An expected gain of 0.005–0.015 is borderline at 100 games and clears the floor at 700. Use the
rotating-window pool. `FINAL_HOLDOUT_GAMES = 700` already exists for this, and window 0 is
byte-identical to what v2-run1..4 scored.

### 7.3 Gates

- **§3 (data and inputs)** — no gate; these are input changes whose effect is read off the §7 table.
- **§4 rung 3** — must improve rollout CRPS, and the foul and rotation probes, on games the
  fine-tune never saw. It must not worsen margin dispersion; if it does, the guard in §4.4 failed.
- **§5.2** — context modulation must not worsen any probe. Cross-roster attention is scored on
  spread MAE.
- **§5.3 fires** only if the Q4 rotation probe moves under §4 and then plateaus.

---

## 8. Sequence

Everything in §3 is free before the retrain and a retrain afterwards, so the order that wastes
nothing is:

1. **Corpus cut, subset re-extract, vocabulary floor.** (§3.1, §3.2)
2. **Prior stats, career stage, season deltas; rebuild the sidecar.** (§3.3, §3.4)
3. **Context modulation and cross-roster attention.** (§5.2) — architecture, so it must be in the
   graph before the retrain or it waits a full cycle.
4. **Retrain all twelve heads.** One retrain.
5. **`rollout_score_fn` bridge; enable rung 2.** (§4.6) Records `epochs_disagree`.
6. **KPI phase.** (§4) Decision logging, per-head metrics, weighted replay.
7. **Development check at 200 games.** Read the probes, not Brier.
8. **Final paired evaluation at 700 games.**

Steps 1–4 are one unit; nothing in them can be A/B'd separately without another retrain, which is
the price of having one retrain. Steps 5–6 are A/B-able against the step-4 bundle, which is why §5.3
is held back — it keeps that comparison clean.

---

## 9. Risks and open questions

**The priors sidecar cut.** §3.1. Silent failure, wrong 2008–2009 priors, model trains fine.
Highest-consequence mistake in the document.

**Thin player embeddings.** §2.2. A decision, taken with the risk stated, mitigated three ways, and
landing on the axis that is already the larger failure. If §7's rookie / role-shifter row comes back
*worse* rather than better, this is the first suspect.

**Eight small heads are unaffected by §2.2** — they were already on the subset. Only the four big
heads change, and only `player` / `substitution` / `sub_decision` carry the player vocabulary.

**Four changes land in one retrain** (data, vocabulary, inputs, two layers). A gain cannot be
attributed among them. Accepted deliberately; §5.3 is held back to stop it becoming five.

**Open — the minimum-games floor.** 20–30 is a starting point, not a measurement. Worth a histogram
of subset games-per-player before fixing it.

**Open — `pf_36` interaction with §4.** The foul-type head gets a new prior *and* a new objective in
the same cycle. If the foul probes move, the two are confounded.

**Open — batched-rollout throughput under the new layers.** Every figure in §6.2 scales directly
with the 37 GPU-min per 1,000 sims rate, measured on the 3.0 graph at `ROLLOUT_BATCH_SIZE = 48`.
Heavier forward passes should improve batching efficiency, but this is unmeasured.

**Test-suite state carries over from 3.0** (`v3_direction.md` §9.6). `pytest` has never run against
this branch; `test_model_persistence` and `test_backbone` have never run at all, and that is exactly
where `num_scalars`, the `regime` input and the manifest meet real save/load. §3.3 and §3.4 take
`num_scalars` to 21 and §5.2 adds `ARCH_KEYS`. **Run the suite on a GPU box before the retrain**, or
the first thing 3.2 discovers will be a reload failure.
