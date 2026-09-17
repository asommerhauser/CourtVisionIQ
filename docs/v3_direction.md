# CourtVisionIQ 3.0 — Direction

**Status: proposal, 2026-09-16, revised after review.** Written from the 2.0 evaluation
(`results/version2/v2-run1..4`) and the analysis that followed it. Every number here is
reproducible from `data/games.parquet`, `data/box_players.parquet`, the per-sim play-by-plays
under `games/*/playbyplay/`, and `data/season2023.csv`; nothing depends on a scratch script. This
document supersedes the *framing* of `v3_planned_changes.md` (which stays as the parked-feature
list) and proposes the next programme. It is a direction, not a build spec — each workstream gets
its own spec on approval.

**Decisions taken at review (2026-09-16):** pre-game win / spread / box lines **stay the
headline**. The holdout stays at ~100 games per run, but runs rotate through different 100-game
windows of the untrained tail of 2022-23 (§4). **W4 goes to rungs 1–2**; rung 3 is conditional.
**W6 is deferred until the W2 gate passes** — the simulator fixes its own centre first. Data
restriction is rejected.

---

## 1. What the 2.0 evaluation established

Six measurements, on the same 96–100 holdout games (2023-01-10 to 2023-01-24), that together
say where the model is and is not.

**(a) The model is statistically indistinguishable from a 20-line baseline on pre-game outcomes.**
"Each team's season-to-date point differential plus home court," fit on the 610 games before the
holdout: acc 0.646 / Brier 0.2266 / spread MAE 9.62. Model (v2-run4): 0.635 / 0.2334 / 9.98.
Paired p = 0.63–0.74. The market sits well above the baseline (~68–70% favourites, ~9.3 MAE).

**(b) On player box lines the model loses to "use his season average" — and the loss is
concentrated exactly where a rare-token model would lose.** Points MAE, all four v2 runs pooled:

| player type | n | model vs season-avg |
|---|---|---|
| rookie / 2nd year (≤80 career games in training) | 1,723 | +10.9% worse |
| 81–400 career games | 3,116 | +8.5% worse |
| veteran (400+) | 2,009 | +2.4% (parity) |
| veteran **and** stable role (<3 ppg from career) | 1,055 | parity, p = 0.32 |
| role shifted >6 ppg from career (the MIP profile) | 817 | **+24.0% worse** |

Desmond Bane (career 14.0 ppg, Jan-2023 21.4): model bias −8.7 ppg. Ja Morant (21.8 → 27.1):
−4.8. The model predicts who they *were*. Regressing model prediction on career-avg and
season-avg gives `0.24·career + 0.61·season` — the season token works, partially, and the 0.24
is the drag.

**(c) The spread is right; the centre is wrong.** Per-player: 65.7% of actuals inside ±1 model sd
(ideal 68%), 94.5% inside ±2 (ideal 95%). Team totals: sim sd 11.73 vs real 11.79. The variance
the simulator produces is correctly calibrated. Nothing in this direction touches how events are
sampled.

**(d) The two teams in a sim do not share a game.** Across the 50 sims of one game,
corr(home pts, away pts) = 0.022; in real 2022-23 games it is 0.351. Sim pace sd 2.96
possessions; real 5.35. Margin sd is therefore 16.5 (= √2 × 11.7) against a real 13.7. Every
"the model is over-confident" symptom traces to this: the margin over-dispersion is entirely a
missing shared component, not a mis-sized one, so no shrinkage dial is the right fix.

**(e) Confidence buckets are unreadable at 100 games, and part of the over-confidence is a
Monte-Carlo artefact.** Selecting on `p̂ ≥ 0.70` from a 50-sim estimate biases the bucket's
realised hit-rate down even for a perfectly calibrated model: +4.6 pp at 50 sims, +9.0 pp at 20,
+0.4 pp at 500. Distinguishing a true 0.78 bucket from a true 0.66 one needs ~660 games. The
four-run v1 ensemble (average of probabilities) scores Brier 0.2133, better than three of its four
members — sim count and averaging are the only way to "keep" a lucky run. The 0.204–0.209 Briers
on record (v2-run3 at n=64, v1 full3) are inside the ±0.026 paired-SE band of the 0.233 runs on
the same games.

**(f) Training curves show the heads reach the base rate in ~5 epochs and then memorise.**
`shot_result` val loss: 0.0416 → best 0.0398 at epoch 16 → rising. `shot_type`: +0.002 gained
after epoch 5. This is not a data-volume or capacity limit; the inputs do not contain the answer.
The embedding table (2,153 × 192) is where the memorisation lives: a rookie's slice trains on
~20k rows. Restricting the corpus (2010+) removes data the loss weight already floors at 0.05 and
makes every rare token rarer; it does not touch the static-identity problem.

---

## 2. Framing

The thesis is *games predict themselves*. §5.4 of the whitepaper already states the consequence:
the season-average baseline "is beaten only by getting the game-specific deviations right."

A player's game line decomposes into (1) who he is, (2) pre-game-knowable adjustments, (3)
in-game emergent effects — foul trouble, blowouts, flow — and (4) noise. The thesis lives in (3).
Before tip-off the simulator can only sample (3) fifty times and average, and the average of
what-might-happen is close to the base rate — so on the **pre-game mean**, the simulator's ceiling
is roughly a good ratings model's, and its edge over one must come from getting (1) right *and*
from the small, real, mean-level effects of game-state dynamics (a likely blowout lowers a
starter's expected minutes; a foul-prone matchup lowers a big's).

**Pre-game stays the headline.** The path to it is therefore: fix (1) so the model is at parity
with season average for *every* player, not only veterans (W2); make the joint structure right so
the margin distribution — and every confidence number read from it — is honest (W3); stop the
training objective and the scored objective from being different things (W4); and average more
(§4). Conditional and joint evaluation (§4) are added as *secondary* scoreboards because they are
where the thesis is directly testable and where the whitepaper's own §6 probes — still *Not
built* — live. They are not the headline.

The literature answer on the hot hand is "real, ~1–3 pp." The large, learnable, rule-driven
effects are game-state dynamics: 4 fouls → sits, +20 in Q4 → starters out, bonus → FT rate,
late-and-close → fouling and timeouts. 3.0 treats "momentum" as game-state dynamics first and
psychology second.

---

## 3. Structural changes

### W1. Measurement first — the thesis probes and the game-state diagnostics

*Inference only. No retrain. Days.*

- **Context sensitivity.** For real game prefixes, compare `p(next | history)` against
  `p(next | history with the last N events replaced by a flat, same-players, same-state prefix)`.
  If the conditional barely moves, the thesis as written is not supported at this scale and the
  programme is W2 + W4 + averaging. If it moves, this is the quantity every later change must
  preserve.
- **Event-head reliability curves / ECE**, per head.
- **Three game-state behaviours, sim vs real:** P(sub-out within 60 s | 4th PF before half);
  starter seconds in Q4 given |margin| ≥ 20 at Q4 start; trailing-team foul rate, last 2:00,
  down 4–9. These say whether 2.0's first trained game-state features produced behaviour.
- **Home/away correlation and pace sd** become standing report metrics (`eval_metrics.py`),
  alongside the season-average and season-net baselines, which are computed inside the report
  rather than in a notebook.

### W2. Identity from context

*Retrain.* At tip-off the model's context window is empty; identity is a single career-averaged
vector per name. Three implementations, stacked, in the order they should be built:

1. **Per-player rates as a cold-start prior**, computed causally in `season_context.py`'s
   chronological pass: `min_pg`, `pts_36`, `fga_36`, `tpa_rate`, `fta_rate`, `ast_36`, `oreb_36`,
   `dreb_36`, `tov_36`, `fg_pct`; season-to-date and last-10; shrunk toward the league mean by
   `n/(n+k)` with previous-season rates as the seed. Attached via the `rest_home` precedent
   (`REST_LIST_COLS`, `roster_set_encoder.scalar_proj`, `RosterEncoderParams.num_scalars` — both
   `get_config` and `from_config`). Team-level: net rating, pace, off/def rating, same pass.
   This is the two-week experiment that says whether the ceiling is there. Inputs do not reduce
   variance: sampling does the variance; inputs move the centre.
2. **A learned recent-games summary.** The player's last K games of play-by-play through the
   same backbone, pooled, attached to his roster slot. Identity learned from how he has been used
   *lately*. (1) becomes the fill-in when K is small.
3. **`(player, season)` embedding**, warm-started from the player embedding, as the cheap
   middle if (2) is deferred.

A strict causality test ships with (1): every prior equals what is computable from
`date < game_date`. Leakage here would make training look better and inference worse.

**Gate:** rookies and >6-ppg role-shifters reach parity with season average on pts/ast/reb.
Bane's bias is the named test case.

### W3. A game-level regime latent

*Retrain.* Sample a small tempo/flow vector once per rollout; condition the time head and the
shot-type head on it; train it to explain within-game persistence (the two teams' pace share a
game). Companion feature, testable first without a retrain: **running pace** — possessions so far
÷ minutes elapsed, from the same `GameStateScan` that builds the possession clock — so the model
can learn autocorrelation from its own game.

**Gate:** sim corr(home, away) ≈ 0.35, pace sd ≈ 5.3, margin sd ≈ 13.7 **with no shrinkage
dial**. `MARGIN_CALIBRATION_*` retires when this passes.

### W4. Train the simulator as a simulator

*The objective changes.* Every head trains and early-stops on next-step NLL with the **real**
history (teacher forcing) and is scored on what 600 **self-fed** steps produce. Three rungs,
cheapest first, each its own A/B. Costs are from the run4 logs: ~27 game-sims/hour/shard on the
16-shard pool, ~37 GPU-minutes per 1,000 batched game-sims; training ~105 s/epoch/head.

1. **Scheduled sampling.** With probability p (annealed from 0), the previous event in the
   training sequence is the model's own sample rather than the real one, so it learns to keep
   going after its own mistakes. *Cost:* one extra forward pass per training step — roughly
   1.5–2× per epoch, i.e. **+1.5–3 GPU-hours on a full train**.
2. **Checkpoint selection on rollout metrics.** Every k-th epoch, roll out 20 games × 10 sims
   and keep the checkpoint by box MAE / margin calibration / W1 probes rather than by NLL.
   *Cost:* 200 game-sims ≈ **7.5 GPU-min per evaluation**; every 3rd epoch over a 30-epoch
   joint stage ≈ **1.3 GPU-hours per train**; every epoch ≈ 4.
3. **Rollout-level fine-tuning.** Simulate whole games from the trained weights, score the
   produced distribution against the real game (CRPS per player, Brier on margin), REINFORCE
   through the samples. *Cost:* each gradient step needs a batch of complete rollouts and
   REINFORCE is high-variance — e.g. 64 games × 8 sims = 512 game-sims ≈ 19 GPU-min **per
   step**; 500 steps ≈ **160 GPU-hours per attempt**, before any hyper-parameter search, with
   the real possibility of no gain. Done last, only after W2–W3, and only if rung 2 shows the
   rollout metrics and NLL disagree about which checkpoint is best (if they agree, rung 3 has
   nothing to buy).

**Gate:** rung 2 must not worsen any W1 probe; rung 3 must improve rollout CRPS on a window the
fine-tune never saw.

### W5. Hierarchical time — *parked, evaluate after W2–W3*

Possession- and stint-level summaries as inputs so the rotation heads see the scale coaches
decide at. Deferred: W2 may absorb most of the rotation error on its own, and this is the most
invasive change to the backbone.

### W6. Copula product layer — *deferred until the W2 gate passes*

*No retrain. Decided at review: not built in parallel; it starts once the simulator's own
centre is right for rookies and role-shifters, so the layer is a product convenience rather than
a patch over a modelling gap.* In plain terms:

- A **pre-game mean model** is any method that produces, before tip-off, one expected number for
  each thing the report scores — Bane's points, MEM's total, the margin. The simulator produces
  that number today as the average of its 50 sims, and for Bane it says ~12.7 when the answer is
  ~21. The simplest mean model is the season average (21.4); a better one is per-36 rates ×
  projected minutes; better still adds matchup and rest. These are the *same numbers* as the W2.1
  priors — used *outside* the model to shift its output rather than *inside* it as an input.
- The **copula** step keeps everything the simulator gets right — the spread (§1c) and the
  correlations (Bane and Morant rising together in a fast game; starters' minutes falling in a
  blowout; a player's line moving with his team's total) — and moves only the centre. Per player,
  per sim: `pts_adj = pts_sim − mean_over_sims(pts_sim) + pts_mean_model`. Same for team totals
  and margin. The joint shape is the simulator's; the level is the mean model's.
- Why it is not cheating: it is how a desk would combine an expensive dependence model with the
  thing that is best at level, and it **isolates** the simulator's contribution — if the
  re-centred sim beats independent-marginals on joint outcomes (both-players-over, player +
  team total), the thesis is demonstrated on the thing it can demonstrate. It also converges: as
  W2 lands and the sim's own mean becomes right, the shift goes to zero.
- Why it is deferred rather than parallel: it puts a number the model did not produce into the
  model's output. Built before W2 it would hide the identity gap the programme is meant to close;
  built after, it is a legitimate product layer whose shift is small by construction.

---

## 4. Evaluation changes

**Holdout: ~100 games per run, rotating windows.** The train cut is at ordered game 26,267 of
26,969; 702 games follow it, all untrained; the current holdout is the first 100. `full_run.py`'s
`[extend]` path already computes `ordered[boundary : boundary + N]`; a window index k selects
`ordered[boundary + 100k : boundary + 100(k+1)]` for k = 0…6. Each run costs what it costs
today; six runs over a season cover 600 *distinct* games; per-run numbers stay as readable as now
and the pooled numbers become readable for the first time (no more four-runs-on-the-same-100
pseudo-replication).

Two rules make the rotation fair:
- **Information window matched.** The weights are frozen at the cut; the baselines must be too
  (season-average and season-net computed from pre-cut games only), or later windows flatter the
  baseline. A second, "deployed" comparison may give both up-to-date information, but not
  one side only.
- **Window k is reported with k.** Later windows are further from the cut (March games on
  January knowledge). Drift with k is itself a finding.

**Sims:** 50 stays the floor for a routine run; **≥ 200** for any run whose confidence buckets
will be read (§1e). `--seed` varies between repeat runs on the same window so repeats are
independent Monte-Carlo draws and can be averaged.

**Baselines in the report:** season-average per player, season-net + HCA per game, and (once W6
is built, after W2) the copula-recentred sim, all computed by `eval_metrics.py` from the same
records.

**Standing metrics:** corr(home, away) across sims; pace sd; per-player z-coverage (±1 sd,
±2 sd); the three W1 game-state behaviours; the paired-Brier SE, printed next to every Brier.

**Scoring rules:** CRPS over `player_std` for rotation and props, not MAE (per the rule in
`dials/README.md`); margin sd against the real residual sd, not a shrunk one.

**Secondary scoreboard — conditional evaluation:** from real game states at end-Q1 and half, roll
out the remainder and score against the actual second half. Not the headline; the venue where the
thesis is directly testable.

---

## 5. Not the goal any more

- **Data restriction** (2010+ or similar) — §1f.
- **Confidence-threshold selection** on the current model: the high-confidence bucket is its
  worst-calibrated (21/32 at a claimed 0.78 on run4), so filtering on confidence concentrates the
  error. Revisit after W3 and a ≥200-sim run.
- **Subset hunting** (roster familiarity, veteran-heavy games, etc.) at n = 100: tested this
  cycle; the one bucket that looked like an edge did not survive the v1 cross-check on the same
  games.
- **Fitting any dial to win / Brier / spread at n = 100** (already a rule in `dials/README.md`;
  restated here because §1e quantifies why).

---

## 6. Sequence and gates

| step | workstream | retrain | gate |
|---|---|---|---|
| 1 | W1 probes + diagnostics; §4 report changes; rotating-window support | no | thesis signal present / absent; game-state behaviours present / absent |
| 2 | W2.1 priors + team priors | yes | rookies + MIP-types at parity with season average |
| 3 | W3 running pace, then latent | same retrain | corr(home, away) ≈ 0.35, margin sd ≈ 13.7, no dial |
| 4 | W4 rungs 1–2 | same retrain | no W1 probe regresses; rollout-selected checkpoint ≥ NLL-selected on the report |
| 5 | W2.2 recent-games summary | yes | cold-start players match W2.1 within K games |
| 6 | W4 rung 3 — *conditional*: only if step 4 shows NLL and rollout metrics pick different checkpoints | yes | rollout CRPS improves on an unseen window |
| 7 | W6 copula layer — *after the W2 gate* | no | re-centred sim beats independent marginals on joint outcomes |

Steps 2–4 share one retrain. Step 1 decides whether steps 5–6 are worth their cost; step 2's
gate unlocks step 7.

---

## 7. Decisions (2026-09-16)

| question | decision |
|---|---|
| Headline metric | **Pre-game** win / spread / box lines. Conditional and joint evaluation are secondary scoreboards. |
| Holdout | **~100 games per run, rotating windows** k = 0…6 over the untrained tail; ≥ 200 sims when confidence buckets are read; baselines frozen at the train cut. |
| Data restriction (2010+) | **Rejected** — §1f. |
| W4 depth | **Rungs 1–2** (+~3–7 GPU-h per train). Rung 3 (~160 GPU-h) only if rung 2 shows NLL and rollout metrics disagree on the best checkpoint. |
| W6 copula layer | **Deferred until the W2 gate passes.** Not built in parallel. |

**Next action:** specs for W1 (probes + report metrics + rotating-window support) and W2.1
(priors), in the format of `v2_planned_changes.md`, followed by one retrain carrying W2.1, W3's
running-pace feature, and W4 rungs 1–2.
