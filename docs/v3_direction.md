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

   **Where the data comes from: the play-by-play we already have. No external source, no join.**
   - `generate_box_score(game_df, home_team, away_team)` in `simulation/box_score.py` already
     turns one real game's event rows into a per-player box — minutes, pts, fga/fgm, tpa/tpm,
     fta/ftm, ast, oreb, dreb, tov, pf. It is what builds `actual_boxscore.txt` for every
     holdout game (`simulation/evaluation.py:338`). **Use it; do not write a second tally.** A
     throwaway re-implementation during the 2.0 analysis matched only 81/96 final scores
     (free-throw and heave edge cases) — the vetted function is the one the report already
     scores against, so the prior and the target agree by construction.
   - `season_context.py` already walks each season's games in date order carrying per-team and
     per-player state (that is how `rest_home` / `games_played` are written). Player averages
     are the same walk with one more accumulator. The order inside the loop is the whole
     causality guarantee:
     ```
     for each game, in chronological order:
         write priors for every rostered player   # from running totals — BEFORE this game
         box = generate_box_score(game_df, ...)
         add box to each player's running totals   # AFTER
     ```
     Game N's prior therefore contains games 1…N−1 only, and training and inference read the
     same column, so they cannot disagree.
   - Per player, keep running sums of minutes, pts, fga, tpa, fta, ast, oreb, dreb, tov and
     games played; derive `min_pg` and the per-36 rates from them (`pts_36 = pts / minutes ×
     36`, etc.) so the rate is role-independent and `min_pg` carries the role. Last-10 is the
     same sums over a rolling window of that player's games.
   - Shrinkage seed: the player's **previous-season** rate from the previous `season<YYYY>.csv`
     (same display-name key, same files), else the league mean. This is what makes opening
     night and a rookie's first ten games sane instead of noise. `k ≈ 10` games to start;
     it is a training-side constant, not a `_TUNING_KEYS` dial.
   - Why no external join: the pipeline has no player IDs (see `v3_planned_changes.md` §2 on
     age — the join is the real work there). Priors key on the same display-name string, in
     the same file, that the model already trains on. `'Desmond Bane'` in the roster column is
     `'Desmond Bane'` in the shot rows.
   - Team priors (net rating, pace, off/def rating) come from the same walk at the team level,
     from `team_totals` / `possessions` in `simulation/stats.py` over the same per-game boxes.

   **Plumbing, following the `rest_home` precedent exactly:** roster-parallel list columns
   produced in `season_context.enrich_df` (add to `NEW_COLUMNS`); normalised and fed by
   `models/season_features.py` (`REST_LIST_COLS`); fused into the player embedding in
   `models/roster_set_encoder.py` through `scalar_proj` by raising
   `RosterEncoderParams.num_scalars` — **in both `get_config` and `from_config`, or weight reload
   silently breaks**; inference side, `simulation/game_input.py` needs the per-player maps
   mirroring `home_rest` / `away_rest`, and `simulation/input_cache.py` (the `rest` block around
   lines 142–156) picks them up. The bench bundle (`sub_decision`) gets the same columns.

   **Tests that ship with it:** (i) causality — for a sample of games, every prior equals what is
   recomputed from `date < game_date` only; (ii) a rookie on game 1 gets the league-mean prior,
   not zero and not a crash; (iii) a player who appears in the roster but never in an event row
   still gets minutes from the roster/`time` columns; (iv) the `num_scalars` round-trip
   (`get_config` → `from_config` → identical weights load).
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

---

## 8. Implementation notes — rules for anyone building 3.0

Collected from the 2.0 cycle. Most of these cost a retrain or a misread result when broken.

**Process**
- **Spec before code.** Each workstream gets a spec in the `v2_planned_changes.md` format
  (what / why / how / next steps, with file:line anchors) and its gate copied from §6 before
  anything is built. A commit at each major implementation step.
- **Training and evaluation run on the GPU box**, not the dev machine. The dev side writes
  the code, the tests, and the spec; the run happens on WSL/CUDA and the report comes back.
  Local pytest and smoke runs stall — do not use them as the check.
- **Before any train, check `encoder/vocabs/norm_stats.json` against git.** The test suite
  rewrites the committed encoder vocabs and norm stats; a train that starts from polluted
  stats is silently wrong.
- **Line endings are mixed CRLF/LF per file.** A Python rewrite of a file flips them and the
  diff becomes unreadable. Edit lines, do not round-trip whole files.

**Data and features**
- Every prior, rate, and game-state feature is computed **once, in one pass, and read by both
  training and the simulator** — `season_context.py` for per-game constants,
  `GameStateScan` (`models/game_state_features.py`) for per-row running state, which
  `simulation/input_cache.py` consumes incrementally. Never compute a feature two ways.
- **Box scores from play-by-play come from `generate_box_score`** (§3, W2.1). Any new tally
  is a bug waiting to disagree with the report.
- **Baselines frozen at the train cut** for every rotating window (§4). A baseline that sees
  games the weights did not is not a baseline.
- New per-player scalars go through `RosterEncoderParams.num_scalars` and **both**
  `get_config` and `from_config`. The kernel shape then fails loudly on a mismatch instead of
  loading quietly.

**Evaluation and reading results**
- **Never fit a dial to win / Brier / spread at n ≈ 100** (rule from `dials/README.md`; §1e
  is the quantification). Gate dials on the box score at 3σ, from `run_summary.parquet`.
- **Never compare confidence buckets across runs with different `--monte-carlo`.** State the
  sim count next to every bucket. ≥ 200 sims before any bucket is read.
- **Print the paired-Brier SE next to every Brier** (sd ≈ 0.133 per game → 0.013 at n = 100).
  Two runs within ±0.026 on the same games are the same model.
- **Report both win probabilities** — the sim-count vote and the Gaussian score prob — and
  never fit the spread calibration into the win path with a nonzero intercept
  (`simulation/eval_metrics.py:90`). `MARGIN_CALIBRATION_SLOPE` stays 1.0 until the W3 gate
  passes, then retires.
- **Repeat runs on the same window use a different `--seed`** so they are independent
  Monte-Carlo draws; average their score probs. That is the only legitimate way to keep a
  lucky run.
- **Window index `k` is recorded** in `report.json` and `run_summary.parquet` and shown on
  every headline. Drift with k is a finding, not noise.
- **Judge rotation on CRPS over `player_std`, not minutes MAE** — MAE structurally prefers a
  flat rotation and gets worse when the rotation model improves.
- A subset that looks like an edge at n = 100 is checked against the v1 runs on the same
  games before it is believed (§5). One bucket in nine at p ≈ 0.05 is the expected false
  positive.

**W1 specifics**
- The three game-state diagnostics run on the **existing** per-sim play-by-plays under
  `results/version2/v2-run4/games/*/playbyplay/sim_*.csv` — no re-simulation. Real-side
  rates come from the same season CSV.
- The context-sensitivity probe needs only the trained heads and real game prefixes: score
  `p(next | prefix)` against `p(next | prefix with its last N rows replaced by a flat prefix of
  the same players and game state)`. Report the mean absolute shift per head and where it
  is largest.
- Home/away correlation across sims and pace sd are added to `eval_metrics.py` and
  `run_summary.parquet` in the same change, so every later run carries them.

**W3 specifics**
- Running pace (possessions so far ÷ minutes elapsed) is a `GameStateScan` key with a fixed
  `_NORM` constant, added to `GAME_STATE_KEYS`; the incremental cache path picks it up
  without a second implementation.
- The regime latent is sampled **once per rollout** and held for the game; it is not a
  per-event input. Its gate is the sim's corr(home, away), not any box number.

**W4 specifics**
- Rung 1 (scheduled sampling): two passes per batch — a no-grad forward to sample the
  replacement events, then the training pass on the mixed sequence. The mixing probability
  anneals from 0 and is logged per epoch.
- Rung 2 (checkpoint selection): an eval hook every 3rd epoch, 20 games × 10 sims from the
  training-era tail (never from a holdout window), scoring box MAE, margin sd against the
  real residual sd, and the W1 game-state behaviours. The chosen epoch and its rollout score
  are written to the run state next to the NLL-best epoch, so the rung-3 condition (§6, step
  6) is a number, not a judgement.

---

## 9. 3.0+ — what the build established, and what is open

**Post-build addendum, added after §6 steps 1–4 shipped on `feature/version3`.** §1–§8 are the
direction as written and are unchanged. This section records what the build answered, what it
changed about the plan, and what is now open. State and numbers live in
[`v3_progress.md`](v3_progress.md); this is the decisions half.

### 9.1 The gates that already answered

**§6 step 1's gate — "game-state behaviours present / absent" — answers ABSENT.** Measured on
v2-run4's 5,357 sim play-by-plays against all 1,320 real 2022-23 games: a man on his 4th first-half
foul comes off within 60 s **28%** of the time in the sim against **96%** in reality, and because
the sim never sits anyone it *reaches* a 4th first-half foul **9.6× as often**. Starters' Q4 minutes
fall 16% in a sim blowout against 47% in reality. `court_fouls_*`, `score_diff` and the period clock
all reached the weights in 2.0 and produced no behaviour.

**This reprioritises W4.** No next-step loss can see a failure of *composition* — each individual
prediction is roughly right. Rung 2 is the only item in the programme that optimises composition
directly, so it moves from fallback to the most interesting rung, and the probe gaps are a term in
its score.

**§1d reproduced and extended.** The sim's *total* sd is too small (16.7 vs 19.7) at the same time
as its margin sd is too large (16.5 vs 13.7), with the per-side marginal right (11.7 vs 12.07). One
missing covariance moves both, in opposite directions. Worth stating because it is a stronger
argument than §1d's: it is not a mis-sized spread, so no dial can fix it.

### 9.2 Two corrections to §8

- **The Brier SE is 0.0178, not 0.013.** §8 estimates the per-game Brier sd at 0.133; measured on
  run4 it is **0.174**. The "two runs are the same model" band is **±0.036**, not ±0.026.
- **§1e is now demonstrated rather than argued.** Paired on the 64 games they share, run3's 0.2110
  against run4's 0.2035 is **+0.0075 ± 0.0109 (z = 0.69)**.

### 9.3 Departures from §3–§4, each forced by a measurement

1. **The priors are a sidecar Parquet, not roster-parallel CSV columns** (§3 W2.1 says to follow the
   `rest_home` precedent exactly). `rest_home` is one integer per player; these are ten floats.
   Measured: ~1,600 characters a row, taking `data/` from 4.2 GB to ~30 GB. The (game, player) grain
   is **559k rows / 56 MB** and rewrites no season file.
2. **Season-to-date only, no last-10** — twenty planes would roughly double the training set's host
   RAM, and 2.0 already lost a train to an OOM. Last-10 is what W2.2 replaces anyway.
3. **Fixed `_NORM` constants, not fitted `norm_stats`** — the only new fitted key in the programme
   is W3's `regime_std`.
4. **The rotating-window POOL, not a relaxed guard.** `FINAL_HOLDOUT_GAMES` becomes the pool
   (300 → 700 = 100 × 7); the existing 100 ids *are* its first 100, so `extend_holdout`'s prefix
   invariant is untouched and window 0 is byte-identical to what v2-run1..4 scored.
5. **The regime latent conditions every head, not the two §3 W3 names** — the seven conditional
   heads share one `cond_*.npz`, so a `regime` input on some and not others desynchronises the
   stored tensors from the graphs.
6. **`running_pace`'s constants came from a measurement.** `poss_ends` is a game-level counter, not
   per-side, so a real game runs ~4 poss/min, not the ~2.08 a 100 team-pace implies. The first
   constants clipped every real row and the feature would have trained as a constant.

### 9.4 Architecture decisions worth carrying forward

- **The player priors enter the set encoder additively, before attention**:
  `x = emb + scalar_proj(stacked)` (`models/roster_set_encoder.py:137`), where `stacked` is the
  fourteen per-player scalars. They therefore participate in the SAB layers — the encoder can
  condition how it attends to the other four slots on the fact that one of them is a 30-ppg scorer.
  Concatenating onto the pooled vector instead would encode the lineup first and bolt the rates on
  after, discarding that interaction. Player priors add **zero** fusion width; only the eight *team*
  priors widen the backbone concat.
- **One `train_step`, not two wrappers.** W3's latent and W4 rung 1 both alter the batch before the
  forward pass, and two `keras.Model` wrappers cannot compose — the outer one's `self.inner(x)`
  hands the inner one a batch it does not declare. `models/train_steps.build_trainer` owns both.
- **The manifest records which inputs a model trained with**, and `ModelBundle.load` refuses a
  mismatch by name. Most width changes already fail at `load_weights`; a same-width key swap did
  not, and that is the case this exists for.

### 9.5 Open, in the order the cost argues for

**(a) Four prior stats that are not in the ten, and are free to add only until the retrain starts.**
`pf`, `ftm`, `tpm`, `stl` and `blk` are in `BOX_STATS` and in every box score, but are not
accumulated, which rules out:

| candidate | why it belongs | head it serves |
|---|---|---|
| `ft_pct` (`ftm/fta`) | the model has an FT *attempt* rate and no *make* rate; FT% is the most stable player stat there is | free throws |
| `tp_pct` (`tpm/tpa`) | `shot_type` emits `corner3_l` / `wing3_r` / `top3` as distinct tokens, and `shot_result` judges them without knowing whether the player can shoot one | `shot_result` |
| `pf_36` | who actually fouls. Note it does **not** fix the 9.6× over-production of 4th fouls, which is a *benching* failure, not an attribution one | `foul_type` |
| `stl_36` / `blk_36` | no defensive-activity prior at all | `turnover_type`, `shot_result` |

Each is one accumulator line, one `_NORM` entry and a ~10-minute sidecar rebuild, taking
`NUM_ROSTER_SCALARS` 14 → 18. **Before the retrain that is free; after it, it is a retrain.**
Recommendation: `ft_pct`, `tp_pct`, `pf_36`; leave `stl` / `blk`, whose events are rare enough that
a prior is mostly noise. Also noted: `fg_pct` mixes shot difficulty with shooting skill, being
`fgm/fga` over all shots.

**(b) `MIXABLE_FIELDS` overstates what rung 1 does.** It lists `event, type, result, player,
secondary_player`; the implementation replaces only `event` (`models/train_steps.py:155`), and the
constant is imported into `train_steps.py` unused. The code is right — within `event_time`'s graph
the only sampleable output is `event_output`, and `type` / `result` / `player` come from other heads
that are not in that graph. Either narrow the constant and the docstring to `("event",)`, or widen
rung 1 to reach the other heads, which is a design change rather than a fix.

**(c) The exposure-bias gap is only partly closed, and the A/B must be read that way.** Rung 1
teaches the model to continue after a wrong *token*, against a true score, clock and lineup. It
never trains on a context the model composed over hundreds of steps. Rung 2 *generates* whole games
mid-train but only scores them — no gradient flows from a rollout. **Training on generated games is
rung 3, and it is not built.** A null result at rung 1 is "token-only scheduled sampling does not
help", not "scheduled sampling does not help".

**(d) Conditional gates, unchanged from §6.** Rung 3 fires only if
`checkpoint_selection.epochs_disagree` is true in the run state. W2.2 and W6 gate on the W2.1
result.

**(e) Two §3/§4 items deliberately not on the step 1–4 path.** The context-sensitivity probe
(§3 W1's first bullet) needs the trained heads and belongs on the GPU box against 3.0 weights.
Conditional evaluation from end-Q1 / half (§4's secondary scoreboard) needs a
`GameController.resume_from(history)`; `start()` always builds a tip-off game and resets ~15 pieces
of rules-side state.

### 9.6 Test-suite state

`pytest` has never run against this branch — the dev box has no GPU and `conftest.py` imports TF
first. Every test function was executed directly instead (667 pass), which caught four real bugs,
but it cannot reproduce pytest's fixtures or its `sys.path`. Two consequences:

- **`test_model_persistence` and `test_backbone` did not run at all.** That is where
  `num_scalars=14`, the `regime` input and the `SCHEMA 2` manifest meet real save/load. It is the
  highest-risk untested surface in the change.
- **Six `test_controller` failures on the offence-side foul path are pre-existing** — verified
  against `main` in a throwaway worktree. They are not 3.0's.
