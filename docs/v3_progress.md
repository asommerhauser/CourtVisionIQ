# CourtVisionIQ 3.0 — Progress Tracker

> **State, not spec.** [`v3_direction.md`](v3_direction.md) is the direction and is not edited
> during the build. This file records what has been built, what was measured, and what was learned —
> including every departure from the direction and the number that forced it. Update it in the same
> commit as the work it describes.

## Standing rules

Carried forward from [`v2_progress.md`](v2_progress.md), with one correction.

1. **Training, rollouts, evals and `pytest` run on the GPU box, not the dev machine.** The dev side
   writes the code, the tests and the spec; the run happens on WSL/CUDA and the report comes back.
   Claude stops at each verification point, hands over the exact command, and waits for a pasted
   result.
2. **Correction to rule 1's stated reason.** `v2_progress.md` records that "TF does not load on the
   Windows side at all". Re-measured this cycle: TensorFlow 2.20 imports fine on Windows **when it
   is imported before pandas**, and fails in `_pywrap_tensorflow_internal` when pandas wins the
   race. The package import order never fixed that for a caller who had already imported pandas; it
   only hid which modules genuinely needed TF. `simulation/__init__` and `reporting/__init__` are
   lazy as of 3.0, so the whole evaluation-report stack and the box-score tally are TF-free by
   construction and the ordering is irrelevant to them. **This does not unlock training or
   rollouts** — there is no GPU visible from Windows — and rule 1 stands unchanged.
3. **Commit at each meaningful step**, with the message saying what changed *and why*, including
   what was measured and what was rejected.
4. **No large evals mid-programme.** The full 3.0 eval happens after the train.
5. **Before any train, check `encoder/vocabs/norm_stats.json` against git.** The test suite rewrites
   the committed encoder vocabs and norm stats.
6. **Line endings are mixed CRLF/LF per file.** Edit lines; never round-trip a whole file.

---

## What is built

All of `v3_direction.md` §6 steps 1–4, on `feature/version3`, as ten branches merged `--no-ff`.
Steps 5–7 (W2.2, W4 rung 3, W6 copula) stay parked — the direction gates them on results this pass
produces.

| branch | workstream | retrain? | state |
|---|---|---|---|
| `v3/tf-free-imports` | lazy `simulation` / `reporting` packages | no | built, verified |
| `v3/w1-standing-metrics` | joint structure, coverage, Brier SE, backfill | no | built, **run** |
| `v3/w1-probes` | three game-state behaviours | no | built, **run** |
| `v3/w1-rotating-windows` | window `k` end to end | no | built, verified |
| `v3/w2-priors` | W2.1 season-to-date priors | yes | built, sidecar **built** |
| `v3/w3-running-pace` | W3a running pace | yes | built, constants measured |
| `v3/w3-regime-latent` | W3b per-game latent | yes | built, verified on CPU |
| `v3/w4-scheduled-sampling` | W4 rung 1 | yes | built, verified on CPU |
| `v3/w4-rollout-checkpoint` | W4 rung 2 | yes | built, verified with stubs |
| `v3/arch-manifest` | input signature + load refusal | no | built, verified |

161 assertions across ten test files, executed directly (not through pytest — `conftest.py` imports
TF first). `pytest` itself is a WSL step.

---

## W1 — the measurements

Everything in this section was computed on the dev machine from artifacts already on disk: the four
v2 runs under `results/version2/`, their 5,357 sim play-by-plays, and `data/season2023.csv`. No
re-simulation.

### 1. Joint structure — §1d reproduced, and extended

Per-sim vectors were backfilled onto all 358 finished v2 games from their play-by-plays, through
`generate_box_score`; every rebuilt game's scores matched `run.json`'s `per_sim_scores` exactly.

| run | n | corr(home, away) | pace sd | margin sd | total sd | per-side pts sd |
|---|---|---|---|---|---|---|
| v2-run1 | 98 | +0.070 ± 0.024 | 2.54 | 15.46 | 16.54 | 11.31 |
| v2-run2 | 100 | −0.075 ± 0.024 | 2.57 | 17.60 | 16.23 | 11.97 |
| v2-run3 | 64 | −0.058 ± 0.018 | 2.54 | 17.58 | 16.59 | 12.06 |
| v2-run4 | 96 | +0.017 ± 0.015 | 2.47 | 16.46 | 16.75 | 11.73 |
| **real 2022-23** | **1,320** | **+0.352** | **4.80** | **13.66** | **19.73** | **12.07** |

**The extension the direction document did not state.** The sim's **total** sd is too *small*
(16.7 vs 19.7) at the same time as its margin sd is too *large* (16.5 vs 13.7), while the per-side
marginal is right (11.7 vs 12.07). That is the exact signature of a missing positive covariance:
Var(H+A) and Var(H−A) both collapse to VarH + VarA when the sides are independent, so one absent
term moves both, in opposite directions. It is not a mis-sized spread, and a shrinkage dial cannot
add a term that is not there. The report now says this on its own, in a standing section.

### 2. Coverage and the Brier standard error — two corrections to §8

| run | player pts ±1sd / ±2sd | margin ±1sd / ±2sd | margin pred sd / residual sd | dispersion |
|---|---|---|---|---|
| v2-run1 | 0.612 / 0.924 | 0.765 / 0.980 | 15.46 / 12.36 | 1.25× |
| v2-run2 | 0.641 / 0.936 | 0.810 / 0.980 | 17.60 / 12.53 | 1.41× |
| v2-run3 | 0.673 / 0.952 | 0.828 / 1.000 | 17.58 / 11.62 | 1.51× |
| v2-run4 | **0.655 / 0.945** | 0.833 / 1.000 | 16.46 / 12.08 | 1.36× |
| ideal | 0.683 / 0.954 | 0.683 / 0.954 | — | 1.00× |

run4's player coverage reproduces §1c's 65.7 / 94.5 exactly, which is what says the metric measures
the right thing.

**Correction A — the Brier SE is larger than §8 assumes.** §8 estimates the per-game Brier sd at
0.133, giving SE 0.013 and a "two runs within ±0.026 are the same model" band. Measured on run4 the
sd is **0.174**, so **SE = 0.0178** and the band is **±0.036**. Every spec should carry the measured
number. The SE is now printed next to every Brier and stored in `run_summary.parquet`.

**Correction B — §1e's claim is now demonstrated, not argued.** Paired on the 64 games run3 and
run4 share, run3's celebrated 0.2110 against run4's 0.2035 is a difference of **+0.0075 ± 0.0109
(z = 0.69)**. They are the same model.

### 3. The three game-state behaviours — §6 step 1's gate answers **absent**

`v2-run4`'s 5,357 sim play-by-plays against all 1,320 real 2022-23 games.

| probe | metric | sim | real |
|---|---|---|---|
| 3rd foul before half | P(off the floor within 60 s) | **0.238** | **0.776** |
| | events per game | 1.413 | 1.189 |
| 4th foul before half | P(off the floor within 60 s) | **0.280** | **0.961** |
| | events per game | **0.371** | **0.039** |
| Q4 blowout rotation | starter seconds, blowout | 3,321 | 2,213 |
| | starter seconds, close | 3,963 | 4,213 |
| | ratio (blowout / close) | **0.838** | **0.525** |
| | blowout frequency | 0.220 | 0.123 |
| Trailing, last 2:00, down 4–9 | fouls per 100 s in state | 1.620 | 1.992 |
| | state seconds per game | 30.1 | 36.4 |

**All three behaviours fire in reality and effectively do not fire in the simulator.** Reality pulls
a starter 47% in a blowout; the sim pulls him 16%. Reality benches a man on his fourth first-half
foul 96% of the time; the sim, 28% — and because it never sits anyone, it *reaches* a fourth
first-half foul **9.6× as often as reality**.

The model has the inputs. `court_fouls_*`, `score_diff` and the period clock all reached the weights
in 2.0. They produced no behaviour.

**This reprioritises W4.** No next-step loss can see any of it: each individual prediction is
roughly right, and it is the composition over hundreds of steps that is wrong. Rung 2 — select the
checkpoint on what it *simulates* — is the only thing in the programme that optimises the
composition directly, and the probe gaps are a term in its score. It is now the more interesting
rung, not the fallback.

**A sizing note that shapes how it is read.** "4th foul before half" yields ~4 real events on a
100-game window and 51 on a full season. It is only readable against the full-season real rate,
which is why the probe reports the season and why the **3rd-foul variant ships as the companion** —
1,570 real events, the one that can actually move.

---

## W2.1 — the priors, built and spot-checked

`data/priors/` now holds **559,032 player-game rows** across 21 seasons, **56 MB**. Built in ~10
minutes on the dev machine; `data/*.csv` was not touched.

Spot-checks against the cases `v3_direction.md` §1b names, on the holdout window:

| player | prior games | min_pg | pts_36 | implied ppg | actual Jan-2023 | career |
|---|---|---|---|---|---|---|
| Desmond Bane | 23 | 30.7 | 24.2 | **20.7** | 21.4 | 14.0 |
| Ja Morant | 36 | 32.4 | 30.5 | **27.5** | 27.1 | 21.8 |
| Nikola Jokić | 39 | 33.4 | 27.2 | 25.2 | ~24.5 | — |

Bane carried a −8.7 ppg bias in 2.0 because the model predicted who he *was*. His prior reads 20.7
against an actual 21.4. Team priors average net rating 0.16 and pace 100.7, against a measured real
2022-23 pace of 100.7.

Zero-history players get the normalized league mean, never zeros — a zero vector is not "no
information", it is the confident claim that a player does nothing, which is precisely what a naive
implementation emits for every rookie.

---

## Departures from the direction, and the number behind each

### 1. The priors are a sidecar Parquet, not roster-parallel CSV columns

§3 W2.1 says to follow the `rest_home` precedent "exactly". `rest_home` holds one integer per
player; the priors hold ten floats. Measured on `data/season2023.csv` — 216 MB, ~743k rows, 291
characters a row, of which `rest_home` is 15.1 — ten rates across five slots and two sides is
**~1,600 characters a row**, taking `data/` from 4.2 GB to roughly **30 GB** (≈13 GB for the
season-to-date half alone) and rewriting every season file in place.

The natural grain is (game, player): 26,969 games × ~21.8 player-lines. **Actual: 559k rows, 56 MB**
— about 1/500th — and the cleaned CSVs are never rewritten, which also removes an in-place overwrite
of 5 GB of untracked, unbacked-up data. `generate_box_score` costs 4 ms/game, so the causal walk is
cheap; the cost is reading the CSVs.

### 2. Season-to-date only. No last-10 window

§3 W2.1 asks for both. Each roster-parallel `(SEQ=600, ROSTER=5)` plane costs 12 KB/game raw, and
`_load_processed` holds the whole npz in host RAM while `from_tensor_slices` materialises it again.
Ten priors × two sides is +240 KB/game against a current ~180 KB; twenty would roughly double the
training set's footprint, and the 2.0 cycle already lost a train to an out-of-memory. Last-10 is
also exactly what W2.2's learned recent-games summary replaces, so it is the right half to cut.

### 3. Fixed normalization constants, not fitted `norm_stats`

§3 W2.1 points at the `rest_mean`/`rest_std` path; this follows `rotation_features._NORM` instead.
Fitted statistics mean two persist sites, a `refit=False` branch, three inference-side read-backs
and ten more keys in the file the test suite is known to overwrite. The rates are already in
natural, era-stable units. The only new `norm_stats` key in the whole programme is W3's
`regime_std`, written once after a fit.

### 4. The rotating-window pool, rather than a redesigned guard

The obvious reading of §4 is to relax `extend_holdout`'s prefix invariant so window k > 0 can be
selected. Instead `FINAL_HOLDOUT_GAMES` is redefined as the **pool** (300 → 700 = 100 × 7, against
702 games available after the cut) and a window is a slice of it. The existing 100 ids *are* the
first 100 of the 700, so widening the pool is a legal extension, the guard is untouched, and window
0 is byte-identical to the games v2-run1..4 were scored on.

### 5. The regime latent conditions every head, not two

§3 W3 names the time head and the shot-type head. All seven conditional heads share one `cond_*.npz`
and one preprocess, so a `regime` input on some and not others would make the stored tensors and the
graphs disagree. The input is uniform; the extra cost is one `Embedding(n_games, 4)` per head.

### 6. `running_pace`'s constants came from a measurement, after a wrong first draft

`poss_ends` is a single **game-level** counter, not per-side, so a real game runs at ~4 possessions a
minute rather than the ~2.08 a 100 team-pace would suggest. The first constants (clip 3.0, divide by
2.0) clipped essentially every real row and the feature would have trained as a constant 1.5, with
no error anywhere. Measured over 485 real games and 240k rows: median 4.03/min, p1 3.37, p99 4.70,
end-of-game 3.98 ± 0.19 (a team pace of 95.5 per 48). Constants are `(0, 8.0, 4.0)`; re-measured,
the normalized feature has mean 0.998, sd 0.113, nothing clipped.

---

## Bugs found by verifying rather than reading

Each of these would have survived to a finished GPU run and been invisible in it.

1. **The latent table received no gradient.** The embedding lookup sat outside the `GradientTape`,
   so it trained as all zeros. The head learns perfectly well either way; the only symptom is
   `regime_std` coming out empty, hours later.
2. **A custom `train_step` that does not feed the loss tracker reports 0.0 every epoch.** Gradients
   flow normally. But `EarlyStopping(monitor="val_loss", restore_best_weights=True)` is this repo's
   only checkpoint selector, and against a constant zero it never improves and restores **epoch 1's
   weights** at the end of a full train.
3. **No loss scaling under `mixed_float16`**, which `configure_gpu` sets for every GPU train —
   fp16 gradients underflow to zero.
4. **`from config import X` at module scope froze three knobs at import time**, so switching them off
   in a test or on the command line did nothing, and the "feature disabled" path was silently
   untested.
5. **The foul-trouble probe keyed pending observations by player**, so a man who took his third and
   fourth fouls before half had only the third recorded — dropping the rarer and more interesting
   half of the sample.
6. **`coverage_metrics` over total rebounds reads 0.0%.** `reb` is derived, and `sd(oreb + dreb)`
   needs a covariance the record does not store. Deriving it the way minutes is derived would give a
   right mean and a silently wrong spread. Requesting an unstored stat now raises.
7. **A stale positional index in `test_game_state_features`.** It read `scan.step(row)[-1]` to mean
   `poss_clock`; appending `running_pace` turned it into a comparison between two different
   quantities. It indexes by name now.

---

## Handover — the WSL/CUDA sequence

Run in order. Each step's gate is what decides whether the next one is worth doing.

```bash
# 0. Pre-flight, per standing rule 5.
cd /mnt/c/Projects/CourtVisionIQ
git checkout feature/version3 && git pull
source ~/cviq-venv/bin/activate
git status --porcelain encoder/vocabs/          # must be empty
```

```bash
# 1. The suite. This is the authority -- nothing on the dev machine ran pytest.
python -m pytest -q
```

**Known state going in.** Six tests in `test_controller.py`, all on the offence-side foul path,
fail on `main` as well — verified by running them against `main` in a throwaway worktree. They are
not from 3.0. Everything else should pass; the 3.0 files are `test_standing_metrics`,
`test_state_probes`, `test_rotating_windows`, `test_player_priors`, `test_regime`,
`test_scheduled_sampling`, `test_rollout_selection`, `test_feature_manifest` and
`test_tf_free_imports`.

```bash
# 2. pytest rewrites the committed encoder vocabs and norm stats. Restore them BEFORE any train.
git checkout -- encoder/vocabs/
git status --porcelain encoder/vocabs/          # must be empty again
```

```bash
# 3. Build the priors sidecar on the GPU box (pure pandas, ~10 min, no GPU needed).
python -m player_priors
```

```bash
# 4. One retrain carrying W2.1 + W3a + W3b + W4 rung 1. Vocabs rebuild: the fusion width changed.
python train.py --full --name version3 --batch-size 64 --rebuild-vocabs
```

```bash
# 5. Widen the holdout pool 100 -> 700. Passes the prefix guard untouched.
python train.py --extend-holdout
```

```bash
# 6. Window 0 -- directly comparable to v2-run1..4, which scored these same 100 games.
python evaluate.py --model version3 --run v3-run1 --window 0 --monte-carlo 200 --procs auto
```

```bash
# 7. The probes and the baselines, on the finished run.
python -m reporting.state_probes results/version3/v3-run1 --seasons 2023
```

**Gates, in order:**

- **W2.1** — rookies and >6-ppg role-shifters reach parity with the season-average baseline on
  pts/ast/reb. Bane is the named case: his bias should move off −8.7.
- **W3** — `corr_home_away` ≈ 0.35, `pace_sd` ≈ 5.3, `margin_sd` ≈ 13.7, `margin_dispersion_ratio`
  ≈ 1.00, **with no shrinkage dial**. Starting point: 0.017 / 2.47 / 16.46 / 1.36×. All four are on
  the report's headline and in `run_summary.parquet`. `MARGIN_CALIBRATION_SLOPE` retires when this
  passes.
- **W4 rung 1** — read the A/B as *token-only* scheduled sampling, and check
  `scheduled_sampling_p` in `epochs.parquet` to confirm what schedule actually ran.
- **W1 probes, re-run** — the three behaviour gaps should narrow. If they do not, that is the
  finding, and it is the argument for rung 2.

Then, and only then, rung 2 as a **second pass** (it needs a finished bundle to roll out):

```bash
# 8. Rung 2: retrain one head warm-started off the finished bundle, selecting on rollout metrics.
#    Set ROLLOUT_SELECTION = True in config.py first.
python train.py --model event_time --name version3
python evaluate.py --model version3 --run v3-run1-rs --window 0 --monte-carlo 200 --procs auto
```

Steps 6 and 8 are a model comparison on the same window, so the **seed is held fixed** — that is the
one case where §8's "repeat runs use a different `--seed`" does not apply, and the run log should
say so. `v3_direction.md` §6 step 6 (rung 3) fires only if `checkpoint_selection.epochs_disagree` is
true in the run state.

---

## Not done, deliberately

- **W2.2** (learned recent-games summary), **W4 rung 3** (rollout-level fine-tuning, ~160 GPU-h),
  **W6** (copula re-centring). All three are gated in `v3_direction.md` §6 on results this pass
  produces.
- **The context-sensitivity probe** (§3 W1's first bullet). It needs the trained heads and a real
  forward pass per prefix; it belongs on the GPU box against the 3.0 weights, where its answer is
  about the model that will actually ship.
- **Conditional evaluation** from end-Q1 / half (§4's secondary scoreboard). `GameController.start`
  always builds a tip-off game and resets ~15 pieces of rules-side state; a `resume_from(history)`
  is a real piece of work and is not on the step 1–4 path.
