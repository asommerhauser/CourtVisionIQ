# CourtVisionIQ 2.0 — Progress Tracker

> **State, not spec.** [`v2_planned_changes.md`](v2_planned_changes.md) is the spec and is not
> edited during the build. This file records what has been built, what has been verified, and what
> was learned. Update it in the same commit as the work it describes.

## Standing rules

1. **GPU/CUDA/TensorFlow work is run by Alec in WSL, never by Claude.** That covers training,
   rollouts, evals, `main.py --clean`, and `pytest` — `tests/conftest.py:14` imports TF before
   anything else, so even a two-test subset is a TF run. Claude stops at each verification point,
   hands over a command, and waits for a pasted result.
2. **No large evals mid-programme.** Short smoke runs only (a handful of games, a few sims) to prove
   things run. The full 2.0 eval happens after the train.
3. **`v1.0` weights stop being meaningful at Gate B.** The re-clean introduces zone tokens, the
   `shooting 2pt` / `shooting 3pt` split and `timeout` — tokens the `v1.0` heads have never seen.
   **Gate A is the last point where an end-to-end sim run against existing weights says anything.**
   After it, verification is pytest plus inspection of cleaner output until the 2.0 train.

## Working pattern, per feature

1. `git checkout feature/version2 && git checkout -b feature/<name>`
2. Implement, committing at each meaningful step.
3. Stop. Hand over: what changed, the command, what to send back, what each outcome means.
4. On a green result: check it off here, commit, merge back to `feature/version2`.

---

## Status

Legend: `[ ]` todo · `[~]` in progress · `[x]` verified and merged

| # | Branch / gate | Phase | Spec | Status | Merge |
|---|---|---|---|---|---|
| 1 | `feature/side-aware-fouls` | 1 | §1 | [x] | f55758d |
| 2 | `feature/dead-ball-state` | 1 | §2 | [x] | 5ea3afd |
| — | **Gate A — Phase 1 short eval** | 1 | | skipped | - |
| 3 | `feature/shot-zone-geometry` | 2 | §3 | [ ] | |
| 4 | `feature/shot-zones` | 2 | §3 | [ ] | |
| 5 | `feature/ft-count-tokens` | 2 | §4 | [ ] | |
| 6 | `feature/fouled-player` | 2 | §5 | [ ] | |
| 7 | `feature/timeouts-team-rebounds` | 2 | §6 | [ ] | |
| 8 | `feature/schema-cleanup` | 2 | §7 | [ ] | |
| — | **Gate B — the re-clean + vocab rebuild** | 2 | | [ ] | |
| 9 | `feature/shared-backbone` | 3 | §9 pre | [ ] | |
| 10 | `feature/local-context` | 3 | §9 | [ ] | |
| 11 | `feature/rotation-model` | 3 | §8 | [ ] | |
| 12 | `feature/training-changes` | 3 | §10 | [ ] | |
| 13 | `feature/quarter-eval-splits` | 4 | §11 | [ ] | |
| — | **Gate C — pre-train checklist, then the 2.0 train** | 4 | | [ ] | |

Starting point: `feature/version2` at `82783c7`, docs-only ahead of `main`. No 2.0 code exists.

---

## Phase 1 — Controller rules

No new tokens, so these run against the existing `v1.0` weights.

### 1. `feature/side-aware-fouls` — §1

**Scope.** Resolve the fouler's side before typing the foul, mask foul types to what that side can
commit, and send free throws to the fouler's opponent in every branch.

Three defects in `simulation/controller.py`, fixed in order:

- `_team_of` (`:671`) reads `sim.home_roster`, which is the **on-court five** — declared
  `game_simulator.py:168`, mutated in place by `_apply_substitution` at `:361`. Any name not in the
  home five (a bench player, a subbed-off player, the `"start"` sentinel) resolves to `AWAY`.
  Replace with a full-roster team map built once at tip-off from `home_full` / `away_full`.
- `_do_foul` (`:467-514`) picks `allowed_types` at `:476` before the fouler exists, draws the fouler
  from `_all_ten()` with no side restriction, and computes `on_defense` at `:489` *after* the type.
  Reorder: sample fouler, resolve side, then mask.
- `_do_shooting_foul` (`:516-545`) never consults `self.possession`, so an offense-side fouler sends
  the free throws to the wrong team; the team-foul increment at `:544` is unconditional where `:493`
  is gated on `on_defense`. With the reorder in place this becomes correct by construction.

Then `_in_bonus` (`:641`) gets the last-two-minutes rule on the second qualifying foul, and the
team-foul definition is shared with the trained game-state feature (see Correction A).

**Verify**
```bash
pytest tests/test_controller.py tests/test_game_state_features.py -q
```
New cases required: an offense-side foul resolves to the offense, and a foul charged to a bench
player resolves to the right team.

**Result:** 61 passed, 0 failed (test_controller.py + test_game_state_features.py). encoder/vocabs/ clean after the run.

**Notes:** Beyond the spec: the side mask taken literally KILLS and-1s, because a made
basket flips possession before the foul is sampled, so the defender who fouled on the shot
reads as an offensive player and `shooting` is masked away. Added `_foul_offense()`, which
treats the scoring team as the offense while the previous row is a made field goal - see
correction F. Two existing tests adjusted: `test_do_foul_charges_the_fouler` scripted an
offense-side `personal` foul (now illegal, switched to `loose ball`), and
`FakeSim.start_with_starters` did not copy the full rosters the way the real simulator does
at `game_simulator.py:777`. Also: **pytest is not in requirements.txt or requirements-gpu.txt**
- it has to be installed into the venv by hand, and `python -m pytest` is the invocation that
works. This will bite at every verification step.

### 2. `feature/dead-ball-state` — §2

**Scope.** A real dead-ball flag, one possession tracker, no play straddling a period boundary.

- Add `self.ball_dead`, set in each `_do_*` handler per the spec's table. Replace the
  `pending_rebound` dead-ball proxy at `:428` and `:453` — today "no rebound pending" is the *only*
  dead-ball notion, so substitutions get injected mid-live-play.
- Delete the simulator's possession tracker (`game_simulator.py:174`, flipped at `:346`, written to
  every row at `:343`). It disagrees with the controller's routinely: a made FG flips the
  controller's but is not in `POSSESSION_FLIP_RESULTS` (`:75`), and a steal emits two flip-triggering
  rows where the controller flips once. **The emitted `possession` column has no consumer anywhere**
  — the only reader in the repo is an assertion at `tests/test_game_simulator.py:156`.
- Clamp `_advance_clock` (`:605`) at `_current_period_end` (`:636`), which already exists and is
  currently used only by `_in_bonus`. Note it is *not* the same value as `self.period_end`.

**Verify**
```bash
pytest tests/test_controller.py tests/test_game_simulator.py tests/test_chronology.py -q
```
New case required: a play sampled across a buzzer does not straddle it.

**Result:** 76 passed (controller + game_simulator + chronology). Full suite: 534 passed,
1 failed - `test_model_naming.py::test_holdout_skips_an_empty_processed_manifest`, pre-existing and
unrelated (see correction G). encoder/vocabs/ clean after both runs.

**Notes:** The boundary clamp resolves the play AT the buzzer rather than discarding it and
re-sampling, which is what the spec's wording asks for. Discarding would mean throwing away an
already-sampled actor and delta and restructuring every handler; clamping gets the stated
invariant (no event straddles a boundary) and the next step samples in the new period anyway.
A play landing exactly at 0.0 is a legal buzzer-beater. Revisit only if diagnostics show
something odd at period ends. Also: `start()` now puts the ball live, so no substitution is
possible until the first whistle - three scheduler tests had to state the dead ball they were
implicitly relying on the old `pending_rebound` proxy to provide. Timeouts and team rebounds
are dead-ball sources that do not exist yet; their hooks land in section 6.

### Gate A — Phase 1 short eval

The last point where a sim against `v1.0` means anything. Short, not a holdout sweep.

```bash
python evaluate.py --model v1.0 --run p1-smoke --holdout 8 --monte-carlo 5 --seed 7
```

Looking for: it runs clean, and FTA / PF / the offensive-defensive foul split move in the expected
direction. Send back the summary table, not the whole log. This is a sanity check, not a verdict.

**Result:** Skipped, deliberately (2026-09-06).

**Notes:** Not run. The cost was not judged worth the signal, and this eval was only ever
a sanity check, not a verdict. The consequence to be aware of: **Phase 1 now has no
measurement of its own.** If a later phase produces an ambiguous result, there is no
datapoint isolating the controller-rule changes from everything downstream, and no way to
recover one - the re-clean at Gate B makes the v1.0 weights unable to produce a meaningful
sim. Phase 1 rides on its 76 passing tests alone.

---

## Phase 2 — The re-clean

All five token changes land in **one** re-clean at Gate B. Each branch below is verified by pytest
alone; nothing here is measurable until the retrain.

### 3. `feature/shot-zone-geometry` — §3, geometry only

**Scope.** New root-level `zones.py` and `tests/test_zones.py`. No consumers, no risk.

Holds the both-baskets fold, the fifteen-token rule, `ZONE_TOKENS`, `ZONE_POINTS` and `is_three()`.
Root level because both the cleaner and the TF-free sim layer import it; it cannot live in
`simulation/stats.py`, which already imports `box_score.py` — its biggest consumer — so that would
cycle.

Also add a `python -m zones` entry point that prints the §3 validation table for a set of seasons,
so Gate B is a command rather than a manual query.

**Verify**
```bash
pytest tests/test_zones.py -q
```
Cases: both baskets map to the same token, every boundary, the raw `type` marker wins over geometry
on the 2pt/3pt call.

**Result:**

**Notes:**

### 4. `feature/shot-zones` — §3, consumers

**Scope.** Wire the fifteen zone tokens through the pipeline. A token-vocabulary widening; nothing
here samples or trains.

- `data_cleaner.py:501-505` stops dropping `converted_x` / `converted_y` / `shot_distance`;
  `:291-294` emits the zone instead of the `2pt`/`3pt` binary, feeding the assist (`:306`), shot
  (`:326`) and block (`:343`) rows. Gate it on shot rows — today it runs on every raw row.
- `models/conditional_type_model.py:114-121` — `shot_type` and `assist_type` `target_tokens` become
  the zone tuple. A loss-mask restriction only; the head already emits over the shared `type` vocab.
- `models/game_state_features.py:136-141` — points arithmetic through `ZONE_POINTS`, bit-identical
  to `simulation/box_score.py:216-241`. That file's `else: # 2pt` catch-all at `:230` becomes
  explicit, so an unknown token cannot silently score 2. `tests/test_game_state_features.py:176`
  pins the two together.
- `simulation/controller.py:48,59` — `SHOT_TYPES`, `FIELD_GOAL_TYPES`; `:271`, `:317` scoring.
- `config.py` — `SHOT_RESULT_BIAS` (`:75`) gains per-zone keys; `TYPE_BIAS` (`:106`) is already keyed
  head to token and needs no mechanism change. Any new dial name must go in `_TUNING_KEYS` (`:236`)
  or `set_dial` raises, and must obey the call-time-read contract (`config.<DIAL>`, never
  `from config import`) enforced by `tests/test_dials.py:110`.
- Six test files carry `"3pt"` literals: `test_box_score.py:35-36`, `test_controller.py:258-280`,
  `test_game_state_features.py:46,137,161,182`, `test_input_cache.py:53,120,224`,
  `test_model_persistence.py:210,217`, `test_predict_game.py:70,85`. `test_data_cleaner.py` does
  **not** assert the shot-type derivation at all — close that gap here.

**Verify**
```bash
pytest tests/ -q
```
This one touches enough surface that the full suite is the right check.

**Result:**

**Notes:**

### 5. `feature/ft-count-tokens` — §4

**Scope.** `shooting` splits into `shooting 2pt` and `shooting 3pt`. Two tokens, not fifteen.

`data_cleaner.py:501-505` stops dropping `outof`; `:443-459` labels each shooting foul from the
following trip's `outof`. The controller reads the token and awards that many free throws, deleting
the `predict_type("shot_type", ...)` call at `controller.py:540` — a head being asked a question it
was never trained to answer. `FOUL_TYPES` (`:52`) gains the two tokens and loses `shooting`.

And-1 stays a structural check on the previous made basket (`:530-533`) and becomes properly
reachable: the controller remembers the scorer, and a foul sampled as the very next play by the
scored-on team resolves as one free throw to that scorer.

This is a **cleaner labeling step, not a model input**. Carrying `num`/`outof` onto free-throw rows
as features is parked in `v3_planned_changes.md` §1.

**Verify**
```bash
pytest tests/test_data_cleaner.py tests/test_controller.py -q
```
Note `tests/test_controller.py:258` is `test_shooting_foul_on_3pt_yields_three_free_throws` — it
tests exactly the mechanism being replaced and will need rewriting, not just retokenizing.

**Result:**

**Notes:**

### 6. `feature/fouled-player` — §5

**Scope.** Foul rows carry the fouled player in `secondary_player`, from the raw `opponent` column.

Zero architectural cost, confirmed: `encoder/encoder.py:81-82` delegates `encode_secondary_player`
straight to `player_vocab` — it is not a separate vocab — and the embedding is weight-tied across
`event_time_model.py:578`, `player_model.py:357`, `conditional_time_model.py:291` and
`conditional_type_model.py:440`. Today the field is hard-set to `"none"` on every foul row
(`data_cleaner.py:455`), so there is no ground truth for who drew a foul.

Write raw `opponent` into it (`none` for technicals), and let `_pick_shooter`
(`controller.py:576`) give way to the fouled player carried on the row.

**Verify**
```bash
pytest tests/test_data_cleaner.py tests/test_controller.py tests/test_encoder.py -q
```

**Result:**

**Notes:**

### 7. `feature/timeouts-team-rebounds` — §6

**Scope.** The two events that most often make the ball dead become real events.

- A seventh entry in `TYPE_GEN_SPECS` (`conditional_type_model.py:114-121`) for `timeout_team` —
  one line; the spec machinery already generalizes over the six existing heads.
- `timeout` joins the legal event set (`controller.py:44`), gated on the dead-ball flag from §2 plus
  a per-team budget: 7 per game, 4 in the fourth, 2 in the last three minutes, +2 per overtime.
- `REBOUND_TYPES` (`:63`) grows to four (`offensive`, `defensive`, `team offensive`,
  `team defensive`); the controller skips the rebounder pick on a team token, and team rebounds land
  on the team line of the box score.
- `DEADBALL_REBOUND_PROB` goes entirely: `config.py:222`, its `_TUNING_KEYS` entry at `:237`, the
  comment at `controller.py:62`, and its single use at `controller.py:358` — where today it silently
  flips possession and emits **no row at all**.

**Verify**
```bash
pytest tests/test_controller.py tests/test_box_score.py tests/test_dials.py -q
```
`test_dials.py` matters here: removing a `_TUNING_KEYS` entry is exactly what it guards.

**Result:**

**Notes:**

### 8. `feature/schema-cleanup` — §7

**Scope.** One row per play where two are emitted today, and rows that should never have been kept.

- Steals collapse from two rows to one turnover row with the stealer as `secondary_player`
  (`data_cleaner.py:396-420`); `controller.py:341-342` must match.
- Offensive fouls emit one foul row with no trailing turnover; the box score counts the turnover
  from the foul.
- Standalone technicals (defensive three seconds, double technicals, coach technicals) become normal
  technical-foul rows where a player is named.
- **Not kept:** jump balls, period-end rows, `possession` as a column, raw ejection and violation
  rows.
- `self.output_columns` (`data_cleaner.py:57-61`) is dead code — never read, already stale. Delete
  it or make it the enforced schema contract; do not leave a third stale copy.

**Verify**
```bash
pytest tests/test_data_cleaner.py tests/test_chronology.py tests/test_preprocess.py -q
```

**Result:**

**Notes:**

### Gate B — the re-clean + vocab rebuild

**Delete the vocabs first.** They are append-only and frozen (`event_vocab.json` has 12 tokens,
`next_token: 12`), so rebuilding without deleting keeps every dead token — `shooting`, the bare
`2pt`/`3pt`, and the rest.

```bash
rm encoder/vocabs/*.json
python main.py --clean --rebuild-vocabs --model event_time
python -m zones --seasons 2003,2013,2023
```

The zone table must match §3 in all three eras: rim make rate climbing, `corner3 > wing3 > top3` in
every era, 3PA share rising 18.1% to 38.4%, the deep corner two dying off, left/right volumes
near-symmetric. A missing zone or a broken make-rate ordering means the geometry is wrong — stop and
fix it rather than training on it.

Send back the printed table. Commit `encoder/vocabs/*.json` after this so a cloud clone matches.

**From here on, `v1.0` weights cannot produce a meaningful sim.**

**Result:**

**Notes:**

---

## Phase 3 — Model

### 9. `feature/shared-backbone` — §9 prerequisite

**Scope.** Extract the transformer backbone that is currently copied into six model files. Pure
refactor, no behaviour change, its own branch so a regression in it stays separable from the local
attention that follows.

Verified as a genuine six-way **byte-identical** copy — the 23-line region from the positional
embedding through `final_ln` hashes the same in every one:

| File | region |
|---|---|
| `models/event_time_model.py` | `:611-633` |
| `models/player_model.py` | `:392-414` |
| `models/conditional_time_model.py` | `:326-348` |
| `models/conditional_type_model.py` | `:482-504` |
| `models/substitution_model.py` | `:524-546` |
| `models/stint_length_model.py` | `:441-463` |

The `def model(self, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=0.2)`
signature is duplicated too, as is the `Dense(D)` fusion / `fusion_ln` / `AddPositionalEmbedding` /
`emb_dropout` / `KeyPaddingMask` prologue. Divergence starts only *below* `final_ln`, at the output
heads, and *above* the region in which conditioning vectors get concatenated.

**Verify**
```bash
pytest tests/ -q
```
`tests/test_model_persistence.py` is the one that matters: layer names must not change, or weight
reload breaks against existing artifacts.

**Result:**

**Notes:**

### 10. `feature/local-context` — §9

**Scope.** Two of the eight attention heads in every block restricted to the last eight rows by a
banded mask — the same mechanism as the padding mask — behind a `LOCAL_ATTENTION_HEADS` config
switch so it is A/B-able without a re-preprocess. Six heads stay global. No new inputs, no custom
kernel.

Plus a shot-clock proxy: one new per-row number, seconds since the current possession started, reset
on a change of possession or an offensive rebound, scaled to the 24-second clock. Derived inside
`GameStateScan` (`models/game_state_features.py:101`) so training and the simulator compute it
identically by construction; `GAME_STATE_KEYS` (`:47`) and `_NORM` (`:54`) gain the key, and
`simulation/input_cache.py:178` picks it up for free through the incremental path.

If diagnostics say the proxy does nothing, it is one key to remove.

**Verify**
```bash
pytest tests/test_game_state_features.py tests/test_game_state_wiring.py tests/test_input_cache.py -q
```

**Result:**

**Notes:**

### 11. `feature/rotation-model` — §8, full version

**Scope.** The largest branch. Substitutions move inside the model; the stint-length scheduler and
the fatigue nudge retire.

- Three more per-player scalars alongside `rest_proj` in `models/roster_set_encoder.py` (`:71`,
  `:99-102`, `:110-120`): seconds in the current stint, minutes played, personal fouls. Every head
  sees them wherever it consumes the lineup.
- A bench bundle: up to ten available bench players through the same set encoder, each with seconds
  since they sat, minutes played, fouls, and whether they have played.
- A new `models/sub_decision_model.py`, registered in `models/registry.py:23,41`, asked only at dead
  balls: per team, how many substitutions follow before the ball is live (`0 / 1 / 2 / 3+`).
- Retire `_schedule_stint` (`controller.py:411`), `_process_scheduled_subs` (`:420`), `_fatigue_bias`
  (`:389`) and `models/stint_length_model.py`. `_maybe_force_sub` (`:447`) stays as the single
  backstop. `STINT_SAMPLE_SIGMA`, `STINT_LENGTH_SCALE`, `STINT_MAX_SECONDS` and `SUB_FATIGUE_WEIGHT`
  leave `_TUNING_KEYS`.

**Two traps.**
- `RosterEncoderParams` is a frozen dataclass with a `get_config` / `from_config` round-trip
  (`:131-146`, `:148-153`). Every new param must be in **both** or weight reload breaks.
- `encoder/vocabs/norm_stats.json` carries exactly one per-player pair today (`rest_mean`,
  `rest_std`). Three new scalars need normalization stats plumbed through `models/norm_stats_io.py`
  — a step the spec's next-steps list does not mention.

**Smaller version, if this proves too much for one train:** keep the stint scheduler but gate it on
dead balls, and ship only the two set-encoder bundles. Bench rest is still learned, no new head.
Does not block anything else.

**Verify**
```bash
pytest tests/ -q
```
Watch `test_model_persistence.py`, `test_substitution_model.py`, `test_oncourt_mask.py`.

**Result:**

**Notes:**

### 12. `feature/training-changes` — §10

**Scope.** Clutch loss weighting, and loss masked to the positions the simulator actually queries.

New `apply_clutch(mask, split)` in `models/game_state_features.py`, a sibling of `apply_recency`
(`models/season_features.py:162`) — the single funnel all six heads already call. Rows that are close
and late (`period_idx >= 3`, `period_time_left <= 300`, `abs(score_diff) <= 8`) count double.
`CLUTCH_LOSS_WEIGHT = 1.0` disables it, which makes an A/B against the same preprocess trivial.

Call sites (the spec's line numbers are stale by up to 12 lines; these are current):

| File | `apply_recency` call |
|---|---|
| `models/event_time_model.py` | `:687` |
| `models/player_model.py` | `:463` |
| `models/conditional_type_model.py` | `:557` |
| `models/substitution_model.py` | `:597` |
| `models/stint_length_model.py` | `:493` |
| `models/conditional_time_model.py` | `:377` |

Two call shapes exist — mask built first then wrapped (`event_time`, `conditional_time`), and wrapped
inline (the other four) — both ending at `sample_weights = {<output_name>: mask}`.

**Shape trap.** The recency weight is per-*game*, `(N,)` reshaped to `(-1,1)`; the clutch weight is
per-*row*, `(N, SEQ)`, and the game-state columns in `split` are `(N, SEQ, 1)` and need a reshape.
Normalization is fixed constants (`_NORM`, `:54`), so thresholds convert to normalized units exactly
— there is no need to carry raw arrays alongside.

Also here: the event and time heads stop training on rows the simulator never asks about —
continuation rows, controller-forced substitution rows, and for the time head the last row before a
period break. The continuation rule is **one function shared with the controller's play expansion**
so the two cannot drift.

`CLUTCH_LOSS_WEIGHT` is a **training** knob and does **not** go in `_TUNING_KEYS`.

`tests/test_game_state_wiring.py` covers event_time, conditional_type, conditional_time and
stint_length but **not** `player` or `substitution`. Close that gap here.

**Verify**
```bash
pytest tests/test_game_state_wiring.py tests/test_recency.py tests/test_preprocess.py -q
```

**Result:**

**Notes:**

---

## Phase 4 — Measurement

### 13. `feature/quarter-eval-splits` — §11

**Scope.** Period-sliced box scores in the eval record and the report, plus a per-zone shot-mix
diagnostic. Build it before the train so the train is measurable.

Confirmed absent today: `simulation/eval_metrics.py` is game-level only, and the existing
"progression" (`reporting/eval_report.py:508`, `:684`) is **tuning** progression — segmenting a run
by dial changes — not game periods. Nothing anywhere splits by quarter.

- `simulation/evaluation.py:313-396` — period-sliced boxes alongside the whole-game one on the
  per-game record, using the same period constants `GameStateScan` uses.
- `reporting/eval_report.py:591-613` — a `_quarter_section` in the sections list, reusing the
  stat-registry-driven `_accuracy_section` (`:412`), so per-quarter pace / eFG / points come nearly
  free.
- `reporting/eval_report.py:767-773` — `box_quarters.parquet` alongside the existing five frames.
- `simulation/diagnostics.py:144` — per-zone shot-mix histogram in `compare_holdout`. It is a
  distribution comparison, not a per-player accuracy stat, so that is the right home.

**Independent of everything else** — it can be pulled forward at any time at no cost if quarter
splits would help read an earlier smoke run.

**Verify**
```bash
pytest tests/test_evaluation.py tests/test_reporting.py -q
python evaluate.py --model v1.0 --run full4-s100 --report-only
```
The `--report-only` rebuild runs over an existing finished run and starts no new sims.

**Result:**

**Notes:**

### Gate C — pre-train

- `pytest tests/` green.
- **Check `encoder/vocabs/norm_stats.json` and the vocab files before training.** pytest overwrites
  the committed encoder artifacts; confirm the freeze came from a real clean, not test residue.
- Zone table matches §3 in all three sampled eras.
- Derived-vs-raw 3pt disagreement under 1% per season.

```bash
python train.py --full --name full_train_3 --batch-size 64 --clean --rebuild-vocabs
```

Train 2's availability masking and capacity settings carry forward unchanged.

**Post-train:** per-zone make rates against the era table; the per-quarter section flat across Q1–Q3
(a Q1 regression means `CLUTCH_LOSS_WEIGHT` is too high); an A/B of `CLUTCH_LOSS_WEIGHT` 1.0 vs 2.0
against the same preprocess; then the dial package fitted from zero, re-keyed per zone.

**Result:**

**Notes:**

---

## Corrections to the spec

Found while verifying `v2_planned_changes.md` against the code. Recorded here so they are not
rediscovered; the spec itself is left alone.

**A. §1's shared team-foul constant is an inclusion/exclusion mismatch.** The spec says to share
`TEAM_FOUL_TYPES` (`simulation/controller.py:72`) with `models/game_state_features.py:42` — but that
site is `NON_TEAM_FOUL_TYPES = {"technical", "offensive"}`, an *exclusion* set. The two are not the
same rule: the exclusion form counts an unknown token as a team foul, the inclusion form does not.
Pick a direction deliberately, and put the shared constant where both a TF-free sim module and a
model module can import it — the same reasoning that puts `zones.py` at root level.

**B. §10's `_make_dataset` line numbers are stale** by up to 12 lines. Current values are in the §12
table above; only `player_model.py:463` still lands exactly.

**C. §8 omits `models/norm_stats_io.py`.** `norm_stats.json` carries exactly one per-player pair
(`rest_mean` / `rest_std`), matching the single scalar in `RosterSetEncoder`. The three new
per-player scalars need normalization stats plumbed through that file.

**D. §1's description of `_do_shooting_foul` is right in substance, imprecise in wording.** It says
`:528` "hard-codes" `shooting_team`; the line actually *derives* it as `self._other(fouler_team)`.
The defect is the same — it assumes the fouler is a defender and never consults `self.possession` —
but the fix is the reorder in `_do_foul`, not an edit to that line.

**E. `DEADBALL_REBOUND_PROB` emits no row today.** `controller.py:358` flips possession and returns
without appending anything. §6 replacing it with learned team-rebound tokens therefore *adds* rows
to the generated sequence, which is a change in row counts, not just in sampling.

**F. The side mask, taken literally, makes and-1s unreachable.** A made basket flips possession
the instant it drops (`_do_shot`), so a foul sampled as the very next play classifies the defender
who fouled on the shot as an *offensive* player - and `shooting` is not in the offensive side's
mask. Section 4 explicitly wants and-1s reachable, so leaving this would have set up a conflict two
branches later. `GameController._foul_offense()` resolves it: while the previous row is a made
field goal, the possession that just ended is the one the foul belongs to, so the scoring team is
the offense. Found and fixed on `feature/side-aware-fouls`.

**G. `test_holdout_skips_an_empty_processed_manifest` fails on a machine that has trained.**
`shell/session.py:149` reads `./training/full_run_state.json` from the CWD, but the test only
isolates `processed_dir` via `tmp_path`. On this machine that file is real (untracked, 100
holdout ids, from the Aug 23 train), so `resolve_holdout` returns at `:153` instead of raising
and the test fails. It is a test-isolation bug, not a product bug, and it is **not** caused by any
2.0 work - it fails identically on `main`. It does block Gate C's "pytest tests/ green", so fix it
before then: the test needs to isolate the training-state path too.

---

## Log

Append one line per merge. Newest last.

| Date | Branch | Commit | Note |
|---|---|---|---|
| 2026-09-06 | `feature/side-aware-fouls` | f55758d | 61 tests green; and-1 fix beyond spec (correction F) |
| 2026-09-06 | `feature/dead-ball-state` | 5ea3afd | 76 green; found pre-existing test-isolation bug (correction G) |
