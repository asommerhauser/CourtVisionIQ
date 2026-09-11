# CourtVisionIQ 2.0 — Progress Tracker

> **State, not spec.** [`v2_planned_changes.md`](v2_planned_changes.md) is the spec and is not
> edited during the build. This file records what has been built, what has been verified, and what
> was learned. Update it in the same commit as the work it describes.

## Standing rules

These are Alec's, given at the outset, and they are not negotiable conveniences — every one of
them exists because the alternative wasted real time.

### 1. Claude never runs anything that touches TensorFlow. Alec runs it, in WSL.

That covers **training, preprocessing, rollouts, evals, `main.py --clean`, `train.py`, and
`pytest`**. `pytest` is on the list because `tests/conftest.py` imports TF before anything else,
so even a two-test subset is a TF run.

Two independent reasons, and either alone would be enough:

- **TF does not load on the Windows side at all.** `import tensorflow` dies in
  `_pywrap_tensorflow_internal` (a DLL init failure). Anything importing `simulation/` hits it too,
  because `simulation/__init__.py` imports `GameSimulator`. So a Claude-side run does not produce
  a wrong answer, it produces a traceback.
- **The GPU is only visible from WSL.** Native Windows TF cannot see the 4070, so even where a run
  starts it falls back to CPU and stalls.

**The pattern is therefore: Claude stops at each verification point, states what changed, hands
over the exact command, says what to send back and what each outcome would mean — then waits for
a pasted result.** Do not guess at the outcome, do not proceed as if it passed, and do not report
a branch as verified on anything less than a pasted run.

### 2. What Claude CAN run, and should

TF-free CPU passes over the cleaned data, which is where most of this programme's real findings
came from:

```bash
python -m zones --seasons 2003,2013,2023
python -m models.game_state_features --seasons 2003,2013,2023
python -m models.rotation_features --seasons 2003,2013,2023
```

Plus anything in plain pandas/numpy — ad-hoc measurement over `data/*.csv` and
`RawData/MasterFiles/*.csv`, and driving the cleaner or a scan directly in-process. Those are
cheap, they are not gated, and **they are what found the jump-ball collapse, the 12% possession
over-count, the roster-snapshot flicker and the split Nene embedding.** None of those was found
by a test.

### 3. Commit at each meaningful step

Not one commit per branch. Each commit's message says what changed and *why*, including what was
measured and what was rejected — the messages are the record of reasoning, and several of them
are the only place a rejected alternative is written down.

### 4. No large evals mid-programme

Short smoke runs only (a handful of games, a few sims) to prove things run. The full 2.0 eval
happens after the train.

### 5. Weights stopped being meaningful at Gate B

The re-clean introduced zone tokens, the `shooting 2pt` / `shooting 3pt` split and `timeout` —
tokens the `v1.0` heads have never seen. Gate A was the last point where a sim against existing
weights meant anything. **There is no loadable model and has not been since Gate B**; verification
is pytest plus TF-free measurement until the 2.0 train.

## START HERE (as of 2026-09-09)

> **2026-09-09, pre-train review:** `v2_review_2026-09-09.md` is the theory review of the whole
> programme — what changed by mechanism, what the build found in the data, a metric-by-metric
> forecast, and the recommended order of post-train measurement (baseline comparison first).
> Read it before Gate C's train.

> **2026-09-11, train 3 cancelled and the tree made train-ready.** Attempts 1 and 2 are dead: a
> Keras 3 `Dense` OOM (correction V, fixed on `fix/rowff-rank2`), a pod running v1.0 code because
> the branch had never been pushed, and a missing subset manifest that would have trained every
> head on the full corpus at 5.4x the epoch cost. All three are written up as corrections V and W.
> The last two are now impossible rather than documented: `full_run` extracts the manifest itself,
> `run_stage` raises on a split conditional group, and the README's pod procedure names a branch,
> checks provenance, and runs the train under `nohup` into a log file. **Push the branch before
> cloning it on a pod** — that is the step that was missing.

**Workstream 11 is complete. Phases 1 and 2, Gate B, all of §9 and all of §8 are merged into
`feature/version2`.** The full suite is green.

**What is left is 12, 13 and Gate C.** Workstream 12 is **built and awaiting its run** — read its
section, not §10, and note that its scope grew twice during the build: `ConditionalTimeModel`
needs the mask too, and unifying the free-throw trip exposed correction T. Nothing in 12 has been
executed under pytest yet.

### Where the tree stands

| | |
|---|---|
| branch | `feature/version2`, clean tree, everything merged |
| cleaned data | 2.0, all 21 seasons, **current** — re-cleaned at workstream 12 (correction Q) |
| vocabularies | frozen and committed, **`be87b39`** — re-frozen at workstream 12's clean, which purged `Nene ` (one token, 2153 -> 2152); the other four are byte-identical to `0ab3956` |
| suite | green, **739** — 719 before workstream 12, +20 from it |
| weights | **none usable** — see standing rule 5 |

**One clean is owed, and workstream 12 is where it rides.** `data/` predates correction Q (a
substitution naming a player already on the floor), and 12 needs a re-preprocess anyway to write
the loss-mask arrays. The vocabulary purge decided at Gate B rides here too, so the command is:

```bash
rm encoder/vocabs/*.json
python main.py --clean --rebuild-vocabs --model event_time
```

The `rm` is deliberate and is the **one** intended vocabulary change. `--rebuild-vocabs` appends
rather than rebuilding, so deleting first is the only way to drop the dead `Nene ` token (trailing
space, id 232, with the real `Nene` at 2152 — correction P). It renumbers every player id above
232, which is free while there are no loadable weights (standing rule 5) and stops being free the
moment train 3 finishes. **Gate C's "the vocabularies must come back byte-identical" therefore
re-baselines against this clean, not against `0ab3956`** — that check was written before the purge
was scheduled, and the two cannot both hold.

No other token changes since the freeze, so nothing else should move.

### Two standing guards

Neither has fired since Gate B. A failure from either is the system working, not a bug to route
around:

1. **`zones.points_for_shot` raises** on any shot `type` that is neither a zone nor
   `free throw`. A stray token aborts the preprocess instead of silently scoring it as two.
2. **`DataCleaner._check_schema` raises** if any emitted event's keys differ from
   `OUTPUT_COLUMNS`. A missing key would otherwise become a silent all-NaN column.
3. **`models.pipeline.run_stage` raises** if some but not all conditional heads are in
   `SUBSET_MODEL_KEYS`. They share one `cond_*.npz`, so they share one partition; the invariant
   used to be a comment that held only because `shot_type` comes first in `TYPE_GEN_SPECS`
   (correction W). Added 2026-09-11.

### What this programme has actually taught, twice each

Both of these were written down mid-programme and then violated inside the same branch that
recorded them. They are here because they keep costing time:

1. **Tests do not find the real bugs; one number against an independent source does.** The
   jump-ball collapse survived 663 green tests. The 12% possession over-count survived 26 tests
   written for it. The roster-snapshot flicker survived everything. Every one was caught by
   computing the same quantity two ways and comparing. **Every workstream should decide, before
   writing code, what its independent number is** — and it is never a unit test.
2. **A list read in more than one place must live somewhere both can import BEFORE anything is
   added to it.** `ARCH_KEYS` existed twice and drifted (correction N). The controller's required
   heads existed four times and drifted, breaking 87 tests that had nothing to do with the change
   (correction R). Same lesson, same branch, weeks apart.

And one about gates: **a loose gate is close to no gate.** The first pace band was drawn from
published NBA figures, which are per-48-minute and exclude playoffs, so it had to be wide enough
that 109.4 only just failed and 104.0 would have passed silently. Prefer a reference computed
from the same file by an independent route.

### Line numbers in this document are unreliable

They have gone stale three times: the backbone extraction removed ~24 lines from each of six
model files, Phases 1-2 grew the controller by ~130, and workstream 11 rewrote its rotation
section entirely. **Grep for the symbol.** That is the lesson, not any table of anchors.

## Working pattern, per feature

1. `git checkout feature/version2 && git checkout -b feature/<name>`
2. Implement, committing at each meaningful step.
3. Stop. Hand over: what changed, the command, what to send back, what each outcome means.
4. On a green result: check it off here, commit, merge back to `feature/version2`.

---

## Status

Legend: `[ ]` todo · `[~]` in progress · `[x]` verified and merged · `[x]*` merged WITHOUT a test run

| # | Branch / gate | Phase | Spec | Status | Merge |
|---|---|---|---|---|---|
| 1 | `feature/side-aware-fouls` | 1 | §1 | [x] | f55758d |
| 2 | `feature/dead-ball-state` | 1 | §2 | [x] | 5ea3afd |
| — | **Gate A — Phase 1 short eval** | 1 | | skipped | - |
| 3 | `feature/shot-zone-geometry` | 2 | §3 | [x] | 8434786 |
| 4 | `feature/shot-zones` | 2 | §3 | [x] | 27a3327 |
| 5 | `feature/ft-count-tokens` | 2 | §4 | [x] | 4b0dd09 |
| 6 | `feature/fouled-player` | 2 | §5 | [x] | 715cedb |
| 7 | `feature/timeouts-team-rebounds` | 2 | §6 | [x] | b54685f |
| 8 | `feature/schema-cleanup` | 2 | §7 | [x] | 0605a35 |
| — | **Gate B — the re-clean + vocab rebuild** | 2 | | [x] | 0ab3956 |
| 9 | `feature/shared-backbone` | 3 | §9 pre | [x] | b9ff4ea |
| 10a | `feature/local-attention` | 3 | §9 | [x] | f402341 |
| 10b | `feature/possession-clock` | 3 | §9 | [x] | bfb52c1 |
| — | `fix/jump-ball-team-binding` | 2 | — | [x] | fac7095 |
| — | `fix/free-throw-possessions` | 3 | §9 | [x] | bfb52c1 |
| 11a | `fix/roster-snapshot-flicker` | 3 | §8 | [x] | c9c1d2d |
| 11b | `feature/lineup-state` | 3 | §8 | [x] | 4fa3713 |
| 11c | `feature/bench-bundle` | 3 | §8 | [x] | 402c182 |
| 11d | `feature/sub-decision-head` | 3 | §8 | [x] | 1b48794 |
| 12 | `feature/training-changes` | 3 | §10† | [x] | be87b39 |
| 13 | `feature/quarter-eval-splits` | 4 | §11 | [x] | a9848c9 |
| — | **Gate C — pre-train checklist, then the 2.0 train** | 4 | | [ ] | |

† Workstream 12 builds only half of §10: the clutch loss weighting was started and dropped
on 2026-09-09 (see its section, and correction S). The loss masking is still to build.

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

**Result:** 42 passed. `python -m zones --seasons 2003,2013,2023` run and **all gates
pass** - the section 3 validation table is already satisfied, ahead of Gate B.

**Notes:** Measured table, matching the spec's independently measured numbers to a rounding
step:

| | 2003 | 2013 | 2023 |
|---|---|---|---|
| `rim` | 30.5% @ 58.9% | 33.1% @ 60.1% | 30.3% @ 66.1% |
| `paint` | 14.7% @ 39.1% | 14.3% @ 38.6% | 19.3% @ 44.4% |
| `mid_corner_l` | 5.4% @ 38.9% | 3.3% @ 40.0% | 0.6% @ 40.4% |
| `corner3_l` | 2.5% @ 36.7% | 3.5% @ 38.5% | 5.1% @ 38.5% |
| `wing3_l` | 4.6% @ 35.2% | 5.9% @ 35.0% | 9.8% @ 35.8% |
| `top3` | 3.9% @ 34.9% | 5.4% @ 34.5% | 10.2% @ 34.9% |
| `heave` | 0.4% @ 4.7% | 0.4% @ 3.8% | 0.4% @ 12.8% |
| 3PA share | 18.4% | 24.4% | 38.8% |
| marker disagreement | 0.39% | 0.23% | 0.11% |
| coord coverage | 100.0% | 100.0% | 99.4% |

One departure from the spec's geometry - see correction H. `python -m zones` is a ~6s CPU
pass per season over the raw files (no TF, no CUDA), so it is cheap to re-run any time the
geometry is touched. Re-run it at Gate B against the re-clean.

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

**Result:** Full suite run and reported green by Alec (verbal, not a pasted
transcript). encoder/vocabs/ clean afterwards.

**Notes:** Thirteen test files carried the old literals, not the six the spec predicted -
every fixture with a made `"2pt"` row now reaches the strict `points_for_shot`. They map to
`paint` and `top3`, which keeps every score assertion numerically identical.

`zones.points_for_shot` **raises** on an unrecognized token rather than defaulting to 2, per
the spec's "an unknown token must not quietly score 2". Both scoring scans call it, so the
box score and the trained score feature are bit-identical by construction rather than by
convention. The trade: one malformed row aborts a 21-season preprocess instead of silently
training the model on a running score that never happened. **Watch for this at Gate B** -
it is the most likely way the re-clean fails loudly.

Verified locally before handoff (box_score and game_state_features import without a TF
session): all fifteen zones score identically in both scans, free throws score 1, and
`"2pt"`/`"3pt"`/garbage all raise. The cleaner was also run end to end on synthetic raw
rows. The training path - preprocessing, the `target_tokens` loss mask, model persistence -
could only be covered by the suite.

New dial `SHOT_RESULT_BIAS_BY_ZONE` (in `_TUNING_KEYS`); `TYPE_BIAS["assist_type"]` emptied
because `"3pt"` can no longer match a token. Both need fitting from zero post-train.

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

**Result:** 146 passed / 1 failed on the targeted set, then 52 passed on dials+shell
after the fix. The failure was mine: `test_get_dials_returns_deep_copies` read
`TYPE_BIAS["foul_type"]["shooting"]` from the live module dict, which the split
renamed. It now names the token off the live dict so the same rename cannot break it
again (commit 5748060).

**Notes:** The spec gives no rule for and-1s, and they are too big to ignore: `outof == 1`
is **24%** of shooting fouls in 2022-23 (4584 of ~18k sampled), and maps to neither token.
98.6% of them sit directly behind a made field goal, so they are labelled from what that
basket was worth - the same question ('was the fouled attempt a 2 or a 3') answered from
the other side. See correction I.

Validated by running the real `_label_shooting_fouls` over the whole 2022-23 file: **27708
of 27708** shooting fouls labelled, 96.4% two-shot / 3.6% three-shot - a realistic
three-shot-trip share, and proof the fallback path is rare rather than quietly absorbing
everything. The lookahead spans an intervening substitution (observed gap up to 7 rows) and
never crosses a period boundary; 0.17% find no trip and fall back to 2pt.

`simulation/controller.py` now imports the two tokens from `data_cleaner` - a new
sim-reads-cleaner dependency direction. Justified by the controller's own docstring ('the
cleaned-data semantics are the source of truth') and better than duplicating literals that
could drift, but if the layering matters later, a small shared constants module is the
alternative.

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

**Result:** 144 passed (data_cleaner + controller + encoder).

**Notes:** Measured before writing anything: the raw `opponent` column is **100% populated
for every non-technical foul type** in 2022-23 and **100% empty for technicals** - exactly
the split the spec assumed - and it matches the actual free-throw shooter **99.5%** of the
time. That last figure is what justifies collapsing the victim and the shooter into one
draw instead of a foul row naming nobody plus an unrelated draw.

`_pick_shooter` survives for **technicals only**, against the spec's "gives way to the
fouled player": a technical genuinely has no victim, and someone still has to shoot.
Collapsing it entirely would mean inventing a name for a field the real data leaves blank.

`_do_shooting_foul` is reordered so the and-1 check runs before the foul row is appended
(`prev` moved from `history[-2]` to `history[-1]`); the and-1 case needs no draw at all
because the scorer is the player who was fouled.

Cost: **every foul now draws a victim**, where before only fouls producing free throws drew
anyone. One extra player-head call per foul, ~40 a game.

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

**Result:** Full suite: 628 passed, 1 failed - the pre-existing correction G failure,
unrelated. Three commits: 7eccf81 team rebounds, 6e13d81 timeouts, b54685f the
bare-team-rebound side recovery.

**Notes:** See correction J for the team-rebound population analysis - the short version is
that the side IS recoverable at 99.8% accuracy, but two thirds of the bare rows are not
rebounds at all.

Timeouts are gated on `timeout_team` being loaded, which the spec does not mention. Alec
confirmed old weights will never meet 2.0 data, so this is now a cheap safety net rather
than a compatibility requirement; the FakeSim tests use it to exercise the off path.

Timeout volume: 14,477 in 2022-23, ~12 a game, `team` populated on 100% and both
abbreviations always resolved before the first one - nothing is dropped in practice.

`_event_menu` extracted from `_step` so the dead-ball + budget + head-loaded gate is one
readable function, testable without a live event head.

**Suite noise:** the full run emits ~18.5k warnings, 99.9% of them one Keras/numpy-2
`__array__ copy keyword` DeprecationWarning fired per array conversion in
test_model_persistence (14,169) and test_game_simulator (4,353). Library-level, not ours.
The repo has no pytest config at all; a `filterwarnings` entry would silence it.

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

**Result:** Full suite: **634 passed, 1 failed** - the pre-existing correction G
failure, unrelated. Run after the merge rather than before it: the branch was merged on
Alec's instruction to assume it passes while moving to a fresh chat for Gate B, and the
run was collected on the next session. The assumption held.

**Notes:** Verified locally without pytest (the cleaner and box score import without a TF
session): steal -> one row with the stealer in `secondary_player`; plain turnover -> `none`;
offensive-foul pair -> one foul row, no turnover; defensive three seconds -> a technical
foul row; coach technical -> dropped; emitted columns matched the contract exactly. Box
score: ball-loser gets the TOV, stealer gets the STL, offensive foul gives `tov=1, pf=1`.

**Standalone technicals were never being cleaned at all** - 632 rows a season file under
`event_type="technical foul"`, which the cleaner did not look at. Listed in the spec as
tidy-up; it is actually data recovery.

**`"steal"` is no longer a result token anywhere** - it survives only as a type. A
vocabulary change that only lands because Gate B deletes the old vocabs.

`OUTPUT_COLUMNS` is now enforced: `parse_file` raises if any emitted event's keys differ.
That check exists precisely for Gate B.

### Gate B — the re-clean + vocab rebuild

**Delete the vocabs first.** They are append-only and frozen, so a rebuild without deleting keeps
every dead token — `2pt`, `3pt`, the bare `shooting`, and `steal` as a result.

```bash
rm encoder/vocabs/*.json
python main.py --clean --rebuild-vocabs --model event_time
python -m zones --seasons 2003,2013,2023
```

The clean is CPU-only and takes a while over 21 seasons. `python -m zones` is ~6s per season and
reads the RAW files, so it validates the geometry independently of whatever the clean produced.

**Checks, in order:**

1. **The clean completes.** If it aborts, read the traceback against the two guards above — that
   is most likely the system catching a real problem, not an incidental crash.
2. **The zone table still passes.** It did on 2026-09-07 (all gates, all three eras — see
   workstream 3). It reads raw files, so a change here means the geometry moved, not the cleaner.
3. **The rebuilt vocabs contain the new tokens and none of the dead ones.** This is the check
   that has no existing tooling, and the one most worth writing:
   - `type_vocab.json` should contain the fifteen zone tokens, `shooting 2pt`, `shooting 3pt`,
     `team offensive`, `team defensive`, `home`, `away` — and should **not** contain `2pt`,
     `3pt`, or a bare `shooting`.
   - `event_vocab.json` should contain `timeout`.
   - `result_vocab.json` should **not** contain `steal`.
4. **Spot-check the cleaned output** against the per-season expectations measured from the raw
   files during the build (2022-23): ~27.7k shooting fouls split 96.4% / 3.6% two-shot vs
   three-shot; ~14.5k timeouts; ~11.9k typed team rebounds plus ~909 recovered bare ones; foul
   rows carrying a fouled player for every non-technical.
5. **`encoder/vocabs/*.json` gets committed** after the clean, so a cloud clone matches.

**Result:** **PASSED**, on the third run. The first two failed check 4 on timeouts (the second
reproduced the first exactly, because the fix had not been merged when it started); the third
ran with `fix/jump-ball-team-binding` in the tree and passes every check.

| check | outcome |
|---|---|
| 1. the clean completes | 21 seasons, no traceback — neither guard fired |
| 2. zone table | all gates, all three eras, matching workstream 3 to a rounding step |
| 3. vocabs | PASS — fifteen zones, both shooting tokens, team rebounds, `home`/`away`, `timeout`; no `2pt`/`3pt`/bare `shooting`; `steal` gone as a result |
| 4. spot-check | all five match (see below) |
| 5. vocabs committed | `0ab3956` |

| 2022-23 | measured at build | after the clean |
|---|---|---|
| shooting fouls | ~27,708, 96.4% / 3.6% | 27,708, 96.4% / 3.6% |
| team rebounds | ~11,884 typed + ~909 bare | 12,793 |
| timeouts | ~14,477 | 14,477, split away 7,287 / home 7,190 |
| steals | one row, stealer named | 19,167, 100% named; `steal` as a result: 0 |
| fouled player | 100% of non-technicals | 99.86% |

**Below is the record of the first run, kept because correction L came out of it.** The clean
completed over all
21 seasons with no traceback, so neither guard fired. The vocab check passes outright: all
fifteen zones, `shooting 2pt`/`shooting 3pt`, `team offensive`/`team defensive`, `home`/`away`
and `free throw` are present; `2pt`, `3pt` and bare `shooting` are gone; `timeout` is in the
event vocab; `steal` is gone from the result vocab and survives only as a type. The 2022-23
spot-check matched the build-time measurements **exactly** on three of four counts:

| | measured at build | after the re-clean |
|---|---|---|
| shooting fouls | ~27,708, 96.4% / 3.6% | 27,708, 96.4% / 3.6% |
| team rebounds | ~11,884 typed + ~909 bare | 12,793 (= 11,884 + 909) |
| steals | one row, stealer named | 100% named; `steal` as a result: 0 |
| fouled player | 100% of non-technicals | 99.9% (technicals 100% `none`) |
| **timeouts** | **~14,477, ~12/game** | **11,134, 8.4/game, split 65/35 home** |

The timeout line is a real bug, not a measurement artefact - see correction L. Fixed on
`fix/jump-ball-team-binding`; **the clean has to run again**, since everything in `./data` was
produced by the broken binding.

**Notes:** The zone table (check 2) reads raw files and is unaffected by the cleaner bug, but
it has not been re-run since workstream 3.

A process note worth keeping: the re-clean was run twice, and the second run reproduced the
first exactly because the fix had not been merged when it started. Check that the fix is in the
working tree - `grep -c _TEAM_AGNOSTIC_EVENTS data_cleaner.py` - before spending half an hour
on a clean.

The preprocess bundled into the clean writes `data/processed` from whatever the tree says, so
the branch checked out at clean time decides how many game-state keys land on disk. Both the
cleaner fix and workstream 10b are merged now, so one more clean brings data, arrays and code
into agreement.

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
git checkout 522ce8f && python scripts/dump_layer_names.py > /tmp/before.txt
git checkout feature/shared-backbone && python scripts/dump_layer_names.py > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt
python -m pytest tests/ -q
```
`tests/test_model_persistence.py` is the one that matters: layer names must not change, or weight
reload breaks against existing artifacts.

**Result:** Layer-name diff **clean** - every hunk was a `Preprocessed ...` line carrying the
run's temp directory, with no layer line among them. Full suite: 639 passed, 12 failed, the
twelve all being the new name test's own order assertion (see Notes); `test_backbone.py` 16
passed after the fix. Correction G is gone from the failure list - the suite is otherwise green.
encoder/vocabs/ untouched after the run.

**Notes:** The six-way copy was confirmed byte-identical by line-range diff before anything
was touched: from `fusion_concat` through `final_ln` the six `def model()` bodies differ in
**exactly one line**, the argument list handed to `Concatenate`. So `build_backbone` takes
that list as `parts` and owns the concat too, rather than leaving a seventh near-copy behind.

`AddPositionalEmbedding` and `KeyPaddingMask` moved into `models/backbone.py` with the stack -
every remaining use was inside the extracted region, so no head imports them any more. Their
`register_keras_serializable(package="cviq")` key is `cviq>ClassName` and carries no module
path, so models saved before the move still deserialize; `test_full_keras_model_loads` covers it.

**Two things I got wrong, both in the checking apparatus rather than the refactor.** The name
test asserted graph order over *every* backbone layer and failed on all twelve cases at
`attn_pad_mask`: that layer hangs off the `pad_mask` input rather than the running tensor, so
it is a side branch and Keras may place it anywhere after that input in the topological sort.
Order is now asserted over the main tensor path only, with presence asserted separately and the
wiring proved functionally by a probe that a masked-out key cannot influence a later real row.
And `scripts/dump_layer_names.py` let preprocess's summary line onto stdout - that line carries
the temp directory, so the first before/after diff showed eleven spurious hunks. It prints to
stderr now.

The duplicated `def model(self, num_layers=..., num_heads=..., ff_dim=..., dropout=0.2)`
signature is deliberately left alone: it is each head's public API, and unifying it means
editing six signatures plus their `from_artifacts` rebuild paths for no behavioural gain.

Correction G was fixed here as the branch's first commit, so the suite is clean going into
the rest of Phase 3.

### 10a. `feature/local-attention` — §9, the banded heads

**Split from workstream 10.** The spec bundles the local heads and the shot-clock proxy into one
feature, but they are independent: one is architecture, the other a preprocessing input; they
verify against different test files, and only the proxy needs Gate B's output. Splitting keeps a
bad result in one separable from the other, which is how Phase 2 was run.

**Scope.** Two of the eight attention heads in every block restricted to the last eight rows by a
banded mask — the same mechanism as the padding mask — behind a `LOCAL_ATTENTION_HEADS` config
switch. Six heads stay global. No new inputs, no new weights, no custom kernel.

Now a one-place change: `models/backbone.py` owns the mask, so this lands once rather than six
times. That was the whole point of workstream 9.

- `config.py` — `LOCAL_ATTENTION_HEADS` and `LOCAL_ATTENTION_WINDOW`. **Not** rollout dials, so
  **not** in `_TUNING_KEYS` — nothing at sim time reads them and `set_dial` must keep rejecting
  them.
- `models/manifest.py:38` — but they **do** belong in `ARCH_KEYS`. See correction K: the
  local/global split changes no weight shapes, so weights trained with local heads would reload
  into an all-global graph *silently*.
- `LOCAL_ATTENTION_HEADS = 0` must emit today's `KeyPaddingMask` unchanged, so the A/B switch is a
  genuine no-op when off and the existing graph is bit-identical.

**Memory to watch.** A per-head mask is `(B, H, SEQ, SEQ)` bool — ~176 MB at `SEQ=600`, `H=8`,
batch 64. Built once outside the block loop and shared across all six blocks, the same lifetime
today's `attn_mask` already has. The fallback if it bites is two `MultiHeadAttention` layers per
block (6 global + 2 local, concatenated), which changes layer names and so is not a drop-in.

**Verify**
```bash
python -m pytest tests/ -q
```
`test_model_persistence.py` and `test_backbone.py` are the ones that matter: with the switch off,
the graph must be identical to today's.

**Result:** Full suite: 663 passed, 1 failed - the failure my own window test (see Notes);
15 passed on `test_local_attention.py` after the fix. `test_model_persistence` green with the
banded path as the default, no OOM and no shape errors anywhere.

**Notes:** The layer emits the band's **lower edge only** - `(i - j) < window`. The upper edge
would be redundant: `MultiHeadAttention(use_causal_mask=True)` already forbids attending
forward, and skipping it saves a comparison over a (B, H, SEQ, SEQ) tensor. The consequence is
that **the emitted mask is not the window** - at `window=1` the band alone is the whole upper
triangle - and only `band AND causal` is meaningful. My first test asserted the identity
matrix from the band alone and failed; the test now checks the conjunction, with a row-count
assertion (row `r` sees `min(r + 1, window)` keys) rather than a restatement of the band
formula. Behaviour was never wrong.

Both mask paths keep the layer name `attn_pad_mask`, so flipping the switch perturbs no layer
naming and the by-name reload contract holds either way. Neither layer has weights, and the
full `.keras` reload records the class in its config, so a saved model rebuilds the right one.

`LOCAL_ATTENTION_HEADS = 0` rebuilds the pre-2.0 graph exactly - same `KeyPaddingMask`, same
(B, 1, SEQ) shape - which is what makes the post-train A/B a real comparison rather than an
approximation. Pinned by a test.

Correction K acted on: both settings are in `ARCH_KEYS`, neither is in `_TUNING_KEYS`, and a
test asserts each.

Side benefit: with the default at 2 and most tiny-dim tests building at `num_heads=2`, the
existing suite now exercises the banded path throughout rather than only the new test file.

**Not measured, and not measurable from here:** the mask is ~176 MB of bool at `SEQ=600`,
`H=8`, batch 64. One allocation shared by all six blocks, not six, but watch memory on the
first real train.

### 10b. `feature/possession-clock` — §9, the shot-clock proxy

**Scope.** One new per-row number: seconds since the current possession started, scaled to the
24-second clock. Derived inside `GameStateScan` (`models/game_state_features.py:103`) so training
and the simulator compute it identically by construction. `GAME_STATE_KEYS` (`:49`) and `_NORM`
(`:56`) gain the key.

**Everything downstream is free** — confirmed, no edit needed: `simulation/input_cache.py:178`
zips over `GAME_STATE_KEYS`; `make_game_state_inputs` / `game_state_projections` iterate the
tuple; each head's `INPUT_KEYS` splats `*GAME_STATE_INPUT_KEYS`.

**The reset rule reads the cleaned row semantics**, which are the authority now that §7 removed
the `possession` column — no `POSSESSION_FLIP_RESULTS` survives anywhere in the repo. A possession
ends on `result == "cop"` (turnover, defensive or team-defensive rebound, made last free throw, a
foul that flips the ball) and on a made field goal; the clock also resets on an offensive rebound
(the real 14-second reset, offense unchanged) and at a period boundary.

**Blocked on Gate B, to measure only.** The rule reads cleaned rows and the ones on disk are still
v1.0 shaped — steals are two rows there, and the `result` semantics differ. Build it any time;
measure it against the re-clean.

**Consequence.** The fusion concat widens by 16, so `fusion_projection`'s kernel changes shape and
every existing weight file stops loading. Already true after Gate B, but from this merge on there
is no local artifact any smoke run can reload.

**Verify**
```bash
python -m pytest tests/test_game_state_features.py tests/test_game_state_wiring.py tests/test_input_cache.py -q
python -m game_state_features --seasons 2003,2013,2023
python -m pytest tests/ -q
```
The measurement entry point follows the `python -m zones` pattern — a cheap TF-free CPU pass.
**Possessions per game not near ~95–105 means the reset rule is wrong**, and it is the only
independent check this feature has before the train.

**Result:** 26 passed on `test_game_state_features.py`. The pace check **failed on its first
real run and found a genuine error** - 104.0 / 103.2 / 109.4 possessions per team per game
against 93.9 / 94.2 / 99.4 by the box-score formula on the same files, ~12% high in every era.
Fixed on `fix/free-throw-possessions`; now 94.1 / 94.6 / 99.2 against 93.9 / 94.2 / 99.4, gaps
of +0.2 / +0.4 / -0.2, with 2022-23 landing on the published 99.2 exactly. Mean possession
13.0-13.4s, median 13-14s. Full suite still to run as part of the Gate B re-verification.

**Notes:** See correction M for the free-throw decomposition. The short version: the spec's
rule ("reset on a change of possession or an offensive rebound") is right about live play and
silent about free throws, and free throws turned out to be the entire error.

**The gate mattered more than the feature.** The rule passed 26 unit tests and every scenario I
could think to write, and was still 12% wrong. What caught it was one number with an
independent source. The gate itself then had to be fixed too: it first compared against a band
drawn from published pace, which is normalized per 48 minutes and excludes playoffs, so the
band had to be loose enough to hide real errors - 109.4 only just failed it, and 104.0 passed.
It now compares against `FGA - OREB + TOV + 0.44*FTA` on the same file: the same quantity by an
independent route, self-calibrating across eras, tolerance 3.

`GameStateScan` gained a `poss_ends` counter so the check counts the same events the clock
resets on, by construction rather than through a second copy of the rule in the diagnostic.

Three of my test expectations were wrong against correct code during this branch (the period
anchor, the and-1 timing, and the first free-throw case). Each is now pinned by a test that
states the reasoning rather than just the number.

### 11. `feature/rotation-model` — §8, full version, split four ways

**Split into 11a–11d**, on the 10a/10b precedent. §8 is the largest workstream in the programme
and the one the spec flags as possibly needing its own train; four branches means a bad result
stays separable, and 11d — the new head and the scheduler's retirement — can be dropped late
without losing the rest.

| | Branch | What it lands |
|---|---|---|
| 11a | `fix/roster-snapshot-flicker` | the cleaner repair, the shared scan, the measurement gate |
| 11b | `feature/lineup-state` | the three per-player scalars through the roster encoder |
| 11c | `feature/bench-bundle` | the bench set input |
| 11d | `feature/sub-decision-head` | the new head; retire the scheduler, `_fatigue_bias`, `STINT_*` |

### 11a. `fix/roster-snapshot-flicker` — §8, the on-court five

**Scope.** Everything §8 derives comes from the on-court five, and the five turned out to be the
thing the raw data is least reliable about. Measured over 200 games of 2022-23 *before* writing
anything: the per-row `roster_home` / `roster_away` snapshots — which feed the roster encoder,
`build_box_score` minutes and any on-court derivation — **do not agree with the substitution
rows**, in two separate ways.

| | per game |
|---|---|
| substitution rows | 46.5 |
| lineup changes carrying **no** substitution row | ~21 |
| of those, one-row flickers (the snapshot reverts on the next row) | ~10.5 pairs |
| permanent desyncs when folding the substitutions forward | ~1.1 |
| team minutes from the snapshots | **exactly 480.0** (530.0 with one OT) |

Taken naively that is ~21 phantom lineup changes a game against 46.5 real substitutions — a
**~45% corruption of exactly the signal this workstream exists to fix**, going straight into
stint seconds and minutes played.

The flicker is a **raw-data artefact, not a cleaner bug**: in
`RawData/MasterFiles/[10-18-2022]-[06-12-2023]-combined-stats.csv`, game `22200001`, rows 86–91
are `free throw, rebound, sub, sub, sub, free throw` and the **second free throw carries the
pre-substitution `h1..h5`**.

- `data_cleaner._repair_fives` — a whole-file pre-pass, the third alongside
  `_label_shooting_fouls` and `_label_team_rebounds`. Carries a running five, updated in place by
  substitution rows, resynced to the snapshot only when the snapshot is *stable* (the next raw row
  of the same game carries the same two sets) **and** does not contradict a substitution made at
  that same instant. Every path that changes the five goes through one `resync`, which records
  what it took to get there — so the five cannot move without a substitution to explain it.
- Real changes with no substitution row are **emitted as substitution rows**, each carrying its
  own progressive roster.
- `models/rotation_features.py` — `LineupScan`, the shared derivation of stint seconds, seconds
  played and personal fouls, structured like `GameStateScan` and used by both preprocessing and
  the simulator. Fixed-constant normalization, so **correction C does not apply**: no
  `norm_stats_io.py` change, no new `norm_stats.json` keys, nothing to load at inference.
- `shell/actions.py:26` — its `ARCH_KEYS` was a second copy and had drifted (see correction N).

**The independent number** is check 2 of the gate: the on-court five read two ways off the same
file — from each row's snapshot, and by folding the substitutions forward from the opening five.
After the repair those are the same quantity by independent routes, so they must agree exactly.

**Verify**
```bash
python -m pytest tests/test_data_cleaner.py tests/test_rotation_features.py tests/test_chronology.py tests/test_box_score.py tests/test_model_naming.py -q
python -m pytest tests/ -q
python main.py --clean --rebuild-vocabs --model event_time
```
No token changes, so the rebuilt vocabularies must come back byte-identical to the frozen
`0ab3956` set **except for one deletion**: `Nene ` leaves the player vocab (correction O). Check
`git status --short encoder/vocabs/` and `git diff` it — any other change means the repair moved
the token set, which it must not.

Then, TF-free and on CPU, against the new clean:
```bash
python -m models.rotation_features --seasons 2003,2013,2023
python -m models.game_state_features --seasons 2003,2013,2023
python -m zones --seasons 2003,2013,2023
```

**Result:** Verified end to end without a full pipeline run: each of the three era master files
was cleaned in process and the gate run over the result. All four checks pass with margin.

| era | disagree/g | dup rows | short % | min fails | subs/team | top10 min | 10+ min |
|---|---|---|---|---|---|---|---|
| 2002-03 | 0.0000 | 0 | 0.0022 | 0 | 22.0 | 34.2 | 8.5 |
| 2012-13 | 0.0000 | 0 | 0.0025 | 0 | 24.0 | 32.9 | 8.9 |
| 2022-23 | 0.0000 | 0 | 0.0000 | 0 | 27.2 | 32.5 | 8.9 |

The possession-clock gate re-run against the first re-clean is unchanged from workstream 10b —
94.1 / 94.6 / 99.2 against 93.9 / 94.2 / 99.4 — so the repair moved nothing it should not have.
The vocabularies came back byte-identical (no token added or removed); `norm_stats.json` moved as
expected, `delta_mean` 5.930 → 5.839, tracking the ~1.5% extra rows sitting at existing
timestamps. **A second clean is still owed** — the data on disk predates corrections Q.

**Notes:** The first full-season run of the gate is what found correction Q, and it found it by
failing a threshold calibrated on an 86-game slice: 0.0573% short lineups against an allowance of
0.05%. Widening the allowance would have buried a defect touching 12 games. `SHORT_LINEUP_PCT` is
now 0.01, four times the worst era's real rate and no more, and a player in two slots has its own
check and its own message rather than surfacing as a short-lineup rate.

`Nene ` is still in the player vocabulary at token 232, with `Nene` at 2152. `--rebuild-vocabs`
*appends* — the vocabs are append-only, which is why Gate B's procedure deletes the files first.
Every row now encodes to 2152, so 232 is simply a dead embedding row. Not worth an hour to purge
on its own; **Gate C re-establishes the freeze and is the place to do it**, and a wholesale
rebuild before then would destroy the byte-identical check that has been useful twice already.

### 11b. `feature/lineup-state` — §8, the per-player scalars

**Scope.** Three more per-player scalars beside rest — seconds in the current stint, seconds
played, personal fouls — through the shared roster encoder, so every head sees them wherever it
consumes the lineup.

- `RosterEncoderParams.num_scalars`, and **one** `Dense` over a stacked `(B, N, S)` tensor rather
  than one per scalar. Those are the same function, but the kernel's first dimension is then the
  count, so a graph rebuilt with the wrong number fails on shapes. That is why `num_scalars` does
  **not** need an `ARCH_KEYS` entry, unlike `LOCAL_ATTENTION_*` (correction K), which changed no
  shapes. **11c's `BENCH_SIZE` will need one** — set sizes change masks, not weight shapes.
- `num_scalars` is in all four config sites (dataclass, `get_config`, `_params_to_config`,
  `_config_to_params`). It is the only key read with `.get`, defaulting to 1, because a model
  saved before this carries no such key.
- `models/rotation_features.py` gains `merge_rotation_features` / `append_rotation_batches` /
  `make_rotation_inputs`, mirroring the season and game-state pairs. Fixed-constant
  normalization, so there is nothing in `norm_stats.json` and **correction C does not apply**.
- Inference: `HistoryEncoder` gets its own `LineupScan`; the uncached oracle derives over the
  full history and windows it. **Not** read off the controller's `player_seconds` / `player_fouls`
  — one scan driven by both sides is train/inference parity by construction.

**The trap was the roster memo cache** (`simulation/input_cache.py:145`). It is keyed by the five
names alone, which is sound only for game-constant quantities. Rest is one; stint, minutes and
fouls move on every row for the same five, so a cache hit would serve a stale value with no error
anywhere. They are written outside it.

Also fixed here, because this branch adds a second per-game scan and would otherwise have doubled
it: `merge_game_state_features` built one dict per row for the whole frame up front (**8.6 GB**
over 21 seasons, measured) and found each game with `np.where` over the full array (**17ms × 27,415
games = 7.8 minutes**). `iter_game_rows` groups once and keeps one game alive. Output is
bit-identical on a full season. `_build_split` still does the same `np.where` scan across its
three calls, another ~7.8 minutes — **not fixed**, six copies, and a clean follow-up.

**Verify**
```bash
python -m pytest tests/test_model_persistence.py tests/test_input_cache.py tests/test_game_state_wiring.py -q
python -m pytest tests/ -q
```
`test_full_keras_model_loads` is the one that matters — the only test that catches a missed
config round-trip site, and it fails as an opaque Keras shape error rather than naming the key.
`test_input_cache` is the second: it asserts the incremental path and the oracle agree array for
array, which is where the memo-cache trap would surface.

**Result:**

**Notes:**

### 11c. `feature/bench-bundle` — §8, the bench

**Scope.** Up to ten available players per side who are not on the floor, each carrying seconds
since he sat down, seconds played, personal fouls, and whether he has played at all, through a
second set encoder — so "he sat down nine seconds ago" is a learned penalty on the incoming pick
rather than a dial.

**It goes to the SubstitutionModel and nowhere else**, and the reason is cost, not taste. The
player head is called on **every event**, so handing it the bench means a full-history scan per
event — quadratic in game length. The substitution head is asked ~50 times a game, where the same
scan is free. So `BENCH_KEYS` ride on that head's own `INPUT_KEYS`, not on `_BASE_INPUT_KEYS`,
which `stint_length` and `conditional_time` share and which decide no rotation. 11d's
`sub_decision` head is asked only at dead balls and can take the same bundle.

- A **separate** encoder instance (`bench_vec`), not the on-court one reused: `roster_size` is
  baked into `build()`, so one instance cannot serve a five-slot and a ten-slot set, and the
  scalars mean different things — seconds since sitting is not seconds into a stint.
- `config.BENCH_SIZE = 10` is an **`ARCH_KEY`**. A set's size changes mask shapes but no weight
  shape, so a graph rebuilt with another value loads quietly and pools over the wrong number of
  slots — correction K's failure mode exactly, which is why 11a folding the two `ARCH_KEYS` lists
  into one mattered here.
- `Encoder.encode_roster` takes a `size` so one function serves both set widths.

**One asymmetry, named rather than hidden.** Training reads availability as everyone who reaches
the floor over the whole game (`game_available`); the simulator reads it off the full rosters
(`_bench_inputs`, matching `_avail_mask`). A player who never checks in is on the bench at rollout
and absent in training. That is the compromise `game_available_mask` already makes; this follows
it rather than inventing a third answer.

**Verify**
```bash
python -m pytest tests/test_substitution_model.py tests/test_oncourt_mask.py tests/test_model_persistence.py tests/test_model_naming.py -q
python -m pytest tests/ -q
```
`test_substitution_model.py:196` builds the head's inputs straight from `INPUT_KEYS`, so a bench
key produced by neither `_build_split` nor the graph fails there first.

**Result:** Full suite green.

**Notes:** The scope narrowed while building it. The bench was going to the player head as well,
until tracing the call pattern: that head is asked on **every event**, and the bench needs a
full-history scan, so it would have been quadratic in game length. Cost decided the scope, and
the spec agreed with the cheaper answer -- §8 only ever asks for the bench on the incoming pick.

### 11d. `feature/sub-decision-head` — §8, the head

**Scope.** The rotation trigger. At every position Rule 3 permits a substitution, predict how
many each side makes before play resumes (`0 / 1 / 2 / 3+`) — replacing a timer that sampled a
stint length per entering player and pulled him when the clock reached it.

- `models/sub_decision_model.py`: two softmaxes on one backbone, so both sides come from a
  single forward pass and the rollout pays one extra head call rather than two. It takes the
  bench bundle and, deliberately, **no next-step conditioning** — the question is about the
  position, not about an event already decided.
- **It does not augment in the opening lineup.** `SubstitutionModel` synthesises ten opening
  substitutions so the incoming-pick head learns starters; a tip-off is not a stoppage, and
  feeding them here would teach it that games open with five substitutions a side.
- Retired: `models/stint_length_model.py`, `_schedule_stint`, `_process_scheduled_subs`,
  `_fatigue_bias`, and four dials — `SUB_FATIGUE_WEIGHT`, `STINT_SAMPLE_SIGMA`,
  `STINT_LENGTH_SCALE`, `STINT_MAX_SECONDS` — with tombstones and a guard test, on the
  `DEADBALL_REBOUND_PROB` precedent. `_maybe_force_sub` stays as the cadence backstop.

**Nine registration sites**, and missing any one is silent rather than loud: both registry
lists, `run_all` **and** `run_stage` (only `run_stage` is on `train.py`'s path),
`LARGE_OUTPUT_MODELS`, `REQUIRED_HEADS`, the controller's required set, a `predict_sub_count`
on `GameSimulator`, and a `ModelTestAdapter` — without which the head gets zero save/load
coverage.

**Two dials worth remembering.** `SUB_FATIGUE_WEIGHT` was a hand-written stand-in for exactly
what the roster encoder now reads directly, since every head sees stint seconds, minutes and
fouls per player. `STINT_LENGTH_SCALE` existed because the head regressed LOG stint, so its
point estimate was a geometric mean under-predicting a right-skewed duration by ~30%, and the
dial multiplied the gap away. **A dial correcting a distributional artefact of the target is a
sign the target is wrong**, and it was: the question was never how long a player stays on.

**Verify**
```bash
python -m pytest tests/ -q
```
`test_model_persistence` bites first (a real one-epoch train plus a full `.keras` reload of the
two-output graph); `test_backbone` catches a head that builds at the wrong width;
`test_dials` iterates `_TUNING_KEYS` live, so the four removals have to be clean.

**Result:** Full suite green, on the third run. The two before it are the Notes below.

**Notes:** Two rounds of failures, and both were the same mistake rather than anything about
rotation.

The first round was 90 failures, 87 of them one thing: `tests/test_controller.py` kept its own
literal copy of the head list its `FakeSim` provides, so adding `sub_decision` to the shell's
copy and the controller's left the third stale and every test that builds a controller died.
That list existed **four** times, counting `test_shell.py`'s `ALL_HEADS`, which still named
`stint_length`. It lives in `config` now — the only module the shell, the controller and the
tests can all import without pulling in TensorFlow.

The second round was six, of which four came from writing something fresh instead of from the
copy that already worked: `_build_split` called `np.stack` on an empty split where
`EventTimeModel._build_split` already guards it, and `model()` read `MODEL_DIM` where every
other head reads `self.model_dim`. The remaining two were test fixtures scripting one half of
what `sample_substitution` pops, and setting a bench before a `start()` stub that silently
resets it.

Recorded because correction N in this same branch is the identical lesson — two `ARCH_KEYS`
lists that drifted — and it was reproduced twice more within days of writing it down.

### 12. `feature/training-changes` — §10, minus the clutch weighting

**Scope changed on 2026-09-09. Read this, not §10.** The spec asks for two things: clutch loss
weighting, and masking the loss to the rows the simulator actually queries. **Only the second is
being built.** The first was started and deliberately dropped — see below.

#### Not building: clutch loss weighting

§10 proposes that rows which are close and late (`period_idx >= 3`, `period_time_left <= 300`,
`abs(score_diff) <= 8`) count double in every head's loss. Alec rejected it, and the reasoning
holds:

- **The features are already inputs.** `period_idx`, `period_time_left` and `score_diff` have been
  plumbed since the game-state work, and 2.0 is simply the first train whose weights consume them.
  Weighting is only justified if the model *under-fits* end-game despite having them — and no
  train has ever consumed them, so there is **no evidence either way**. Turning it on is a guess.
- **It biases the metric we report.** The programme is scored on box-score accuracy over whole
  games. A Q1 rebound counts toward a player's total exactly as much as a Q4 one, and the
  overwhelming majority of every box score comes from non-clutch rows. Doubling a thin slice buys
  accuracy there by spending it everywhere else.
- **§10 half-concedes this** — it says to watch the per-quarter splits for early-game drift, and
  that a Q1 regression means the weight is too high. A knob that anticipates its own harm should
  not ship on.

If the post-train per-quarter splits (workstream 13) show end-game genuinely mis-modelled — no
intentional fouling when trailing, no three-point hunting — that is the evidence, and the
mechanism can be built then. It is ~30 lines and the A/B would be two trains against the **same**
preprocess, so nothing is lost by waiting for a reason.

#### Building: the loss mask

The event and time heads stop training on rows the simulator never asks about.

**Already done**, in `EventTimeModel._make_dataset`: rows whose next event is a substitution are
already zero-weighted, because rotation is injected by the controller and never sampled from the
event stream.

**Still to do:**

- **Continuation rows.** The controller expands one sampled event into several emitted rows, and
  the event head is never asked "what next" at the intermediate ones. From the `_append` calls in
  `simulation/controller.py`, the expansions are exactly four:

  | first row | continuation | emitted by |
  |---|---|---|
  | `assist` | `shot` | `_do_assist` |
  | `shot` (blocked) | `block` | `_do_shot` |
  | `foul` | `shot`/`free throw` | `_do_foul` / `_do_shooting_foul` -> `_free_throws` |
  | `shot`/`free throw` | the next attempt of the same trip | `_free_throws` |

  §10 requires this be **one function shared with the controller's play expansion** so the two
  cannot drift. The controller's expansion is imperative, so the practical form is: the rule lives
  as a predicate beside the other cleaned-row semantics (`models/game_state_features.py`, next to
  `possession_boundary`), preprocessing builds the mask from it, and a test walks the controller's
  own emitted rows asserting every within-play adjacent pair IS a continuation by that predicate
  and every across-play pair is NOT.
- **For the time head only:** the last row before a period break.
- Only `EventTimeModel` needs the new array, which keeps the plumbing to one head.

`tests/test_game_state_wiring.py` covers event_time, conditional_type, conditional_time and
sub_decision but **not** `player` or `substitution`. Close that gap here.

**A candidate for 12's independent number, already measured.** `docs/technical_specs.md` records
M4 as "Play-boundary loss masking — **21.9% of event-head training positions never occur at
inference**". That figure predates 2.0 and predates the substitution mask that is already in
`EventTimeModel._make_dataset`, so it is not the answer — but re-deriving it against the current
cleaned data gives the branch a number to move: how many rows the event head trains on that the
controller never asks about, before and after. A masking change with no such number is unfalsifiable.

**Scope grew by one head, and by one bug found on the way in.**

`ConditionalTimeModel` needs the mask too. The plan above said only `EventTimeModel` did; that
was wrong. `_advance_for` (`simulation/controller.py`) routes through `predict_delta` whenever
the conditional head is loaded, so it is the sim's actual clock — and it is called **once per
sampled play**, never at a continuation row: the block, the assisted shot and every free throw
are appended with no clock advance at all. It is therefore asked about exactly the positions the
event head is asked about. Leaving it out would have left the artifact in the head that sets
pace. Its `_make_dataset` already carried the twin substitution mask, which is the same argument
applied to substitutions.

**The free-throw trip had a hole, and finding it moved the pace gate.** See correction T. The
short version: the continuation rule needs to know whether a trip is open, `GameStateScan`
already tracked one, and two notions of the same thing in one class is corrections N and R for
the third time. Unifying them exposed that a substitution or timeout inside a trip closed it
early — ~7,500 trips a season in every era. The fix drops the derived possession count by ~2.1
per team per game, and the pace gate then failed at -3.1 to -3.7 against a tolerance of 3.0. The
tolerance was **not** touched; the reference was corrected instead, because `0.44 * FTA` charges
a fraction of a possession to and-1s and retaining-foul trips that cannot end one.

**The independent number, measured before and after.** Route A is the predicate over cleaned
rows; route B is the controller's own `_append` ledger, asserted to agree exactly in
`tests/test_controller.py`. Route A, from the production scan:

```
  season   positions      sub   contin.   before    after
    2003     625,528   56,102   130,175     9.0%    29.8%
    2013     639,521   63,197   129,619     9.9%    30.2%
    2023     674,703   71,840   140,554    10.6%    31.5%
```

`docs/technical_specs.md` records M4 as 21.9%, which is neither the before nor the after — it
predates the substitution mask and is close to the *new* half by coincidence. The masked share
is stable to about a point across twenty years.

**Why the free-throw arm is a trip question and not a row-pair one.** Two measured reasons,
either alone fatal to the pairwise form the plan above proposed:

- The foul row does not say whether free throws followed. `determine_foul_result` writes
  `nothing` on a common foul and the **bonus is the controller's decision** (`_foul_outcome` ->
  `_in_bonus`). In 2022-23, 3,375 `personal`/`nothing` fouls are followed directly by a free
  throw, plus 714 `loose ball`/`op` and 100 `away from play`/`nothing`.
- Substitutions and timeouts sit inside the trip, at depths up to six or more. About one free
  throw in five is separated from its foul that way, and the position AT the interposed row is
  one the controller never queries either.

Together those are 13-16% of the whole mask — 17,000 to 21,000 positions a season that a
pairwise predicate keyed on the foul's result token drops silently.

**The mask is A/B-able without a re-preprocess.** It is written into the npz unconditionally and
`config.MASK_CONTINUATION_ROWS` switches whether the dataset applies it. That restores the second
cheap ablation Gate C had lost, alongside `LOCAL_ATTENTION_HEADS = 0`.

**Verify**
```bash
python -m models.game_state_features --seasons 2003,2013,2023
python -m pytest tests/ -q
```
The first is TF-free and was run here — gate green, table above. The second is the handover: the
new controller-parity tests and the two new wiring tests have never been executed, because both
import TensorFlow.

**Then the clean, which is the one owed since correction Q:**
```bash
rm encoder/vocabs/*.json
python main.py --clean --rebuild-vocabs --model event_time
```
`rm` first is deliberate and is the **one** intended vocabulary change: `--rebuild-vocabs`
appends rather than rebuilds, so it is the only way to drop the dead `Nene ` token (correction
P). It renumbers every player id above 232, which is free now and stops being free the moment
train 3 finishes. Gate C's "byte-identical" check re-baselines against this clean.

**Result:** **739 passed** (2026-09-09), against 719 before this branch. That delta is the check
that matters: 20 tests were added and 20 appeared, so the new files collected — a green run where
they had silently not collected would have looked identical. The clean and the vocabulary purge
ran in the same session; everything below was verified here, TF-free, against the artifacts they
produced rather than taken on trust.

**Notes:**

**The vocabulary purge landed exactly as intended.** `Nene ` is gone; the real `Nene` now sits at
id 232 (it took the freed slot on the rebuild); `next_token` went 2153 -> 2152, exactly one token
removed. `event`, `type`, `result` and `season` vocabularies are byte-identical to the freeze —
they do not even appear in `git status` — so nothing else moved. This is the new baseline for
Gate C's byte-identical check.

**`norm_stats.json` is a legitimate refit, not test residue.** It is timestamped twenty minutes
after the vocabularies, which is exactly the signature the Gate C checklist warns about. It is not
pollution: `train.npz` / `test.npz` / `holdout.npz` / `event_time_norm_stats.json` all carry the
same 16:59 stamp, i.e. the event_time preprocess finishing its pass over 21 seasons, and the four
scalars moved only in the seventh significant figure (`delta_mean` 5.8389745 -> 5.8389098),
consistent with correction Q's few hundred repaired rows. Test residue from the two-game synthetic
fixture would not be corpus-scale. **The timestamp alone does not settle this — check that the
npz share the stamp and that the values are corpus-scale.**

**The mask arrays verified end-to-end at corpus scale**, straight out of the real npz:

| | train.npz | test.npz |
|---|---|---|
| games | 18,878 | 5,394 |
| positions | 9,355,829 | 2,676,448 |
| continuation | 1,919,475 (20.52%) | 549,212 (20.52%) |
| period break | 76,662 (0.82%) | 21,923 (0.82%) |
| event head trains on | 79.48% | 79.48% |
| time head trains on | 78.66% | 78.66% |

Both arrays are 0/1 only and a strict subset of `loss_mask`. Train and test agreeing to two
decimals is the sign the rule is reading the data and not an artifact of one split. The 20.52%
here against ~20.6% averaged over the three sampled eras is the season mix, not a discrepancy —
the npz covers all 21 seasons.

**The other five heads' npz are two months stale (2026-07-05) and that is harmless.**
`--model event_time` preprocesses only its own head, so `cond_*`, `condtime_*`, `player_*`,
`sub_*` and `stint_*` still predate Gate B. `models/pipeline.run_stage` calls `preprocess()`
unconditionally per head (gated only on the resume list, empty on a fresh train), so Gate C's
`train.py --full` rebuilds all of them from the cleaned data — including
`ConditionalTimeModel`'s, which is where the second copy of the mask lands. **Worth knowing:
`condtime_train.npz` does not carry the mask today**, and cannot until that train, so the only
evidence for that head is the wiring test on synthetic rows. `stint_*` is a leftover from the head
retired in 11d.

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

**Why it matters more, not less, now that clutch weighting is dropped.** §11 motivated this as a
way to watch for clutch-induced Q1 drift, and that reason is gone. The real one is stronger:
per-quarter splits are the only way to see whether the model gets end-game basketball right at
all — and that measurement is exactly the evidence that would justify revisiting the weighting
later (correction S). Without it, "is end-game mis-modelled?" stays unanswerable. The per-zone
shot-mix half was never about clutch either.

**Independent of everything else** — it can be pulled forward at any time at no cost.

**The verify below does not work, and the reason is worth keeping.** `full4-s100` is a v1.0-era
run: its archived play-by-play carries `2pt` / `3pt`, and `zones.points_for_shot` refuses any type
that is not a zone or `free throw` (standing guard 1). Any 2.0 code that rebuilds a box score from
those rows aborts. So the archived run cannot be the reference, and the cleaned corpus is used
instead — which is the better target anyway: ~26,000 real games in current tokens, no weights, no
TensorFlow.

**The independent number: per-period boxes must sum to the whole-game box.** Stat for stat, player
for player, over every counting field and minutes. Two routes to one quantity — one scan over all
rows, and a scan per period recombined. **1,500 games across three eras, zero mismatches**, with
period counts that are real basketball (463 / 34 / 3 regulation / 1OT / 2OT in 2002-03).

It found three defects, none of which a unit test would have proposed:

- **A phantom fifth period in every regulation game.** `period_index` is half-open, so a clock
  exactly on a boundary opens the next period. That is right for the game STATE (at 2880 there
  really are 300 seconds of a new period) and wrong for an EVENT: a shot at 0.0 is a buzzer-beater
  belonging to the quarter it ended, and the final shot, its block and the `end` sentinel all sit
  at exactly 2880. A zero-duration tail sitting exactly on its own period start folds back.
- **Minutes leaking at every break.** A slice beginning mid-game has real elapsed time before its
  first event, credited to nobody without a seed. Worse: a player substituted off AT the break was
  credited those seconds by `line()` and then dropped from the box, because the players sets are
  built only from rows in the slice — so his minutes left the total altogether.
- **Side membership derived per period.** A player who records a stat in a period he was not on the
  floor for resolves to no side, his line is never emitted, and his stats vanish from that period's
  team total. One game in 1500 — Brent Barry fouling and turning it over at the 2002-03 buzzer while
  off the floor. Which side a player is on is a fact about the game, so it is computed once.

**Where the code went.** `split_by_period` / `period_box_scores` / `side_membership` in
`simulation/box_score.py`, next to the thing they slice, importing the period rule from
`models.game_state_features` rather than restating it. **Known duplicate, deliberately left:**
`GameController._period_index` is a third copy of that arithmetic. It reads `self.clock` rather
than a parameter and sits on the rollout's hot path (a call per event against inline arithmetic,
with millions of events per run), so unifying it is a perf question, not a tidiness one. Recorded
here so it is not rediscovered.

`build_game_record` gained `period_boxes`, and `_PbpSink` computes the split as each sim lands
rather than keeping histories alive — the 8.6 GB record spike the lineup-state branch removed is
not worth reintroducing for a table of team totals. **The actual game's split is always derived**,
because `game_df` is always to hand, so a record carries the real per-quarter box even when the
predicted one is unavailable.

**The per-zone shot mix** is in `simulation/diagnostics.compare_holdout`, reporting SHARE of
attempts per zone rather than counts: a pace difference must not read as a shot-selection
difference, and the report already measures pace. Cross-checked against a route sharing no code
with it — `python -m zones` derives the era table from raw coordinates, this derives it from
cleaned tokens, and on 2022-23 they agree (corner3_l 5.2% vs 5.1% at 38.5% both ways, wing3_l 9.6%
vs 9.8% at 35.8% both ways, 3PA share 38.6% vs 38.8%; the residual is 200 games against a season).

**Verify**
```bash
python -m pytest tests/test_box_score.py tests/test_evaluation.py tests/test_diagnostics.py -q
python -m pytest tests/ -q
```
The suite was 739 before this branch and gains 25 (17 box-score slicing, 8 shot-mix), so **764 is
the number that says they collected**. The slicing half was additionally run here against 1,500
real games; the report and diagnostics halves are pytest-only, because `reporting/eval_report.py`
imports `report_artifacts`, which imports Keras.

**Result:** Suite green (2026-09-09), reported by Alec. The slicing half was additionally run
here against 1,500 real cleaned games across three eras with zero mismatches; the report and
diagnostics halves are pytest-only, because `reporting/eval_report.py` imports `report_artifacts`,
which imports Keras.

**Notes:** `reporting/eval_report.py` is the one CRLF file this branch touched, and the patch
flipped it to LF -- reading in text mode normalizes CRLF, and writing back with `newline=""` then
persists it. Caught from the diff stat (789 deletions for ~100 added lines), not from a test,
because nothing in the suite looks at line endings. Restored. Worth watching on any future patch
to that file.

### Gate C — pre-train

**Pre-train checklist run 2026-09-09. Every item that does not need TensorFlow is green.**

| check | result |
|---|---|
| `pytest tests/` | green (reported; 739 before workstream 13, +25 from it) |
| committed encoder artifacts | clean, and `norm_stats.json` equals `data/processed/event_time_norm_stats.json` bit for bit |
| vocabulary purge | `next_token` 2152, `Nene` at 232, `Nene ` absent |
| zone table vs §3, three eras | all gates passed |
| derived-vs-raw 3pt disagreement | 0.39% / 0.23% / 0.11% — under 1% in every era |
| coordinate coverage | 100% / 100% / 99.4% |
| pace gate | -1.7 / -1.6 / -1.9 against the de-biased reference (correction T), tolerance 3.0 |
| loss-mask share | 9.0% -> 29.8% / 9.9% -> 30.1% / 10.6% -> 31.5% |
| branch provenance on the pod | **added after train 3** — `git log --oneline -1` is the tip, and grep finds `timeout_team` |
| subset regime | **added after train 3** — auto-extracted by `full_run`; the banner reads a few thousand games, not 21,014 |

Reproduced by two commands, both TF-free and both cheap:

```bash
python -m zones --seasons 2003,2013,2023
python -m models.game_state_features --seasons 2003,2013,2023
```

**The norm-stats check is stronger than "the file is unmodified."** The committed value equals what
the preprocess wrote, which is what distinguishes a real freeze from pytest residue — an unmodified
file only says nothing has touched it since the last commit, not that the last commit held a real
clean. Compare the two files, not the git status.

**The train command drops `--clean --rebuild-vocabs`.** Workstream 12's clean already ran, with the
purge, and `main.py --clean` enriches as `train.py --full --clean` does — the cleaned data carries
the season-context columns. Re-cleaning would reproduce the same bytes and re-appending to the
vocabulary would reproduce the same ids, so both are ~40 minutes of failure surface for no gain
before a long train. Without `--rebuild-vocabs`, `run_stage` loads and freezes the committed
vocabulary, which is the path the other five heads already take.

```bash
nohup python train.py --full --name <name> --batch-size 64 > /workspace/train.log 2>&1 &
tail -n 200 /workspace/train.log | tr '\r' '\n' | tail -30
```

Use `--clean --rebuild-vocabs` only if `data/` or `encoder/vocabs/` has been touched since
2026-09-09; then the vocabularies must come back byte-identical to the purged freeze.

**Two things to read before the first head, both consequences of correction W.** Neither is
optional and both are cheap:

- **Provenance, before the card is spent.** `git log --oneline -1` must show the branch tip you
  expect, and `grep -rl timeout_team models/` must return files. Train 3 ran pre-2.0 code for
  twelve hours because the branch had never been pushed and nothing checked.
- **The subset banner, in the log's first screen.** `full_run` extracts the manifest itself now,
  so the banner is always printed: the small heads on a few thousand games, `event_time` /
  `player` / `substitution` / `sub_decision` on the full corpus. If those counts look like the
  whole corpus, stop the train rather than pay 5.4x for it. `python -m training.subset show`
  prints the same summary afterwards.

The train runs under `nohup` into a log file on the volume. It is not that the process needs
protecting — it survives a dropped SSH session perfectly well — but that its *output* does
not, and with no `SYS_PTRACE` in the container there is no recovering it from the orphan.

Train 2's availability masking and capacity settings carry forward unchanged. `run_stage`
preprocesses every head unconditionally on a fresh train, so the five heads whose npz still predate
Gate B are rebuilt — that is where `ConditionalTimeModel` gets its copy of the loss mask, which has
no corpus-scale evidence before this train.

**Post-train:** per-zone make rates against the era table; the per-quarter section read for what
end-game behaviour the model actually produces (does a trailing team foul, does it hunt threes);
then the dial package fitted from zero, re-keyed per zone.

The clutch A/B that used to sit here is gone with the weighting (correction S), but workstream 12
put a second one back. **Two cheap ablations, both against the same preprocess:**

- `LOCAL_ATTENTION_HEADS = 0` — rebuilds the pre-2.0 graph exactly.
- `MASK_CONTINUATION_ROWS = False` — trains the event and time heads on all positions again. The
  mask arrays ship in the npz either way, so this costs a train and no re-clean.

Worth knowing before reading an ambiguous result: eleven workstreams land in this train and their
effects confound. These two are the only knobs that separate cleanly.

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

**B. §10's `_make_dataset` line numbers are stale, and went stale a second time.** They were
already off by up to 12 lines when the spec was written; the backbone extraction (workstream 9)
then removed ~24 lines from each of the six model files and moved every one of them again. The
§12 table carries the values measured on 2026-09-08. **Treat any line number in either document
as a hint and grep for the symbol** — this is the second time these particular ones drifted.

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
2.0 work - it fails identically on `main`. It does block Gate C's "pytest tests/ green", so it was
fixed on `feature/shared-backbone` (commit 5674516): `monkeypatch.chdir(tmp_path)` isolates both
CWD-relative fallbacks at once - the train state and `./results/<model>/`. The fallback chain
itself is deliberate (a machine that only loads someone else's weights has no train state), so
the fix belongs in the test, not the product. **Closed.**

**H. The spec's fold is wrong for exactly the shots `heave` exists to catch.** Section 3 decides
which basket is being attacked by which half the shot came from (`end_a = y < 47`). That is right
for every normal attempt and backwards for a genuine backcourt heave, launched from the shooter's
own end: 2022-23 has real rows like `shot_distance=65` at `(20.3, 24.4)`, which the half-court
rule folds to the *near* hoop at 19.7 ft and drops into `top3`. A 5-15% prayer would then sit
inside a real zone dragging its make rate down - precisely what the separate token is meant to
prevent, so `heave` would have been unreachable for the shots that need it most. `zones.fold()`
picks the basket by whichever hoop `shot_distance` agrees with, falling back to the half-court
rule when that column is missing. Impact: the basket changes for 0.193% of 2022-23 coordinate
rows and the zone for 161 of them. This is why the measured `heave` volumes run slightly above
the spec's 0.3-0.4%.

**I. And-1s need a labelling rule section 4 does not give.** The spec says to label each shooting
foul from the following trip's `outof`, but `outof == 1` - an and-1 - matches neither
`shooting 2pt` nor `shooting 3pt`, and it is 24% of all shooting fouls, far too many to drop or
default. `data_cleaner._label_shooting_fouls` labels them from the point value of the made
basket they follow (98.6% have one within five rows). The controller still overrides the count
structurally - an and-1 is one attempt whatever the token says - so the token is only ever read
as 'the fouled attempt was a 2 or a 3', which is exactly what it means in every other case.

**J. A bare team rebound's side IS recoverable - but most of those rows are not rebounds.**
I first claimed the side could not be inferred. That was wrong: the heuristic was bad, not the
data. Counting "the next event that names a team" includes fouls, which are usually committed by
the team WITHOUT the ball, so it inverted the answer and gave an implausible 65/12 split.
Restricting the lookahead to events that actually indicate possession (shot, free throw,
turnover) is **99.8% accurate**, validated against the 11,884 playerless rebounds whose side is
recorded, in the same structural position. The raw `possession` column is no help - it is a
jump-ball arrow, blank on 99.6% of rows and on all 9,374 of these.

The population, though, is mostly not boards. Of 9,374 bare rows in 2022-23: **6,336** follow a
missed free throw that was not the last of its trip (6,078 are literally "missed 1 of 2") - the
ball is dead and the shooter shoots again; **13** follow a made free throw; **2,116** are
end-of-period boards with no following possession, genuinely undecidable and already excluded by
section 7's "period-end rows are not kept"; leaving **909** real, decidable team rebounds, which
are now recovered. Emitting the 6,336 would have injected phantom boards into the head whose
entire job is the offensive/defensive split.

**K. §9 calls the local-attention switch "a config switch" without saying it is an architecture
key.** Restricting two heads to a banded mask changes no weight shapes, so a model trained with
local heads reloads into an all-global graph **silently** — no error, no shape mismatch, just
quietly wrong attention in every rollout. `models/manifest.py:38 ARCH_KEYS` is the existing
mechanism for exactly this ("a mismatch means the weights will not load") and does not cover it
today, so `LOCAL_ATTENTION_HEADS` and `LOCAL_ATTENTION_WINDOW` go in it. They stay out of
`_TUNING_KEYS`: nothing at sim time reads them, and they are not A/B-able without a retrain.

**L. A jump ball's `team` column is the team that WON THE TIP, not the team of the player it
names.** `_update_teams` bound each side's abbreviation from the first row whose actor sat in
that side's five, taking the row's `team` as that side's id. The two jumpers are opponents by
definition, so about half the time a jump ball pairs an away player with the home abbreviation -
and it is the first action row of nearly every game, so it bound first. In game 5084 the jump
ball names Joel Embiid (PHI) with `team=BOS`, setting `away_team=BOS`; Marcus Smart's shot then
set `home_team=BOS` too.

Both sides collapsed onto one abbreviation in **~47% of games in every era** - 615/1320 in
2022-23, 572/1277 in 2002-03, 638/1314 in 2012-13. Two consequences, the second worse than the
first: every timeout by the unbound team was dropped, and every timeout that survived in a
collapsed game was labelled `home`, because `_side_of_team` tests home first. **3,391 of the
11,134 kept in 2022-23 were mislabelled by construction**, about half of them wrongly - and that
is the exact field the new `timeout_team` head trains on. In the 705 uncollapsed games the split
is 3,944 away / 3,799 home, the 51/49 a timeout should be. The `home_team` / `away_team` context
columns carried the same collapse.

Fix: jump balls do not bind, plus a guard that the two sides cannot share an abbreviation.
Validated through the real `DataCleaner` methods over three eras: 0 collapsed, 0 unresolved, and
the bound home abbreviation matches the majority abbreviation over shot rows by home players in
all 3,911 games. Timeouts kept now equals timeouts attributable - 14,477 of 14,477 in 2022-23;
the ~2,400 still dropped in the older files carry `team=nan` with raw type `unknown` and have no
side to attribute.

`_label_team_rebounds` compares raw `team` values to each other rather than to the bound
abbreviations, so correction J's work is unaffected - which is why its counts matched exactly.

**Why 663 green tests missed it:** every existing timeout test bound from shot rows, so nothing
in the suite reached the jump-ball path. This is the case for Gate B's step 4 existing at all -
no unit test was going to find it, and a spot-check against a known real-world quantity did.

**M. §9's possession-clock rule is silent about free throws, and free throws were the whole
error.** The spec says the clock resets "on a change of possession or an offensive rebound",
which is right for live play. Reading a made free throw as a change of possession - the obvious
reading, since the cleaner normalizes free throws under `shot` - over-counted possessions by
12% in every era. Three causes, each measured over the 2022-23 file, together 4.38 per team per
game against a 4.3 residual:

| | per team per game |
|---|---|
| a two-shot trip ended on each made attempt, counting twice | ~5.7 |
| an and-1 ended it again, after the made basket that drew the foul already had | 3.52 |
| a technical, flagrant or take foul ended it at all, when the shooting team keeps the ball | 0.86 |

None of the three is visible in a single row, so `possession_boundary` no longer decides free
throws. `GameStateScan` carries the open trip: the foul row records whether it retains
possession and whether it followed a made basket, each made attempt records its time, and the
trip resolves on the first row that is not one of its own free throws. That needs no
`num`/`outof` index - which the cleaned data does not carry, and which §4 deferred to v3 - and
it dates the next possession from the LAST made attempt rather than the first.

**N. Correction K landed in the manifest and not in the check.** `ARCH_KEYS` existed twice —
`models/manifest.py:43` and `shell/actions.py:26` — and workstream 10a added
`LOCAL_ATTENTION_HEADS` / `LOCAL_ATTENTION_WINDOW` only to the first. The manifest therefore
*recorded* both settings and LOAD *compared* neither, so the silent local/global reload
correction K exists to prevent was still fully available. `shell/actions.py` now imports the one
list; a test pins that. Worth remembering for 11c, which adds `BENCH_SIZE` to it.

**O. `"null"` reaches no vocabulary, in any column.** The cleaner writes `"null"` as the sentinel
for a missing player, result or type at about ten sites. `data_loading.py:53` reads the cleaned
CSVs with `pd.read_csv(p)` and pandas' default NA list contains `"null"`, so every one of them
becomes NaN before a vocabulary is built. Checked against the frozen 2.0 vocabularies: `"null"`
is in none of the five, while `"none"` is in all of them. The substitution path is switched to
`"none"` on 11a, because there it decides who is on the floor; the other sites are left, the same
defect with a wider blast radius and nothing depending on them structurally. **Not yet fixed.**

**P. 2002-03 spells one player two ways, and the vocabulary carries both.** The raw `entered` /
`left` columns give `"Nene "` where `h1..h5` give `"Nene"`. The two never matched, so folding the
substitutions desynced for the rest of every Denver game — and the frozen 2.0 `player_vocab.json`
holds `"Nene"` and `"Nene "` as two players with two embeddings. `parse_file` now trims every
player-valued raw column once, on read, before the pre-passes and the row loop; a dozen call
sites each stripping their own would drift. One token leaves the vocabulary at the next clean.

**Q. A substitution row can name the wrong incoming player, and applying it grows the five to
six.** 2002-03 has rows like "Gerald Wallace out, Jim Jackson in" where Jim Jackson is already on
the floor; the lineup snapshot shows the real arrival was Doug Christie. `_repair_fives` checked
only that the *outgoing* player was on the floor, so it applied the row and put one player in two
slots. Every comparison downstream is by membership, so the five then grew to six and never
recovered — 12 games in 2002-03, one of them for 148 rows, 359 short/duplicated rows out of 9 bad
rows in the source. A substitution is now applicable only if the outgoing player is on the floor
**and** the incoming one is not; otherwise the pairing is refused, the transition is recovered
from the lineup, and the raw row is dropped.

The refusal has to be a **flag**, not something inferred from whether the recovery emitted
anything: where the lineup never moves the recovery emits nothing, and reading it that way let
the contradictory row out. That version measured 0.088 disagreeing rows a game in 2002-03 — a
*pass* against the 0.1 tolerance, and worth nothing. The rule it encodes is the one the whole
branch rests on: **the lineup is the authority, and a substitution the five does not corroborate
is not emitted.**

**R. The same list existed four times, and adding to it broke 87 tests.** The heads
`GameController` requires were a literal in `shell/actions.py`, another in the controller's own
constructor, a third in `tests/test_controller.py`'s FakeSim and a fourth in
`tests/test_shell.py`'s `ALL_HEADS`. Registering `sub_decision` in two of them left the other
two stale, and every test that builds a controller died on a missing head — a failure with
nothing to do with the change that caused it. It lives in `config.REQUIRED_HEADS` now, the only
module all four can import without TensorFlow.

This is correction N a second time (two `ARCH_KEYS` lists that drifted), and it was reproduced
within the same branch that recorded N. **When a list is read in more than one place, put it
somewhere both can import before adding to it, not after.**

**S. Clutch loss weighting was rejected, and the reasoning generalises.** §10 asks that
close-and-late rows count double in every head's loss. The features it keys on are already model
inputs, and 2.0 is the first train that consumes them, so there is no evidence the model
under-fits end-game — and the programme is scored on box-score accuracy over whole games,
where the overwhelming majority of every box score comes from non-clutch rows. Weighting a thin
slice buys accuracy there by spending it on the reported metric.

The general form is worth keeping: **a knob whose own documentation tells you to watch for the
harm it causes should not ship on.** §10 says to watch the per-quarter splits for early-game
drift and that a Q1 regression means the weight is too high. Build the measurement first, then
decide. Nothing is lost by waiting — the mechanism is ~30 lines and the A/B is two trains
against the same preprocess.

**T. A dead-ball row inside a free-throw trip closed it early, and the pace gate hid it by
cancelling against a biased reference.** Found while unifying the trip state for workstream 12's
continuation rule, which needs the same "is a trip open" question correction M's possession clock
already answered.

`GameStateScan.step` resolved the trip on **any** non-free-throw row. But the trip is
whistle-to-whistle: the ball is dead for its whole length, which is exactly when the rotation
scheduler may substitute and when a coach may call timeout. Measured over the cleaned corpus:
7,474 trips in 2022-23 carry a substitution or timeout strictly between two attempts, ~7,500 a
season in every era, and **99.5% have the same shooter on both sides of the gap** — one trip, not
two.

The cost was worse than the split trip counting twice. Resolving early also **cleared**
`ft_after_basket` and `ft_retains`, so both exclusions correction M was written to remove came
back whenever a dead-ball row landed mid-trip. Six of six sampled divergences were and-1s or take
fouls counted a second time. Fixing it drops the derived count by ~2.1 possessions per team per
game.

**And that made the gate fail: -3.1 / -3.1 / -3.7 against `PACE_TOLERANCE = 3.0`.** The tolerance
was not touched — that is the "a loose gate is close to no gate" lesson pointing the wrong way.
The reference was the biased side. `FGA - OREB + TOV + 0.44 * FTA` charges 0.44 of a possession
to *every* free throw, including the two families that provably cannot end one: and-1s, where the
made basket already ended it, and retaining fouls, where the shooting team keeps the ball. Those
are ~3.4 and ~1.3 FTA per team per game, about 1.5 possessions of pure reference bias.
`_non_ending_fta` removes them from **raw tokens alone**, with no possession rule involved, so
independence is preserved.

Rejected: detecting an and-1 as "a made field goal preceded the foul". Most defensive fouls
follow somebody's made basket, so that measured 9.3 and-1s per team per game against a true 2.0.
The free-throw shooter *being* the scorer is the only unambiguous raw signal.

Result: the old +0.2 / +0.4 / -0.2 agreement was two errors cancelling. Both sides now measure
the same quantity — -1.7 / -1.6 / -1.9, stable across twenty years, gate green with the tolerance
unchanged. **The residual is a real, unexplained ~1.8 and it is worth its own look**: it did not
exist as a visible quantity before, because the double-count was filling it.

The general form, and it is the third time: **when two numbers agree, check they are measuring
the same thing before believing them.** Published NBA pace matches the uncorrected formula
because it carries the same 0.44 — which makes it a confirmation of the bias, not of the rule.

**U. Correction O's `"null"` sentinel is contained, not live — I claimed otherwise and was
wrong.** The cleaner writes `"null"` at about ten sites and `data_loading.py:53` coerces it to
NaN, so training encodes those cells as the string `"nan"` — a real token in the player, type and
result vocabularies. I read that as a train/inference mismatch on every offensive rebound (the
controller emits `result="null"` at `simulation/controller.py:504`). It is not.
`game_simulator._norm_cat` reproduces the same pandas coercion at every categorical encode site,
and `simulation/input_cache.py` routes through it too, so `"null"` becomes `"nan"` before
encoding and train and inference agree by construction.

What is left is cosmetic: the corpus's semantic sentinel is spelled `"nan"` in the vocabularies.
Changing it costs a re-clean and a fresh vocabulary freeze for no behavioural gain, so it is
**deliberately not fixed** — unlike correction P's `Nene `, which is a genuine duplicate player
embedding and is purged at workstream 12's clean.

**V. Train 3's OOM was a Keras 3 Dense, not a batch size.** The run died 298 steps into epoch 1
on a 2.34 GiB allocation for
`gradient_tape/.../roster_vec/roster_encoder/pma/mab/rff/fc2/MatMul/MatMul_1`, with 22.4 GiB of a
23.2 GiB card in use.

The op is TF's gradient for a *broadcasting* `BatchMatMul`, which materialises one kernel gradient
per batch element — `(B, d_model, d_ff)` — before reducing it to the `(d_model, d_ff)` the kernel
actually is. Keras 3's `Dense` is `ops.matmul(inputs, kernel)` and lowers to that op for any input
of rank > 2; Keras 2's `Dense` reshaped internally first, so the cost is new in 2.0 even though the
set transformer is not. `SequenceRosterEncoder` collapses time into the batch axis, so B there is
`batch x SEQ` = 38400 at batch 64, and `38400 x 128 x 256 x 2` bytes is 2.34 GiB — per Dense, with
eight of them per roster application (2 SABs + PMA's pre-rFF and MAB rFF, two Dense each), applied
twice a step for home and away.

**The allocator dump is the proof, and it is worth reading before touching batch size.** At the
failure there was ~10 GiB free, in blocks of 2.33 / 1.86 / 1.62 / 1.47 / 1.39 / 1.33 GiB — the
holes left by these same transients — and the largest missed the request by 9.8 MB. That is
fragmentation on top of a genuine 96%-of-VRAM peak, which is why it survived 297 steps: nothing
changed at 298 except that the heap finally had no contiguous slab left.

Fix: `RowFF.call` flattens to rank 2 before the two Dense layers and restores the shape after.
rFF is row-wise, so the function is identical; the kernel gradient goes from 2.34 GiB to 128 KiB.
`tests/test_row_ff.py` asserts it by inspecting the emitted graph for `BatchMatMul` — an output
test cannot tell the two graphs apart, which is exactly why this shipped.

Not changed, and noted for whoever needs the next slice: `RosterSetEncoder.scalar_proj` has the
same shape (39 MB an application, not worth the diff), and the deeper waste is that the encoder
re-encodes all 600 timesteps when rosters only move on substitutions.

**W. Train 3 also ran v1.0 code on the pod, and would have trained every head on the full corpus
anyway. Both failures were silent, and neither was in the model.** The RowFF fix (correction V) was
the only one of the three that announced itself.

*The branch was never pushed.* `origin/feature/version2` sat at `f2f0db8` — pre-2.0 — so the
pod's clone produced a v1.0 tree. Verified there after the fact: `git log --oneline -1` returned
`f2f0db8`, and grep found zero occurrences of both `timeout_team` and `lead = tf.shape`. Twelve
hours of training (event_time 5h20m, player 3h43m, event_time_cond 3h20m) were worthless before
the OOM ever mattered. Nothing about the run looked wrong from the outside: same files, same
commands, same banner. The README's clone step named no branch and the procedure had no provenance
check; both are fixed, and Gate C's checklist now carries the check as an item.

*The subset manifest was absent, so every head trained on the full corpus.* `full_run.train()`
called `load_subset_games()`, which returns `None` when `./training/subset_games.json` is missing,
and then trained the seven small heads on all 21,014 games while printing one line of stdout
saying so. Measured from full_train_2's reports, which had the manifest: full-corpus heads ran
391-428 sec/epoch on 21,014 games, subset heads 72-74 on 3,239. 6.5x the games for 5.4x the time,
about sixteen hours of rented card.

The manifest's absence was not carelessness — **there was no point in the documented flow at
which it could have been created.** `extract` reads `full_run_state.json`, which only `setup()`
writes, and `train.py --full` calls `setup()` then `train()` in the same breath. A separate
`python -m training.subset extract` has never had a window to run in. So the fix is auto-extract
inside `train()`, at the one moment the state file exists and no head has started;
`_subset_games()` returns a set and never `None`, and the full-corpus branch is gone rather than
guarded. `retrain_model()` had the same fallback with no message at all and now shares the helper.
The manifest stays untracked, in `.gitignore` beside the state file it derives from: a fresh clone
then carries none, so one is always built from that machine's own state and cannot arrive stale.
Residual, deliberately not coded against: a re-clean on a machine that already holds a manifest
leaves the old one in place, and deleting the file is what re-derives it.

*`timeout_team` was not the third victim, and the epoch times would have said otherwise.* It is a
seventh `ConditionalTypeModel` and it is not in `SUBSET_MODEL_KEYS`, which reads as ~2.7h on the
full corpus against ~25 min for its six siblings. It is not. All seven conditional heads share one
`cond_*.npz`, built once in `run_stage` from `_pp(cond_keys[0])`, and `cond_keys[0]` is `shot_type`
— which is listed. `timeout_team` has been training on subset rows since workstream 13. The
config list was wrong, not the behaviour, and adding the seventh key changes no bytes of any npz.

What was genuinely dangerous is that `run_stage` stated the all-or-none invariant in a comment. It
held only because `shot_type` happens to come first in `TYPE_GEN_SPECS`; reorder those specs and
every conditional head flips to the full corpus with no diff anywhere to show for it. `run_stage`
now raises instead, in the `DataCleaner._check_schema` style.

*`sub_decision` stays on the full corpus, deliberately.* It replaced full-corpus `stint_length`
(correction R) and has no history of its own, so the question was open. It is
`class SubDecisionModel(SubstitutionModel)`: it inherits the roster encoder and trains `emb_player`
against `player_vocab.next_token`, which makes it a player-vocab head by `training/subset.py`'s own
criterion, sitting with `substitution` rather than with the conditionals. Subsetting it on its
first ever train would also confound "new head" with "new regime" in the report. Revisit once
train 3 shows whether it saturates. Two stale `stint_length` strings — the train banner and
`subset.py`'s docstring — are corrected while here.

---

## Log

Append one line per merge. Newest last.

| Date | Branch | Commit | Note |
|---|---|---|---|
| 2026-09-06 | `feature/side-aware-fouls` | f55758d | 61 tests green; and-1 fix beyond spec (correction F) |
| 2026-09-06 | `feature/dead-ball-state` | 5ea3afd | 76 green; found pre-existing test-isolation bug (correction G) |
| 2026-09-06 | `feature/shot-zone-geometry` | 8434786 | 42 green; zone validation table passes all gates (correction H) |
| 2026-09-07 | `feature/shot-zones` | 27a3327 | full suite green; 13 test files re-tokenized, not 6 |
| 2026-09-07 | `feature/ft-count-tokens` | 4b0dd09 | phantom shot_type sample deleted; and-1 rule added (correction I) |
| 2026-09-07 | `feature/fouled-player` | 715cedb | 144 green; opponent 100% populated except technicals |
| 2026-09-07 | `feature/timeouts-team-rebounds` | b54685f | 628 green; team-rebound side recovered at 99.8% (correction J) |
| 2026-09-07 | `feature/schema-cleanup` | 0605a35 | merged on instruction, verified after: 634 green; Phase 2 complete |
| 2026-09-07 | `feature/shared-backbone` | b9ff4ea | layer-name diff clean; correction G closed; workstream 10 split into 10a/10b |
| 2026-09-07 | `feature/local-attention` | f402341 | 663 green; switch-off parity pinned; ARCH_KEYS entry added (correction K) |
| 2026-09-07 | `fix/jump-ball-team-binding` | fac7095 | Gate B found it: ~47% of games bound both sides to one abbreviation (correction L) |
| 2026-09-07 | `feature/possession-clock` | 53224b3 | merged ahead of its run so one clean could settle data, arrays and code together |
| 2026-09-07 | `fix/free-throw-possessions` | bfb52c1 | pace gate caught a 12% over-count; free throws resolve by trip now (correction M) |
| 2026-09-08 | **Gate B** | 0ab3956 | passed on the third clean; vocabs frozen and committed; 685 green |
| 2026-09-08 | `fix/roster-snapshot-flicker` | c9c1d2d | 707 green; the five disagreed with the substitutions ~21x/game (corrections N-Q) |
| 2026-09-09 | `feature/lineup-state` | 4fa3713 | stint/minutes/fouls per player through the roster encoder; 8.6 GB record spike removed |
| 2026-09-09 | `feature/bench-bundle` | 402c182 | ten bench slots per side, second set encoder; BENCH_SIZE into ARCH_KEYS |
| 2026-09-09 | `feature/sub-decision-head` | 1b48794 | rotation is a decision, not a timer; stint head + 4 dials retired (correction R) |
| 2026-09-09 | — | — | clutch weighting rejected before building (correction S); workstream 11 complete |
| 2026-09-09 | `feature/training-changes` | be87b39 | mask 10.6% -> 31.5% of event-head positions; free-throw trip unified and the pace reference de-biased (correction T) |
| 2026-09-09 | `feature/quarter-eval-splits` | a9848c9 | per-quarter boxes + per-zone shot mix; sum identity over 1,500 games found 3 defects |
