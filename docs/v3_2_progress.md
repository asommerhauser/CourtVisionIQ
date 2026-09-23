# CourtVisionIQ 3.2 — Progress Tracker

> **State, not spec.** [`v3_2_direction.md`](v3_2_direction.md) is the direction and is not edited
> during the build; [`v3_2_planned_changes.md`](v3_2_planned_changes.md) is the build guide. This file
> records what has been built, what was measured, and what was learned — including every departure
> from the direction and the number that forced it. Update it in the same commit as the work it
> describes.

## Standing rules

Carried forward from [`v3_progress.md`](v3_progress.md), with **rule 1 amended**.

1. **Trains, rollouts and evals run on the GPU box, not the dev machine.** The dev side writes the
   code, the tests and the spec; the run happens on WSL/CUDA and the report comes back. Claude stops
   at each verification point, hands over the exact command, and waits for a pasted result.
2. **Amendment to rule 1: `pytest`'s local tier runs on the dev machine, after every workstream.**
   `v3_progress.md` rule 1 excluded `pytest` because "the dev box has no GPU and `conftest.py` imports
   TF first", and 3.0 worked around it by executing 667 test functions directly. Measured here on
   2026-09-17: **TensorFlow 2.20.0 imports on Windows in 6.6 s**, `list_physical_devices('GPU')` is
   `[]`, and **`python -m pytest --collect-only -q` collects all 891 tests in 10.7 s** — the same 891
   that passed on WSL on 2026-09-16. The real constraint is **no GPU**, not "no TensorFlow". Keras
   graph construction, `get_config`/`from_config` round-trips, `load_weights` shape refusals, the
   layer-name contract and the one-epoch tiny trains are all CPU work. See W-T below for the tiering.
3. **Commit at each meaningful step**, with the message saying what changed *and why*, including what
   was measured and what was rejected.
4. **No large evals mid-programme.** The full 3.2 eval happens after the train.
5. **Still check `encoder/vocabs/` against git before any train.** W-T adds a session fixture that
   snapshots and restores the directory, but that is a net rather than a fix, and a session teardown
   does not survive a kill. **Measured: the suite does not currently write the committed vocabs at
   all** — every test passes `Encoder(vocab_dir=tmp_path)`, and the shared `norm_stats.json` path is
   keyed off `encoder.vocab_dir`, not `config.NORM_STATS_PATH`
   (`models/event_time_model.py:259-266`). So `v3_progress.md` rule 5 describes a hazard that was
   already fixed test by test; what was missing was anything making it an invariant rather than a
   convention. The fixture is that, and it matters most in W4, which deletes and rebuilds the vocabs
   on purpose.
6. **Line endings are mixed CRLF/LF per file.** Edit lines; never round-trip a whole file.
7. **Non-ASCII in a Bash heredoc reaches Python mangled.** Use a dedicated write, or `chr(0x2014)`.

---

## What is built

**All twelve workstreams are built.** Branches below in the order they were merged; every one is a
`--no-ff` merge onto `feature/version3.2`.

| branch | workstream | retrain? | state |
|---|---|---|---|
| `v3.2/w0-spec` | direction, build guide, this tracker | no | **built** |
| `v3.2/test-tiers` | WT: `pytest.ini` markers, two conftest fixtures | no | **built**, 49-test subset green |
| `v3.2/priors-join` | W1: one definition of the `game_id` numbering | yes (sidecar rebuild) | **built**, 153 tests green |
| `v3.2/corpus-cut` | W2: corpus floor, applied to rows | yes | **built**, 165 tests green |
| `v3.2/corpus-cut-2011` | W2a: the floor moves 2008 -> 2011 | yes | **built** |
| `v3.2/vocab-floor` | W4: the floor, and anonymous slots by graph colouring | yes | **built**, 20 new tests; verified on the real corpus |
| `v3.2/prior-scalars` | W5: three rates, career stage, deltas; 14 -> 21 scalars | yes | **built**, sidecar rebuilt |
| `v3.2/film-context` | W6: per-block scale and shift from a game-context vector | yes | **built**, identity at init verified |
| `v3.2/vocab-floor-refresh` | W4 fix: reload the alias map before a vocab rebuild | yes | **built**, guard verified by breaking it |
| `v3.2/cross-roster` | W7: each lineup attends over the other before pooling | yes | **built**, 13 tests; 87-test graph/persistence run green |
| `v3.2/rung2-bridge` | W8: the three missing pieces of checkpoint selection | no | **built**, 29 tests |
| `v3.2/head-metrics` | W9: twelve heads, twelve metrics, measured normalisers | no | **built**, 21 tests |
| `v3.2/decision-log` | W10: what each head sampled, per sim | no | **built**, 20 tests |
| `v3.2/weighted-replay` | W11: advantages, the filter, the weight channel | no | **built**, 20 tests |
| `v3.2/ab-harness` | W12: three arms, paired statistics, run-state records | no | **built**, 18 tests |
| `v3.2/subset-all-heads` | W3: all twelve heads on the subset; coverage retired | yes | **built**, 124 tests green |

**W1 verified two ways.** `tests/test_player_priors.py` gains three tests that build a real sidecar
over two season files carrying *the same raw ids* -- the case that used to collapse -- and assert the
corpus is covered, that the causal seed chain still crosses the season boundary, and that a
`--seasons` partial rebuild still lands on corpus ids. And `season_offsets("./data")` was compared
against the offsets computed by replicating the original inline walk on the real 21-season corpus:
**all 21 match exactly, so the numbering is unchanged.** That is what keeps the stored
`holdout_game_ids` valid and window 0 comparable to v2-run1..4.

**W1 and W5 are both verified against the rebuilt sidecar.** `python -m player_priors` was re-run
after W5 changed the columns (~13 min, pure pandas, 559k rows across 21 seasons). Measured on the real
corpus afterwards:

| | before W1 | after |
|---|---|---|
| games the sidecar and the corpus agree on | **1,277 of 26,969** | **26,969 of 26,969** |

`require_priors` reports 26,969 and nothing is uncovered, so `train.py --full` no longer aborts before
its first epoch. The 2023 table carries 20 columns with the last 17 in `PLAYER_PRIOR_KEYS` order, and
`career_stage` reads mean 4.86 / max 19 against the 4.84 / 19 measured independently beforehand.

---

## Measurements taken before the build

All on the dev machine, read-only, from artifacts already on disk. No simulation, no training.

### 1. The priors sidecar joins 1,277 of 26,969 games — and it blocks the retrain

`player_priors.build` reads each season CSV directly (`player_priors.py:308`), so `data/priors/` is
keyed by **per-season** `game_id`. `data_loading.load_all_cleaned` shifts each file's ids by the
running maximum (`data_loading.py:52-55`), so the training rows carry **cumulative** ids. Computed
over the real corpus by replicating the offset walk:

| | |
|---|---|
| games in the corpus, as `load_all_cleaned` keys them | 26,969 |
| game ids in `data/priors/` | 26,969 |
| **ids the two agree on** | **1,277** |
| sidecar ids that are raw-but-not-offset | 25,692 |

1,277 is season 2003 — the one file whose offset is zero. `merge_prior_features` raises on
`covered < total` (`models/prior_features.py:207-217`), so `train.py --full` would abort before the
first epoch with "the priors sidecar ... is stale".

**This is a live defect in untrained 3.0 code, not a 3.2 one.** It is loud rather than silent, which
is exactly why it has survived: the 3.0 handover never got past step 3 (`python -m player_priors`),
and step 4 is where it fires. W1 fixes it first.

Raw per-season id ranges are also **not ordered by season** — 2016 holds 1313–2628, 2019 holds
1–1312, 2003 holds 15556–16832 — so the offset walk is order-dependent and ids are positional
artifacts, not intrinsic. That is what forces W2's placement.

### 2. What the vocabulary floor actually costs — measured at both candidate cuts

Computed from the 559k-row priors sidecar, weighting each season by the live subset sampling rate
(`SUBSET_RECENT_SEASON_RATES = (1.0, 0.70, 0.50)`, `SUBSET_RECENCY_HALFLIFE_SEASONS = 5.0`) and halving
the newest season for the `FINAL_SEASON_FRACTION` cut. These are **expected exposures**, not a real
extraction — the real histogram comes from `training/subset_games.json` after W3, and
`training.subset extract` now prints it.

| | cut at 2008 | **cut at 2011 (chosen)** |
|---|---|---|
| seasons kept | 16 | **13** |
| train pool | ~19,800 games | **~15,875 games** |
| expected subset | ~5,750 games | **~5,374 games** |
| players with any exposure | 1,797 | **1,614** |
| median expected subset games | 31 | **34** |

So §3.2's "start at 20–30" sits **at the median** either way, which is a far deeper cut than the
document implies.

| floor | vocab kept | → anonymous | minutes anon (all) | minutes anon (2021+) | games with ≥2 | max in one game |
|---|---|---|---|---|---|---|
| 10 | 1,151 | 463 | 1.60% | 0.91% | 11.6% | 8 |
| 15 | 1,056 | 558 | 2.63% | 1.30% | 19.6% | 10 |
| **20** | **959** | **655** | **4.03%** | **2.02%** | **28.7%** | **12** |
| 25 | 905 | 709 | 5.11% | 2.57% | 34.8% | 13 |
| 30 | 843 | 771 | 6.68% | 3.30% | 43.9% | 15 |

**The minutes cost is affordable; the collision count is what forced a design change.** At a floor of
20, two or more below-floor players are rostered in **28.7% of games** (mean 1.23, max 12). A single
shared `UNK` id cannot tell them apart, and `game_available_mask`
(`models/event_time_model.py:219-239`) zeroes `PAD` at `:238` but **not** `UNK`, so the player and
substitution heads can spend probability mass on an unresolvable token. Hence W4's per-game anonymous
slots (`ANON_SLOTS = 16`, comfortably above the measured maximum, with `UNK` as overflow).

**One cross-check worth recording:** the 2021+ minute shares are *identical* at both cuts
(0.91 / 1.30 / 2.02 / 2.57 / 3.30). Removing 2008–2010 barely changes which *recent* players clear the
floor, which is what the scored holdout is actually made of — so the choice between the two cuts is
about the old tail, not about the games being predicted.

### 3. The suite, at the end of the build

**1,054 tests, all green**, against 891 when 3.2 started — so the build added **163**. Measured on the dev
box with CPU-only TensorFlow:

| tier | tests | wall time |
|---|---|---|
| fast (`pytest -m "not slow"`) | **1,005 passed** | 9 min 55 s |
| slow (the graph-heavy modules) | **49** | part of a 29-min, 87-test run |

The `slow` marker is a measurement, not a guess: `test_model_persistence` and `test_local_attention` run
`preprocess` plus a one-epoch train per test. A separate 87-test run covering those plus `test_backbone`,
`test_substitution_model` and `test_oncourt_mask` took 29 minutes and is what verified W7's save/load
across every head.

*One earlier claim withdrawn.* Mid-build I inferred from counting progress dots that FiLM and
cross-attention had roughly doubled the tiny-model test time. The full numbers do not support it — 87 tests
in 1,739 s is ~20 s a test against ~31 s for the 49-test run before, and the difference is module mix, not
a slowdown.

New test files: `test_subset.py`, `test_player_floor.py`, `test_cross_roster.py`, `test_head_metrics.py`,
`test_decision_log.py`, `test_replay.py`, `test_ab_harness.py` — five of which cover machinery that had
**no** tests before (the subset sampler, `MAB` in cross mode, and all three rung-3 pieces).

### 4. The suite runs here, but some modules are slow on CPU rather than instant

891 tests collect in 10.7 s on Windows with CPU-only TensorFlow, and a full run reached **562 tests
with zero failures** before it was stopped at ~29 minutes of CPU time and a 2.9 GB working set. It was
progressing throughout — sampled twice, it held ~100% of one core — so the older note that local runs
"stall" is really "the modules that build real Keras graphs cost minutes per test on CPU, not
seconds". `tests/test_game_simulator.py` is where the cost becomes obvious.

**Consequence for how the suite is used.** The per-workstream gate is the *relevant* modules, run
targeted and fast; the full local sweep is a milestone check. A per-module duration sweep is still
**pending**, so no module carries the `slow` marker yet — marking them is a measurement, and an
unmeasured guess in `pytest.ini` would be worse than an empty marker list.

---

## Departures from the direction, and the reason for each

### 1. The corpus floor is applied after the game-id offset walk, not in `cleaned_csvs`

§3.1 says "`data_loading` / `training.chronology` gain a minimum-season bound" without saying where.
Placed in `cleaned_csvs`, it would renumber every remaining game (measurement 1), invalidating
`full_run_state.json`, `training/subset_games.json` and every `results/<run>/holdout.json`, and
breaking §7.2's claim that window 0 is byte-identical to the games v2-run1..4 scored. Applied inside
`load_all_cleaned` **after** the offset walk, every id survives and only `pos` / `boundary_idx`
renumber.

It also makes §3.1's own warning — "cut the training games, never the sidecar", which the document
calls "the single easiest thing in 3.2 to get wrong" — true **by construction**, since
`player_priors` and `season_context` walk `cleaned_csvs` directly and are therefore untouched.

**How it is expressed.** `load_all_cleaned` gains an opt-in `min_season=`, and
`data_loading.load_training_corpus` is the one caller that opts in. The four heads' `_load_all`,
`training.chronology.game_index` and `training.subset` all go through it; everything that genuinely
wants 21 seasons — the box-score validator, the shell, the report stack, the priors sidecar, season
context — keeps calling `load_all_cleaned` and says so by doing it. One name, so the floor cannot
apply in some training paths and not others.

`data_loading.training_min_season()` reads `config` **at call time**, not via a module-scope
`from config import`. That is 3.0's bug 4 (`from config import X` froze three knobs at import, so
switching one off in a test did nothing and the disabled path was silently untested), and there is a
test pinning it.

**Four invariants are tested, not assumed:** ids survive the cut unchanged while `pos` renumbers; the
holdout lands on the *same games* across a cut (the real content of §7.2's window-0 claim); a floor
past the whole corpus raises rather than yielding an empty frame downstream; and `load_all_cleaned`
still sees every season.

### 2. The corpus is cut at 2011, not the 2008 the direction proposed

**User decision, 2026-09-17**, taken after the floor arithmetic was measured both ways (measurement
2). Three more seasons leave the training pool: ~19,800 games becomes ~15,875, and the expected subset
~5,750 becomes ~5,374.

Two things make it a comfortable decision rather than a trade. Every season it removes is **already
pinned at `RECENCY_FLOOR = 0.05`** in the loss — a halflife of 3.0 reaches the floor about seven
seasons back, so 2016 and older are all there — and their subset sampling rates run 0.11 (2010) down
to 0.0021 (2003). Multiplying the two, a 2010 game carries about **0.5% of the gradient a current game
does**, and 2003–2010 together come to roughly **37 current-game equivalents**: one to two percent of
the total. So the expectation of neutrality is, if anything, better supported than at 2008.

And it is the deeper corpus cut but the **milder vocabulary cut**. It removes 183 players outright
(1,797 → 1,614), almost all low-exposure old-era names, so the median player's subset exposure *rises*
from 31 games to 34 and every floor keeps a smaller fraction anonymous. The cost the direction names
is unchanged and still stands: rare tokens get rarer, and the rarest foul and rebound sub-types are
where it would show.

### 3. Below-floor players are aliased to anonymous slots — and the slots come from a graph colouring

§3.2 maps them all to `UNK`. Measurement 2 shows that is ambiguous in 28.7% of games, up to twelve
anonymous players in one, so a single shared id cannot stand for them.

**The planned fix was a per-game map. It turned out not to be necessary, which is the useful finding.**
Build the co-occurrence graph over below-floor players — an edge between any two who ever appear in the
same game — and colour it greedily, highest degree first. Measured on the real corpus: **36 slots, zero
collisions across all 16,535 games**, against a busiest game of twelve. A naive `rank mod slots`
assignment collides in 7.4% of games at 16 slots and 2.2% at 64; the colouring collides in none.

So one **global, stateless** `name -> token` map delivers exactly the guarantee a per-game map would,
and avoids all of its cost: no grouping by `game_id` in six heads' preprocess, no second parse of every
roster cell, and nothing threaded through `simulation/input_cache.py` or `simulation/game_input.py`. The
slot count is therefore not a constant — it is whatever the colouring needs, with `ANON_SLOTS_MAX = 128`
as a bound that raises rather than truncating.

**Measured against the real corpus at the 2011 cut and a floor of 20:** 970 players keep their own
embedding row, 644 are aliased into 36 slots, and the table goes from 2,152 rows to about 1,011 — a 53%
cut in exactly the place §1f blames for the memorisation. Rebuilding the assignment twice gives an
identical result, which matters because it is persisted and the whole vocabulary depends on it.

**Applied inside `Encoder`**, because the alias map is part of *the language*: `encode_roster`,
`encode_player` and `encode_secondary_player` all route names through `Encoder.alias`, so no call site
changes, and the map is saved and loaded with the vocabs, snapshotted into `artifacts/<name>/vocabs/` by
the existing `snapshot_vocabs`, and fingerprinted by the manifest. `secondary_player` shares the
aliasing because it shares the vocab — otherwise a below-floor assister would be one id there and
another on the floor.

**`game_available_mask` needed no change at all**, which is a consequence of the design rather than
luck: it marks the ids that actually appear in a game, so the anonymous tokens present are available
and the rest are not. The `UNK`-is-samplable worry that motivated the whole design disappears, because
no real player maps to `UNK` any more. Verified end to end — `PAD` and `UNK` both read 0, and two
anonymous players sharing a game get distinct available ids.

**A third gap, found by tracing the handover order rather than by reading the code.** A head's `Encoder`
is constructed when the head object is created, which is **before** `full_run.train` extracts the subset
— and the extract is what writes `anon_slots.json`. So at construction the map is legitimately empty, and
a rebuild trusting that in-memory copy would register every below-floor player under his own name and the
floor would do nothing. **None of the existing guards see it:** `require_player_floor` checks the file,
which exists; `assert_aliases_absent` returns early on an empty map; and the only symptom is an embedding
table that did not shrink. `Encoder.prepare_for_rebuild()` re-reads the map, and all six heads' rebuild
branches call it — with a test that greps for it in each, verified to fail when the call is removed from
one head.

**Two silent failures made loud.** `Encoder.freeze_all` refuses a vocab that still holds a row for an
aliased player: `Vocab` is append-only, so a rebuild over a pre-floor vocab would keep every below-floor
name, the floor would do nothing, and the only symptom would be a table that did not shrink. And
`require_player_floor`, called from `full_run.train` beside `require_priors`, refuses a configured floor
with no alias map on disk.

**A known limit, out of scope.** A name absent from the training corpus entirely still encodes as `UNK`,
so two genuinely unseen players on the same floor remain ambiguous. That is pre-existing behaviour and
only bites upcoming-season inference, where such a player has no priors either.

### 4. The seven new scalars split into rates and derived, and only the rates are shrunk

§3.3 and §3.4 list seven additions and treat them alike. They are not alike, and the implementation
says so: `PLAYER_RATE_KEYS` holds the thirteen shrunk rates, `PLAYER_DERIVED_KEYS` holds `career_stage`
and the three deltas. `shrink`'s default `keys` narrows to the rates accordingly — shrinking a career
year toward a seed is meaningless, and a delta is *already* a difference of two shrunk quantities.

**Each delta is taken against the seed** (last season's final rate) from the shrunk current rate, so on
opening night the shrunk rate still is the seed and the delta reads zero, growing as the season
accumulates evidence. That is the honest shape for "has his role changed": not yet known. A player with
no previous season gets exactly 0.0 rather than a delta against the league mean — which would be a
statement about how good he is wearing the clothes of a statement about change, and indistinguishable
from a real role shift.

**Two things the tests caught rather than the reading.**

`career_stage` for a debutant is **0.0, not the league default**. The cold-start test asserted that
every key equals `LEAGUE_DEFAULTS`, and it failed — correctly. A player in his first season *is* at
stage 0; `LEAGUE_DEFAULTS["career_stage"] = 4.5` is what an *unknown name* reads through
`_DEFAULT_PLAYER`, which is a different situation (no record at all, versus a record saying "season
one"). The test now says both, and the distinction is the point of the input.

And the test fixture stamped `season=2023` on every row regardless of which season file it was written
to, so `career_stage` read 0 everywhere and the new tests failed for a fixture reason rather than a code
one. Worth recording because the same fixture is now used by W1's join tests.

**Normalization divisors are measured, not guessed**, since a test asserts an average player reads near
1.0: `ft_pct` 0.78, `tp_pct` 0.36, `pf_36` 2.95 (2022-23 via `generate_box_score`: 0.7825 / 0.3600 /
2.9689), `career_stage` 4.5 (mean 4.84 over 2011+ player-games, median 4). The three deltas centre on
0.0 and are excluded from the [0.4, 1.6] band — a narrowing, not a weakening, since the band is a claim
about rates. Shifting them to centre on 1.0 was rejected: it would make "no change" indistinguishable
from "no information".

**`feature_mismatch` finally has a caller.** It was written in 3.0 and tested, but nothing called it, so
the 14 -> 21 change would have surfaced as a raw Keras kernel-shape error from `scalar_proj`.
`shell/actions._check_features` now runs beside `_check_arch` on every load.

### 5. FiLM is zero-initialised, so the A/B starts from an exact identity

§5.2 asks for "a small network emits a per-block scale and shift". The implementation adds two things the
section does not specify, and both are load-bearing.

**Zero-initialised kernel *and* bias, applied as `h * (1 + gamma) + beta`.** At initialisation gamma and
beta are exactly zero, so the modulation is the identity and a FiLM graph is numerically the same as one
built without it — verified with a **maximum observed difference of 0.0**. Predicting the scale directly
would perturb the residual stream before training begins, and the W6 gate ("must not worsen any probe")
would then be comparing two different initialisations rather than FiLM against no FiLM. A companion test
pins that it does not stay inert: one gamma set to 0.5 changes the output.

**A shared `FILM_DIM = 64` bottleneck**, because "negligible parameters" is not automatic. Projecting the
raw 160-wide context straight to a scale and shift in every block would cost **1.48M parameters a head**;
through the bottleneck it is **609,344, or 5.7% of the backbone's 10.76M**. That matters in a cycle whose
whole argument is that capacity is not the binding limit — W4 removes about 1,100 embedding rows, and it
would be odd to hand most of that back as modulation width.

Two smaller choices worth recording. The scale and shift are separate `Dense(d_model)` layers rather than
one `Dense(2·d_model)` that is sliced, and the combination is `Add([h, Multiply([h, gamma]), beta])`,
because both avoid a `Lambda` in a graph that has to reload by name. And the season embedding is captured
**by name** inside each head's embedding loop rather than indexed out of `embs`, so a change to
`CATEGORICAL_FIELDS` order cannot silently hand FiLM the wrong tensor.

The per-head name test now passes `film=config.FILM_ENABLED`. With `film=False` the non-FiLM names are
still present and still in order, so the assertion would have passed while checking nothing about W6.

### 6. Cross-roster attention needed the encoder split, and MAB needed a wrapper

§5.2 says "``layers/mab.py`` already has the block", which is true and not sufficient. `MAB` was written
for cross-attention -- `call(X, Y)` already means "X attends to Y" -- but had **never been used that
way**: `SAB` passes `Y = X` and `PMA` passes learned seeds, and there is no `tests/test_mab.py`, so the
path it exists for was entirely unexercised. Three consequences, each of which bites in a functional
graph: no `compute_output_shape` (and a second positional tensor is not in the standard `inputs` slot, so
Keras has nothing to infer from), lazily-created variables (which under `load_weights` means saved weights
have nowhere to land), and post-norm against a pre-norm backbone.

The last one turns out to be fine, but only because of *where* this sits: inside the roster encoder, among
the post-norm `SAB` layers it was designed alongside, rather than spliced into the residual stream.

And **"before pooling" forced a split**. Cross-attention needs both sides' slots live at the same moment,
and a layer that returns a pooled vector has already thrown them away -- so `RosterSetEncoder` grew
`encode_slots` and `pool_slots`, with `call` now exactly those two in sequence and a test asserting the
split changed nothing about what it computes.

### 7. The A/B thresholds needed the paired difference sd, not a single run's

A correction caught by trying to reproduce §7.2's table and failing. **0.174 is one run's per-game Brier
sd**; a comparison rests on the sd of the per-game *difference*, which is about half that because two arms
on the same games make correlated errors. §9.2 records the measurement that pins it: paired run3-vs-run4
came to ±0.0109 at one standard error over their 64 shared games, so the difference sd is
0.0109 × √64 ≈ **0.087**. With both constants the harness reproduces all three published figures.

Using the single-run figure for a paired comparison would have **doubled the threshold and hidden every
gain 3.2 expects**.

One discrepancy recorded rather than silently fixed: a strictly-correct unpaired comparison of two
independent runs carries a further √2, making that row 0.049 rather than §7.2's 0.036. The 0.036 is 2 SE of
a *single* run's Brier, which is the form §9.2 recorded and the project's documents quote, so it is
reproduced as recorded with the discrepancy named in the docstring.

### 8. Two pieces of the replay pass turned out to be free

§4.2 describes logging context tensors and building a weighted pass. Both are smaller than that.

**The labels need no construction.** A sim's play-by-play *is* what the model sampled, so running it
through the ordinary preprocess yields "what this sim did" as targets.

**The weight needs no new machinery.** Every head already multiplies its loss mask by a per-game weight
through `season_features.apply_recency`, which reads `split["recency_weight"]` — so an advantage is that
existing channel with a different number in it, and all twelve heads already honour it. A test runs the
result through `apply_recency` to prove the two actually meet.

What is *not* free is the context: `build_model_inputs` serves a dict memoised per row and shared across
the ~5 head calls at one position, and the prior columns alone run to megabytes per position at
`SEQ = 600`. So the decision log is a thin index and the context is re-derived — which is also why the log
is written *beside* `playbyplay/` rather than inside it, since `harvest.py` prunes that directory and would
take the labels with it.

### 9. Coverage-completeness is retired, and the sampler gets its first tests

§2.2 argues the guarantee is inert under a minimum-games floor: anyone it rescues with a single game
falls below the floor anyway, and it drags old games into a deliberately modern-heavy sample to do it.
Retiring it also removed the per-game recency weights, which only ever broke ties when choosing
*which* game to add for a rare player — the per-season fill is uniform by construction, so the modern
tilt now lives entirely in the rates.

**The sampler had zero test coverage.** Nothing in the suite referenced `build_subset`,
`season_sample_rates`, `load_subset_games`, `SUBSET_MODEL_KEYS` or `subset_train_games`, so retiring
the guarantee would have produced no failure in either direction. `tests/test_subset.py` is new and
starts from zero: per-season rates and their decay, per-season fill targets, seed determinism, the
subset-⊆-train property, the empty pool, manifest round-tripping, and a test that pins the behaviour
change itself — a player who appears only in a zero-rate season is now simply absent rather than
rescued.

It also pins the routing as a **membership** test against `STAGE_MODEL_KEYS` rather than a count. The
list was wrong before in exactly the way a count would not catch: it named six of the seven
conditional heads, which was true of nothing, because `timeout_team` already trained on subset rows.

`build_subset` now emits `stats["players"]`, the per-player game count **inside the subset**, which is
what W4's floor reads; and `extract` prints the distribution plus a kept/anonymous table at floors
10-30, so the floor is chosen against the real histogram rather than the estimate in measurement 2.

### 10. The replay estimator keeps only positive advantages

§4.2: "the advantage is the sample weight. Positive advantage reinforces those choices, negative makes
them less likely." A negative weight on cross-entropy is `-w·log p` with `w < 0`, which is **minimized
by driving that action's probability to zero and the loss to −∞** — unbounded. Signed REINFORCE via
weighted cross-entropy needs advantage clipping plus a KL leash to stay stable, which is three
constants that cannot be tuned inside a single 3.2 GPU-hour pass. Filtering to the sims that beat
their nine siblings is bounded by construction, needs no hyper-parameters, and never pushes away from
anything.

### 11. Two §4 citations corrected rather than implemented

- §4.4 rule 2 says to "drop `seconds` from the team aggregate entirely" and that "`eval_metrics`
  already says this in its headline block". **`_BOX_ACCURACY_STATS`
  (`simulation/eval_metrics.py:30-31`) carries derived `minutes` and no `seconds` at all**, so the
  first half is already done. And there is no such statement in the code: the nearest comments are
  `eval_metrics.py:27-31` (why `minutes` is derived from stored `seconds`), `stats.py:19-35` (the same),
  and `game_state_features.py:71-73` ("past 24 s the exact value carries no information"), which is
  about the shot clock.
- §4.2 says `apply_query_mask` "already marks these positions — roughly 68.5% of rows".
  `apply_query_mask` (`models/game_state_features.py:557`) is called by **2 of 12 heads**
  (`event_time`'s two outputs and `event_time_cond`); the figure is the event/time head's kept share.
  The other ten heads' queried positions are event-token-gated (`next_event == <token>`) and much
  sparser, so the decision log is defined per head from its own sampling call.

### 12. Rung 2 stays scoped to `event_time`

§4.6 prices the bridge at one day. `rollout_score_fn` exists only on `EventTimeModel.train`
(`models/event_time_model.py:711`); the other five head classes have the identical signature without
it, and `models/pipeline.run_stage._train` has no channel to pass one. Widening to all twelve heads is
a design change, not a bridge, and is 3.3.

---

### 13. Rung 2 cannot run on the pass that builds the bundle — found by running it

`ROLLOUT_SELECTION = True` was committed on, and the first 3.2 full train died three epochs into
`event_time`: `ROLLOUT_EVAL_EVERY = 3`, the callback fired, `make_sim()` called
`GameSimulator.load(artifacts_root)`, and nothing was there — because `event_time` is the head being
trained and the other eleven have no weights yet. `FileNotFoundError`, two wasted epochs, on a rented card.

`_rollout_score_factory`'s own docstring already said why ("mid-first-train the other eleven heads have
no weights of their own, so a scored rollout would be scoring a bundle that does not exist"). Departure 12
scoped rung 2 to a second pass. Neither fact was enforced anywhere: the flag meant "attempt it", and the
only thing standing between a first-ever train and a crash was remembering to turn the flag off by hand.

Two changes, both in `training/full_run.py`:

* **The factory pre-flights the bundle.** `missing_heads(artifacts_root)` lists the heads with no
  `.weights.h5` on disk; when any are missing the factory prints which ones, prints the remedy
  (`python train.py --model event_time`), and returns `(None, None)`. `run_stage` already tolerated a
  factory returning nothing, so the train simply proceeds without selection. Same shape as the priors and
  floor pre-flights: refuse in words, before the epochs are paid for, rather than raising hours in.
* **`retrain_model` passes the channel.** It never did. It called `run_stage` without
  `rollout_score_fn_factory` and never called `record_selection`, so the handover's arm 2 — the pass whose
  entire purpose is rung 2 — would have trained a fresh `event_time`, scored no rollouts, recorded no
  `epochs_disagree`, and looked completely normal. The crash is what sent anyone to read that function.
  The recorder is now one method (`_record_selection`) shared by the full train and the second pass.

`tests/test_full_run.py` pins all four behaviours: the head inventory, the refusal and its wording, the
factory arming on a complete bundle, and the channel reaching `run_stage` from `retrain_model`.

**The pod is running with `ROLLOUT_SELECTION = False`** set by hand in `config.py`, because the train was
relaunched before this fix existed. That is arm 1 behaving as designed either way; the flag has to go back
to `True` for step 6.

### 14. W11 and W12 were libraries nothing imported — arm 3 was arm 1 with a new name

The same discovery as departure 13, twice over and larger. `models/replay.py` (the estimator),
`simulation/decision_log.py` (what each head sampled) and `reporting/ab_harness.py` (arm identities,
the comparability refusal, the paired test) were all built, all tested, and **imported by nothing
outside `tests/`**. The handover's arm 3 was `evaluate.py --run v32-a3`, which evaluates the bundle
arm 1 produced and labels the result the third arm; its comparison was two Brier numbers read by eye.

So the pass needed a driver and the comparison needed one:

* **`training/replay_pass.py`** — select one game in ten of the subset, simulate ten sims each, score
  every sim per head (with the three game-state probes computed from the sim's own rows, so the foul
  and rotation behaviour the gate names actually reaches the weights), keep the sims that beat their
  siblings, and run one weighted pass through `run_stage`.
* **`training/replay_corpus.py`** — the labels. A sim's play-by-play *is* what the model sampled, but
  it is not a corpus: three things a simulation cannot know are carried from the real fixture
  (season context, per-slot rest re-laid against each sim row's own roster, and the priors, re-keyed
  onto the sim ids because `merge_prior_features` raises on partial coverage).
* **`reporting/ab_report.py`** — read each arm, refuse an incomparable set, run the paired test over
  both increments and the whole, write JSON and a page, and record it in the run state.

Two things `run_stage` grew to make the pass safe, both enforced rather than remembered.
`processed_dir`, because preprocessing a corpus of sims at the default path would overwrite
`./data/processed` and the next `--continue` would train on simulations silently. And
`on_preprocessed`, the one seam where a weight can reach a split that only exists on disk — routed
through a single `_prep` helper so it cannot be attached to five of the six preprocess call sites,
which is the shape of bug this file has had twice already.

**Departures taken here, both stated rather than hidden.** `REPLAY_LR = 3e-5`, a tenth of the train
LR: the spec names no learning rate, and one pass over ~5,200 sim-games at 3e-4 moves the weights
about as far as several ordinary epochs, which cannot be judged against a gate that says "must not
worsen margin dispersion". And the labels come from re-preprocessing the sim corpus rather than from
the decision log's `(position, head, output, token)` rows — the log's own docstring says the context
is re-derived by replaying the rows, and the rows already carry every head's choices, so the log
stays a cross-check rather than a second label path.

### 15. "The final paired evaluation at 700 games" was not expressible — an arm is now several runs

`--window K` scores exactly `HOLDOUT_WINDOW_GAMES` (100). There is no flag that evaluates the whole
700-game pool, so the handover's step 7 — the sentence carrying the entire argument for widening the
pool — had no command behind it. Written as it was, it would have scored window 0 and reported it
as 700.

700 games is **seven runs per arm** (`--window 0` through `--window 6`), pooled at the point of
comparison rather than merged into one evaluation: each run records its own k, and drift with k is a
finding (`v3_direction.md` SS4), so merging them would destroy a measurement to save a column.
`read_arm` therefore takes one run dir or several, and refuses an arm whose runs mix seeds or sim
counts, or whose windows overlap — a game scored twice would be paired twice and inflate n.

`assert_comparable` judges a single window field, which an arm spanning seven runs cannot express:
an arm on windows 0-6 and an arm on window 0 alone both report 0 and would pass. `compare` checks the
window *set* as well, because the failure is not an exception — the pairing is by game id, so a
mismatch silently compares 700 games against the 100 they contain and reports it under the wider
arm's name.

### 16. Rung 2's first real invocation ran 9.5 hours without finishing one evaluation

`python train.py --model event_time --name version3.2`, 2026-09-22. Epochs 1 and 2 took 187 s and
103 s. Epoch 3 — the first `ROLLOUT_EVAL_EVERY` boundary — went silent and stayed silent for
**9 h 27 m**, at which point the process had 20 GB resident, 11 h of CPU against 9 h of wall clock
(122%, i.e. ~1.2 cores), **1% GPU utilisation**, no open data files and byte-identical RSS across
samples. `py-spy` could not attach: these pods have no `SYS_PTRACE`. The diagnosis came from `/proc`
instead, and it was not one defect but three, each of which alone would have been survivable.

**The inference path.** `simulation/game_simulator.py` wraps each head in a cached `tf.function`
— and that wrapper is **opt-in, default off** (`CVIQ_TF_INFER`), because its payoff is GPU-specific
and was never measured on hardware. Its own comment predicts the observed failure exactly: called
eagerly, each tiny forward pass pays Python-side op-dispatch that dominates, and *"the GPU sits
mostly idle waiting on Python"*. Every path had run eagerly until now and been merely slower; rung 2
is the first caller that makes ~200 game-sims of tiny passes **inside the training loop**, where
"slower" compounds into a run whose remaining cost is already unaffordable. `evaluate.py` hides this
behind `--procs`, which buys back a factor of eight with processes and leaves the per-sim cost
unexamined. The env var stayed off on the pod (`CVIQ_TF_INFER=[]`), so arm 1 was scored on the eager
path too — which means **pass 3's 43 GPU-h budget rests on the same unmeasured number.**

**Nothing timed or bounded an evaluation.** One evaluation was a single blocking call that printed
on completion. A rollout that never returns and a rollout that is merely slow are therefore
indistinguishable from outside, which is why nine hours passed before anyone could say which it was.

**The corpus was parsed whole to keep twenty games.** `_games()` called
`load_all_cleaned(parse_rosters=True)` over all 21 seasons and filtered afterwards — an
`ast.literal_eval` per row for ~13M rows, in the training process, to keep the ~1,000 rows of the
scored games. That is most of the 20 GB, held for the whole train beside the training graph and a
twelve-head simulator, on the host-RAM budget that already cost 2.0 a train.

**What changed.** `ROLLOUT_COMPILED_INFERENCE` (default on) opts rung 2's simulator into the
compiled forward per-simulator, via a new `enabled=` argument on `_compiled_forward`, without
flipping the process-wide default for paths where it is still unmeasured. The evaluation times
itself, announces its shape and inference path on the first call, emits a progress line a minute
through `simulate_games` streaming mode, and aborts through `ROLLOUT_EVAL_BUDGET_MIN` (25 min) with
the measured number when one evaluation runs past it. `load_all_cleaned` grew a `game_ids=` filter
applied **before** roster parsing. The streaming rewiring moves where probe histories come from
— the return value is empty by contract in that mode — which is a silent-wrong failure if got
wrong, so `tests/test_rollout_bridge.py` covers it specifically.

**What is still unknown, and should not be written up as fixed.** The compiled path has *never been
measured on a GPU in this repository*. Its own comment warns that if `reduce_retracing` fails to
collapse the varying sequence-length dimension it retraces per event and runs **slower**. So the next
invocation is a measurement, and the budget guard is what makes taking that measurement cheap: it
costs 25 minutes to learn the answer instead of a night. The honest state of rung 2 is that its cost
model — `config.py`'s ~7.5 GPU-min per evaluation, derived from 37 GPU-min per 1,000 sims on the
**3.0** graph — has one real observation against it and none in its favour.

## Gaps found in 3.0's code while planning 3.2

Each would have surfaced as a failure or a silent constant in the first real 3.0 train.

1. **The priors sidecar join** (measurement 1). Blocking, and it fires at the start of the train.
2. **`eval_game_ids` reads a state key nobody writes.** `state["train_tail_game_ids"]`
   (`models/rollout_selection.py:70`) appears only there and in its own test; `FullRun.setup` writes
   `boundary_idx`, `holdout_game_ids`, `n_games`, `eval_batch`, `status` and `trained_models`. So
   rung 2's game set is unreachable from a real run state, and it would have returned `[]`.
   `boundary_idx` is read at `:67` and then unused.
3. **Nothing constructs `rollout_score_fn`**, and `self._checkpoint_selection`
   (`models/event_time_model.py:887`) is written and never read. `ROLLOUT_EVAL_SIMS` (`config.py:81`)
   is defined and never read.
4. **`rollout_score` expects `probes["rows"]` and nothing produces that shape.**
   `reporting/state_probes.compare()` returns `{"run", "seasons", "sim", "real"}`.
5. **`models.manifest.feature_mismatch` has no caller.** `shell/actions.py` checks `ARCH_KEYS` and
   vocab sizes only, so a feature-signature change surfaces as a raw Keras kernel-shape error rather
   than the named refusal the function was written for.
6. **`game_available_mask` does not mask `UNK`** (`models/event_time_model.py:238` zeroes `PAD`
   only). Harmless while every player has his own row; a correctness bug the moment a floor exists.
7. **The subset sampler has zero test coverage.** No `tests/test_subset.py`, and no reference to
   `build_subset`, `season_sample_rates`, `load_subset_games`, `SUBSET_MODEL_KEYS` or
   `subset_train_games` in any test file.
8. **`layers/mab.py` has never been called in cross mode** and has no `compute_output_shape`; there
   is no test for `MAB`, `SAB` or `PMA`. It is also post-norm while the backbone is pre-norm.
9. **The committed `player_vocab.json` holds 2,152 entries**, not the 2,153 `technical_specs.md:127`
   and `methodology_whitepaper.md:107` both state.

---

## Handover — the WSL/CUDA sequence

Everything in §3–§5 of the build is in the tree and green locally. What is left needs a GPU.

**Read the ordering note first, and note what it cost.** `training.subset extract` reads
`training/full_run_state.json`, which only `FullRun.setup` writes — and `setup` runs as part of
`train.py --full`. The 3.2 handover said the extract could therefore not precede the first setup and
listed it as a step anyway; on the pod it died with `FileNotFoundError: training/full_run_state.json`.
`setup` is pure bookkeeping (it computes the cut and writes the state file plus a stub manifest, and
deletes nothing), so the fix is to run it on its own first — step 3b below — and let `train.py --full`
re-run it identically afterwards. The train would extract the subset itself, *before* the vocab rebuild,
which is the order W4's floor requires; the explicit extract exists only so the real games-per-player
histogram can be read and the floor confirmed before hours of training start.

```bash
cd /mnt/c/Projects/CourtVisionIQ
git checkout feature/version3.2 && git pull
source ~/cviq-venv/bin/activate
```

```bash
# 1. The suite, every tier. The local tier (901 tests) is already green on the dev box; this adds the
#    49 `slow` ones, which are the graph-heavy modules -- 87 tests including them took 29 minutes there.
python -m pytest -q -m ""
```

```bash
# 2. The priors sidecar. Already rebuilt on the dev box with W1's fix and W5's seventeen columns, so this
#    is only needed if data/ moved or was re-cleaned. Pure pandas, ~13 min, no GPU.
#    Gate: it must report 26,969 games covered. Before W1 it covered 1,277.
python -m player_priors
```

```bash
# 3. Set up the run under the 2011 floor. boundary_idx moves (it is a POSITION and 6,100 games left the
#    pool); the holdout game IDS do not, which is what keeps window 0 comparable to v2-run1..4.
python train.py --full --name version3.2 --batch-size 64 --rebuild-vocabs
```

**Before step 3, two things that cannot be recovered afterwards.**

```bash
# 3a. Vocab is APPEND-ONLY, so it must be deleted or the floor cannot shrink it. Encoder.freeze_all now
#     refuses a stale vocab by name, so this fails loudly rather than silently training a 2,152-row table.
rm encoder/vocabs/*.json
```

```bash
# 3b. Setup ALONE, because 3c reads the state file it writes. Same defaults --full would pass, so the
#     setup --full re-runs is identical. Expect the cut line and `holdout = 700 games`.
python -c "from training.full_run import FullRun; FullRun().setup(name='version3.2', batch_size=64)"
```

```bash
# 3c. Read the REAL games-per-player histogram and confirm the floor. Measurement 2 is an expected-exposure
#     estimate (subset ~5,374 games, median 34 games a player, 959 rows kept at a floor of 20); the
#     extraction is the number. `extract` prints the distribution and a kept/anonymous table at floors
#     10-30. Change MIN_PLAYER_SUBSET_GAMES now if the real histogram disagrees -- after the train it is a
#     retrain.
python -m training.subset extract
```

**Expected at the start of step 3**, in order: the priors coverage line, `rung 2 ... OFF for this pass`,
`subset heads [all twelve]`,
`vocabulary floor 20: N players aliased to anonymous slots`, then the vocab rebuild. If the floor line is
missing, `anon_slots.json` was not written and the floor is doing nothing — that is what
`require_player_floor` refuses, so it should not be possible to get past it silently.

```bash
# 4. Widen the holdout pool 100 -> 700. Passes extend_holdout's prefix guard untouched.
python train.py --extend-holdout
```

### The three passes

Everything past step 4 is **three passes, not nine steps**. Each answers a question the next one is
not entitled to ask, and the sim count is set by that question rather than carried forward.

| | question | shape | cost |
|---|---|---|---|
| Pass 1 | Did the retrain fix the behaviour probes? | 1 arm, 100 games × 20 sims | ~1.2 GPU-h |
| Pass 2 | Did rung 2 and the replay pass change anything at all? | 2 trains, no eval | ~4 GPU-h |
| Pass 3 | Which arm is better? | 2 arms, 700 games × 50 sims | ~43 GPU-h |

**Corrected 2026-09-20.** The version of this section committed with the build ran all three arms at
window 0 at `--monte-carlo 200`, in arm order, and read the comparison off them. That is 60,000
game-sims, ~37 GPU-h, spent where no comparison exists: the paired Brier SE at a 100-game window is
~0.0140 (`config.py`, measured on v2-run2 against v1.0 full4-s100), so every arm difference lands
inside one sigma by construction. §6.2 of `v3_2_direction.md` budgets **two** arms at 50 sims for one
final paired evaluation, and nothing at all for a three-arm window-0 sweep. **Sims buy probes; games
buy Brier** — the paired SE falls as 1/sqrt(games), while sims only remove the per-game Monte-Carlo
term.

#### Pass 1 — the diagnostic

```bash
# 5. Arm 1: the retrained bundle. Window 0 is the same 100 games v2-run1..4 scored.
python evaluate.py --model version3.2 --run v32-a1 --window 0 --monte-carlo 20 --procs auto
python -m reporting.state_probes results/version3.2/v32-a1 --seasons 2023
```

**20 sims is the right number here, not a budget cut.** `state_probes` scores the real side against
the WHOLE SEASON and never the window, so only the sim side scales with `--monte-carlo`: 100 games ×
20 sims puts ~2,400 events under the 3rd-foul probe (SE ~0.009) and ~80 under the 4th (SE ~0.05),
against gaps of 0.54 and 0.68. Both are 10-60× their standard error, and 50 sims resolves nothing
that 20 does not.

**Read the probes here, not Brier.** Brier moves least and that is structural: the winner is mostly
decided by pre-game team strength, which lives in the priors, not by how faithfully the fourth
quarter composes. The four numbers that matter are the 3rd- and 4th-foul benching rates (0.238 and
0.280 against a real 0.776 and 0.961), the 4th-foul event rate (9.5× too high), and the Q4 blowout
rotation ratio (0.838 against 0.525).

**At 20 sims, Brier and spread corr are biased rather than merely noisy.** The Monte-Carlo term
inflates Brier by ~0.010 and pulls spread corr from a signal 0.420 down to 0.375 (`config.py`). They
are not readable off this run, and never comparable to a run at a different sim count.

**If the probes have not moved, pass 2 is the wrong next spend.** Rung 2 and the replay pass are both
refinements to how a bundle is selected, and neither reaches the regime latent — which the
2026-09-19 subset train fitted at 1e-4 sd against a 0.01 init. A simulator that still never benches
anyone in foul trouble is not waiting on better checkpoint selection.

#### Pass 2 — the two retrains, no evaluation

```bash
# 6a. Copy the arm-1 bundle to its own NAME first. On a --model retrain, `--name` is a GUARD checked
#     against the state's version (train.py), NOT a destination, so arm 2 overwrites
#     artifacts/version3.2/event_time/ in place and arm 1 stops existing. Nothing compares a
#     manifest's `name` field to its directory, so the copy loads fine as version3.2-a1 — rewrite
#     the field anyway, so no report can attribute a run to the wrong bundle.
cp -r artifacts/version3.2 artifacts/version3.2-a1
python -c "import json,pathlib; p=pathlib.Path('artifacts/version3.2-a1/manifest.json'); m=json.loads(p.read_text()); m['name']='version3.2-a1'; p.write_text(json.dumps(m, indent=2))"
```

```bash
# 6b. Arm 2: rung 2. ROLLOUT_SELECTION is already True, setup wrote train_tail_game_ids, and
#     retrain_model now passes the score channel and records the selection (it did neither before
#     — see departure 13), so this second pass over the finished bundle scores rollouts.
python train.py --model event_time --name version3.2
```

**Read epoch 3, then decide.** The first `ROLLOUT_EVAL_EVERY` boundary is where this pass either
works or does not, and departure 16 is what happens when nobody looks. The rollout now announces
itself on the first evaluation and emits a progress line a minute:

```
[rollout] scoring 20 games x 10 sims every 3 epochs, compiled inference, budget 25 min/evaluation.
[rollout] epoch 3: 48/200 sims, 1.0 min elapsed, ~4 min projected
[rollout] epoch 3: score 8.1234 over 20 games x 10 sims in 4.2 min
```

`compiled inference` is the word that matters; `EAGER` means `ROLLOUT_COMPILED_INFERENCE` is off and
you are about to repeat the 9.5-hour run. The **projection** in the second line is readable inside
the first minute — if it says hours, kill it there rather than waiting. If an evaluation does run
past `ROLLOUT_EVAL_BUDGET_MIN`, the train aborts with `RolloutBudgetExceeded` and the measured
number, which is the intended outcome and not a crash to work around. **This is the first GPU
measurement of the compiled path in this repository**, so treat epoch 3's elapsed time as the result
of the step, and record it.

```bash
# 7. Arm 3: the KPI replay pass itself. One game in ten of the subset at ten sims each, every sim
#    scored against the real game per head, the ones that beat their siblings kept, then ONE
#    weighted pass. Writes a NEW bundle (version3.2-kpi) and leaves arm 2 intact.
python train.py --replay-pass
```

**Both gates are printed by the TRAINS, which is why this pass has no eval.** Arm 2's W8 number is
`epochs_disagree`, recorded during selection. Arm 3's first read is the per-head kept counts: a head
where no sim beat its siblings trains on an all-zero weight, which is a no-op pass that looks
identical to a clean one. An arm whose own train says it changed nothing does not go to pass 3.

#### Pass 3 — the comparison, once

Seven windows per arm at 50 sims, and only for the arms pass 2 says are real. Each window needs its
own `--run`: `--window` is pinned to the run dir on first use, so a later call at a different window
is an error rather than a re-slice.

```bash
# 8. One arm, seven windows. Repeat for the other arm, changing only --model and the run prefix:
#    arm 2 is `--model version3.2` (the overwritten bundle), arm 3 `--model version3.2-kpi`.
for w in 0 1 2 3 4 5 6; do
  python evaluate.py --model version3.2-a1 --run v32-a1-w$w --window $w --monte-carlo 50 --procs auto
done
```

**Hold the seed fixed across arms.** §8's "repeat runs use a different `--seed`" is for repeats of one
model; this is a model comparison, where a shared seed makes both arms face the same Monte-Carlo
draw. The harness refuses a mismatched pair, and the run log should say why.

**700 games at 50 sims is ~4.2 GB of play-by-play per arm** (0.12 MB per game-sim), so run
`harvest.py` beside the pool — a pod disk quota already killed one eval at 32/100 games.

```bash
# 9. The comparison. One positional per ARM, that arm's windows comma-separated inside it. 700 games
#    puts the readable threshold at 0.007 against an expected gain of 0.005-0.015; at 100 games the
#    threshold was 0.017 and the answer was never readable — see departure 15.
python -m reporting.ab_report \
  "$(printf 'results/version3.2-a1/v32-a1-w%s,' 0 1 2 3 4 5 6 | sed 's/,$//')" \
  "$(printf 'results/version3.2/v32-a2-w%s,' 0 1 2 3 4 5 6 | sed 's/,$//')" \
  --state training/full_run_state.json
```

### Gates, in order

- **W1** — `require_priors` reports 26,969 games. Already verified on the dev box.
- **W2–W5** — no gate of their own; read off `v3_2_direction.md` §7. Expect the cut to be neutral, and
  watch the rookie / role-shifter row (+10.9% / +24.0% against the season average), which is where the
  seven new scalars and the vocabulary floor either pay or do not. If it comes back **worse**, the first
  suspect is the four big heads moving onto the subset (departure in `config.py`, stated when taken).
- **W6** — context modulation must not worsen any probe. It starts as an exact identity, so any change is
  learned rather than an initialisation artifact.
- **W7** — scored on spread MAE.
- **W8** — records `epochs_disagree`. That number is rung 3's original firing condition; 3.2 builds rung 3
  anyway, so it is now evidence rather than a gate.
- **W11** — `python train.py --replay-pass` (departure 14). Must improve rollout CRPS and the foul and
  rotation probes on games the pass never saw, and must **not** worsen margin dispersion. If dispersion worsens, `ROLLOUT_SCORE_DISPERSION_WEIGHT` failed
  and the pass found the collapse-the-spread shortcut.

- **W12** — `python -m reporting.ab_report <arm dirs>` prints and records the paired test. The
  comparison refuses rather than reports when the arms are not comparable, which is the gate:
  a number that survives it is readable, and one that does not was never a finding.

### Known-open, deliberately

- **Nine changes land in one retrain** (the corpus cut, the vocabulary floor, seven scalars, two layers,
  plus 3.0's priors / running pace / regime latent / rung 1, none of which ever trained). A gain cannot be
  attributed among them. Accepted; §5.3's multi-scale time is held for 3.3 to stop it becoming ten.
- **`pf_36` and the foul objective land together**, so if the foul probes move, the two are confounded.
- **The replay pass has never run.** Every part of it is built and unit-tested, but the chain
  (simulate -> score -> corpus -> preprocess -> weighted pass) has only ever run against fixtures. The
  first real invocation is on a pod, and the first thing to read is the per-head kept counts: a head
  where no sim beat its siblings trains on an all-zero weight, which is a no-op pass rather than a
  change, and looks identical to a pass that did nothing wrong.
- **Batched-rollout throughput under the new layers is unmeasured.** Every figure in §6.2 scales with the
  37 GPU-min per 1,000 sims measured on the 3.0 graph at `ROLLOUT_BATCH_SIZE = 48`.
- **A name absent from the training corpus still encodes as `UNK`**, so two genuinely unseen players on
  one floor remain ambiguous. Pre-existing, and it only bites upcoming-season inference, where such a
  player has no priors either.
