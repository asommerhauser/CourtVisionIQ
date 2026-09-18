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

| branch | workstream | retrain? | state |
|---|---|---|---|
| `v3.2/w0-spec` | direction, build guide, this tracker | no | **built** |
| `v3.2/test-tiers` | WT: `pytest.ini` markers, two conftest fixtures | no | **built**, 49-test subset green |
| `v3.2/priors-join` | W1: one definition of the `game_id` numbering | yes (sidecar rebuild) | **built**, 153 tests green |
| `v3.2/corpus-cut` | W2: `MIN_TRAIN_SEASON = 2008`, applied to rows | yes | **built**, 165 tests green |
| `v3.2/subset-all-heads` | W3: all twelve heads on the subset; coverage retired | yes | **built**, 124 tests green |

**W1 verified two ways.** `tests/test_player_priors.py` gains three tests that build a real sidecar
over two season files carrying *the same raw ids* -- the case that used to collapse -- and assert the
corpus is covered, that the causal seed chain still crosses the season boundary, and that a
`--seasons` partial rebuild still lands on corpus ids. And `season_offsets("./data")` was compared
against the offsets computed by replicating the original inline walk on the real 21-season corpus:
**all 21 match exactly, so the numbering is unchanged.** That is what keeps the stored
`holdout_game_ids` valid and window 0 comparable to v2-run1..4.

**The on-disk sidecar is still stale** -- it was built with raw ids. The rebuild is deferred to W5,
which changes the columns anyway, so it is paid once.

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

### 2. What the vocabulary floor actually costs

Computed from the 559k-row priors sidecar for seasons 2008+, weighting each season by the live subset
sampling rate (`SUBSET_RECENT_SEASON_RATES = (1.0, 0.70, 0.50)`,
`SUBSET_RECENCY_HALFLIFE_SEASONS = 5.0`) and halving the newest season for the `FINAL_SEASON_FRACTION`
cut. These are expected exposures, not a real extraction — the real histogram comes from
`training/subset_games.json` after W3.

| | |
|---|---|
| train pool, 2008+ | ~19,800 games |
| expected subset size | ~5,750 games (against ~6,110 uncut — so ~360 games leave, not §3.1's ~300) |
| players with any subset exposure | 1,797 |
| **median expected subset games per player** | **~31** |

So §3.2's "start at 20–30" sits **at the median**, which is a far deeper cut than the document
implies.

| floor | vocab kept | → anonymous | minutes anonymous (all) | minutes anonymous (2021+) | games with ≥2 anonymous |
|---|---|---|---|---|---|
| 5 | 1,435 | 362 | 0.72% | 0.29% | — |
| 10 | 1,253 | 544 | 1.86% | 0.91% | 13.5% |
| 15 | 1,152 | 645 | 3.00% | 1.30% | — |
| **20** | **1,050** | **747** | **4.64%** | **2.02%** | **32.7%** |
| 25 | 981 | 816 | 6.21% | 2.57% | — |
| 30 | 911 | 886 | 8.15% | 3.30% | 49.0% |

**The minutes cost is affordable; the collision count is what forced a design change.** At a floor of
20, two or more below-floor players are rostered in **32.7% of games** (mean 1.39, p90 4, max 13).
A single shared `UNK` id cannot tell them apart, and `game_available_mask`
(`models/event_time_model.py:219-239`) zeroes `PAD` at `:238` but **not** `UNK`, so the player and
substitution heads can spend probability mass on an unresolvable token. Hence W4's per-game anonymous
slots (`ANON_SLOTS = 16`, covering the measured maximum of 13, with `UNK` as overflow).

### 3. The suite runs here, but some modules are slow on CPU rather than instant

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

### 2. Below-floor players get per-game anonymous slots, not one `UNK`

§3.2 maps them all to `UNK`. Measurement 2 shows that is ambiguous in a third of games. Sixteen
reserved tokens assigned per game by sorted name cost 16 embedding rows and make the token resolvable,
and the embedding becomes an honest "generic bench slot" with all identity flowing through the
seventeen prior scalars — which is a truer reading of §3.2's own argument than one shared id.

### 3. Coverage-completeness is retired, and the sampler gets its first tests

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

### 4. The replay estimator keeps only positive advantages

§4.2: "the advantage is the sample weight. Positive advantage reinforces those choices, negative makes
them less likely." A negative weight on cross-entropy is `-w·log p` with `w < 0`, which is **minimized
by driving that action's probability to zero and the loss to −∞** — unbounded. Signed REINFORCE via
weighted cross-entropy needs advantage clipping plus a KL leash to stay stable, which is three
constants that cannot be tuned inside a single 3.2 GPU-hour pass. Filtering to the sims that beat
their nine siblings is bounded by construction, needs no hyper-parameters, and never pushes away from
anything.

### 5. Two §4 citations corrected rather than implemented

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

### 6. Rung 2 stays scoped to `event_time`

§4.6 prices the bridge at one day. `rollout_score_fn` exists only on `EventTimeModel.train`
(`models/event_time_model.py:711`); the other five head classes have the identical signature without
it, and `models/pipeline.run_stage._train` has no channel to pass one. Widening to all twelve heads is
a design change, not a bridge, and is 3.3.

---

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

Filled in as the build lands. The shape it will take:

```bash
cd /mnt/c/Projects/CourtVisionIQ
git checkout feature/version3.2 && git pull
source ~/cviq-venv/bin/activate
python -m pytest -q -m ""                 # every tier; the local tier is already green here
python -m player_priors                   # W1 + W5: rebuild the sidecar (~10 min, no GPU)
python -m training.subset extract         # W2 + W3: the cut subset, and the per-player game counts
rm encoder/vocabs/*.json                  # W4: Vocab is append-only; it must be deleted to shrink
python train.py --full --name version3.2 --batch-size 64 --rebuild-vocabs
python train.py --extend-holdout          # pool 100 -> 700
python evaluate.py --model version3.2 --run v32-run1 --window 0 --monte-carlo 200 --procs auto
python -m reporting.state_probes results/version3.2/v32-run1 --seasons 2023
```

**Check the real games-per-player histogram from `training/subset_games.json` before the train** and
confirm `MIN_PLAYER_SUBSET_GAMES`. Measurement 2 is an expected-exposure estimate; the extraction is
the number.
