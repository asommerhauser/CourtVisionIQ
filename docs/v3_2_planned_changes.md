# CourtVisionIQ 3.2 — Development Guide

> What 3.2 builds, why each piece exists, and the order we build it in. Written to be worked from:
> every workstream ends with **Next steps** naming concrete entry points, and names its **gate** from
> [`v3_2_direction.md`](v3_2_direction.md) §7.3.
>
> [`v3_2_direction.md`](v3_2_direction.md) is the *direction* and is not edited during the build.
> [`v3_2_progress.md`](v3_2_progress.md) is *state* — where this document and the code disagree,
> the progress tracker wins.

## Status

Branch `feature/version3.2`, off `feature/version3`. Nothing in this document is implemented yet.

3.0's code is built but **was never trained** — there is no `artifacts/version3`, and the WSL
handover in [`v3_progress.md`](v3_progress.md) is still pending. So the single retrain at the end of
this document carries **3.0 and 3.2 together**: priors, running pace, the regime latent and rung 1
from 3.0, plus the corpus cut, the vocabulary floor, seven new scalars and two new layers from 3.2.
Nine changes, one train, no attribution between them. That is deliberate and is the price of one
retrain; `v3_2_direction.md` §7 already assumes it, crediting `corr(home, away)` to "regime latent
(3.0)".

## Decisions taken before the build (2026-09-17)

On top of the twelve in `v3_2_direction.md` §Decisions:

1. **Below-floor players get per-game anonymous slot tokens, not one shared `UNK`** (W4). Measured:
   at a floor of 20 subset games, **32.7% of games roster two or more below-floor players**, which
   one `UNK` id cannot represent — and `game_available_mask` marks it available, so the player head
   can sample a man who is not identifiable.
2. **The replay estimator keeps only positive advantages** (W11). §4.2's signed sample weight is
   `-w·log p` with `w < 0`, which is minimized by driving that action's probability to zero and the
   loss to −∞. Filtering is bounded by construction and needs no trust region.
3. **Everything that does not need a GPU is verified locally, continuously** (WT). See below.

## The dev machine can run the suite

`v3_progress.md` standing rule 1 says `pytest` runs only on the GPU box, because "the dev box has no
GPU and `conftest.py` imports TF first". Rule 2 corrected half of that. This is the other half:

| measured on the Windows dev box, 2026-09-17 | |
|---|---|
| `import tensorflow` | **6.6 s**, version 2.20.0 |
| `tf.config.list_physical_devices('GPU')` | **`[]`** |
| `python -m pytest --collect-only -q` | **891 tests in 10.7 s** |

The real constraint is **no GPU**, not "no TensorFlow". Keras graph construction,
`get_config`/`from_config` round-trips, `load_weights` shape refusals, the layer-name contract and
the one-epoch tiny trains are all CPU work. **Rule 1 is amended: the local tier runs here after
every workstream.** Trains, rollouts and evals stay on WSL.

## The workstreams

Ordered as we build them. `retrain?` means the change must be in the graph or the data before the
one retrain, so it cannot be A/B'd separately afterwards.

| # | Workstream | Retrain? | Data | Model | Engine |
|---|---|---|---|---|---|
| [WT](#wt--the-test-tiers) | The test tiers | no | — | — | — |
| [W0](#w0--spec-and-tracker) | Spec and tracker | no | — | — | — |
| [W1](#w1--the-priors-join) | The priors join — **blocking** | yes | Light | — | — |
| [W2](#w2--corpus-cut-at-2008) | Corpus cut at 2008 | yes | Moderate | — | — |
| [W3](#w3--all-twelve-heads-on-the-subset) | All twelve heads on the subset | yes | Light | — | — |
| [W4](#w4--vocabulary-floor-and-per-game-anonymous-slots) | Vocabulary floor, anonymous slots | yes | Moderate | Moderate | Moderate |
| [W5](#w5--three-prior-stats-career-stage-season-deltas) | Three prior stats, career stage, deltas | yes | Moderate | Light | Light |
| [W6](#w6--context-modulation-film) | Context modulation (FiLM) | yes | — | Moderate | — |
| [W7](#w7--cross-roster-attention) | Cross-roster attention | yes | — | Moderate | — |
| [W8](#w8--the-rung-2-bridge) | The rung-2 bridge | no | — | Light | Moderate |
| [W9](#w9--per-head-kpi-metrics) | Per-head KPI metrics | no | — | — | Moderate |
| [W10](#w10--decision-logging-during-rollout) | Decision logging during rollout | no | — | — | Heavy |
| [W11](#w11--the-weighted-replay-pass) | The weighted replay pass | no | — | Heavy | Moderate |
| [W12](#w12--ab-harness-and-run-state-records) | A/B harness, run-state records | no | — | — | Light |

W1–W7 are one unit: nothing in them can be A/B'd separately without a second retrain. W8–W11 are
A/B-able against the retrained bundle, which is why §5.3's multi-scale time is held for 3.3 — it
keeps that comparison clean.

---

## WT — the test tiers

**What it is.** A `pytest.ini` with markers, and a fixture that stops the suite from mutating the
committed encoder vocabs.

**Why.** Two reasons. The suite has no tiering at all today — no `pytest.ini`, no markers — so
"run what is safe here" is not expressible. And the suite **rewrites `encoder/vocabs/*.json` and
`norm_stats.json`**, which is why standing rule 5 exists and why the 3.0 handover has to tell the
operator to `git checkout -- encoder/vocabs/` between the suite and a train. A fixture makes that
structural instead of a thing to remember, and it is the precondition for running the suite locally
at all.

**How it works.**

1. `pytest.ini`: `--strict-markers`, markers `gpu` (needs CUDA) and `slow` (over ~60 s), and
   `addopts = -m "not gpu"` so the bare invocation is the local tier. WSL runs `-m ""`.
2. An autouse session fixture in `tests/conftest.py` that snapshots `encoder/vocabs/` and restores it
   at the end of the session.
3. `tests/conftest.py`'s `_restore_dials` covers only `config._TUNING_KEYS`. The new corpus and
   architecture constants (`MIN_TRAIN_SEASON`, `MIN_PLAYER_SUBSET_GAMES`, `FILM_ENABLED`,
   `CROSS_ROSTER_ENABLED`) need their own restore list — `config.py:629` explains why a training knob
   does not belong in `_TUNING_KEYS`.

**Next steps.** `pytest.ini` (new), `tests/conftest.py`. Measure the local tier module by module and
mark `slow` from the measurement, not from a guess — `tests/test_model_persistence.py` runs
`preprocess` plus a one-epoch train three times over eight heads and is the first candidate.

**Gate.** None; it is infrastructure. Its output is a recorded local-tier pass count and wall time.

---

## W0 — spec and tracker

`docs/v3_2_direction.md` (committed verbatim, so the reasoning stays attributable and the document
is not edited during the build), this file, and `docs/v3_2_progress.md`.

---

## W1 — the priors join

**Blocking, and independent of everything else in 3.2.** It fixes a live defect in untrained 3.0 code.

**What it is.** One shared definition of the per-season `game_id` offset, so the priors sidecar and
the training corpus agree on what a game id means.

**Why.** `player_priors.build` reads each season CSV directly with `pd.read_csv`
(`player_priors.py:308`), so the sidecar is keyed by **per-season** `game_id`.
`data_loading.load_all_cleaned` shifts each file's ids by the running maximum
(`data_loading.py:52-55`), so the training rows carry **cumulative** ids. Measured over the real
corpus:

| | |
|---|---|
| games in the corpus, as `load_all_cleaned` keys them | 26,969 |
| game ids in `data/priors/` | 26,969 |
| **ids the two agree on** | **1,277** |

1,277 is season 2003 — the one file whose offset is zero. `merge_prior_features` raises on
`covered < total` (`models/prior_features.py:207-217`), so `train.py --full` aborts before the first
epoch. It is loud rather than silent, which is why nobody has hit it: the 3.0 handover never got past
step 3.

**How it works.** `data_loading.season_offsets(data_dir) -> dict[Path, int]` becomes the single
definition, read by both `load_all_cleaned` and `player_priors.build`. A rule read in two places
drifts — the same argument `apply_query_mask`'s docstring makes.

**Next steps.** `data_loading.py`, `player_priors.py:293-316`, `tests/test_player_priors.py`.

**Gate.** `require_priors` reports 26,969 and `merge_prior_features` does not raise. Locally: a
sidecar built over a multi-season fixture covers every id `load_all_cleaned` emits.

---

## W2 — corpus cut at 2008

**What it is.** `MIN_TRAIN_SEASON = 2008`. Seasons 2003–2007 leave the training pool.

**Why.** Not as a fix for identity — `v3_direction.md` §1f is right that it is not one — but as the
simplification that makes W4's vocabulary surgery possible. Old seasons are already discounted twice,
once in subset sampling and once in the loss, so a 2006 game contributes roughly three thousandths of
what a current game does. **Expect the cut to be neutral on every metric** (`v3_2_direction.md` §2.1).

**How it works — and the one way to get it wrong.** Game ids are **positional**: each season file's
ids are shifted by the running maximum, and the raw per-season ranges are not even ordered by season
(2016 holds 1313–2628, 2019 holds 1–1312). So cutting inside `cleaned_csvs` would renumber every
remaining game, invalidating `full_run_state.json`, `subset_games.json` and every
`results/<run>/holdout.json`, and breaking §7.2's claim that window 0 is byte-identical to the games
v2-run1..4 scored.

**The floor is therefore applied inside `load_all_cleaned`, after the offset walk.** Every id
survives; only `pos` and `boundary_idx` renumber. `cleaned_csvs` stays unfiltered, which also makes
§3.1's "cut the training games, never the sidecar" true **by construction** — `player_priors` and
`season_context` walk `cleaned_csvs` directly, so the priors chain still seeds 2008 from real 2007
production. That is the highest-consequence mistake in the direction document, and this removes the
opportunity to make it.

**The invariant to test.** `boundary` is `reg["pos"].min() + int(FINAL_SEASON_FRACTION * len(reg))`
over the last season's regular games (`training/full_run.py:83`), so removing games before it shifts
`reg["pos"].min()` by exactly the number removed and lands on the **same game**. Assert that the
holdout ids are unchanged across the cut.

**Note on the fixtures.** 18 test modules build 2003 corpora. `MIN_TRAIN_SEASON` therefore defaults
to `None` under `tests/conftest.py`, and only the tests that exercise the floor set it. No existing
fixture changes.

**Next steps.** `config.py`, `data_loading.py`, `training/chronology.py`, `training/full_run.py`
(record the floor as corpus provenance next to `boundary_idx`/`n_games`), `tests/test_chronology.py`,
`tests/test_full_run.py`.

**Gate.** None of its own; read off `v3_2_direction.md` §7.

---

## W3 — all twelve heads on the subset

**What it is.** `SUBSET_MODEL_KEYS` becomes all twelve heads, and the subset sampler's
coverage-completeness guarantee is retired.

**Why.** Consistency and compute: the four full-corpus heads drop from 26,267 games to ~5,200, taking
their epochs from ~420 s to ~84 s, which roughly pays for W6 and W7. **The risk is real and stated**
— thinning every player embedding lands on the identity axis, which is already the larger of the two
failures. Three things mitigate it (`v3_2_direction.md` §2.2), the third being W4's floor, which
removes the players who would have had the thinnest embeddings rather than leaving them in the table
under-trained.

Coverage-completeness goes because under a floor it is **inert**: anyone it rescues with a single
game falls below the floor anyway, and it drags old games into the sample for players who will be
anonymous regardless.

**How it works.** One config edit; the all-or-nothing conditional guard
(`models/pipeline.py:248-265`) is then satisfied trivially. Phase 1 of `build_subset`
(`training/subset.py:128-139`) and its `covered`/`freq` bookkeeping come out, along with the three
stats keys and the two prints. `extract()` gains a `player_subset_games` map — the input W4 reads —
which the sampler already computes at `training/subset.py:118-123`.

**Two consequences to record, not fix.** `event_time` owns the vocab build
(`models/event_time_model.py:341-349`) and is the head `refit_norm_stats` fits `norm_stats.json`
from. Both now run against the subset slice rather than the full train pool.

**Next steps.** `config.py:725`, `training/subset.py`, `training/full_run.py:236` (a hardcoded
`"full corpus -> event_time, player, substitution, sub_decision"`), and **`tests/test_subset.py`,
which is new** — the sampler has zero test coverage today, so nothing would have noticed the
guarantee being retired in either direction.

**Gate.** None of its own.

---

## W4 — vocabulary floor and per-game anonymous slots

**The highest-risk block in 3.2.**

**What it is.** The player vocabulary is rebuilt from players clearing a minimum number of games
**within the subset**; everyone below becomes an anonymous per-game slot.

**Why.** `v3_direction.md` §1f names the 2,152 × 192 embedding table as where the memorisation lives.
This shrinks it by removing long-retired players outright and the thin rows the floor catches — the
one capacity change in 3.2, and it goes *downward*.

**What the floor actually costs, measured.** Over the 2008+ priors at the live sampling rates:

| floor | vocab kept | minutes anonymous (all) | minutes anonymous (2021+) | games with ≥2 anonymous |
|---|---|---|---|---|
| 10 | 1,253 | 1.86% | 0.91% | 13.5% |
| **20** | **1,050** | **4.64%** | **2.02%** | **32.7%** |
| 30 | 911 | 8.15% | 3.30% | 49.0% |

The median player has ~31 expected subset games, so the document's "start at 20–30" sits at the
median — a far deeper cut than §3.2 implies. Two things follow.

**Order of operations, which matters.** Cut the corpus (W2) → carve the subset (W3) → count games per
player **within the subset** → build the vocabulary from those clearing the floor.

**Anonymous slots, not one `UNK`.** At a floor of 20, two or more below-floor players are rostered in
a third of games, up to a maximum of 13. One shared id cannot tell them apart, and the player and
substitution heads would spend probability on an unresolvable token. So `ANON_SLOTS = 16` tokens are
reserved and assigned **per game, by sorted name within that game**, so preprocess and inference
agree; `UNK` stays as the overflow. The embedding then becomes an honest "generic bench slot" and all
identity flows through the seventeen prior scalars — which is a truer version of §3.2's own argument
than one shared token would be.

**Why anonymity is affordable.** The prior scalars enter additively at
`x = emb + scalar_proj(stacked)` (`models/roster_set_encoder.py:137`), *before* the SAB layers. An
anonymous player loses his identity row and keeps his season-to-date production. For a deep-bench
player that is the right trade; the embedding row was mostly noise anyway.

**The one hard correctness item.** `game_available_mask` (`models/event_time_model.py:219-239`) sets
`mask[ids] = 1.0` over the ids in a game and zeroes `PAD` at `:238` — **but not `UNK`**. It must admit
exactly the anonymous tokens present in that game and zero the rest, or the heads can sample a player
who is not in the game.

**Also.** `Vocab` is append-only (`encoder/vocab.py:36-43`), so `encoder/vocabs/*.json` must be
deleted before the rebuild or the table never shrinks. (Noted in passing: the committed
`player_vocab.json` holds **2,152** entries, not the 2,153 `technical_specs.md:127` and
`methodology_whitepaper.md:107` both claim.)

**Next steps.** `config.py`, `encoder/encoder.py` (`SPECIALS`, `encode_roster`),
`models/event_time_model.py:219-239` and `:341-349`, `simulation/game_input.py`,
`simulation/input_cache.py`, `tests/test_encoder.py`, `tests/test_vocab.py`.

**Gate.** None of its own, but it is the first suspect if §7's rookie / role-shifter row comes back
*worse* rather than better.

---

## W5 — three prior stats, career stage, season deltas

**What it is.** `NUM_ROSTER_SCALARS` 14 → 21: `ft_pct`, `tp_pct`, `pf_36`, career stage, and
season-over-season deltas on `pts_36`, `min_pg`, `fga_36`.

**Why.** The model has a free-throw *attempt* rate and no *make* rate; `shot_type` emits
`corner3_l` / `wing3_r` / `top3` as distinct tokens and `shot_result` judges them without knowing
whether the player can shoot one; nothing says who actually fouls. And §1b's failure axis *is* career
games and role shift, which career stage and the deltas index directly. `stl_36` / `blk_36` stay out:
rare enough that the prior is mostly noise.

**Note `pf_36` does not fix the 9.6× over-production of fourth fouls.** That is a benching failure,
not an attribution one, and W8–W11 are what address it.

**How it works.** All seven ride the existing `prior_home` / `prior_away` planes, so
`NUM_ROSTER_SCALARS = 4 + N_PLAYER_PRIORS` (`models/rotation_features.py:633`) reaches 21 with **no
new named inputs** and no new plumbing through six heads' `INPUT_KEYS`, `_build_split`,
`append_*_batches`, `simulation/input_cache.py` or `simulation/game_input.py`. That is the change with
the smallest surface, and it is why the two scalar-order tests
(`tests/test_rotation_features.py:262`, `:275`) keep passing unedited — they assert the *relation*,
not a literal.

- `PlayerTotals.__slots__` gains `tpm`, `ftm`, `pf` (`player_priors.py:109`); `add()` and `rates()`
  follow. The box-score dataclass already carries all three (`simulation/box_score.py:56,58,66`), so
  nothing changes there. `ft_pct` / `tp_pct` follow `fg_pct`'s `_safe(num, den, default)`; `pf_36`
  follows `tov_36`.
- `PriorCarry` gains a `first_season` map and exposes the previous season's final rates.
  **A player with no previous season emits deltas of exactly 0.0 and career stage 0** — not a delta
  against the league mean, which is what `seed_for`'s fallback would otherwise hand every rookie.
- `PLAYER_PRIOR_KEYS` 10 → 17, appended in this order: `ft_pct, tp_pct, pf_36, career_stage,
  d_pts_36, d_min_pg, d_fga_36`. The order is load-bearing three ways — parquet column order, the
  `_NORM` index loop, and `ops.unstack` — and is pinned by `tests/test_player_priors.py:225`.
- `models/prior_features._NORM` gains seven entries, and `LEAGUE_DEFAULTS` seven (`pf_36`'s must be
  > 0, per `tests/test_player_priors.py:134`). The three rate priors are sized so their league
  default normalizes into [0.4, 1.6]. **The four delta and stage keys legitimately centre on 0.0 and
  cannot satisfy that band** — `test_normalization_puts_an_average_player_near_one` gains a
  `DELTA_KEYS` exclusion plus a companion assertion that a no-history delta reads exactly 0.0. The
  assertion's rationale is about *rate* priors, so this is a narrowing, not a weakening.

**A readable refusal instead of a shape error.** `models/manifest.feature_mismatch` is written and
tested but **has no caller** — `shell/actions.py` checks `ARCH_KEYS` and vocab sizes only. So today
the 14 → 21 change surfaces as a raw Keras kernel-shape error from `scalar_proj`. Wiring
`feature_mismatch` in beside `_check_arch` (`shell/actions.py:117-129`) is what that function exists
for.

**Next steps.** `player_priors.py`, `models/prior_features.py:58-79`, `shell/actions.py`,
`tests/test_player_priors.py` (including the fake box line at `:112-114`, which needs the three new
fields or `PlayerTotals.add` raises `AttributeError`), `tests/test_feature_manifest.py:32,59`
(hardcoded `14`). Rebuild the sidecar (~10 min). **Regenerate any saved `holdout_inputs.json`** —
those specs carry 10-float prior vectors and `pad_priors` will raise on them.

**Gate.** `v3_2_direction.md` §7's rookie / role-shifter row: +10.9% / +24.0% against the season
average, expected moderate, 50%.

---

## W6 — context modulation (FiLM)

**What it is.** One game-context vector drives a per-block scale and shift on the residual stream, so
season and the night shape every layer's computation instead of being one column at the bottom.

**Why.** Season, priors, rest and the regime latent all enter **once**, as columns in a wide concat
projected to `MODEL_DIM = 384`, and then have to survive six residual blocks on their own
(`models/backbone.py:167-169`). **Season is not under-weighted, it is under-plumbed.**

**Why not more width.** §5.4: heads reach the base rate in ~5 epochs and then memorise; the inputs do
not contain the answer. Widening now would add capacity to a model that already overfits, in the same
cycle the corpus shrinks roughly fivefold. FiLM is two vectors of 384 per block — negligible
parameters, no memorisation surface — and W4 removes parameters from precisely the table §1f blames.

**How it works.** `build_backbone` (`models/backbone.py:150`) gains a keyword-only
`film_context=None`. The context is a concat of the season embedding, the eight team priors and the
regime latent — all already `(B, SEQ, ·)`, so modulation is per-row with no broadcasting. One
`Dense(2·d_model)` per block emits scale and shift, applied to the **normalized branch** (after
`block{i}_ln1` and after `block{i}_ln2`), never to `x` itself — modulating `x` destroys the residual
identity path.

- `None` must reproduce today's graph byte-for-byte, the way `_attention_mask` already handles
  `LOCAL_ATTENTION_HEADS <= 0` (`models/backbone.py:142-147`). Assert it.
- `tests/test_backbone.py`'s `backbone_chain_names` / `SIDE_BRANCH_LAYERS` must list the new layers
  in graph order — the layer names are a persistence contract, and
  `test_head_carries_the_backbone_layer_names` runs the assertion against all eight heads.
- `FILM_ENABLED` goes into `models/manifest.ARCH_KEYS`: a flag that is off produces a graph missing
  layers, which is the `LOCAL_ATTENTION_*` failure class that list exists for.

**Next steps.** `models/backbone.py`, six production call sites (`event_time_model.py:605`,
`player_model.py:424`, `conditional_time_model.py:362`, `conditional_type_model.py:521`,
`substitution_model.py:601`, `sub_decision_model.py:305`), two test call sites
(`tests/test_backbone.py:69`, `tests/test_local_attention.py:44`), `models/manifest.py:43`,
`scripts/dump_layer_names.py` for the before/after check.

**Gate.** Must not worsen any probe (§7.3).

---

## W7 — cross-roster attention

**What it is.** One attention block where each roster attends over the other before pooling.

**Why.** The two rosters pass through one weight-tied encoder **independently** and meet only at the
fusion concat (`models/backbone.py:167`). There is no structural way to represent one lineup
*against* another. Shared weights, so it generalises rather than memorises.

**How it works — and why `layers/mab.py` is not a drop-in.** `MAB` is written correctly for
cross-attention but has never been used that way: `SAB` calls it with `Y = X` and `PMA` with seeds.
Three consequences. It has **no `compute_output_shape`**, so a second positional tensor in a
functional graph has nothing to infer from. It is **post-norm** while the backbone is pre-norm. And
nothing builds the `(B, 5, 5)` cross mask — the encoder builds only its own `(B, 1, N)`
(`models/roster_set_encoder.py:133`). So this lands as `layers/cross_roster.py`, a wrapper layer with
its own `build()` forcing child variable creation — the `RosterSetEncoder.build` precedent
(`models/roster_set_encoder.py:105-120`), and for the same reload reason.

**Next steps.** `layers/cross_roster.py` (new), the six heads' roster-encoder call sites,
`models/manifest.py:43` (`CROSS_ROSTER_ENABLED`), **`tests/test_cross_roster.py` (new)** — there is
no test for `MAB`, `SAB` or `PMA` at all today.

**Gate.** Scored on spread MAE (§7.3).

---

## W8 — the rung-2 bridge

**What it is.** The missing wiring that lets checkpoint selection on rollout metrics actually run.

**Why.** `models/rollout_selection.py` is written and tested — the policy, the callback, the scoring
formula, the EarlyStopping ordering, the `epochs_disagree` record. It is one day of work and the
cheapest item in 3.2. Three things stop it:

1. **`ROLLOUT_SELECTION = False`.**
2. **Nothing constructs `rollout_score_fn`**, which `models/event_time_model.py:865` expects. The
   parameter exists on `EventTimeModel.train` only, no caller passes it,
   `models/pipeline.run_stage._train` has no channel for it, and `self._checkpoint_selection` is
   written at `:887` and never read.
3. **`eval_game_ids` reads `state["train_tail_game_ids"]`, which nothing writes**
   (`models/rollout_selection.py:70`), so the selector's game set is unreachable from a real run
   state. `boundary_idx` is read at `:67` and then unused.

**Also.** `rollout_score` expects `probes["rows"]` and nothing produces that shape —
`reporting/state_probes.compare()` returns `{"sim", "real"}` and `_rows_for_frame` returns a bare
list. And `ROLLOUT_EVAL_SIMS` is defined and never read.

**How it works.** A new `models/rollout_bridge.py` builds `score_fn(epoch) -> float | None`: roll out
`ROLLOUT_EVAL_GAMES × ROLLOUT_EVAL_SIMS` from the training-era tail through
`simulation.evaluation.simulate_games`, aggregate, adapt the probes to `{"rows": [...]}`, and return
`rollout_score(aggregate, probes)`. Note `_rows_for_frame` mixes rates, seconds, ratios and per-game
counts into one list, so the behaviour term is a mean over heterogeneous relative gaps — weight it
deliberately rather than feeding it whole.

**Scoped to `event_time`**, matching the 3.0 handover's step 8. Widening to the other eleven heads is
3.3.

**Never from a holdout window.** Selecting against the holdout turns the report into a training
metric and nothing downstream would look wrong.

**Next steps.** `models/rollout_bridge.py` (new), `training/full_run.py` (write
`train_tail_game_ids`; pass the callable), `models/pipeline.py:215-228`, `config.py:78`,
`tests/test_rollout_selection.py`.

**Gate.** Records `epochs_disagree` — the number `v3_direction.md` §6 step 6 fires on.

---

## W9 — per-head KPI metrics

**What it is.** Each head scored on the metric closest to its own output, instead of one shared
scalar.

**Why.** A shared scalar punishes a head that did its job for another head's failure. That
mis-assignment is why the generic estimator needs hundreds of steps; matching each head to its own
metric is the largest available variance reduction and it costs nothing.

**How it works.** Against the existing `_aggregate` (`simulation/eval_metrics.py:360-443`), which
already carries Brier, score-Brier, and per-player MAE and reliability for every stat in
`_BOX_ACCURACY_STATS` — so this is a weighting change, not new machinery.

`event_time` is scored on the histogram of its own eight output tokens. Six map straight onto
`BOX_STATS`; **substitution and timeout counts need a small counter off the play-by-play, on both the
sim and the real side**. Three exclusions are deliberate: free throws drop out (there is no
free-throw token — attempts are produced by the rules engine downstream of a shooting foul, so
scoring the head on them charges it for a rule it does not control), steals stay with
`turnover_type`, and substitutions are counted rather than converted to minutes.

**Rate-normalisation is the principle behind the two least obvious rows.** `shot_result` gets eFG%
rather than team points, because dividing by attempts stops a too-fast simulator being charged to the
shooting head. `shot_type` must **not** get eFG% — it controls the mix, and the cheapest way to win
an efficiency metric by changing the mix is to shoot more threes, which eFG counts at 1.5. Same
reason `player` gets **shares** of its team's total rather than raw counts.

**Three implementation rules that decide whether this works** (§4.4):

1. **Win probability from `simulation.stats.score_win_prob`** (the Gaussian margin approximation),
   never from counting which side won — from one sim the winner is 0/1 and the Brier term is a coin
   flip.
2. **Every stat normalized by a frozen cross-game standard deviation**, in the style of the existing
   `_NORM` blocks. *Correction to §4.4:* the second half of this rule is already satisfied and needs
   no work — `_BOX_ACCURACY_STATS` (`simulation/eval_metrics.py:30`) carries derived `minutes` and no
   `seconds` at all. There is also no "`eval_metrics` already says this in its headline block"
   statement in the code; the nearest comments are about the shot clock and about why `minutes` is
   derived.
3. **The dispersion guard stays on.** MAE is minimised by collapsing the spread, and a head that
   fixes the box by killing variance destroys the joint structure the whole thesis rests on.
   `ROLLOUT_SCORE_DISPERSION_WEIGHT` is not optional under a KPI objective.

**Next steps.** `simulation/eval_metrics.py`, `models/rollout_selection.py:80-113`,
`reporting/state_probes.py`, `tests/test_standing_metrics.py`, `tests/test_rollout_selection.py`.

**Gate.** Feeds W11's.

---

## W10 — decision logging during rollout

**What it is.** A record of what each head actually sampled, per queried position, per sim.

**Why.** It is the input the replay pass trains on.

**How it works — log a thin index, not tensors.** `base_inputs()` returns a dict **memoized per row
and shared across the ~5 head calls at one position** (`simulation/input_cache.py:253-268`), and the
prior columns alone are megabytes per position at `SEQ = 600`. Storing context per queried position
across thousands of game-sims is not feasible, and retaining the dict would alias it.

So the log records `(game_id, sim_index, history_position, head, output_name, sampled_token)`, and
**context is re-derived by replaying the sim's own play-by-play through the normal preprocess**.
`_PbpSink` (`simulation/stage_eval.py:141`) already writes each sim's play-by-play in the cleaned-row
schema, and `reporting/state_probes.py` already runs `LineupScan` and `GameStateScan` over those
files — which is the proof the round trip works.

**Where it attaches.** The only stochastic picks over head logits are `_masked_sample`
(`simulation/game_simulator.py:413`) and `predict_sub_count` (`:658`); `_head_logits` (`:574`) knows
the head and the decision position. Attach in the `predict_*` wrappers, where head, position and pick
are all in hand.

**Per-`_WorkerSim` buffers, flushed in `on_complete`.** Up to 48 slots call `_infer` concurrently
from worker threads (`simulation/batched_rollout.py:64-80`); a shared list would need a lock and would
lose the sim identity.

**Write the log beside `playbyplay/`, never inside it** — `harvest.py` prunes that directory.

**A note on which positions count.** `apply_query_mask` (`models/game_state_features.py:557`) is
called by **2 of 12 heads**, so §4.2's "~68.5% of rows" is the event/time head's kept share. The other
ten heads' queried sets are event-token-gated (`next_event == <token>`) and much sparser. The log is
therefore defined per head from its own sampling call, not from the query mask.

**Next steps.** `simulation/game_simulator.py`, `simulation/batched_rollout.py`,
`simulation/stage_eval.py`, `tests/test_batched_rollout.py`.

---

## W11 — the weighted replay pass

**What it is.** W4 rung 3, in the cheap form: the same score-function estimator with the cost moved.
**3.2 GPU-hours, against the ~160 `v3_direction.md` priced.**

**Why the gradient cannot flow through a rollout.** `simulation.controller.GameController` is
pure-Python rules on worker threads, and every sample goes through numpy. The TensorFlow graph is
discarded at each step. There is no reparameterisation trick and no backprop through the game; only
score-function estimators exist.

**Why it is built unconditionally.** `v3_direction.md` made rung 3 conditional on
`epochs_disagree`. The §1 measurement makes composition the headline failure, and rung 3 is the only
item in the programme that puts a gradient on it. **No next-step loss can see this**, because each
individual prediction is roughly right and it is the composition over hundreds of steps that is wrong.

**How it works.**

1. Score each finished sim against the real game it simulated, using W9's per-head metrics.
2. **Advantage** = the mean of the other nine sibling sims of the same game, minus this one — signed
   so that positive means "more realistic than its siblings". Same matchup, same date, so the
   baseline is tightly matched and most of the variance cancels. `simulate_games`' `job_owner`
   already gives the `(game_index, sim_index)` grouping (`simulation/evaluation.py:188-199`).
3. **Keep only positive advantages.** *Departure from §4.2, taken before the build:* the document
   specifies a signed sample weight, but `-w·log p` with `w < 0` is minimized by driving that
   action's probability to zero and the loss to −∞. Filtering is bounded by construction, introduces
   no hyper-parameters, and fits a single 3.2 GPU-hour pass that has no budget for a search. It never
   pushes away from anything; it reinforces what worked.
4. **One weighted pass**, reusing `models.train_steps.build_trainer` rather than adding a second
   training path. Per-head sample weights are already a plain dict pass-through
   (`models/train_steps.py:160-183`), so nothing there changes. Call it with **`n_games=0`** so it
   returns `inner` unchanged — a sim has no valid `game_index` row and a regime lookup on one would
   be meaningless.

**The sims are the cost, and they are shared: one batch of rollouts updates all twelve heads**, each
filtering to its own decisions. Twelve label/mask/weight bundles come from one decision log; heads
4–10 share a single `cond_*.npz` and one event-token mask convention, and `sub_decision` shares one
mask array across its two outputs (`models/sub_decision_model.py:346-349`).

**Cadence and sampling (§4.5).** One pass, **after** the main train, over one game in ten of the
subset, ten sims per game — ~520 games, 5,200 game-sims. After rather than during, because mid-train
the other eleven heads are still moving and a scored rollout came from a bundle that stops existing a
few epochs later. Ten sims of the *same* game, not one sim of ten games, because the sibling set is
what makes the leave-one-out baseline work. From the **subset**, never a flat stride over the corpus
(which would be era-neutral and fine-tune `shot_result` toward the old game) and **never from a
holdout window**. **Do not repeat the pass per epoch** — one pass is 3.2 GPU-hours; the same pass
every epoch of a thirty-epoch stage is ~96, worse than the naive form this design exists to avoid.

**Next steps.** `models/rollout_bridge.py`, a new replay module, `models/pipeline.py`,
`training/full_run.py`.

**Gate.** Must improve rollout CRPS, and the foul and rotation probes, on games the fine-tune never
saw. **Must not worsen margin dispersion** — if it does, the guard in §4.4 failed.

---

## W12 — A/B harness and run-state records

Three arms on the same window with a **fixed seed** — the one case where §8's "repeat runs use a
different `--seed`" does not apply, because this is a model comparison rather than an independent
Monte-Carlo draw, and the run log should say so. The arms: the retrained bundle, + rung 2, + the KPI
pass. The KPI phase and its filtered-sim counts are recorded in the run state next to
`checkpoint_selection`.

**Read it at 700 games, paired.** The per-game Brier sd measured on run4 is **0.174**, not the 0.133
§8 assumed, so the 2 SE detection threshold is 0.017 paired at n = 100 and 0.007 paired at n = 700.
An expected gain of 0.005–0.015 is borderline at 100 games and clears the floor at 700.
`FINAL_HOLDOUT_GAMES = 700` already exists for this, and window 0 is byte-identical to what
v2-run1..4 scored — **provided W2's cut preserves game ids**, which is why it is placed where it is.

---

## What 3.2 does not do

- **Multi-scale time** (§5.3). Held for 3.3, and not on cost — 4–6 days — but on **attribution**: the
  rollout objective is the largest unknown in the programme and deserves an A/B with nothing standing
  in front of it. If the Q4 rotation probe moves under W11 and then plateaus, that is the evidence
  multi-scale time is the limit, and 3.3 gets a clean before-and-after. It also would not fix the
  foul-benching gap, which is an objective failure, not a scale failure.
- **Player-by-season embeddings.** Each slice trains on at most 80 games — the memorisation problem
  in a new hat.
- **Broadcasting the season token into the roster slots.** Cheap and harmless, and carries
  information the backbone already has.
- **Widening `MODEL_DIM` / `NUM_LAYERS`.** §5.4, and see W6.
- **Changing the recency weighting.** `RECENCY_WEIGHTING = True`, halflife 3.0, floor 0.05. The
  double discount is the reason the cutoff is neutral; changing the weighting at the same time would
  confound that.
- **`stl_36` / `blk_36`, and `fg_pct`.** The first two are noise as priors; `fg_pct` mixes shot
  difficulty with shooting skill — noted, not changed.
