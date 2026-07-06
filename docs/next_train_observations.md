# Next-Train Observations

> A running log of issues we've diagnosed and the changes made across trains. Entries
> start as OPEN/STAGED reasoning and are marked `APPLIED` (with commit) once they land.
> Entries 01–04 are the **Train 2.5** scope (in the working tree, pending the cloud run).

Each entry records: what we observed, the root cause, the planned change, why, and
status. Keep entries append-only; mark a change `APPLIED` (with commit) once it lands
in a train.

---

## Entry format

```
### NN. <short title>
- **Observed:** the symptom in simulated output.
- **Root cause:** the mechanism in the code/training that produces it.
- **Planned change:** what we'll change for the next train.
- **Why:** the principle behind it.
- **Status:** OPEN / STAGED / APPLIED (commit)
- **Open questions:** anything unresolved.
```

---

### 01. Star players get too few minutes — candidate-masked substitution loss

- **Observed:** Stars (especially injury-prone stars who miss chunks of a season)
  are under-played in simulated games — they don't get re-subbed in and don't reach
  their real minutes load.

- **Root cause:** The substitution head softmaxes over the **entire player vocab**
  and only restricts to the legal bench at *inference* (`_masked_sample` in
  `simulation/game_simulator.py` slices + renormalizes the full-vocab logits). The
  training loss (`SparseCategoricalCrossentropy` over `player_vocab_size`,
  `models/substitution_model.py`) never sees that restriction. So the relative logits
  among bench players still carry each player's **global appearance frequency**. A
  star who missed games appears in fewer rows → lower logit everywhere → he loses the
  bench competition to durable role players even when healthy and available. The
  decision is not made *relative to who was actually available*.

- **Planned change:** Train the substitution head with the **same legality mask the
  simulator applies at inference** — i.e. restrict the softmax denominator to the
  legal candidate set (the actual bench at that row: full roster minus on-court). The
  candidate set is recoverable in preprocess from the per-row rosters. The model then
  learns `P(incoming | available bench)` instead of `P(incoming | whole league)`, and
  the global frequency prior cancels because every player is only ever scored against
  the teammates genuinely selectable alongside him.
  - Cheaper approximation if the masked denominator is awkward to build:
    balanced-softmax / logit-adjustment — subtract `log(appearance_freq)` per player
    from the logits in the loss.

- **Why:** **Train/inference legality parity.** At inference we constrain to legal
  candidates; at train we let the model spend probability mass on players who weren't
  even on the floor. Closing that gap is the general principle (see Entry 02). This
  also keeps the fix learned purely from play-by-play — no hand-fed role feature —
  because removing the popularity shortcut *forces* the embedding to carry a player's
  role from context (the company he keeps + his in-game behavior) rather than from his
  row count.

- **Status:** APPLIED (`7a067de`, Train 2.5) — but via **in-graph** candidate masking,
  not a per-row preprocess mask. `OnCourtCandidateMask` (`models/event_time_model.py`)
  builds the legal set from the roster inputs already in the graph and biases off-set
  logits to −1e9 in both train and inference. The substitution head's denominator is
  `avail_mask` minus the on-court ten (the legal bench).

- **Resolved open questions:**
  - *Per-row mask vs. logit-adjustment:* neither — the in-graph roster-derived mask
    avoids a ~130 GB per-row npz and the appearance-frequency shortcut entirely.
  - *Does the incoming fix suffice, or does the actor head need it too?* The actor head
    got the same treatment this train (→ Entry 02, also APPLIED).
  - *Game-day active list vs. season pool?* **Answered by code inspection:** the eval's
    availability set is the union of per-row lineups (`GameInput.home_roster/away_roster`),
    i.e. the game-day actives minus DNPs — not a diluted season pool. No dilution bug.

---

### 02. Candidate-masked loss as a general principle (which heads?)

- **Observed (conceptual):** Entry 01's fix feels like something we'd want
  *everywhere*, not just the substitution head.

- **Where it applies:** Any head that, at inference, masks a **small varying subset**
  out of a **large** vocab — i.e. the **player-vocab heads**:
  - substitution / incoming (Entry 01),
  - the actor/player head (shooter, rebounder, fouler, etc.) — candidates are the
    on-court five of one team out of the whole player vocab, masked only at inference,
    so it carries the same appearance-frequency prior (smaller magnitude than subs,
    since shot attempts correlate with global usage more than sub timing does, but the
    injury-prone-star deflation is the same bug).

- **Where it does *not* apply:** The small fixed-vocab heads — `shot_type` (2pt/3pt),
  `shot_result` (made/missed/blocked), `foul_type`, `rebound_type`, etc. There the
  inference candidate set is essentially the *whole meaningful vocab*, so the mask is a
  no-op, and we actively *want* the learned base rate (the real make %, the real ~25%
  offensive-rebound share). Don't mask these.

- **Principle:** Train-time legality masking should **match** inference-time legality
  masking, per head. Apply the candidate-restricted loss wherever the two differ
  (the player-vocab heads); leave the full softmax where inference already uses ~the
  full vocab.

- **Status:** APPLIED (`7a067de`, Train 2.5) — rolled into the actor head **this** train
  alongside the substitution head. Actor-head denominator = the on-court ten (both teams;
  the head also serves the fouler pool); inference still restricts to the acting five, a
  far smaller train/inference gap than the old ~25-player game pool. A `target_on_court`
  loss-mask guard zeroes rows whose target isn't in the candidate set (e.g. the pre-`end`
  sentinel) so a −1e9 target logit can't produce NaN loss.

- **Open questions:**
  - The `PLAYER_TEMPERATURE=2.0` flattening was a *band-aid* for the un-normalized actor
    softmax. After this train, retune it **down** (expect ~1.0–1.3) from the lineup probe —
    a properly normalized head shouldn't need heavy flattening, and the flattening was
    itself destroying star-vs-role usage separation (a driver of the baseline-MAE loss).
  - If masked loss alone doesn't recover star role, the next lever is model
    **capacity/layers** (not external features) — revisit only after measuring this train.

---

### 03. Score-blindness — game-state features into every head

- **Observed:** Sims drift as unanchored random walks. Eval: spread bias **+3.48**, margin
  correlation **0.197**, FTA **−3.9** / PF **−3.6** per team-game (no catch-up/bonus fouling),
  and the model loses to the season-to-date baseline on **all 12** per-player counting stats.

- **Root cause:** `GameController` tracks score / possession / period / team fouls, but none
  of it entered `GameSimulator.build_model_inputs()`. The fusion saw event tokens, rosters,
  absolute game time, and season context — never the *state of the game*. A transformer can't
  reconstruct the running score by exact arithmetic over ~500 event tokens, so score-conditional
  behavior (leaders sitting on leads, trailers fouling/shooting threes, garbage-time margin
  compression, the bonus manufacturing FTs) was unlearnable.

- **Change:** New `models/game_state_features.py` (mirrors `season_features.py`) feeds six
  per-row scalars — `score_diff`, `score_total`, `period_idx`, `period_time_left`,
  `team_fouls_home/away` — `Dense(16)`-projected into the fusion of all six heads. One pure
  `derive_game_state()` scan feeds BOTH preprocess and the simulator, so train/inference values
  are identical by construction (fixed-constant normalization, no new `norm_stats` keys).

- **Why:** Same philosophy as feeding rest days explicitly and the controller enforcing rules:
  hand the model the state, spend capacity on the conditional behavior, not on bookkeeping.

- **Status:** APPLIED (`827f3a1`, Train 2.5).

- **Open questions:**
  - Deferred to a later train (roster-parallel plumbing): **possession** indicator and
    **per-player foul counts** for the on-court ten (foul-trouble anticipation).
  - After the train, re-measure the FTA/PF deficit; residual is a `EVENT_BIAS["foul"]` dial,
    not a retrain.

---

### 04. Post-train inference dials (no retrain) + eval resolution

- **`EVENT_BIAS` / `TYPE_BIAS`** (`config.py`, APPLIED `7fb87b7`): per-event and per-head-token
  logit offsets threaded through `_sample_event` / `predict_type` (same hook as `SHOT_RESULT_BIAS`).
  Default `{}` = no-op. Tune post-train from the eval team-bias table (expect `foul` +, `assist` −,
  `turnover`/`steal` −) once the game-state features have shifted the base event mix.
- **`STAGE_SIMS` 11 → 21** (`config.py`, APPLIED `b806fa7`): the eval averages sims before scoring,
  so more sims cut box-score sampling-noise MAE and halve the win-vote quantization (1/11 → 1/21)
  that flattered Brier.
- **Status:** APPLIED. These are the levers to reach for **after** the weights come back, before
  any thought of another train.
