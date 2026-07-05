# Next-Train Observations

> A running log of issues we've diagnosed and the changes we intend to make on the
> **next full train**. Nothing here is applied to the live code yet — this is the
> staging ground so we don't lose the reasoning between trains.

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

- **Status:** STAGED (for next train)

- **Open questions:**
  - Build the per-row candidate mask in preprocess vs. the logit-adjustment shortcut —
    which first?
  - Outgoing pick is currently identity-blind to stardom (the fatigue nudge pulls a
    star mid-stint as readily as a scrub). Does the masked-loss fix to the *incoming*
    head plus the stint head close the gap, or does the outgoing/actor head need the
    same treatment? (→ Entry 02)
  - Orthogonal data check: is the sim's available roster the **game-day active list**,
    not a season-level player set? Inactive players in the pool dilute star minutes
    regardless of any head change.

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

- **Status:** OPEN — decide whether to roll the masked loss into the actor head this
  train alongside the substitution head, or stage the actor head separately to isolate
  its effect.

- **Open questions:**
  - If the masked loss alone doesn't recover star role, the next lever is model
    **capacity/layers** (not external features) — revisit only after measuring the
    masked-loss train.
