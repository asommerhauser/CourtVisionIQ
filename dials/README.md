# Dial packages

A dial package is a JSON object of `DIAL -> value` consumed by `evaluate.py --dials FILE`
(and `cviq> load ... --dials`). It is the only thing that pins every process of a sharded or
pooled run to one tuning: a child re-reads `config.py` from disk, so a parent that tuned at
runtime must hand the values over explicitly. `eval_pool.run_procs` writes the live values to
`<run_dir>/dials.json` and `assert_one_tuning` checks after the fact.

## `v2-run1.json` -- the first 2.0 eval

The rule this package follows: **a dial that corrected a defect 2.0 fixed at the source goes to
neutral; a dial that is not a fit stays.** Everything in `config.py` today was fitted against
v1.0's heads, and every one of those comments already says "re-fit from zero after the 2.0 train"
(`docs/v2_review_2026-09-09.md` sec 6.2 orders it pace -> event mix -> shot result -> home court).

Carrying the v1.0 package into run 1 would not be a neutral choice, it would double-correct.
`TYPE_BIAS.foul_type["loose ball"] = +1.2` existed because loose-ball fouls were near-absent
(0.06 vs 2.70/game) -- a consequence of the foul-type masking bug that workstream 1 fixed. Keep
the offset on top of the fix and the token over-produces. `EVENT_BIAS.foul = +0.11` is the same
shape: it corrected an event mix distorted by continuation rows that workstream 12 masked out.

| Dial | v1.0 | run 1 | Why |
|---|---|---|---|
| `SHOT_RESULT_BIAS` | `{made:+0.40, blocked:-0.15}` | `{}` | 15 zone tokens replace the one global make-rate knob (sec 4.1); refit per zone from the raw rates |
| `SHOT_RESULT_BIAS_BY_ZONE` | `{}` | `{}` | was unusable in v1.0 -- no zones existed to key it on |
| `EVENT_BIAS` | `{foul:+0.11, turnover:-0.08}` | `{}` | event head no longer trains on continuation rows |
| `TYPE_BIAS` | 5 foul types + 3 heads | `{}` | side-aware masking; `assist_type` keys cannot match the zone tokens at all |
| `DELTA_TIME_SCALE` | 0.97 | 1.0 | fit to v1.0 pace; 2.0 re-counted possessions (FT trip unification, the 12% over-count). Pace is first in the refit order, so measure it raw |
| `SUB_INCOMING_TEMPERATURE` | 0.45 | 1.0 | fit to deep-bench over-play under the **stint scheduler**, which workstream 11 deleted. Holding it would mask whether the sub_decision head learned the rotation |

Held, because none of these is a fit against v1.0's heads:

- `FOUL_OUT_LIMIT` 6 -- an NBA rule.
- `MAX_DELTA` 60.0, `SUB_MAX_GAP_SECONDS` 600.0 -- clamps and safety nets, not calibration.
- `HOME_COURT_SHOT_BIAS` 0.055 -- supplies information the model structurally lacks. The rollout
  is home/away symmetric and 2.0 added no team identity (sec 6.4), so nothing in this train
  changed what this dial stands in for.
- `PLAYER_TEMPERATURE` 2.0 -- the one deliberate exception, and the one to re-probe next. Raw, the
  head puts 0.55-0.85 of its mass on a single player once restricted to the on-court five, which
  produces 50/15/14 lines that pollute points, rebounds and assists at once. That pathology was
  measured, it is severe, and 2.0 did not target it. Run 1 is not the place to discover it again;
  sec 4.2 pencils in <= 1.7 and asks for a probe.

**The size of the package that comes back is the verdict** (sec 4.2), which is exactly why run 1
must not start from v1's. If the refit lands as large as v1's, the workstreams did not reach the
weights and the next step is the ablations, not more dials.
