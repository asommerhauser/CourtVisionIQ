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

## `v2-run2.json` -- the first 2.0 refit

Run 1 was the measurement: every fitted dial at neutral, so the size of the package that came
back would mean something. This is that package, and **it is small where it counts.**

The headline: `SHOT_RESULT_BIAS["made"]` fits at **-0.022**. v1.0 needed **+0.40** there, carried
across three runs and never fully closing the gap. eFG bias with the dial *off* was -0.008 against
v1.0's -0.017 *with* it. The make-rate defect is fixed in the weights, not dialled around. Same
story on the three-point mix: `tpa` bias +2.31 -> **+0.60** and `tpm` -0.60 -> **-0.30** with
`assist_type` empty, so the zone tokens did their job. By sec 4.2's rule, that is the verdict, and
it is a pass.

What *did* come back big is one bug and one head, and neither is a calibration failure:

### The fouler's side was never drawn (a code fix, not a dial)

`_do_foul` sampled the fouler from all ten and then looked up his side, so the side was a
consequence of the pick. The player head does not know who has the ball and
`PLAYER_TEMPERATURE = 2.0` flattens what little it infers, so it split ~50/50 against a real ~13%.
Half of every foul was masked to `OFFENSIVE_SIDE_FOUL_TYPES`, where `shooting 2pt` is not legal.

That one mechanism is **FTA -10.68, tov +4.50 and most of pts -6.55**. Box turnovers count
offensive fouls, and those ran 13.34/game against 3.82 real; turnover *events* were slightly
*under* the whole time. FT% was exact at 77.6% vs 77.6% -- only the count was wrong. PF read fine
at -0.31 because box PF excludes technicals, which is how a +3.37/game foul excess hid in plain
sight.

No dial reaches this. The mass is on the wrong side of the mask, so suppressing `offensive` only
spills it into `loose ball`, already 2.9x over. The controller now draws the side first and samples
the fouler from that side's five -- the order `_do_rebound` has always used -- with the rate under
the new `FOUL_OFFENSE_SIDE_PROB`.

### `rebound_type` over-produces the two new TEAM tokens 3.2x

31.7 team rebounds/game against 9.25 real. A team rebound credits no player, so it leaves the box
score entirely: that is the whole of **dreb -10.45**, and it is why `oreb_pct` read .345 vs .242 --
the denominator collapsed, not the numerator. Rebound *events* were right the entire time (97.4
vs 98.35/game), which is exactly why nothing caught it until the split was counted directly. One
more for the list of things a test would never have found.

All four tokens are always allowed and the head is sampled with `next_player=None`, so no mask is
involved and this is a clean log-ratio fit.

### The package

| Dial | run 1 | run 2 | Why |
|---|---|---|---|
| `FOUL_OFFENSE_SIDE_PROB` | *(did not exist)* | 0.1286 | solved jointly with `foul_type` below; agrees with an independent read of the same file (offensive fouls 3.82/game + ~half of loose ball/technical/flagrant ~1.8 = 13.7% of 40.99) |
| `TYPE_BIAS.foul_type` | `{}` | 7 tokens | the two-sided masked multinomial, fitted with the side prob rather than after it |
| `TYPE_BIAS.rebound_type` | `{}` | 3 tokens | log-ratio; `defensive` is the zero |
| `SHOT_RESULT_BIAS` | `{}` | `{blocked: -0.134}` | blocks 11.00/game vs 9.55. `made` fits at -0.022 and is left at 0 |
| `EVENT_BIAS` | `{}` | `{assist: -0.087}` | assists 54.34/game vs 48.55 |
| `SUB_INCOMING_TEMPERATURE` | 1.0 | 1.0 | neutral beat v1.0's sharpened 0.45 on player-minutes MAE, 5.552 vs 5.664. It stays off |

Held at neutral **with the measurement in hand**, because fitting them now would make run 2
unreadable:

- **`DELTA_TIME_SCALE` 1.0**, though pace fits at 1.0198 (102.57 vs 100.58). Pace is FGA-driven
  and run 1 was missing 40% of its free throws, so that +2.0 was measured under a regime that no
  longer exists. Run 3's first knob.
- **`PLAYER_TEMPERATURE` 2.0**. It is implicated in flattening the fouler's side signal, but the
  side fix pins that rate directly and moving this shifts every head at once. Still the next thing
  to probe, and sec 4.2 still pencils in <= 1.7.
- **`EVENT_BIAS.foul`** (-0.053) and **`.turnover`** (+0.060). The `foul_type` refit already takes
  total fouls to 40.93/game on its own, and box turnovers land at 12.93 vs 12.92 once offensive
  fouls are right. Both would double-correct.
- **`TYPE_BIAS.turnover_type`** (error +0.167, violation +0.371). Steals land exactly (14.53 vs
  14.52) and steals are what the box score measures; applying these renormalizes mass off the one
  number here that is already right.
- **`TYPE_BIAS.assist_type`**. Mid-range zones are mildly inflated, but eFG and the 3pt mix are the
  best they have ever been. Do not tune what 2.0 just fixed.

### What run 2 should show if this is right

FTA ~24/team (from 13.29), dreb ~32/team (from 21.86), tov ~12.9/team (from 17.42), pts ~115
(from 109.41), `oreb_pct` ~.24 (from .345). If FTA and dreb land and the win/spread numbers do
*not* move, the remaining gap is the rotation and the margin correlation, not the box score --
and `PLAYER_TEMPERATURE` plus the sim-side rotation table are where run 3 starts.

One caveat on reading the comparison: run 1 used **20 sims** against v1.0 `full4-s100`'s **100**.
The predicted margin is a mean over sims, so its sampling error is sqrt(5) larger, which accounts
for a meaningful part of spread corr 0.299 vs 0.468 -- roughly 0.39 equivalent, not 0.30. Match the
sim count before calling that a regression.
