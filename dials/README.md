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

## `v2-run3.json` -- the share refit; every volume dial held

Run 2's own exit criterion (the end of the previous section) was: *"If FTA and dreb land and the
win/spread numbers do not move, the remaining gap is the rotation and the margin correlation, not
the box score."* Both halves came true, so this package is built on that verdict.

FTA 13.29 -> 20.60, dreb 21.86 -> 32.32, tov 17.42 -> 13.13, pts 109.41 -> 114.34, `oreb_pct`
.345 -> .252. And the win/spread numbers did not move: paired over the same 100 holdout game ids
against v1.0 `full4-s100`, Brier +0.0128 (SE 0.0140), score-view Brier +0.0029 (SE 0.0133),
|margin error| +0.48 (SE 0.58). Nothing at even one sigma. **At 100 games the win and spread
metrics cannot resolve a dial change** -- which is what the eval-config change beside this package
is for. Do not fit anything against them.

So run 3 fits only what 100 games *can* resolve, and splits the residual by what a dial reaches:

| Box residual | sigma (200 team-games) | Verdict |
|---|---|---|
| `ft_rate` -.0452, `fta` -3.28, `ftm` -2.60 | 7.8 / 6.9 / 6.5 | **code fix, not a dial** (below) |
| `fga` +3.10, pace +1.52 | 6.7 / 4.6 | held -- the code fix moves this regime |
| `efg` -.0136 | 3.2 | **fitted**, as mix + per-zone rate |
| `blk` +0.46 | 3.1 | held -- 1.7 sigma as a *share*; the count rides the FGA excess |
| pts, fgm, tpa, tpm, oreb, dreb, ast, stl, tov, pf, `oreb_pct` | all < 3 | held, with the measurement |
| spread bias +1.10 | **0.9** | `HOME_COURT_SHOT_BIAS` stays 0.055 |

That last row is worth stating plainly: the spread-bias SE at 100 games is 1.25, so v1.0's three
successive retunings of `HOME_COURT_SHOT_BIAS` (0.10 -> 0.07 -> 0.047 -> 0.055, each chasing a bias
of +1.47, +1.29, -0.69) were fitting noise. It does not move again until the holdout is big enough
to see it.

### The free-throw deficit is the and-1 check, and no dial reaches it

Attributing every free throw in the run to the foul that caused it (2,000 sim play-by-plays against
the same 100 games' real ones) puts 88% of the -6.55 FT/game on one number: free throws per
`shooting 2pt` foul, sim **1.56** against real **1.75**. The controller gives an and-1 one attempt
and everything else two, so that ratio *is* the and-1 rate: 8.41/game sim against 5.10 real.

`_do_shooting_foul`'s and-1 test asks whether the previous row is a made field goal by the shooting
team. It never asks *when*. Bucketing those fouls by the gap since that basket:

| gap since the made FG | sim/game | real/game | real 1-FT trips |
|---|---|---|---|
| 0s | 0.32 | 5.00 | 5.00 |
| 1-2s | 1.47 | 0.00 | 0.00 |
| 3-5s | 1.81 | 0.13 | 0.00 |
| 6-12s | 3.81 | 1.56 | 0.00 |
| 13s+ | 1.28 | 2.88 | 0.00 |

In the real data **every** and-1 sits at dt = 0, and a shooting foul three or more seconds after a
basket *always* gets two. The sim treats all 8.68 as and-1s, 8.37 of them on a later possession.
The fix is a dt == 0 guard, and it is the same shape as run 2's fouler-side bug: the mass is in the
wrong branch, so no offset on `foul_type` or `FOUL_OFFENSE_SIDE_PROB` can reach it -- pushing
shooting fouls up just produces more mis-classified ones.

**This is why every volume dial is held.** The guard alone moves free throws 41.20 -> ~49.6 against
a real 47.75, i.e. it *over*-corrects the deficit by ~1.8/game. An FTA dial written now would
double-correct by about that much in the wrong direction. And free throws enter possessions at 0.44
each, so the guard adds ~+1.4 to a pace bias that is already +1.52: `DELTA_TIME_SCALE` fits at
**1.015** against run 2 as it stands (pace 102.20 vs 100.68), and that number is void the moment
the guard lands. Pace is still run 3's knob, one run later than the previous section pencilled it.

The guard also exposes the other half, a diagnostic rather than a dial: the sim emits a same-instant
foul after a made basket **0.32/game against a real 5.00**. It over-produces false and-1s and
under-produces true ones by 15x, so the guard's net effect depends on both. Measure it before
fitting free throws again.

Two smaller reads from the same pass, both held: the foul `result` token `op` is 2.70/game real and
**0.00** in the sim (loose-ball fouls resolve somewhere else), though the FTs they generate match
(1.25 vs 1.28), so it looks like a label mapping rather than a behaviour; and `timeout` runs
7.08/game against 10.96 real, which matters because timeouts consume clock -- part of the pace
excess is the missing timeout volume, and `DELTA_TIME_SCALE` would paper over it.

### What is fitted

Every value below is a log-ratio from the pooled play-by-play (2,000 sim games against the same 100
real ones), kept only at >= 3 sigma. Shares are scale-free, so none of them is disturbed by the
and-1 fix -- which is exactly why they can ship alongside it.

| Dial | run 2 | run 3 | sigma | Why |
|---|---|---|---|---|
| `TYPE_BIAS.shot_type` | *(never dialled)* | 4 zones | 4.1-6.9 | the live-FG mix funnels threes to the top of the arc and twos to the paint |
| `SHOT_RESULT_BIAS_BY_ZONE` | `{}` | 12 zones | 4.0-4.5 | first use ever; 2P% is exact, the whole rate error is 3pt and mid-range |
| `TYPE_BIAS.rebound_type` | 3 tokens | 2 re-fitted | 3.7-7.2 | the TEAM tokens are still over, 13.44/game against 10.04 |

**`shot_type`** (relative to `rim` = 0), from 364,770 sim and 17,619 real live attempts: `top3`
-0.196 (share .1165 vs .1022), `paint` -0.173 (.2181 vs .1959), `wing3_r` +0.052, `wing3_l` +0.031.
Held: `heave` fits at +1.566 and is 12.9 sigma, but its real share is .0047 -- a large offset on a
near-zero-mass token is precisely v1.0's `loose ball` +1.2 mistake, and a heave is an end-of-period
behaviour, not a calibration. Both corner threes and all seven mid-range zones are under 3 sigma
and stay at 0.

**`SHOT_RESULT_BIAS_BY_ZONE`** -- the hook this dial was added for. Decomposing `efg` -.0136 by
holding one factor at a time: the mix error is worth +.0056 and the per-zone rate error +.0096.
Pooled to the four physical groups, because the real side is only 100 games and fifteen separate
make-rate fits would be noise:

| group | sim att | sim FG% | real FG% | d logit | sigma | applied |
|---|---|---|---|---|---|---|
| rim | 104,661 | .6696 | .6638 | -0.026 | 0.9 | no |
| paint | 79,561 | .4337 | .4389 | +0.021 | 0.6 | no |
| mid (7 zones) | 41,318 | .4678 | .4216 | **-0.187** | 4.0 | yes |
| three (5 zones) | 138,871 | .3490 | .3761 | **+0.117** | 4.5 | yes |
| heave | 359 | .3398 | .1807 | -0.847 | 2.8 | no |

Two-point efficiency is already right (55.0% sim against 54.8% real). The entire make-rate error is
3P% 34.90 against 37.61, plus mid-range running 4.6pp too generous. `made` is still 0 in the global
`SHOT_RESULT_BIAS` and stays there -- the correction is per-zone or it is nothing, which is what the
previous section's "made fits at -0.022" was already saying. `blocked` -0.134 carries over
unchanged: per-zone entries merge key-by-key on top of the global, so every zone keeps it.

**`rebound_type`**: `team offensive` -1.617 -> **-1.963**, `team defensive` -1.653 -> **-1.872**
(increments -0.346 / -0.219 on top of run 2's, since the shares were measured with run 2's dials
live). `offensive` stays at -0.564 -- it fits at +0.003, 0.1 sigma, it landed exactly.

Held at neutral **with the measurement**, on the same rule as the previous two packages:

- **`turnover_type`.** `violation` is 1.5x under as a *share* (.1287 vs .1835, 7.9 sigma) and
  fitting it would put +0.445 on the token. But that renormalises mass off `steal`, and box `stl`
  is only +0.30 at **1.4 sigma** -- applying it trades a +0.30 error for about -0.34. The two
  references also disagree on what a turnover is: the real play-by-play has 23.54 turnover events
  plus 3.83 offensive fouls = 27.37/game where the real box says 25.94, so no single value satisfies
  both. The box is what is scored, and box `tov` is +0.16 at 0.6 sigma. Nothing to win here.
- **`EVENT_BIAS`.** `assist` now lands *exactly* on the open-play mix (share .17389 sim against
  .17390 real) with -0.087 applied -- leave it. `turnover` +0.083 and `foul` +0.032 would both
  break box numbers that are already inside a sigma (`tov` +0.16, `pf` +0.05).
- **`SHOT_RESULT_BIAS.blocked`.** Box `blk` +0.46 is 3.1 sigma, but as a share of live attempts it
  is .0571 against .0539 -- 1.7 sigma, and the FGA excess it rides on is +3.5%. Fix the denominator
  first.
- **`foul_type`, `FOUL_OFFENSE_SIDE_PROB`.** The run 2 solve still reproduces every token
  (`shooting 2pt` 19.26 vs 20.49, `personal` 12.36 vs 10.60, `offensive` 3.84 vs 3.83, total 40.90
  vs 40.79) and it was fitted jointly. The and-1 guard changes what a shooting foul *costs*, not how
  often one is called, so this refits after the guard or not at all.
- **`PLAYER_TEMPERATURE` 2.0, `SUB_INCOMING_TEMPERATURE` 1.0.** See the probe below.

## `v2-run3-pt17.json` -- the rotation probe, and why minutes MAE must not judge it

Identical to `v2-run3.json` except `PLAYER_TEMPERATURE` 1.7, the value sec 4.2 pencilled in. Run it
as an A/B against `v2-run3.json` on the same games and the same `--seed`.

Run 2 compresses the rotation harder than v1.0 did. Starters (36+ actual minutes) are predicted at
32.8 against 38.8, bias **-5.98** against v1.0's -4.96; the deep bench (0-8 actual) at 12.4 against
4.4, **+8.01** against +6.28. sd of predicted minutes is 8.86 against a real 10.98 (v1.0: 9.96).

The trap: a cross-validated monotone recalibration of predicted minutes -- map predicted to expected
actual on four folds, rescale each player's counting stats by the ratio, score the fifth -- makes
every number **worse**. Minutes MAE +2.6%, pts MAE +1.0%, minutes RMSE +0.5%. MAE and RMSE under
uncertainty are minimised by shrinking toward the middle, so the compression is already at the
error-minimising point and un-flattening the rotation *costs* box-score MAE by construction.

Which means the previous section's reason for holding `SUB_INCOMING_TEMPERATURE` at neutral --
"neutral beat v1.0's sharpened 0.45 on player-minutes MAE, 5.552 vs 5.664" -- was decided on a
metric that structurally prefers the flatter rotation. That is not evidence the `sub_decision` head
learned the rotation; it is evidence MAE likes compression. Judge this probe on win/spread and on a
proper scoring rule (CRPS / pinball over the sim's own minutes distribution, which the run already
stores as `player_std`), and expect minutes MAE to get *worse* if it is working.

The reason to care is the margin. Correcting for Monte-Carlo noise with the within-game margin sd
(17.60 at 20 sims = 3.94 on the mean), run 2's spread corr of 0.375 is 0.420 of signal against
0.483 for v1.0 `full4-s100` -- much closer than the raw numbers, and inside the noise at n=100. But
decomposing it says the deficit is not knowledge:

| | v1.0 full4 | v2 run 2 | real |
|---|---|---|---|
| corr(pred team pts, actual), home / away | 0.313 / 0.204 | **0.391 / 0.297** | -- |
| corr(pred game total, actual total) | 0.170 | **0.323** | -- |
| corr(ORtg differential) | 0.452 | 0.432 | -- |
| corr(pred home pts, pred away pts) | +0.450 | **+0.120** | +0.342 |
| corr(pred home poss, pred away poss) | +0.958 | +0.932 | +0.883 |
| corr(pred home eFG, pred away eFG) | -0.086 | -0.154 | +0.162 |

2.0 is better at every level except the margin. v1.0 *over*-coupled the two teams' scores, so shared
game-level error cancelled out of the difference and flattered its margins; 2.0 is under-coupled, so
per-side error survives the subtraction. Possessions are still coupled correctly -- it is efficiency
that has come apart. That is the thing to chase after the rotation, and it is structural, not a dial.
