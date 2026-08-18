# CourtVisionIQ — Pitch, with notes

The pitch as written, with notes under each paragraph on what it means and where the
soft spots are.

---

> CourtVisionIQ is a generative world-model for basketball. It applies the same class of
> technology behind modern LLMs, a large transformer trained to predict what comes next, to the
> sequence of a basketball game rather than to text. It does not regress a final score from team
> ratings. It generates the game, play by play, sampling each step from a learned distribution
> the way a language model samples a sentence, then runs the matchup many times over to produce
> a full distribution of outcomes: score, pace, box score, win probability.

**Notes.** Everyone else builds a rating model: two teams in, one number out. We play the game
possession by possession and then look at who won. The win probability is what falls out of
running it many times, not what we calculate. That's the whole difference, and everything else
in the pitch depends on it.

If asked how many times we run it: tens of simulations per matchup in the runs these numbers
come from. More is a compute dial, not a different model.

---

> Trained on a sample drawn from more than twenty seasons and tested against a season it has
> never seen, v1.0 picks winners on 67 percent of held-out games at a Brier score of 0.220, where
> lower is better and a coin flip is 0.250. That lands us inside the pack of established public
> forecasters on our first version. FiveThirtyEight's NBA models pick games in the mid to high
> 60s. ESPN's pregame number is 0.219, and inpredictable's is 0.216. The betting market, priced
> by decades of capital and every data source available, sits near 0.19. We are one version in,
> and the gap between us and the sharpest money in sports is three hundredths.

**Notes.** Brier score grades probabilities, not just picks — being confident and wrong costs you.
Coin flip is 0.250, lower is better. It's the standard metric in forecasting.

Two things to know:

- FiveThirtyEight was shut down in 2025. Use past tense or a sports-analytics audience will
  catch it.
- The market's 0.19 includes injury reports and confirmed lineups that land minutes before tip.
  We don't ingest any of that yet. That's a data integration, not a modeling gap. Worth saying if
  someone pushes on the 0.220 vs 0.19 comparison — but don't oversell it, they're still better.

---

> And we are matching those models on a metric that is the only thing they produce. They return
> one number, the probability that one team beats another. We return the entire game, and the box
> scores that come out of it already land within a few attempts on shooting, a few possessions on
> pace, and well under a single assist of bias on playmaking. The win condition is one of the many
> things we can measure, not the whole of what we do.

**Notes.** Those three are real: pace bias is +0.04 possessions, shot attempts +1.6, assists
−0.33. "Bias" means systematically off in a direction, which is the number that matters for a
simulator — individual games are supposed to vary.

The soft spot: **simulated teams score about 11–12 points too few per game** (104 vs a real 116).
The shape of the game is right; the finishing rates run low and we draw too few free throws.
It's diagnosed, the fix is in the code, and the confirming run hasn't happened yet.

So: don't claim projected totals or over/unders work today. If someone asks what we'd project for
a game total, say it's the open calibration item and route it to Alec. It doesn't affect the win
probability numbers above.

---

> These numbers come from multi-day runs on rented remote GPUs, which is also why the holdout is
> a hundred games and not a thousand. The pipeline that makes those runs possible is built and
> working, so the obvious levers, more compute, more simulation, more experimentation, more
> manpower, are ones we already know how to pull.

**Notes.** Worth raising the sample size yourself rather than waiting to be challenged on it. A
hundred games is directional, not settled. The reason it isn't a thousand is GPU hours, not
capability — the same harness runs the thousand unchanged.

Also true and worth saying if cost comes up: this is cheap by AI standards. A full train plus
evaluation is dollars of rented GPU time. Basketball has a few hundred things that can happen;
English has a hundred thousand words. You don't need a giant model for it.

---

> Because the simulator generates whole games rather than a single number, every product is a
> query against one engine instead of a new model: matchup and win probability, projected box
> scores, player evaluation from simulated usage and efficiency. Roster is a live input, so you
> can swap any player for a league-average replacement, re-simulate, and read wins over
> replacement directly from the change in outcomes, a causal answer rather than a regression
> coefficient. Tracking data is the next unlock, turning shot location into something the model
> generates rather than looks up, which gives predictive shot maps against the specific defense on
> the floor. The concept is sound and the results are already showing up.

**Notes.** The replacement-player point is the strongest thing here. Every existing wins-above-
replacement number in sports is fit after the fact from correlations. Ours re-runs the season
without the player and reads the difference. That's an experiment, not a coefficient.

Tracking data needs a license — commercial terms go to Alec.

One thing the model genuinely does not do yet: it doesn't track the score and clock, so
end-of-game and clutch behavior isn't modeled. If someone asks about crunch time, say it's a
known roadmap item. Don't claim it.

---

## Route to Alec

Data sources and licensing, commercial terms, anything score-level (totals, over/unders),
production latency, feature timelines, other sports, deployment.

"I'd rather have that exactly right than fast — let me get you Alec" is a fine answer and costs
nothing.
