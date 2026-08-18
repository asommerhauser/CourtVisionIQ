# CourtVisionIQ — Speaker's Notes

*Companion to the one-page pitch. For the person delivering it, not the audience.*

You do not need to understand the math. You need to understand **what we built, why it is
different, and where the edges are** — so you sound confident where we're strong and honest
where we're early. Nothing in this document is proprietary; you can say all of it out loud.

Rule of thumb for the whole conversation: **"I'm not the engineer, let me get Alec on that"**
is a completely acceptable answer and it makes you *more* credible, not less. A salesperson who
knows the boundary of their knowledge reads as trustworthy. One who bluffs a technical answer
gets caught in one follow-up.

---

## 1. The thing itself, in plain language

### The one-liner
> "It's autocomplete, but for a basketball game instead of a sentence."

That's the whole idea and it's genuinely accurate. ChatGPT reads the words so far and predicts
the next word. We read the plays so far — shot, miss, rebound, foul, substitution — and predict
the next play, and how many seconds until it happens. Then we feed that prediction back in and
predict the play after that. Do that a few hundred times and you have generated an entire
basketball game that never happened but easily could have.

### The critical distinction (say this one slowly)
Everyone else in this space builds a **rating model**: they take two teams, feed in season
statistics and adjustments, and output a number — the chance Team A wins. One input, one output.

We built a **game generator**. We don't calculate who wins. We *play the game*, from tip to
buzzer, and then we look at who won. Then we play it again. And again. A few dozen to a few
hundred times. The win probability is just "how often did Team A win across all those
simulations" — it's a **byproduct**, not the product.

That difference is the entire pitch. Everything else follows from it.

### Why the difference matters commercially
A rating model gives you one number and that number is all you will ever get from it. If a
customer wants projected rebounds, you have to build a second model. Points in the paint? Third
model. What happens if we trade for this guy? Fourth model. Each one is a separate build,
separately trained, separately maintained, and they will all quietly contradict each other.

Because we generate the whole game, **every one of those is a question you ask the same engine.**
The rebounds were already in the simulation. So were the minutes, the fouls, the assists, the
run in the third quarter. We don't build a new model, we run a different query. That's the
architectural argument, and it's the reason this is a platform rather than a product.

### The analogy that lands with non-technical people
A weather rating model tells you "70% chance of rain tomorrow." A weather *simulation* runs the
atmosphere forward a thousand times and shows you a thousand possible tomorrows. You get the 70%
out of it, but you also get when it starts, how hard it comes down, and how bad the worst case
is. We are the second thing. Sports analytics has mostly been doing the first thing.

---

## 2. The numbers — what each one actually means

### Brier score
The single most likely thing you'll get asked to explain. Keep it simple:

> "It scores how good your probabilities are, not just whether you picked right. Lower is
> better. A coin flip scores 0.250. Being confident and wrong is punished hard. It's the
> standard scoreboard in forecasting — the same metric election and weather forecasters get
> graded on."

Why it's the right metric and not a dodge: picking winners is easy (favorites win most NBA
games, so a model that always picks the home favorite already looks decent). Brier catches
whether you *knew how sure to be*, which is what any real customer is buying.

### Our headline: 67% of games picked correctly, 0.220 Brier, on 100 games it had never seen

| Who | Brier | Comment |
|---|---|---|
| Coin flip | 0.250 | The floor |
| **CourtVisionIQ v1.0** | **0.220** | First version |
| ESPN pregame | ~0.219 | Established public model |
| inpredictable | ~0.216 | Established public model |
| Betting market | ~0.19 | Decades of capital, every data source, injury news minutes before tip |

The framing: **we are one version in and we're already sitting on the public forecasters.** The
gap to the sharpest money in sports is about three hundredths — and the market has advantages
we haven't even tried to claim yet (see §5, the injury question).

### The box-score numbers, and how to use them
This is where we go somewhere nobody else can follow, so it's worth being precise:

- **Pace** — essentially exact (bias +0.04 possessions). Say "we get the tempo of the game right."
- **Shot attempts** — within about 1.6 attempts per team per game.
- **Assists** — bias of about a third of an assist. Say "under a single assist of bias."

These are *bias* figures — how far off we are on average across the sample, i.e. whether the
model is systematically wrong in a direction. That's the number that matters for a simulator,
because individual games are *supposed* to vary; the distribution is the product.

### ⚠️ The one soft spot — know it before someone finds it
Our simulated teams currently score about 11–12 points too few per game (roughly 104 vs a real
116). The shape of the game is right — right number of possessions, right number of shots, right
passing — but the finishing rates are a touch low and we draw too few free throws, and those
compound over a full game.

**This is diagnosed, understood, and already addressed in code; the confirming run hasn't been
re-run yet.** It's a calibration issue, not a modeling failure — the equivalent of a scale that
weighs consistently three pounds light.

How to handle it:
- **Don't volunteer it** in the pitch. It's not the headline and it's mid-fix.
- **Don't claim projected totals or over/unders work today.** If someone asks "what would you
  project for the total?" — that's the question to route to Alec, and you should route it.
- **If someone catches it,** the honest answer is strong: *"Good eye. Scoring level is our
  known open calibration item — the game shape is right, the finishing rates run low. Alec has
  the fix in and the confirming run queued. It doesn't move the win-probability numbers, which
  is what those results are measuring."* That answer makes us look rigorous. Bluffing does not.

### The honest caveat you should raise yourself
100 games is a small sample. **Say this before they do** — it's disarming and it's true:

> "Fair warning on the sample: that's a hundred held-out games, not a thousand. The error bar on
> a 67% figure at that sample size is real. We'd call it directional, not settled. That's a
> compute budget question, not a research question — the pipeline that produced it already runs."

Volunteering a limitation buys you credibility you can spend later.

---

## 3. Why this is hard to copy

Four points, in descending order of how much you should lean on them:

1. **It's an architecture choice, not a feature.** A competitor with a rating model can't add
   "simulate the game" as a feature. They'd have to throw it away and start over, on a different
   data shape, with different infrastructure, different evaluation. Ratings and generation are
   not on the same road.

2. **The pipeline is the hard part and it's built.** Cleaning twenty-plus seasons of messy
   play-by-play into something a model can learn from, keeping every simulated game legal
   (you can't have six men on the floor, or a rebound off a made basket), scoring simulated games
   against real ones — that plumbing is most of the work, and it exists and runs.

3. **The scaling levers are known and boring.** More data, more compute, more simulations per
   matchup, bigger model. We know exactly which ones to pull and roughly what they cost. That's
   a very different risk profile from "we need a research breakthrough."

4. **The results are already competitive.** Version one, tested against a season the model has
   never seen, matching public forecasters that have been iterated on for years.

---

## 4. What we can sell out of one engine

Frame every one of these as **"same engine, different question"** — that's the point.

- **Matchup / win probability.** Table stakes. Everyone has this; we have it too, at a
  competitive number.
- **Projected box scores.** Full stat lines for every player, with a distribution — not just
  "18 points" but the spread of outcomes, including the 20% chance he goes off. Relevant to
  fantasy, media, and player props. *(Gated on the scoring calibration above — sell the concept,
  not a delivery date, until Alec clears it.)*
- **Player evaluation.** We see simulated usage and efficiency *in context* — against a specific
  opponent, with specific teammates — not a season average.
- **Wins over replacement, done causally.** This is the demo that makes people sit up. The
  roster is a *live input*. Swap a player for a league-average replacement, re-simulate, and read
  the win difference directly. Every existing "wins above replacement" number in sports is a
  regression coefficient — a correlation fit after the fact. Ours is an experiment: we ran the
  world both ways. **Say it like that.**
- **What-if / front office.** Trades, injuries, lineup combinations. Same mechanism as above.
- **Live / in-game (roadmap).** The model reads a game in progress natively, since that's just a
  shorter history to continue from. Frame as roadmap, not shipped — and see the score-awareness
  note in §5.
- **Tracking data (the next unlock).** Today the model knows *what* happened. With
  player-tracking data it would know *where*, which turns shot charts from a lookup of history
  into something the model generates against the specific defense on the floor. Predictive shot
  maps rather than descriptive ones. Requires a data license — route commercial terms to Alec.

---

## 5. Questions you will get, and answers

### Skeptic questions

**"People have simulated sports for decades. What's new?"**
> "Where the probabilities come from. A classic Monte Carlo sim is a hand-built rulebook —
> someone decides the odds of a turnover and types the number in. Ours learned every one of those
> probabilities from twenty-plus seasons of real play, and learned them *conditionally* — they
> shift based on what just happened in this game. Nobody typed in a number."

**"Your win probability is basically ESPN's. Why would I switch?"**
> "You wouldn't switch on that number, and I wouldn't ask you to. That number is the one thing
> our engine and theirs both produce, so it's the honest comparison to lead with — we wanted to
> prove we're competitive on their turf before talking about ours. But ESPN's model can only ever
> hand you that number. Ask ours what the box score looks like, what happens if the starting
> center sits, what this game looks like at a faster tempo — same engine, different query. That's
> the part they can't answer at any price."

**"The market is at 0.19 and you're at 0.220. You can't bet this."**
> "Correct, and we're not pitching betting alpha today — I'd be lying to you if I did. Two things
> worth knowing though. One, the closing line has information we don't even ingest yet: injury
> reports, confirmed lineups, and news that lands minutes before tip. That's a data integration,
> not a modeling problem. Two, that's version one, on a fraction of the compute, on a
> hundred-game sample. The distance from a coin flip to the market is six hundredths. We covered
> three of them on the first try."

**"A hundred games proves nothing."**
> "Agreed, and that's the right question to ask. It's directional, not settled. The reason it's a
> hundred and not a thousand is rented GPU hours, not capability — the harness that produced
> those hundred runs the thousand unchanged. It's a line item, not a research program."

**"How do I know it didn't just memorize?"**
> "It's tested against a season it has never seen — the training data and the test season don't
> overlap. That's the whole point of holding a season out."

**"Isn't this just fancy statistics with an AI label on it?"**
> "It's the same class of model as the ones behind the chatbots — a transformer trained to
> predict what comes next in a sequence. We pointed it at basketball possessions instead of
> English sentences. The 'AI' part isn't marketing; it's literally the same machinery, which is
> also why the improvement levers are so well understood."

### Technical-ish questions (safe to answer)

**"What data do you train on?"**
> "Public play-by-play — the event log of what happened in each game. Twenty-plus seasons,
> sampled. Alec can give you the exact composition."

**"How big is the model? Is this a huge expensive AI thing?"**
> "Deliberately not. It's small and cheap by modern AI standards — a full training run plus a
> full evaluation is dollars of rented GPU time, not millions. Basketball has a vocabulary of a
> few hundred things that can happen; English has a hundred thousand. You don't need a
> hundred-billion-parameter model for a game with a dozen kinds of events. That's a feature — our
> cost to iterate is low."

**"How many simulations per game?"**
> "Tens per matchup in the runs those results come from, and that's purely a compute dial — more
> simulations tightens the distribution, it doesn't change the model. Alec can tell you where the
> returns flatten out."

**"Does it know the score and the clock? What about crunch time?"**
> "That's a sharp question and the honest answer is that it's a current limitation — the model
> reads the sequence of plays, and score-awareness for end-of-game behavior is a known roadmap
> item, not something shipped. Alec can walk you through what that takes."
> *(Do not claim clutch modeling. This one is genuinely not built yet.)*

**"Could this work for college / WNBA / soccer / anything else?"**
> "In principle yes — nothing about the architecture is basketball-specific, it just needs an
> event log of the sport. It'd be a retraining exercise rather than a rebuild. Alec should size
> that before I promise it."

**"Why a transformer?"**
> "Because our core bet is that basketball is path-dependent — that what happened three minutes
> ago changes what happens next, that runs and cold streaks are real. A transformer's whole
> mechanism is looking back over the sequence and weighting what matters. It's the right
> instrument for that claim. And it's testable: if momentum turns out to be weak, the model tells
> us that too."

**"What's the biggest weakness?"**
> Pick one and own it. *"Sample size on the evaluation, and rotation realism — getting simulated
> minutes distributed across a bench the way a real coach does it is genuinely hard, and it's an
> active work item."* That's true, it's not the scoring issue, and it makes you sound like you've
> read the roadmap.

### Business questions

**"Who built this?"**
> "Alec — engineering, data, models, the whole pipeline." Don't pad the team.

**"What do you need?"**
> Compute, data (especially tracking-data licensing), and people. Everything on the roadmap is
> gated on one of those three, not on a research unknown. *(If they push on amounts or terms,
> that's for you and Alec to have decided in advance — don't improvise a number.)*

**"When does it beat the market?"**
> Do not promise this. *"I won't put a date on that — anyone who does is selling you something.
> What I'll say is the gap is three hundredths on version one, and the levers to close it are
> known and mostly boring: more data, more compute, injury and lineup feeds."*

**"Who else is doing this?"**
> "On win probability, plenty — ESPN's pregame model, inpredictable, the betting market itself,
> and the private analytics shops that sell into teams and sportsbooks. On generating the full
> game and answering everything else out of the same engine, we haven't found a public
> equivalent."
> *(Note: FiveThirtyEight's models are the classic public benchmark, but the outlet was shut down
> in 2025 — refer to them in the past tense, or a sports-analytics audience will catch it.)*

---

## 6. Traps — things not to say

- ❌ **"We beat Vegas."** We don't. Saying it once destroys the whole conversation.
- ❌ **"It predicts the future."** It produces a *distribution* of plausible futures. That
  distinction is the product; blur it and we sound like every other tout.
- ❌ **Quoting box-score accuracy as uniformly dialed in.** Pace, attempts, and assists are.
  Scoring level is not, yet. Stay on the three that are.
- ❌ **Inventing a technical answer.** One follow-up exposes it, and then everything else you
  said is suspect too. "Let me get Alec" costs you nothing.
- ❌ **Over-explaining the architecture.** Nobody is buying transformer layers. They're buying
  "one engine answers every question." Get back to that.
- ❌ **Apologizing for being early.** One version in and matching public forecasters is the
  strongest thing in the deck. Deliver it as the achievement it is.

---

## 7. Route these straight to Alec

Anything about: exact data sources and licensing rights · commercial terms · latency and
throughput at production scale · projected totals / over-unders / anything score-level ·
infrastructure and deployment · timelines for a specific feature · other sports · the details of
what the model does and doesn't ingest · anything where being wrong would be embarrassing later.

Framing that keeps you strong: *"That's the kind of thing I'd rather have exactly right than
fast — let me get you Alec."*

---

## 8. Glossary

| Term | What it means |
|---|---|
| **Play-by-play** | The event log of a game — every shot, rebound, foul, substitution, with a timestamp. Our raw material. |
| **Holdout / held-out** | Games the model was never trained on, kept aside to test it honestly. Our test season is one it has never seen. |
| **Brier score** | Grade for probability forecasts. 0.250 = coin flip, lower is better, confident-and-wrong is punished. |
| **Calibration** | Whether stated confidence matches reality — of everything you called 70%, did about 70% happen? |
| **Bias** | Being wrong in a consistent direction (a scale that reads 3 lbs light), as opposed to being noisy. |
| **Monte Carlo** | Running something many times with randomness to see the range of outcomes. |
| **Pace / possessions** | How many times each team has the ball — the tempo of the game. |
| **eFG%** | Shooting efficiency, adjusted so a three counts more than a two. |
| **Transformer** | The model architecture behind modern AI. Reads a sequence, predicts what comes next. |
| **Generative** | Produces new examples (a whole game) rather than scoring an existing one. |
| **Tracking data** | Player and ball *positions* on the floor, many times per second. Licensed, expensive, and our next big unlock. |

---

## 9. The thirty-second version, if that's all you get

> "Everyone else in sports prediction builds a calculator that outputs one number: who wins.
> We built a simulator that plays the game — possession by possession, a few hundred times — and
> the win probability falls out as a byproduct, along with the box score, the minutes, and the
> answer to any what-if you want to ask. On our first version, tested on a season it had never
> seen, we're already matching the established public forecasters on the one number they produce.
> And that number is the smallest thing we make."
