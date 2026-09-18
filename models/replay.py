"""
The weighted replay pass: W4 rung 3, in the cheap form (3.2 W11).

``v3_direction.md`` priced rung 3 at ~160 GPU-hours per attempt and made it conditional. This is the same
score-function estimator with the cost moved, at **3.2 GPU-hours**, and it is built unconditionally
because the composition failure is the programme's headline and this is the only item that puts a
gradient on it. No next-step loss can see it: each individual prediction is roughly right, and it is the
composition over hundreds of steps that is wrong.

**Why no gradient can flow through the rollout.** ``simulation.controller.GameController`` is pure-Python
rules on worker threads and every sample goes through numpy, so the TensorFlow graph is discarded at each
step. There is no reparameterisation available and no backprop through the game; only score-function
estimators exist, which is exactly why the original costing was what it was.

**The three pieces, and the one that turned out to be free.**

1. *Score each sim* against the real game it simulated, per head, through ``models.head_metrics``.
2. *Advantage* = the mean of the other siblings of the same game minus this one. Same matchup, same date,
   so the baseline is tightly matched and most of the variance cancels. Signed so that **positive means
   more realistic than its siblings**.
3. *Weight one pass by it.* This is the free part. Every head already multiplies its loss mask by a
   per-game weight -- ``season_features.apply_recency`` reads ``split["recency_weight"]`` -- so an
   advantage is not a new mechanism, it is the existing per-game loss weight with a different number in
   it. And the labels need no construction either: a sim's play-by-play **is** what the model sampled, so
   running it through the ordinary preprocess yields exactly "what this sim did" as targets.

**Positive advantages only.** A departure from ``v3_2_direction.md`` §4.2, which specifies a signed sample
weight. A negative weight on cross-entropy is ``-w·log p`` with ``w < 0``, which is minimised by driving
that action's probability to zero and the loss to ``-inf``: unbounded, and it diverges. Staying stable
would need advantage clipping plus a KL leash to the pre-pass weights, which is three constants that
cannot be tuned inside a single 3.2 GPU-hour pass. Filtering is bounded by construction, introduces no
hyper-parameters, and never pushes away from anything -- it reinforces what worked and declines to learn
from the rest.

**Cadence, unchanged from §4.5.** One pass, *after* the main train, over one game in ten of the subset,
ten sims per game. After rather than during, because mid-train the other eleven heads are still moving
and a scored rollout came from a bundle that stops existing a few epochs later -- credit assigned to a
lineup that gets substituted before the next play. Ten sims of the *same* game, not one sim of ten games,
because the sibling set is what makes the leave-one-out baseline work. From the training subset, never a
holdout window, and never a flat stride over the corpus: a flat stride is era-neutral and would fine-tune
``shot_result`` toward the old game, undoing a v1.1 rate increase that exists because it was under-fitting
modern efficiency. And **not once per epoch** -- one pass is 3.2 GPU-hours, the same pass every epoch of a
thirty-epoch stage is ~96, which is worse than the naive form this design exists to avoid.
"""
from __future__ import annotations

import numpy as np

from models.head_metrics import head_errors, total_error

#: Per-head scores are errors (lower is better), so an advantage is *sibling mean minus mine*.
MIN_SIBLINGS = 2


def sim_head_errors(real_stats, sim_stats_list, *, probes=None) -> list[dict]:
    """Per-head error for each sim of one game. ``probes`` may be one report or one per sim."""
    out = []
    for i, sim in enumerate(sim_stats_list):
        p = probes[i] if isinstance(probes, (list, tuple)) else probes
        out.append(head_errors(sim, real_stats, probes=p))
    return out


def sim_totals(real_stats, sim_stats_list, *, probes=None) -> list[float]:
    """One scalar per sim, for ranking a game's siblings."""
    return [total_error(sim, real_stats,
                        probes=(probes[i] if isinstance(probes, (list, tuple)) else probes))
            for i, sim in enumerate(sim_stats_list)]


def advantages(scores) -> list[float]:
    """Leave-one-out advantage per sim: the mean of the OTHER siblings' error minus this one.

    Positive means this sim reproduced the real game better than its siblings did. A game with fewer
    than :data:`MIN_SIBLINGS` sims has no baseline and yields zeros -- not a fabricated advantage of
    zero-versus-itself, which would read as "exactly average" and slip through the filter at 0.0.
    """
    values = [float(s) for s in scores]
    n = len(values)
    if n < MIN_SIBLINGS:
        return [0.0] * n
    total = sum(values)
    return [((total - v) / (n - 1)) - v for v in values]


def keep_positive(advs) -> list[int]:
    """Indices of the sims that beat their siblings. Strictly greater than zero.

    Exactly zero is not kept: on an even split it means "indistinguishable from the baseline", and
    training on it adds a gradient with no evidence behind it.
    """
    return [i for i, a in enumerate(advs) if a > 0.0]


def positive_weights(advs, *, normalise: bool = True) -> dict:
    """``{sim_index: weight}`` over the kept sims, weights strictly positive.

    Normalised so the kept weights average 1.0, which keeps the pass's effective learning rate
    comparable to an ordinary epoch's regardless of how large the advantages happen to be. Without it the
    step size would ride on the arbitrary scale of the score, and the one pass this design allows is not
    the place to discover that.
    """
    kept = keep_positive(advs)
    if not kept:
        return {}
    weights = {i: float(advs[i]) for i in kept}
    if normalise:
        mean = sum(weights.values()) / len(weights)
        if mean > 0:
            weights = {i: w / mean for i, w in weights.items()}
    return weights


def head_weights(per_head_errors, *, normalise: bool = True) -> dict:
    """``{head: {sim_index: weight}}`` -- each head weighted by ITS OWN advantage.

    The point of per-head metrics: a sim that got the shot mix right and the rotation wrong should teach
    ``shot_type`` and not ``substitution``. One shared scalar would hand both heads the same weight and
    reintroduce exactly the mis-assignment ``head_metrics`` exists to remove.
    """
    if not per_head_errors:
        return {}
    heads = list(per_head_errors[0])
    out = {}
    for head in heads:
        scores = [e.get(head, float("inf")) for e in per_head_errors]
        out[head] = positive_weights(advantages(scores), normalise=normalise)
    return out


def replay_weight_array(game_ids, weights_by_game, *, default: float = 0.0) -> np.ndarray:
    """A ``(N,)`` per-game weight array aligned to a split's game order.

    This is the seam that makes the pass cheap: every head already multiplies its ``(N, SEQ)`` loss mask
    by a per-game weight through ``season_features.apply_recency``, which reads
    ``split["recency_weight"]``. So an advantage is not new machinery -- it is that existing channel with
    a different number in it, and all twelve heads already honour it.

    Games with no weight get ``default`` (0.0), which is how a filtered-out sim contributes nothing
    without having to be removed from the tensors.
    """
    return np.array([float(weights_by_game.get(g, default)) for g in game_ids], dtype=np.float32)


def apply_advantages(split: dict, game_ids, weights_by_game) -> dict:
    """Put the advantages into the split where the per-game loss weight already lives.

    Deliberately the same key ``attach_recency_weights`` writes: the replay pass is a *reweighting* of
    the ordinary training path, so it should travel down the ordinary channel rather than add a second
    one that every head would have to learn about.

    Recency itself does not apply here -- these are simulations of games from one narrow slice, not a
    corpus spanning twenty-one seasons -- so overwriting rather than multiplying is the right composition.
    """
    split = dict(split)
    split["recency_weight"] = replay_weight_array(game_ids, weights_by_game)
    return split


def kept_fraction(advs) -> float:
    """Share of sims that survive the filter. Expected near 0.5; far from it is worth looking at."""
    return (len(keep_positive(advs)) / len(advs)) if advs else 0.0


__all__ = [
    "MIN_SIBLINGS", "advantages", "apply_advantages", "head_weights", "keep_positive",
    "kept_fraction", "positive_weights", "replay_weight_array", "sim_head_errors", "sim_totals",
]
