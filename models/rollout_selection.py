"""
W4 rung 2: choose the checkpoint by what it SIMULATES, not by its next-step likelihood.

Training and scoring have never been the same objective here. Every head early-stops on
``val_loss`` -- one-step-ahead negative log-likelihood against the real history -- and the model is
then judged on a 600-step self-fed rollout. The 2.0 training curves say the two decouple early:
``shot_result`` reaches its best val loss at epoch 16 and then rises, while ``shot_type`` gains
0.002 after epoch 5. Whether the epoch that minimises NLL is the epoch that rolls out best is an
open question, and this is the cheapest way to answer it.

**And this cycle's W1 probes make it the more urgent rung.** Measured on v2-run4 against the real
2022-23 season:

  3rd foul before half -> off the floor within 60 s   sim 0.238   real 0.776
  4th foul before half -> off within 60 s             sim 0.280   real 0.961
  4th-foul events per game                            sim 0.371   real 0.039
  Q4 starter seconds, blowout / close                 sim 0.838   real 0.525
  trailing-team fouls per 100 s, last 2:00 down 4-9   sim 1.62    real 1.99

The model has the inputs -- personal fouls, score, clock all reached the weights in 2.0 -- and
produces none of the behaviour. No next-step loss can see that, because each individual next-step
prediction is roughly right; it is the composition over hundreds of steps that is wrong. Selecting
on rollout behaviour is the only thing in the programme that optimises the composition directly.

**How it coexists with EarlyStopping, which is the part that must be designed rather than assumed.**
``EarlyStopping(restore_best_weights=True)`` restores at ``on_train_end`` -- after every callback's
own ``on_epoch_end`` and before ``save_artifacts``. A callback that merely snapshots the weights it
likes is therefore silently overwritten, and the run reports a rollout-selected epoch that is not
what is on disk. So this selector **never restores**. It holds one snapshot, and ``train()`` applies
it after ``fit`` returns, which is strictly later than ``on_train_end``. The ordering is explicit and
does not depend on callback list position.

**A second pass, not part of the first train.** Scoring a rollout needs every head, and during a
from-scratch twelve-head run ``event_time`` trains first with no bundle to roll out. Rather than a
"no-op for the first head" branch, rung 2 runs on its own: finish the train, then retrain the head
under test warm-started off the finished bundle. That also makes the A/B clean -- identical data,
identical everything, one head swapped.

**The rung-3 condition becomes a number.** ``docs/v3_direction.md`` §6 step 6 fires only if NLL and
rollout metrics pick different checkpoints. Both epochs and both scores are recorded, so it is a
comparison rather than a judgement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config

# Where the record lands, next to the trained-head list the full run already keeps.
STATE_KEY = "checkpoint_selection"


def eval_game_ids(state: dict, *, n_games: int | None = None, seed: int = 0) -> list[int]:
    """Games to roll out during training: a sample from the TRAINING-ERA TAIL.

    Never a holdout window. The holdout is what the finished model is scored on, and selecting a
    checkpoint against it would turn the report into a training metric -- the most expensive way
    there is to fool yourself, because nothing downstream would look wrong.

    The tail is the last ``ROLLOUT_EVAL_TAIL`` games before the train cut: recent enough to look
    like the holdout's era, and trained on, which is fine. This measures whether a checkpoint
    *behaves*, not whether it generalises; generalisation is what the holdout is for.
    """
    boundary = int(state.get("boundary_idx", 0))
    tail = int(config.ROLLOUT_EVAL_TAIL)
    n_games = int(n_games or config.ROLLOUT_EVAL_GAMES)
    ordered = [int(g) for g in state.get("train_tail_game_ids", [])]
    if not ordered:
        return []
    window = ordered[-tail:] if len(ordered) > tail else ordered
    if len(window) <= n_games:
        return window
    rng = np.random.default_rng(seed)
    return sorted(int(g) for g in rng.choice(window, size=n_games, replace=False))


def rollout_score(aggregate: dict, probes: dict | None = None) -> float:
    """One number per evaluated epoch, lower is better.

    Three terms, because a checkpoint can be good at one and bad at the others:

    * **box MAE** -- the headline the report already leads with, in points.
    * **margin dispersion** -- predicted margin sd over the realised residual sd, penalised by its
      distance from 1.0 and scaled into points so it is commensurable. This is what W3 is trying to
      move, and a checkpoint that fixes the box by collapsing the spread must not win on that.
    * **game-state behaviour** -- the mean absolute gap between the sim's three probe rates and the
      real ones, times a weight that makes a fully-absent behaviour cost about as much as a point
      of box MAE. Without this term nothing in the objective notices that the simulator does not
      bench a player in foul trouble.

    Every input is read defensively: a probe block that could not be computed contributes nothing
    rather than a zero, which would read as perfect agreement.
    """
    headline = (aggregate or {}).get("headline") or {}
    score = float(headline.get("points_mae") or 0.0)

    coverage = ((aggregate or {}).get("coverage") or {}).get("margin") or {}
    ratio = float(coverage.get("dispersion_ratio") or 0.0)
    if ratio > 0:
        score += config.ROLLOUT_SCORE_DISPERSION_WEIGHT * abs(ratio - 1.0)

    gaps = []
    for row in (probes or {}).get("rows", []):
        sim, real = row.get("sim"), row.get("real")
        if sim is None or real is None or not real:
            continue
        gaps.append(abs(float(sim) - float(real)) / abs(float(real)))
    if gaps:
        score += config.ROLLOUT_SCORE_BEHAVIOUR_WEIGHT * float(np.mean(gaps))
    return score


class _Selection:
    """The record written next to ``trained_models``, so rung 3's condition is a comparison."""

    def __init__(self) -> None:
        self.best_epoch: int | None = None
        self.best_score: float | None = None
        self.scores: dict[int, float] = {}

    def offer(self, epoch: int, score: float) -> bool:
        self.scores[int(epoch)] = float(score)
        if self.best_score is None or score < self.best_score:
            self.best_epoch, self.best_score = int(epoch), float(score)
            return True
        return False

    def as_record(self, *, nll_best_epoch: int | None) -> dict:
        return {
            "rollout_best_epoch": self.best_epoch,
            "rollout_best_score": self.best_score,
            "nll_best_epoch": nll_best_epoch,
            "rollout_score_at_nll_best": self.scores.get(nll_best_epoch),
            "scores_by_epoch": {str(k): v for k, v in sorted(self.scores.items())},
            # docs/v3_direction.md §6 step 6 fires on exactly this.
            "epochs_disagree": (self.best_epoch is not None and nll_best_epoch is not None
                                and self.best_epoch != nll_best_epoch),
        }


def build_selector(model, *, score_fn, every: int | None = None):
    """The callback, or None when rung 2 is off.

    ``score_fn(epoch) -> float`` does the rollout. It is injected rather than built here so this
    module stays TensorFlow-free and so the tests can drive the whole selection policy without a
    simulator.
    """
    if not config.ROLLOUT_SELECTION:
        return None
    from tensorflow import keras

    every = int(every or config.ROLLOUT_EVAL_EVERY)
    selection = _Selection()

    class RolloutSelector(keras.callbacks.Callback):
        """Scores a rollout every ``every`` epochs and keeps the best weights. NEVER restores.

        Restoring here would be undone: EarlyStopping(restore_best_weights=True) runs at
        ``on_train_end``, after this. ``train()`` applies :attr:`best_weights` once ``fit`` has
        returned, which is strictly later.
        """

        def __init__(self, target) -> None:
            super().__init__()
            self.selection = selection
            self.best_weights = None
            # The model is held explicitly rather than read off ``self.model``. Keras injects that
            # through a read-only property, so depending on it makes the callback untestable
            # without a compiled model -- and this is precisely the policy that needs testing
            # without a GPU, a bundle or a simulator.
            self.target = target

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % every:
                return
            score = score_fn(epoch)
            if score is None:
                return
            if logs is not None:
                logs["rollout_score"] = float(score)
            if self.selection.offer(epoch, score):
                # One snapshot at a time: a full head is ~250 MB, and keeping every candidate
                # would cost more RAM than the train itself.
                self.best_weights = [np.array(w, copy=True)
                                     for w in self.target.get_weights()]

    return RolloutSelector(model)


def apply_selection(model, selector, *, history=None, monitor: str = "val_loss") -> dict | None:
    """Restore the rollout-selected weights, and return the record. Call AFTER ``fit`` returns.

    Returns ``None`` when rung 2 is off or nothing was ever scored, in which case the weights are
    exactly what EarlyStopping restored and the run is unchanged.
    """
    if selector is None or selector.best_weights is None:
        return None
    nll_best = None
    if history is not None and monitor in getattr(history, "history", {}):
        values = history.history[monitor]
        if values:
            nll_best = int(np.argmin(values))
    model.set_weights(selector.best_weights)
    record = selector.selection.as_record(nll_best_epoch=nll_best)
    print(f"[rollout] selected epoch {record['rollout_best_epoch']} "
          f"(score {record['rollout_best_score']:.4f}); NLL-best was epoch {nll_best}"
          f"{' -- THEY DISAGREE' if record['epochs_disagree'] else ''}")
    return record


def record_selection(state_path, key: str, record: dict) -> None:
    """Merge one head's selection record into the full-run state file."""
    path = Path(state_path)
    if not path.is_file():
        return
    state = json.loads(path.read_text(encoding="utf-8"))
    state.setdefault(STATE_KEY, {})[key] = record
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


__all__ = ["STATE_KEY", "eval_game_ids", "rollout_score", "build_selector",
           "apply_selection", "record_selection"]
