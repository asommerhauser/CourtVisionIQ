"""
Scheduled sampling: train the event head to keep going after its own mistakes.

Every head trains on next-step likelihood with the REAL history in its context window, and is then
scored on what 600 self-fed steps produce. Those are different tasks. Under teacher forcing the
model never sees a context it generated, so it never learns to recover from one -- the classic
exposure-bias gap, and it is the cheapest of the three rungs in docs/v3_direction.md SS3 W4.

The mechanism: with probability p, the previous event's categorical tokens in the training sequence
are replaced by the model's own sample instead of the real ones, so the next-step target has to be
predicted from a context the model produced. p anneals from 0, and is logged per epoch.

**The honest scope limit, stated here because it decides how the A/B is read.** Only the
*categorical* columns are replaced -- event, type, result, player, secondary_player. Everything
derived stays real: ``GAME_STATE_KEYS`` (score, clock, period, team fouls, shot clock, running pace)
and ``ROSTER_STATE_KEYS`` (stint seconds, minutes, personal fouls) are produced by pure-Python folds
(``GameStateScan``, ``LineupScan``) that cannot be recomputed inside a graph. Re-implementing them
in TensorFlow would be the second implementation the project forbids, and it would drift.

So this is TOKEN-level robustness, not counterfactual game state. Two things follow, and both are
deliberate:

- It makes the task HARDER, never easier. The model sees a wrong previous token against a true
  score, clock and lineup, so there is no optimistic leakage -- if anything the mismatch is a mild
  adversarial perturbation.
- If rung 1 shows no gain, the finding is "token-only scheduled sampling does not help", not
  "scheduled sampling does not help". Write it that way.

**Where it mixes.** Only at positions the controller actually queries. ``apply_query_mask`` already
marks them -- roughly 68.5% of rows, the rest being continuation rows the rollout expands from one
sample -- and they arrive here as a non-zero sample weight. Mixing anywhere else would teach the
model to recover from mistakes it will never make.

**Which head.** ``event_time`` only in this retrain. It drives the rollout, it trains on the full
corpus, and restricting to it keeps the cost at the low end of the estimate (one extra forward pass
per step, ~1.5-2x per epoch). The mixin is available to the others; nothing else is wired. It must
NOT go on ``player`` / ``substitution`` / ``sub_decision``, whose inputs ARE lineup state -- exactly
the thing this leaves stale.
"""
from __future__ import annotations

# `import config`, not `from config import ...`: these knobs are switched in tests and on the
# command line, and a value bound at import time would ignore both.
import config

# The columns a sample may replace. Categorical tokens only -- see the module docstring on why
# nothing derived is in this list, and why adding one would need a second GameStateScan.
MIXABLE_FIELDS = ("event", "type", "result", "player", "secondary_player")


def mixing_probability(epoch: int) -> float:
    """``p`` for a 0-based epoch: zero through the warmup, then a linear ramp to the cap.

    Annealed from 0 so the model first learns the task at all. A model that has never fit the
    next-step distribution has nothing worth sampling from, and mixing early just trains it on
    noise it generated.
    """
    if not config.SCHEDULED_SAMPLING:
        return 0.0
    ramp = max(config.SCHEDULED_SAMPLING_RAMP_EPOCHS, 1)
    progress = (epoch - config.SCHEDULED_SAMPLING_WARMUP_EPOCHS) / ramp
    return float(config.SCHEDULED_SAMPLING_MAX_P * min(max(progress, 0.0), 1.0))


class ScheduledSamplingSchedule:
    """Sets ``p`` on the model at each epoch and records it in the logs.

    A callback rather than a computation inside ``train_step`` because ``p`` is a per-epoch value
    and the logs entry is what lands it in ``epochs.parquet`` through ``ReportingCallback`` -- which
    is the only way an A/B can say what schedule actually ran.
    """

    def __new__(cls, model):
        from tensorflow import keras

        class _Schedule(keras.callbacks.Callback):
            def __init__(self, target):
                super().__init__()
                self.target = target

            def on_epoch_begin(self, epoch, logs=None):
                self.target.set_mixing_probability(mixing_probability(epoch))

            def on_epoch_end(self, epoch, logs=None):
                if logs is not None:
                    logs["scheduled_sampling_p"] = mixing_probability(epoch)

        return _Schedule(model)


def build_scheduled_sampling_model(inner, *, n_games: int = 0):
    """Deprecated shim -- see ``models/train_steps.build_trainer``, which owns the one train_step."""
    from models.train_steps import build_trainer

    return build_trainer(inner, n_games=n_games, scheduled_sampling=True)


__all__ = ["MIXABLE_FIELDS", "mixing_probability", "ScheduledSamplingSchedule",
           "build_scheduled_sampling_model"]
