"""
A per-game regime latent: one small vector, drawn once per rollout and held for the game.

The 2.0 evaluation's sharpest finding is that the two teams in a sim do not share a game. Across
the 50 sims of one game, corr(home pts, away pts) is 0.02; in real 2022-23 games it is 0.35. Pace sd
within a game is 2.5 against a real 4.8 across games. The consequence is arithmetic: with
independent sides, Var(H-A) and Var(H+A) both collapse to VarH + VarA, so the margin comes out at
16.5 against a real 13.7 while the total comes out at 16.7 against a real 19.7. Both halves of that
are one missing term. It is a *missing* shared component, not a mis-sized one, which is why no
shrinkage dial is the right fix and why MARGIN_CALIBRATION_SLOPE is meant to retire rather than be
tuned.

What is missing is a game-level quantity that both teams' events condition on. Running pace
(``models/game_state_features.running_pace``) is the observable half: the model can read how fast
the game has gone so far. This is the unobservable half -- whatever it is about a particular night
that makes both teams shoot well, or the whistle swallow, or the pace drag, before any of it has
happened.

**How it is trained.** One free embedding row per training game, L2-regularized toward zero, tiled
across the sequence and concatenated into the fusion. Gradient descent puts into that row whatever
about the game the other inputs cannot explain -- a per-game random effect, the same device a mixed
model uses. The regularizer is what stops it memorising the game outright: it can only pay for
structure that repeats across many events of the same game, which is exactly the shared component
the sim is missing.

**How it is used.** After training, the per-dimension standard deviation of the fitted table is
written to ``norm_stats["regime_std"]``. At rollout the simulator draws ``z ~ N(0, regime_std)``
**once per game** and holds it for every row -- so two sims of the same matchup get different nights,
and within one sim both teams share the same night. That is the whole mechanism.

**What validation loss will do, and why not to "fix" it.** A validation game's embedding row is
never trained, so it holds its initialization. ``val_loss`` therefore measures the model under an
uninformative latent -- which is exactly what a rollout gets, and so is the honest number. It will
read worse than a run without the latent. Leaking validation indices into the table to make that
number look better would be measuring the one thing the model can never have at inference.

**Gate** (docs/v3_direction.md §3 W3): sim corr(home, away) ~ 0.35, pace sd ~ 5.3, margin sd ~ 13.7,
with no shrinkage dial. Measured starting point, from the four v2 runs: corr 0.017, pace sd 2.47,
margin sd 16.46, dispersion 1.36x. The gate is read off ``eval_metrics.joint_metrics``; it is not a
box-score number and must not be judged as one.
"""
from __future__ import annotations

import numpy as np

from config import REGIME_DIM

# The model input the latent arrives on, and the training-only column that says which game a row is.
REGIME_KEY = "regime"
GAME_INDEX_KEY = "game_index"


def make_regime_input(seq_len: int):
    """The ``(SEQ, REGIME_DIM)`` Keras Input. Tiled across time because the fusion is per-row."""
    from tensorflow import keras

    return keras.Input(shape=(seq_len, REGIME_DIM), name=REGIME_KEY, dtype="float32")


def regime_projection(regime_input):
    """``Dense(16)``, matching every other continuous projection into the fusion."""
    from tensorflow.keras import layers

    return layers.Dense(16, name="regime_proj")(regime_input)


def append_regime_batches(batches: dict, game_pos: int, seq_len: int) -> None:
    """Zeros for the latent plane, and this game's index.

    The plane is zeros on disk: :class:`RegimeModel` overwrites it inside ``train_step`` from the
    embedding table, and at rollout the simulator writes its own draw. Storing zeros rather than a
    sample keeps the npz honest about what it contains -- there is no per-game value until a table
    has been fitted.
    """
    batches[REGIME_KEY].append(np.zeros((seq_len, REGIME_DIM), dtype=np.float32))
    batches[GAME_INDEX_KEY].append(np.array([game_pos], dtype=np.int32))


def regime_std(table: np.ndarray) -> list[float]:
    """Per-dimension sd of a fitted table, mean removed. What the rollout samples from.

    The mean is removed because a constant offset is not a regime -- the heads can absorb it into a
    bias, and sampling around it would just add noise with no shared structure.
    """
    table = np.asarray(table, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] < 2:
        return [0.0] * REGIME_DIM
    centered = table - table.mean(axis=0, keepdims=True)
    return [float(v) for v in centered.std(axis=0)]


def sample_regime(rng, std) -> np.ndarray:
    """One game's draw. ``std`` of all zeros returns zeros, which reproduces a model without one."""
    std = np.asarray(std, dtype=np.float32)
    if std.size != REGIME_DIM or not np.any(std > 0):
        return np.zeros((REGIME_DIM,), dtype=np.float32)
    return (rng.standard_normal(REGIME_DIM) * std).astype(np.float32)


def build_regime_model(inner, n_games: int):
    """Deprecated shim. The single training wrapper lives in ``models/train_steps.py``.

    Kept so the reason is on the record: a second ``keras.Model`` wrapper could not compose with
    scheduled sampling's -- the outer one's ``self.inner(x)`` would hand the inner wrapper a batch
    it does not declare, and the ordering between the two would be implicit rather than chosen.
    """
    from models.train_steps import build_trainer

    return build_trainer(inner, n_games=n_games)


__all__ = ["REGIME_KEY", "GAME_INDEX_KEY", "make_regime_input", "regime_projection",
           "append_regime_batches", "regime_std", "sample_regime", "build_regime_model"]
