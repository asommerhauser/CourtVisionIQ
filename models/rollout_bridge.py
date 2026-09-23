"""
The bridge that lets checkpoint selection on rollout metrics actually run (3.2 W8).

``models/rollout_selection.py`` was written and tested in 3.0 -- the policy, the Keras callback, the
scoring formula, the EarlyStopping ordering, the ``epochs_disagree`` record. Three things stopped it:

* ``ROLLOUT_SELECTION = False``.
* **Nothing constructed the ``rollout_score_fn``** that ``models/event_time_model.py`` expects. The
  parameter existed on one head, no caller passed it, ``models.pipeline.run_stage`` had no channel for
  it, and ``self._checkpoint_selection`` was written and never read.
* **``eval_game_ids`` read ``state["train_tail_game_ids"]``, which nothing wrote**, so the selector's
  game set was unreachable from a real run state and it would have scored nothing at all.

This module is the first of those two, plus the piece ``rollout_score`` needs and nobody produced: it
expects ``probes["rows"]``, and ``reporting.state_probes.compare`` returns ``{"sim", "real"}``.

**Which probe rows are scored, and why not all of them.** ``_rows_for_frame`` emits ten rows mixing a
probability, per-game counts, raw seconds in the thousands, a ratio and a rate. ``rollout_score`` averages
*relative* gaps, so the units cancel -- but three of the ten describe the same behaviour and would
triple-weight it, and two are denominators rather than behaviours. So the selection is deliberate:

===========================================  ======  ==========================================
row                                          scored  why
===========================================  ======  ==========================================
``foul_trouble_3 / p_benched_within_60s``     yes     1,570 real events: the one that can move
``foul_trouble_4 / p_benched_within_60s``     yes     the headline gap, 0.28 against 0.96
``foul_trouble_4 / events_per_game``          yes     the 9.5x over-production, a distinct failure
``blowout_q4 / ratio``                        yes     rotation, already rate-normalised
``late_foul / rate_per_100s``                 yes     trailing-team fouling, per unit of time
``blowout_q4 / starter_seconds_blowout``      no      subsumed by the ratio
``blowout_q4 / starter_seconds_close``        no      subsumed by the ratio
``blowout_q4 / blowout_frequency``            no      a consequence of the sim's scoring, not a
                                                      rotation behaviour
``foul_trouble_3 / events_per_game``          no      already close (1.41 against 1.19); scoring it
                                                      mostly adds noise
``late_foul / state_seconds_per_game``        no      how often the state occurs, not what is done
===========================================  ======  ==========================================

The sim side is computed **from the histories in memory**, not from play-by-play files: ``probe_game``
takes rows, and writing thousands of CSVs per scored epoch to read them straight back would dominate the
cost. The real side is walked once and cached, because it does not change between epochs.
"""
from __future__ import annotations

import threading
import time

import config

#: Seconds between rollout progress lines. One evaluation is minutes to hours; a line a minute is
#: enough to project the total and cheap enough to ignore.
_HEARTBEAT_SECONDS = 60.0
from models.rollout_selection import eval_game_ids, rollout_score

#: ``(probe, metric)`` pairs that enter the behaviour term. See the module docstring for the exclusions.
SCORED_PROBES = (
    ("foul_trouble_3", "p_benched_within_60s"),
    ("foul_trouble_4", "p_benched_within_60s"),
    ("foul_trouble_4", "events_per_game"),
    ("blowout_q4", "ratio"),
    ("late_foul", "rate_per_100s"),
)


def scored_probe_rows(report: dict) -> dict:
    """``{"rows": [...]}`` for ``rollout_score``, narrowed to :data:`SCORED_PROBES`.

    ``report`` is what ``reporting.state_probes.compare`` returns, or any dict with ``sim`` and ``real``
    summaries of the same shape.
    """
    from reporting.state_probes import _rows_for_frame
    wanted = set(SCORED_PROBES)
    rows = [r for r in _rows_for_frame(report) if (r["probe"], r["metric"]) in wanted]
    return {"rows": rows}


def sim_probe_summary(histories) -> dict:
    """Pool the three game-state probes over in-memory sim histories."""
    from reporting.state_probes import _blank, accumulate, probe_game, summarize
    pool = _blank()
    for rows in histories:
        if rows:
            pool = accumulate(pool, probe_game(list(rows)))
    return summarize(pool)


class RolloutBudgetExceeded(RuntimeError):
    """One scored evaluation ran past ``config.ROLLOUT_EVAL_BUDGET_MIN``."""


def _check_budget(elapsed_min: float, epoch: int) -> None:
    """Abort the train when one evaluation costs more than the pass can afford.

    Rung 2 pays this cost at every ``ROLLOUT_EVAL_EVERY`` epochs for the whole run, so an evaluation
    that is 10x its estimate is not a slow run -- it is a run whose remaining cost is already decided
    and unaffordable. Raising here surfaces that inside one epoch with the measured number attached.

    The alternative is what happened on 2026-09-22: the first real invocation went quiet for 9.5 hours
    with no output of any kind, because nothing timed the evaluation and nothing bounded it. A silent
    stall is indistinguishable from slow progress from the outside, which is the whole problem.
    """
    budget = float(getattr(config, "ROLLOUT_EVAL_BUDGET_MIN", 0) or 0)
    if budget <= 0 or elapsed_min <= budget:
        return
    every = max(1, int(config.ROLLOUT_EVAL_EVERY))
    raise RolloutBudgetExceeded(
        f"rollout evaluation at epoch {epoch + 1} took {elapsed_min:.1f} min, over the "
        f"{budget:.0f} min budget (config.ROLLOUT_EVAL_BUDGET_MIN). At one evaluation every "
        f"{every} epochs that is ~{elapsed_min / every:.1f} min of selection per epoch for the "
        "rest of the run. Stopping now with the number rather than discovering it overnight. "
        "Either raise the budget deliberately, or cut ROLLOUT_EVAL_GAMES / ROLLOUT_EVAL_SIMS, "
        "or raise ROLLOUT_EVAL_EVERY."
    )


def build_rollout_score_fn(state: dict, *, make_sim, live_model=None, data_dir: str = "./data",
                           seasons=("2023",), n_games: int | None = None, n_sims: int | None = None,
                           seed: int = 0, echo=print):
    """A ``score_fn(epoch) -> float | None`` for ``rollout_selection.build_selector``.

    ``make_sim()`` returns a ready ``GameSimulator`` -- every head loaded from the finished bundle. Rung 2
    runs as a *second* pass over an already-trained bundle for exactly this reason: mid-train, the other
    eleven heads have no current weights of their own, so a rollout scored against them would be scoring
    a bundle that does not exist.

    ``live_model`` is a zero-argument callable returning the head's **inner** functional model, whose
    weights are copied into the simulator before each rollout. It must be the inner model and not the
    trainer wrapper: ``build_trainer`` may add a regime-latent table, so the wrapper's weight list is
    longer than the graph the simulator holds.

    Returns ``None`` -- meaning "abstain, do not record an epoch" -- when there are no games to roll out.
    That is the honest answer when ``train_tail_game_ids`` is missing, and it keeps a misconfigured run
    from silently selecting on a score computed over nothing.
    """
    game_ids = eval_game_ids(state, n_games=n_games, seed=seed)
    if not game_ids:
        if echo:
            echo("[rollout] no train-tail games in the run state; checkpoint selection will abstain. "
                 "Re-run setup so it records train_tail_game_ids.")
        return None

    sims_per_game = int(n_sims or config.ROLLOUT_EVAL_SIMS)
    cache: dict = {}

    def _real_summary() -> dict:
        """The real side, walked once. It cannot change between epochs."""
        if "real" not in cache:
            from reporting.state_probes import probe_real
            cache["real"] = probe_real(data_dir, seasons=seasons, echo=None)
        return cache["real"]

    def _games():
        """The real rows for each scored game, loaded once.

        ``game_input_for_game`` re-reads the whole cleaned corpus per call, which over twenty games and
        every third epoch would cost more than the rollouts it is scoring. One load, sliced and kept.

        The slice happens INSIDE the load (``game_ids=``), not after it. Loading all 21 seasons with
        ``parse_rosters=True`` and filtering afterwards decodes a roster literal per row for ~13M rows
        to keep twenty games, and it happens in the training process: measured 2026-09-22, that left
        20 GB resident alongside the training graph and a full twelve-head simulator, on a box whose
        host RAM has already cost this project one train.
        """
        if "games" not in cache:
            from data_loading import load_all_cleaned
            rows = load_all_cleaned(data_dir, parse_rosters=True, game_ids=game_ids)
            cache["games"] = [(gid, rows[rows["game_id"] == gid]) for gid in game_ids]
        return cache["games"]

    def score_fn(epoch: int):
        from simulation.evaluation import (
            _real_starters, build_game_record, simulate_games)
        from simulation.eval_metrics import _aggregate
        from simulation.game_input import extract_game_input

        started = time.monotonic()
        sim = make_sim()
        # Rung 2 is THE path the compiled forward was written for: ~200 game-sims of tiny passes per
        # evaluation, where eager op-dispatch leaves the GPU idle and one core does Python. Set on the
        # master simulator, which is the seam every batched worker routes through
        # (simulation/batched_rollout._WorkerSim._infer -> master._infer).
        if getattr(config, "ROLLOUT_COMPILED_INFERENCE", True):
            sim.compiled_inference = True
        if echo and "announced" not in cache:
            cache["announced"] = True
            mode = "compiled" if getattr(sim, "compiled_inference", False) else "EAGER"
            echo(f"[rollout] scoring {len(game_ids)} games x {sims_per_game} sims every "
                 f"{config.ROLLOUT_EVAL_EVERY} epochs, {mode} inference, "
                 f"budget {config.ROLLOUT_EVAL_BUDGET_MIN:.0f} min/evaluation.")
            if mode == "EAGER":
                echo("[rollout] EAGER inference: expect the GPU to idle while one core does Python. "
                     "This is the 9.5-hour configuration; set ROLLOUT_COMPILED_INFERENCE=True.")
        if live_model is not None:
            current = live_model()
            if current is not None:
                # The simulator holds its own copy of the event/time graph, built from the bundle on
                # disk. Point it at the weights THIS epoch produced, or every epoch scores the same
                # finished model and the selection is meaningless.
                sim.model.set_weights(current.get_weights())

        games, ids = [], []
        for gid, game_df in _games():
            if game_df.empty:
                continue
            spec = extract_game_input(game_df, data_dir=data_dir)
            try:
                home_starters, away_starters = _real_starters(game_df)
            except ValueError:
                home_starters = away_starters = None
            games.append((spec, home_starters, away_starters, "HOME", "AWAY"))
            ids.append(gid)
        if not games:
            return None

        # A heartbeat, and the reason this runs in STREAMING mode. ``on_sim`` fires on the slot thread
        # as each sim lands, which is the only progress signal a rollout emits: without it one
        # evaluation is a single blocking call that prints nothing until it returns, and an
        # evaluation that never returns is indistinguishable from one that is merely slow. That cost
        # 9.5 hours of silence on 2026-09-22. The projection is the number worth reading -- it is
        # accurate within the first minute, long before the evaluation finishes.
        total_sims = len(games) * sims_per_game
        streamed: list = []
        beat = {"done": 0, "last": started, "lock": threading.Lock()}

        def _on_sim(_g, _s, history, _box) -> None:
            if history:
                streamed.append(list(history))   # list.append is atomic under the GIL
            with beat["lock"]:
                beat["done"] += 1
                done, now = beat["done"], time.monotonic()
                due = (now - beat["last"]) >= _HEARTBEAT_SECONDS
                if due:
                    beat["last"] = now
            if echo and due:
                mins = (now - started) / 60.0
                echo(f"[rollout] epoch {epoch + 1}: {done}/{total_sims} sims, {mins:.1f} min "
                     f"elapsed, ~{mins / max(done, 1) * total_sims:.0f} min projected")

        # One batched call for every game and sim: a single game keeps only ~2 sims on the same head
        # at once, so pooling is what fills the batch.
        per_game = simulate_games(sim, games, n_sims=sims_per_game, seed0=seed + epoch,
                                  batch_size=config.ROLLOUT_BATCH_SIZE, game_ids=ids,
                                  on_sim=_on_sim)

        # Streaming mode returns empty history lists by contract -- every history was handed to
        # ``_on_sim`` instead, so the probes read what that collected. The box scores still come back
        # per game, which is what ``build_game_record`` needs.
        records = []
        histories = streamed
        for (gid, game_df), (boxes, _hists) in zip(_games(), per_game):
            if not boxes:
                continue
            records.append(build_game_record(game_df, boxes, n_sims=sims_per_game))
        if not records:
            return None

        aggregate = _aggregate(records)
        probes = scored_probe_rows({"sim": sim_probe_summary(histories), "real": _real_summary()})
        value = rollout_score(aggregate, probes)
        elapsed = (time.monotonic() - started) / 60.0
        if echo:
            echo(f"[rollout] epoch {epoch + 1}: score {value:.4f} over {len(records)} games "
                 f"x {sims_per_game} sims in {elapsed:.1f} min")
        _check_budget(elapsed, epoch)
        return value

    return score_fn


__all__ = ["RolloutBudgetExceeded", "SCORED_PROBES", "build_rollout_score_fn",
           "scored_probe_rows", "sim_probe_summary"]
