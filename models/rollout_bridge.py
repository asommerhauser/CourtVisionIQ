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

import config
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
        """
        if "games" not in cache:
            from data_loading import load_all_cleaned
            df = load_all_cleaned(data_dir, parse_rosters=True)
            wanted = set(game_ids)
            rows = df[df["game_id"].isin(wanted)]
            cache["games"] = [(gid, rows[rows["game_id"] == gid]) for gid in game_ids]
        return cache["games"]

    def score_fn(epoch: int):
        from simulation.evaluation import (
            _real_starters, build_game_record, simulate_games)
        from simulation.eval_metrics import _aggregate
        from simulation.game_input import extract_game_input

        sim = make_sim()
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

        # One batched call for every game and sim: a single game keeps only ~2 sims on the same head
        # at once, so pooling is what fills the batch.
        per_game = simulate_games(sim, games, n_sims=sims_per_game, seed0=seed + epoch,
                                  batch_size=config.ROLLOUT_BATCH_SIZE, game_ids=ids)

        records, histories = [], []
        for (gid, game_df), (boxes, hists) in zip(_games(), per_game):
            if not boxes:
                continue
            records.append(build_game_record(game_df, boxes, n_sims=sims_per_game))
            histories.extend(h for h in (hists or []) if h)
        if not records:
            return None

        aggregate = _aggregate(records)
        probes = scored_probe_rows({"sim": sim_probe_summary(histories), "real": _real_summary()})
        value = rollout_score(aggregate, probes)
        if echo:
            echo(f"[rollout] epoch {epoch + 1}: score {value:.4f} over {len(records)} games "
                 f"x {sims_per_game} sims")
        return value

    return score_fn


__all__ = ["SCORED_PROBES", "build_rollout_score_fn", "scored_probe_rows", "sim_probe_summary"]
