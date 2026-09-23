"""
The weighted replay pass, end to end: simulate, score, weight, train once (3.2 W11, W4 rung 3).

Three pieces already existed and none of them were joined to anything. ``models/replay.py`` holds the
estimator (advantages, the positive filter, the weight arrays), ``simulation/decision_log.py`` records
what each head sampled, and ``models/head_metrics.py`` scores a sim per head. Nothing outside the tests
imported any of them, so arm 3 of the A/B was a command that evaluated arm 1's bundle under a different
run name. This is the driver that makes it an arm.

**The shape of one pass.**

1. **Choose the games.** One in ten of the training subset, drawn once and deterministically
   (``training/replay_corpus.select_replay_games``). From the subset, never a holdout window.
2. **Simulate.** Ten sims of each game, pooled into batched rollouts. Ten sims of the *same* game,
   because the sibling set is what makes the leave-one-out baseline work.
3. **Score each sim** against the real game it simulated, per head, through ``models.head_metrics`` --
   plus the three game-state probes, computed here from the sim's own rows rather than from a finished
   run directory, so the foul and rotation behaviour W11 is aimed at actually reaches the weights.
4. **Advantage** = the mean of the other nine siblings' error minus this one's, per head. Keep the
   positives. A sim that got the shot mix right and the rotation wrong teaches ``shot_type`` and not
   ``substitution``; that is the entire point of scoring per head.
5. **One weighted pass**, through ``models.pipeline.run_stage`` -- the ordinary training path, with the
   sims as the corpus and the advantages riding the per-game loss weight every head already honours.

**What this pass must not damage, and how that is enforced here.**

* *The real tensors.* It preprocesses a corpus of simulations, so it runs with its own
  ``processed_dir``. Left at the default it would overwrite ``./data/processed`` and the next
  ``--continue`` would train on simulations silently. That is why ``run_stage`` grew the parameter.
* *The bundle it started from.* It writes a NEW artifacts root (``<name>-kpi``). Arm 3 is compared
  against arm 2, so arm 2 has to still exist afterwards.
* *The normalisation.* ``refit_norm_stats=False``: the stats stay the ones the bundle was trained
  under. Refitting them on simulated deltas would move the scale the weights were fitted to, and every
  head would be reading a different Delta-t language than the one it learned.

**Weights reach the heads through the split on disk**, because that is where a head reads them from
(``_load_processed``). The hook rewrites ``recency_weight`` in the preprocessed train file between
preprocess and train, and **refuses on a length mismatch** rather than broadcasting or truncating: the
split's row order is "sorted unique game ids present", so a head that dropped a game would otherwise
receive every subsequent game's weight shifted by one, which is a silent, plausible-looking wrong answer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import (REPLAY_ARTIFACTS_SUFFIX, REPLAY_EPOCHS, REPLAY_GAME_FRACTION, REPLAY_LR,
                    REPLAY_SIMS_PER_GAME, ROLLOUT_BATCH_SIZE, SEED, SUBSET_MODEL_KEYS)
from models.replay import head_weights, sim_head_errors
from models.registry import STAGE_MODEL_KEYS
from training.replay_corpus import (assert_ids_fit, select_replay_games, sim_frame,
                                    sim_game_id, write_corpus, write_priors)

#: Fraction of the replayed games held out of the weighted pass as its validation split. Small and
#: present rather than absent: one epoch never early-stops, but a head with no validation data reports
#: no val loss at all, and the run report is how this pass is read afterwards.
VAL_FRACTION = 0.05

#: Rewritten between preprocess and train. Matched by substring because each head names its own file
#: ("train.npz", "player_train.npz", "cond_train.npz") and a registry of those names would be a second
#: place to forget a head -- which is the shape of bug this workstream keeps finding.
TRAIN_FILE_MARKER = "train"


# ===================================================================== #
# --- Scoring one game's sims                                          --
# ===================================================================== #

def real_probe_summary(data_dir: str, seasons) -> dict:
    """The real side of the three game-state probes, walked once for the whole pass."""
    from reporting.state_probes import probe_real, summarize
    return summarize(probe_real(data_dir, seasons=seasons, echo=None))


def sim_probe_report(rows, real_summary: dict) -> dict:
    """One sim's probes, in the shape ``head_metrics.probe_gap`` reads.

    ``probe_gap`` skips rows whose real or sim side is missing, which is what makes a per-sim report
    workable at all: most single games contain no blowout fourth quarter and no late-foul state, so
    those rows are ``None`` and simply do not count toward that sim's behaviour gap.
    """
    from reporting.state_probes import _blank, _rows_for_frame, accumulate, probe_game, summarize
    pool = accumulate(_blank(), probe_game(rows))
    return {"rows": _rows_for_frame({"sim": summarize(pool), "real": real_summary})}


def score_game(real_box, real_rows, sim_boxes, sim_frames, real_summary: dict) -> list[dict]:
    """Per-head error for each sim of one game. Lower is better."""
    from models.head_metrics import game_stats

    real_stats = game_stats(real_box, real_rows.to_dict("records"))
    sim_stats, probes = [], []
    for box, frame in zip(sim_boxes, sim_frames):
        rows = frame.to_dict("records")
        sim_stats.append(game_stats(box, rows))
        probes.append(sim_probe_report(rows, real_summary))
    return sim_head_errors(real_stats, sim_stats, probes=probes)


# ===================================================================== #
# --- The rollout                                                      --
# ===================================================================== #

def replay_one_chunk(sim, chunk, real_frames, real_summary, *, n_sims, batch_size,
                     data_dir: str = "./data", echo=print):
    """Simulate and score a handful of games. Returns ``(frames, weights_by_head)``.

    Games are chunked rather than run one at a time because a single game keeps only ~2 sims on the same
    head at once -- pooling is what fills the batch (``simulation/evaluation.simulate_games``).
    """
    from simulation.box_score import generate_box_score
    from simulation.evaluation import _real_starters, simulate_games
    from simulation.game_input import extract_game_input

    specs, ids, inputs = [], [], []
    for gid in chunk:
        frame = real_frames[gid]
        spec = extract_game_input(frame, data_dir=data_dir)
        try:
            home_starters, away_starters = _real_starters(frame)
        except ValueError:
            home_starters = away_starters = None
        specs.append((spec, home_starters, away_starters, "HOME", "AWAY"))
        inputs.append(spec)
        ids.append(int(gid))

    results = simulate_games(sim, specs, n_sims=n_sims, seed0=SEED, batch_size=batch_size,
                             game_ids=ids)

    frames, weights_by_head = [], {}
    for gid, spec, (boxes, histories) in zip(ids, inputs, results):
        real_rows = real_frames[gid]
        sim_frames = [sim_frame(h, spec, real_rows, real_id=gid, sim_index=i)
                      for i, h in enumerate(histories)]
        if len(sim_frames) < 2:
            echo(f"    game {gid}: {len(sim_frames)} sim(s) finished, no sibling baseline -- skipped")
            continue
        real_box = generate_box_score(real_rows)
        per_head = score_game(real_box, real_rows, boxes, sim_frames, real_summary)
        for head, by_index in head_weights(per_head).items():
            for sim_index, weight in by_index.items():
                weights_by_head.setdefault(head, {})[sim_game_id(gid, sim_index)] = weight
        frames.extend(sim_frames)
    return frames, weights_by_head


def merge_weights(into: dict, more: dict) -> dict:
    for head, by_game in more.items():
        into.setdefault(head, {}).update(by_game)
    return into


# ===================================================================== #
# --- The weighted training pass                                       --
# ===================================================================== #

def reweight_split(processed_dir, expected_ids, weights_by_game, *, echo=print) -> list[Path]:
    """Overwrite ``recency_weight`` in every preprocessed train file under ``processed_dir``.

    Every head's split orders its rows by sorted game id, so the weight vector is built in that order
    and its length is the check: a head whose split holds a different number of games would otherwise be
    handed the next game's weight for every row after the first gap.
    """
    from models.replay import replay_weight_array

    ordered = sorted(int(g) for g in expected_ids)
    weights = replay_weight_array(ordered, weights_by_game)
    touched = []
    for path in sorted(Path(processed_dir).glob("*.npz")):
        if TRAIN_FILE_MARKER not in path.name:
            continue
        with np.load(path) as data:
            split = {k: data[k] for k in data.files}
        if "recency_weight" not in split:
            continue
        have = int(split["recency_weight"].shape[0])
        if have != len(ordered):
            raise ValueError(
                f"{path.name} holds {have} games but the pass replayed {len(ordered)}; the weight "
                f"vector cannot be aligned to it. Something dropped games in preprocess -- fix that "
                f"rather than padding, or every weight after the gap lands on the wrong game.")
        split["recency_weight"] = weights
        np.savez_compressed(path, **split)
        touched.append(path)
        echo(f"    reweighted {path.name}: {int((weights > 0).sum())}/{len(ordered)} games carry a weight")
    return touched


def run_replay_pass(state: dict, *, out_root: str | None = None, work_dir: str | None = None,
                    fraction: float = REPLAY_GAME_FRACTION, n_sims: int = REPLAY_SIMS_PER_GAME,
                    seed: int = SEED, batch_size: int = ROLLOUT_BATCH_SIZE,
                    games_per_chunk: int = 4, echo=print) -> dict:
    """Run the whole pass and return its summary (also written to ``out_root/replay_pass.json``)."""
    import pandas as pd

    from data_loading import load_all_cleaned
    from models.pipeline import run_stage
    from simulation.game_simulator import GameSimulator
    from training.subset import load_subset_games

    data_dir = state["data_dir"]
    in_root = state["artifacts_root"]
    out_root = out_root or f"{in_root.rstrip('/')}{REPLAY_ARTIFACTS_SUFFIX}"
    work_dir = Path(work_dir or Path(state.get("processed_dir", "./data/processed")).parent / "replay")
    corpus_dir = work_dir / "corpus"
    tensors_dir = work_dir / "processed"

    subset = load_subset_games()
    if not subset:
        raise SystemExit("no training subset on disk; run `python -m training.subset extract` first")
    games = select_replay_games(subset, fraction=fraction, seed=seed)
    assert_ids_fit(games)
    echo(f"[replay] {len(games)} games x {n_sims} sims = {len(games) * n_sims} game-sims "
         f"(subset of {len(subset)}), from {in_root} -> {out_root}")

    echo("[replay] loading the real rows for the replayed games ...")
    # Filter BEFORE roster parsing: parsing the whole corpus to keep a few hundred games is the
    # ~13M-row literal_eval and most-of-20-GB footprint rung 2 already paid for once (departure 16).
    df = load_all_cleaned(data_dir, parse_rosters=True, game_ids=games)
    wanted = set(games)
    rows = df[df["game_id"].isin(wanted)]
    real_frames = {int(g): part for g, part in rows.groupby("game_id")}
    missing = [g for g in games if g not in real_frames]
    if missing:
        raise SystemExit(f"{len(missing)} replayed games have no rows in {data_dir} (first: {missing[0]})")
    seasons = sorted({int(part["season"].iloc[0]) for part in real_frames.values()})
    real_summary = real_probe_summary(data_dir, seasons)

    sim = GameSimulator.load(artifacts_root=in_root)
    frames, weights_by_head = [], {}
    for start in range(0, len(games), games_per_chunk):
        chunk = games[start:start + games_per_chunk]
        got, weights = replay_one_chunk(sim, chunk, real_frames, real_summary, n_sims=n_sims,
                                        batch_size=batch_size, data_dir=data_dir, echo=echo)
        frames.extend(got)
        merge_weights(weights_by_head, weights)
        echo(f"[replay] {min(start + games_per_chunk, len(games))}/{len(games)} games simulated")

    if not frames:
        raise SystemExit("no sims survived the pass; nothing to train on")
    sim_ids = sorted({int(f["game_id"].iloc[0]) for f in frames})
    echo(f"[replay] writing the corpus: {len(sim_ids)} sim games -> {corpus_dir}")
    write_corpus(frames, corpus_dir)
    write_priors(data_dir, corpus_dir, sim_ids)

    n_val = max(1, int(round(len(sim_ids) * VAL_FRACTION)))
    val_ids = set(sim_ids[-n_val:])
    train_ids = [g for g in sim_ids if g not in val_ids]
    partition = (set(train_ids), val_ids, set())

    def on_preprocessed(key: str, model) -> None:
        by_game = weights_by_head.get(key, {})
        echo(f"[replay] '{key}': {len(by_game)} of {len(train_ids)} sims kept a positive advantage")
        reweight_split(model.processed_dir, train_ids, by_game, echo=echo)

    echo(f"[replay] one weighted pass, {REPLAY_EPOCHS} epoch(s) at lr {REPLAY_LR} -> {out_root}")
    trained = run_stage(
        str(corpus_dir), partition, artifacts_root=out_root,
        warm_start=True, warm_start_root=in_root, refit_norm_stats=False,
        epochs=REPLAY_EPOCHS, lr=REPLAY_LR, batch_size=state.get("batch_size", 64),
        report=True, run_name=f"{state.get('run_name', 'replay')}-kpi",
        subset_keys=SUBSET_MODEL_KEYS, subset_train_games=set(train_ids),
        processed_dir=str(tensors_dir), on_preprocessed=on_preprocessed,
    )

    summary = {
        "in_root": in_root, "out_root": out_root,
        "games": len(games), "sims_per_game": n_sims, "sim_games": len(sim_ids),
        "train_games": len(train_ids), "val_games": len(val_ids),
        "kept_by_head": {h: len(w) for h, w in sorted(weights_by_head.items())},
        "heads_trained": trained, "epochs": REPLAY_EPOCHS, "lr": REPLAY_LR,
        "corpus_dir": str(corpus_dir), "processed_dir": str(tensors_dir),
    }
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay_pass.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    missing_heads = [k for k in STAGE_MODEL_KEYS if k not in weights_by_head]
    if missing_heads:
        echo(f"[replay] NOTE: no sim earned a positive advantage for {missing_heads} -- those heads "
             f"trained on an all-zero weight, which is a no-op pass rather than a change")
    echo(f"[replay] done. Summary -> {out / 'replay_pass.json'}")
    return summary


__all__ = ["VAL_FRACTION", "merge_weights", "real_probe_summary", "replay_one_chunk",
           "reweight_split", "run_replay_pass", "score_game", "sim_probe_report"]
