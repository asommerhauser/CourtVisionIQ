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
import threading
import time
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


def _echo(msg: str) -> None:
    """``print`` that reaches a ``nohup`` log as it happens.

    Every other long job here streams only because Keras flushes stdout with each progress-bar
    update, carrying our buffered prints out with it. The rollout half of this pass has no Keras
    in it, so without the flush its progress sat in the buffer for the whole simulation.
    """
    print(msg, flush=True)


#: Seconds between heartbeat lines while the rollout runs. On a timer, not on completions: a chunk's
#: sims all land together at the end, so a completion-driven line goes quiet for the whole chunk.
HEARTBEAT_SECONDS = 60.0


def _rss_gb() -> float | None:
    """Resident memory of this process in GB, or None where /proc is absent (Windows).

    Printed on every heartbeat because the likeliest silent death here is the OOM killer: SIGKILL
    leaves no traceback, so the only evidence is the last RSS the log recorded before it stopped.
    """
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    return None


def _mem() -> str:
    rss = _rss_gb()
    return f"RSS {rss:.1f} GB" if rss is not None else "RSS n/a"


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
                     data_dir: str = "./data", echo=_echo, progress=None):
    """Simulate and score a handful of games. Returns one ``(game_id, frames, weights_by_head)`` per game.

    Per game rather than merged, so each game can be saved the moment its chunk lands: a shard that
    dies keeps every game it finished, and nothing piles up in memory across the whole pass.

    ``progress`` (a ``batched_rollout._Progress``) is shared across chunks so the heartbeat counts
    sims for the whole pass, not just the current chunk.

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
                             game_ids=ids, progress=progress)

    out = []
    for gid, spec, (boxes, histories) in zip(ids, inputs, results):
        real_rows = real_frames[gid]
        sim_frames = [sim_frame(h, spec, real_rows, real_id=gid, sim_index=i)
                      for i, h in enumerate(histories)]
        if len(sim_frames) < 2:
            # Still returned (empty), so the game is marked done and a resume does not retry it.
            echo(f"    game {gid}: {len(sim_frames)} sim(s) finished, no sibling baseline -- skipped")
            out.append((gid, [], {}))
            continue
        real_box = generate_box_score(real_rows)
        per_head = score_game(real_box, real_rows, boxes, sim_frames, real_summary)
        weights_by_head = {}
        for head, by_index in head_weights(per_head).items():
            for sim_index, weight in by_index.items():
                weights_by_head.setdefault(head, {})[sim_game_id(gid, sim_index)] = weight
        out.append((gid, sim_frames, weights_by_head))
    return out


def merge_weights(into: dict, more: dict) -> dict:
    for head, by_game in more.items():
        into.setdefault(head, {}).update(by_game)
    return into


# ===================================================================== #
# --- The per-game store, and the shard that fills it                  --
# ===================================================================== #
# The rollout is GIL-bound Python at ~3.5 sims/min per process, so 520 games x 10 sims in one process
# is ~25 hours with the card mostly idle -- and the first attempt held every sim in memory until the
# end, so when it died nothing was left. Now each game lands on disk as it finishes, under the same
# ``games/<id>/record.json`` marker eval_pool counts, and N processes fill disjoint slices of it.
# A rerun resumes: a game with a record is never simulated twice.

def work_dir_for(state: dict, work_dir=None) -> Path:
    return Path(work_dir or Path(state.get("processed_dir", "./data/processed")).parent / "replay")


def _game_dir(sims_dir: Path, gid) -> Path:
    return Path(sims_dir) / "games" / str(int(gid))


def done_games(sims_dir) -> set[int]:
    return {int(p.parent.name) for p in (Path(sims_dir) / "games").glob("*/record.json")}


def save_game(sims_dir, gid, frames, weights_by_head) -> None:
    """Frames first, record.json LAST and atomically: the record is the done-marker."""
    import pickle

    d = _game_dir(sims_dir, gid)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "frames.pkl", "wb") as fh:
        pickle.dump(frames, fh, protocol=pickle.HIGHEST_PROTOCOL)
    record = {"sims": len(frames),
              "weights": {h: {str(k): float(v) for k, v in w.items()} for h, w in weights_by_head.items()}}
    tmp = d / "record.json.tmp"
    tmp.write_text(json.dumps(record), encoding="utf-8")
    tmp.replace(d / "record.json")


def load_games(sims_dir, games) -> tuple[list, dict]:
    """Every saved game's frames and weights, merged in game order."""
    import pickle

    frames, weights_by_head = [], {}
    for gid in games:
        d = _game_dir(sims_dir, gid)
        record = json.loads((d / "record.json").read_text(encoding="utf-8"))
        if record["sims"]:
            with open(d / "frames.pkl", "rb") as fh:
                frames.extend(pickle.load(fh))
        merge_weights(weights_by_head, {h: {int(k): v for k, v in w.items()}
                                        for h, w in record["weights"].items()})
    return frames, weights_by_head


def _manifest(sims_dir) -> dict:
    return json.loads((Path(sims_dir) / "manifest.json").read_text(encoding="utf-8"))


def run_replay_shard(state: dict, index: int, total: int, *, work_dir=None,
                     batch_size: int = ROLLOUT_BATCH_SIZE, games_per_chunk: int = 4,
                     echo=_echo) -> int:
    """Simulate and save this shard's slice (``games[index-1::total]``) of the pass. Returns games made.

    Everything that must be identical across shards -- the game list, sims per game, the bundle, the
    real side of the probes -- comes from the manifest the parent wrote, never recomputed here.
    """
    import pickle

    from data_loading import load_all_cleaned
    from simulation.batched_rollout import _Progress
    from simulation.game_simulator import GameSimulator

    tag = f"[replay {index}/{total}]"
    sims_dir = work_dir_for(state, work_dir) / "sims"
    manifest = _manifest(sims_dir)
    n_sims, in_root, data_dir = manifest["n_sims"], manifest["in_root"], state["data_dir"]
    mine = manifest["games"][index - 1::total]
    done = done_games(sims_dir)
    todo = [g for g in mine if g not in done]
    echo(f"{tag} {len(todo)} of {len(mine)} games to simulate ({len(mine) - len(todo)} already saved)")
    if not todo:
        return 0
    with open(sims_dir / "real_summary.pkl", "rb") as fh:
        real_summary = pickle.load(fh)

    df = load_all_cleaned(data_dir, parse_rosters=True, game_ids=todo)
    real_frames = {int(g): part for g, part in df[df["game_id"].isin(set(todo))].groupby("game_id")}
    del df
    echo(f"{tag} real rows loaded, loading {in_root} ... ({_mem()})")
    sim = GameSimulator.load(artifacts_root=in_root)
    echo(f"{tag} simulator ready ({_mem()})")

    total_sims = len(todo) * n_sims
    progress = _Progress(total=total_sims, enabled=False)   # counts only; the pulse prints
    started = time.monotonic()
    made = [0]
    stop = threading.Event()

    def _pulse() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            with progress._lock:
                sims_done, passes = progress.completed, progress.passes
            mins = (time.monotonic() - started) / 60.0
            echo(f"{tag} heartbeat: {made[0]}/{len(todo)} games, {sims_done}/{total_sims} sims, "
                 f"{mins:.1f} min, {passes} fwd passes, {_mem()}")

    threading.Thread(target=_pulse, daemon=True).start()
    try:
        for start in range(0, len(todo), games_per_chunk):
            chunk = todo[start:start + games_per_chunk]
            for gid, frames, weights in replay_one_chunk(
                    sim, chunk, real_frames, real_summary, n_sims=n_sims, batch_size=batch_size,
                    data_dir=data_dir, echo=echo, progress=progress):
                save_game(sims_dir, gid, frames, weights)
                real_frames.pop(gid, None)
            made[0] = min(start + games_per_chunk, len(todo))
            mins = (time.monotonic() - started) / 60.0
            echo(f"{tag} {made[0]}/{len(todo)} games saved, {mins:.1f} min, "
                 f"~{mins / made[0] * (len(todo) - made[0]):.0f} min left, {_mem()}")
    finally:
        stop.set()
    return made[0]


def _run_shards(state_path: str, sims_dir: Path, n_games: int, n_procs: int, *, echo) -> None:
    """Launch ``train.py --replay-shard i/N`` children and log one progress line a minute."""
    import sys

    import eval_pool

    logs = sims_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    base = len(done_games(sims_dir))
    last = [0.0]

    def build(wave: int, remaining: int) -> list:
        n = max(1, min(n_procs, remaining))
        cards = eval_pool.visible_gpus()
        return [eval_pool.Shard(
                    index=i, total=n, log=logs / f"wave{wave}-shard{i}of{n}.log",
                    cmd=[sys.executable, "train.py", "--replay-shard", f"{i}/{n}",
                         "--state", str(state_path)],
                    gpu=cards[(i - 1) % len(cards)] if len(cards) > 1 else None)
                for i in range(1, n + 1)]

    def tick() -> None:
        now = time.monotonic()
        if now - last[0] < HEARTBEAT_SECONDS:
            return
        last[0] = now
        done = len(done_games(sims_dir))
        mins = (now - started) / 60.0
        rate = (done - base) / mins if mins > 0 else 0.0
        line = (f"[replay] {done}/{n_games} games saved, {mins:.1f} min elapsed, "
                f"{rate * 60:.0f} games/hr")
        if rate > 0:
            line += f", ~{(n_games - done) / rate:.0f} min left"
        echo(line + f"  (per-shard logs: {logs})")

    eval_pool.run_waves(build=build, run_dir=sims_dir, total_games=n_games, echo=echo,
                        render=False, on_tick=tick)


# ===================================================================== #
# --- The weighted training pass                                       --
# ===================================================================== #

def reweight_split(processed_dir, expected_ids, weights_by_game, *, echo=_echo) -> list[Path]:
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
                    games_per_chunk: int = 4, procs=None, state_path: str | None = None,
                    echo=_echo) -> dict:
    """Run the whole pass and return its summary (also written to ``out_root/replay_pass.json``).

    ``procs`` (an int or ``"auto"``) splits the rollout across that many ``--replay-shard`` processes,
    sized like ``evaluate.py --procs``; it needs ``state_path`` so the children read the same run.
    Finished games are kept under ``<work_dir>/sims``, so a rerun picks up where the last one died.
    """
    import pickle
    import shutil

    from data_loading import load_all_cleaned
    from models.pipeline import run_stage
    from training.subset import load_subset_games

    data_dir = state["data_dir"]
    in_root = state["artifacts_root"]
    out_root = out_root or f"{in_root.rstrip('/')}{REPLAY_ARTIFACTS_SUFFIX}"
    work_dir = work_dir_for(state, work_dir)
    corpus_dir = work_dir / "corpus"
    tensors_dir = work_dir / "processed"
    sims_dir = work_dir / "sims"

    subset = load_subset_games()
    if not subset:
        raise SystemExit("no training subset on disk; run `python -m training.subset extract` first")
    games = select_replay_games(subset, fraction=fraction, seed=seed)
    assert_ids_fit(games)
    echo(f"[replay] {len(games)} games x {n_sims} sims = {len(games) * n_sims} game-sims "
         f"(subset of {len(subset)}), from {in_root} -> {out_root}")

    # Saved sims are only reusable by a pass with the same bundle, games and sims per game. Anything
    # else would train on another model's rollouts, so a mismatch starts the store over.
    want = {"in_root": in_root, "games": [int(g) for g in games], "n_sims": int(n_sims), "seed": int(seed)}
    if (sims_dir / "manifest.json").exists() and _manifest(sims_dir) != want:
        echo(f"[replay] saved sims in {sims_dir} are from a different pass setup -- discarding them")
        shutil.rmtree(sims_dir)
    sims_dir.mkdir(parents=True, exist_ok=True)
    if not (sims_dir / "manifest.json").exists():
        (sims_dir / "manifest.json").write_text(json.dumps(want), encoding="utf-8")

    # An empty file is a pass killed mid-write before the write was atomic; treat it as missing.
    summary_path = sims_dir / "real_summary.pkl"
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        echo(f"[replay] walking the real side of the probes ... ({_mem()})")
        # Filter BEFORE roster parsing: parsing the whole corpus to keep a few hundred games is the
        # ~13M-row literal_eval and most-of-20-GB footprint rung 2 already paid for once (departure 16).
        df = load_all_cleaned(data_dir, parse_rosters=True, game_ids=games)
        found = set(df["game_id"].unique().tolist())
        missing = [g for g in games if g not in found]
        if missing:
            raise SystemExit(f"{len(missing)} replayed games have no rows in {data_dir} (first: {missing[0]})")
        seasons = sorted({int(s) for s in df["season"].unique()})
        del df
        # Summarize BEFORE opening the file, then swap it in: a kill during the slow walk must not
        # leave an empty pickle that every later run trusts because it exists.
        real_summary = real_probe_summary(data_dir, seasons)
        tmp = summary_path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(real_summary, fh)
        tmp.replace(summary_path)

    remaining = len(games) - len(done_games(sims_dir))
    if remaining:
        import eval_pool

        n_procs, why = eval_pool.autosize_procs(holdout_games=remaining, requested=procs or 1)
        echo(f"[replay] {remaining} games to simulate, {why}")
        if n_procs > 1:
            if not state_path:
                raise SystemExit("--procs needs the state file path so the shards read the same run")
            _run_shards(state_path, sims_dir, len(games), n_procs, echo=echo)
        else:
            run_replay_shard(state, 1, 1, work_dir=work_dir, batch_size=batch_size,
                             games_per_chunk=games_per_chunk, echo=echo)
    unfinished = [g for g in games if g not in done_games(sims_dir)]
    if unfinished:
        raise SystemExit(f"[replay] {len(unfinished)} games never finished (first: {unfinished[0]}). "
                         f"Rerun the same command to resume; shard logs are in {sims_dir / 'logs'}")

    echo(f"[replay] all {len(games)} games simulated; loading the saved sims ... ({_mem()})")
    frames, weights_by_head = load_games(sims_dir, games)

    if not frames:
        raise SystemExit("no sims survived the pass; nothing to train on")
    sim_ids = sorted({int(f["game_id"].iloc[0]) for f in frames})
    echo(f"[replay] writing the corpus: {len(sim_ids)} sim games -> {corpus_dir} ({_mem()})")
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
