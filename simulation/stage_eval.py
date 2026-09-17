"""
stage_eval.py — per-stage curriculum evaluation: predict the next block of real games.

After a curriculum stage finishes training, we predict its sequential holdout (the next
``HOLDOUT_GAMES`` real games) by simulating each ``STAGE_SIMS`` times. This wraps the existing
``simulation.evaluation`` scoring with two stage-specific additions the user asked for:

  1. **Per-game prediction folders** (descriptive names) under
     ``artifacts/predictions/<stage_name>/`` holding, for each game: the actual box score, the
     *averaged* predicted box score over the sims (game score included), the actual play-by-play,
     and every generated play-by-play. Each game's evaluation record is also cached
     (``record.json``) so an interrupted eval **resumes** — finished games are reloaded, not
     re-simulated.
  2. **A stage-level overall report** (the standard HTML + Parquet eval report) capturing win/
     spread/box accuracy and the std across the sims, written once all games are done.
"""
from __future__ import annotations

# Import TensorFlow before pandas-using project modules (see evaluation.py / main.py).
try:  # noqa: SIM105
    import tensorflow  # noqa: F401
except Exception:
    pass

import atexit
import json
import os
import re
import signal
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from config import (EVAL_GAMES_PER_BATCH, EVAL_MAX_CONSECUTIVE_GAME_FAILURES, EVAL_POOL_JOBS,
                    HOLDOUT_MANIFEST_NAME, ROLLOUT_BATCH_SIZE, STAGE_SIMS)
from data_loading import load_all_cleaned
from models.artifacts import DEFAULT_ARTIFACTS_ROOT
from reporting.game_report import render_game_html
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT
from simulation.box_score import BoxScore, PlayerLine, generate_box_score
from simulation.eval_metrics import _aggregate, print_summary, reported_sims
from simulation.box_score import period_box_scores
from simulation.evaluation import build_game_record, simulate_games
from simulation.game_input import extract_game_input
from simulation.game_simulator import GameSimulator
from simulation.predict_game import (
    CLEANED_COLUMNS,
    DEFAULT_OUTPUT_ROOT,
    _real_starters,
    history_to_cleaned_frame,
)
from simulation.stats import BOX_STATS


def _game_labels(game) -> tuple[str, str, str]:
    """Team labels + a filesystem-safe per-game folder name from a cleaned game's first row."""
    first = game.iloc[0]
    gid = int(game["game_id"].iloc[0])

    def _clean(val, default):
        s = str(val).strip() if val is not None else ""
        return s if s and s.lower() != "nan" else default

    home = _clean(first.get("home_team"), "HOME")
    away = _clean(first.get("away_team"), "AWAY")
    date = _clean(first.get("game_date"), "")
    label = f"game{gid}_{date}_{away}at{home}" if date else f"game{gid}_{away}at{home}"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label)
    return home, away, label


def _averaged_box(record: dict, home_team: str, away_team: str) -> BoxScore:
    """Build a BoxScore from a record's per-sim per-player averages (game score included)."""
    def _lines(side: str) -> list[PlayerLine]:
        out = []
        for name, stats in record["player_avg"][side].items():
            pl = PlayerLine(player=name)
            for f in BOX_STATS:
                setattr(pl, f, round(float(stats[f]), 1))
            out.append(pl)
        return out

    return BoxScore(home=_lines("home"), away=_lines("away"),
                    home_score=round(float(record["pred_home_score"]), 1),
                    away_score=round(float(record["pred_away_score"]), 1),
                    home_team=home_team, away_team=away_team)


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + os.replace, so no reader ever sees a half-written file.

    ``record.json`` is the run's completion marker: ``eval_pool.finished_games`` counts it and the
    supervisor now rebuilds the report from these files *while shards are still writing them*. A
    plain write_text is visible (and empty) the instant it is created.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_game_folder(out_dir: Path, game, boxes, record,
                       home_team: str, away_team: str) -> None:
    """Persist one game's actual + averaged-prediction box scores, an HTML report and its record.

    The per-sim play-by-plays are NOT written here -- :class:`_PbpSink` streams each one out as its
    sim finishes, so a killed process keeps the sims it already ran. ``record.json`` is written last
    and atomically: it is what marks this game done.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    actual_box = generate_box_score(game, home_team=home_team, away_team=away_team)
    actual_box.to_frame("home").to_csv(out_dir / "actual_boxscore_home.csv", index=False)
    actual_box.to_frame("away").to_csv(out_dir / "actual_boxscore_away.csv", index=False)
    (out_dir / "actual_boxscore.txt").write_text(actual_box.render(), encoding="utf-8")

    pred_box = _averaged_box(record, home_team, away_team)
    pred_box.to_frame("home").to_csv(out_dir / "pred_boxscore_home.csv", index=False)
    pred_box.to_frame("away").to_csv(out_dir / "pred_boxscore_away.csv", index=False)
    (out_dir / "pred_boxscore.txt").write_text(pred_box.render(), encoding="utf-8")

    # Per-game HTML: predicted (mean) / actual (raw) / variance box scores.
    (out_dir / "game.html").write_text(
        render_game_html(record, home_team=home_team, away_team=away_team), encoding="utf-8")

    run_meta = {
        "game_id": record["game_id"],
        "home_team": home_team, "away_team": away_team,
        "predicted_score": {"home": record["pred_home_score"], "away": record["pred_away_score"]},
        "actual_score": {"home": record["actual_home_score"], "away": record["actual_away_score"]},
        "win_prob_home": record["win_prob_home"],
        "per_sim_scores": [{"home": b.home_score, "away": b.away_score} for b in boxes],
        "n_sims": len(boxes),
    }
    _atomic_write(out_dir / "run.json", json.dumps(run_meta, indent=2))
    # Last, and atomically: this file is the completion marker the resume path and the pool read.
    name = "record.partial.json" if record.get("partial") else "record.json"
    _atomic_write(out_dir / name, json.dumps(record, indent=2))


class _PbpSink:
    """Writes each sim's play-by-play the moment that sim finishes.

    The old path held every history until the pool drained and then wrote all of them at once, so a
    game killed at sim 99 of 100 left nothing on disk at all. Writing inline on the slot thread
    costs ~50ms against a ~100s sim (0.05%); a queue and a writer thread would cost a failure mode
    where the writer dies, the queue fills and every slot blocks -- which is the disaster this is
    supposed to prevent.
    """

    def __init__(self, chunk: list[dict], *, n_sims: int) -> None:
        self.chunk = chunk
        # Width from the REQUESTED sim count, not from a finished list (there isn't one any more).
        # At 100 sims a fixed :02d sorts sim_9 after sim_100.
        self.width = max(2, len(str(n_sims)))
        # Per-period team totals, accumulated here because this is the only place a history is
        # in hand. Splitting the history NOW and keeping the small result is what lets the
        # per-quarter record exist without holding every history alive to the end of the chunk --
        # which is the 8.6 GB record spike the lineup-state branch removed, and not worth
        # reintroducing for a table of team totals.
        self.period_boxes: list[list] = [[None] * n_sims for _ in chunk]

    def prepare(self) -> None:
        """Create each game's playbyplay/ dir, clear any stale sims, write the actual play-by-play.

        The clear matters on a re-run: a first attempt killed at sim 43 of 100, re-run at 21 sims,
        would otherwise leave sim_022..sim_043 orphaned next to a record.json claiming 21. A
        salvaged record.partial.json goes too -- a full re-run supersedes it. If the run never comes
        back to this game, nothing here runs and the partial data stays put.
        """
        for p in self.chunk:
            pbp_dir = p["out_dir"] / "playbyplay"
            pbp_dir.mkdir(parents=True, exist_ok=True)
            for stale in pbp_dir.glob("sim_*_playbyplay.csv"):
                try:
                    stale.unlink()
                except OSError:
                    pass
            try:
                (p["out_dir"] / "record.partial.json").unlink(missing_ok=True)
            except OSError:
                pass
            p["game"].reindex(columns=CLEANED_COLUMNS).to_csv(
                pbp_dir / "actual_playbyplay.csv", index=False)

    def __call__(self, g: int, s: int, history, box) -> None:
        p = self.chunk[g]
        try:
            self.period_boxes[g][s] = period_box_scores(
                history, home_team=p["home_team"], away_team=p["away_team"])
        except Exception as e:      # noqa: BLE001 - a bad split costs the quarter table, not the sim
            print(f"  game {p['gid']} sim {s + 1}: could not split by period ({e!r})")
        try:
            frame = history_to_cleaned_frame(history, p["spec"], game_id=int(p["gid"]))
            path = p["out_dir"] / "playbyplay" / f"sim_{s + 1:0{self.width}d}_playbyplay.csv"
            frame.to_csv(path, index=False)
        except Exception as e:      # noqa: BLE001 - a bad write costs one pbp, not the sim
            print(f"  game {p['gid']} sim {s + 1}: could not write play-by-play ({e!r})")


@dataclass
class _InFlight:
    """One game currently being simulated, and the box list filling up as its sims land."""
    gid: int
    out_dir: Path
    game: object
    home_team: str
    away_team: str
    n_sims: int
    seed0: int
    boxes: list = field(default_factory=list)


_INFLIGHT: list[_InFlight] = []
_SALVAGE_LOCK = threading.Lock()
_SALVAGED = False
_HANDLERS_INSTALLED = False


def _salvage_one(g: _InFlight, reason: str) -> None:
    """Write what this game has finished so far as ``record.partial.json``.

    Deliberately NOT ``record.json``: that stays the sole completion marker, so ``finished_games``
    does not count this game, the remainder wave re-runs it at full precision, and ``reported_sims``
    (which takes the min across records) can never relabel a 100-sim run from one killed game.
    The data still lands -- the streamed sim CSVs plus these box scores.
    """
    if (g.out_dir / "record.json").exists():
        return
    boxes = [b for b in list(g.boxes) if b is not None]
    if not boxes:
        return
    record = build_game_record(g.game, boxes, n_sims=len(boxes), seed_base=g.seed0,
                              home_team=g.home_team, away_team=g.away_team)
    record["partial"] = True
    record["requested_n_sims"] = g.n_sims
    _write_game_folder(g.out_dir, g.game, boxes, record, g.home_team, g.away_team)
    print(f"  [salvage] game {g.gid}: {len(boxes)}/{g.n_sims} sims saved to record.partial.json "
          f"({reason}). The remainder wave will re-run this game at full precision.")


def _salvage_all(reason: str) -> None:
    """Finalize every in-flight game. Runs from a signal handler, so it must never raise."""
    global _SALVAGED
    with _SALVAGE_LOCK:
        if _SALVAGED:
            return
        pending = list(_INFLIGHT)
        if not pending:
            return              # nothing to lose yet; don't burn the once-only guard on a no-op
        _SALVAGED = True
    print(f"\n[salvage] {reason}: writing partial records for {len(pending)} in-flight game(s)...")
    for g in pending:
        try:
            _salvage_one(g, reason)
        except Exception as e:      # noqa: BLE001 - one game's failure must not skip the others
            print(f"  [salvage] game {g.gid}: could not save ({e!r})")


def _on_sigterm(signum, frame):     # pragma: no cover - exercised by the smoke test, not pytest
    _salvage_all(f"signal {signum}")
    # Re-raise natively so the exit status is a real signal death and the pool's _reap sees -15.
    try:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    except Exception:               # noqa: BLE001
        os._exit(128 + signum)


def _install_salvage_handlers() -> None:
    """SIGTERM (RunPod pod-stop, which never runs atexit) plus atexit for everything else.

    No SIGINT handler on purpose: KeyboardInterrupt already propagates and atexit fires, so Ctrl-C
    in the cviq shell keeps behaving exactly as it does today.
    """
    global _HANDLERS_INSTALLED
    if _HANDLERS_INSTALLED:
        return
    _HANDLERS_INSTALLED = True
    atexit.register(_salvage_all, "atexit")
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError, AttributeError):
        pass                        # not the main thread (resident shell), or no SIGTERM


def evaluate_stage(stage_name: str, *, sim=None, df=None, run_label: str | None = None,
                   holdout_ids: list[int] | None = None,
                   n_sims: int = STAGE_SIMS, max_new: int | None = None,
                   report_every: int | None = None,
                   data_dir: str = "./data", processed_dir: str = "./data/processed",
                   artifacts_root: str = DEFAULT_ARTIFACTS_ROOT,
                   reports_root: str = DEFAULT_REPORTS_ROOT,
                   predictions_root: str = DEFAULT_OUTPUT_ROOT, seed0: int = 0,
                   batch_size: int = ROLLOUT_BATCH_SIZE,
                   games_per_batch: int = EVAL_GAMES_PER_BATCH,
                   results_run_dir: str | Path | None = None,
                   write_report: bool = True, window: int = 0) -> dict:
    """Predict a stage's holdout games (``n_sims`` each), write per-game folders + a stage report.

    ``holdout_ids`` defaults to the manifest the stage's preprocess wrote (``holdout_games.json``).
    Finished games (those with a cached ``record.json``) are reloaded rather than re-simulated, so a
    killed eval resumes. ``max_new`` caps how many *new* games are simulated this call (the rest are
    left for a later call) — used to predict the holdout a batch at a time; pass ``None`` to run the
    whole holdout straight through in one process (a paid-GPU run).

    ``sim`` accepts an already-loaded :class:`~simulation.game_simulator.GameSimulator` instead of
    loading one from ``artifacts_root`` -- how the resident ``cviq`` shell runs many evals against
    one set of weights without paying the ~11-head rebuild each time. ``df`` likewise accepts an
    already-parsed cleaned frame, skipping a full re-read of every season CSV. ``run_label`` is the
    run's own name for the report (defaults to ``stage_name``, which is the *model* name).

    ``report_every`` (when set) writes
    an intermediate report every N newly-finished games so progress is visible during a straight run;
    a final report is always written at the end. Already-finished games still load into the report, so
    it covers everything done so far. Returns the report dict (with ``run_dir``, ``done``, ``total``).

    ``write_report=False`` still builds and returns the aggregate but writes no report files
    (report.html / report.json / data/*.parquet) -- used by sharded evals, where several processes
    share one run dir and each would otherwise rewrite those files wholesale from its own slice,
    clobbering the others. Per-game folders are keyed by game id and stay disjoint, so the merge is
    a later ``write_report=True`` pass over the full holdout (``evaluate.py --report-only``).
    """
    from reporting.eval_report import build_report, write_eval_report

    if holdout_ids is None:
        manifest = Path(processed_dir) / HOLDOUT_MANIFEST_NAME
        if not manifest.exists():
            raise FileNotFoundError(f"No holdout manifest at {manifest}; preprocess the stage first.")
        holdout_ids = [int(g) for g in json.loads(manifest.read_text(encoding="utf-8"))]
    if not holdout_ids:
        raise ValueError(f"stage '{stage_name}' has an empty holdout — nothing to evaluate.")

    if df is None:
        df = load_all_cleaned(data_dir, parse_rosters=True)
    # Results layout (new): per-game folders under <results_run_dir>/games/, report at the run root.
    # Legacy layout: per-game folders under artifacts/predictions/<stage_name>/, report in reports/.
    results_run_dir = Path(results_run_dir) if results_run_dir is not None else None
    stage_dir = (results_run_dir / "games") if results_run_dir is not None \
        else (Path(predictions_root) / stage_name)
    # None -> lazily loaded below only if there's an unfinished game to simulate. A caller-supplied
    # sim is used as-is (and never unloaded here; the shell owns its lifetime).

    def _flush_report() -> dict:
        """Build (and unless ``write_report=False``, write) the report over everything finished so
        far; return the report dict."""
        aggregate = _aggregate(records)
        rep = build_report(records=records, aggregate=aggregate,
                           n_sims=reported_sims(records, default=n_sims),
                           run_name=run_label or stage_name, window=window)
        if not write_report:
            rd = results_run_dir if results_run_dir is not None else Path(reports_root)
        elif results_run_dir is not None:
            rd = write_eval_report(rep, run_dir=results_run_dir)
        else:
            rd = write_eval_report(rep, reports_root=reports_root)
        print_summary(aggregate, len(records), rep["n_sims"])
        rep["run_dir"] = str(rd)
        rep["predictions_dir"] = str(stage_dir)
        rep["done"] = len(records)
        rep["total"] = len(holdout_ids)
        return rep

    records: list[dict] = []
    new_done = 0

    # Pass 1: reload cached (finished) games; collect the rest as pending work.
    pending: list[dict] = []
    for gid in holdout_ids:
        game = df[df["game_id"] == int(gid)].sort_values("time")
        if game.empty:
            print(f"  game {gid}: not found in cleaned data — skipping")
            continue
        home_team, away_team, label = _game_labels(game)
        out_dir = stage_dir / label

        cached = out_dir / "record.json"
        if cached.exists():
            print(f"  game {gid}: already evaluated -> reusing {cached}")
            records.append(json.loads(cached.read_text(encoding="utf-8")))
            continue

        pending.append({"gid": gid, "game": game, "home_team": home_team,
                        "away_team": away_team, "out_dir": out_dir})

    if max_new is not None:
        pending = pending[:max_new]  # cap NEW games this call; the rest wait for a later call

    # Pass 2: simulate the pending games in pools of ``games_per_batch`` so each batched rollout
    # fills the GPU (one game alone leaves it ~10% utilized). Resolve each game's matchup + real
    # starters once, then pool their sims into a single run.
    # Pool width is capped in *jobs*, not games: the batched rollout keeps every finished history
    # until the pool drains, so 6 games x 100 sims would hold 600 of them. At STAGE_SIMS this is a
    # no-op (126/21 = 6); at 100 sims it drops to one game per pool, which also lands each game's
    # record.json as soon as it finishes.
    games_per_batch = max(1, min(games_per_batch, EVAL_POOL_JOBS // max(1, n_sims)))

    if pending:
        _install_salvage_handlers()
        if sim is None:
            sim = GameSimulator.load(artifacts_root=artifacts_root)
        for p in pending:
            p["spec"] = extract_game_input(p["game"])
            try:
                p["home_starters"], p["away_starters"] = _real_starters(p["game"])
            except ValueError:
                p["home_starters"] = p["away_starters"] = None

        total_new = len(pending)
        consecutive_failures = 0
        for start in range(0, total_new, games_per_batch):
            chunk = pending[start:start + games_per_batch]
            inflight: list[_InFlight] = []
            try:
                gids = ", ".join(str(p["gid"]) for p in chunk)
                print(f"  simulating games {start + 1}-{start + len(chunk)}/{total_new} "
                      f"({n_sims} sims each, {len(chunk)} games pooled): {gids} ...")
                game_specs = [(p["spec"], p["home_starters"], p["away_starters"],
                               p["home_team"], p["away_team"]) for p in chunk]

                # Stream each sim to disk as it lands, and expose the boxes collected so far so a
                # SIGTERM mid-game can still write what has been run.
                sink = _PbpSink(chunk, n_sims=n_sims)
                sink.prepare()
                boxes_out: list[list] = [[None] * n_sims for _ in chunk]
                inflight = [_InFlight(gid=p["gid"], out_dir=p["out_dir"], game=p["game"],
                                      home_team=p["home_team"], away_team=p["away_team"],
                                      n_sims=n_sims, seed0=seed0, boxes=boxes_out[i])
                            for i, p in enumerate(chunk)]
                _INFLIGHT.extend(inflight)

                results = simulate_games(sim, game_specs, n_sims=n_sims, seed0=seed0,
                                         batch_size=batch_size, show_progress=True,
                                         game_ids=[p["gid"] for p in chunk],
                                         on_sim=sink, boxes_out=boxes_out)
                for _ci, (p, (boxes, _)) in enumerate(zip(chunk, results)):
                    try:
                        if not boxes:
                            print(f"  game {p['gid']}: every sim failed; skipping "
                                  f"(the remainder wave will retry it).")
                            continue
                        if len(boxes) < n_sims:
                            print(f"  game {p['gid']}: {n_sims - len(boxes)} of {n_sims} sims "
                                  f"failed; recording it at {len(boxes)} sims.")
                        # len(boxes), not n_sims: a record must say what it was actually built
                        # from, or the report's precision is a fiction.
                        record = build_game_record(
                            p["game"], boxes, n_sims=len(boxes), seed_base=seed0,
                            home_team=p["home_team"], away_team=p["away_team"],
                            period_boxes=[q for q in sink.period_boxes[_ci]
                                          if q is not None])
                        _write_game_folder(p["out_dir"], p["game"], boxes, record,
                                           p["home_team"], p["away_team"])
                        records.append(record)
                        new_done += 1
                        if report_every and new_done % report_every == 0:
                            print(f"  [report] {new_done} new games done — flushing intermediate "
                                  f"report...")
                            _flush_report()
                    except Exception as e:      # noqa: BLE001 - one game must not end the shard
                        print(f"  game {p['gid']}: FAILED to finalize ({e!r}); skipping.")
                        traceback.print_exc()
                consecutive_failures = 0
            except Exception as e:              # noqa: BLE001 - nor must one chunk
                consecutive_failures += 1
                print(f"  games {start + 1}-{start + len(chunk)} FAILED ({e!r}); skipping.")
                traceback.print_exc()
                # A CUDA OOM usually poisons TF for the life of the process, so grinding through
                # every remaining game to fail identically just burns GPU hours. Die instead and
                # let the pool's remainder wave respawn with a clean context.
                if consecutive_failures >= EVAL_MAX_CONSECUTIVE_GAME_FAILURES:
                    print(f"  {consecutive_failures} chunks failed in a row — stopping this "
                          f"process so the pool can restart it cleanly.")
                    raise
            finally:
                for g in inflight:
                    try:
                        _INFLIGHT.remove(g)
                    except ValueError:
                        pass

    report = _flush_report()
    print(f"\n  {len(records)}/{len(holdout_ids)} holdout games done "
          f"({new_done} predicted this call)")
    print(f"  per-game predictions -> {stage_dir.resolve()}")
    print(f"  stage report        -> {report['run_dir']}")
    return report


__all__ = ["evaluate_stage"]
