"""
stage_eval.py — per-stage curriculum evaluation: predict the next block of real games.

After a curriculum stage finishes training, we predict its sequential holdout (the next
``HOLDOUT_GAMES`` real games) by simulating each ``STAGE_SIMS`` times. This wraps the existing
``simulation.evaluation`` scoring with two stage-specific additions the user asked for:

  1. **Per-game prediction folders** (descriptive names) under
     ``artifacts/predictions/<stage_name>/`` holding, for each game: the actual box score, the
     11-sim *averaged* predicted box score (game score included), the actual play-by-play, and all
     11 generated play-by-plays. Each game's evaluation record is also cached (``record.json``) so
     an interrupted eval **resumes** — finished games are reloaded, not re-simulated.
  2. **A stage-level overall report** (the standard HTML + Parquet eval report) capturing win/
     spread/box accuracy and the std across the 11 sims, written once all games are done.
"""
from __future__ import annotations

# Import TensorFlow before pandas-using project modules (see evaluation.py / main.py).
try:  # noqa: SIM105
    import tensorflow  # noqa: F401
except Exception:
    pass

import json
import re
from pathlib import Path

from config import HOLDOUT_MANIFEST_NAME, STAGE_SIMS
from data_loading import load_all_cleaned
from models.artifacts import DEFAULT_ARTIFACTS_ROOT
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT
from simulation.box_score import BoxScore, PlayerLine, generate_box_score
from simulation.evaluation import _aggregate, _print_summary, build_game_record, simulate_repeated
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
    """Build a BoxScore from a record's 11-sim per-player averages (game score included)."""
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


def _write_game_folder(out_dir: Path, game, spec, boxes, histories, record,
                       home_team: str, away_team: str) -> None:
    """Persist one game's actual + averaged-prediction box scores and all sim play-by-plays."""
    out_dir.mkdir(parents=True, exist_ok=True)

    actual_box = generate_box_score(game, home_team=home_team, away_team=away_team)
    actual_box.to_frame("home").to_csv(out_dir / "actual_boxscore_home.csv", index=False)
    actual_box.to_frame("away").to_csv(out_dir / "actual_boxscore_away.csv", index=False)
    (out_dir / "actual_boxscore.txt").write_text(actual_box.render(), encoding="utf-8")
    game.reindex(columns=CLEANED_COLUMNS).to_csv(out_dir / "actual_playbyplay.csv", index=False)

    pred_box = _averaged_box(record, home_team, away_team)
    pred_box.to_frame("home").to_csv(out_dir / "pred_boxscore_home.csv", index=False)
    pred_box.to_frame("away").to_csv(out_dir / "pred_boxscore_away.csv", index=False)
    (out_dir / "pred_boxscore.txt").write_text(pred_box.render(), encoding="utf-8")

    for i, history in enumerate(histories, start=1):
        frame = history_to_cleaned_frame(history, spec, game_id=int(record["game_id"]))
        frame.to_csv(out_dir / f"sim_{i:02d}_playbyplay.csv", index=False)

    run_meta = {
        "game_id": record["game_id"],
        "home_team": home_team, "away_team": away_team,
        "predicted_score": {"home": record["pred_home_score"], "away": record["pred_away_score"]},
        "actual_score": {"home": record["actual_home_score"], "away": record["actual_away_score"]},
        "win_prob_home": record["win_prob_home"],
        "per_sim_scores": [{"home": b.home_score, "away": b.away_score} for b in boxes],
        "n_sims": len(boxes),
    }
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    (out_dir / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def evaluate_stage(stage_name: str, *, holdout_ids: list[int] | None = None,
                   n_sims: int = STAGE_SIMS, data_dir: str = "./data",
                   processed_dir: str = "./data/processed",
                   artifacts_root: str = DEFAULT_ARTIFACTS_ROOT,
                   reports_root: str = DEFAULT_REPORTS_ROOT,
                   predictions_root: str = DEFAULT_OUTPUT_ROOT, seed0: int = 0) -> dict:
    """Predict a stage's holdout games (``n_sims`` each), write per-game folders + a stage report.

    ``holdout_ids`` defaults to the manifest the stage's preprocess wrote (``holdout_games.json``).
    Finished games (those with a cached ``record.json``) are reloaded rather than re-simulated, so
    a killed eval resumes. Returns the report dict (with ``run_dir``).
    """
    from reporting.eval_report import build_report, write_eval_report

    if holdout_ids is None:
        manifest = Path(processed_dir) / HOLDOUT_MANIFEST_NAME
        if not manifest.exists():
            raise FileNotFoundError(f"No holdout manifest at {manifest}; preprocess the stage first.")
        holdout_ids = [int(g) for g in json.loads(manifest.read_text(encoding="utf-8"))]
    if not holdout_ids:
        raise ValueError(f"stage '{stage_name}' has an empty holdout — nothing to evaluate.")

    df = load_all_cleaned(data_dir, parse_rosters=True)
    stage_dir = Path(predictions_root) / stage_name
    sim = None  # lazily loaded only if there's an unfinished game to simulate

    records: list[dict] = []
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

        if sim is None:
            sim = GameSimulator.load(artifacts_root=artifacts_root)
        print(f"  game {gid} ({away_team} at {home_team}): {n_sims} sims...")
        spec = extract_game_input(game)
        try:
            home_starters, away_starters = _real_starters(game)
        except ValueError:
            home_starters = away_starters = None
        boxes, histories = simulate_repeated(
            sim, spec, home_starters, away_starters, n_sims=n_sims, seed0=seed0,
            home_team=home_team, away_team=away_team, return_histories=True,
        )
        record = build_game_record(game, boxes, n_sims=n_sims,
                                   home_team=home_team, away_team=away_team)
        _write_game_folder(out_dir, game, spec, boxes, histories, record, home_team, away_team)
        records.append(record)

    aggregate = _aggregate(records)
    report = build_report(records=records, aggregate=aggregate, n_sims=n_sims, run_name=stage_name)
    run_dir = write_eval_report(report, reports_root=reports_root)
    _print_summary(aggregate, len(records), n_sims)
    print(f"\n  per-game predictions -> {stage_dir.resolve()}")
    print(f"  stage report        -> {run_dir.resolve()}")
    report["run_dir"] = str(run_dir)
    report["predictions_dir"] = str(stage_dir)
    return report


__all__ = ["evaluate_stage"]
