"""
predict_game — run ONE prediction on ONE match and write it out.

This is the user-facing entry point that ties the pieces together: load the trained models
(:class:`~simulation.game_simulator.GameSimulator`), play a full game with the rule engine
(:class:`~simulation.controller.GameController`), then emit two artifacts under
``artifacts/predictions/<run_id>/``:

  1. ``playbyplay.csv`` — the generated game in the **exact 12-column cleaned-data format**
     (``game_id, roster_home, roster_away, time, event, player, type, result,
     secondary_player, home/away, season, playoff``), so it parses with the same tooling as
     ``data/season2003.csv`` (``data_loading.load_all_cleaned`` / ``box_score._roster``).
  2. ``boxscore_home.csv`` / ``boxscore_away.csv`` — full NBA box scores (MIN, FG/3PT/FT with
     percentages, OREB/DREB/REB, AST/STL/BLK/TO/PF, **+/-**, PTS) plus a TEAM totals row.

When the matchup comes from a real game (``--from-game`` / :func:`predict_from_game`), the
**actual** game's box score, play-by-play and score are written alongside (``actual_*``) so the
prediction can be compared directly against what really happened.

Plus a small ``run.json`` (the matchup spec + predicted/actual scores) and a human-readable
``boxscore.txt``.
"""
from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import ROSTER_SIZE
from models.artifacts import DEFAULT_ARTIFACTS_ROOT
from simulation.box_score import BoxScore, generate_box_score
from simulation.controller import GameController
from simulation.game_input import GameInput, game_input_for_game
from simulation.game_simulator import GameSimulator

# The cleaned-data column order (must match data_cleaner.py output exactly).
CLEANED_COLUMNS = [
    "game_id", "roster_home", "roster_away", "time", "event", "player", "type", "result",
    "secondary_player", "home/away", "season", "playoff",
]

DEFAULT_OUTPUT_ROOT = "artifacts/predictions"
SKIP_HOME_AWAY_EVENTS = {"start", "end", "none", "PAD", "UNK"}


def predict_game(game_input: GameInput, *, home_team: str = "HOME", away_team: str = "AWAY",
                 possession: str = "home", seed: int | None = None, greedy: bool = False,
                 game_id: int = 1, artifacts_root: str = DEFAULT_ARTIFACTS_ROOT,
                 output_root: str = DEFAULT_OUTPUT_ROOT,
                 actual_box: BoxScore | None = None,
                 actual_pbp: pd.DataFrame | None = None,
                 home_starters: list[str] | None = None,
                 away_starters: list[str] | None = None) -> dict:
    """Generate one game from a :class:`GameInput` spec and write CSV + box-score artifacts.

    If ``actual_box`` / ``actual_pbp`` are supplied (the real game this matchup was drawn from),
    they are written as ``actual_*`` artifacts and folded into ``run.json`` for side-by-side
    comparison. Returns a dict with the ``run_id``, output paths, scores and the :class:`BoxScore`.

    ``home_starters`` / ``away_starters`` (when given) seed the exact opening five for each team
    instead of letting the SubstitutionModel choose it — used to anchor a real-game prediction
    to that game's actual tip-off lineup.
    """
    sim = GameSimulator.load(artifacts_root=artifacts_root)
    controller = GameController(sim, seed=seed, greedy=greedy)
    controller.start(game_input.home_roster, game_input.away_roster,
                     possession=possession, season=str(game_input.season),
                     home_starters=home_starters, away_starters=away_starters,
                     season_context=game_input.season_context())
    history = controller.run()

    # 12-column cleaned-format play-by-play + box scores.
    pbp = history_to_cleaned_frame(history, game_input, game_id=game_id)
    box = generate_box_score(history, home_team=home_team, away_team=away_team)

    run_id = _new_run_id()
    out_dir = Path(output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pbp_path = out_dir / "playbyplay.csv"
    pbp.to_csv(pbp_path, index=False)
    home_path = out_dir / "boxscore_home.csv"
    away_path = out_dir / "boxscore_away.csv"
    box.to_frame("home").to_csv(home_path, index=False)
    box.to_frame("away").to_csv(away_path, index=False)
    (out_dir / "boxscore.txt").write_text(box.render(), encoding="utf-8")

    predicted_score = _score_dict(home_team, away_team, box.home_score, box.away_score)
    result = {"run_id": run_id, "out_dir": str(out_dir), "playbyplay": str(pbp_path),
              "boxscore_home": str(home_path), "boxscore_away": str(away_path),
              "box": box, "history": history, "final_score": predicted_score}
    run_meta = {
        "run_id": run_id,
        "predicted_score": predicted_score,   # the headline result
        "possession_start": possession,
        "season": game_input.season,
        "playoff": game_input.playoff,
        "seed": seed,
        "events": len(history),
        "spec": game_input.to_dict(),
    }

    # --- Actual game (for comparison), when predicting from a real game ---
    actual_score = None
    if actual_box is not None:
        actual_box.to_frame("home").to_csv(out_dir / "actual_boxscore_home.csv", index=False)
        actual_box.to_frame("away").to_csv(out_dir / "actual_boxscore_away.csv", index=False)
        (out_dir / "actual_boxscore.txt").write_text(actual_box.render(), encoding="utf-8")
        actual_score = _score_dict(home_team, away_team,
                                   actual_box.home_score, actual_box.away_score)
        run_meta["actual_score"] = actual_score
        result["actual_box"] = actual_box
        result["actual_score"] = actual_score
    if actual_pbp is not None:
        actual_pbp.to_csv(out_dir / "actual_playbyplay.csv", index=False)

    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 52)
    print(f"  PREDICTED  {predicted_score['line']}   ({len(history)} events)")
    if actual_score is not None:
        print(f"  ACTUAL     {actual_score['line']}")
    print("=" * 52)
    print(f"  Artifacts: {out_dir}")
    return result


def predict_from_game(game_id: int, *, data_dir: str = "./data", home_team: str | None = None,
                      away_team: str | None = None, use_model_starters: bool = False,
                      **kwargs) -> dict:
    """Convenience: pull a real game's matchup spec (rosters/season/playoff) and predict it.

    Loads the cleaned game once and derives the matchup spec **and** the actual box score /
    play-by-play from it, so the prediction is written next to the real game for comparison. The
    cleaned data has no team-name column, so labels default to ``HOME``/``AWAY`` unless overridden.

    By default the prediction is seeded with the game's **actual** tip-off five (read from the
    ``start`` row). Pass ``use_model_starters=True`` to instead let the SubstitutionModel choose
    the opening five (the spec's whole roster is handed to it, no real starters seeded).
    """
    from data_loading import load_all_cleaned
    from simulation.game_input import extract_game_input

    df = load_all_cleaned(data_dir, parse_rosters=True)
    game = df[df["game_id"] == int(game_id)].sort_values("time")
    if game.empty:
        raise ValueError(f"game_id {game_id} not found in cleaned data under {data_dir!r}")

    ht, at = home_team or "HOME", away_team or "AWAY"
    spec = extract_game_input(game)
    actual_box = generate_box_score(game, home_team=ht, away_team=at)
    actual_pbp = game.reindex(columns=CLEANED_COLUMNS)

    home_starters = away_starters = None
    if not use_model_starters:
        home_starters, away_starters = _real_starters(game)

    return predict_game(spec, home_team=ht, away_team=at, game_id=game_id,
                        actual_box=actual_box, actual_pbp=actual_pbp,
                        home_starters=home_starters, away_starters=away_starters, **kwargs)


def _real_starters(game: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Read the actual tip-off five for each team from a cleaned game's ``start`` row.

    The ``start`` row's ``roster_home`` / ``roster_away`` cells hold the real starters (written
    by ``data_cleaner.py``). Non-player tokens are filtered out and each side is capped at five.
    """
    from simulation.box_score import _roster
    from simulation.game_input import _NON_PLAYERS

    starts = game[game["event"] == "start"]
    if starts.empty:
        raise ValueError("no 'start' row in game; cannot read real starters.")
    start_row = starts.iloc[0]

    def _five(cell) -> list[str]:
        players = [p for p in _roster(cell) if str(p).strip() not in _NON_PLAYERS]
        return players[:ROSTER_SIZE]

    return _five(start_row["roster_home"]), _five(start_row["roster_away"])


def history_to_cleaned_frame(history: list[dict], game_input: GameInput,
                             *, game_id: int = 1) -> pd.DataFrame:
    """Turn the controller's event history into the 12-column cleaned-data DataFrame."""
    playoff_flag = 2 if game_input.playoff == 1 else 1   # cleaned: 1=regular, 2=playoff
    rows = []
    for r in history:
        home_roster = list(r.get("roster_home", []))
        away_roster = list(r.get("roster_away", []))
        rows.append({
            "game_id": game_id,
            "roster_home": str(home_roster),
            "roster_away": str(away_roster),
            # Cleaned data stores whole-second timestamps (data_cleaner.convert_time); round the
            # fractional rollout clock to match that format (the internal clock stays fractional).
            "time": int(round(float(r["time"]))),
            "event": r["event"],
            "player": r["player"],
            "type": r["type"],
            "result": r["result"],
            "secondary_player": r.get("secondary_player", "none"),
            "home/away": _home_away_flag(r, home_roster, away_roster),
            "season": game_input.season,
            "playoff": playoff_flag,
        })
    return pd.DataFrame(rows, columns=CLEANED_COLUMNS)


def _home_away_flag(row: dict, home_roster: list[str], away_roster: list[str]) -> int:
    """1 = home, 2 = away, 0 = game boundary — mirrors data_cleaner.home_indicator.

    For substitutions the post-swap lineup holds the incoming player, so the flag keys off the
    incoming (``secondary_player``); otherwise it keys off the acting ``player``.
    """
    event = str(row.get("event"))
    if event in SKIP_HOME_AWAY_EVENTS:
        return 0
    ref = row.get("secondary_player") if event == "substitution" else row.get("player")
    if ref in home_roster:
        return 1
    if ref in away_roster:
        return 2
    return 0


def _score_dict(home_team: str, away_team: str, home_score: int, away_score: int) -> dict:
    """Build a uniform score block (used for both the predicted and actual final scores)."""
    winner = (home_team if home_score > away_score
              else away_team if away_score > home_score else "TIE")
    return {
        "home_team": home_team, "home_score": home_score,
        "away_team": away_team, "away_score": away_score,
        "winner": winner,
        "line": f"{home_team} {home_score} - {away_score} {away_team}",
    }


def _new_run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


__all__ = ["predict_game", "predict_from_game", "history_to_cleaned_frame"]
