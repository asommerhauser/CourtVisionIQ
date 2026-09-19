"""
The corpus a weighted replay pass trains on: each sim's own play-by-play, in cleaned-data shape (3.2 W11).

``models/replay.py`` holds the estimator -- advantages, the positive filter, the weight arrays. This
holds the other half, which is less interesting and more load-bearing: **the labels**. A sim's
play-by-play *is* what the model sampled, so "what this sim did" needs no construction. It needs a
corpus every head's ordinary ``preprocess`` can read, and that is not the same thing as a list of
events.

**Why a corpus rather than a tensor.** ``simulation/decision_log.py`` records only where and what
(``game_id, sim_index, position, head, output, token``) because retaining the context per queried
position across thousands of game-sims is not feasible. The context is re-derived by replaying the
sim's rows through the ordinary preprocess -- the same round trip ``reporting/state_probes`` already
performs over written play-by-play. So the pass writes a corpus and the heads read it exactly as they
read ``./data``. No second preprocessing path, and no tensor format that can drift from the real one.

**Three things a sim does not carry, and where they come from.**

1. *Season context* -- ``game_date``, the two ``*_games_played`` and ``*_days_rest`` columns. These are
   properties of the matchup, not of the simulation: the sim IS that fixture on that date, so they are
   copied from the real game's rows. Recomputing them from a synthetic schedule would invent a
   different season.
2. *Per-slot rest* -- ``rest_home`` / ``rest_away`` are lists aligned to **that row's** roster, and a
   sim substitutes, so the real row's list is aligned to the wrong men within a few events. Carried as
   a player -> rest-days map built from the real game and re-laid against each sim row's roster.
3. *Priors* -- ``models/prior_features.merge_prior_features`` joins by ``game_id`` and **raises** when
   coverage is incomplete, which is correct and is exactly what a synthetic id would trip. The pass
   writes its own sidecar beside the corpus: the real game's prior rows, re-keyed to the sim ids. A
   sim of game G faces G's opponent on G's date, so G's priors are its priors.

**Sim ids are offset, never hashed.** ``sim_game_id`` is ``REPLAY_ID_BASE + real_id * 100 + sim_index``,
so the real id is recoverable by arithmetic (:func:`real_game_id`) and a collision with a real id is
impossible rather than unlikely -- :func:`assert_ids_fit` refuses a corpus that reaches the base. The
recoverability is not a convenience: the advantage is computed per real game over its sibling sims, and
the weight has to find its way back to the rows it belongs to.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from config import REPLAY_ID_BASE, REPLAY_SIMS_PER_GAME, ROSTER_SIZE

#: Per-game columns a sim inherits from the real fixture it simulated. Scalars: one value per game.
CONTEXT_COLUMNS = ("game_date", "home_team", "away_team",
                   "home_games_played", "away_games_played",
                   "home_days_rest", "away_days_rest")

#: Per-slot columns that must be re-laid against each row's own roster, not copied.
REST_COLUMNS = {"rest_home": "roster_home", "rest_away": "roster_away"}


# ===================================================================== #
# --- Ids                                                              --
# ===================================================================== #

def sim_game_id(real_id, sim_index: int) -> int:
    """The corpus id for one sim of one real game. Reversible by construction."""
    if not 0 <= int(sim_index) < 100:
        raise ValueError(f"sim_index {sim_index} outside 0..99; widen the multiplier first")
    return int(REPLAY_ID_BASE) + int(real_id) * 100 + int(sim_index)


def real_game_id(sim_id) -> int:
    """The real game a sim id came from."""
    return (int(sim_id) - int(REPLAY_ID_BASE)) // 100


def sim_index_of(sim_id) -> int:
    return (int(sim_id) - int(REPLAY_ID_BASE)) % 100


def assert_ids_fit(real_ids) -> None:
    """Refuse a corpus whose real ids reach the offset base.

    The whole scheme rests on sim ids living in a range no real id occupies. That is true by a factor
    of ~3,000 on this corpus and would stop being true silently.
    """
    biggest = max((int(g) for g in real_ids), default=0)
    if biggest * 100 >= int(REPLAY_ID_BASE):
        raise ValueError(
            f"real game id {biggest} times 100 reaches REPLAY_ID_BASE ({REPLAY_ID_BASE}); "
            f"sim ids would collide with real ones. Raise REPLAY_ID_BASE.")


# ===================================================================== #
# --- Which games get replayed                                         --
# ===================================================================== #

def select_replay_games(subset_ids, *, fraction: float, seed: int) -> list[int]:
    """One game in ``fraction`` of the training subset, drawn once and deterministically.

    From the SUBSET, never a flat stride over the corpus and never a holdout window (W11 SS4.5): a flat
    stride is era-neutral and would fine-tune ``shot_result`` toward the old game, and a holdout window
    would turn the report into a training metric.
    """
    ordered = sorted(int(g) for g in subset_ids)
    if not ordered:
        raise ValueError("no subset games to replay; run `python -m training.subset extract` first")
    n = max(1, int(round(len(ordered) * float(fraction))))
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(len(ordered), size=min(n, len(ordered)), replace=False)
    return sorted(ordered[i] for i in picked)


# ===================================================================== #
# --- One sim's rows                                                    --
# ===================================================================== #

def _roster(cell) -> list[str]:
    if isinstance(cell, (list, tuple)):
        return [str(p) for p in cell]
    try:
        return [str(p) for p in ast.literal_eval(str(cell))]
    except (ValueError, SyntaxError):
        return []


def rest_by_player(real_rows: pd.DataFrame) -> tuple[dict, float, float]:
    """``(player -> rest days, home default, away default)`` from the real game's rows.

    Read over every row rather than the first, because a man who starts on the bench appears in no
    starting roster and would otherwise fall to the default in every sim that plays him.
    """
    out: dict[str, float] = {}
    defaults = {}
    for rest_col, roster_col in REST_COLUMNS.items():
        side_values: list[float] = []
        for cell, roster_cell in zip(real_rows[rest_col], real_rows[roster_col]):
            values = _roster(cell)
            names = _roster(roster_cell)
            for name, value in zip(names, values):
                try:
                    out.setdefault(str(name), float(value))
                except (TypeError, ValueError):
                    continue
            side_values.extend(float(v) for v in values if str(v).replace(".", "", 1).isdigit())
        defaults[rest_col] = float(np.median(side_values)) if side_values else 0.0
    return out, defaults["rest_home"], defaults["rest_away"]


def attach_context(frame: pd.DataFrame, real_rows: pd.DataFrame) -> pd.DataFrame:
    """Add the columns a sim cannot know: the fixture's season context and its per-slot rest.

    ``frame`` is :func:`~simulation.predict_game.history_to_cleaned_frame` output (the twelve columns a
    sim does produce). Returns a new frame in the full cleaned-column order.
    """
    out = frame.copy()
    first = real_rows.iloc[0]
    for col in CONTEXT_COLUMNS:
        out[col] = first[col] if col in real_rows.columns else np.nan

    rest_map, home_default, away_default = rest_by_player(real_rows)
    for rest_col, roster_col in REST_COLUMNS.items():
        default = home_default if rest_col == "rest_home" else away_default
        laid = []
        for roster_cell in out[roster_col]:
            names = _roster(roster_cell)[:ROSTER_SIZE]
            laid.append(str([float(rest_map.get(n, default)) for n in names]))
        out[rest_col] = laid
    return out


def sim_frame(history, game_input, real_rows: pd.DataFrame, *, real_id, sim_index: int) -> pd.DataFrame:
    """One sim, as cleaned rows under its own id, ready to be concatenated into a corpus."""
    from simulation.predict_game import history_to_cleaned_frame

    gid = sim_game_id(real_id, sim_index)
    frame = history_to_cleaned_frame(history, game_input, game_id=gid)
    return attach_context(frame, real_rows)


# ===================================================================== #
# --- The corpus on disk                                                --
# ===================================================================== #

def write_corpus(frames, out_dir) -> list[Path]:
    """Write the sim rows as ``season<YEAR>.csv`` files, the shape ``data_loading`` expects.

    Split by season because that is how the cleaned corpus is laid out and how ``cleaned_csvs``
    enumerates it -- one giant file would load identically today and diverge the first time anything
    reads a season off a filename.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(list(frames), ignore_index=True)
    written = []
    for season, part in combined.groupby("season"):
        path = out_dir / f"season{int(season)}.csv"
        part.to_csv(path, index=False)
        written.append(path)
    return written


def write_priors(data_dir, out_dir, sim_ids) -> list[Path]:
    """Re-key the real games' prior rows onto their sim ids, beside the replay corpus.

    ``merge_prior_features`` raises when the sidecar does not cover every game in the frame, which is
    the behaviour W1 added on purpose: a partially-covered corpus trains some games on real rates and
    the rest on league means, and looks exactly like a model that learned nothing from the priors.
    """
    from player_priors import PRIORS_DIRNAME

    src = Path(data_dir) / PRIORS_DIRNAME
    dst = Path(out_dir) / PRIORS_DIRNAME
    dst.mkdir(parents=True, exist_ok=True)

    wanted: dict[int, list[int]] = {}
    for sid in sim_ids:
        wanted.setdefault(real_game_id(sid), []).append(int(sid))

    written = []
    for pattern in ("players_*.parquet", "teams_*.parquet"):
        for path in sorted(src.glob(pattern)):
            frame = pd.read_parquet(path)
            hit = frame[frame["game_id"].isin(wanted)]
            if hit.empty:
                continue
            parts = []
            for real_id, group in hit.groupby("game_id"):
                for sid in wanted[int(real_id)]:
                    copy = group.copy()
                    copy["game_id"] = sid
                    parts.append(copy)
            target = dst / path.name
            pd.concat(parts, ignore_index=True).to_parquet(target, index=False)
            written.append(target)
    if not written:
        raise ValueError(
            f"no prior rows found under {src} for the replayed games; the sidecar does not cover "
            f"them, and a pass without priors would train every player as the league mean")
    return written


def expected_sims(n_games: int, n_sims: int = REPLAY_SIMS_PER_GAME) -> int:
    """How many game-sims a pass of this shape costs. Stated so a run can check it before paying."""
    return int(n_games) * int(n_sims)


__all__ = ["CONTEXT_COLUMNS", "REST_COLUMNS", "assert_ids_fit", "attach_context", "expected_sims",
           "real_game_id", "rest_by_player", "select_replay_games", "sim_frame", "sim_game_id",
           "sim_index_of", "write_corpus", "write_priors"]
