"""
Per-player and per-team season-to-date priors, as model inputs.

``player_priors.py`` computes them causally and writes them to ``data/priors/``. This module is the
other half: it joins them onto each row's roster slots, normalizes them, and hands them to the
roster encoder as extra per-player scalars, exactly where ``rest`` already goes.

Three decisions live here, each with a reason that cost something to learn.

**Fixed normalization constants, not fitted ``norm_stats``.** ``season_features`` z-scores rest
against a train-fit ``rest_mean`` / ``rest_std`` pair, and that choice ripples: two persist sites, a
``refit=False`` branch for staged training, and three separate places on the inference side that
read the values back. Every fitted statistic is another way for training and inference to disagree,
and another key in ``encoder/vocabs/norm_stats.json`` -- the file the test suite is known to
overwrite, which is why "check it against git before any train" is a standing rule. These rates are
already in natural, era-stable units with known ranges, so a clip and a divisor is honest and a
z-score buys nothing. This follows ``models/rotation_features.py``'s ``_NORM`` precedent instead.

**One 3-D tensor per side, not ten 2-D ones.** ``prior_home`` is ``(SEQ, ROSTER_SIZE, N_PRIORS)``.
Ten separate named inputs per side would mean twenty new entries in every head's ``INPUT_KEYS``,
twenty buffers in the incremental cache and twenty columns in the npz, for the same numbers.
``side_prior_scalars`` unstacks it at the graph edge into exactly the per-player scalar list the
roster encoder wants.

**Season-to-date only. No last-10 window, despite SS3 W2.1 asking for both.** Each roster-parallel
``(SEQ=600, ROSTER=5)`` float plane costs 12 KB per game raw, and ``_load_processed`` holds the whole
npz in host RAM while ``from_tensor_slices`` materializes it again. Ten priors across two sides is
+240 KB per game against a current total near 180 KB; twenty would roughly double the training set's
footprint, and the 2.0 cycle already lost a train to an out-of-memory. Stored as float16 (these are
three-significant-digit rates and the GPU path is mixed-precision anyway) the ten cost +120 KB.
Last-10 is also precisely what W2.2's learned recent-games summary is meant to replace, so it is the
right half to cut.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROSTER_SIZE
from player_priors import (
    LEAGUE_DEFAULTS,
    PLAYER_PRIOR_KEYS,
    TEAM_DEFAULTS,
    TEAM_PRIOR_KEYS,
    priors_dir,
)

# Roster-parallel per-player priors: (SEQ, ROSTER_SIZE, N_PRIORS), fed to the roster encoder.
PRIOR_LIST_COLS = ("prior_home", "prior_away")
# Team-level priors: (SEQ, 1) each, projected and concatenated into the fusion.
TEAM_PRIOR_COLS = tuple(f"{side}_prior_{key}"
                        for side in ("home", "away") for key in TEAM_PRIOR_KEYS)
PRIOR_INPUT_KEYS = (*PRIOR_LIST_COLS, *TEAM_PRIOR_COLS)

N_PLAYER_PRIORS = len(PLAYER_PRIOR_KEYS)

# (clip_lo, clip_hi, divisor). Chosen so an average player lands near 1.0 and the whole realistic
# range lands inside about [0, 2] -- the same shape rotation_features._NORM aims for. The clips are
# not cosmetic: they are what stops a garbage-time line (four minutes, two threes) from arriving as
# a 60-points-per-36 input.
_NORM = {
    "min_pg":    (0.0, 44.0, 24.0),
    "pts_36":    (0.0, 40.0, 18.0),
    "fga_36":    (0.0, 32.0, 14.0),
    "fg_pct":    (0.0, 1.0, 0.46),
    "tpa_rate":  (0.0, 1.0, 0.40),
    "fta_rate":  (0.0, 1.0, 0.30),
    "ast_36":    (0.0, 14.0, 4.0),
    "oreb_36":   (0.0, 8.0, 2.0),
    "dreb_36":   (0.0, 16.0, 5.0),
    "tov_36":    (0.0, 10.0, 2.5),
    # 3.2 W5. Divisors are the MEASURED league means (2022-23, via generate_box_score: ft_pct 0.7825,
    # tp_pct 0.3600, pf_36 2.9689), so an average player reads 1.0 -- the property
    # test_normalization_puts_an_average_player_near_one pins.
    "ft_pct":    (0.0, 1.0, 0.78),
    "tp_pct":    (0.0, 1.0, 0.36),
    "pf_36":     (0.0, 10.0, 2.95),
    # Seasons since first appearance. Divisor is the measured mean over 2011+ player-games (4.84,
    # median 4), clipped at 20 -- the longest career in the corpus is 19 seasons.
    "career_stage": (0.0, 20.0, 4.5),
    # The three deltas are DIFFERENCES, so their neutral value is 0.0 and they are the only prior
    # inputs that do not read near 1.0 for an average player. Clipped symmetrically at roughly the
    # largest real season-over-season move, and divided by the clip so the feature lands in [-1, 1].
    # That is deliberate, not an oversight: shifting them to centre on 1.0 would make "no change"
    # indistinguishable from "no information" for a player with no previous season, which is exactly
    # the distinction career_stage and these three exist to draw.
    "d_pts_36":  (-15.0, 15.0, 15.0),
    "d_min_pg":  (-20.0, 20.0, 20.0),
    "d_fga_36":  (-12.0, 12.0, 12.0),
    "net_rating": (-20.0, 20.0, 8.0),
    "pace":       (85.0, 115.0, 100.0),
    "off_rating": (90.0, 130.0, 112.0),
    "def_rating": (90.0, 130.0, 112.0),
}

# The league-average player, already normalized -- what a PAD slot and an unknown name both read as.
# Deliberately not zeros: a zero vector says "a player who does nothing", which is a strong and
# wrong claim, where the league mean says "no information".
_DEFAULT_PLAYER = np.array(
    [(min(max(LEAGUE_DEFAULTS[k], _NORM[k][0]), _NORM[k][1]) / _NORM[k][2])
     for k in PLAYER_PRIOR_KEYS], dtype=np.float32)


def normalize_player(values) -> np.ndarray:
    """Clip-and-divide one player's raw prior vector, in ``PLAYER_PRIOR_KEYS`` order."""
    out = np.asarray(values, dtype=np.float32).copy()
    for i, key in enumerate(PLAYER_PRIOR_KEYS):
        lo, hi, div = _NORM[key]
        out[i] = min(max(float(out[i]), lo), hi) / div
    return out


def normalize_team(key: str, value: float) -> float:
    lo, hi, div = _NORM[key]
    return min(max(float(value), lo), hi) / div


@lru_cache(maxsize=8)
def priors_for_season(data_dir: str, season: str):
    """``load_priors`` for one season, memoised.

    The inference path needs one season at a time, and holding all 21 as dict-of-dict-of-array
    would cost hundreds of megabytes of Python object overhead in a process whose whole job is to
    keep VRAM free for the rollout. Training loads the lot deliberately, through ``load_priors``.
    """
    return load_priors(data_dir, seasons=(str(season),))


def load_priors(data_dir: str = "./data", *, seasons=None):
    """``(player_map, team_map)`` for the corpus.

    ``player_map[game_id][name]`` is a normalized ``(N_PLAYER_PRIORS,)`` vector;
    ``team_map[game_id][side]`` is a dict of normalized team rates. Both are keyed the way the
    cleaned rows are -- integer game id, display-name string -- so the join needs no player ids,
    which the pipeline does not have.
    """
    root = priors_dir(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"no priors at {root}. Build them first:  python -m player_priors")

    player_map: dict[int, dict[str, np.ndarray]] = {}
    team_map: dict[int, dict[str, dict[str, float]]] = {}

    for path in sorted(root.glob("players_*.parquet")):
        if seasons and path.stem.split("_")[-1] not in {str(s) for s in seasons}:
            continue
        frame = pd.read_parquet(path)
        raw = frame[list(PLAYER_PRIOR_KEYS)].to_numpy(dtype=np.float32)
        for i, key in enumerate(PLAYER_PRIOR_KEYS):
            lo, hi, div = _NORM[key]
            raw[:, i] = np.clip(raw[:, i], lo, hi) / div
        for gid, name, vec in zip(frame["game_id"].to_numpy(), frame["player"].to_numpy(), raw):
            player_map.setdefault(int(gid), {})[str(name)] = vec

    for path in sorted(root.glob("teams_*.parquet")):
        if seasons and path.stem.split("_")[-1] not in {str(s) for s in seasons}:
            continue
        frame = pd.read_parquet(path)
        for row in frame.itertuples(index=False):
            team_map.setdefault(int(row.game_id), {})[str(row.side)] = {
                key: normalize_team(key, getattr(row, key)) for key in TEAM_PRIOR_KEYS}
    return player_map, team_map


def pad_priors(names, lookup, size: int = ROSTER_SIZE) -> np.ndarray:
    """One row's roster slots as ``(size, N_PLAYER_PRIORS)``.

    Slots past the end of the roster, and names the sidecar has never seen, read as the league mean
    rather than as zeros -- see ``_DEFAULT_PLAYER``. The PAD slots are separately masked out inside
    the roster encoder (it derives its mask from the id column), so what they hold does not reach an
    attention output; the league mean is simply the honest filler.
    """
    out = np.repeat(_DEFAULT_PLAYER[None, :], size, axis=0).astype(np.float32)
    for i, name in enumerate(list(names)[:size]):
        vec = lookup.get(str(name))
        if vec is not None:
            out[i] = vec
    return out


def build_raw_prior_cols(df: pd.DataFrame, rosters: dict, player_map, team_map) -> dict:
    """Per-row prior planes, aligned to ``df``'s positional index.

    ``rosters`` maps ``"home_roster"`` / ``"away_roster"`` to the parsed name lists the caller
    already built for the encoder, so the roster strings are parsed once for the whole pipeline.
    """
    cols: dict[str, np.ndarray] = {}
    game_ids = df["game_id"].to_numpy()
    for col, roster_key in (("prior_home", "home_roster"), ("prior_away", "away_roster")):
        planes = np.empty((len(df), ROSTER_SIZE, N_PLAYER_PRIORS), dtype=np.float32)
        for i, (gid, names) in enumerate(zip(game_ids, rosters[roster_key])):
            planes[i] = pad_priors(names, player_map.get(int(gid), {}))
        cols[col] = planes
    for side in ("home", "away"):
        for key in TEAM_PRIOR_KEYS:
            default = normalize_team(key, TEAM_DEFAULTS[key])
            cols[f"{side}_prior_{key}"] = np.array(
                [team_map.get(int(gid), {}).get(side, {}).get(key, default) for gid in game_ids],
                dtype=np.float32)
    return cols


def merge_prior_features(df: pd.DataFrame, cols: dict, rosters: dict,
                         data_dir: str = "./data") -> dict:
    """Load the sidecar and fold the prior columns into ``cols``. The heads' single entry point.

    Mirrors ``merge_season_features`` / ``merge_game_state_features`` / ``merge_rotation_features``
    so every head's preprocess reads the same way, and mutates-and-returns ``cols`` as they do.
    """
    if not priors_dir(data_dir).is_dir():
        # No sidecar at all. Every player reads as the league mean, which is a legitimate state --
        # a synthetic test fixture, or predicting a game on a machine that carries weights but no
        # data. It is NOT a legitimate state for a real train, so it is loud, and full_run's
        # pre-flight (require_priors) refuses it outright before a train starts.
        print(f"[priors] WARNING: no sidecar at {priors_dir(data_dir)}; every player will read as "
              f"the league mean. For a real train, build it first: "
              f"python -m player_priors --data-dir {data_dir}")
        player_map, team_map = {}, {}
    else:
        player_map, team_map = load_priors(data_dir)
        covered = sum(1 for gid in df["game_id"].unique() if int(gid) in player_map)
        total = df["game_id"].nunique()
        if covered < total:
            # A sidecar that exists but does not cover the corpus is the dangerous case: the data
            # was re-cleaned and the priors were not rebuilt, so SOME games train on real rates and
            # the rest silently on league means. That looks exactly like a model that learned
            # nothing from the priors, so it raises rather than warning.
            raise ValueError(
                f"the priors sidecar at {priors_dir(data_dir)} covers {covered} of {total} games "
                f"in this corpus -- it is stale. Rebuild it: "
                f"python -m player_priors --data-dir {data_dir}")
    cols.update(build_raw_prior_cols(df, rosters, player_map, team_map))
    return cols


def require_priors(data_dir: str = "./data") -> int:
    """Pre-flight for a real train: the sidecar must exist. Returns how many games it covers.

    ``merge_prior_features`` only warns when the sidecar is missing, because a synthetic fixture and
    a weights-only machine both legitimately have none. A train does not: starting one without the
    priors produces a model whose whole W2.1 input is a constant, and the first sign of it would be
    the eval, hours later.
    """
    root = priors_dir(data_dir)
    if not root.is_dir():
        raise SystemExit(
            f"no priors sidecar at {root}. A train without it feeds every player the league mean." \
            f"\nBuild it first (pure pandas, no GPU, ~10 min):  python -m player_priors")
    games = set()
    for path in sorted(root.glob("players_*.parquet")):
        games.update(int(g) for g in pd.read_parquet(path, columns=["game_id"])["game_id"])
    if not games:
        raise SystemExit(f"the priors sidecar at {root} is empty. Rebuild:  python -m player_priors")
    return len(games)


def append_prior_batches(batches: dict, cols: dict, idx, n: int, seq_len: int) -> None:
    """Pad one game's prior columns into ``(SEQ, ...)`` buffers, mirroring append_season_batches."""
    for key in PRIOR_LIST_COLS:
        buf = np.repeat(_DEFAULT_PLAYER[None, None, :], seq_len * ROSTER_SIZE, axis=0) \
            .reshape(seq_len, ROSTER_SIZE, N_PLAYER_PRIORS).astype(np.float32)
        buf[:n] = cols[key][idx]
        batches[key].append(buf)
    for key in TEAM_PRIOR_COLS:
        buf = np.zeros((seq_len, 1), dtype=np.float32)
        buf[:n, 0] = cols[key][idx]
        batches[key].append(buf)


def make_prior_inputs(seq_len: int):
    """``(player_inputs, team_inputs)`` Keras Inputs. Keras is imported here to keep this TF-free."""
    from tensorflow import keras

    players = {
        key: keras.Input(shape=(seq_len, ROSTER_SIZE, N_PLAYER_PRIORS), name=key, dtype="float32")
        for key in PRIOR_LIST_COLS
    }
    team = {key: keras.Input(shape=(seq_len, 1), name=key, dtype="float32")
            for key in TEAM_PRIOR_COLS}
    return players, team


def side_prior_scalars(prior_inputs: dict, side: str) -> list:
    """The ten per-player prior planes for one side, in ``PLAYER_PRIOR_KEYS`` order.

    Unstacked here rather than stored separately, so the model graph sees the same flat list of
    ``(B, SEQ, ROSTER_SIZE)`` scalars the roster encoder has always taken.

    ``keras.ops``, not ``tf``: under Keras 3 a raw TensorFlow op on a symbolic ``KerasTensor``
    raises rather than tracing, because the functional graph is backend-agnostic.
    """
    from keras import ops

    return ops.unstack(prior_inputs[f"prior_{side}"], num=N_PLAYER_PRIORS, axis=-1)


def prior_team_projections(team_inputs: dict) -> list:
    """A Dense(16) per team prior, mirroring season_features.season_team_projections."""
    from tensorflow.keras import layers

    return [layers.Dense(16, name=f"{key}_proj")(team_inputs[key]) for key in TEAM_PRIOR_COLS]


__all__ = [
    "PRIOR_LIST_COLS", "TEAM_PRIOR_COLS", "PRIOR_INPUT_KEYS", "N_PLAYER_PRIORS",
    "normalize_player", "normalize_team", "load_priors", "priors_for_season", "pad_priors",
    "build_raw_prior_cols", "merge_prior_features", "require_priors",
    "append_prior_batches",
    "make_prior_inputs",
    "side_prior_scalars", "prior_team_projections",
]
