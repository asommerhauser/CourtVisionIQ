"""
Per-head KPI metrics: each head scored on the metric closest to its own output (3.2 W9).

**Why not one shared scalar.** A single score punishes a head that did its job for another head's
failure, and that mis-assignment is why a generic score-function estimator needs hundreds of steps to
learn anything. Matching each head to the quantity it actually controls is the largest available
variance reduction and it costs nothing at rollout time -- the sims are already run.

**Everything is normalised by a measured cross-game standard deviation.** Raw averaging is dominated by
whichever number happens to be biggest: points (sd 12.1 per team-game) would drown blocks (sd 2.5), and
`seconds` would drown everything. So each error is expressed in "how many games' worth of variation is
this", which is what makes errors on different stats commensurable.

*Correction to* ``v3_2_direction.md`` §4.4: the rule also says to "drop ``seconds`` from the team
aggregate entirely" and that ``eval_metrics`` "already says this in its headline block". The first half
is already done -- ``_BOX_ACCURACY_STATS`` carries derived ``minutes`` and no ``seconds`` at all -- and
there is no such statement in the code. Nothing to do; recorded so the next reader does not go looking.

**Rate-normalisation is the principle behind the two least obvious rows.** ``shot_result`` is scored on
effective field goal percentage rather than team points, because dividing by attempts stops a
too-fast simulator being charged to the shooting head. ``shot_type`` must **not** get eFG%: it controls
the shot mix, and the cheapest way to win an efficiency metric by changing the mix is to shoot more
threes, which eFG counts at 1.5. Same reason ``player`` is scored on each man's *share* of his team's
total rather than on raw counts.

**Three exclusions from the event head's own histogram**, which follow from the token vocabulary rather
than from preference. Free throws drop out -- there is no free-throw *event* token, attempts are produced
by the rules engine downstream of a shooting foul, so scoring the head on them charges it for a rule it
does not control. Steals stay with ``turnover_type``, because the event head emits ``turnover`` and
whether it is credited as a steal is the sub-type split. And substitutions are *counted*, not converted
to minutes: the head controls how often an opportunity fires, not who goes in or for how long.
"""
from __future__ import annotations

import math

from simulation.stats import team_totals

# Measured on the real 2022-23 season (1,320 games / 2,640 team-games) via ``generate_box_score``.
# Frozen constants in the style of ``models/prior_features._NORM`` and ``rotation_features._NORM``: a
# fitted statistic would mean a persist site, a refit branch and read-backs, for a quantity that is
# era-stable in natural units.
#
#   python -c "..."  -- see docs/v3_2_progress.md for the measurement that produced these.
TEAM_SD = {
    "pts": 12.065, "fga": 7.312, "fgm": 5.095, "tpa": 7.009, "tpm": 3.952,
    "fta": 6.882, "ftm": 5.791, "oreb": 3.975, "dreb": 5.364, "ast": 4.891,
    "stl": 2.872, "blk": 2.460, "tov": 3.867, "pf": 4.076,
    "efg": 0.066, "tpa_rate": 0.077, "oreb_share": 0.075,
}

#: Per-game counts over both teams, for the two event tokens that are not box stats.
GAME_SD = {"substitutions": 8.512, "timeouts": 1.916, "possessions": 5.678, "margin": 13.66}

#: One player's share of his team's total, and his minutes. Shares are bounded so their spread is
#: small; minutes is the rotation scale the substitution heads are judged on.
PLAYER_SD = {"share": 0.06, "minutes": 9.0}

#: The event head's eight real tokens, and the box quantity each one produces. ``event_vocab.json``
#: also holds PAD / UNK / start / end / none, which are not sampled outcomes.
EVENT_TOKEN_STATS = {
    "shot": ("fga",),
    "rebound": ("oreb", "dreb"),
    "assist": ("ast",),
    "turnover": ("tov",),
    "block": ("blk",),
    "foul": ("pf",),
    "substitution": ("substitutions",),
    "timeout": ("timeouts",),
}

#: What each head is scored on. Keys are ``models.registry.STAGE_MODEL_KEYS``.
HEAD_METRICS = {
    # The histogram of its own output vocabulary.
    "event_time": ("fga", "oreb", "dreb", "ast", "tov", "blk", "pf", "substitutions", "timeouts"),
    # How long the game takes and how many possessions fit in it.
    "event_time_cond": ("possessions",),
    # The mix, never the efficiency.
    "shot_type": ("tpa_rate", "tpa", "fta"),
    # The efficiency, rate-normalised.
    "shot_result": ("efg",),
    "assist_type": ("ast",),
    "turnover_type": ("tov", "stl"),
    "foul_type": ("pf",),
    "rebound_type": ("oreb_share",),
    "timeout_team": ("timeouts",),
    # Who did it, as a share of the team's total, weighted by real minutes.
    "player": ("player_share",),
    # The rotation scale.
    "substitution": ("player_minutes",),
    "sub_decision": ("player_minutes",),
}

#: Every head also carries a small shared term, because the winner is a joint product of all twelve.
#: Per SIM that term is the margin error -- a Brier needs a distribution over sims, so it belongs to the
#: aggregate (``rollout_selection.rollout_score``) rather than to a single game-sim.
SHARED_WEIGHT = 0.25

#: The foul head additionally carries the game-state probes, which is the only place a behaviour rather
#: than a count reaches a head. Weighted so a fully-absent benching behaviour costs about as much as a
#: standard deviation of foul-count error.
PROBE_WEIGHT = 1.0
PROBE_HEADS = ("foul_type", "substitution", "sub_decision")


def count_events(rows) -> dict:
    """Substitution and timeout counts for one game's rows, on either the sim or the real side.

    Six of the event head's eight tokens map onto ``BOX_STATS`` and come free from the box score. These
    two do not: nobody accrues a substitution, so it appears in no stat line.
    """
    subs = timeouts = 0
    for row in rows:
        event = str(row.get("event", "") if hasattr(row, "get") else row["event"])
        if event == "substitution":
            subs += 1
        elif event == "timeout":
            timeouts += 1
    return {"substitutions": subs, "timeouts": timeouts}


def _rates(total: dict) -> dict:
    """The three rate-normalised team quantities, from one side's counting totals."""
    fga = float(total.get("fga", 0.0) or 0.0)
    fgm = float(total.get("fgm", 0.0) or 0.0)
    tpm = float(total.get("tpm", 0.0) or 0.0)
    tpa = float(total.get("tpa", 0.0) or 0.0)
    oreb = float(total.get("oreb", 0.0) or 0.0)
    dreb = float(total.get("dreb", 0.0) or 0.0)
    reb = oreb + dreb
    return {
        "efg": ((fgm + 0.5 * tpm) / fga) if fga else 0.0,
        "tpa_rate": (tpa / fga) if fga else 0.0,
        "oreb_share": (oreb / reb) if reb else 0.0,
    }


def game_stats(box, rows=None) -> dict:
    """Everything the per-head metrics read, for one game (one sim, or the real thing).

    ``box`` is a ``BoxScore``; ``rows`` its play-by-play, needed only for the two counted tokens.
    """
    sides = {}
    for name, lines in (("home", box.home), ("away", box.away)):
        total = team_totals(lines)
        stats = {k: float(total.get(k, 0.0) or 0.0) for k in
                 ("pts", "fga", "fgm", "tpa", "tpm", "fta", "ftm",
                  "oreb", "dreb", "ast", "stl", "blk", "tov", "pf")}
        stats.update(_rates(total))
        # Possessions from the same definition simulation.stats uses, so sim and real agree.
        from simulation.stats import possessions
        stats["possessions"] = float(possessions(total))
        sides[name] = stats
        sides[name]["players"] = {
            str(pl.player): {"pts": float(pl.pts), "fga": float(pl.fga), "ast": float(pl.ast),
                             "reb": float(pl.oreb) + float(pl.dreb),
                             "minutes": float(pl.seconds) / 60.0}
            for pl in lines
        }
    counts = count_events(rows) if rows is not None else {"substitutions": 0.0, "timeouts": 0.0}
    margin = sides["home"]["pts"] - sides["away"]["pts"]
    return {"sides": sides, "counts": counts, "margin": margin,
            "counted": rows is not None}


def _team_error(sim: dict, real: dict, key: str) -> float:
    """Mean absolute error over the two sides, in standard deviations."""
    sd = TEAM_SD.get(key) or GAME_SD.get(key) or 1.0
    gaps = [abs(sim["sides"][side].get(key, 0.0) - real["sides"][side].get(key, 0.0))
            for side in ("home", "away")]
    return (sum(gaps) / len(gaps)) / sd


def _count_error(sim: dict, real: dict, key: str) -> float:
    if not (sim.get("counted") and real.get("counted")):
        return 0.0            # no play-by-play on one side: abstain rather than score a zero as perfect
    sd = GAME_SD.get(key, 1.0)
    return abs(float(sim["counts"].get(key, 0.0)) - float(real["counts"].get(key, 0.0))) / sd


def _player_share_error(sim: dict, real: dict) -> float:
    """Each player's share of his team's total, weighted by his REAL minutes.

    Shares rather than counts, for the same reason ``shot_result`` gets eFG% and not points: a simulator
    playing too fast inflates every count, and that is the event head's failure, not the player head's.
    Weighted by real minutes so the men who actually played dominate the number.
    """
    num = den = 0.0
    for side in ("home", "away"):
        s_players, r_players = sim["sides"][side]["players"], real["sides"][side]["players"]
        for stat in ("pts", "fga", "ast", "reb"):
            s_tot = sum(p[stat] for p in s_players.values()) or 1.0
            r_tot = sum(p[stat] for p in r_players.values()) or 1.0
            for name, r in r_players.items():
                w = max(r["minutes"], 0.0)
                if w <= 0.0:
                    continue
                s = s_players.get(name, {}).get(stat, 0.0)
                num += w * abs((s / s_tot) - (r[stat] / r_tot))
                den += w
    return (num / den) / PLAYER_SD["share"] if den else 0.0


def _player_minutes_error(sim: dict, real: dict) -> float:
    """Minutes error over the men who really played, in standard deviations.

    Judged here rather than on rotation CRPS because a single sim has no distribution to score. The
    standing rule that minutes MAE structurally prefers a flat rotation still holds -- which is why the
    substitution heads ALSO carry the game-state probes, where a flat rotation is what loses.
    """
    num = den = 0.0
    for side in ("home", "away"):
        s_players, r_players = sim["sides"][side]["players"], real["sides"][side]["players"]
        for name, r in r_players.items():
            if r["minutes"] <= 0.0:
                continue
            s = s_players.get(name, {}).get("minutes", 0.0)
            num += abs(s - r["minutes"])
            den += 1.0
    return (num / den) / PLAYER_SD["minutes"] if den else 0.0


def probe_gap(probes: dict | None) -> float:
    """Mean relative gap over the scored probe rows, or 0.0 when there are none."""
    rows = (probes or {}).get("rows") or []
    gaps = []
    for row in rows:
        sim, real = row.get("sim"), row.get("real")
        if sim is None or real is None or not real:
            continue
        gaps.append(abs(float(sim) - float(real)) / abs(float(real)))
    return (sum(gaps) / len(gaps)) if gaps else 0.0


def head_errors(sim: dict, real: dict, *, probes: dict | None = None) -> dict:
    """Per-head error for one sim against the real game. Lower is better, roughly in sd units.

    ``sim`` and ``real`` are :func:`game_stats` outputs. ``probes`` is the behaviour report, which only
    the rotation-adjacent heads carry.
    """
    shared = SHARED_WEIGHT * (abs(sim["margin"] - real["margin"]) / GAME_SD["margin"])
    behaviour = probe_gap(probes)

    out = {}
    for head, metrics in HEAD_METRICS.items():
        parts = []
        for metric in metrics:
            if metric == "player_share":
                parts.append(_player_share_error(sim, real))
            elif metric == "player_minutes":
                parts.append(_player_minutes_error(sim, real))
            elif metric in ("substitutions", "timeouts"):
                parts.append(_count_error(sim, real, metric))
            else:
                parts.append(_team_error(sim, real, metric))
        score = (sum(parts) / len(parts)) if parts else 0.0
        if head in PROBE_HEADS:
            score += PROBE_WEIGHT * behaviour
        out[head] = score + shared
    return out


def total_error(sim: dict, real: dict, *, probes: dict | None = None) -> float:
    """One number for the whole sim: the mean of the per-head errors.

    Used where a scalar is genuinely wanted -- ranking a game's sibling sims, for instance -- while the
    per-head values are what the replay pass weights each head by.
    """
    errors = head_errors(sim, real, probes=probes)
    finite = [v for v in errors.values() if math.isfinite(v)]
    return (sum(finite) / len(finite)) if finite else float("inf")


__all__ = [
    "EVENT_TOKEN_STATS", "GAME_SD", "HEAD_METRICS", "PLAYER_SD", "PROBE_HEADS", "PROBE_WEIGHT",
    "SHARED_WEIGHT", "TEAM_SD", "count_events", "game_stats", "head_errors", "probe_gap",
    "total_error",
]
