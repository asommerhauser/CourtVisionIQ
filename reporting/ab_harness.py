"""
The A/B harness for 3.2's post-retrain arms, and the run-state records behind it (W12).

Three arms, in the order they can exist: the retrained bundle, that bundle with rung 2's checkpoint
selection, and that with the KPI replay pass. Steps 1-7 of the build are one unit -- nothing in them can
be A/B'd separately without a second retrain, which is the price of having one retrain -- so these three
are the only comparisons 3.2 can actually make, and §5.3's multi-scale time is held for 3.3 to keep them
clean.

**The seed is held FIXED across arms.** ``v3_direction.md`` §8 says repeat runs on the same window use a
different ``--seed`` so they are independent Monte-Carlo draws. That rule is for *repeats of one model*.
This is a model comparison, where a shared seed makes the two arms face the same Monte-Carlo draw and
removes that variance from the difference -- and the run log has to say so, or the next reader will apply
the standing rule and conclude the comparison was done wrong.

**Read it at 700 games, paired.** The per-game Brier sd measured on run4 is **0.174**, not the 0.133 §8
assumed, and the per-game sd of the *difference* between two arms is about half that again (0.087, from §9.2's
paired run3-vs-run4 measurement). So the two-standard-error detection threshold is 0.036 unpaired at
n = 100, 0.017 paired at n = 100, and 0.007 paired at n = 700. The expected gain is 0.005-0.015, which is borderline at 100 games
and clears the floor at 700. ``FINAL_HOLDOUT_GAMES = 700`` exists for exactly this, and window 0 is
byte-identical to the games v2-run1..4 scored -- provided the corpus cut preserved game ids, which is why
W2 applies its floor to rows rather than to the file list.

**Brier will move least, and that is structural.** It is the metric the project most wants and the one
this cycle helps least: the winner is mostly decided by pre-game team strength, which lives in the priors,
not by how faithfully the fourth quarter composes. The one real mechanism is indirect -- margin is
over-dispersed at 16.5 against a real 13.7, and over-dispersion flattens win probabilities toward 0.5, so
tightening it makes the same picks score better. **Read the probes at the development check, not Brier.**
"""
from __future__ import annotations

import json
import math
from pathlib import Path

#: One run's per-game Brier sd, measured on v2-run4. ``v3_direction.md`` §8 assumed 0.133; §9.2 corrected
#: it to this, and quotes 2 SE of it -- ±0.036 at n = 100 -- as the "two runs are the same model" band.
BRIER_SD = 0.174

#: The per-game sd of the DIFFERENCE between two arms, which is the quantity a comparison actually rests
#: on and is much smaller because two arms on the same games make correlated errors. Derived from the
#: measurement §9.2 records: paired run3-vs-run4 came to ±0.0109 at one standard error over the 64 games
#: they share, so the difference sd is 0.0109 × sqrt(64) ≈ 0.087 -- almost exactly half a single run's.
#:
#: Using BRIER_SD for a paired comparison would overstate the threshold by 2x and hide every gain 3.2
#: expects. That is the whole reason §7.2's table has two rows for n = 100.
PAIRED_DIFF_SD = 0.087

#: The arms, in the order they can exist. Each is a superset of the one before it.
ARMS = ("retrained", "rung2", "kpi")

ARM_DESCRIPTIONS = {
    "retrained": "one retrain carrying 3.0 and 3.2 together (W1-W7)",
    "rung2": "+ checkpoint selection on rollout metrics (W8)",
    "kpi": "+ the weighted replay pass (W9-W11)",
}

#: Where each phase's record lands in the run state, beside ``checkpoint_selection``.
PHASE_KEYS = {"kpi": "replay_pass", "ab": "ab_arms"}


def detection_threshold(n: int, *, sd: float | None = None, paired: bool = True) -> float:
    """Two standard errors of a Brier comparison at ``n`` games.

    Reproduces ``v3_2_direction.md`` §7.2: **0.036 unpaired at n = 100, 0.017 paired at n = 100, 0.007
    paired at n = 700.** Paired is the one to use -- two arms on the same games make correlated errors, so
    the sd of the per-game *difference* is about half either arm's own sd, and using the single-run figure
    would double the threshold and hide every gain 3.2 expects.

    One honest caveat on the unpaired row: a strictly-correct unpaired comparison of two independent runs
    carries a further factor of sqrt(2), which would make it 0.049 rather than 0.036. The 0.036 is 2 SE of
    a *single* run's Brier, which is the form §9.2 recorded and the project's documents quote. Reproduced
    as recorded rather than silently corrected, because the number appears elsewhere and a harness that
    disagreed with it would just look wrong.
    """
    if n <= 0:
        return float("inf")
    if sd is None:
        sd = PAIRED_DIFF_SD if paired else BRIER_SD
    return 2.0 * (sd / math.sqrt(n))


def readable_at(n: int, expected_gain: float, **kw) -> bool:
    """Whether a gain of ``expected_gain`` clears two standard errors at ``n`` games."""
    return abs(float(expected_gain)) >= detection_threshold(n, **kw)


def describe_arm(name: str, *, model: str, run: str, window: int, seed: int,
                 monte_carlo: int, **extra) -> dict:
    """One arm's identity, in the form the comparison has to be able to defend later.

    ``monte_carlo`` is recorded because confidence buckets are not comparable across runs with different
    sim counts -- selecting on ``p_hat >= 0.70`` from n sims is biased even for a perfect model, by +9pp
    at 20 sims and +4.6 at 50 -- so the sim count belongs next to every number it produced.
    """
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; expected one of {ARMS}")
    return {"arm": name, "description": ARM_DESCRIPTIONS[name], "model": model, "run": run,
            "window": int(window), "seed": int(seed), "monte_carlo": int(monte_carlo), **extra}


def assert_comparable(arms: list[dict]) -> None:
    """Refuse a comparison that is not one.

    Three ways a pair of arms stops being a comparison, all of which have bitten this project: a
    different holdout window (different games), a different sim count (biased buckets, and a different
    Monte-Carlo inflation of Brier), and a different seed (an independent draw, which is right for a
    repeat and wrong for a model comparison).
    """
    if len(arms) < 2:
        return
    for field, why in (("window", "they scored different games"),
                       ("monte_carlo", "Brier's Monte-Carlo inflation differs with the sim count, and "
                                       "confidence buckets are not comparable across sim counts"),
                       ("seed", "a different seed is an independent Monte-Carlo draw -- right for a "
                                "repeat of one model, wrong for a comparison of two")):
        values = {a.get(field) for a in arms}
        if len(values) > 1:
            raise ValueError(f"arms differ in {field} ({sorted(values)}): {why}")


def compare_arms(records_a: list[dict], records_b: list[dict], *, label_a: str, label_b: str,
                 score: bool = False) -> dict:
    """Paired Brier comparison of two arms, with the threshold the result has to clear.

    ``verdict`` is deliberately blunt: "the same model" is the correct reading of a difference inside two
    standard errors, and the 2.0 cycle spent a lot of effort on differences that were not there.
    """
    from simulation.eval_metrics import paired_brier
    paired = paired_brier(records_a, records_b, score=score)
    n = int(paired.get("n", 0))
    threshold = detection_threshold(n)
    diff = float(paired.get("diff", 0.0))
    return {
        "a": label_a, "b": label_b, "n": n, "score_view": bool(score),
        **{k: paired[k] for k in ("brier_a", "brier_b", "diff", "diff_se", "z") if k in paired},
        "threshold_2se": threshold,
        "separated": abs(diff) >= threshold,
        "verdict": ("separated" if abs(diff) >= threshold else "the same model"),
    }


def replay_pass_record(*, games: int, sims_per_game: int, head_kept: dict,
                       seed: int, window: int | None = None) -> dict:
    """What the KPI pass did, for the run state.

    ``head_kept`` is ``{head: n_sims_kept}``. Recorded per head because the filter is per head: a head
    every sim failed equally gets no update at all, and that is a finding rather than a failure -- without
    this number it would look identical to a pass that trained on everything.
    """
    total = games * sims_per_game
    kept = sum(int(v) for v in head_kept.values())
    return {
        "games": int(games), "sims_per_game": int(sims_per_game), "game_sims": total,
        "seed": int(seed), "window": window,
        "head_kept": {str(k): int(v) for k, v in sorted(head_kept.items())},
        "kept_total": kept,
        "kept_fraction_mean": (kept / (len(head_kept) * total)) if head_kept and total else 0.0,
    }


def record_phase(state_path, key: str, record: dict) -> None:
    """Merge a phase record into the run state, beside ``checkpoint_selection``.

    Same read-modify-write shape as ``rollout_selection.record_selection``, and the same silence when the
    state file is absent: a record is worth having, and worth nothing at the cost of failing a finished
    run over it.
    """
    p = Path(state_path)
    if not p.is_file():
        return
    state = json.loads(p.read_text(encoding="utf-8"))
    field = PHASE_KEYS.get(key, key)
    existing = state.get(field)
    if isinstance(existing, dict):
        existing.update(record)
    else:
        state[field] = dict(record)
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


__all__ = [
    "ARMS", "ARM_DESCRIPTIONS", "BRIER_SD", "PAIRED_DIFF_SD", "PHASE_KEYS",
    "assert_comparable",
    "compare_arms", "describe_arm", "detection_threshold", "readable_at", "record_phase",
    "replay_pass_record",
]
