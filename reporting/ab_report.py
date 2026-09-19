"""
The three arms, actually compared (3.2 W12's driver).

``reporting/ab_harness.py`` holds the statistics -- arm identities, the comparability refusal, the
paired Brier test and the detection threshold a result has to clear. Nothing called any of it. So the
3.2 handover could run all three arms and still have no comparison: three run folders, three separate
reports, and the reader left to eyeball two Brier numbers that a paired test says are the same model.

**What this does.** Reads the finished per-game records out of each arm's run directory, reconstructs
each arm's identity from what the run itself recorded, refuses the comparison if the arms are not
comparable, and runs the paired test over every pair -- consecutive arms, and first against last.

**Why the pairs and not a single number.** The arms are supersets: rung 2 is the retrained bundle with
checkpoint selection, and the KPI pass starts from rung 2's weights. So 1-vs-2 and 2-vs-3 are the two
increments, and 1-vs-3 is what 3.2 as a whole bought. A single "best arm" would hide which of the two
changes did the work, and this programme has already spent a cycle on differences that were not there.

**The comparison can refuse.** ``assert_comparable`` raises when the arms scored different windows,
different sim counts or different seeds. That is not defensive coding: a different window is different
games; a different sim count moves Brier's Monte-Carlo inflation and biases confidence buckets; a
different seed is an independent draw, which is right for a repeat of one model and wrong for a
comparison of two. All three have bitten this project, which is why the refusal is loud and early
rather than a footnote under a table of numbers that look fine.

**Both views.** The vote view (``win_prob_home``, the share of sims that won) and the Gaussian score
view are reported side by side, because they disagree in a readable way: the vote view saturates at
small sim counts and the score view does not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from reporting.ab_harness import ARMS, assert_comparable, compare_arms, describe_arm

#: Written beside the first arm's run directory unless told otherwise.
REPORT_NAME = "ab_report"


def _report_json(run_dir: Path) -> dict:
    path = run_dir / "report.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_arm(run_dir, arm: str) -> tuple[dict, list[dict]]:
    """``(identity, records)`` for one arm, read from what the run itself wrote.

    The identity is reconstructed rather than passed in, so a mislabelled comparison is impossible from
    the command line: the window, the sim count and the seed come from the run directory, and
    ``assert_comparable`` then judges them.
    """
    from eval_pool import finished_records

    run_dir = Path(run_dir)
    records = finished_records(run_dir)
    if not records:
        raise SystemExit(f"{run_dir} holds no finished games (games/*/record.json). "
                         f"Has this arm been evaluated?")
    report = _report_json(run_dir)
    sims = report.get("n_sims") or max(int(r.get("n_sims", 0) or 0) for r in records)
    seeds = {int(r.get("seed_base") or 0) for r in records}
    if len(seeds) > 1:
        raise SystemExit(f"{run_dir} mixes seed bases {sorted(seeds)}; it is not one run's worth of "
                         f"games and cannot stand as one arm")
    identity = describe_arm(
        arm,
        model=report.get("model") or run_dir.parent.name,
        run=report.get("run_name") or run_dir.name,
        window=int(report.get("window") or 0),
        seed=sorted(seeds)[0],
        monte_carlo=int(sims),
        n_games=len(records),
        run_dir=str(run_dir),
    )
    return identity, records


def pairs_for(n_arms: int) -> list[tuple[int, int]]:
    """Consecutive increments, plus first-against-last when there are three.

    With two arms the two are the same pair, and reporting it twice would suggest two findings.
    """
    if n_arms < 2:
        return []
    out = [(i, i + 1) for i in range(n_arms - 1)]
    if n_arms > 2:
        out.append((0, n_arms - 1))
    return out


def compare(run_dirs, *, arms=ARMS, state_path=None, out_dir=None, echo=print) -> dict:
    """Read every arm, refuse an incomparable set, and run the paired test over each pair."""
    from reporting.ab_harness import record_phase

    names = list(arms)[:len(run_dirs)]
    read = [read_arm(d, name) for d, name in zip(run_dirs, names)]
    identities = [identity for identity, _ in read]
    assert_comparable(identities)

    for identity in identities:
        echo(f"[ab] {identity['arm']:<10} {identity['n_games']:>4} games  "
             f"window {identity['window']}  seed {identity['seed']}  "
             f"{identity['monte_carlo']} sims  {identity['run_dir']}")

    comparisons = []
    for i, j in pairs_for(len(read)):
        for score in (False, True):
            result = compare_arms(read[i][1], read[j][1],
                                  label_a=identities[i]["arm"], label_b=identities[j]["arm"],
                                  score=score)
            comparisons.append(result)
            view = "score" if score else "vote"
            echo(f"[ab] {result['a']} vs {result['b']} ({view}): "
                 f"{result['brier_a']:.4f} -> {result['brier_b']:.4f}  "
                 f"diff {result['diff']:+.4f} +- {result['diff_se']:.4f} (2 SE "
                 f"{result['threshold_2se']:.4f})  {result['verdict']}")

    report = {"arms": identities, "comparisons": comparisons}
    out_dir = Path(out_dir or run_dirs[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{REPORT_NAME}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / f"{REPORT_NAME}.html").write_text(render_html(report), encoding="utf-8")
    echo(f"[ab] wrote {out_dir / REPORT_NAME}.json and .html")

    if state_path:
        record_phase(state_path, "ab", {"arms": identities, "comparisons": comparisons})
        echo(f"[ab] recorded in {state_path}")
    return report


# ===================================================================== #
# --- Output                                                           --
# ===================================================================== #

def render_html(report: dict) -> str:
    """A small self-contained page. Same shape as ``state_probes.render_html``, for the same reason:
    a comparison that lives only in a terminal is one nobody re-reads."""
    def esc(v):
        return ("" if v is None else str(v)).replace("&", "&amp;").replace("<", "&lt;")

    def num(v, spec=".4f"):
        return "-" if v is None else format(float(v), spec)

    arms = "".join(
        f"<tr><td>{esc(a['arm'])}</td><td>{esc(a['description'])}</td><td>{esc(a.get('n_games'))}</td>"
        f"<td>{esc(a['window'])}</td><td>{esc(a['seed'])}</td><td>{esc(a['monte_carlo'])}</td>"
        f"<td>{esc(a.get('run_dir'))}</td></tr>"
        for a in report["arms"])
    rows = "".join(
        f"<tr><td>{esc(c['a'])} vs {esc(c['b'])}</td>"
        f"<td>{'score' if c['score_view'] else 'vote'}</td><td>{esc(c['n'])}</td>"
        f"<td>{num(c.get('brier_a'))}</td><td>{num(c.get('brier_b'))}</td>"
        f"<td>{num(c.get('diff'), '+.4f')}</td><td>{num(c.get('diff_se'))}</td>"
        f"<td>{num(c.get('threshold_2se'))}</td>"
        f"<td class='{'sep' if c['separated'] else 'same'}'>{esc(c['verdict'])}</td></tr>"
        for c in report["comparisons"])
    return (
        "<!doctype html><meta charset='utf-8'><title>3.2 A/B arms</title>"
        "<style>body{font:14px system-ui;margin:2rem;max-width:70rem}"
        "table{border-collapse:collapse;margin:1rem 0;width:100%}"
        "th,td{border:1px solid #ccc;padding:.35rem .5rem;text-align:left}"
        "th{background:#f3f3f3}.sep{font-weight:700}.same{color:#666}"
        "p{color:#444}</style>"
        "<h1>3.2 A/B arms</h1>"
        "<p>Paired Brier over the games each pair shares. A difference inside two standard errors "
        "reads as the same model, which is the honest verdict rather than a cautious one.</p>"
        "<h2>Arms</h2><table><tr><th>arm</th><th>what it is</th><th>games</th><th>window</th>"
        f"<th>seed</th><th>sims</th><th>run</th></tr>{arms}</table>"
        "<h2>Comparisons</h2><table><tr><th>pair</th><th>view</th><th>n</th><th>brier a</th>"
        "<th>brier b</th><th>diff</th><th>SE</th><th>2 SE</th><th>verdict</th></tr>"
        f"{rows}</table>")


# ===================================================================== #
# --- CLI                                                              --
# ===================================================================== #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compare the 3.2 arms (retrained, rung2, kpi) with a paired Brier test.")
    ap.add_argument("run_dirs", nargs="+", metavar="RUNDIR",
                    help="Each arm's results dir, IN ARM ORDER: retrained, rung2, kpi.")
    ap.add_argument("--state", default=None,
                    help="Full-run state file to record the comparison in (optional).")
    ap.add_argument("--out", default=None,
                    help="Where the report goes (default: the first arm's run dir).")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if len(args.run_dirs) > len(ARMS):
        raise SystemExit(f"at most {len(ARMS)} arms ({', '.join(ARMS)}), got {len(args.run_dirs)}")
    compare(args.run_dirs, state_path=args.state, out_dir=args.out)


if __name__ == "__main__":
    main()


__all__ = ["REPORT_NAME", "compare", "pairs_for", "read_arm", "render_html"]
