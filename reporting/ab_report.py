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


def read_arm(run_dirs, arm: str) -> tuple[dict, list[dict]]:
    """``(identity, records)`` for one arm, read from what its runs themselves wrote.

    The identity is reconstructed rather than passed in, so a mislabelled comparison is impossible from
    the command line: the windows, the sim count and the seed come from the run directories, and
    ``assert_comparable`` then judges them.

    **An arm may be several runs.** ``--window K`` scores exactly ``HOLDOUT_WINDOW_GAMES``, so the 700
    games W11's detection threshold rests on are seven runs, not one: at 100 games a 0.005-0.015 gain
    sits inside 2 SE (0.017) and is unreadable, and at 700 the threshold is 0.007 and it is not. The
    windows stay separate runs on purpose -- each records its own k, and drift with k is a finding --
    so they are pooled here, at the point of comparison, rather than merged into one evaluation.
    """
    from eval_pool import finished_records

    dirs = [Path(d) for d in ([run_dirs] if isinstance(run_dirs, (str, Path)) else run_dirs)]
    records, windows, sims_seen, seeds = [], [], set(), set()
    for run_dir in dirs:
        got = finished_records(run_dir)
        if not got:
            raise SystemExit(f"{run_dir} holds no finished games (games/*/record.json). "
                             f"Has this arm been evaluated?")
        report = _report_json(run_dir)
        records.extend(got)
        windows.append(int(report.get("window") or 0))
        sims_seen.add(int(report.get("n_sims") or max(int(r.get("n_sims", 0) or 0) for r in got)))
        seeds.update(int(r.get("seed_base") or 0) for r in got)

    if len(seeds) > 1:
        raise SystemExit(f"arm {arm!r} mixes seed bases {sorted(seeds)}; a shared seed is what makes "
                         f"two arms face the same Monte-Carlo draw, so this is not one arm")
    if len(sims_seen) > 1:
        raise SystemExit(f"arm {arm!r} mixes sim counts {sorted(sims_seen)}; Brier's Monte-Carlo "
                         f"inflation differs with the sim count, so these runs cannot be pooled")
    seen = {int(r["game_id"]) for r in records}
    if len(seen) != len(records):
        raise SystemExit(f"arm {arm!r} covers {len(records)} records over {len(seen)} distinct games; "
                         f"the windows given overlap, and a game scored twice would be paired twice")
    return describe_arm(
        arm,
        model=_report_json(dirs[0]).get("model") or dirs[0].parent.name,
        run=", ".join(d.name for d in dirs),
        window=min(windows),
        seed=sorted(seeds)[0],
        monte_carlo=sorted(sims_seen)[0],
        n_games=len(records),
        windows=sorted(windows),
        run_dir=", ".join(str(d) for d in dirs),
    ), records


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
    # assert_comparable judges one window field, which an arm spanning several runs cannot express:
    # arms on windows 0-6 and on window 0 alone both report a window of 0 and would pass. The pairing
    # is by game id, so a mismatch here is not fatal to the arithmetic -- it just silently compares
    # 700 games against the 100 they contain, and reports it as 700.
    spans = {tuple(i.get("windows", [i["window"]])) for i in identities}
    if len(spans) > 1:
        raise ValueError(f"arms cover different holdout windows ({sorted(spans)}); they would be "
                         f"compared on the games they happen to share, under the wider arm's name")

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
    first = run_dirs[0]
    out_dir = Path(out_dir or (first if isinstance(first, (str, Path)) else first[0]))
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
                    help="Each arm's results dir(s), IN ARM ORDER: retrained, rung2, kpi. An arm "
                         "spanning several holdout windows is one comma-separated argument, e.g. "
                         "results/version3.2/v32-a1-w0,results/version3.2/v32-a1-w1 -- 700 games is "
                         "seven windows, and that is what the detection threshold rests on.")
    ap.add_argument("--state", default=None,
                    help="Full-run state file to record the comparison in (optional).")
    ap.add_argument("--out", default=None,
                    help="Where the report goes (default: the first arm's run dir).")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if len(args.run_dirs) > len(ARMS):
        raise SystemExit(f"at most {len(ARMS)} arms ({', '.join(ARMS)}), got {len(args.run_dirs)}")
    per_arm = [[d for d in group.split(",") if d] for group in args.run_dirs]
    compare(per_arm, state_path=args.state, out_dir=args.out)


if __name__ == "__main__":
    main()


__all__ = ["REPORT_NAME", "compare", "pairs_for", "read_arm", "render_html"]
