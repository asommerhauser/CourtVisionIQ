"""
growth_report.py — the end-of-curriculum "how the model grew" report.

After every stage is evaluated, this stitches the per-stage records into one cross-stage view:
for each stage, its cumulative training size, the per-model best validation loss (from that stage's
training runs), and the eval headline on its next-N holdout (win-pick accuracy, Brier, spread MAE,
points MAE). The result is the storyline the user asked for — does the simulator predict real games
better as it sees more of the league's history? Written as a self-contained HTML table plus a
queryable Parquet (same reports/ conventions as the training + eval reports).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reporting.report_artifacts import DEFAULT_REPORTS_ROOT, ReportArtifacts, new_run_id

# Eval headline metrics carried into the growth table (from each stage's eval aggregate).
_HEADLINE = ("pick_accuracy", "brier", "spread_mae", "points_mae")


def build_growth_report(state: dict, *, reports_root: str = DEFAULT_REPORTS_ROOT) -> str:
    """Build the cross-stage growth report from the curriculum ``state``; return its run dir."""
    from reporting.query import load_runs

    runs = load_runs(reports_root)
    has_runs = not runs.empty and {"run_name", "model_key", "best_val_loss"} <= set(runs.columns)

    rows: list[dict] = []
    for entry in state["schedule"]:
        n = entry["stage"]
        sdict = state.get("stages", {}).get(str(n), {})
        row = {
            "stage": n, "boundary_type": entry["boundary_type"], "season": entry["season"],
            "train_games": entry["train_games"], "status": sdict.get("status", "pending"),
        }

        # Per-model best val loss from this stage's training runs (latest wins on a resumed retrain).
        run_name = sdict.get("report_run_name")
        if has_runs and run_name:
            stage_runs = runs[runs["run_name"] == run_name].drop_duplicates("model_key", keep="first")
            losses = []
            for _, r in stage_runs.iterrows():
                val = r.get("best_val_loss")
                row[f"val_{r['model_key']}"] = None if pd.isna(val) else float(val)
                if pd.notna(val):
                    losses.append(float(val))
            row["mean_best_val_loss"] = float(np.mean(losses)) if losses else None

        # Eval headline on this stage's next-N holdout.
        eval_dir = sdict.get("eval_run_dir")
        if eval_dir and (Path(eval_dir) / "report.json").exists():
            headline = json.loads((Path(eval_dir) / "report.json").read_text(encoding="utf-8")) \
                .get("aggregate", {}).get("headline", {})
            for k in _HEADLINE:
                row[k] = headline.get(k)

        rows.append(row)

    df = pd.DataFrame(rows)

    arts = ReportArtifacts.for_run("growth", new_run_id("growth"), root=reports_root)
    run_dir = arts.ensure_dir()
    df.to_parquet(run_dir / "growth.parquet", index=False)
    (run_dir / "growth.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    (run_dir / "growth.html").write_text(_render_html(df, state), encoding="utf-8")
    return str(run_dir)


def _render_html(df: pd.DataFrame, state: dict) -> str:
    table = df.to_html(index=False, na_rep="", border=0,
                       float_format=lambda x: f"{x:.4f}", classes="growth")
    n_stages = len(state.get("schedule", []))
    n_games = state.get("n_games", "?")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Curriculum growth report</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.4rem; }}
 p.meta {{ color: #555; }}
 table.growth {{ border-collapse: collapse; font-size: 0.85rem; }}
 table.growth th, table.growth td {{ padding: 4px 10px; border-bottom: 1px solid #e2e2e2;
   text-align: right; }}
 table.growth th {{ background: #f5f5f5; text-align: right; position: sticky; top: 0; }}
 table.growth td:nth-child(2), table.growth th:nth-child(2) {{ text-align: left; }}
</style></head><body>
<h1>Curriculum growth report</h1>
<p class="meta">{n_stages} stages over {n_games} games. Per stage: cumulative training size,
per-model best validation loss, and the simulator's accuracy on that stage's next-N holdout
(win-pick accuracy, Brier, point-spread MAE, team-points MAE).</p>
{table}
</body></html>"""


__all__ = ["build_growth_report"]
