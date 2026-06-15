"""
Convenience API for querying the Parquet report data model.

Reports live one-folder-per-run (reports/<model_key>/<run_id>/). These helpers
glob the whole tree and concatenate it into tidy DataFrames so you can analyze
every run at once with plain pandas — no database server, no manual file walking.

Examples
--------
    from reporting.query import load_runs, load_epochs, learning_curve, best_runs

    runs = load_runs()                       # one row per run
    runs[runs.batch_size == 32]              # filter by hyperparameter
    best_runs(metric="best_val_loss")        # leaderboard, lowest first

    ep = load_epochs()                       # long format: run/epoch/metric/value
    learning_curve("20260614-101500-ab12cd") # val/train loss over epochs for a run

For ad-hoc SQL, the same frames work with DuckDB:
    import duckdb; duckdb.sql("SELECT model_key, MIN(best_val_loss) FROM runs GROUP BY 1")
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from reporting.report_artifacts import DEFAULT_REPORTS_ROOT


def _glob(root: str | Path, filename: str) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.glob(f"*/*/{filename}"))


def load_runs(root: str | Path = DEFAULT_REPORTS_ROOT) -> pd.DataFrame:
    """All run summaries across every model, one row per run."""
    files = _glob(root, "run.parquet")
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    if "started_at" in df.columns:
        df = df.sort_values("started_at", ascending=False, ignore_index=True)
    return df


def load_epochs(root: str | Path = DEFAULT_REPORTS_ROOT) -> pd.DataFrame:
    """All per-epoch metrics across every run, in long format."""
    files = _glob(root, "epochs.parquet")
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def best_runs(metric: str = "best_val_loss", ascending: bool = True,
              root: str | Path = DEFAULT_REPORTS_ROOT) -> pd.DataFrame:
    """Leaderboard of runs ordered by a summary metric (lowest loss first)."""
    runs = load_runs(root)
    if runs.empty or metric not in runs.columns:
        return runs
    return runs.sort_values(metric, ascending=ascending, ignore_index=True)


def learning_curve(run_id: str, metric: str = "loss",
                   root: str | Path = DEFAULT_REPORTS_ROOT) -> pd.DataFrame:
    """Train vs validation values of one metric over epochs for a single run."""
    ep = load_epochs(root)
    if ep.empty:
        return ep
    sub = ep[(ep.run_id == run_id) & (ep.metric == metric)]
    return sub.pivot_table(index="epoch", columns="split", values="value").reset_index()
