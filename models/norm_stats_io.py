"""
Per-model normalization-stats persistence for staged (curriculum) training.

Each model's ``preprocess`` standardizes its continuous inputs/targets with stats computed from
its train split (time scaling, per-player rest, and — for the stint head — log-stint scaling).
In a single full train that's fine. But the curriculum **warm-starts** each stage from the prior
stage's weights, so the standardization MUST stay fixed across stages: if ``delta_std`` (or the
stint scale) shifted between stages, the loaded regression head would predict in the wrong units.

So the warmup-fit pass computes every model's stats once over the full corpus and persists them
here; every staged ``preprocess(refit_norm_stats=False)`` then *loads* these instead of recomputing
from its (smaller, shifting) slice. Stored per model under ``processed_dir`` keyed by model key.
"""
from __future__ import annotations

import json
from pathlib import Path


def norm_stats_path(processed_dir, key: str) -> Path:
    return Path(processed_dir) / f"{key}_norm_stats.json"


def save_norm_stats(processed_dir, key: str, stats: dict) -> Path:
    p = norm_stats_path(processed_dir, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return p


def load_norm_stats(processed_dir, key: str) -> dict:
    p = norm_stats_path(processed_dir, key)
    if not p.exists():
        raise FileNotFoundError(
            f"No persisted norm stats for '{key}' at {p.resolve()}; run the warmup fit "
            f"(preprocess with refit_norm_stats=True) before a staged refit_norm_stats=False run."
        )
    return json.loads(p.read_text(encoding="utf-8"))


__all__ = ["norm_stats_path", "save_norm_stats", "load_norm_stats"]
