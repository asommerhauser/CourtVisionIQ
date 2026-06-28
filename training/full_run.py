"""
full_run.py — one recency-weighted full train + a batched 100-game holdout.

The pivot away from the staged curriculum: train every model **once** on the whole corpus (up to a
cut partway through the most recent season), with older seasons down-weighted (see
``season_features`` recency weighting), then predict the next ``FINAL_HOLDOUT_GAMES`` real games a
batch at a time. Reuses the cleaned data + frozen vocab + warmup that ``init`` already produced — no
re-clean, no re-warmup — and writes its weights to ``FULL_ARTIFACTS_ROOT`` so the curriculum's
``./artifacts`` is never touched.

State machine (``full_run_state.json``), each step user-launched:

  setup : compute the cut (``FINAL_SEASON_FRACTION`` through the last season) + the next-100 holdout.
  train : one fresh full train of every model on the train slice -> FULL_ARTIFACTS_ROOT, then STOP.
  eval  : predict the next ``EVAL_BATCH`` holdout games (11 sims each) + write a report, then STOP.
  report: rebuild the aggregate report over everything finished.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config import (
    EVAL_BATCH, FINAL_HOLDOUT_GAMES, FINAL_SEASON_FRACTION, FULL_ARTIFACTS_ROOT,
    SEED, STAGE_SIMS, SUBSET_MODEL_KEYS, TEST_FRAC,
)
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT
from training.chronology import game_index, sequential_partition
from training.curriculum import STAGE_MODEL_KEYS
from training.subset import load_subset_games

DEFAULT_STATE_PATH = "./training/full_run_state.json"
RUN_NAME = "full_train"


class FullRun:
    """Single full-train + batched holdout eval; each public method is one CLI subcommand."""

    def __init__(self, state_path: str = DEFAULT_STATE_PATH):
        self.state_path = Path(state_path)
        self.state: dict = json.loads(self.state_path.read_text(encoding="utf-8")) \
            if self.state_path.exists() else {}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _require(self) -> None:
        if not self.state:
            raise SystemExit("No full-run state. Run:  python full_train.py setup")

    # --------------------------------------------------------------- setup
    def setup(self, *, data_dir: str = "./data", processed_dir: str = "./data/processed",
              epochs: int = 50, batch_size: int = 32) -> None:
        """Compute the train/holdout cut from the already-cleaned data (no re-clean / re-warmup)."""
        idx = game_index(data_dir)
        last_season = int(idx["season"].max())
        reg = idx[(idx["season"] == last_season) & idx["is_regular"]]
        if reg.empty:
            raise SystemExit(f"no regular-season games for the last season ({last_season}).")
        boundary = int(reg["pos"].min()) + int(FINAL_SEASON_FRACTION * len(reg))

        if boundary + FINAL_HOLDOUT_GAMES > len(idx):
            raise SystemExit(
                f"not enough games after the cut for a {FINAL_HOLDOUT_GAMES}-game holdout "
                f"(boundary {boundary}, corpus {len(idx)})."
            )
        _, _, holdout = sequential_partition(idx, boundary, n_holdout=FINAL_HOLDOUT_GAMES,
                                             val_frac=TEST_FRAC, seed=SEED)
        ordered = idx["game_id"].to_numpy()
        holdout_ids = [int(g) for g in ordered[boundary:boundary + FINAL_HOLDOUT_GAMES]]

        self.state = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "data_dir": data_dir, "processed_dir": processed_dir,
            "artifacts_root": FULL_ARTIFACTS_ROOT, "reports_root": DEFAULT_REPORTS_ROOT,
            "epochs": epochs, "batch_size": batch_size, "run_name": RUN_NAME,
            "n_games": int(len(idx)), "boundary_idx": boundary,
            "holdout_game_ids": holdout_ids, "eval_batch": EVAL_BATCH,
            "status": "setup", "trained_models": [],
        }
        self._save()

        by_id = idx.set_index("game_id")
        first, last = by_id.loc[holdout_ids[0]], by_id.loc[holdout_ids[-1]]
        print(f"[setup] last season {last_season}: cut at {int(FINAL_SEASON_FRACTION * 100)}% "
              f"of its regular season -> {boundary} train games.")
        print(f"[setup] holdout = {len(holdout_ids)} games (g{holdout_ids[0]} .. g{holdout_ids[-1]}, "
              f"{first['game_date']} .. {last['game_date']}), predicted {EVAL_BATCH} at a time.")
        print(f"[setup] full-train weights -> {FULL_ARTIFACTS_ROOT} (curriculum ./artifacts untouched)")
        print(f"State -> {self.state_path}\nNext:  python full_train.py train")

    # --------------------------------------------------------------- train
    def train(self) -> None:
        from models.pipeline import run_stage

        self._require()
        if self.state["status"] == "trained":
            print("[train] already trained — run:  python full_train.py eval")
            return
        idx = game_index(self.state["data_dir"])
        partition = sequential_partition(idx, self.state["boundary_idx"],
                                         n_holdout=FINAL_HOLDOUT_GAMES, val_frac=TEST_FRAC, seed=SEED)
        print(f"[train] one fresh full train on {self.state['boundary_idx']} games "
              f"(recency-weighted) -> {self.state['artifacts_root']}")

        # Small heads (config.SUBSET_MODEL_KEYS) train on the compact, modern-heavy per-season
        # subset if it has been extracted (python -m training.subset extract); the big player-vocab
        # heads keep the full corpus. No subset file => everything trains full, as before.
        subset_train = load_subset_games()
        if subset_train is not None:
            print(f"[train] small heads {list(SUBSET_MODEL_KEYS)} -> {len(subset_train)}-game subset; "
                  f"player/substitution/stint_length -> full corpus.")
        else:
            print("[train] no subset extracted — all heads train on the full corpus. "
                  "(Run `python -m training.subset extract` to enable the small-head subset.)")

        self.state["status"] = "training"
        self._save()

        sdict = self.state
        def on_trained(key: str) -> None:
            if key not in sdict["trained_models"]:
                sdict["trained_models"].append(key)
            self._save()

        run_stage(
            self.state["data_dir"], partition, artifacts_root=self.state["artifacts_root"],
            warm_start=False, refit_norm_stats=True, epochs=self.state["epochs"],
            batch_size=self.state["batch_size"], report=True, run_name=RUN_NAME,
            done=sdict["trained_models"], on_trained=on_trained,
            subset_keys=SUBSET_MODEL_KEYS, subset_train_games=subset_train,
        )

        self.state["status"] = "trained"
        self._save()
        print("\n" + "=" * 70)
        print("STOP — full train done. Predict the holdout 10 at a time:")
        print(f"  python full_train.py eval   (run it until all {FINAL_HOLDOUT_GAMES} are done)")
        print("=" * 70)

    # --------------------------------------------------------------- eval
    def eval(self) -> None:
        from simulation.stage_eval import evaluate_stage

        self._require()
        if self.state["status"] != "trained":
            print("[eval] not trained yet — run:  python full_train.py train")
            return
        report = evaluate_stage(
            RUN_NAME, holdout_ids=self.state["holdout_game_ids"], n_sims=STAGE_SIMS,
            max_new=self.state["eval_batch"], data_dir=self.state["data_dir"],
            processed_dir=self.state["processed_dir"], artifacts_root=self.state["artifacts_root"],
            reports_root=self.state["reports_root"],
        )
        done, total = report["done"], report["total"]
        print("\n" + "=" * 70)
        if done >= total:
            print(f"STOP — all {total} holdout games predicted. Final report:")
            print(f"  {report['run_dir']}")
        else:
            print(f"STOP — {done}/{total} holdout games done. Take a break if you like.")
            print("  When ready, predict the next batch:  python full_train.py eval")
        print("=" * 70)

    # ------------------------------------------------------- retrain-shot-type
    def retrain_shot_type(self) -> None:
        """Targeted retrain of ONLY the shot_type head, reusing the existing cond_*.npz.

        The shot_type head now masks its loss to live field goals ({2pt, 3pt}); free-throw shot
        rows — which the simulator never asks shot_type to choose (FTs come from fouls) — no longer
        pollute the binary 2pt-vs-3pt task. The shared conditional tensors are untouched (shot_result
        still needs the FT rows), so this reuses them as-is: no re-preprocess, no other head retrained.
        Fresh init; weights -> ``artifacts_root``/shot_type, overwriting the old head.
        """
        from encoder.encoder import Encoder
        from models.conditional_type_model import CONDITIONAL_MODEL_CLASSES

        self._require()
        cond_train = Path(self.state["processed_dir"]) / "cond_train.npz"
        if not cond_train.exists():
            raise SystemExit(
                f"{cond_train} not found — the conditional preprocess has not run yet. "
                f"Run `python full_train.py train` first (it writes the cond_*.npz)."
            )
        cls = next(c for c in CONDITIONAL_MODEL_CLASSES if c.KEY == "shot_type")
        model = cls(Encoder(), path=self.state["data_dir"], processed_dir=self.state["processed_dir"])
        print(f"[retrain-shot-type] fresh train on existing {cond_train.name} "
              f"(live field goals only) -> {self.state['artifacts_root']}/shot_type")
        model.train(
            epochs=self.state["epochs"], batch_size=self.state["batch_size"],
            artifacts_root=self.state["artifacts_root"], report=True, run_name=RUN_NAME,
            init_weights_root=None,
        )
        if "shot_type" not in self.state.setdefault("trained_models", []):
            self.state["trained_models"].append("shot_type")
            self._save()
        print("[retrain-shot-type] done — shot_type now trained on {2pt, 3pt} only.")

    # --------------------------------------------------------------- report / status
    def report(self) -> None:
        from simulation.stage_eval import evaluate_stage

        self._require()
        report = evaluate_stage(
            RUN_NAME, holdout_ids=self.state["holdout_game_ids"], n_sims=STAGE_SIMS, max_new=0,
            data_dir=self.state["data_dir"], processed_dir=self.state["processed_dir"],
            artifacts_root=self.state["artifacts_root"], reports_root=self.state["reports_root"],
        )
        print(f"[report] {report['done']}/{report['total']} games -> {report['run_dir']}")

    def status(self) -> None:
        self._require()
        print(f"Full run: status={self.state['status']}, "
              f"models trained {len(self.state.get('trained_models', []))}/{len(STAGE_MODEL_KEYS)}, "
              f"holdout {len(self.state['holdout_game_ids'])} games, "
              f"weights {self.state['artifacts_root']}")


__all__ = ["FullRun"]
