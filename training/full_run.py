"""
full_run.py — one recency-weighted full train + a batched 100-game holdout (the engine behind the
``train.py`` / ``evaluate.py`` CLIs).

Train every model **once** on the whole corpus (up to a cut partway through the most recent season),
with older seasons down-weighted (see ``season_features`` recency weighting), then predict the next
``FINAL_HOLDOUT_GAMES`` real games a batch at a time. Weights go to a versioned root
``artifacts/v<version>/`` (see ``models.artifacts.version_root``) so each train keeps its own dir.

State machine (``full_run_state.json``), each step user-launched:

  setup      : compute the cut (``FINAL_SEASON_FRACTION`` through the last season) + next-N holdout.
  train      : one fresh full train of every model on the train slice -> artifacts/v<version>/.
  retrain    : retrain ONE head in place, keeping the rest (train.py --model <name>).
  eval       : predict the next ``EVAL_BATCH`` holdout games + write a report (evaluate.py).
  report     : rebuild the aggregate report over everything finished.

``train.py --full --version X.Y --batch-size N`` runs setup+train; ``--continue`` re-runs train
(resumes at the next unfinished head); ``--model <name>`` runs retrain.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config import (
    DEFAULT_VERSION, EVAL_BATCH, EVAL_GAMES_PER_BATCH, FINAL_HOLDOUT_GAMES, FINAL_SEASON_FRACTION,
    ROLLOUT_BATCH_SIZE, SEED, STAGE_SIMS, SUBSET_MODEL_KEYS, TEST_FRAC,
)
from models.artifacts import version_root
from models.registry import STAGE_MODEL_KEYS
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT
from training.chronology import game_index, sequential_partition
from training.subset import load_subset_games

DEFAULT_STATE_PATH = "./training/full_run_state.json"


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
            raise SystemExit("No full-run state. Run:  python train.py --full --version X.Y --batch-size N")

    # --------------------------------------------------------------- setup
    def setup(self, *, version: str | None = None, data_dir: str = "./data",
              processed_dir: str = "./data/processed", epochs: int = 50, batch_size: int = 64) -> None:
        """Compute the train/holdout cut from the already-cleaned data (no re-clean / re-warmup).

        ``version`` (e.g. ``"1.0"``) names the weights dir (``artifacts/v<version>/``) and the report
        label (``v<version>``). Defaults to ``DEFAULT_VERSION`` so existing callers/tests keep working.
        """
        version = version or DEFAULT_VERSION
        artifacts_root = version_root(version)
        run_name = f"v{version}"
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
            "version": version, "data_dir": data_dir, "processed_dir": processed_dir,
            "artifacts_root": artifacts_root, "reports_root": DEFAULT_REPORTS_ROOT,
            "epochs": epochs, "batch_size": batch_size, "run_name": run_name,
            "n_games": int(len(idx)), "boundary_idx": boundary,
            "holdout_game_ids": holdout_ids, "eval_batch": EVAL_BATCH,
            "status": "setup", "trained_models": [],
        }
        self._save()

        by_id = idx.set_index("game_id")
        first, last = by_id.loc[holdout_ids[0]], by_id.loc[holdout_ids[-1]]
        print(f"[setup] version {version}: cut at {int(FINAL_SEASON_FRACTION * 100)}% "
              f"of season {last_season}'s regular schedule -> {boundary} train games.")
        print(f"[setup] holdout = {len(holdout_ids)} games (g{holdout_ids[0]} .. g{holdout_ids[-1]}, "
              f"{first['game_date']} .. {last['game_date']}), predicted {EVAL_BATCH} at a time.")
        print(f"[setup] full-train weights -> {artifacts_root}")
        print(f"State -> {self.state_path}\nNext:  python train.py --full --version {version} "
              f"--batch-size {batch_size}")

    # --------------------------------------------------------------- train
    def train(self, *, rebuild_vocabs: bool = False) -> None:
        from models.pipeline import run_stage

        self._require()
        if self.state["status"] == "trained":
            print("[train] already trained — run:  python evaluate.py --version "
                  f"{self.state.get('version', DEFAULT_VERSION)}")
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
            batch_size=self.state["batch_size"], report=True, run_name=self.state["run_name"],
            done=sdict["trained_models"], on_trained=on_trained,
            subset_keys=SUBSET_MODEL_KEYS, subset_train_games=subset_train,
            rebuild_vocabs=rebuild_vocabs,
        )

        self.state["status"] = "trained"
        self._save()
        version = self.state.get("version", DEFAULT_VERSION)
        print("\n" + "=" * 70)
        print("STOP — full train done. Evaluate the holdout:")
        print(f"  python evaluate.py --version {version}")
        print("=" * 70)

    # --------------------------------------------------------------- eval
    def eval(self, *, version: str | None = None, name: str | None = None,
             n_sims: int | None = None, concurrency: int | None = None,
             max_new: int | None = None, report_every: int | None = None) -> None:
        """Predict the holdout into a results run at results/v<version>/<eval-name>/.

        ``version`` defaults to the trained run's version. ``name`` names the eval folder (default:
        auto-increment ``eval-NNN``, resuming the latest incomplete one). ``n_sims`` = Monte-Carlo
        sims per game (--monte-carlo). ``concurrency`` = concurrent game-sims per batched GPU forward
        pass — the **VRAM knob**: attention memory grows with batch × seq², so lower it if you OOM,
        raise it to use more of the card. It is decoupled from n_sims: the holdout's pooled sims run
        in cohorts of this width, so memory is bounded by ``concurrency`` no matter how many games/sims
        are pooled. ``max_new`` caps NEW games this call (batched / interrupt-friendly); ``None`` runs
        the whole holdout, flushing an intermediate report every ``report_every`` games.
        """
        from reporting.eval_report import resolve_results_run_dir
        from simulation.stage_eval import evaluate_stage

        self._require()
        if self.state["status"] != "trained":
            print("[eval] not trained yet — run:  python train.py --full --version "
                  f"{self.state.get('version', DEFAULT_VERSION)} --batch-size {self.state['batch_size']}")
            return

        version = version or self.state.get("version", DEFAULT_VERSION)
        n_sims = n_sims or STAGE_SIMS
        # concurrency = concurrent game-sims per GPU forward pass (VRAM-bound). games_per_batch just
        # pools enough games that cohorts stay full as sims desync; the actual GPU batch is capped at
        # `batch_size`, so VRAM is bounded by concurrency regardless of the total pooled.
        batch_size = concurrency or ROLLOUT_BATCH_SIZE
        games_per_batch = EVAL_GAMES_PER_BATCH
        holdout = self.state["holdout_game_ids"]
        run_dir = resolve_results_run_dir(version, name=name, holdout_total=len(holdout))
        self.state["last_eval_name"] = run_dir.name
        self._save()

        report = evaluate_stage(
            f"v{version}", holdout_ids=holdout, n_sims=n_sims, max_new=max_new,
            report_every=report_every, data_dir=self.state["data_dir"],
            processed_dir=self.state["processed_dir"], artifacts_root=version_root(version),
            results_run_dir=run_dir, batch_size=batch_size, games_per_batch=games_per_batch,
        )
        done, total = report["done"], report["total"]
        print("\n" + "=" * 70)
        if done >= total:
            print(f"DONE — all {total} holdout games predicted. Report:")
        else:
            print(f"STOP — {done}/{total} holdout games done. Re-run to continue:")
            print(f"  python evaluate.py --version {version} --name {run_dir.name}")
        print(f"  {report['run_dir']}")
        print("=" * 70)

    # --------------------------------------------------------------- retrain one head
    def retrain_model(self, name: str) -> None:
        """Retrain exactly ONE head in place, keeping the other heads' weights untouched.

        Backs ``train.py --model <name>``. Reuses the full pipeline (``run_stage``) with every OTHER
        head marked ``done`` so only ``name`` preprocesses + trains, overwriting
        ``artifacts/v<version>/<name>/``. Fresh init, same recency-weighted train slice as a full
        train. ``ModelBundle.load`` tolerates the untouched heads.
        """
        from models.pipeline import run_stage

        self._require()
        if name not in STAGE_MODEL_KEYS:
            raise SystemExit(f"unknown model '{name}'. Choose one of: {', '.join(STAGE_MODEL_KEYS)}")

        idx = game_index(self.state["data_dir"])
        partition = sequential_partition(idx, self.state["boundary_idx"],
                                         n_holdout=FINAL_HOLDOUT_GAMES, val_frac=TEST_FRAC, seed=SEED)
        subset_train = load_subset_games()
        print(f"[retrain] '{name}' only -> {self.state['artifacts_root']}/{name} "
              f"(other heads left in place)")

        sdict = self.state
        def on_trained(key: str) -> None:
            if key not in sdict.setdefault("trained_models", []):
                sdict["trained_models"].append(key)
            self._save()

        run_stage(
            self.state["data_dir"], partition, artifacts_root=self.state["artifacts_root"],
            warm_start=False, refit_norm_stats=True, epochs=self.state["epochs"],
            batch_size=self.state["batch_size"], report=True, run_name=self.state["run_name"],
            done=[k for k in STAGE_MODEL_KEYS if k != name], on_trained=on_trained,
            subset_keys=SUBSET_MODEL_KEYS, subset_train_games=subset_train,
        )
        print(f"[retrain] '{name}' done.")

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
                f"Run a full train first (it writes the cond_*.npz)."
            )
        cls = next(c for c in CONDITIONAL_MODEL_CLASSES if c.KEY == "shot_type")
        model = cls(Encoder(), path=self.state["data_dir"], processed_dir=self.state["processed_dir"])
        print(f"[retrain-shot-type] fresh train on existing {cond_train.name} "
              f"(live field goals only) -> {self.state['artifacts_root']}/shot_type")
        model.train(
            epochs=self.state["epochs"], batch_size=self.state["batch_size"],
            artifacts_root=self.state["artifacts_root"], report=True, run_name=self.state["run_name"],
            init_weights_root=None,
        )
        if "shot_type" not in self.state.setdefault("trained_models", []):
            self.state["trained_models"].append("shot_type")
            self._save()
        print("[retrain-shot-type] done — shot_type now trained on {2pt, 3pt} only.")

    # --------------------------------------------------------------- report / status
    def report(self, *, version: str | None = None, name: str | None = None) -> None:
        """Rebuild the aggregate report over an eval run's finished games (no new sims)."""
        from reporting.eval_report import resolve_results_run_dir
        from simulation.stage_eval import evaluate_stage

        self._require()
        version = version or self.state.get("version", DEFAULT_VERSION)
        name = name or self.state.get("last_eval_name")
        holdout = self.state["holdout_game_ids"]
        run_dir = resolve_results_run_dir(version, name=name, holdout_total=len(holdout))
        report = evaluate_stage(
            f"v{version}", holdout_ids=holdout, n_sims=STAGE_SIMS, max_new=0,
            data_dir=self.state["data_dir"], processed_dir=self.state["processed_dir"],
            artifacts_root=version_root(version), results_run_dir=run_dir,
        )
        print(f"[report] {report['done']}/{report['total']} games -> {report['run_dir']}")

    def status(self) -> None:
        self._require()
        print(f"Full run: version={self.state.get('version', '?')}, status={self.state['status']}, "
              f"models trained {len(self.state.get('trained_models', []))}/{len(STAGE_MODEL_KEYS)}, "
              f"holdout {len(self.state['holdout_game_ids'])} games, "
              f"weights {self.state['artifacts_root']}")


__all__ = ["FullRun"]
