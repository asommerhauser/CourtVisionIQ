"""
curriculum.py — the staged, checkpointed, user-gated training driver.

This is the "full script": it clean-trains the whole corpus in contiguous, cumulative stages a
human can pause between. The transitions are deliberately manual — every step stops and waits for
the user to run the next command (nothing auto-advances), so the week-long run can be interrupted
at any milestone and the machine handed back.

State machine (persisted to ``curriculum_state.json``):

  init  --fresh : wipe all prior artifacts/data, re-clean + enrich every season, build the vocab +
                  normalization stats once over the full corpus (the warmup fit), auto-generate the
                  stop-point schedule, and print it for approval.
  train         : train the current stage (warm-started from the live previous-stage weights) on
                  its cumulative slice, then STOP. Resumable per model: if interrupted, re-running
                  picks up at the next unfinished model.
  eval          : predict the current stage's next-N holdout (11 sims/game), write the per-game
                  prediction folders + the stage report, then STOP and advance to the next stage.
  status        : show the schedule and per-stage progress.
  report        : once every stage is evaluated, build the cross-stage growth report.

Training itself is launched by the user on their GPU box — this module only sequences + checkpoints
it; it never decides to train on its own.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from config import (
    BOOTSTRAP_SEASONS, HOLDOUT_GAMES, NORM_STATS_PATH, SEED, STAGE_SIMS, TEST_FRAC, VOCAB_DIR,
)
from models.artifacts import DEFAULT_ARTIFACTS_ROOT
from models.conditional_type_model import CONDITIONAL_MODEL_CLASSES
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT
from training.chronology import build_schedule, format_schedule, game_index, sequential_partition

DEFAULT_STATE_PATH = "./training/curriculum_state.json"

# Every model trained per stage, in dependency order (matches pipeline.run_stage).
STAGE_MODEL_KEYS = [
    "event_time", "player", *[c.KEY for c in CONDITIONAL_MODEL_CLASSES],
    "substitution", "stint_length",
]


class Curriculum:
    """Load/advance the curriculum state machine; each public method is one CLI subcommand."""

    def __init__(self, state_path: str = DEFAULT_STATE_PATH):
        self.state_path = Path(state_path)
        self.state: dict = json.loads(self.state_path.read_text(encoding="utf-8")) \
            if self.state_path.exists() else {}

    # --------------------------------------------------------------- state io
    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _stage_entry(self, stage_no: int) -> dict:
        return self.state["schedule"][stage_no - 1]

    def _stage_status(self, stage_no: int) -> dict:
        return self.state.setdefault("stages", {}).setdefault(
            str(stage_no), {"status": "pending", "trained_models": []}
        )

    # --------------------------------------------------------------- init
    def init_fresh(self, *, data_dir: str = "./data", processed_dir: str = "./data/processed",
                   artifacts_root: str = DEFAULT_ARTIFACTS_ROOT,
                   reports_root: str = DEFAULT_REPORTS_ROOT,
                   epochs: int = 50, batch_size: int = 64) -> None:
        """Destructive reset → re-clean → warmup fit → schedule. Requires the explicit caller."""
        print("=" * 70)
        print("FULL CLEAN + RETRAIN — wiping prior cleaned data / artifacts / reports")
        print("=" * 70)
        self._wipe(data_dir, processed_dir, artifacts_root, reports_root)

        print("\n[init] cleaning raw master files -> ./data/season*.csv")
        from data_cleaner import DataCleaner
        DataCleaner().run()

        print("[init] enriching with season context (rest / games played)")
        from season_context import enrich
        enrich(data_dir)

        print("[init] warmup fit: building vocab + normalization stats over the full corpus")
        idx = game_index(data_dir)
        self._warmup_fit(idx, data_dir, processed_dir)

        print("[init] building the stop-point schedule")
        schedule = build_schedule(idx, bootstrap_seasons=BOOTSTRAP_SEASONS, n_holdout=HOLDOUT_GAMES)

        self.state = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "data_dir": data_dir, "processed_dir": processed_dir,
            "artifacts_root": artifacts_root, "reports_root": reports_root,
            "epochs": epochs, "batch_size": batch_size,
            "n_games": int(len(idx)), "current_stage": 1,
            "schedule": schedule, "stages": {},
        }
        self._save()

        print("\n" + format_schedule(idx, schedule))
        print(f"\n[init] {len(schedule)} stages over {len(idx)} games. State -> {self.state_path}")
        print("Review the schedule above, then run:  python train_full.py train")

    def _wipe(self, data_dir, processed_dir, artifacts_root, reports_root) -> None:
        for csv in Path(data_dir).glob("*.csv"):
            csv.unlink()
        for d in (processed_dir, artifacts_root, reports_root):
            shutil.rmtree(d, ignore_errors=True)
        for j in Path(VOCAB_DIR).glob("*.json"):
            j.unlink()
        Path(NORM_STATS_PATH).unlink(missing_ok=True)
        self.state_path.unlink(missing_ok=True)

    def _warmup_fit(self, idx, data_dir: str, processed_dir: str) -> None:
        """Build + freeze the vocab and persist every model's norm stats over the FULL corpus.

        Stats are fixed here once (warm-start needs stable standardization across stages), and the
        vocab is built over every game so future players get real embedding rows (never UNK). Uses
        an all-games train partition; the (full) tensors it writes are harmless and get overwritten
        by stage 1's slice.
        """
        from encoder.encoder import Encoder
        from models.event_time_model import EventTimeModel
        from models.player_model import PlayerModel
        from models.stint_length_model import StintLengthModel
        from models.substitution_model import SubstitutionModel

        all_games = {int(g) for g in idx["game_id"]}
        partition = (all_games, set(), set())
        common = dict(path=data_dir, processed_dir=processed_dir)

        def _fit(model, rebuild):
            model.preprocess(rebuild_vocabs=rebuild, game_partition=partition, refit_norm_stats=True)

        _fit(EventTimeModel(Encoder(), **common), rebuild=True)       # builds + freezes vocab
        _fit(PlayerModel(Encoder(), **common), rebuild=False)
        _fit(CONDITIONAL_MODEL_CLASSES[0](Encoder(), **common), rebuild=False)
        _fit(SubstitutionModel(Encoder(), **common), rebuild=False)
        _fit(StintLengthModel(Encoder(), **common), rebuild=False)

    # --------------------------------------------------------------- train
    def train_stage(self) -> None:
        from models.pipeline import run_stage

        self._require_init()
        stage_no = self.state["current_stage"]
        if stage_no > len(self.state["schedule"]):
            print("[train] all stages complete — run:  python train_full.py report")
            return
        entry = self._stage_entry(stage_no)
        sdict = self._stage_status(stage_no)
        if sdict["status"] in ("trained", "evaluated"):
            print(f"[train] stage {stage_no} already trained — run:  python train_full.py eval")
            return

        run_name = self._run_name(stage_no, entry)
        idx = game_index(self.state["data_dir"])
        partition = sequential_partition(idx, entry["boundary_idx"],
                                         n_holdout=HOLDOUT_GAMES, val_frac=TEST_FRAC, seed=SEED)
        print(f"\n[train] stage {stage_no}/{len(self.state['schedule'])} "
              f"({entry['boundary_type']}, season {entry['season']}): "
              f"{entry['train_games']} train games, warm-start from live weights")
        sdict["status"] = "training"
        self._save()

        def on_trained(key: str) -> None:
            if key not in sdict["trained_models"]:
                sdict["trained_models"].append(key)
            self._save()

        run_stage(
            self.state["data_dir"], partition, artifacts_root=self.state["artifacts_root"],
            epochs=self.state["epochs"], batch_size=self.state["batch_size"],
            report=True, run_name=run_name, done=sdict["trained_models"], on_trained=on_trained,
        )

        sdict["status"] = "trained"
        sdict["report_run_name"] = run_name
        self._save()
        print("\n" + "=" * 70)
        print(f"STOP — stage {stage_no} trained. Take a break if you like.")
        print(f"  When ready, predict its next {HOLDOUT_GAMES} games:  python train_full.py eval")
        print("=" * 70)

    # --------------------------------------------------------------- eval
    def eval_stage(self) -> None:
        from simulation.stage_eval import evaluate_stage

        self._require_init()
        stage_no = self.state["current_stage"]
        if stage_no > len(self.state["schedule"]):
            print("[eval] all stages complete — run:  python train_full.py report")
            return
        entry = self._stage_entry(stage_no)
        sdict = self._stage_status(stage_no)
        if sdict["status"] == "pending":
            print(f"[eval] stage {stage_no} not trained yet — run:  python train_full.py train")
            return
        if sdict["status"] == "evaluated":
            print(f"[eval] stage {stage_no} already evaluated.")
            return

        run_name = self._run_name(stage_no, entry)
        holdout = entry["holdout_game_ids"]
        if not holdout:                       # the final full-corpus stage has nothing to predict
            print(f"[eval] stage {stage_no} is the final full-corpus stage — no holdout to predict.")
        else:
            print(f"\n[eval] stage {stage_no}: predicting the next {len(holdout)} games "
                  f"({STAGE_SIMS} sims each)")
            report = evaluate_stage(
                run_name, holdout_ids=holdout, n_sims=STAGE_SIMS,
                data_dir=self.state["data_dir"], processed_dir=self.state["processed_dir"],
                artifacts_root=self.state["artifacts_root"], reports_root=self.state["reports_root"],
            )
            sdict["eval_run_dir"] = report.get("run_dir")
            sdict["eval_run_id"] = report.get("run_id")

        sdict["status"] = "evaluated"
        self.state["current_stage"] = stage_no + 1
        self._save()

        print("\n" + "=" * 70)
        if stage_no >= len(self.state["schedule"]):
            print("STOP — final stage evaluated. Build the growth report:")
            print("  python train_full.py report")
        else:
            print(f"STOP — stage {stage_no} evaluated. Take a break if you like.")
            print(f"  When ready, train the next stage:  python train_full.py train")
        print("=" * 70)

    # --------------------------------------------------------------- status
    def status(self) -> None:
        self._require_init()
        cur = self.state["current_stage"]
        print(f"Curriculum: {len(self.state['schedule'])} stages over "
              f"{self.state['n_games']} games  (current: stage {cur})")
        print(f"{'stage':>5}  {'type':<14}  {'season':>6}  {'train':>7}  {'status':<10}  models")
        for entry in self.state["schedule"]:
            n = entry["stage"]
            sdict = self.state.get("stages", {}).get(str(n), {"status": "pending", "trained_models": []})
            marker = "->" if n == cur else "  "
            print(f"{marker}{n:>3}  {entry['boundary_type']:<14}  {entry['season']:>6}  "
                  f"{entry['train_games']:>7}  {sdict['status']:<10}  "
                  f"{len(sdict.get('trained_models', []))}/{len(STAGE_MODEL_KEYS)}")

    # --------------------------------------------------------------- report
    def growth_report(self) -> None:
        from reporting.growth_report import build_growth_report

        self._require_init()
        run_dir = build_growth_report(self.state, reports_root=self.state["reports_root"])
        print(f"[report] growth report -> {run_dir}")

    # --------------------------------------------------------------- helpers
    def _require_init(self) -> None:
        if not self.state:
            raise SystemExit("No curriculum state. Run:  python train_full.py init --fresh")

    @staticmethod
    def _run_name(stage_no: int, entry: dict) -> str:
        return f"stage{stage_no:02d}_{entry['boundary_type'].replace(':', '')}_s{entry['season']}"


__all__ = ["Curriculum", "STAGE_MODEL_KEYS"]
