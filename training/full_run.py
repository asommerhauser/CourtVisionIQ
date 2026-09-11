"""
full_run.py — one recency-weighted full train + a batched 100-game holdout (the engine behind the
``train.py`` / ``evaluate.py`` CLIs).

Train every model **once** on the whole corpus (up to a cut partway through the most recent season),
with older seasons down-weighted (see ``season_features`` recency weighting), then predict the next
``FINAL_HOLDOUT_GAMES`` real games a batch at a time. Weights go to a versioned root
``artifacts/<name>/`` (see ``models.artifacts.model_root``) so each train keeps its own dir.

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
    DEFAULT_MODEL, EVAL_BATCH, EVAL_GAMES_PER_BATCH, FINAL_HOLDOUT_GAMES, FINAL_SEASON_FRACTION,
    FULL_RUN_STATE_PATH, ROLLOUT_BATCH_SIZE, SEED, STAGE_SIMS, SUBSET_GAMES_PATH,
    SUBSET_MODEL_KEYS, TEST_FRAC,
)
# model_name is re-exported: it lives in models.artifacts (TF-free, so eval_pool can reach
# it), but train.py and the tests have always imported it from here.
from models.artifacts import model_name, model_root  # noqa: F401
from models.manifest import (new_manifest, record_head, snapshot_vocabs, vocab_fingerprint,
                             write_manifest)
from models.registry import STAGE_MODEL_KEYS
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT
from training.chronology import game_index, sequential_partition
from training.subset import extract as extract_subset, load_subset_games

DEFAULT_STATE_PATH = FULL_RUN_STATE_PATH   # re-exported: train.py imports it from here


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
            raise SystemExit("No full-run state. Run:  python train.py --full --name <name> --batch-size N")

    # --------------------------------------------------------------- setup
    def setup(self, *, name: str | None = None, version: str | None = None,
              data_dir: str = "./data", processed_dir: str = "./data/processed",
              epochs: int = 50, batch_size: int = 64) -> None:
        """Compute the train/holdout cut from the already-cleaned data (no re-clean / re-warmup).

        ``name`` is the model name: it is both the weights dir (``artifacts/<name>/``) and the
        report label. Free-form -- ``"v1.0"``, ``"endgame-feats"``. ``version`` is a deprecated
        alias accepting a bare ``"1.0"``. Defaults to ``DEFAULT_MODEL``.
        """
        name = model_name(name or version or DEFAULT_MODEL)
        artifacts_root = model_root(name)
        run_name = name
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
            "version": name, "data_dir": data_dir, "processed_dir": processed_dir,
            "artifacts_root": artifacts_root, "reports_root": DEFAULT_REPORTS_ROOT,
            "epochs": epochs, "batch_size": batch_size, "run_name": run_name,
            "n_games": int(len(idx)), "boundary_idx": boundary,
            "holdout_game_ids": holdout_ids, "eval_batch": EVAL_BATCH,
            "status": "setup", "trained_models": [],
        }
        self._save()

        # Stub manifest up front so an interrupted train still leaves readable weights: it records
        # the architecture the graph will be rebuilt from, the seed, and the holdout.
        write_manifest(artifacts_root, **new_manifest(
            name, epochs=epochs, batch_size=batch_size, seed=SEED, data_dir=data_dir,
            processed_dir=processed_dir, n_games=int(len(idx)), boundary_idx=boundary,
            holdout_game_ids=holdout_ids))

        by_id = idx.set_index("game_id")
        first, last = by_id.loc[holdout_ids[0]], by_id.loc[holdout_ids[-1]]
        print(f"[setup] {name}: cut at {int(FINAL_SEASON_FRACTION * 100)}% "
              f"of season {last_season}'s regular schedule -> {boundary} train games.")
        print(f"[setup] holdout = {len(holdout_ids)} games (g{holdout_ids[0]} .. g{holdout_ids[-1]}, "
              f"{first['game_date']} .. {last['game_date']}), predicted {EVAL_BATCH} at a time.")
        print(f"[setup] full-train weights -> {artifacts_root}")
        print(f"State -> {self.state_path}\nNext:  python train.py --full --name {name} "
              f"--batch-size {batch_size}")

    # -------------------------------------------------------------- subset
    def _subset_games(self, *, tag: str) -> set[int]:
        """The small-head training subset, extracted on demand. Never returns None.

        A missing manifest used to mean "train every head on the full corpus", announced by a
        single line of stdout. Measured against full_train_2's reports that is 391-428 sec/epoch
        on 21,014 games where the subset heads ran 72-74 on 3,239 — 5.4x, silently, for about
        sixteen hours of a rented card. Absence cannot select a training regime any more.

        Extracting here is not a convenience, it is the only point in the flow where it fits:
        ``extract`` reads ``full_run_state.json``, which ``setup()`` writes and which
        ``train.py --full`` consumes in the same breath, so there has never been a window in
        which a separate ``python -m training.subset extract`` could have run.
        """
        games = load_subset_games()
        if games is None:
            print(f"[{tag}] no subset manifest at {SUBSET_GAMES_PATH} — extracting one now. "
                  f"It reads every cleaned season to map players to games; a few minutes.")
            extract_subset(state_path=str(self.state_path))
            games = load_subset_games()
            if games is None:
                raise SystemExit(
                    f"subset extract wrote no games to {SUBSET_GAMES_PATH}; the small heads "
                    f"would have no train pool."
                )
        bar = "=" * 70
        print(f"\n{bar}\n[{tag}] subset heads {list(SUBSET_MODEL_KEYS)}\n"
              f"[{tag}]   -> {len(games)} games\n"
              f"[{tag}] full corpus -> event_time, player, substitution, sub_decision\n{bar}")
        return games

    # --------------------------------------------------------------- train
    def train(self, *, rebuild_vocabs: bool = False) -> None:
        from models.pipeline import run_stage

        self._require()
        if self.state["status"] == "trained":
            print("[train] already trained — run:  python evaluate.py --model "
                  f"{self.state.get('version', DEFAULT_MODEL)}")
            return
        idx = game_index(self.state["data_dir"])
        partition = sequential_partition(idx, self.state["boundary_idx"],
                                         n_holdout=FINAL_HOLDOUT_GAMES, val_frac=TEST_FRAC, seed=SEED)
        print(f"[train] one fresh full train on {self.state['boundary_idx']} games "
              f"(recency-weighted) -> {self.state['artifacts_root']}")

        # Small heads (config.SUBSET_MODEL_KEYS) train on the compact, modern-heavy per-season
        # subset; the big player-vocab heads keep the full corpus. Extracted here when absent, so
        # there is no "no subset file" branch left to fall into.
        subset_train = self._subset_games(tag="train")

        self.state["status"] = "training"
        self._save()

        sdict = self.state
        root = self.state["artifacts_root"]

        def on_trained(key: str) -> None:
            if key not in sdict["trained_models"]:
                sdict["trained_models"].append(key)
            self._save()
            record_head(root, key)
            # event_time owns the vocab build, so once it is done the vocabs are final. Copying
            # them into the model dir pins the token ids these weights were trained against --
            # save_artifacts() writes the SHARED encoder/vocabs/, which the next train rewrites.
            if key == "event_time":
                from encoder.encoder import Encoder
                enc = Encoder()
                snapshot_vocabs(enc, root)
                write_manifest(root, vocabs=vocab_fingerprint(enc))

        run_stage(
            self.state["data_dir"], partition, artifacts_root=self.state["artifacts_root"],
            warm_start=False, refit_norm_stats=True, epochs=self.state["epochs"],
            batch_size=self.state["batch_size"], report=True, run_name=self.state["run_name"],
            done=sdict["trained_models"], on_trained=on_trained,
            subset_keys=SUBSET_MODEL_KEYS, subset_train_games=subset_train,
            rebuild_vocabs=rebuild_vocabs,
        )

        write_manifest(self.state["artifacts_root"],
                       finished_at=datetime.now().isoformat(timespec="seconds"))
        self.state["status"] = "trained"
        self._save()
        version = self.state.get("version", DEFAULT_MODEL)
        print("\n" + "=" * 70)
        print("STOP — full train done. Evaluate the holdout:")
        print(f"  python evaluate.py --model {version}")
        print("=" * 70)

    # --------------------------------------------------------------- eval
    def eval(self, *, version: str | None = None, name: str | None = None,
             n_sims: int | None = None, concurrency: int | None = None,
             max_new: int | None = None, report_every: int | None = None,
             shard: tuple[int, int] | None = None, seed: int = 0,
             subset: int | None = None) -> None:
        """Predict the holdout into a results run at results/v<version>/<eval-name>/.

        ``version`` defaults to the trained run's version. ``name`` names the eval folder (default:
        auto-increment ``eval-NNN``, resuming the latest incomplete one). ``n_sims`` = Monte-Carlo
        sims per game (--monte-carlo). ``concurrency`` = concurrent game-sims per batched GPU forward
        pass — the **VRAM knob**: attention memory grows with batch × seq², so lower it if you OOM,
        raise it to use more of the card. It is decoupled from n_sims: the holdout's pooled sims run
        in cohorts of this width, so memory is bounded by ``concurrency`` no matter how many games/sims
        are pooled. ``max_new`` caps NEW games this call (batched / interrupt-friendly); ``None`` runs
        the whole holdout, flushing an intermediate report every ``report_every`` games.

        ``subset`` (--holdout N) narrows the run to an N-game slice of the holdout, pinned to the
        run dir so every later call against it -- resume, shard, merge -- covers the same games.
        It applies BEFORE the shard stride, so the pool still splits exactly the run's own games.

        ``shard`` = ``(i, n)``, 1-based: simulate only ``holdout[i-1::n]``, so n concurrent
        processes can split the holdout across CPU cores while sharing one GPU. Within a process
        the rollout's worker threads are GIL-bound, so processes are what turn spare cores into
        throughput. The n slices are disjoint and cover the holdout exactly once, per-game seeds do
        not depend on position, and each game writes its own folder -- so a sharded run's games are
        identical to an unsharded run's. Sharded calls do NOT write the aggregate report (concurrent
        partial writes would clobber each other); merge with :meth:`report` once every shard
        finishes. They also leave the run state untouched, since concurrent writers would race it.
        """
        from reporting.eval_report import pin_run_holdout, resolve_results_run_dir, subset_holdout
        from simulation.stage_eval import evaluate_stage

        self._require()
        if self.state["status"] != "trained":
            print("[eval] not trained yet — run:  python train.py --full --name "
                  f"{self.state.get('version', DEFAULT_MODEL)} --batch-size {self.state['batch_size']}")
            return

        name_ = model_name(version or self.state.get("version", DEFAULT_MODEL))
        n_sims = n_sims or STAGE_SIMS
        # concurrency = concurrent game-sims per GPU forward pass (VRAM-bound). games_per_batch just
        # pools enough games that cohorts stay full as sims desync; the actual GPU batch is capped at
        # `batch_size`, so VRAM is bounded by concurrency regardless of the total pooled.
        batch_size = concurrency or ROLLOUT_BATCH_SIZE
        games_per_batch = EVAL_GAMES_PER_BATCH
        full_holdout = self.state["holdout_game_ids"]
        # Resolve the run dir against the count this run will actually cover (an auto eval-NNN
        # decides "still incomplete?" from it), then pin the ids inside it.
        run_dir = resolve_results_run_dir(
            name_, name=name, holdout_total=len(subset_holdout(full_holdout, subset)))
        holdout = pin_run_holdout(run_dir, full_holdout, subset=subset)
        run_total = len(holdout)
        if len(holdout) != len(full_holdout):
            print(f"[eval] holdout subset: {run_total} of {len(full_holdout)} games "
                  f"(g{holdout[0]} .. g{holdout[-1]}, pinned in {run_dir.name}/holdout.json)")
        if shard is None:
            self.state["last_eval_name"] = run_dir.name
            self._save()
        else:
            i, n = shard
            holdout = holdout[i - 1::n]
            print(f"[eval] shard {i}/{n}: {len(holdout)} of "
                  f"{run_total} holdout games -> {run_dir}")

        report = evaluate_stage(
            name_, holdout_ids=holdout, n_sims=n_sims, max_new=max_new, seed0=seed,
            report_every=report_every, data_dir=self.state["data_dir"],
            processed_dir=self.state["processed_dir"], artifacts_root=model_root(name_),
            results_run_dir=run_dir, batch_size=batch_size, games_per_batch=games_per_batch,
            write_report=shard is None,
        )
        done, total = report["done"], report["total"]
        print("\n" + "=" * 70)
        if shard is not None:
            i, n = shard
            if done >= total:
                print(f"SHARD {i}/{n} DONE — all {total} of its games predicted. Once EVERY "
                      f"shard is done, merge the report:")
                print(f"  python evaluate.py --model {name_} --run {run_dir.name} --report-only")
            else:
                print(f"SHARD {i}/{n} STOP — {done}/{total} of its games done. Re-run it:")
                print(f"  python evaluate.py --model {name_} --run {run_dir.name} --shard {i}/{n}")
        elif done >= total:
            print(f"DONE — all {total} holdout games predicted. Report:")
        else:
            print(f"STOP — {done}/{total} holdout games done. Re-run to continue:")
            print(f"  python evaluate.py --model {name_} --run {run_dir.name}")
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
        subset_train = self._subset_games(tag="retrain")
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
    def report(self, *, version: str | None = None, name: str | None = None,
               subset: int | None = None) -> None:
        """Rebuild the aggregate report over an eval run's finished games (no new sims).

        The sim count is read back off the finished records rather than assumed, and ``subset`` is
        normally unnecessary: a subset run pinned its ids in the run dir when it was created.
        """
        from reporting.eval_report import pin_run_holdout, resolve_results_run_dir, subset_holdout
        from simulation.stage_eval import evaluate_stage

        self._require()
        name_ = model_name(version or self.state.get("version", DEFAULT_MODEL))
        name = name or self.state.get("last_eval_name")
        full_holdout = self.state["holdout_game_ids"]
        run_dir = resolve_results_run_dir(
            name_, name=name, holdout_total=len(subset_holdout(full_holdout, subset)))
        holdout = pin_run_holdout(run_dir, full_holdout, subset=subset)
        report = evaluate_stage(
            name_, holdout_ids=holdout, max_new=0,
            data_dir=self.state["data_dir"], processed_dir=self.state["processed_dir"],
            artifacts_root=model_root(name_), results_run_dir=run_dir,
        )
        print(f"[report] {report['done']}/{report['total']} games -> {report['run_dir']}")

    def status(self) -> None:
        self._require()
        print(f"Full run: version={self.state.get('version', '?')}, status={self.state['status']}, "
              f"models trained {len(self.state.get('trained_models', []))}/{len(STAGE_MODEL_KEYS)}, "
              f"holdout {len(self.state['holdout_game_ids'])} games, "
              f"weights {self.state['artifacts_root']}")


__all__ = ["FullRun"]
