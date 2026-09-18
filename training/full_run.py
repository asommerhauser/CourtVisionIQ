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
  extend     : re-cut the holdout to ``FINAL_HOLDOUT_GAMES`` WITHOUT re-running setup, so a
               finished train survives an eval-sizing change (train.py --extend-holdout).
  eval       : predict the next ``EVAL_BATCH`` holdout games + write a report (evaluate.py).
  report     : rebuild the aggregate report over everything finished.

``train.py --full --version X.Y --batch-size N`` runs setup+train; ``--continue`` re-runs train
(resumes at the next unfinished head); ``--model <name>`` runs retrain; ``--extend-holdout`` runs
extend.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config import (
    DEFAULT_MODEL, EVAL_BATCH, EVAL_GAMES_PER_BATCH, FINAL_HOLDOUT_GAMES, FINAL_SEASON_FRACTION,
    HOLDOUT_WINDOW_GAMES,
    FULL_RUN_STATE_PATH, ROLLOUT_BATCH_SIZE, ROLLOUT_EVAL_TAIL, SEED, STAGE_SIMS,
    SUBSET_GAMES_PATH,
    SUBSET_MODEL_KEYS, TEST_FRAC, VOCAB_DIR,
)
# model_name is re-exported: it lives in models.artifacts (TF-free, so eval_pool can reach
# it), but train.py and the tests have always imported it from here.
from models.artifacts import model_name, model_root  # noqa: F401
from data_loading import training_min_season
from models.manifest import (new_manifest, record_head, snapshot_vocabs, vocab_fingerprint,
                             write_manifest)
from models.registry import STAGE_MODEL_KEYS
from reporting.report_artifacts import DEFAULT_REPORTS_ROOT
from training.chronology import game_index, sequential_partition
from models.rollout_selection import record_selection
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

        # FINAL_HOLDOUT_GAMES is a TARGET for the pool, not a requirement. Since 3.0 it is 700 --
        # seven rotating 100-game windows -- against 702 games after the cut on the real corpus,
        # which is two games of slack. A smaller corpus (a test fixture, an experiment on one
        # season) should take the tail it has rather than refuse to set up at all; window_ids()
        # already warns when a window comes out short. Only an EMPTY tail is fatal.
        pool_size = min(FINAL_HOLDOUT_GAMES, len(idx) - boundary)
        if pool_size <= 0:
            raise SystemExit(
                f"no games sit after the cut (boundary {boundary}, corpus {len(idx)}); "
                f"there is nothing to hold out.")
        if pool_size < FINAL_HOLDOUT_GAMES:
            print(f"[setup] WARNING: only {pool_size} games after the cut, against a "
                  f"FINAL_HOLDOUT_GAMES pool of {FINAL_HOLDOUT_GAMES}. That is "
                  f"{pool_size // HOLDOUT_WINDOW_GAMES} full rotating window(s).")
        _, _, holdout = sequential_partition(idx, boundary, n_holdout=pool_size,
                                             val_frac=TEST_FRAC, seed=SEED)
        ordered = idx["game_id"].to_numpy()
        holdout_ids = [int(g) for g in ordered[boundary:boundary + pool_size]]

        self.state = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "version": name, "data_dir": data_dir, "processed_dir": processed_dir,
            "artifacts_root": artifacts_root, "reports_root": DEFAULT_REPORTS_ROOT,
            "epochs": epochs, "batch_size": batch_size, "run_name": run_name,
            "n_games": int(len(idx)), "boundary_idx": boundary,
            # The corpus this state describes. boundary_idx is a POSITION, so it moves when the
            # floor moves even though the games it points at do not -- recording the floor is what
            # makes a stale state file diagnosable instead of merely wrong.
            "min_train_season": training_min_season(),
            # W8: the last games before the cut, which rollout_selection.eval_game_ids samples from.
            # It read this key and NOTHING wrote it, so the selector's game set was unreachable from a
            # real run state and checkpoint selection would have scored nothing at all. Capped at
            # ROLLOUT_EVAL_TAIL so the state file stays small; they are trained-on games on purpose --
            # this measures whether a checkpoint BEHAVES, and generalisation is what the holdout is for.
            "train_tail_game_ids": [int(g) for g in ordered[:boundary][-ROLLOUT_EVAL_TAIL:]],
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

    # ----------------------------------------------------- extend holdout
    def extend_holdout(self, *, data_dir: str | None = None) -> None:
        """Re-cut the holdout window to ``FINAL_HOLDOUT_GAMES`` on an ALREADY-TRAINED model.

        :meth:`setup` is the only other consumer of that constant, and it also stamps
        ``status="setup"`` and ``trained_models=[]`` -- so raising the constant and re-running
        setup would throw away a finished train to change an eval knob. This does the one thing
        that is actually wanted, and nothing else.

        Three invariants, because the failure mode here is leaking training games into the
        holdout and never noticing:

        * the boundary is read from ``state["boundary_idx"]``, never recomputed, so the
          train/holdout cut cannot move -- the window only extends FORWARD from where this
          model actually stopped training;
        * the new list must START WITH the existing one, or this refuses. That keeps the first
          N ids the same N ids, so earlier runs stay directly comparable rather than being
          re-sliced under a total they never covered (their own ``results/<run>/holdout.json``
          pin holds them at their original count either way -- see ``pin_run_holdout``);
        * it refuses to SHRINK, since finished games in an existing run would fall outside the
          set they were scored against.

        ``status`` and ``trained_models`` are untouched, so the next call is ``evaluate.py``.
        """
        self._require()
        if "boundary_idx" not in self.state:
            raise SystemExit("state has no boundary_idx -- this predates the full-run cut; "
                             "re-run setup for a fresh model instead.")
        boundary = int(self.state["boundary_idx"])
        data_dir = data_dir or self.state.get("data_dir", "./data")
        idx = game_index(data_dir)
        ordered = idx["game_id"].to_numpy()
        available = len(ordered) - boundary

        if len(idx) != self.state.get("n_games", len(idx)):
            print(f"[extend] WARNING: corpus is {len(idx)} games, state recorded "
                  f"{self.state['n_games']} -- the data has been re-cleaned since the train. "
                  f"The prefix check below is what decides whether that matters.")
        if FINAL_HOLDOUT_GAMES > available:
            raise SystemExit(
                f"only {available} games sit after the train cut (boundary {boundary}, corpus "
                f"{len(ordered)}); FINAL_HOLDOUT_GAMES is {FINAL_HOLDOUT_GAMES}.")

        current = [int(g) for g in self.state.get("holdout_game_ids", [])]
        wanted = [int(g) for g in ordered[boundary:boundary + FINAL_HOLDOUT_GAMES]]
        if len(wanted) < len(current):
            raise SystemExit(
                f"refusing to shrink the holdout {len(current)} -> {len(wanted)}: games already "
                f"simulated in an existing run would fall outside the set they were scored "
                f"against. Raise FINAL_HOLDOUT_GAMES instead.")
        if wanted[:len(current)] != current:
            first = next(i for i, (a, b) in enumerate(zip(wanted, current)) if a != b)
            raise SystemExit(
                f"refusing to extend: the corpus no longer starts the holdout with the same "
                f"games (position {first}: {current[first]} recorded, {wanted[first]} now). "
                f"Earlier runs would stop being comparable. Re-clean and re-train, or evaluate "
                f"a new model name.")
        if wanted == current:
            print(f"[extend] holdout is already {len(current)} games "
                  f"(g{current[0]} .. g{current[-1]}); nothing to do.")
            return

        self.state["holdout_game_ids"] = wanted
        self._save()
        # Merge-only write: the manifest keeps its arch snapshot / seed / head records.
        write_manifest(self.state["artifacts_root"], holdout_game_ids=wanted)

        by_id = idx.set_index("game_id")
        first_g, last_g = by_id.loc[wanted[0]], by_id.loc[wanted[-1]]
        print(f"[extend] holdout {len(current)} -> {len(wanted)} games "
              f"(g{wanted[0]} .. g{wanted[-1]}, {first_g['game_date']} .. "
              f"{last_g['game_date']}); the first {len(current)} are unchanged.")
        print(f"[extend] train cut untouched at boundary {boundary}; "
              f"status stays '{self.state.get('status')}'.")
        print(f"State -> {self.state_path}")

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
        # 3.2: every head is on the subset, so there is no full-corpus group left to
        # name. The old line hardcoded the four head names and would have kept printing
        # them after the routing changed underneath it.
        print(f"\n{bar}\n[{tag}] subset heads {list(SUBSET_MODEL_KEYS)}\n"
              f"[{tag}]   -> {len(games)} games, every head\n{bar}")
        return games

    # ----------------------------------------------------------- rung 2
    def _rollout_score_factory(self):
        """``(factory, holder)`` for W4 rung 2, or ``(None, None)`` when selection is off.

        ``factory(head)`` returns the ``score_fn(epoch)`` the Keras callback calls. It is a factory
        because ``run_stage`` constructs the heads itself, so there is nothing for this to close over
        until the head exists -- and the score function must read the weights of *this* epoch, which
        live only on that instance.

        The simulator it rolls out with is loaded once, with every head from the finished bundle. That
        is why the handover runs rung 2 as a SECOND pass: mid-first-train the other eleven heads have no
        weights of their own, so a scored rollout would be scoring a bundle that does not exist.
        """
        import config as _config
        if not getattr(_config, "ROLLOUT_SELECTION", False):
            return None, None
        from models.rollout_bridge import build_rollout_score_fn

        root = self.state["artifacts_root"]
        data_dir = self.state["data_dir"]
        holder: dict = {}

        def make_sim():
            if "sim" not in holder:
                from simulation.game_simulator import GameSimulator
                holder["sim"] = GameSimulator.load(artifacts_root=root)
            return holder["sim"]

        def factory(head):
            holder["head"] = head
            return build_rollout_score_fn(
                self.state, make_sim=make_sim,
                live_model=lambda: getattr(head, "_live_model", None),
                data_dir=data_dir, seed=SEED)

        return factory, holder

    # --------------------------------------------------------------- train
    def train(self, *, rebuild_vocabs: bool = False) -> None:
        from models.pipeline import run_stage
        from models.prior_features import require_priors

        self._require()
        if self.state["status"] == "trained":
            print("[train] already trained — run:  python evaluate.py --model "
                  f"{self.state.get('version', DEFAULT_MODEL)}")
            return
        # W2.1's inputs come from a sidecar the cleaner does not build. merge_prior_features only
        # WARNS when it is missing, because a synthetic fixture and a weights-only machine both
        # legitimately have none -- but a train without it feeds every player the league mean, and
        # the first sign of that would be the eval, hours later. Refuse here instead.
        covered = require_priors(self.state["data_dir"])
        print(f"[train] priors sidecar covers {covered:,} games")
        idx = game_index(self.state["data_dir"])
        partition = sequential_partition(idx, self.state["boundary_idx"],
                                         n_holdout=FINAL_HOLDOUT_GAMES, val_frac=TEST_FRAC, seed=SEED)
        print(f"[train] one fresh full train on {self.state['boundary_idx']} games "
              f"(recency-weighted) -> {self.state['artifacts_root']}")

        # Every head (config.SUBSET_MODEL_KEYS is all twelve as of 3.2) trains on the compact,
        # modern-heavy per-season subset. Extracted here when absent, so there is no "no subset file"
        # branch left to fall into.
        subset_train = self._subset_games(tag="train")

        # W4's floor has the same shape of silent failure as the priors sidecar: configured but never
        # materialised, every player keeps his own embedding row and nothing looks wrong. Checked HERE,
        # after the extract, because the extract is what writes the map -- checking before it would
        # refuse every first-ever train on a fresh box, which is precisely the case that is fine.
        import config as _config
        from player_floor import require_player_floor
        n_aliased = require_player_floor(VOCAB_DIR, getattr(_config, "MIN_PLAYER_SUBSET_GAMES", None))
        if n_aliased:
            print(f"[train] vocabulary floor {_config.MIN_PLAYER_SUBSET_GAMES}: "
                  f"{n_aliased:,} players aliased to anonymous slots")

        # Rung 2's score function, and the holder the pipeline fills with the live head so the
        # function can read this epoch's weights rather than the bundle on disk.
        score_factory, holder = self._rollout_score_factory()

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
            rollout_score_fn_factory=score_factory,
        )

        # W8: the record rung 3's condition is a number rather than a judgement. It was written to
        # EventTimeModel._checkpoint_selection and never read by anything.
        head = holder.get("head") if holder else None
        selection = getattr(head, "_checkpoint_selection", None) if head is not None else None
        if selection is not None:
            record_selection(self.state_path, "event_time", selection)
            print(f"[train] checkpoint selection recorded: rollout-best epoch "
                  f"{selection.get('rollout_best_epoch')}, NLL-best "
                  f"{selection.get('nll_best_epoch')}"
                  f"{' -- THEY DISAGREE' if selection.get('epochs_disagree') else ''}")

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
    def window_ids(self, k: int = 0) -> list[int]:
        """The ``k``-th rotating holdout window out of the pool in state.

        The pool (``holdout_game_ids``) is every untrained game the model is allowed to be scored
        on; a window is the ~100 of them one run actually simulates. Windows are disjoint and
        contiguous, and window 0 is the first games after the train cut -- the ones every run
        before 3.0 used -- so ``k = 0`` reproduces history exactly.

        Later windows sit further from the cut (March games on January knowledge), which is why
        every run records its own k: drift with k is a finding, not noise (docs/v3_direction.md §4).
        """
        pool = [int(g) for g in self.state.get("holdout_game_ids", [])]
        width = HOLDOUT_WINDOW_GAMES
        if k < 0:
            raise SystemExit(f"window index must be >= 0, got {k}")
        start = k * width
        if start >= len(pool):
            raise SystemExit(
                f"window {k} starts at pool position {start} but the pool holds {len(pool)} "
                f"games. Widen it first:  python train.py --extend-holdout  "
                f"(FINAL_HOLDOUT_GAMES is {FINAL_HOLDOUT_GAMES}).")
        window = pool[start:start + width]
        if len(window) < width:
            print(f"[eval] WARNING: window {k} is short -- {len(window)} games, not {width}. "
                  f"Its numbers are not directly comparable with a full window.")
        return window

    def eval(self, *, version: str | None = None, name: str | None = None,
             n_sims: int | None = None, concurrency: int | None = None,
             max_new: int | None = None, report_every: int | None = None,
             shard: tuple[int, int] | None = None, seed: int = 0,
             subset: int | None = None, window: int = 0) -> None:
        """Predict the holdout into a results run at results/v<version>/<eval-name>/.

        ``version`` defaults to the trained run's version. ``name`` names the eval folder (default:
        auto-increment ``eval-NNN``, resuming the latest incomplete one). ``n_sims`` = Monte-Carlo
        sims per game (--monte-carlo). ``concurrency`` = concurrent game-sims per batched GPU forward
        pass — the **VRAM knob**: attention memory grows with batch × seq², so lower it if you OOM,
        raise it to use more of the card. It is decoupled from n_sims: the holdout's pooled sims run
        in cohorts of this width, so memory is bounded by ``concurrency`` no matter how many games/sims
        are pooled. ``max_new`` caps NEW games this call (batched / interrupt-friendly); ``None`` runs
        the whole holdout, flushing an intermediate report every ``report_every`` games.

        ``window`` (--window K) picks which ~100-game slice of the holdout pool this run scores.
        It applies FIRST, before ``subset`` and before the shard stride, so "window 2, 50 games,
        4 shards" means a 50-game subset OF window 2, split four ways -- never a subset of the
        whole pool that happens to land near window 2. The order is asserted in the tests.

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
        # Window first, then subset, then shard -- see the docstring.
        full_holdout = self.window_ids(window)
        # Resolve the run dir against the count this run will actually cover (an auto eval-NNN
        # decides "still incomplete?" from it), then pin the ids inside it.
        run_dir = resolve_results_run_dir(
            name_, name=name, holdout_total=len(subset_holdout(full_holdout, subset)))
        holdout = pin_run_holdout(run_dir, full_holdout, subset=subset, window=window)
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
            write_report=shard is None, window=window,
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
    def retrain_model(self, name: str, batch_size: int | None = None) -> None:
        """Retrain exactly ONE head in place, keeping the other heads' weights untouched.

        Backs ``train.py --model <name>``. Reuses the full pipeline (``run_stage``) with every OTHER
        head marked ``done`` so only ``name`` preprocesses + trains, overwriting
        ``artifacts/v<version>/<name>/``. Fresh init, same recency-weighted train slice as a full
        train. ``ModelBundle.load`` tolerates the untouched heads.

        ``batch_size`` overrides the state's for this head only, and is NOT written back: a
        one-head rescue on a smaller card must not silently re-scope the next ``--continue``.
        It used to be accepted on the command line and dropped on the floor here, so a
        ``--batch-size 24`` aimed at an OOM re-ran at the state's 64 and died the same way.
        """
        from models.pipeline import run_stage

        self._require()
        if name not in STAGE_MODEL_KEYS:
            raise SystemExit(f"unknown model '{name}'. Choose one of: {', '.join(STAGE_MODEL_KEYS)}")

        idx = game_index(self.state["data_dir"])
        partition = sequential_partition(idx, self.state["boundary_idx"],
                                         n_holdout=FINAL_HOLDOUT_GAMES, val_frac=TEST_FRAC, seed=SEED)
        subset_train = self._subset_games(tag="retrain")
        bs = batch_size or self.state["batch_size"]
        print(f"[retrain] '{name}' only -> {self.state['artifacts_root']}/{name} "
              f"(other heads left in place)")
        print(f"[retrain] batch {bs}"
              + (f" (override; state says {self.state['batch_size']})" if batch_size else ""))

        sdict = self.state
        def on_trained(key: str) -> None:
            if key not in sdict.setdefault("trained_models", []):
                sdict["trained_models"].append(key)
            self._save()

        run_stage(
            self.state["data_dir"], partition, artifacts_root=self.state["artifacts_root"],
            warm_start=False, refit_norm_stats=True, epochs=self.state["epochs"],
            batch_size=bs, report=True, run_name=self.state["run_name"],
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
