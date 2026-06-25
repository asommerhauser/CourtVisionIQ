"""
End-to-end orchestration: preprocess + train every model in dependency order, then
load them all back.

The order is fixed by the shared vocab "language" and the autoregressive chain:

  1. ``event_time`` — rebuilds + freezes the shared vocab language and writes the base
     tensors + time-norm stats. MUST run first; everything downstream loads that frozen
     language.
  2. ``player`` — loads the frozen language; writes its own tensors.
  3. the conditional type/result heads — one shared preprocess (``cond_*.npz``) then train
     all five in turn: shot_type, shot_result, assist_type, turnover_type, foul_type.

Loading is the inverse and already generic: ``ModelBundle.load`` reloads every registered
model that has artifacts on disk, so ``load_all()`` here is a thin, friendly wrapper.

A fresh ``Encoder`` is built per model (mirroring how ``main.py`` runs each model in its
own process): the Event/Time rebuild persists the vocabs to disk, and each later model
loads + freezes them from disk — no shared mutable encoder state across stages.
"""
from __future__ import annotations

from config import HOLDOUT_FRAC
from encoder.encoder import Encoder
from models.artifacts import DEFAULT_ARTIFACTS_ROOT
from models.conditional_type_model import CONDITIONAL_MODEL_CLASSES
from models.event_time_model import EventTimeModel
from models.model_bundle import ModelBundle
from models.player_model import PlayerModel
from models.stint_length_model import StintLengthModel
from models.substitution_model import SubstitutionModel


def run_all(data_dir: str = "./data", artifacts_root: str = DEFAULT_ARTIFACTS_ROOT,
            holdout_frac: float = HOLDOUT_FRAC, epochs: int = 50, batch_size: int = 64,
            report: bool = True, run_name: str | None = None,
            skip_preprocess: bool = False, train: bool = True) -> ModelBundle | None:
    """
    Preprocess (unless ``skip_preprocess``) and, if ``train``, train every model in the
    dependency order above. Returns the loaded ``ModelBundle`` when training ran, else None.

    Same per-model semantics as ``main.py`` for a single model — this just sequences them
    so the shared vocab/tensors are built once, in the right order, before each train.
    """
    def _train(model, key: str):
        print(f"\n{'=' * 70}\n[pipeline] training '{key}'\n{'=' * 70}")
        model.train(epochs=epochs, batch_size=batch_size, artifacts_root=artifacts_root,
                    report=report, run_name=run_name)

    # 1) Event/Time — builds + freezes the shared vocab language (rebuild_vocabs=True).
    et = EventTimeModel(Encoder(), path=data_dir)
    if not skip_preprocess:
        print("[pipeline] preprocess 'event_time' (rebuild vocabs)")
        et.preprocess(rebuild_vocabs=True, holdout_frac=holdout_frac)
    if train:
        _train(et, EventTimeModel.KEY)

    # 2) Player — loads the now-frozen language.
    pl = PlayerModel(Encoder(), path=data_dir)
    if not skip_preprocess:
        print("[pipeline] preprocess 'player'")
        pl.preprocess(rebuild_vocabs=False, holdout_frac=holdout_frac)
    if train:
        _train(pl, PlayerModel.KEY)

    # 3) Conditional type/result heads — one shared preprocess, then train each.
    for i, cls in enumerate(CONDITIONAL_MODEL_CLASSES):
        model = cls(Encoder(), path=data_dir)
        if not skip_preprocess and i == 0:
            print("[pipeline] preprocess conditional heads (shared cond_*.npz)")
            model.preprocess(rebuild_vocabs=False, holdout_frac=holdout_frac)
        if train:
            _train(model, cls.KEY)

    # 4) Substitution head — self-contained preprocess (own sub_*.npz, with the
    # synthesized opening subs), then train.
    sub = SubstitutionModel(Encoder(), path=data_dir)
    if not skip_preprocess:
        print("[pipeline] preprocess 'substitution' (sub_*.npz w/ opening subs)")
        sub.preprocess(rebuild_vocabs=False, holdout_frac=holdout_frac)
    if train:
        _train(sub, SubstitutionModel.KEY)

    # 5) Stint-length head — self-contained preprocess (own stint_*.npz, with the
    # synthesized opening subs), then train. Drives rotation timing at inference.
    stint = StintLengthModel(Encoder(), path=data_dir)
    if not skip_preprocess:
        print("[pipeline] preprocess 'stint_length' (stint_*.npz w/ opening subs)")
        stint.preprocess(rebuild_vocabs=False, holdout_frac=holdout_frac)
    if train:
        _train(stint, StintLengthModel.KEY)

    if train:
        print(f"\n[pipeline] all models trained -> loading bundle from {artifacts_root}")
        return load_all(artifacts_root=artifacts_root)
    return None


def run_stage(data_dir: str, game_partition, *, artifacts_root: str = DEFAULT_ARTIFACTS_ROOT,
              warm_start_root: str | None = None, warm_start: bool = True,
              refit_norm_stats: bool = False, epochs: int = 50, batch_size: int = 32,
              report: bool = True, run_name: str | None = None,
              done: list[str] | None = None, on_trained=None) -> list[str]:
    """Train every model on ``game_partition`` (one stage / one full train) and return the keys
    trained this call.

    Mirrors ``run_all``'s dependency order. Two modes:
      * **Curriculum stage** (defaults): ``warm_start=True`` continues the previous stage's weights
        from ``warm_start_root`` (defaults to ``artifacts_root``), and ``refit_norm_stats=False``
        reuses the warmup-fit stats so standardization stays fixed across stages.
      * **Single full train**: ``warm_start=False`` trains fresh (``init_weights_root=None``) and
        ``refit_norm_stats=True`` recomputes stats on this train slice. Vocabs are never rebuilt
        here (the warmup fit froze them).

    ``done`` lists keys already trained this call (a resumed run): their train — and, where a whole
    group is done, their preprocess — is skipped, so an interruption picks up at the next unfinished
    model. ``on_trained(key)`` (when given) is called right after each model finishes so the caller
    can persist progress before the next (crash-resilient) model starts.
    """
    warm_start_root = (warm_start_root or artifacts_root) if warm_start else None
    done = set(done or [])
    trained: list[str] = []
    pp = dict(rebuild_vocabs=False, game_partition=game_partition, refit_norm_stats=refit_norm_stats)

    def _train(model, key: str) -> None:
        if key in done:
            print(f"[stage] '{key}' already trained this call — skipping")
            return
        origin = warm_start_root if warm_start_root else "fresh init"
        print(f"\n{'=' * 70}\n[stage] training '{key}' ({origin})\n{'=' * 70}")
        model.train(epochs=epochs, batch_size=batch_size, artifacts_root=artifacts_root,
                    report=report, run_name=run_name, init_weights_root=warm_start_root)
        trained.append(key)
        if on_trained is not None:
            on_trained(key)

    # 1) Event/Time, 2) Player — each its own preprocess + train.
    et = EventTimeModel(Encoder(), path=data_dir)
    if EventTimeModel.KEY not in done:
        et.preprocess(**pp)
        _train(et, EventTimeModel.KEY)

    pl = PlayerModel(Encoder(), path=data_dir)
    if PlayerModel.KEY not in done:
        pl.preprocess(**pp)
        _train(pl, PlayerModel.KEY)

    # 3) Conditional heads — one shared preprocess (only if a head still needs training), then each.
    cond_keys = [cls.KEY for cls in CONDITIONAL_MODEL_CLASSES]
    if any(k not in done for k in cond_keys):
        CONDITIONAL_MODEL_CLASSES[0](Encoder(), path=data_dir).preprocess(**pp)
    for cls in CONDITIONAL_MODEL_CLASSES:
        _train(cls(Encoder(), path=data_dir), cls.KEY)

    # 4) Substitution, 5) Stint-length — self-contained preprocess + train.
    sub = SubstitutionModel(Encoder(), path=data_dir)
    if SubstitutionModel.KEY not in done:
        sub.preprocess(**pp)
        _train(sub, SubstitutionModel.KEY)

    stint = StintLengthModel(Encoder(), path=data_dir)
    if StintLengthModel.KEY not in done:
        stint.preprocess(**pp)
        _train(stint, StintLengthModel.KEY)

    print(f"\n[stage] trained this call: {trained}")
    return trained


def load_all(artifacts_root: str = DEFAULT_ARTIFACTS_ROOT, encoder: Encoder | None = None,
             **kwargs) -> ModelBundle:
    """Load every trained model that has artifacts under ``artifacts_root``.

    Thin wrapper over ``ModelBundle.load`` (the generic, registry-driven loader): the
    Event/Time, Player, and all conditional heads come back keyed by their KEY. Extra
    kwargs (e.g. ``sequence_length=``, ``path=``) flow through to each ``from_artifacts``.
    """
    bundle = ModelBundle.load(root=artifacts_root, encoder=encoder, **kwargs)
    print(f"[pipeline] loaded models: {sorted(bundle.keys())}")
    return bundle
