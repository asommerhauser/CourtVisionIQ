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

    if train:
        print(f"\n[pipeline] all models trained -> loading bundle from {artifacts_root}")
        return load_all(artifacts_root=artifacts_root)
    return None


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
