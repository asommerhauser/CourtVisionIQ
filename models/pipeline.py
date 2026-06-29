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
from models.conditional_time_model import ConditionalTimeModel
from models.conditional_type_model import CONDITIONAL_MODEL_CLASSES
from models.event_time_model import EventTimeModel
from models.model_bundle import ModelBundle
from models.player_model import PlayerModel
from models.stint_length_model import StintLengthModel
from models.substitution_model import SubstitutionModel


# Heavier heads that get a capped batch so the chain fits a tight (~10 GB) GPU alongside the shared
# roster encoder (~8 GB at batch 32). PlayerModel + SubstitutionModel emit logits over the large
# *player* vocab (+~0.5–1 GB of logits/gradients — the actual OOM cause); StintLengthModel has a
# scalar output but is the substitution model's sibling (two player-embedding conditioning inputs),
# capped here too as a precaution since it's the last head to train. Every other head (event / type /
# result / conditional-time — tiny outputs) trains at the full batch_size.
LARGE_OUTPUT_MODELS = {PlayerModel.KEY, SubstitutionModel.KEY, StintLengthModel.KEY}
# Lowered 24 -> 16 for the train-2 capacity bump (model_dim 384, player embed 192): the bigger
# backbone + player-vocab logits/gradients need more headroom on a tight (~10 GB) GPU. Raise back
# toward 24 if the first epoch shows VRAM to spare.
LARGE_OUTPUT_BATCH = 16


def _batch_for(key: str, batch_size: int) -> int:
    """Per-head batch size: cap the player-vocab heads so the chain fits on a tight GPU."""
    return min(batch_size, LARGE_OUTPUT_BATCH) if key in LARGE_OUTPUT_MODELS else batch_size


def _free_gpu() -> None:
    """Release the previous model's graph + VRAM before the next one builds.

    Every head trains sequentially in one process; without this, TF holds the prior model's
    allocations and the chain can OOM on a later (heavier) head. Best-effort: a failure here must
    never abort a multi-day run.
    """
    try:
        import gc
        import keras
        keras.backend.clear_session()
        gc.collect()
    except Exception:
        pass


def run_all(data_dir: str = "./data", artifacts_root: str = DEFAULT_ARTIFACTS_ROOT,
            holdout_frac: float = HOLDOUT_FRAC, epochs: int = 50, batch_size: int = 64,
            lr: float = 3e-4, patience: int = 15, dropout: float = 0.15, warmup_epochs: int = 2,
            report: bool = True, run_name: str | None = None,
            skip_preprocess: bool = False, train: bool = True) -> ModelBundle | None:
    """
    Preprocess (unless ``skip_preprocess``) and, if ``train``, train every model in the
    dependency order above. Returns the loaded ``ModelBundle`` when training ran, else None.

    Same per-model semantics as ``main.py`` for a single model — this just sequences them
    so the shared vocab/tensors are built once, in the right order, before each train.
    """
    def _train(model, key: str):
        _free_gpu()
        bs = _batch_for(key, batch_size)
        print(f"\n{'=' * 70}\n[pipeline] training '{key}' (batch {bs})\n{'=' * 70}")
        model.train(epochs=epochs, batch_size=bs, lr=lr, patience=patience,
                    dropout=dropout, warmup_epochs=warmup_epochs, artifacts_root=artifacts_root,
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

    # 2b) Conditional time head — own preprocess (condtime_*.npz, raw stream), then train.
    ct = ConditionalTimeModel(Encoder(), path=data_dir)
    if not skip_preprocess:
        print("[pipeline] preprocess 'event_time_cond' (condtime_*.npz)")
        ct.preprocess(rebuild_vocabs=False, holdout_frac=holdout_frac)
    if train:
        _train(ct, ConditionalTimeModel.KEY)

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
              lr: float = 3e-4, patience: int = 15, dropout: float = 0.15,
              warmup_epochs: int = 2,
              report: bool = True, run_name: str | None = None,
              done: list[str] | None = None, on_trained=None,
              subset_keys=None, subset_train_games=None) -> list[str]:
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

    ``subset_keys`` / ``subset_train_games`` (optional): the small heads listed in ``subset_keys``
    preprocess + train on ``subset_train_games`` instead of the full train pool — a compact,
    recency-weighted, coverage-complete slice (see ``training.subset``). Their val/holdout stay the
    full partition's, so early stopping + the reserved holdout are unchanged; only the train set
    shrinks. Heads not listed (player / substitution / stint_length) keep the full corpus.
    """
    warm_start_root = (warm_start_root or artifacts_root) if warm_start else None
    done = set(done or [])
    trained: list[str] = []

    subset_keys = set(subset_keys or ())
    _, _val, _holdout = game_partition
    subset_partition = (
        ({int(g) for g in subset_train_games}, set(_val), set(_holdout))
        if subset_keys and subset_train_games is not None else None
    )

    def _pp(key: str) -> dict:
        """Preprocess kwargs for one head: the subset partition for the small heads, else full."""
        part = subset_partition if (key in subset_keys and subset_partition is not None) else game_partition
        return dict(rebuild_vocabs=False, game_partition=part, refit_norm_stats=refit_norm_stats)

    def _train(model, key: str) -> None:
        if key in done:
            print(f"[stage] '{key}' already trained this call — skipping")
            return
        _free_gpu()
        bs = _batch_for(key, batch_size)
        origin = warm_start_root if warm_start_root else "fresh init"
        print(f"\n{'=' * 70}\n[stage] training '{key}' ({origin}, batch {bs})\n{'=' * 70}")
        model.train(epochs=epochs, batch_size=bs, lr=lr, patience=patience,
                    dropout=dropout, warmup_epochs=warmup_epochs, artifacts_root=artifacts_root,
                    report=report, run_name=run_name, init_weights_root=warm_start_root)
        trained.append(key)
        if on_trained is not None:
            on_trained(key)

    # 1) Event/Time, 2) Player — each its own preprocess + train.
    et = EventTimeModel(Encoder(), path=data_dir)
    if EventTimeModel.KEY not in done:
        et.preprocess(**_pp(EventTimeModel.KEY))
        _train(et, EventTimeModel.KEY)

    pl = PlayerModel(Encoder(), path=data_dir)
    if PlayerModel.KEY not in done:
        pl.preprocess(**_pp(PlayerModel.KEY))
        _train(pl, PlayerModel.KEY)

    # 2b) Conditional time head — own preprocess + train (raw stream; conditions on event + actor).
    ct = ConditionalTimeModel(Encoder(), path=data_dir)
    if ConditionalTimeModel.KEY not in done:
        ct.preprocess(**_pp(ConditionalTimeModel.KEY))
        _train(ct, ConditionalTimeModel.KEY)

    # 3) Conditional heads — one shared preprocess (only if a head still needs training), then each.
    # They share one cond_*.npz file, so they share one partition: all conditional heads are in
    # subset_keys together (see config.SUBSET_MODEL_KEYS) or none are.
    cond_keys = [cls.KEY for cls in CONDITIONAL_MODEL_CLASSES]
    if any(k not in done for k in cond_keys):
        CONDITIONAL_MODEL_CLASSES[0](Encoder(), path=data_dir).preprocess(**_pp(cond_keys[0]))
    for cls in CONDITIONAL_MODEL_CLASSES:
        _train(cls(Encoder(), path=data_dir), cls.KEY)

    # 4) Substitution, 5) Stint-length — self-contained preprocess + train.
    sub = SubstitutionModel(Encoder(), path=data_dir)
    if SubstitutionModel.KEY not in done:
        sub.preprocess(**_pp(SubstitutionModel.KEY))
        _train(sub, SubstitutionModel.KEY)

    stint = StintLengthModel(Encoder(), path=data_dir)
    if StintLengthModel.KEY not in done:
        stint.preprocess(**_pp(StintLengthModel.KEY))
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
