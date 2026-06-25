"""
Central registry of trainable models, keyed by their stable artifact KEY.

Adding a new model is a one-line entry here. Anything that needs to act over "all
models" (the ModelBundle manager, the parametrized persistence tests) iterates this
mapping rather than hard-coding model classes.

Each registered class must implement the persistence contract:
  - class attribute `KEY: str`
  - `save_artifacts(self, model, root=...) -> ModelArtifacts`
  - classmethod `from_artifacts(cls, root=..., encoder=None, **kwargs) -> (instance, model)`
"""
from __future__ import annotations

from models.conditional_type_model import CONDITIONAL_MODEL_CLASSES
from models.event_time_model import EventTimeModel
from models.player_model import PlayerModel
from models.stint_length_model import StintLengthModel
from models.substitution_model import SubstitutionModel

# key -> model wrapper class
MODEL_REGISTRY: dict[str, type] = {
    EventTimeModel.KEY: EventTimeModel,
    PlayerModel.KEY: PlayerModel,
    # Conditional type/result heads (shot_type, shot_result, assist_type,
    # turnover_type, foul_type) — each a spec-bound ConditionalTypeModel subclass.
    **{cls.KEY: cls for cls in CONDITIONAL_MODEL_CLASSES},
    # Substitution head: predicts the incoming player of a substitution.
    SubstitutionModel.KEY: SubstitutionModel,
    # Stint-length head: regresses how long an entering player stays on the floor.
    StintLengthModel.KEY: StintLengthModel,
}
