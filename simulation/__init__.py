"""
Simulation / generation layer.

Holds the trained model(s) and drives game *generation* — the self-feeding loop
that the training side (encoder + EventTimeModel + ModelBundle) was built to enable.
This package is the home for the rollout driver, and later the Controller and the
Monte-Carlo harness (see docs/technical_specs.md → Evaluation Strategy).
"""
# game_simulator imports TensorFlow, which on Windows must initialize before pandas
# (see main.py) — keep it first so importing the package never loads pandas ahead of TF.
from simulation.game_simulator import GameSimulator
from simulation.box_score import BoxScore, PlayerLine, box_score_for_game, generate_box_score
from simulation.game_input import (
    GameInput,
    extract_game_input,
    game_input_for_game,
    holdout_game_inputs,
    write_holdout_inputs,
)

__all__ = [
    "GameSimulator",
    "BoxScore",
    "PlayerLine",
    "generate_box_score",
    "box_score_for_game",
    "GameInput",
    "extract_game_input",
    "game_input_for_game",
    "holdout_game_inputs",
    "write_holdout_inputs",
]
