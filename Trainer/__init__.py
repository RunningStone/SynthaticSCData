"""
Training and evaluation modules for trajectory models
- SBTrainer: Trainer for Schrödinger Bridge models
- UnifiedTrainer: Trainer for OT and VAE models
- Evaluator: Evaluation metrics and comparison plots
"""

from .sb_trainer import SBTrainer
from .unified_trainer import UnifiedTrainer
from .sb_evaluator import Evaluator
from .utils import train_model, run_experiment_from_config

__all__ = [
    "SBTrainer",
    "UnifiedTrainer",
    "Evaluator",
    "train_model",
    "run_experiment_from_config",
]
