"""
Training and evaluation modules for trajectory models

Base Classes:
- BaseTrainer: Abstract base trainer with common training infrastructure

Concrete Trainers:
- SBTrainer: Trainer for Schrödinger Bridge models
- UnifiedTrainer: Trainer for OT and VAE models
- BatchOTTrainer: Trainer for Batch OT models

Evaluation:
- Evaluator: Evaluation metrics and comparison plots

Utilities:
- train_model: Train a single model from config
- run_experiment_from_config: Run complete experiment from config
"""

from .base_trainer import BaseTrainer
from .sb_trainer import SBTrainer
from .unified_trainer import UnifiedTrainer
from .batch_ot_trainer import BatchOTTrainer
from .sb_evaluator import Evaluator
from .utils import train_model, run_experiment_from_config

__all__ = [
    'BaseTrainer',
    'SBTrainer',
    'UnifiedTrainer',
    'BatchOTTrainer',
    'Evaluator',
    'train_model',
    'run_experiment_from_config'
]
