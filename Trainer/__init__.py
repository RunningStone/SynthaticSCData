"""
Training and evaluation modules
"""

from .trainer import ModelTrainer, Trainer
from .evaluator import ModelEvaluator
from .metrics import (
    compute_wasserstein_distance,
    compute_entropy,
    BoundaryFidelityMetric,
    PathFidelityMetric,
    EntropyEvolutionMetric,
    GeometricStructureMetric,
    GeneralizationMetric
)

__all__ = [
    "ModelTrainer",
    "Trainer",
    "ModelEvaluator",
    "compute_wasserstein_distance",
    "compute_entropy",
    "BoundaryFidelityMetric",
    "PathFidelityMetric",
    "EntropyEvolutionMetric",
    "GeometricStructureMetric",
    "GeneralizationMetric",
]
