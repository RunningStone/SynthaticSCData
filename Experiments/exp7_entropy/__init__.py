"""
Experiment 7: Entropy Evolution Analysis

Tests whether models can reproduce the non-monotonic entropy evolution
(entropy increase → entropy decrease) observed in real EMT trajectories.

This is the most direct validation of the core hypothesis:
boundary conditions alone are insufficient to constrain non-monotonic dynamics.

Modules:
    - entropy_estimators: KNN and Gaussian entropy estimation
    - analyze_entropy_evolution: Entropy curve computation and analysis
    - run_entropy_analysis: Main analysis script
"""

from .entropy_estimators import (
    estimate_entropy_knn,
    estimate_entropy_gaussian,
    estimate_entropy_both_methods,
    batch_estimate_entropy
)

from .analyze_entropy_evolution import (
    compute_entropy_curve,
    compute_entropy_curve_from_real_data,
    analyze_entropy_peak,
    compute_entropy_curve_similarity,
    compare_multiple_models
)

__all__ = [
    # Entropy estimators
    'estimate_entropy_knn',
    'estimate_entropy_gaussian',
    'estimate_entropy_both_methods',
    'batch_estimate_entropy',
    # Entropy curve analysis
    'compute_entropy_curve',
    'compute_entropy_curve_from_real_data',
    'analyze_entropy_peak',
    'compute_entropy_curve_similarity',
    'compare_multiple_models'
]
