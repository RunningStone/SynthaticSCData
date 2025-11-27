"""
Analyser module for visualizing model generation results and pre-experiment analysis

Modular Architecture:
- BaseVisualizer: Common infrastructure for all visualizers
- DataManager: Data loading and management
- ModelManager: Model loading and inference
- EmbeddingComputer: Dimensionality reduction (PHATE, LMNN+PCA)
- MetricsPlotter: Metrics visualization
- GenerationPlotter: Generation comparison visualization
- DataSplitAnalyzer: Data distribution analysis and sampling parameter calculation
- ModelParamAnalyzer: Model parameter counting and memory estimation
- AblationAnalyzer: Timepoint ablation marginal contribution analysis

Main entry points:
- step0_precalc_exps.py: Pre-experiment analysis
- step3_multi_setting_visualization.py: Result visualization
- step3_analyze_ablation.sh: Ablation study analysis
"""

# Modular components
from .base_visualizer import BaseVisualizer
from .data_manager import DataManager
from .model_manager import ModelManager
from .embedding_computer import EmbeddingComputer
from .metrics_plotter import MetricsPlotter
from .generation_plotter import GenerationPlotter
from .data_split_analyzer import DataSplitAnalyzer
from .model_param_analyzer import ModelParamAnalyzer
from .ablation_analyzer import AblationAnalyzer
from .interpolation_analyzer import InterpolationAnalyzer
from .entropy_analyzer import EntropyAnalyzer
from .figure_plotters import (
    plot_performance_radar,
    plot_phate_3x3_grid,
    plot_ablation_bar_chart,
    plot_ablation_heatmap,
    plot_entropy_marginal_scatter,
    plot_causal_ablation_comparison,
    plot_interpolation_ladder,
    extract_metrics_from_results,
    METRICS_CONFIG,
    SETTING_COLORS,
    MODEL_COLORS,
    TIMEPOINT_COLORS,
)

__all__ = [
    "BaseVisualizer",
    "DataManager",
    "ModelManager",
    "EmbeddingComputer",
    "MetricsPlotter",
    "GenerationPlotter",
    "DataSplitAnalyzer",
    "ModelParamAnalyzer",
    "AblationAnalyzer",
    "InterpolationAnalyzer",
    "EntropyAnalyzer",
    # Figure plotters
    "plot_performance_radar",
    "plot_phate_3x3_grid",
    "plot_ablation_bar_chart",
    "plot_ablation_heatmap",
    "plot_entropy_marginal_scatter",
    "plot_causal_ablation_comparison",
    "plot_interpolation_ladder",
    "extract_metrics_from_results",
    "METRICS_CONFIG",
    "SETTING_COLORS",
    "MODEL_COLORS",
    "TIMEPOINT_COLORS",
]
