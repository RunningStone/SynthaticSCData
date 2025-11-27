#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure Plotters - Specialized Visualization Functions for Publication Figures

This module provides reusable plotting functions for:
- Figure 1: Performance radar charts and PHATE 3x3 grids
- Figure 2: Ablation bar charts, heatmaps, and entropy scatter plots
- Figure 3: Causal ablation and linear interpolation ladder charts

Author: Shi Pan
Date: 2024-11-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Constants and Color Schemes
# =============================================================================

# Metrics configuration
METRICS_CONFIG = {
    'frechet_distance': {'display': 'FD', 'lower_better': True},
    'mae': {'display': 'MAE', 'lower_better': True},
    'pcc': {'display': 'PCC', 'lower_better': False},
    'wasserstein_distance': {'display': 'W-Dist', 'lower_better': True},
    'mmd': {'display': 'MMD', 'lower_better': True},
    'js_divergence': {'display': 'JS-Div', 'lower_better': True},
}

# Color schemes
SETTING_COLORS = {
    'Setting1': '#3498db',  # Blue (cold)
    'Setting2': '#e74c3c',  # Red (warm)
    'Setting3': '#2ecc71',  # Green
    'Setting4': '#9b59b6',  # Purple
    'Setting5': '#f39c12',  # Orange
    'Setting6': '#1abc9c',  # Teal
}

MODEL_COLORS = {
    'ot': '#1f77b4',
    'vae': '#ff7f0e',
    'sb': '#2ca02c',
    'batch_ot': '#d62728',
    'sb_mlplus': '#9467bd',
}

# Time point colors for PHATE plots
TIMEPOINT_COLORS = {
    '0d': '#1f77b4',
    '8h': '#ff7f0e',
    '1d': '#2ca02c',
    '3d': '#d62728',
    '7d': '#9467bd',
    '8h_rm': '#8c564b',
    '1d_rm': '#e377c2',
    '3d_rm': '#7f7f7f',
}


# =============================================================================
# Figure 1: Radar Chart and PHATE Grid
# =============================================================================

def plot_performance_radar(
    metrics_setting1: Dict[str, Dict[str, float]],
    metrics_setting2: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Performance Comparison: Setting1 vs Setting2",
    metrics_keys: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 300
) -> Path:
    """
    Create a radar chart comparing metrics between Setting1 and Setting2.
    
    Args:
        metrics_setting1: Dict mapping model_name -> metrics dict for Setting1
        metrics_setting2: Dict mapping model_name -> metrics dict for Setting2
        output_path: Path to save the figure
        title: Figure title
        metrics_keys: List of metric keys to plot (default: all in METRICS_CONFIG)
        figsize: Figure size
        dpi: DPI for saving
    
    Returns:
        Path to saved figure
    """
    if metrics_keys is None:
        metrics_keys = list(METRICS_CONFIG.keys())
    
    # Prepare data
    labels = [METRICS_CONFIG[k]['display'] for k in metrics_keys]
    n_metrics = len(metrics_keys)
    
    # Compute angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Cold colors for Setting1, warm colors for Setting2
    cold_colors = ['#3498db', '#2980b9', '#1abc9c']  # Blues/teals
    warm_colors = ['#e74c3c', '#e67e22', '#f39c12']  # Reds/oranges
    
    # Normalize metrics for radar chart (0-1 scale)
    all_values = {}
    for metric in metrics_keys:
        all_vals = []
        for model_metrics in list(metrics_setting1.values()) + list(metrics_setting2.values()):
            if metric in model_metrics:
                all_vals.append(model_metrics[metric])
        if all_vals:
            all_values[metric] = (min(all_vals), max(all_vals))
    
    def normalize_metric(value, metric):
        """Normalize metric to 0-1 scale, with 1 being better."""
        if metric not in all_values:
            return 0.5
        min_val, max_val = all_values[metric]
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        # Invert if lower is better
        if METRICS_CONFIG[metric]['lower_better']:
            normalized = 1 - normalized
        return normalized
    
    # Plot Setting1 models (cold colors)
    for idx, (model_name, metrics) in enumerate(metrics_setting1.items()):
        values = [normalize_metric(metrics.get(k, 0), k) for k in metrics_keys]
        values += values[:1]  # Close polygon
        
        color = cold_colors[idx % len(cold_colors)]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'S1-{model_name}', color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # Plot Setting2 models (warm colors)
    for idx, (model_name, metrics) in enumerate(metrics_setting2.items()):
        values = [normalize_metric(metrics.get(k, 0), k) for k in metrics_keys]
        values += values[:1]  # Close polygon
        
        color = warm_colors[idx % len(warm_colors)]
        ax.plot(angles, values, 's--', linewidth=2, label=f'S2-{model_name}', color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # Configure axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_phate_3x3_grid(
    phate_coords: np.ndarray,
    real_labels: np.ndarray,
    time_labels: List[str],
    generated_coords_setting1: Dict[str, np.ndarray],
    generated_coords_setting2: Dict[str, np.ndarray],
    start_timepoint: str,
    end_timepoint: str,
    output_path: Path,
    figsize: Tuple[int, int] = (15, 15),
    dpi: int = 300
) -> Path:
    """
    Create a 3x3 PHATE embedding grid.
    
    Row 1: Real data views (all categories, start/end highlighted, intermediate highlighted)
    Row 2: Setting1 models (OT, VAE, SB) - generated in red, real in gray
    Row 3: Setting2 models (Batch_OT, VAE, SB_MLPlus) - generated in red, real in gray
    
    Args:
        phate_coords: PHATE coordinates for all data (real + generated combined)
        real_labels: Labels for real data points
        time_labels: List of time label strings
        generated_coords_setting1: Dict mapping model_name -> generated PHATE coords
        generated_coords_setting2: Dict mapping model_name -> generated PHATE coords
        start_timepoint: Start timepoint label (e.g., '0d')
        end_timepoint: End timepoint label (e.g., '7d')
        output_path: Path to save the figure
        figsize: Figure size
        dpi: DPI for saving
    
    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    
    # Get indices for start and end timepoints
    start_idx = time_labels.index(start_timepoint) if start_timepoint in time_labels else 0
    end_idx = time_labels.index(end_timepoint) if end_timepoint in time_labels else len(time_labels) - 1
    
    # Number of real samples
    n_real = len(real_labels)
    real_coords = phate_coords[:n_real]
    
    # Define colors for each timepoint
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(time_labels)))
    tp_colors = {tp: colors[i] for i, tp in enumerate(time_labels)}
    
    # =========================================================================
    # Row 1: Real data views
    # =========================================================================
    
    # Row 1, Col 1: All categories with different colors
    ax = axes[0, 0]
    for time_idx, time_label in enumerate(time_labels):
        mask = (real_labels == time_idx)
        ax.scatter(real_coords[mask, 0], real_coords[mask, 1],
                  c=[tp_colors[time_label]], label=time_label,
                  alpha=0.6, s=15, edgecolors='none')
    ax.set_title('Real Data: All Categories', fontweight='bold', fontsize=11)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.5)
    ax.set_xlabel('PHATE 1', fontsize=10)
    ax.set_ylabel('PHATE 2', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Row 1, Col 2: Start/end highlighted, others gray
    ax = axes[0, 1]
    for time_idx, time_label in enumerate(time_labels):
        mask = (real_labels == time_idx)
        if time_label in [start_timepoint, end_timepoint]:
            ax.scatter(real_coords[mask, 0], real_coords[mask, 1],
                      c=[tp_colors[time_label]], label=time_label,
                      alpha=0.7, s=20, edgecolors='none')
        else:
            ax.scatter(real_coords[mask, 0], real_coords[mask, 1],
                      c='lightgray', alpha=0.3, s=10, edgecolors='none')
    ax.set_title(f'Highlighted: {start_timepoint} & {end_timepoint}', fontweight='bold', fontsize=11)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.5)
    ax.set_xlabel('PHATE 1', fontsize=10)
    ax.set_ylabel('PHATE 2', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Row 1, Col 3: Intermediate highlighted, start/end gray
    ax = axes[0, 2]
    for time_idx, time_label in enumerate(time_labels):
        mask = (real_labels == time_idx)
        if time_label not in [start_timepoint, end_timepoint]:
            ax.scatter(real_coords[mask, 0], real_coords[mask, 1],
                      c=[tp_colors[time_label]], label=time_label,
                      alpha=0.7, s=20, edgecolors='none')
        else:
            ax.scatter(real_coords[mask, 0], real_coords[mask, 1],
                      c='lightgray', alpha=0.3, s=10, edgecolors='none')
    ax.set_title('Highlighted: Intermediate', fontweight='bold', fontsize=11)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.5)
    ax.set_xlabel('PHATE 1', fontsize=10)
    ax.set_ylabel('PHATE 2', fontsize=10)
    ax.grid(alpha=0.3)
    
    # =========================================================================
    # Row 2: Setting1 models
    # =========================================================================
    setting1_models = ['ot', 'vae', 'sb']
    for col_idx, model_name in enumerate(setting1_models):
        ax = axes[1, col_idx]
        
        # Plot real data in gray
        ax.scatter(real_coords[:, 0], real_coords[:, 1],
                  c='lightgray', alpha=0.3, s=10, edgecolors='none', label='Real')
        
        # Plot generated data in red
        if model_name in generated_coords_setting1:
            gen_coords = generated_coords_setting1[model_name]
            ax.scatter(gen_coords[:, 0], gen_coords[:, 1],
                      c='#e74c3c', alpha=0.7, s=25, edgecolors='darkred',
                      linewidths=0.5, label=f'Generated', marker='*')
        
        ax.set_title(f'Setting1: {model_name.upper()}', fontweight='bold', fontsize=11)
        ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.5)
        ax.set_xlabel('PHATE 1', fontsize=10)
        ax.set_ylabel('PHATE 2', fontsize=10)
        ax.grid(alpha=0.3)
    
    # =========================================================================
    # Row 3: Setting2 models
    # =========================================================================
    setting2_models = ['batch_ot', 'vae', 'sb_mlplus']
    for col_idx, model_name in enumerate(setting2_models):
        ax = axes[2, col_idx]
        
        # Plot real data in gray
        ax.scatter(real_coords[:, 0], real_coords[:, 1],
                  c='lightgray', alpha=0.3, s=10, edgecolors='none', label='Real')
        
        # Plot generated data in red
        if model_name in generated_coords_setting2:
            gen_coords = generated_coords_setting2[model_name]
            ax.scatter(gen_coords[:, 0], gen_coords[:, 1],
                      c='#e74c3c', alpha=0.7, s=25, edgecolors='darkred',
                      linewidths=0.5, label=f'Generated', marker='*')
        
        ax.set_title(f'Setting2: {model_name.upper()}', fontweight='bold', fontsize=11)
        ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.5)
        ax.set_xlabel('PHATE 1', fontsize=10)
        ax.set_ylabel('PHATE 2', fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Generation Quality Visualization (PHATE)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


# =============================================================================
# Figure 2: Ablation Analysis
# =============================================================================

def _compute_normalized_change(
    full_val: float,
    variant_val: float,
    lower_better: bool,
    max_change: float
) -> float:
    """
    Compute normalized change relative to full model.
    
    Maps changes to [0.5, 1.5] range where:
    - 1.0 = same as full model
    - Values < 1.0 = worse performance
    - Values > 1.0 = better performance
    
    Args:
        full_val: Value from full model (baseline)
        variant_val: Value from variant
        lower_better: Whether lower values are better for this metric
        max_change: Maximum absolute change across all variants (for normalization)
    
    Returns:
        Normalized value in [0.5, 1.5] range
    """
    if abs(full_val) < 1e-10 or abs(max_change) < 1e-10:
        return 1.0
    
    # Compute raw change
    raw_change = variant_val - full_val
    
    # For lower-is-better metrics, positive change (increase) is bad
    # For higher-is-better metrics, negative change (decrease) is bad
    if lower_better:
        # Positive change = worse, so negate to make positive = better
        direction = -raw_change
    else:
        # Positive change = better
        direction = raw_change
    
    # Normalize to [-0.5, 0.5] range based on max_change
    normalized = (direction / max_change) * 0.5
    
    # Shift to [0.5, 1.5] range (1.0 = baseline)
    return 1.0 + normalized


def plot_ablation_bar_chart(
    metrics_full: Dict[str, float],
    metrics_ablations: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Timepoint Ablation: Relative Performance Change",
    metrics_keys: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 300
) -> Path:
    """
    Create a grouped bar chart showing ablation performance with normalized changes.
    
    Layout: Subplots grouped by metrics, with Full model (baseline=1.0) and 
    ablation variants showing relative change in [0.5, 1.5] range.
    
    Args:
        metrics_full: Metrics from full model (Setting2) - used as baseline (1.0)
        metrics_ablations: Dict mapping removed_timepoint -> metrics dict
        output_path: Path to save the figure
        title: Figure title
        metrics_keys: List of metric keys to plot
        figsize: Figure size
        dpi: DPI for saving
    
    Returns:
        Path to saved figure
    """
    if metrics_keys is None:
        metrics_keys = list(METRICS_CONFIG.keys())
    
    ablation_names = list(metrics_ablations.keys())
    n_metrics = len(metrics_keys)
    n_variants = len(ablation_names) + 1  # +1 for Full model
    
    # First pass: compute max absolute change for each metric (for normalization)
    max_changes = {}
    for metric_key in metrics_keys:
        full_val = metrics_full.get(metric_key, 0)
        max_change = 0
        for abl_name in ablation_names:
            abl_val = metrics_ablations[abl_name].get(metric_key, full_val)
            change = abs(abl_val - full_val)
            max_change = max(max_change, change)
        max_changes[metric_key] = max_change if max_change > 0 else 1.0
    
    # Create subplots: one per metric
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    # Colors: Full model + ablation variants
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']  # Green for Full, others for ablations
    
    for metric_idx, metric_key in enumerate(metrics_keys):
        ax = axes[metric_idx]
        
        x = np.arange(n_variants)
        width = 0.6
        
        full_val = metrics_full.get(metric_key, 0)
        lower_better = METRICS_CONFIG[metric_key]['lower_better']
        max_change = max_changes[metric_key]
        
        # Compute normalized values: Full = 1.0, variants mapped to [0.5, 1.5]
        normalized_values = [1.0]  # Full model is always 1.0
        labels = ['Full']
        
        for abl_name in ablation_names:
            abl_val = metrics_ablations[abl_name].get(metric_key, full_val)
            norm_val = _compute_normalized_change(full_val, abl_val, lower_better, max_change)
            normalized_values.append(norm_val)
            labels.append(f'-{abl_name}')
        
        bars = ax.bar(x, normalized_values, width, 
                     color=[colors[i % len(colors)] for i in range(n_variants)],
                     alpha=0.8, edgecolor='black')
        
        # Add value labels on top of bars
        for bar, norm_val in zip(bars, normalized_values):
            height = bar.get_height()
            ax.annotate(f'{norm_val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add baseline reference line at 1.0
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Variant', fontsize=10, fontweight='bold')
        ax.set_title(METRICS_CONFIG[metric_key]['display'], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0.4, 1.6)  # Fixed range to show [0.5, 1.5] clearly
        ax.set_ylabel('Relative Score', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(title + '\n(1.0 = Full Model, >1.0 = Better, <1.0 = Worse)', 
                fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_ablation_heatmap(
    metrics_full: Dict[str, float],
    metrics_ablations: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str = "Ablation Sensitivity Heatmap",
    metrics_keys: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300
) -> Path:
    """
    Create a heatmap showing normalized relative performance change.
    
    Values are normalized to [0.5, 1.5] range where:
    - 1.0 = same as full model (baseline)
    - >1.0 = better performance
    - <1.0 = worse performance
    
    Args:
        metrics_full: Metrics from full model
        metrics_ablations: Dict mapping removed_timepoint -> metrics dict
        output_path: Path to save the figure
        title: Figure title
        metrics_keys: List of metric keys to plot
        figsize: Figure size
        dpi: DPI for saving
    
    Returns:
        Path to saved figure
    """
    if metrics_keys is None:
        metrics_keys = list(METRICS_CONFIG.keys())
    
    ablation_names = list(metrics_ablations.keys())
    
    # First pass: compute max absolute change for each metric
    max_changes = {}
    for metric in metrics_keys:
        full_val = metrics_full.get(metric, 0)
        max_change = 0
        for abl_name in ablation_names:
            abl_val = metrics_ablations[abl_name].get(metric, full_val)
            change = abs(abl_val - full_val)
            max_change = max(max_change, change)
        max_changes[metric] = max_change if max_change > 0 else 1.0
    
    # Compute normalized relative performance
    data = []
    for metric in metrics_keys:
        row = []
        full_val = metrics_full.get(metric, 0)
        lower_better = METRICS_CONFIG[metric]['lower_better']
        max_change = max_changes[metric]
        
        for abl_name in ablation_names:
            abl_val = metrics_ablations[abl_name].get(metric, full_val)
            norm_val = _compute_normalized_change(full_val, abl_val, lower_better, max_change)
            row.append(norm_val)
        data.append(row)
    
    df = pd.DataFrame(data, 
                     index=[METRICS_CONFIG[k]['display'] for k in metrics_keys],
                     columns=[f'Remove {abl}' for abl in ablation_names])
    
    # Create heatmap with fixed range [0.5, 1.5]
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use diverging colormap centered at 1.0
    # Green = better (>1.0), Red = worse (<1.0)
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0,
               vmin=0.5, vmax=1.5,
               cbar_kws={'label': 'Relative Score (1.0 = Full Model)'},
               ax=ax, linewidths=0.5)
    
    ax.set_xlabel('Ablation Variant', fontsize=12, fontweight='bold')
    ax.set_ylabel('Evaluation Metric', fontsize=12, fontweight='bold')
    ax.set_title(title + '\n(>1.0 = Better, <1.0 = Worse)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_entropy_marginal_scatter(
    entropy_values: Dict[str, float],
    marginal_contributions: Dict[str, float],
    output_path: Path,
    title: str = "Entropy vs Marginal Contribution",
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 300
) -> Path:
    """
    Create a scatter plot of entropy vs marginal contribution for each timepoint.
    
    Args:
        entropy_values: Dict mapping timepoint -> entropy value
        marginal_contributions: Dict mapping timepoint -> marginal contribution
        output_path: Path to save the figure
        title: Figure title
        figsize: Figure size
        dpi: DPI for saving
    
    Returns:
        Path to saved figure
    """
    timepoints = list(entropy_values.keys())
    x = [entropy_values[tp] for tp in timepoints]
    y = [marginal_contributions[tp] for tp in timepoints]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(timepoints)))
    for i, tp in enumerate(timepoints):
        ax.scatter(x[i], y[i], c=[colors[i]], s=150, edgecolors='black',
                  linewidths=1.5, label=tp, zorder=5)
        ax.annotate(tp, (x[i], y[i]), xytext=(5, 5), textcoords='offset points',
                   fontsize=11, fontweight='bold')
    
    # Fit and plot regression line
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2, label='Linear fit')
        
        # Compute correlation
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}',
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Marginal Contribution (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


# =============================================================================
# Figure 3: Causal and Interpolation Analysis
# =============================================================================

def plot_causal_ablation_comparison(
    metrics_setting2: Dict[str, Dict[str, float]],
    metrics_setting5: Dict[str, Dict[str, float]],
    output_path: Path,
    models_to_compare: List[str] = ['vae', 'sb_mlplus', 'batch_ot'],
    title: str = "Sequential vs Shuffled Intermediate States",
    metrics_keys: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 300
) -> Path:
    """
    Create a grouped bar chart comparing Setting2 (sequential) vs Setting5 (shuffled).
    
    Layout: Subplots grouped by metrics, with multiple models per subplot.
    Each subplot shows one metric, with bars for each model comparing S2 vs S5.
    
    Args:
        metrics_setting2: Dict mapping model_name -> metrics dict for Setting2
        metrics_setting5: Dict mapping model_name -> metrics dict for Setting5
        output_path: Path to save the figure
        models_to_compare: List of model names to compare
        title: Figure title
        metrics_keys: List of metric keys to plot
        figsize: Figure size
        dpi: DPI for saving
    
    Returns:
        Path to saved figure
    """
    if metrics_keys is None:
        metrics_keys = list(METRICS_CONFIG.keys())
    
    n_metrics = len(metrics_keys)
    n_models = len(models_to_compare)
    
    # Create subplots: one per metric
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    colors_s2 = '#3498db'  # Blue for Setting2
    colors_s5 = '#e74c3c'  # Red for Setting5
    
    for metric_idx, metric_key in enumerate(metrics_keys):
        ax = axes[metric_idx]
        
        x = np.arange(n_models)
        width = 0.35
        
        s2_values = [metrics_setting2.get(m, {}).get(metric_key, 0) for m in models_to_compare]
        s5_values = [metrics_setting5.get(m, {}).get(metric_key, 0) for m in models_to_compare]
        
        bars1 = ax.bar(x - width/2, s2_values, width, label='Setting2 (Sequential)',
                      color=colors_s2, alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, s5_values, width, label='Setting5 (Shuffled)',
                      color=colors_s5, alpha=0.8, edgecolor='black')
        
        # Add value labels on top of bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7, rotation=45)
        
        ax.set_xlabel('Model', fontsize=10, fontweight='bold')
        ax.set_title(METRICS_CONFIG[metric_key]['display'], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models_to_compare], rotation=45, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=0)  # Hide y-axis values
        ax.grid(axis='y', alpha=0.3)
        
        if metric_idx == 0:
            ax.legend(loc='best', fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_interpolation_ladder(
    metrics_setting1: Dict[str, Dict[str, float]],
    metrics_setting2: Dict[str, Dict[str, float]],
    metrics_setting6: Dict[str, Dict[str, float]],
    output_path: Path,
    model_groups: Optional[List[Tuple[str, str, str]]] = None,
    title: str = "Linear Interpolation Enhancement: Performance Ladder",
    metrics_keys: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 8),
    dpi: int = 300
) -> Path:
    """
    Create a ladder chart showing Setting1 -> Setting6 -> Setting2 progression.
    
    Args:
        metrics_setting1: Dict mapping model_name -> metrics dict for Setting1
        metrics_setting2: Dict mapping model_name -> metrics dict for Setting2
        metrics_setting6: Dict mapping model_name -> metrics dict for Setting6
        output_path: Path to save the figure
        model_groups: List of (S1_model, S6_model, S2_model) tuples
        title: Figure title
        metrics_keys: List of metric keys to plot
        figsize: Figure size
        dpi: DPI for saving
    
    Returns:
        Path to saved figure
    """
    if metrics_keys is None:
        metrics_keys = list(METRICS_CONFIG.keys())
    
    if model_groups is None:
        model_groups = [
            ('vae', 'vae', 'vae'),
            ('sb', 'sb_mlplus', 'sb_mlplus'),
            ('ot', 'batch_ot', 'batch_ot'),
        ]
    
    n_metrics = len(metrics_keys)
    n_groups = len(model_groups)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']  # S1, S6, S2
    
    for metric_idx, metric_key in enumerate(metrics_keys):
        ax = axes[metric_idx]
        
        x = np.arange(n_groups)
        width = 0.25
        
        s1_values = []
        s6_values = []
        s2_values = []
        group_labels = []
        
        for s1_model, s6_model, s2_model in model_groups:
            s1_val = metrics_setting1.get(s1_model, {}).get(metric_key, 0)
            s6_val = metrics_setting6.get(s6_model, {}).get(metric_key, 0)
            s2_val = metrics_setting2.get(s2_model, {}).get(metric_key, 0)
            
            s1_values.append(s1_val)
            s6_values.append(s6_val)
            s2_values.append(s2_val)
            group_labels.append(f'{s1_model}→{s2_model}')
        
        bars1 = ax.bar(x - width, s1_values, width, label='Setting1', color=colors[0], alpha=0.8)
        bars2 = ax.bar(x, s6_values, width, label='Setting6', color=colors[1], alpha=0.8)
        bars3 = ax.bar(x + width, s2_values, width, label='Setting2', color=colors[2], alpha=0.8)
        
        # Add value labels on top of bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7, rotation=45)
        
        ax.set_xlabel('Model Group', fontsize=10, fontweight='bold')
        ax.set_title(METRICS_CONFIG[metric_key]['display'], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=8)
        ax.tick_params(axis='y', labelsize=0)  # Hide y-axis values
        ax.grid(axis='y', alpha=0.3)
        
        if metric_idx == 0:
            ax.legend(loc='best', fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


# =============================================================================
# Utility Functions
# =============================================================================

def save_figure_multi_format(
    fig: plt.Figure,
    output_path: Path,
    formats: List[str] = ['pdf', 'png'],
    dpi: int = 300
) -> List[Path]:
    """
    Save figure in multiple formats.
    
    Args:
        fig: Matplotlib figure
        output_path: Base output path (without extension)
        formats: List of formats to save
        dpi: DPI for raster formats
    
    Returns:
        List of saved file paths
    """
    saved_paths = []
    base_path = output_path.with_suffix('')
    
    for fmt in formats:
        path = base_path.with_suffix(f'.{fmt}')
        if fmt in ['png', 'jpg', 'jpeg']:
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(path, bbox_inches='tight')
        saved_paths.append(path)
    
    return saved_paths


def extract_metrics_from_results(
    results_dict: Dict,
    model_name: str
) -> Dict[str, float]:
    """
    Extract metrics for a specific model from evaluation results.
    
    Args:
        results_dict: Evaluation results dictionary
        model_name: Name of the model
    
    Returns:
        Dictionary of metric values
    """
    if model_name not in results_dict:
        return {}
    
    model_data = results_dict[model_name]
    
    # Handle nested 'metrics' key
    if 'metrics' in model_data:
        return model_data['metrics']
    
    return model_data
