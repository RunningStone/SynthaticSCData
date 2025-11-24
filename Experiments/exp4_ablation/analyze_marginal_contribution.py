#!/usr/bin/env python3
"""
Experiment 4: Marginal Contribution Analysis
============================================

This script analyzes the marginal contribution of each intermediate timepoint
by comparing the performance of models trained with and without that timepoint.

Usage:
    python analyze_marginal_contribution.py --output_base /path/to/OUTPUTs/SynthaticSCData

Author: Shi Pan
Date: 2024-11-18
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple


# Metric names and their display properties
METRICS = [
    'test_loss', 'frechet_distance', 'mae', 'pcc', 
    'wasserstein_distance', 'mmd', 'r2_mean', 'js_divergence',
    'correlation_frobenius_diff', 'correlation_structure_corr'
]

METRIC_DISPLAY_NAMES = {
    'test_loss': 'Test Loss',
    'frechet_distance': 'Fréchet Distance',
    'mae': 'MAE',
    'pcc': 'Pearson Corr.',
    'wasserstein_distance': 'Wasserstein Dist.',
    'mmd': 'MMD',
    'r2_mean': 'R² Mean',
    'js_divergence': 'JS Divergence',
    'correlation_frobenius_diff': 'Corr. Frob. Diff',
    'correlation_structure_corr': 'Corr. Struct. Corr'
}

# Metrics where lower is better
LOWER_IS_BETTER = {
    'test_loss', 'frechet_distance', 'mae', 'wasserstein_distance',
    'mmd', 'js_divergence', 'correlation_frobenius_diff'
}


def load_results(output_base: Path, experiment_name: str) -> Dict:
    """Load results.json from an experiment directory."""
    results_path = output_base / experiment_name / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def extract_metrics(results: Dict, model_name: str = 'sb_mlplus') -> Dict[str, float]:
    """Extract evaluation metrics for a specific model."""
    if model_name not in results:
        raise KeyError(f"Model '{model_name}' not found in results")
    
    eval_metrics = results[model_name].get('evaluation', {})
    return {m: eval_metrics.get(m, np.nan) for m in METRICS}


def compute_marginal_contribution(
    P_full: Dict[str, float],
    P_ablations: Dict[str, Dict[str, float]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute marginal contribution for each timepoint.
    
    Args:
        P_full: Metrics from full model (Setting2)
        P_ablations: Dict of {timepoint: metrics} for ablation variants
    
    Returns:
        delta_P: DataFrame of absolute marginal contributions (ΔP)
        I_margin: DataFrame of relative marginal contributions (%)
    """
    timepoints = list(P_ablations.keys())
    
    # Initialize DataFrames
    delta_P = pd.DataFrame(index=METRICS, columns=timepoints, dtype=float)
    I_margin = pd.DataFrame(index=METRICS, columns=timepoints, dtype=float)
    
    for metric in METRICS:
        p_full = P_full[metric]
        
        for tp in timepoints:
            p_ablation = P_ablations[tp][metric]
            
            # For metrics where lower is better, positive ΔP means removing the timepoint hurts performance
            # For metrics where higher is better, negative ΔP means removing the timepoint hurts performance
            if metric in LOWER_IS_BETTER:
                delta = p_ablation - p_full  # Positive = performance degraded
            else:
                delta = p_full - p_ablation  # Positive = performance degraded
            
            delta_P.loc[metric, tp] = delta
            
            # Relative contribution (%)
            if abs(p_full) > 1e-6:
                I_margin.loc[metric, tp] = (delta / abs(p_full)) * 100
            else:
                I_margin.loc[metric, tp] = 0.0
    
    return delta_P, I_margin


def identify_critical_timepoints(
    I_margin: pd.DataFrame,
    threshold_multiplier: float = 1.5
) -> Dict[str, List[str]]:
    """
    Identify critical timepoints for each metric.
    
    A timepoint is critical if I_margin(t) > threshold_multiplier * mean(I_margin)
    """
    critical = {}
    
    for metric in I_margin.index:
        values = I_margin.loc[metric].values.astype(float)
        mean_val = np.mean(values)
        threshold = threshold_multiplier * mean_val
        
        critical_tps = [
            tp for tp, val in zip(I_margin.columns, values)
            if val > threshold
        ]
        critical[metric] = critical_tps
    
    return critical


def compute_consistency(I_margin: pd.DataFrame) -> pd.Series:
    """
    Compute cross-metric consistency for each timepoint.
    
    Consistency(t) = fraction of metrics where t ranks in top 2
    """
    timepoints = I_margin.columns
    consistency = {}
    
    for tp in timepoints:
        # Count how many metrics rank this timepoint in top 2
        top2_count = 0
        for metric in I_margin.index:
            values = I_margin.loc[metric].values.astype(float)
            ranks = np.argsort(-values)  # Descending order
            if tp in I_margin.columns[ranks[:2]]:
                top2_count += 1
        
        consistency[tp] = top2_count / len(I_margin.index)
    
    return pd.Series(consistency)


def plot_marginal_contribution(
    delta_P: pd.DataFrame,
    I_margin: pd.DataFrame,
    output_dir: Path
):
    """Generate visualizations for marginal contribution analysis."""
    
    # 1. Bar plot for each metric (absolute contribution)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        values = delta_P.loc[metric].values.astype(float)
        timepoints = delta_P.columns
        
        colors = ['#d62728' if v > 0 else '#2ca02c' for v in values]
        ax.bar(timepoints, values, color=colors, alpha=0.7)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_title(METRIC_DISPLAY_NAMES.get(metric, metric), fontsize=10)
        ax.set_ylabel('Δ' + metric, fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'marginal_contribution_absolute.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'marginal_contribution_absolute.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'marginal_contribution_absolute.png'}")
    plt.close()
    
    # 2. Heatmap (relative contribution %)
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Convert to numeric and handle NaN
    I_margin_numeric = I_margin.astype(float)
    
    sns.heatmap(
        I_margin_numeric,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        center=0,
        cbar_kws={'label': 'Relative Contribution (%)'},
        ax=ax,
        linewidths=0.5
    )
    ax.set_xlabel('Timepoint Removed', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    ax.set_title('Marginal Contribution Heatmap\n(Positive = Performance Degraded)', fontsize=14)
    
    # Replace metric names with display names
    yticklabels = [METRIC_DISPLAY_NAMES.get(m, m) for m in I_margin.index]
    ax.set_yticklabels(yticklabels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'marginal_contribution_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'marginal_contribution_heatmap.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'marginal_contribution_heatmap.png'}")
    plt.close()
    
    # 3. Summary bar plot (average across metrics)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    avg_contribution = I_margin_numeric.mean(axis=0)
    std_contribution = I_margin_numeric.std(axis=0)
    
    timepoints = avg_contribution.index
    x_pos = np.arange(len(timepoints))
    
    ax.bar(x_pos, avg_contribution.values, yerr=std_contribution.values,
           capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(timepoints, fontsize=12)
    ax.set_ylabel('Average Relative Contribution (%)', fontsize=12)
    ax.set_xlabel('Timepoint Removed', fontsize=12)
    ax.set_title('Average Marginal Contribution Across All Metrics', fontsize=14)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'marginal_contribution_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'marginal_contribution_summary.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'marginal_contribution_summary.png'}")
    plt.close()


def generate_report(
    delta_P: pd.DataFrame,
    I_margin: pd.DataFrame,
    critical_timepoints: Dict[str, List[str]],
    consistency: pd.Series,
    output_dir: Path
):
    """Generate a comprehensive text report."""
    
    report_path = output_dir / 'ablation_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Experiment 4: Marginal Contribution Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. Summary statistics
        f.write("1. SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        avg_contribution = I_margin.mean(axis=0)
        f.write("Average Relative Contribution (%) across all metrics:\n")
        for tp, val in avg_contribution.items():
            f.write(f"  {tp:>4s}: {val:>6.2f}%\n")
        f.write("\n")
        
        # 2. Cross-metric consistency
        f.write("2. CROSS-METRIC CONSISTENCY\n")
        f.write("-" * 80 + "\n")
        f.write("Fraction of metrics where timepoint ranks in top 2:\n")
        for tp, val in consistency.items():
            f.write(f"  {tp:>4s}: {val:>5.2%}\n")
        f.write("\n")
        
        # 3. Critical timepoints
        f.write("3. CRITICAL TIMEPOINTS (I_margin > 1.5 × mean)\n")
        f.write("-" * 80 + "\n")
        for metric, tps in critical_timepoints.items():
            if tps:
                f.write(f"  {METRIC_DISPLAY_NAMES.get(metric, metric):30s}: {', '.join(tps)}\n")
            else:
                f.write(f"  {METRIC_DISPLAY_NAMES.get(metric, metric):30s}: None\n")
        f.write("\n")
        
        # 4. Detailed contribution table
        f.write("4. DETAILED MARGINAL CONTRIBUTION TABLE\n")
        f.write("-" * 80 + "\n")
        f.write(I_margin.to_string())
        f.write("\n\n")
        
        # 5. Interpretation
        f.write("5. INTERPRETATION\n")
        f.write("-" * 80 + "\n")
        
        # Identify the most critical timepoint
        most_critical = avg_contribution.idxmax()
        max_contribution = avg_contribution.max()
        
        f.write(f"Most critical timepoint: {most_critical} ({max_contribution:.2f}% average contribution)\n")
        f.write(f"Most consistent timepoint: {consistency.idxmax()} ({consistency.max():.2%} consistency)\n\n")
        
        # Pattern detection
        std_dev = avg_contribution.std()
        mean_val = avg_contribution.mean()
        
        if std_dev < 0.2 * mean_val:
            f.write("Pattern: UNIFORM CONTRIBUTION (Mode 1)\n")
            f.write("All timepoints contribute roughly equally. This suggests that EMT information\n")
            f.write("is uniformly distributed across time, weakening the hypothesis that specific\n")
            f.write("intermediate states are particularly critical.\n")
        elif avg_contribution['3d'] > 2 * avg_contribution[['8h', '1d']].mean():
            f.write("Pattern: LATE-STAGE SENSITIVITY (Mode 2)\n")
            f.write("The 3d timepoint (close to EMT peak) is significantly more important than\n")
            f.write("early timepoints. This supports the hypothesis that late intermediate states\n")
            f.write("contain irreplaceable information about the state space exploration boundary.\n")
        else:
            f.write("Pattern: STAGE-DEPENDENT CONTRIBUTION (Mode 3)\n")
            f.write("Different timepoints show asymmetric contributions, suggesting that certain\n")
            f.write("stages of EMT are more difficult to model than others.\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze marginal contribution of timepoints in EMT ablation study'
    )
    parser.add_argument(
        '--output_base',
        type=str,
        default='/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData',
        help='Base directory containing experiment outputs'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='sb_mlplus',
        help='Model name to analyze'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save analysis results (default: output_base/exp4_ablation_analysis)'
    )
    
    args = parser.parse_args()
    
    output_base = Path(args.output_base)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = output_base / 'exp4_ablation_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Experiment 4: Marginal Contribution Analysis")
    print("=" * 80)
    print(f"Output base: {output_base}")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load results
    print("Loading results...")
    try:
        results_full = load_results(output_base, 'EMT_Part1_Setting2')
        results_remove_8h = load_results(output_base, 'experiment_EMT_Part1_setting4_ablation_remove_8h')
        results_remove_1d = load_results(output_base, 'experiment_EMT_Part1_setting4_ablation_remove_1d')
        results_remove_3d = load_results(output_base, 'experiment_EMT_Part1_setting4_ablation_remove_3d')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure all experiments have been run:")
        print("  1. EMT_Part1_Setting2 (full model)")
        print("  2. experiment_EMT_Part1_setting4_ablation_remove_8h")
        print("  3. experiment_EMT_Part1_setting4_ablation_remove_1d")
        print("  4. experiment_EMT_Part1_setting4_ablation_remove_3d")
        return
    
    # Extract metrics
    print("Extracting metrics...")
    P_full = extract_metrics(results_full, args.model)
    P_ablations = {
        '8h': extract_metrics(results_remove_8h, args.model),
        '1d': extract_metrics(results_remove_1d, args.model),
        '3d': extract_metrics(results_remove_3d, args.model),
    }
    
    # Compute marginal contributions
    print("Computing marginal contributions...")
    delta_P, I_margin = compute_marginal_contribution(P_full, P_ablations)
    
    # Identify critical timepoints
    print("Identifying critical timepoints...")
    critical_timepoints = identify_critical_timepoints(I_margin)
    
    # Compute consistency
    print("Computing cross-metric consistency...")
    consistency = compute_consistency(I_margin)
    
    # Save numerical results
    print("Saving numerical results...")
    delta_P.to_csv(output_dir / 'delta_P.csv')
    I_margin.to_csv(output_dir / 'I_margin.csv')
    consistency.to_csv(output_dir / 'consistency.csv')
    
    with open(output_dir / 'critical_timepoints.json', 'w') as f:
        json.dump(critical_timepoints, f, indent=2)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_marginal_contribution(delta_P, I_margin, output_dir)
    
    # Generate report
    print("Generating report...")
    generate_report(delta_P, I_margin, critical_timepoints, consistency, output_dir)
    
    print()
    print("=" * 80)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
