#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Analyzer - Marginal Contribution Analysis

This module analyzes the marginal contribution of each intermediate timepoint
by comparing the performance of models trained with and without that timepoint.

Author: Shi Pan
Date: 2024-11-18
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .base_visualizer import BaseVisualizer


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


class AblationAnalyzer(BaseVisualizer):
    """
    Analyzer for timepoint ablation studies.
    
    This class provides methods to:
    - Load experiment results from multiple ablation variants
    - Compute marginal contribution of each timepoint
    - Identify critical timepoints
    - Generate comprehensive visualizations and reports
    """
    
    def __init__(
        self,
        output_dir: str,
        device: str = 'cpu',
        random_seed: int = 42
    ):
        """
        Initialize ablation analyzer.
        
        Args:
            output_dir: Directory to save analysis outputs
            device: Device for computation ('cuda' or 'cpu')
            random_seed: Random seed for reproducibility
        """
        super().__init__(output_dir, device, random_seed)
    
    def load_experiment_results(
        self,
        output_base: Path,
        experiment_name: str
    ) -> Dict:
        """
        Load results.json from an experiment directory.
        
        Args:
            output_base: Base directory containing experiment outputs
            experiment_name: Name of the experiment subdirectory
        
        Returns:
            Dictionary containing experiment results
        
        Raises:
            FileNotFoundError: If results file doesn't exist
        """
        results_path = output_base / experiment_name / "results.json"
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def extract_metrics(
        self,
        results: Dict,
        model_name: str = 'sb_mlplus'
    ) -> Dict[str, float]:
        """
        Extract evaluation metrics for a specific model.
        
        Args:
            results: Results dictionary from experiment
            model_name: Name of the model to extract metrics for
        
        Returns:
            Dictionary mapping metric names to values
        
        Raises:
            KeyError: If model not found in results
        """
        if model_name not in results:
            raise KeyError(f"Model '{model_name}' not found in results")
        
        eval_metrics = results[model_name].get('evaluation', {})
        return {m: eval_metrics.get(m, np.nan) for m in METRICS}
    
    def compute_marginal_contribution(
        self,
        P_full: Dict[str, float],
        P_ablations: Dict[str, Dict[str, float]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute marginal contribution for each timepoint.
        
        Args:
            P_full: Metrics from full model (all timepoints)
            P_ablations: Dict of {timepoint: metrics} for ablation variants
        
        Returns:
            Tuple of (delta_P, I_margin):
                - delta_P: DataFrame of absolute marginal contributions (ΔP)
                - I_margin: DataFrame of relative marginal contributions (%)
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
        self,
        I_margin: pd.DataFrame,
        threshold_multiplier: float = 1.5
    ) -> Dict[str, List[str]]:
        """
        Identify critical timepoints for each metric.
        
        A timepoint is critical if I_margin(t) > threshold_multiplier * mean(I_margin)
        
        Args:
            I_margin: DataFrame of relative marginal contributions
            threshold_multiplier: Multiplier for mean threshold
        
        Returns:
            Dictionary mapping metric names to lists of critical timepoints
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
    
    def compute_consistency(self, I_margin: pd.DataFrame) -> pd.Series:
        """
        Compute cross-metric consistency for each timepoint.
        
        Consistency(t) = fraction of metrics where t ranks in top 2
        
        Args:
            I_margin: DataFrame of relative marginal contributions
        
        Returns:
            Series mapping timepoints to consistency scores
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
        self,
        delta_P: pd.DataFrame,
        I_margin: pd.DataFrame
    ):
        """
        Generate visualizations for marginal contribution analysis.
        
        Creates three plots:
        1. Bar plots for each metric (absolute contribution)
        2. Heatmap (relative contribution %)
        3. Summary bar plot (average across metrics)
        
        Args:
            delta_P: DataFrame of absolute marginal contributions
            I_margin: DataFrame of relative marginal contributions
        """
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
        self._save_figure(fig, 'marginal_contribution_absolute')
        self._print_success(f"Saved: {self.output_dir / 'marginal_contribution_absolute.png'}")
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
        self._save_figure(fig, 'marginal_contribution_heatmap')
        self._print_success(f"Saved: {self.output_dir / 'marginal_contribution_heatmap.png'}")
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
        self._save_figure(fig, 'marginal_contribution_summary')
        self._print_success(f"Saved: {self.output_dir / 'marginal_contribution_summary.png'}")
        plt.close()
    
    def generate_report(
        self,
        delta_P: pd.DataFrame,
        I_margin: pd.DataFrame,
        critical_timepoints: Dict[str, List[str]],
        consistency: pd.Series
    ):
        """
        Generate a comprehensive text report.
        
        Args:
            delta_P: DataFrame of absolute marginal contributions
            I_margin: DataFrame of relative marginal contributions
            critical_timepoints: Dictionary of critical timepoints per metric
            consistency: Series of consistency scores per timepoint
        """
        report_path = self.output_dir / 'ablation_analysis_report.txt'
        
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
            elif avg_contribution.get('3d', 0) > 2 * avg_contribution[['8h', '1d']].mean():
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
        
        self._print_success(f"Saved: {report_path}")
    
    def run_analysis(
        self,
        output_base: Path,
        full_exp_name: str,
        ablation_exp_names: Dict[str, str],
        model_name: str = 'sb_mlplus'
    ):
        """
        Run complete ablation analysis.
        
        Args:
            output_base: Base directory containing experiment outputs
            full_exp_name: Name of the full experiment (all timepoints)
            ablation_exp_names: Dict mapping timepoints to ablation experiment names
            model_name: Name of the model to analyze
        
        Example:
            >>> analyzer = AblationAnalyzer(output_dir='analysis_output')
            >>> analyzer.run_analysis(
            ...     output_base=Path('/path/to/outputs'),
            ...     full_exp_name='EMT_Part1_Setting2',
            ...     ablation_exp_names={
            ...         '8h': 'experiment_EMT_Part1_setting4_ablation_remove_8h',
            ...         '1d': 'experiment_EMT_Part1_setting4_ablation_remove_1d',
            ...         '3d': 'experiment_EMT_Part1_setting4_ablation_remove_3d'
            ...     },
            ...     model_name='sb_mlplus'
            ... )
        """
        self._print_section("Experiment 4: Marginal Contribution Analysis")
        self._print_info(f"Output base: {output_base}")
        self._print_info(f"Model: {model_name}")
        self._print_info(f"Output directory: {self.output_dir}")
        print()
        
        # Load results
        self._print_subsection("Loading Results")
        try:
            results_full = self.load_experiment_results(output_base, full_exp_name)
            self._print_success(f"Loaded full experiment: {full_exp_name}")
            
            results_ablations = {}
            for tp, exp_name in ablation_exp_names.items():
                results_ablations[tp] = self.load_experiment_results(output_base, exp_name)
                self._print_success(f"Loaded ablation experiment (remove {tp}): {exp_name}")
        
        except FileNotFoundError as e:
            self._print_error(str(e))
            print("\nPlease ensure all experiments have been run:")
            print(f"  1. {full_exp_name} (full model)")
            for tp, exp_name in ablation_exp_names.items():
                print(f"  2. {exp_name} (remove {tp})")
            return
        
        # Extract metrics
        self._print_subsection("Extracting Metrics")
        P_full = self.extract_metrics(results_full, model_name)
        P_ablations = {
            tp: self.extract_metrics(results_ablations[tp], model_name)
            for tp in ablation_exp_names.keys()
        }
        self._print_success("Metrics extracted successfully")
        
        # Compute marginal contributions
        self._print_subsection("Computing Marginal Contributions")
        delta_P, I_margin = self.compute_marginal_contribution(P_full, P_ablations)
        self._print_success("Marginal contributions computed")
        
        # Identify critical timepoints
        self._print_subsection("Identifying Critical Timepoints")
        critical_timepoints = self.identify_critical_timepoints(I_margin)
        self._print_success("Critical timepoints identified")
        
        # Compute consistency
        self._print_subsection("Computing Cross-Metric Consistency")
        consistency = self.compute_consistency(I_margin)
        self._print_success("Consistency computed")
        
        # Save numerical results
        self._print_subsection("Saving Numerical Results")
        delta_P.to_csv(self.output_dir / 'delta_P.csv')
        self._print_success("Saved: delta_P.csv")
        
        I_margin.to_csv(self.output_dir / 'I_margin.csv')
        self._print_success("Saved: I_margin.csv")
        
        consistency.to_csv(self.output_dir / 'consistency.csv')
        self._print_success("Saved: consistency.csv")
        
        with open(self.output_dir / 'critical_timepoints.json', 'w') as f:
            json.dump(critical_timepoints, f, indent=2)
        self._print_success("Saved: critical_timepoints.json")
        
        # Generate visualizations
        self._print_subsection("Generating Visualizations")
        self.plot_marginal_contribution(delta_P, I_margin)
        
        # Generate report
        self._print_subsection("Generating Report")
        self.generate_report(delta_P, I_margin, critical_timepoints, consistency)
        
        print()
        self._print_section("Analysis Complete!")
        self._print_info(f"Results saved to: {self.output_dir}")
