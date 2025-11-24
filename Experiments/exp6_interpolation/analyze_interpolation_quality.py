#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 6: Analyze Interpolation Quality
Compute specialized metrics for evaluating interpolation effectiveness
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def compute_interpolation_effectiveness_index(
    real_data: np.ndarray,
    generated_data_setting1: np.ndarray,
    generated_data_interpolated: np.ndarray
) -> float:
    """
    Compute Interpolation Effectiveness Index (IEI).
    
    IEI measures how much interpolation improves over boundary-only training:
    IEI = 1 - (E_interp / E_setting1)
    
    where E is the prediction error (MAE).
    
    Args:
        real_data: Real intermediate time point data
        generated_data_setting1: Generated data from Setting 1 (boundary only)
        generated_data_interpolated: Generated data from interpolated training
        
    Returns:
        IEI value (higher is better, 1.0 means perfect improvement)
    """
    # Compute MAE for both settings
    mae_setting1 = np.mean(np.abs(real_data - generated_data_setting1))
    mae_interpolated = np.mean(np.abs(real_data - generated_data_interpolated))
    
    # IEI = 1 - (E_interp / E_setting1)
    if mae_setting1 == 0:
        return 0.0
    
    iei = 1.0 - (mae_interpolated / mae_setting1)
    
    return float(iei)


def compute_residual_structure_index(
    real_data: np.ndarray,
    interpolated_data: np.ndarray,
    n_components: int = 10
) -> Dict[str, float]:
    """
    Compute Residual Structure Index (RSI).
    
    RSI measures how much structured information remains in the residuals
    between real and interpolated data. High RSI indicates that interpolation
    fails to capture systematic patterns.
    
    RSI = (variance explained by top-k PCs) / (total variance)
    
    Args:
        real_data: Real intermediate time point data (n_samples, n_features)
        interpolated_data: Interpolated data (n_samples, n_features)
        n_components: Number of top principal components to consider
        
    Returns:
        Dictionary with RSI and related metrics
    """
    # Compute residuals
    residuals = real_data - interpolated_data
    
    # Fit PCA on residuals
    pca = PCA(n_components=min(n_components, residuals.shape[0], residuals.shape[1]))
    pca.fit(residuals)
    
    # RSI = variance explained by top components
    rsi = np.sum(pca.explained_variance_ratio_[:n_components])
    
    # Additional metrics
    total_variance = np.var(residuals, axis=0).sum()
    explained_variance = np.sum(pca.explained_variance_)
    
    return {
        'rsi': float(rsi),
        'total_residual_variance': float(total_variance),
        'explained_variance_top_k': float(explained_variance),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'n_components': n_components
    }


def compute_per_timepoint_metrics(
    real_data_dict: Dict[str, np.ndarray],
    generated_data_dict: Dict[str, np.ndarray],
    timepoint_labels: List[str]
) -> pd.DataFrame:
    """
    Compute metrics for each intermediate time point separately.
    
    Args:
        real_data_dict: Dictionary mapping timepoint -> real data
        generated_data_dict: Dictionary mapping timepoint -> generated data
        timepoint_labels: List of intermediate timepoint labels
        
    Returns:
        DataFrame with per-timepoint metrics
    """
    results = []
    
    for tp in timepoint_labels:
        if tp not in real_data_dict or tp not in generated_data_dict:
            continue
        
        real = real_data_dict[tp]
        gen = generated_data_dict[tp]
        
        # Ensure same number of samples
        n_samples = min(real.shape[0], gen.shape[0])
        real = real[:n_samples]
        gen = gen[:n_samples]
        
        # Compute metrics
        mae = np.mean(np.abs(real - gen))
        mse = np.mean((real - gen) ** 2)
        
        # Per-feature correlation
        correlations = []
        for i in range(real.shape[1]):
            try:
                corr = np.corrcoef(real[:, i], gen[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            except:
                pass
        
        mean_corr = np.mean(correlations) if correlations else 0.0
        
        # R² score
        try:
            r2 = r2_score(real.flatten(), gen.flatten())
        except:
            r2 = 0.0
        
        results.append({
            'timepoint': tp,
            'mae': mae,
            'mse': mse,
            'mean_correlation': mean_corr,
            'r2_score': r2,
            'n_samples': n_samples
        })
    
    return pd.DataFrame(results)


def visualize_interpolation_quality(
    results_df: pd.DataFrame,
    output_dir: Path
):
    """
    Create visualizations for interpolation quality analysis.
    
    Args:
        results_df: DataFrame with per-timepoint metrics
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: MAE per timepoint
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MAE
    ax = axes[0, 0]
    ax.bar(results_df['timepoint'], results_df['mae'], color='steelblue', alpha=0.7)
    ax.set_xlabel('Time Point', fontsize=12)
    ax.set_ylabel('Mean Absolute Error', fontsize=12)
    ax.set_title('MAE per Intermediate Time Point', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Correlation
    ax = axes[0, 1]
    ax.bar(results_df['timepoint'], results_df['mean_correlation'], color='coral', alpha=0.7)
    ax.set_xlabel('Time Point', fontsize=12)
    ax.set_ylabel('Mean Correlation', fontsize=12)
    ax.set_title('Mean Correlation per Time Point', fontsize=14, fontweight='bold')
    ax.axhline(y=0.8, color='red', linestyle='--', label='Target: 0.8')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # R² score
    ax = axes[1, 0]
    ax.bar(results_df['timepoint'], results_df['r2_score'], color='mediumseagreen', alpha=0.7)
    ax.set_xlabel('Time Point', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Score per Time Point', fontsize=14, fontweight='bold')
    ax.axhline(y=0.7, color='red', linestyle='--', label='Target: 0.7')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # MSE
    ax = axes[1, 1]
    ax.bar(results_df['timepoint'], results_df['mse'], color='mediumpurple', alpha=0.7)
    ax.set_xlabel('Time Point', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('MSE per Time Point', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_timepoint_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'per_timepoint_metrics.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved per-timepoint metrics plot to {output_dir}")


def visualize_residual_structure(
    rsi_results: Dict[str, float],
    output_dir: Path
):
    """
    Visualize residual structure analysis.
    
    Args:
        rsi_results: Dictionary with RSI results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot explained variance ratio
    explained_var_ratio = rsi_results['explained_variance_ratio']
    n_components = len(explained_var_ratio)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    ax = axes[0]
    ax.bar(range(1, n_components + 1), explained_var_ratio, color='steelblue', alpha=0.7)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax.set_title('Residual PCA Scree Plot', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Cumulative explained variance
    ax = axes[1]
    cumsum = np.cumsum(explained_var_ratio)
    ax.plot(range(1, n_components + 1), cumsum, marker='o', color='coral', linewidth=2)
    ax.axhline(y=0.7, color='red', linestyle='--', label='70% threshold')
    ax.axhline(y=rsi_results['rsi'], color='green', linestyle='--', 
               label=f"RSI (top {rsi_results['n_components']}): {rsi_results['rsi']:.3f}")
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_structure_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'residual_structure_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved residual structure plot to {output_dir}")


def create_summary_report(
    iei_results: Dict[str, float],
    rsi_results: Dict[str, float],
    per_timepoint_df: pd.DataFrame,
    output_dir: Path
):
    """
    Create a comprehensive summary report.
    
    Args:
        iei_results: IEI results for each timepoint
        rsi_results: RSI results
        per_timepoint_df: Per-timepoint metrics
        output_dir: Directory to save report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / 'interpolation_quality_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EXPERIMENT 6: INTERPOLATION QUALITY ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. INTERPOLATION EFFECTIVENESS INDEX (IEI)\n")
        f.write("-"*70 + "\n")
        f.write("IEI = 1 - (E_interp / E_setting1)\n")
        f.write("Higher values indicate better interpolation effectiveness\n\n")
        
        for tp, iei in iei_results.items():
            f.write(f"  {tp}: {iei:.4f}\n")
        
        avg_iei = np.mean(list(iei_results.values()))
        f.write(f"\n  Average IEI: {avg_iei:.4f}\n")
        
        # Interpretation
        f.write("\nInterpretation:\n")
        if avg_iei > 0.85:
            f.write("  ✓ INTERPOLATION EFFECTIVE: Geometric interpolation captures most dynamics\n")
        elif avg_iei > 0.5:
            f.write("  ~ PARTIAL EFFECTIVENESS: Interpolation helps but misses key information\n")
        else:
            f.write("  ✗ INTERPOLATION INEFFECTIVE: Real intermediate states contain crucial information\n")
        
        f.write("\n" + "="*70 + "\n\n")
        
        f.write("2. RESIDUAL STRUCTURE INDEX (RSI)\n")
        f.write("-"*70 + "\n")
        f.write("RSI = Variance explained by top-k principal components of residuals\n")
        f.write("Higher values indicate systematic structure in interpolation errors\n\n")
        
        f.write(f"  RSI (top {rsi_results['n_components']} PCs): {rsi_results['rsi']:.4f}\n")
        f.write(f"  Total residual variance: {rsi_results['total_residual_variance']:.2f}\n")
        
        # Interpretation
        f.write("\nInterpretation:\n")
        if rsi_results['rsi'] > 0.7:
            f.write("  ✗ HIGH STRUCTURE: Residuals contain systematic, low-rank information\n")
            f.write("     → Real data has non-interpolable features\n")
        elif rsi_results['rsi'] > 0.3:
            f.write("  ~ MODERATE STRUCTURE: Some systematic patterns in residuals\n")
        else:
            f.write("  ✓ LOW STRUCTURE: Residuals mostly random noise\n")
            f.write("     → Interpolation captures main patterns\n")
        
        f.write("\n" + "="*70 + "\n\n")
        
        f.write("3. PER-TIMEPOINT METRICS\n")
        f.write("-"*70 + "\n\n")
        f.write(per_timepoint_df.to_string(index=False))
        
        f.write("\n\n" + "="*70 + "\n\n")
        
        f.write("4. OVERALL CONCLUSION\n")
        f.write("-"*70 + "\n")
        
        if avg_iei > 0.85 and rsi_results['rsi'] < 0.3:
            f.write("✓ HYPOTHESIS WEAKENED: Interpolation is effective\n")
            f.write("  → Problem may be data coverage rather than dynamics complexity\n")
        elif avg_iei < 0.7 and rsi_results['rsi'] > 0.7:
            f.write("✓ HYPOTHESIS SUPPORTED: Interpolation fails to capture real dynamics\n")
            f.write("  → Real intermediate states contain non-interpolable information\n")
        else:
            f.write("~ MIXED RESULTS: Partial support for hypothesis\n")
            f.write("  → Further analysis needed\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Saved summary report to {report_path}")


if __name__ == "__main__":
    print("This module provides analysis functions for Experiment 6.")
    print("Use the main analysis script to run the complete analysis.")
