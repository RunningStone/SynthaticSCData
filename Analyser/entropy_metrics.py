#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy Metrics for Continuous Time Data Quality Assessment
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma, loggamma
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def estimate_entropy_knn(X: np.ndarray, k: int = 10) -> float:
    """
    Estimate entropy using k-nearest neighbors method
    Based on Kozachenko-Leonenko estimator
    
    Args:
        X: Data matrix (n_samples, n_features)
        k: Number of nearest neighbors
    
    Returns:
        Estimated entropy value
    """
    n_samples, n_dims = X.shape
    
    if n_samples < k + 1:
        return np.nan
    
    # Fit k-NN
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Get k-th nearest neighbor distance (excluding self)
    rk = distances[:, k]
    
    # Kozachenko-Leonenko estimator
    log_volume_unit_ball = (n_dims / 2) * np.log(np.pi) - loggamma(n_dims / 2 + 1)
    log_rk = np.log(rk + 1e-10)
    mean_log_rk = np.mean(log_rk)
    
    digamma_n = digamma(n_samples)
    digamma_k = digamma(k)
    
    entropy_estimate = n_dims * mean_log_rk + log_volume_unit_ball + digamma_n - digamma_k
    
    return entropy_estimate


def calculate_entropy_timeline(
    adata: sc.AnnData,
    time_column: str = 'continuous_time',
    time_slice_column: str = 'time_slice',
    method: str = 'knn',
    k: int = 10
) -> pd.DataFrame:
    """
    Calculate entropy for each time slice
    
    Args:
        adata: AnnData object with continuous time data
        time_column: Column name for continuous time values
        time_slice_column: Column name for time slice labels
        method: Entropy estimation method ('knn')
        k: Number of neighbors for KNN method
    
    Returns:
        DataFrame with time_slice, continuous_time, entropy, n_cells
    """
    results = []
    
    # Get unique time slices and their average continuous times
    time_slice_info = []
    for time_slice in adata.obs[time_slice_column].unique():
        mask = adata.obs[time_slice_column] == time_slice
        avg_time = adata.obs.loc[mask, time_column].mean()
        time_slice_info.append((time_slice, avg_time))
    
    # Sort by continuous time (not by string label)
    time_slice_info.sort(key=lambda x: x[1])
    
    for time_slice, _ in time_slice_info:
        mask = adata.obs[time_slice_column] == time_slice
        
        if mask.sum() == 0:
            continue
        
        # Get expression data
        X = adata[mask].X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Calculate entropy
        entropy = estimate_entropy_knn(X, k=k)
        
        # Get average continuous time for this slice
        avg_time = adata.obs.loc[mask, time_column].mean()
        n_cells = mask.sum()
        
        results.append({
            'time_slice': time_slice,
            'continuous_time': avg_time,
            'entropy': entropy,
            'n_cells': n_cells
        })
    
    return pd.DataFrame(results)


def plot_entropy_timeline(
    entropy_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Entropy Timeline",
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot entropy timeline
    
    Args:
        entropy_df: DataFrame from calculate_entropy_timeline
        output_path: Path to save plot (if None, display only)
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot line
    ax.plot(entropy_df['continuous_time'], entropy_df['entropy'], 
            'b-', linewidth=2, alpha=0.7, label='Entropy')
    
    # Plot markers
    ax.scatter(entropy_df['continuous_time'], entropy_df['entropy'],
               c='blue', s=100, zorder=3, edgecolors='black', linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Continuous Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Entropy (KNN)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_entropy_by_category(
    adata: sc.AnnData,
    category_column: str,
    time_column: str = 'continuous_time',
    time_slice_column: str = 'time_slice',
    k: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Calculate entropy timeline for each category
    
    Args:
        adata: AnnData object
        category_column: Column name for categories (e.g., 'is_real')
        time_column: Column name for continuous time
        time_slice_column: Column name for time slices
        k: Number of neighbors for KNN
    
    Returns:
        Dict mapping category values to entropy DataFrames
    """
    results = {}
    
    for category in adata.obs[category_column].unique():
        mask = adata.obs[category_column] == category
        adata_subset = adata[mask].copy()
        
        entropy_df = calculate_entropy_timeline(
            adata_subset,
            time_column=time_column,
            time_slice_column=time_slice_column,
            k=k
        )
        
        results[str(category)] = entropy_df
    
    return results


def plot_entropy_comparison(
    entropy_dict: Dict[str, pd.DataFrame],
    output_path: Optional[str] = None,
    title: str = "Entropy Comparison",
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot entropy timelines for multiple categories
    
    Args:
        entropy_dict: Dict from calculate_entropy_by_category
        output_path: Path to save plot
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (category, df) in enumerate(entropy_dict.items()):
        color = colors[i % len(colors)]
        
        ax.plot(df['continuous_time'], df['entropy'],
                linewidth=2, alpha=0.7, label=category, color=color)
        ax.scatter(df['continuous_time'], df['entropy'],
                   s=80, zorder=3, edgecolors='black', linewidth=1.5, color=color)
    
    ax.set_xlabel('Continuous Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Entropy (KNN)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_continuous_data_quality(
    adata_path: str,
    output_dir: str,
    time_column: str = 'continuous_time',
    time_slice_column: str = 'time_slice'
):
    """
    Complete entropy analysis for continuous time data
    
    Args:
        adata_path: Path to continuous time h5ad file
        output_dir: Directory to save outputs
        time_column: Column name for continuous time
        time_slice_column: Column name for time slices
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    print(f"Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    # Overall entropy timeline
    print("\nCalculating overall entropy timeline...")
    entropy_df = calculate_entropy_timeline(adata, time_column, time_slice_column)
    
    # Save results
    csv_path = output_path / "entropy_timeline.csv"
    entropy_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Plot
    plot_path = output_path / "entropy_timeline.png"
    plot_entropy_timeline(entropy_df, output_path=plot_path, title="Overall Entropy Timeline")
    
    # Entropy by real/generated
    if 'is_real' in adata.obs.columns:
        print("\nCalculating entropy by real/generated...")
        entropy_by_type = calculate_entropy_by_category(
            adata, 'is_real', time_column, time_slice_column
        )
        
        # Save
        for category, df in entropy_by_type.items():
            csv_path = output_path / f"entropy_timeline_{category}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")
        
        # Plot comparison
        plot_path = output_path / "entropy_comparison_real_vs_generated.png"
        plot_entropy_comparison(
            entropy_by_type,
            output_path=plot_path,
            title="Entropy: Real vs Generated Cells"
        )
    
    print(f"\n✓ Analysis complete. Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze entropy of continuous time data')
    parser.add_argument('--input', type=str, required=True, help='Input h5ad file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--time_column', type=str, default='continuous_time')
    parser.add_argument('--time_slice_column', type=str, default='time_slice')
    
    args = parser.parse_args()
    
    analyze_continuous_data_quality(
        adata_path=args.input,
        output_dir=args.output,
        time_column=args.time_column,
        time_slice_column=args.time_slice_column
    )
