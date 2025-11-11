#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced evaluation metrics for generative models
Includes distribution-based and correlation-based metrics
"""

import torch
import numpy as np
from typing import Tuple, Dict
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import r2_score


def compute_wasserstein_distance(
    real: np.ndarray,
    generated: np.ndarray,
    per_feature: bool = True
) -> float:
    """
    Compute Wasserstein Distance (Earth Mover's Distance) between real and generated data.
    
    For high-dimensional data, computes per-feature distance and averages.
    
    Args:
        real: Real data (n_samples, n_features)
        generated: Generated data (n_samples, n_features)
        per_feature: If True, compute per feature and average; if False, use sliced approach
        
    Returns:
        Wasserstein distance
    """
    if per_feature:
        # Compute per-feature Wasserstein distance
        n_features = real.shape[1]
        distances = []
        
        for i in range(n_features):
            wd = wasserstein_distance(real[:, i], generated[:, i])
            distances.append(wd)
        
        return float(np.mean(distances))
    else:
        # For multivariate case, use sliced Wasserstein
        return compute_sliced_wasserstein(real, generated)


def compute_sliced_wasserstein(
    real: np.ndarray,
    generated: np.ndarray,
    n_projections: int = 100
) -> float:
    """
    Compute Sliced Wasserstein Distance.
    
    Projects high-dimensional data onto random 1D directions and computes
    Wasserstein distance for each projection, then averages.
    
    Args:
        real: Real data (n_samples, n_features)
        generated: Generated data (n_samples, n_features)
        n_projections: Number of random projections
        
    Returns:
        Sliced Wasserstein distance
    """
    n_features = real.shape[1]
    distances = []
    
    for _ in range(n_projections):
        # Random projection direction
        direction = np.random.randn(n_features)
        direction = direction / np.linalg.norm(direction)
        
        # Project data
        real_proj = real @ direction
        gen_proj = generated @ direction
        
        # Compute 1D Wasserstein distance
        wd = wasserstein_distance(real_proj, gen_proj)
        distances.append(wd)
    
    return float(np.mean(distances))


def compute_mmd(
    real: np.ndarray,
    generated: np.ndarray,
    kernel: str = 'rbf',
    sigma: float = None
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between real and generated distributions.
    
    MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    
    Args:
        real: Real data (n_samples, n_features)
        generated: Generated data (n_samples, n_features)
        kernel: Kernel type ('rbf' or 'linear')
        sigma: Bandwidth for RBF kernel (if None, use median heuristic)
        
    Returns:
        MMD value
    """
    n_real = real.shape[0]
    n_gen = generated.shape[0]
    
    if kernel == 'rbf':
        # Compute bandwidth using median heuristic if not provided
        if sigma is None:
            # Sample subset for efficiency
            sample_size = min(1000, n_real, n_gen)
            real_sample = real[np.random.choice(n_real, sample_size, replace=False)]
            gen_sample = generated[np.random.choice(n_gen, sample_size, replace=False)]
            combined = np.vstack([real_sample, gen_sample])
            
            # Pairwise distances
            dists = np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=2)
            sigma = np.sqrt(np.median(dists[dists > 0]))
        
        # RBF kernel
        def kernel_func(x, y):
            dists = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
            return np.exp(-dists / (2 * sigma ** 2))
    else:
        # Linear kernel
        def kernel_func(x, y):
            return x @ y.T
    
    # Compute kernel matrices
    k_xx = kernel_func(real, real)
    k_yy = kernel_func(generated, generated)
    k_xy = kernel_func(real, generated)
    
    # MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
    # Remove diagonal for unbiased estimate
    mmd_sq = (np.sum(k_xx) - np.trace(k_xx)) / (n_real * (n_real - 1))
    mmd_sq += (np.sum(k_yy) - np.trace(k_yy)) / (n_gen * (n_gen - 1))
    mmd_sq -= 2 * np.mean(k_xy)
    
    return float(np.sqrt(max(mmd_sq, 0)))


def compute_r2_per_gene(
    real: np.ndarray,
    generated: np.ndarray
) -> Dict[str, float]:
    """
    Compute R² (coefficient of determination) for each gene/feature.
    
    Args:
        real: Real data (n_samples, n_features)
        generated: Generated data (n_samples, n_features)
        
    Returns:
        Dictionary with mean, median, std, min, max R² values
    """
    n_samples = min(real.shape[0], generated.shape[0])
    n_features = real.shape[1]
    
    r2_scores = []
    for i in range(n_features):
        try:
            r2 = r2_score(real[:n_samples, i], generated[:n_samples, i])
            r2_scores.append(r2)
        except:
            # Handle constant features
            r2_scores.append(0.0)
    
    r2_scores = np.array(r2_scores)
    
    return {
        'r2_mean': float(np.mean(r2_scores)),
        'r2_median': float(np.median(r2_scores)),
        'r2_std': float(np.std(r2_scores)),
        'r2_min': float(np.min(r2_scores)),
        'r2_max': float(np.max(r2_scores)),
        'r2_per_gene': r2_scores.tolist()  # Full array for detailed analysis
    }


def compute_js_divergence(
    real: np.ndarray,
    generated: np.ndarray,
    n_bins: int = 50
) -> float:
    """
    Compute Jensen-Shannon Divergence between real and generated distributions.
    
    For high-dimensional data, computes per-feature JS divergence and averages.
    
    Args:
        real: Real data (n_samples, n_features)
        generated: Generated data (n_samples, n_features)
        n_bins: Number of bins for histogram estimation
        
    Returns:
        JS divergence (0 to 1, where 0 means identical distributions)
    """
    n_features = real.shape[1]
    js_divs = []
    
    for i in range(n_features):
        # Get feature data
        real_feat = real[:, i]
        gen_feat = generated[:, i]
        
        # Determine common range
        min_val = min(real_feat.min(), gen_feat.min())
        max_val = max(real_feat.max(), gen_feat.max())
        
        # Avoid division by zero for constant features
        if max_val - min_val < 1e-10:
            js_divs.append(0.0)
            continue
        
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        # Compute histograms
        hist_real, _ = np.histogram(real_feat, bins=bins, density=True)
        hist_gen, _ = np.histogram(gen_feat, bins=bins, density=True)
        
        # Normalize to probability distributions
        hist_real = hist_real / (hist_real.sum() + 1e-10)
        hist_gen = hist_gen / (hist_gen.sum() + 1e-10)
        
        # Add small epsilon to avoid log(0)
        hist_real = hist_real + 1e-10
        hist_gen = hist_gen + 1e-10
        
        # Compute JS divergence
        js = jensenshannon(hist_real, hist_gen)
        js_divs.append(js)
    
    return float(np.mean(js_divs))


def compute_correlation_structure_similarity(
    real: np.ndarray,
    generated: np.ndarray,
    n_samples: int = 100
) -> Dict[str, float]:
    """
    Compute similarity of cell-cell correlation structure.
    
    Measures whether the generated data preserves the correlation patterns
    between cells observed in real data.
    
    Args:
        real: Real data (n_samples, n_features)
        generated: Generated data (n_samples, n_features)
        n_samples: Number of samples to use for correlation computation
        
    Returns:
        Dictionary with Frobenius norm difference and correlation coefficient
    """
    # Subsample if needed
    n_real = min(n_samples, real.shape[0])
    n_gen = min(n_samples, generated.shape[0])
    
    real_subset = real[:n_real]
    gen_subset = generated[:n_gen]
    
    # Compute correlation matrices (cells x cells)
    corr_real = np.corrcoef(real_subset)
    corr_gen = np.corrcoef(gen_subset)
    
    # Match sizes
    n_min = min(corr_real.shape[0], corr_gen.shape[0])
    corr_real = corr_real[:n_min, :n_min]
    corr_gen = corr_gen[:n_min, :n_min]
    
    # Frobenius norm of difference
    frobenius_diff = np.linalg.norm(corr_real - corr_gen, 'fro')
    
    # Correlation between correlation matrices (flatten and correlate)
    # Remove diagonal (self-correlation = 1)
    mask = ~np.eye(n_min, dtype=bool)
    corr_real_flat = corr_real[mask]
    corr_gen_flat = corr_gen[mask]
    
    # Pearson correlation between the two correlation structures
    corr_of_corr = np.corrcoef(corr_real_flat, corr_gen_flat)[0, 1]
    
    return {
        'correlation_frobenius_diff': float(frobenius_diff),
        'correlation_structure_corr': float(corr_of_corr)
    }


def compute_all_metrics(
    real: np.ndarray,
    generated: np.ndarray,
    include_detailed: bool = False
) -> Dict[str, float]:
    """
    Compute all evaluation metrics in one call.
    
    Args:
        real: Real data (n_samples, n_features)
        generated: Generated data (n_samples, n_features)
        include_detailed: If True, include per-gene R² values
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Wasserstein Distance
    try:
        metrics['wasserstein_distance'] = compute_wasserstein_distance(real, generated)
    except Exception as e:
        print(f"Warning: Could not compute Wasserstein distance: {e}")
        metrics['wasserstein_distance'] = float('nan')
    
    # MMD
    try:
        metrics['mmd'] = compute_mmd(real, generated)
    except Exception as e:
        print(f"Warning: Could not compute MMD: {e}")
        metrics['mmd'] = float('nan')
    
    # R² per gene
    try:
        r2_results = compute_r2_per_gene(real, generated)
        if include_detailed:
            metrics.update(r2_results)
        else:
            # Only include summary statistics
            metrics['r2_mean'] = r2_results['r2_mean']
            metrics['r2_median'] = r2_results['r2_median']
            metrics['r2_std'] = r2_results['r2_std']
    except Exception as e:
        print(f"Warning: Could not compute R² per gene: {e}")
        metrics['r2_mean'] = float('nan')
        metrics['r2_median'] = float('nan')
        metrics['r2_std'] = float('nan')
    
    # JS Divergence
    try:
        metrics['js_divergence'] = compute_js_divergence(real, generated)
    except Exception as e:
        print(f"Warning: Could not compute JS divergence: {e}")
        metrics['js_divergence'] = float('nan')
    
    # Correlation Structure
    try:
        corr_results = compute_correlation_structure_similarity(real, generated)
        metrics.update(corr_results)
    except Exception as e:
        print(f"Warning: Could not compute correlation structure: {e}")
        metrics['correlation_frobenius_diff'] = float('nan')
        metrics['correlation_structure_corr'] = float('nan')
    
    return metrics
