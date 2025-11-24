#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy Estimators for High-Dimensional Single-Cell Data

Implements two complementary non-parametric entropy estimation methods:
1. K-Nearest Neighbors (KNN) - Kozachenko-Leonenko estimator
2. Multivariate Gaussian with Ledoit-Wolf shrinkage

Author: Generated for Experiment 7
Date: 2024-11
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from scipy.special import digamma, gammaln
from typing import Union, Tuple


def estimate_entropy_knn(
    X: Union[np.ndarray, torch.Tensor],
    k: int = 5,
    normalize: bool = False
) -> float:
    """
    K-Nearest Neighbors entropy estimation using Kozachenko-Leonenko estimator.
    
    This method is non-parametric and works well for arbitrary distributions
    in high dimensions without assuming a specific functional form.
    
    Formula:
        H_KNN = (d/N) * Σ log(ρ_k(x_i)) + log(N-1) - ψ(k) + log(c_d)
    
    where:
        - ρ_k(x_i): distance to k-th nearest neighbor of x_i
        - ψ: digamma function
        - c_d: volume constant for d-dimensional unit ball
        - N: number of samples
        - d: dimension
    
    Args:
        X: Data matrix of shape (N, d) where N is samples, d is dimensions
        k: Number of nearest neighbors (default: 5 for bias-variance balance)
        normalize: If True, normalize data to zero mean and unit variance
    
    Returns:
        Estimated differential entropy in nats (natural log units)
    
    References:
        Kozachenko, L. F. & Leonenko, N. N. (1987). Sample estimate of the 
        entropy of a random vector. Problems of Information Transmission.
    """
    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    
    N, d = X.shape
    
    if N <= k:
        raise ValueError(f"Sample size N={N} must be larger than k={k}")
    
    # Optional normalization
    if normalize:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    
    # Compute k-nearest neighbor distances
    # Use k+1 because the first neighbor is the point itself
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='euclidean')
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # Extract k-th nearest neighbor distance (index k, since index 0 is self)
    rho_k = distances[:, k]
    
    # Avoid log(0) for identical points
    rho_k = np.maximum(rho_k, 1e-10)
    
    # Compute volume constant: c_d = π^(d/2) / Γ(d/2 + 1)
    # Using log to avoid numerical overflow
    log_c_d = (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1)
    
    # Kozachenko-Leonenko estimator
    H_knn = (d / N) * np.sum(np.log(rho_k)) + np.log(N - 1) - digamma(k) + log_c_d
    
    return float(H_knn)


def estimate_entropy_gaussian(
    X: Union[np.ndarray, torch.Tensor],
    shrinkage: bool = True,
    regularization: float = 1e-6
) -> float:
    """
    Multivariate Gaussian entropy estimation with Ledoit-Wolf shrinkage.
    
    Assumes data approximately follows a multivariate normal distribution.
    Uses shrinkage estimation to improve covariance matrix conditioning
    in high dimensions.
    
    Formula:
        H_Gauss = (d/2) * log(2πe) + (1/2) * log(det(Σ))
    
    where Σ is the shrinkage-estimated covariance matrix:
        Σ_shrink = (1-α)*Σ + α*(tr(Σ)/d)*I
    
    Args:
        X: Data matrix of shape (N, d)
        shrinkage: If True, use Ledoit-Wolf shrinkage (recommended for d large)
        regularization: Small constant added to diagonal for numerical stability
    
    Returns:
        Estimated differential entropy in nats
    
    References:
        Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for 
        large-dimensional covariance matrices. Journal of Multivariate Analysis.
    """
    # Convert to numpy if needed
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    
    N, d = X.shape
    
    if N < 2:
        raise ValueError(f"Need at least 2 samples, got N={N}")
    
    if shrinkage and N > d:
        # Use Ledoit-Wolf shrinkage estimator
        lw = LedoitWolf()
        lw.fit(X)
        Sigma = lw.covariance_
    else:
        # Simple sample covariance
        Sigma = np.cov(X, rowvar=False)
        
        # Manual shrinkage if N <= d
        if N <= d:
            trace_Sigma = np.trace(Sigma)
            alpha = 0.5  # Fixed shrinkage intensity
            Sigma = (1 - alpha) * Sigma + alpha * (trace_Sigma / d) * np.eye(d)
    
    # Add small regularization to diagonal for numerical stability
    Sigma = Sigma + regularization * np.eye(d)
    
    # Compute log determinant (numerically stable)
    sign, logdet = np.linalg.slogdet(Sigma)
    
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite")
    
    # Gaussian entropy formula
    H_gauss = 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * logdet
    
    return float(H_gauss)


def estimate_entropy_both_methods(
    X: Union[np.ndarray, torch.Tensor],
    k: int = 5,
    shrinkage: bool = True
) -> Tuple[float, float, float]:
    """
    Estimate entropy using both KNN and Gaussian methods for cross-validation.
    
    Args:
        X: Data matrix of shape (N, d)
        k: Number of nearest neighbors for KNN method
        shrinkage: Whether to use Ledoit-Wolf shrinkage for Gaussian method
    
    Returns:
        Tuple of (H_knn, H_gauss, H_avg) where:
            - H_knn: KNN entropy estimate
            - H_gauss: Gaussian entropy estimate
            - H_avg: Average of the two methods
    
    Usage:
        If both methods give similar values, the estimate is robust.
        If they differ significantly, the data may have strong non-Gaussian features.
    """
    H_knn = estimate_entropy_knn(X, k=k)
    H_gauss = estimate_entropy_gaussian(X, shrinkage=shrinkage)
    H_avg = (H_knn + H_gauss) / 2.0
    
    return H_knn, H_gauss, H_avg


def batch_estimate_entropy(
    X_list: list,
    method: str = 'knn',
    k: int = 5,
    shrinkage: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Estimate entropy for multiple datasets (e.g., different time points).
    
    Args:
        X_list: List of data matrices, each of shape (N_i, d)
        method: 'knn', 'gaussian', or 'both'
        k: KNN parameter
        shrinkage: Gaussian shrinkage parameter
        verbose: Print progress
    
    Returns:
        Array of entropy estimates for each dataset
    """
    entropies = []
    
    for i, X in enumerate(X_list):
        if verbose:
            print(f"Estimating entropy for dataset {i+1}/{len(X_list)}...")
        
        if method == 'knn':
            H = estimate_entropy_knn(X, k=k)
        elif method == 'gaussian':
            H = estimate_entropy_gaussian(X, shrinkage=shrinkage)
        elif method == 'both':
            H_knn, H_gauss, H_avg = estimate_entropy_both_methods(X, k=k, shrinkage=shrinkage)
            H = H_avg
            if verbose:
                print(f"  KNN: {H_knn:.4f}, Gaussian: {H_gauss:.4f}, Avg: {H_avg:.4f}")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        entropies.append(H)
    
    return np.array(entropies)


# Unit tests for known distributions
if __name__ == "__main__":
    print("="*70)
    print("Testing Entropy Estimators on Known Distributions")
    print("="*70)
    
    # Test 1: Standard Gaussian in d dimensions
    # True entropy: H = (d/2) * log(2πe) = (d/2) * (1 + log(2π))
    print("\nTest 1: Standard Multivariate Gaussian")
    for d in [10, 50, 100]:
        N = 1000
        X = np.random.randn(N, d)
        
        H_true = 0.5 * d * (1 + np.log(2 * np.pi))
        H_knn = estimate_entropy_knn(X, k=5)
        H_gauss = estimate_entropy_gaussian(X, shrinkage=True)
        
        print(f"  d={d:3d}: True={H_true:.2f}, KNN={H_knn:.2f}, Gauss={H_gauss:.2f}")
        print(f"         Error: KNN={abs(H_knn-H_true)/H_true*100:.1f}%, "
              f"Gauss={abs(H_gauss-H_true)/H_true*100:.1f}%")
    
    # Test 2: Check consistency across sample sizes
    print("\nTest 2: Sample Size Sensitivity (d=50)")
    d = 50
    H_true = 0.5 * d * (1 + np.log(2 * np.pi))
    
    for N in [500, 1000, 2000]:
        X = np.random.randn(N, d)
        H_knn = estimate_entropy_knn(X, k=5)
        H_gauss = estimate_entropy_gaussian(X, shrinkage=True)
        
        print(f"  N={N:4d}: KNN={H_knn:.2f} (err={abs(H_knn-H_true)/H_true*100:.1f}%), "
              f"Gauss={H_gauss:.2f} (err={abs(H_gauss-H_true)/H_true*100:.1f}%)")
    
    # Test 3: Cross-validation between methods
    print("\nTest 3: Method Consistency")
    d = 100
    N = 1000
    X = np.random.randn(N, d)
    
    H_knn, H_gauss, H_avg = estimate_entropy_both_methods(X, k=5, shrinkage=True)
    H_true = 0.5 * d * (1 + np.log(2 * np.pi))
    
    print(f"  KNN:      {H_knn:.2f}")
    print(f"  Gaussian: {H_gauss:.2f}")
    print(f"  Average:  {H_avg:.2f}")
    print(f"  True:     {H_true:.2f}")
    print(f"  Relative difference: {abs(H_knn - H_gauss) / H_avg * 100:.1f}%")
    
    print("\n" + "="*70)
    print("✓ All tests completed successfully!")
    print("="*70)
