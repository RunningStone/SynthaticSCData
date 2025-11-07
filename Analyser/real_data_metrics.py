#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Metrics for Evaluating Generated vs Test Data
Includes Frechet Distance, MAE, PCC, and Entropy
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import linalg
from scipy.stats import pearsonr
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, 
                               mu2: np.ndarray, sigma2: np.ndarray,
                               eps: float = 1e-6) -> float:
    """
    Calculate Frechet Distance between two multivariate Gaussians
    
    FD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    
    Args:
        mu1: Mean of first distribution (n_features,)
        sigma1: Covariance of first distribution (n_features, n_features)
        mu2: Mean of second distribution (n_features,)
        sigma2: Covariance of second distribution (n_features, n_features)
        eps: Small value for numerical stability
    
    Returns:
        Frechet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        print("Warning: FID calculation produces singular product; adding epsilon to diagonal")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    return float(fd)


def calculate_statistics(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and covariance of data
    
    Args:
        X: Data matrix (n_samples, n_features)
    
    Returns:
        (mu, sigma) - mean vector and covariance matrix
    """
    mu = np.mean(X, axis=0)
    sigma = np.cov(X, rowvar=False)
    return mu, sigma


def calculate_mae(X_true: np.ndarray, X_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error
    
    Args:
        X_true: True data (n_samples, n_features)
        X_pred: Predicted data (n_samples, n_features)
    
    Returns:
        MAE value
    """
    return np.mean(np.abs(X_true - X_pred))


def calculate_pcc(X_true: np.ndarray, X_pred: np.ndarray) -> Tuple[float, float]:
    """
    Calculate average Pearson Correlation Coefficient across features
    
    Args:
        X_true: True data (n_samples, n_features)
        X_pred: Predicted data (n_samples, n_features)
    
    Returns:
        (mean_pcc, std_pcc) - mean and std of PCC across features
    """
    n_features = X_true.shape[1]
    pccs = []
    
    for i in range(n_features):
        if np.std(X_true[:, i]) > 0 and np.std(X_pred[:, i]) > 0:
            corr, _ = pearsonr(X_true[:, i], X_pred[:, i])
            if not np.isnan(corr):
                pccs.append(corr)
    
    if len(pccs) == 0:
        return 0.0, 0.0
    
    return np.mean(pccs), np.std(pccs)


def evaluate_generated_vs_test(
    X_test: np.ndarray,
    X_generated: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate generated data against test data
    
    Args:
        X_test: Test data (n_test, n_features)
        X_generated: Generated data (n_gen, n_features)
        verbose: Print results
    
    Returns:
        Dictionary with metrics
    """
    # Calculate statistics
    mu_test, sigma_test = calculate_statistics(X_test)
    mu_gen, sigma_gen = calculate_statistics(X_generated)
    
    # Frechet Distance
    fd = calculate_frechet_distance(mu_test, sigma_test, mu_gen, sigma_gen)
    
    # For MAE and PCC, we need paired samples
    # Use minimum number of samples
    n_samples = min(X_test.shape[0], X_generated.shape[0])
    
    # Randomly sample if needed
    if X_test.shape[0] > n_samples:
        idx_test = np.random.choice(X_test.shape[0], n_samples, replace=False)
        X_test_sampled = X_test[idx_test]
    else:
        X_test_sampled = X_test
    
    if X_generated.shape[0] > n_samples:
        idx_gen = np.random.choice(X_generated.shape[0], n_samples, replace=False)
        X_gen_sampled = X_generated[idx_gen]
    else:
        X_gen_sampled = X_generated
    
    # MAE
    mae = calculate_mae(X_test_sampled, X_gen_sampled)
    
    # PCC
    mean_pcc, std_pcc = calculate_pcc(X_test_sampled, X_gen_sampled)
    
    results = {
        'frechet_distance': fd,
        'mae': mae,
        'mean_pcc': mean_pcc,
        'std_pcc': std_pcc,
        'n_test_samples': X_test.shape[0],
        'n_generated_samples': X_generated.shape[0],
        'n_features': X_test.shape[1]
    }
    
    if verbose:
        print("\n" + "="*50)
        print("Evaluation Metrics: Generated vs Test Data")
        print("="*50)
        print(f"Frechet Distance: {fd:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Mean PCC: {mean_pcc:.4f} ± {std_pcc:.4f}")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Generated samples: {X_generated.shape[0]}")
        print(f"Features: {X_test.shape[1]}")
        print("="*50)
    
    return results


def evaluate_model_on_dataset(
    adata_continuous: sc.AnnData,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    generated_indices: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model by comparing generated data with test data
    
    Args:
        adata_continuous: AnnData with continuous time data
        train_indices: Indices of training data
        test_indices: Indices of test data
        generated_indices: Indices of generated data (if None, use non-real cells)
        verbose: Print results
    
    Returns:
        Dictionary with metrics
    """
    # Get expression data
    X = adata_continuous.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Get test data
    X_test = X[test_indices]
    
    # Get generated data
    if generated_indices is None:
        # Use all generated (non-real) cells
        is_real = adata_continuous.obs['is_real'].values
        generated_mask = ~is_real
        X_generated = X[generated_mask]
    else:
        X_generated = X[generated_indices]
    
    # Evaluate
    results = evaluate_generated_vs_test(X_test, X_generated, verbose=verbose)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate generated vs test data')
    parser.add_argument('--input', type=str, required=True, help='Input h5ad file')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test ratio')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    adata = sc.read_h5ad(args.input)
    
    # Simple train/test split
    n_samples = adata.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_test = int(n_samples * args.test_ratio)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Evaluate
    results = evaluate_model_on_dataset(
        adata,
        train_indices=train_indices,
        test_indices=test_indices,
        verbose=True
    )
