#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy Evolution Analysis for Cell State Transitions

Analyzes whether models can reproduce the non-monotonic entropy evolution
(entropy increase → entropy decrease) observed in real EMT trajectories.

This is the most direct test of the core hypothesis.

Author: Generated for Experiment 7
Date: 2024-11
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .entropy_estimators import (
    estimate_entropy_knn,
    estimate_entropy_gaussian,
    estimate_entropy_both_methods
)


def compute_entropy_curve(
    model: torch.nn.Module,
    initial_states: torch.Tensor,
    time_grid: torch.Tensor,
    time_labels: List[str],
    method: str = 'knn',
    k: int = 5,
    device: str = 'cuda',
    verbose: bool = True
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute entropy curve along generated trajectory.
    
    This function:
    1. Generates complete trajectory from initial states
    2. Computes entropy at each time point
    3. Returns entropy curve and trajectory snapshots
    
    Args:
        model: Trained SB model with generate_trajectory method
        initial_states: Initial cell states (N, d) from test set t0
        time_grid: Normalized time points [0, t1, ..., 1]
        time_labels: Human-readable labels ['0d', '8h', ..., '7d']
        method: 'knn', 'gaussian', or 'both'
        k: Number of neighbors for KNN method
        device: 'cuda' or 'cpu'
        verbose: Show progress bar
    
    Returns:
        Tuple of:
            - entropy_curve: Array of shape (K,) with entropy at each time
            - trajectory_list: List of K arrays, each (N, d) for states at each time
    
    Example:
        >>> model.load_state_dict(torch.load('best_model.pt'))
        >>> x0 = test_data[test_data.obs['timepoint'] == '0d'].X
        >>> time_grid = torch.linspace(0, 1, 5)
        >>> H_curve, trajectory = compute_entropy_curve(
        ...     model, x0, time_grid, ['0d', '8h', '1d', '3d', '7d']
        ... )
    """
    model.eval()
    model = model.to(device)
    initial_states = initial_states.to(device)
    time_grid = time_grid.to(device)
    
    N = initial_states.shape[0]
    K = len(time_grid)
    
    # Generate complete trajectory
    # Use deterministic integration (no noise) for stable entropy estimation
    if verbose:
        print(f"Generating trajectory for {N} cells across {K} time points...")
    
    # Note: Don't use torch.no_grad() here because SB model needs gradients for drift computation
    # SB model's generate_trajectory returns (batch_size, n_time, d)
    trajectory_tensor = model.generate_trajectory(
        initial_states,
        time_grid,
        method='deterministic'  # No noise for reproducibility
    )
    
    # Convert to list of numpy arrays (one per time point)
    trajectory_list = []
    for j in range(K):
        X_t = trajectory_tensor[:, j, :].detach().cpu().numpy()
        trajectory_list.append(X_t)
    
    # Compute entropy at each time point
    entropy_curve = []
    
    iterator = enumerate(trajectory_list)
    if verbose:
        iterator = tqdm(iterator, total=K, desc="Computing entropy")
    
    for j, X_t in iterator:
        if method == 'knn':
            H_t = estimate_entropy_knn(X_t, k=k)
        elif method == 'gaussian':
            H_t = estimate_entropy_gaussian(X_t, shrinkage=True)
        elif method == 'both':
            H_knn, H_gauss, H_avg = estimate_entropy_both_methods(X_t, k=k)
            H_t = H_avg
            if verbose and j == 0:
                print(f"  Using average of KNN and Gaussian methods")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        entropy_curve.append(H_t)
    
    return np.array(entropy_curve), trajectory_list


def compute_entropy_curve_from_real_data(
    test_data,  # AnnData object
    time_column: str,
    time_labels: List[str],
    n_samples: int = 1000,
    method: str = 'knn',
    k: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute entropy curve from real data snapshots.
    
    This provides the ground truth entropy evolution for comparison.
    
    Args:
        test_data: AnnData object with test set
        time_column: Name of obs column with time labels
        time_labels: Ordered list of time labels to analyze
        n_samples: Number of cells to sample per time point
        method: Entropy estimation method
        k: KNN parameter
        random_state: Random seed for reproducibility
        verbose: Show progress
    
    Returns:
        Tuple of (entropy_curve, data_list)
    """
    import scanpy as sc
    
    np.random.seed(random_state)
    
    data_list = []
    entropy_curve = []
    
    iterator = time_labels
    if verbose:
        iterator = tqdm(time_labels, desc="Processing real data")
    
    for time_label in iterator:
        # Extract cells at this time point
        mask = test_data.obs[time_column] == time_label
        cells_at_t = test_data[mask]
        
        # Sample if we have more than n_samples
        if cells_at_t.n_obs > n_samples:
            indices = np.random.choice(cells_at_t.n_obs, n_samples, replace=False)
            cells_at_t = cells_at_t[indices]
        
        # Get expression matrix
        if hasattr(cells_at_t.X, 'toarray'):
            X_t = cells_at_t.X.toarray()
        else:
            X_t = cells_at_t.X
        
        data_list.append(X_t)
        
        # Compute entropy
        if method == 'knn':
            H_t = estimate_entropy_knn(X_t, k=k)
        elif method == 'gaussian':
            H_t = estimate_entropy_gaussian(X_t, shrinkage=True)
        elif method == 'both':
            _, _, H_t = estimate_entropy_both_methods(X_t, k=k)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        entropy_curve.append(H_t)
    
    return np.array(entropy_curve), data_list


def analyze_entropy_peak(
    entropy_curve: np.ndarray,
    time_labels: List[str]
) -> Dict:
    """
    Analyze entropy peak characteristics.
    
    Detects:
    1. Peak position and value
    2. Peak amplitude
    3. Non-monotonicity (inverted-U shape)
    4. Entropy change rates (explore vs. collapse)
    
    Args:
        entropy_curve: Array of entropy values at each time point
        time_labels: Time labels corresponding to entropy values
    
    Returns:
        Dictionary with analysis results:
            - peak_idx: Index of maximum entropy
            - peak_time: Time label at peak
            - peak_value: Entropy value at peak
            - amplitude: Peak height above boundary minimum
            - is_nonmonotonic: Whether curve has inverted-U shape
            - explore_rate: Entropy increase rate (0 → peak)
            - collapse_rate: Entropy decrease rate (peak → end)
            - asymmetry: Ratio of explore to collapse rate
    """
    # Find peak
    peak_idx = int(np.argmax(entropy_curve))
    peak_time = time_labels[peak_idx]
    peak_value = float(entropy_curve[peak_idx])
    
    # Compute amplitude (peak height above boundary minimum)
    boundary_min = min(entropy_curve[0], entropy_curve[-1])
    amplitude = peak_value - boundary_min
    
    # Check non-monotonicity (inverted-U)
    is_nonmonotonic = (
        (peak_value > entropy_curve[0]) and 
        (peak_value > entropy_curve[-1])
    )
    
    # Compute entropy change rates
    if peak_idx > 0:
        explore_rate = (peak_value - entropy_curve[0]) / peak_idx
    else:
        explore_rate = 0.0
    
    if peak_idx < len(entropy_curve) - 1:
        collapse_rate = (entropy_curve[-1] - peak_value) / (len(entropy_curve) - peak_idx - 1)
    else:
        collapse_rate = 0.0
    
    # Asymmetry: ratio of rates
    if abs(collapse_rate) > 1e-10:
        asymmetry = abs(explore_rate) / abs(collapse_rate)
    else:
        asymmetry = float('inf')
    
    return {
        'peak_idx': peak_idx,
        'peak_time': peak_time,
        'peak_value': peak_value,
        'amplitude': amplitude,
        'is_nonmonotonic': is_nonmonotonic,
        'explore_rate': explore_rate,
        'collapse_rate': collapse_rate,
        'asymmetry': asymmetry
    }


def compute_entropy_curve_similarity(
    H_real: np.ndarray,
    H_gen: np.ndarray,
    metric: str = 'mse'
) -> float:
    """
    Compute similarity between real and generated entropy curves.
    
    Args:
        H_real: Real entropy curve
        H_gen: Generated entropy curve
        metric: 'mse' (mean squared error) or 'dtw' (dynamic time warping)
    
    Returns:
        Similarity score (lower is better for MSE)
    """
    if metric == 'mse':
        return float(np.mean((H_real - H_gen) ** 2))
    
    elif metric == 'dtw':
        try:
            from dtaidistance import dtw
            distance = dtw.distance(H_real, H_gen)
            return float(distance)
        except ImportError:
            print("Warning: dtaidistance not installed, falling back to MSE")
            return float(np.mean((H_real - H_gen) ** 2))
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compare_multiple_models(
    models_dict: Dict[str, torch.nn.Module],
    initial_states: torch.Tensor,
    time_grid: torch.Tensor,
    time_labels: List[str],
    real_entropy_curve: Optional[np.ndarray] = None,
    method: str = 'knn',
    k: int = 5,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Compare entropy evolution across multiple models.
    
    Args:
        models_dict: Dictionary mapping model names to model objects
        initial_states: Shared initial states for all models
        time_grid: Time grid for trajectory generation
        time_labels: Time labels
        real_entropy_curve: Optional ground truth for comparison
        method: Entropy estimation method
        k: KNN parameter
        device: Computing device
        verbose: Show progress
    
    Returns:
        Dictionary mapping model names to results:
            - entropy_curve: Entropy values
            - peak_analysis: Peak characteristics
            - similarity_to_real: (if real_entropy_curve provided)
    """
    results = {}
    
    for model_name, model in models_dict.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"Analyzing model: {model_name}")
            print(f"{'='*70}")
        
        # Compute entropy curve
        entropy_curve, trajectory = compute_entropy_curve(
            model=model,
            initial_states=initial_states,
            time_grid=time_grid,
            time_labels=time_labels,
            method=method,
            k=k,
            device=device,
            verbose=verbose
        )
        
        # Analyze peak
        peak_analysis = analyze_entropy_peak(entropy_curve, time_labels)
        
        # Prepare results
        model_results = {
            'entropy_curve': entropy_curve,
            'peak_analysis': peak_analysis,
            'trajectory': trajectory
        }
        
        # Compare to real if provided
        if real_entropy_curve is not None:
            similarity = compute_entropy_curve_similarity(real_entropy_curve, entropy_curve)
            model_results['similarity_to_real'] = similarity
            
            if verbose:
                print(f"\nSimilarity to real data (MSE): {similarity:.4f}")
        
        # Print peak analysis
        if verbose:
            print(f"\nPeak Analysis:")
            print(f"  Peak time: {peak_analysis['peak_time']}")
            print(f"  Peak value: {peak_analysis['peak_value']:.4f}")
            print(f"  Amplitude: {peak_analysis['amplitude']:.4f}")
            print(f"  Non-monotonic: {peak_analysis['is_nonmonotonic']}")
            print(f"  Explore rate: {peak_analysis['explore_rate']:.4f}")
            print(f"  Collapse rate: {peak_analysis['collapse_rate']:.4f}")
            print(f"  Asymmetry: {peak_analysis['asymmetry']:.4f}")
        
        results[model_name] = model_results
    
    return results


# Example usage and unit test
if __name__ == "__main__":
    print("="*70)
    print("Testing Entropy Evolution Analysis")
    print("="*70)
    
    # Create a simple test: generate synthetic trajectory with known entropy evolution
    print("\nTest 1: Synthetic trajectory with inverted-U entropy")
    
    # Simulate entropy evolution: low → high → low
    time_labels = ['t0', 't1', 't2', 't3', 't4']
    real_entropy = np.array([5.0, 7.0, 9.0, 7.5, 5.5])  # Inverted-U
    
    # Analyze peak
    peak_analysis = analyze_entropy_peak(real_entropy, time_labels)
    
    print(f"  Entropy curve: {real_entropy}")
    print(f"  Peak at: {peak_analysis['peak_time']} (index {peak_analysis['peak_idx']})")
    print(f"  Peak value: {peak_analysis['peak_value']:.2f}")
    print(f"  Amplitude: {peak_analysis['amplitude']:.2f}")
    print(f"  Non-monotonic: {peak_analysis['is_nonmonotonic']}")
    print(f"  Explore rate: {peak_analysis['explore_rate']:.2f}")
    print(f"  Collapse rate: {peak_analysis['collapse_rate']:.2f}")
    
    # Test 2: Compare with monotonic curve
    print("\nTest 2: Monotonic curve (should fail non-monotonicity test)")
    monotonic_entropy = np.array([5.0, 5.5, 6.0, 6.5, 7.0])  # Monotonic increase
    
    peak_analysis_mono = analyze_entropy_peak(monotonic_entropy, time_labels)
    print(f"  Entropy curve: {monotonic_entropy}")
    print(f"  Non-monotonic: {peak_analysis_mono['is_nonmonotonic']}")
    
    # Test 3: Curve similarity
    print("\nTest 3: Curve similarity")
    gen_entropy_good = np.array([5.1, 6.9, 8.8, 7.6, 5.4])  # Close to real
    gen_entropy_bad = np.array([5.0, 5.5, 6.0, 6.5, 7.0])   # Monotonic
    
    mse_good = compute_entropy_curve_similarity(real_entropy, gen_entropy_good, metric='mse')
    mse_bad = compute_entropy_curve_similarity(real_entropy, gen_entropy_bad, metric='mse')
    
    print(f"  MSE (good match): {mse_good:.4f}")
    print(f"  MSE (poor match): {mse_bad:.4f}")
    print(f"  Good match is better: {mse_good < mse_bad}")
    
    print("\n" + "="*70)
    print("✓ All tests completed successfully!")
    print("="*70)
