#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 6: Generate Linearly Interpolated Data
Generate synthetic intermediate time points by linear interpolation between boundary points
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def generate_linear_interpolated_data(
    adata_full: sc.AnnData,
    boundary_timepoints: List[str] = ["0d", "7d"],
    intermediate_timepoints: List[str] = ["8h", "1d", "3d"],
    time_column: str = "Ground_truth",
    n_samples_per_timepoint: int = 750,
    time_to_hours: Dict[str, float] = None,
    random_seed: int = 42
) -> sc.AnnData:
    """
    Generate linearly interpolated data between boundary time points.
    
    This function creates synthetic intermediate states by linear interpolation
    between boundary cells, testing whether geometric interpolation can replace
    real intermediate observations.
    
    Args:
        adata_full: Full AnnData object with all time points
        boundary_timepoints: List of boundary time point labels (e.g., ["0d", "7d"])
        intermediate_timepoints: List of intermediate time point labels to generate
        time_column: Column name in obs containing time labels
        n_samples_per_timepoint: Number of cells to generate per time point
        time_to_hours: Dictionary mapping time labels to hours (for interpolation weights)
        random_seed: Random seed for reproducibility
        
    Returns:
        New AnnData object with boundary (real) + intermediate (interpolated) data
    """
    np.random.seed(random_seed)
    
    # Default time mapping if not provided
    if time_to_hours is None:
        time_to_hours = {
            "0d": 0.0,
            "8h": 8.0,
            "1d": 24.0,
            "3d": 72.0,
            "7d": 168.0
        }
    
    print("="*70)
    print("Generating Linearly Interpolated Data")
    print("="*70)
    print(f"Boundary timepoints: {boundary_timepoints}")
    print(f"Intermediate timepoints to generate: {intermediate_timepoints}")
    print(f"Samples per timepoint: {n_samples_per_timepoint}")
    
    # Extract boundary data
    t0_label, tn_label = boundary_timepoints[0], boundary_timepoints[1]
    t0_hours = time_to_hours[t0_label]
    tn_hours = time_to_hours[tn_label]
    
    mask_t0 = adata_full.obs[time_column] == t0_label
    mask_tn = adata_full.obs[time_column] == tn_label
    
    X_t0 = adata_full[mask_t0].X
    X_tn = adata_full[mask_tn].X
    
    # Convert sparse to dense if needed
    if hasattr(X_t0, 'toarray'):
        X_t0 = X_t0.toarray()
    if hasattr(X_tn, 'toarray'):
        X_tn = X_tn.toarray()
    
    print(f"\n✓ Extracted boundary data:")
    print(f"  - {t0_label}: {X_t0.shape[0]} cells")
    print(f"  - {tn_label}: {X_tn.shape[0]} cells")
    
    # Prepare containers for interpolated data
    all_X = []
    all_obs = []
    
    # Add boundary data (real)
    for label, X, mask in [(t0_label, X_t0, mask_t0), (tn_label, X_tn, mask_tn)]:
        # Sample n_samples_per_timepoint cells
        n_available = X.shape[0]
        if n_available > n_samples_per_timepoint:
            indices = np.random.choice(n_available, n_samples_per_timepoint, replace=False)
            X_sampled = X[indices]
        else:
            X_sampled = X
        
        all_X.append(X_sampled)
        
        # Create obs entries
        obs_subset = adata_full.obs[mask].iloc[:X_sampled.shape[0]].copy()
        obs_subset['data_source'] = 'real'
        all_obs.append(obs_subset)
    
    print(f"\n✓ Added boundary data (real):")
    print(f"  - {t0_label}: {all_X[0].shape[0]} cells")
    print(f"  - {tn_label}: {all_X[1].shape[0]} cells")
    
    # Generate interpolated intermediate data
    print(f"\n✓ Generating interpolated intermediate data:")
    
    for tk_label in intermediate_timepoints:
        tk_hours = time_to_hours[tk_label]
        
        # Compute interpolation weight: λ_k = (t_n - t_k) / (t_n - t_0)
        lambda_k = (tn_hours - tk_hours) / (tn_hours - t0_hours)
        
        print(f"  - {tk_label} (t={tk_hours}h): λ={lambda_k:.3f}")
        
        # Random pairing between boundary cells
        n_pairs = n_samples_per_timepoint
        n_t0 = X_t0.shape[0]
        n_tn = X_tn.shape[0]
        
        idx_t0 = np.random.choice(n_t0, n_pairs, replace=True)
        idx_tn = np.random.choice(n_tn, n_pairs, replace=True)
        
        # Linear interpolation: x_k = λ_k * x_0 + (1 - λ_k) * x_n
        X_tk = lambda_k * X_t0[idx_t0] + (1 - lambda_k) * X_tn[idx_tn]
        
        all_X.append(X_tk)
        
        # Create obs entries for interpolated data
        obs_tk = pd.DataFrame({
            time_column: [tk_label] * n_pairs,
            'data_source': ['interpolated'] * n_pairs,
            'interpolation_weight': [lambda_k] * n_pairs,
            'source_t0_idx': idx_t0,
            'source_tn_idx': idx_tn
        })
        all_obs.append(obs_tk)
        
        print(f"    Generated {X_tk.shape[0]} interpolated cells")
    
    # Combine all data
    X_combined = np.vstack(all_X)
    obs_combined = pd.concat(all_obs, ignore_index=True)
    
    # Create new AnnData object
    adata_interpolated = sc.AnnData(
        X=X_combined,
        obs=obs_combined,
        var=adata_full.var.copy()
    )
    
    print(f"\n" + "="*70)
    print(f"✓ Generated interpolated dataset:")
    print(f"  - Total cells: {adata_interpolated.shape[0]}")
    print(f"  - Total genes: {adata_interpolated.shape[1]}")
    print(f"  - Time points: {sorted(adata_interpolated.obs[time_column].unique())}")
    print(f"\nData source breakdown:")
    print(adata_interpolated.obs['data_source'].value_counts())
    print(f"\nCells per time point:")
    print(adata_interpolated.obs[time_column].value_counts().sort_index())
    print("="*70)
    
    return adata_interpolated


def main():
    """Main function to generate interpolated data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate linearly interpolated data')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input h5ad file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output h5ad file')
    parser.add_argument('--n_samples', type=int, default=750,
                        help='Number of samples per timepoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input}")
    adata = sc.read_h5ad(args.input)
    
    # Generate interpolated data
    adata_interp = generate_linear_interpolated_data(
        adata_full=adata,
        boundary_timepoints=["0d", "7d"],
        intermediate_timepoints=["8h", "1d", "3d"],
        time_column="Ground_truth",
        n_samples_per_timepoint=args.n_samples,
        random_seed=args.seed
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_interp.write_h5ad(output_path)
    print(f"\n✓ Saved interpolated data to: {output_path}")


if __name__ == "__main__":
    main()
