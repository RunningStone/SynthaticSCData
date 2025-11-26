#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolated Data Loader for Experiment 6

Extends BaseDataLoader to generate linearly interpolated intermediate states
between boundary timepoints.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from typing import List, Dict, Optional

from .base_data_loader import BaseDataLoader


class InterpolatedDataLoader(BaseDataLoader):
    """
    Data loader that generates linearly interpolated intermediate time points.
    
    Inherits from BaseDataLoader and adds interpolation functionality to create
    synthetic intermediate states between boundary timepoints. This is useful for
    testing model generalization to unseen intermediate states.
    
    Key features:
    - Inherits all standard data loading from BaseDataLoader
    - Generates linearly interpolated intermediate timepoints
    - Preserves real boundary data
    - Marks interpolated vs real data in obs
    """
    
    def __init__(
        self,
        file_path: str,
        n_hvg: int = 100,
        obs_time_column: str = 'Ground_truth',
        time_labels: Optional[List[str]] = None,
        time_label_order: Optional[List[str]] = None,
        biology_split: Optional[Dict] = None,
        random_seed: int = 42,
        interpolation_params: Optional[Dict] = None
    ):
        """
        Initialize interpolated data loader.
        
        Args:
            file_path: Path to h5ad file
            n_hvg: Number of highly variable genes
            obs_time_column: Column name for time labels in obs
            time_labels: List of time labels to use
            time_label_order: Order of time labels
            biology_split: Dictionary for train/test split
            random_seed: Random seed
            interpolation_params: Dictionary with:
                - boundary_timepoints: List of boundary time labels (e.g., ["0d", "7d"])
                - intermediate_timepoints: List of intermediate time labels to interpolate
                - n_samples_per_timepoint: Number of samples per timepoint
                - time_to_hours: Dictionary mapping time labels to hours
        """
        # Initialize parent class
        super().__init__(
            file_path=file_path,
            n_hvg=n_hvg,
            obs_time_column=obs_time_column,
            time_labels=time_labels,
            time_label_order=time_label_order,
            biology_split=biology_split,
            random_seed=random_seed
        )
        
        self.interpolation_params = interpolation_params or {}
        self.adata_interpolated = None
    
    def load_and_analyze(self):
        """
        Load data and generate interpolated intermediate states.
        
        Overrides parent method to add interpolation step after standard loading.
        """
        # First, load the original data using parent method
        super().load_and_analyze()
        
        # Then, generate interpolated data if parameters are provided
        if self.interpolation_params:
            print("\n" + "="*70)
            print("Generating Interpolated Data")
            print("="*70)
            
            self._generate_interpolated_data()
            
            # Replace adata_hvg with interpolated version
            self.adata_hvg = self.adata_interpolated
            
            # Recreate split masks for the new data
            self._create_split_masks()
            
            print("\n" + "="*70)
            print("Interpolated Data Loading Complete")
            print("="*70)
    
    def _generate_interpolated_data(self):
        """
        Generate linearly interpolated intermediate time points.
        
        For each intermediate timepoint t_k between boundaries t_0 and t_n:
        - Compute interpolation weight: λ_k = (t_n - t_k) / (t_n - t_0)
        - Generate synthetic cells: x_k = λ_k * x_0 + (1 - λ_k) * x_n
        - Random pairing between boundary cells
        """
        # Extract parameters
        boundary_timepoints = self.interpolation_params.get(
            'boundary_timepoints', ["0d", "7d"]
        )
        intermediate_timepoints = self.interpolation_params.get(
            'intermediate_timepoints', ["8h", "1d", "3d"]
        )
        n_samples_per_timepoint = self.interpolation_params.get(
            'n_samples_per_timepoint', 750
        )
        time_to_hours = self.interpolation_params.get('time_to_hours', {
            "0d": 0.0, "8h": 8.0, "1d": 24.0, "3d": 72.0, "7d": 168.0
        })
        
        print(f"Boundary timepoints: {boundary_timepoints}")
        print(f"Intermediate timepoints to generate: {intermediate_timepoints}")
        print(f"Samples per timepoint: {n_samples_per_timepoint}")
        
        # Extract boundary data from HVG-filtered data
        t0_label, tn_label = boundary_timepoints[0], boundary_timepoints[1]
        t0_hours = time_to_hours[t0_label]
        tn_hours = time_to_hours[tn_label]
        
        mask_t0 = self.adata_hvg.obs[self.obs_time_column] == t0_label
        mask_tn = self.adata_hvg.obs[self.obs_time_column] == tn_label
        
        X_t0 = self.adata_hvg[mask_t0].X
        X_tn = self.adata_hvg[mask_tn].X
        
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
                indices = np.random.choice(
                    n_available, n_samples_per_timepoint, replace=False
                )
                X_sampled = X[indices]
                obs_indices = np.where(mask)[0][indices]
            else:
                X_sampled = X
                obs_indices = np.where(mask)[0][:X_sampled.shape[0]]
            
            all_X.append(X_sampled)
            
            # Create obs entries - preserve original obs columns
            obs_subset = self.adata_hvg.obs.iloc[obs_indices].copy()
            obs_subset['data_source'] = 'real'
            obs_subset['interpolation_weight'] = np.nan
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
            # Use the first real cell's obs as template, but modify key columns
            template_obs = self.adata_hvg.obs.iloc[0:1].copy()
            obs_tk = pd.concat([template_obs] * n_pairs, ignore_index=True)
            
            # Update key columns
            obs_tk[self.obs_time_column] = tk_label
            obs_tk['data_source'] = 'interpolated'
            obs_tk['interpolation_weight'] = lambda_k
            obs_tk['source_t0_idx'] = idx_t0
            obs_tk['source_tn_idx'] = idx_tn
            
            all_obs.append(obs_tk)
            
            print(f"    Generated {X_tk.shape[0]} interpolated cells")
        
        # Combine all data
        X_combined = np.vstack(all_X)
        obs_combined = pd.concat(all_obs, ignore_index=True)
        
        # Create new AnnData object
        self.adata_interpolated = sc.AnnData(
            X=X_combined,
            obs=obs_combined,
            var=self.adata_hvg.var.copy()
        )
        
        print(f"\n✓ Generated interpolated dataset:")
        print(f"  - Total cells: {self.adata_interpolated.shape[0]}")
        print(f"  - Total genes: {self.adata_interpolated.shape[1]}")
        print(f"  - Time points: {sorted(self.adata_interpolated.obs[self.obs_time_column].unique())}")
        print(f"\nData source breakdown:")
        print(self.adata_interpolated.obs['data_source'].value_counts())
        print(f"\nCells per time point:")
        print(self.adata_interpolated.obs[self.obs_time_column].value_counts().sort_index())
