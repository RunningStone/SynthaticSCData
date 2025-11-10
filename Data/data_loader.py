#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Loader for Time Series Single-Cell Data
Loads real h5ad files and provides analysis and validation
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class RealDataLoader:
    """
    Load and analyze real single-cell time series data
    """
    
    def __init__(
        self,
        file_path: str,
        n_hvg: int = 100,
        obs_time_column: str = 'Ground_truth',
        time_labels: List[str] = None,
        time_label_order: List[str] = None,
        biology_split: Dict = None,
        random_seed: int = 42
    ):
        """
        Initialize real data loader
        
        Args:
            file_path: Path to h5ad file
            n_hvg: Number of highly variable genes
            obs_time_column: Column name for time labels in obs
            time_labels: List of time labels to use
            time_label_order: Order of time labels
            biology_split: Dictionary with keys:
                - "train_val_column": column name or "random"
                - "train": list of values for training
                - "test": list of values for testing
            random_seed: Random seed
        """
        self.file_path = file_path
        self.n_hvg = n_hvg
        self.obs_time_column = obs_time_column
        self.time_labels = time_labels or []
        self.time_label_order = time_label_order or time_labels or []
        self.biology_split = biology_split or {}
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        self.adata = None
        self.adata_hvg = None
        self.hvg_genes = None
        self.train_mask = None
        self.test_mask = None
        
    def load_and_analyze(self):
        """Load data and output obs column analysis"""
        print("="*70)
        print("Loading Real Data")
        print("="*70)
        print(f"File: {self.file_path}")
        
        self.adata = sc.read_h5ad(self.file_path)
        print(f"\nLoaded: {self.adata.shape[0]} cells × {self.adata.shape[1]} genes")
        
        # Analyze obs columns
        print("\n" + "="*70)
        print("Obs Columns Analysis")
        print("="*70)
        for col in self.adata.obs.columns:
            unique_vals = self.adata.obs[col].unique()
            n_unique = len(unique_vals)
            print(f"\n{col}:")
            print(f"  - Unique values: {n_unique}")
            if n_unique <= 20:
                print(f"  - Values: {sorted(unique_vals.tolist())}")
                # Show counts
                counts = self.adata.obs[col].value_counts()
                for val, count in counts.items():
                    print(f"    * {val}: {count} cells")
        
        # Filter to specified time labels
        if self.time_labels:
            mask = self.adata.obs[self.obs_time_column].isin(self.time_labels)
            self.adata = self.adata[mask].copy()
            print(f"\n✓ Filtered to {len(self.time_labels)} time labels: {self.adata.shape[0]} cells")
        
        # Select HVGs
        self._select_hvgs()
        
        # Create train/test masks
        self._create_split_masks()
        
        print("\n" + "="*70)
        print("Data Loading Complete")
        print("="*70)
        
    def _select_hvgs(self):
        """Select highly variable genes"""
        print(f"\nSelecting top {self.n_hvg} HVGs...")
        
        # Clean data
        X = self.adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        self.adata.X = X
        
        # Compute HVGs
        if 'highly_variable' not in self.adata.var.columns:
            try:
                sc.pp.highly_variable_genes(
                    self.adata,
                    n_top_genes=self.n_hvg,
                    flavor='seurat_v3',
                    subset=False
                )
            except (ImportError, ValueError):
                print("  seurat_v3 failed, using variance-based selection...")
                variances = np.var(X, axis=0)
                top_indices = np.argsort(variances)[-self.n_hvg:]
                self.adata.var['highly_variable'] = False
                self.adata.var.iloc[top_indices, self.adata.var.columns.get_loc('highly_variable')] = True
        
        # Get HVG genes
        if 'highly_variable' in self.adata.var.columns:
            hvg_mask = self.adata.var['highly_variable']
            self.hvg_genes = self.adata.var_names[hvg_mask][:self.n_hvg].tolist()
        else:
            self.hvg_genes = self.adata.var_names[:self.n_hvg].tolist()
        
        print(f"✓ Selected {len(self.hvg_genes)} HVGs")
        
        # Create HVG-filtered AnnData
        self.adata_hvg = self.adata[:, self.hvg_genes].copy()
        
    def _create_split_masks(self):
        """Create train/test split masks based on biology_split"""
        if not self.biology_split:
            print("\nWarning: No biology_split provided, using random 80/20 split")
            n_cells = self.adata_hvg.shape[0]
            indices = np.arange(n_cells)
            np.random.shuffle(indices)
            n_train = int(n_cells * 0.8)
            self.train_mask = np.zeros(n_cells, dtype=bool)
            self.train_mask[indices[:n_train]] = True
            self.test_mask = ~self.train_mask
            return
        
        train_val_column = self.biology_split.get("train_val_column", "random")
        
        if train_val_column == "random":
            # Random split
            n_cells = self.adata_hvg.shape[0]
            indices = np.arange(n_cells)
            np.random.shuffle(indices)
            n_train = int(n_cells * 0.8)
            self.train_mask = np.zeros(n_cells, dtype=bool)
            self.train_mask[indices[:n_train]] = True
            self.test_mask = ~self.train_mask
        else:
            # Biology-based split
            train_values = self.biology_split.get("train", [])
            test_values = self.biology_split.get("test", [])
            
            self.train_mask = self.adata_hvg.obs[train_val_column].isin(train_values).values
            self.test_mask = self.adata_hvg.obs[train_val_column].isin(test_values).values
        
        print(f"\n✓ Train cells: {self.train_mask.sum()}")
        print(f"✓ Test cells: {self.test_mask.sum()}")
        
    def validate_biology_split(self):
        """Validate that train and test sets contain all time point categories"""
        print("\n" + "="*70)
        print("Biology Split Validation")
        print("="*70)
        
        if self.train_mask is None or self.test_mask is None:
            print("Error: Split masks not created yet. Call load_and_analyze() first.")
            return False
        
        # Get time labels in train and test
        train_times = set(self.adata_hvg.obs.loc[self.train_mask, self.obs_time_column].unique())
        test_times = set(self.adata_hvg.obs.loc[self.test_mask, self.obs_time_column].unique())
        all_times = set(self.time_label_order)
        
        print(f"\nExpected time points: {sorted(all_times)}")
        print(f"Train time points: {sorted(train_times)}")
        print(f"Test time points: {sorted(test_times)}")
        
        # Check if all time points are present
        train_missing = all_times - train_times
        test_missing = all_times - test_times
        
        valid = True
        if train_missing:
            print(f"\n⚠️  WARNING: Train set missing time points: {sorted(train_missing)}")
            valid = False
        else:
            print("\n✓ Train set contains all time points")
        
        if test_missing:
            print(f"⚠️  WARNING: Test set missing time points: {sorted(test_missing)}")
            valid = False
        else:
            print("✓ Test set contains all time points")
        
        # Show cell counts per time point
        print("\nCell counts per time point:")
        print("-" * 50)
        for time_label in self.time_label_order:
            train_count = ((self.adata_hvg.obs[self.obs_time_column] == time_label) & self.train_mask).sum()
            test_count = ((self.adata_hvg.obs[self.obs_time_column] == time_label) & self.test_mask).sum()
            print(f"{time_label:>5s}: Train={train_count:>5d}, Test={test_count:>5d}")
        
        print("="*70)
        return valid
    
    def get_data_for_setting(
        self,
        setting: int = 1,
        cells_per_timepoint: int = 2000,
        total_cells: Optional[int] = None,
        balance_strategy: str = 'per_timepoint'
    ) -> tuple:
        """
        Get data for specific setting
        
        Args:
            setting: 1 for boundary only, 2 for all timepoints
            cells_per_timepoint: Number of cells per timepoint
            total_cells: Total number of cells (only used when balance_strategy='total')
            balance_strategy: 'per_timepoint' (default) or 'total'
                - 'per_timepoint': Each timepoint has cells_per_timepoint samples
                - 'total': Total samples = total_cells, evenly distributed across timepoints
            
        Returns:
            (X_train, y_train, X_test, y_test)
        """
        # Get HVG expression matrix
        X = self.adata_hvg.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Get time labels as integers
        time_to_idx = {label: idx for idx, label in enumerate(self.time_label_order)}
        y = np.array([time_to_idx[t] for t in self.adata_hvg.obs[self.obs_time_column]])
        
        if setting == 1:
            # Setting 1: Only first and last timepoints
            first_time = self.time_label_order[0]
            last_time = self.time_label_order[-1]
            selected_times = [first_time, last_time]
            
            X_train_list, y_train_list = [], []
            X_test_list, y_test_list = [], []
            
            for time_label in selected_times:
                time_idx = time_to_idx[time_label]
                time_mask = (y == time_idx)
                
                # Train
                train_time_mask = time_mask & self.train_mask
                train_indices = np.where(train_time_mask)[0]
                if cells_per_timepoint and len(train_indices) > cells_per_timepoint:
                    train_indices = np.random.choice(train_indices, cells_per_timepoint, replace=False)
                X_train_list.append(X[train_indices])
                y_train_list.append(y[train_indices])
                
                # Test
                test_time_mask = time_mask & self.test_mask
                test_indices = np.where(test_time_mask)[0]
                if cells_per_timepoint and len(test_indices) > cells_per_timepoint:
                    test_indices = np.random.choice(test_indices, cells_per_timepoint, replace=False)
                X_test_list.append(X[test_indices])
                y_test_list.append(y[test_indices])
            
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            X_test = np.vstack(X_test_list)
            y_test = np.concatenate(y_test_list)
            
        else:  # setting == 2
            # Setting 2: All timepoints
            n_timepoints = len(self.time_label_order)
            
            # Determine cells per timepoint based on balance strategy
            if balance_strategy == 'total' and total_cells:
                # Total balance: distribute total_cells evenly across timepoints
                cells_per_tp = total_cells // n_timepoints
            else:
                # Per-timepoint balance (default): each timepoint has cells_per_timepoint
                cells_per_tp = cells_per_timepoint
            
            X_train_list, y_train_list = [], []
            X_test_list, y_test_list = [], []
            
            for time_label in self.time_label_order:
                time_idx = time_to_idx[time_label]
                time_mask = (y == time_idx)
                
                # Train
                train_time_mask = time_mask & self.train_mask
                train_indices = np.where(train_time_mask)[0]
                if cells_per_tp and len(train_indices) > cells_per_tp:
                    train_indices = np.random.choice(train_indices, cells_per_tp, replace=False)
                X_train_list.append(X[train_indices])
                y_train_list.append(y[train_indices])
                
                # Test
                test_time_mask = time_mask & self.test_mask
                test_indices = np.where(test_time_mask)[0]
                if cells_per_tp and len(test_indices) > cells_per_tp:
                    test_indices = np.random.choice(test_indices, cells_per_tp, replace=False)
                X_test_list.append(X[test_indices])
                y_test_list.append(y[test_indices])
            
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            X_test = np.vstack(X_test_list)
            y_test = np.concatenate(y_test_list)
        
        print(f"\nSetting {setting} data prepared:")
        print(f"  Train: {X_train.shape[0]} cells × {X_train.shape[1]} genes")
        print(f"  Test: {X_test.shape[0]} cells × {X_test.shape[1]} genes")
        
        return X_train, y_train, X_test, y_test


def create_default_emt_data_loader(
    file_path: Optional[str] = None,
    n_hvg: int = 100
) -> RealDataLoader:
    """
    Create data loader with default EMT dataset configuration
    
    Args:
        file_path: Path to h5ad file (if None, uses default)
        n_hvg: Number of HVGs
        
    Returns:
        RealDataLoader instance
    """
    if file_path is None:
        file_path = "/home/pan/Experiments/EXPs/2024_EMT_LM_workspace/EMT-LM_Data/Step_0_data/preprocessed/GSE147405_Cook/2024_12_04_Cook_emt_dataset_with_removal_scBERT.h5ad"
    
    loader = RealDataLoader(
        file_path=file_path,
        n_hvg=n_hvg,
        obs_time_column='Ground_truth',
        time_labels=['0d', '8h', '1d', '3d', '7d'],
        time_label_order=['0d', '8h', '1d', '3d', '7d'],
        biology_split={
            "train_val_column": "batches",
            "train": ["Mix1", "Mix2", "Mix3"],
            "test": ["Mix4"]
        }
    )
    
    return loader


if __name__ == "__main__":
    # Test the data loader
    loader = create_default_emt_data_loader()
    loader.load_and_analyze()
    loader.validate_biology_split()
    
    # Test getting data for both settings
    print("\n" + "="*70)
    print("Testing Setting 1 (boundary only)")
    print("="*70)
    X_train_s1, y_train_s1, X_test_s1, y_test_s1 = loader.get_data_for_setting(
        setting=1, cells_per_timepoint=2000
    )
    
    print("\n" + "="*70)
    print("Testing Setting 2 (all timepoints)")
    print("="*70)
    X_train_s2, y_train_s2, X_test_s2, y_test_s2 = loader.get_data_for_setting(
        setting=2, total_cells=X_train_s1.shape[0]
    )
