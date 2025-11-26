#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Data Loader for Time Series Single-Cell Data

Defines the abstract interface for all data loaders in the project.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class BaseDataLoader(ABC):
    """
    Abstract base class for all time series single-cell data loaders.
    
    Provides common infrastructure for:
    - Loading h5ad files
    - Selecting highly variable genes (HVGs)
    - Creating train/test splits
    - Validating data quality
    - Extracting data for different experimental settings
    
    Subclasses customize:
    - Data transformation logic (e.g., shuffling, interpolation)
    - Dataset creation methods
    - Additional preprocessing steps
    """
    
    def __init__(
        self,
        file_path: str,
        n_hvg: int = 100,
        obs_time_column: str = 'Ground_truth',
        time_labels: Optional[List[str]] = None,
        time_label_order: Optional[List[str]] = None,
        biology_split: Optional[Dict] = None,
        random_seed: int = 42
    ):
        """
        Initialize base data loader.
        
        Args:
            file_path: Path to h5ad file
            n_hvg: Number of highly variable genes to select
            obs_time_column: Column name for time labels in obs
            time_labels: List of time labels to use (filters data to these labels)
            time_label_order: Ordered list of time labels (defines time sequence)
            biology_split: Dictionary defining train/test split:
                - "train_val_column": column name for split or "random"
                - "train": list of values for training set
                - "test": list of values for test set
            random_seed: Random seed for reproducibility
        """
        self.file_path = file_path
        self.n_hvg = n_hvg
        self.obs_time_column = obs_time_column
        self.time_labels = time_labels or []
        self.time_label_order = time_label_order or time_labels or []
        self.biology_split = biology_split or {}
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        # Data containers (to be populated by load_and_analyze)
        self.adata = None  # Original full data
        self.adata_hvg = None  # HVG-filtered data
        self.hvg_genes = None  # List of HVG gene names
        self.train_mask = None  # Boolean mask for training samples
        self.test_mask = None  # Boolean mask for test samples
    
    def load_and_analyze(self):
        """
        Load data and perform initial analysis.
        
        This is the main entry point. Default implementation:
        1. Load h5ad file
        2. Analyze obs columns
        3. Filter to specified time labels
        4. Select HVGs
        5. Create train/test split
        
        Subclasses can override to add custom preprocessing steps.
        """
        print("="*70)
        print(f"Loading Data: {self.__class__.__name__}")
        print("="*70)
        print(f"File: {self.file_path}")
        
        # Load data
        self.adata = sc.read_h5ad(self.file_path)
        print(f"\nLoaded: {self.adata.shape[0]} cells × {self.adata.shape[1]} genes")
        
        # Analyze obs columns
        self._analyze_obs_columns()
        
        # Filter to specified time labels
        if self.time_labels:
            self._filter_time_labels()
        
        # Select HVGs
        self._select_hvgs()
        
        # Create train/test split
        self._create_split_masks()
        
        print("\n" + "="*70)
        print("Data Loading Complete")
        print("="*70)
    
    def _analyze_obs_columns(self):
        """Analyze and print obs column statistics"""
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
    
    def _filter_time_labels(self):
        """Filter data to specified time labels"""
        mask = self.adata.obs[self.obs_time_column].isin(self.time_labels)
        self.adata = self.adata[mask].copy()
        print(f"\n✓ Filtered to {len(self.time_labels)} time labels: {self.adata.shape[0]} cells")
    
    def _select_hvgs(self):
        """
        Select highly variable genes.
        
        Uses Seurat v3 method if available, otherwise falls back to variance-based selection.
        """
        print(f"\nSelecting top {self.n_hvg} HVGs...")
        
        # Clean data
        X = self.adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=np.finfo(np.float32).max,
            neginf=np.finfo(np.float32).min
        )
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
                self.adata.var.iloc[
                    top_indices,
                    self.adata.var.columns.get_loc('highly_variable')
                ] = True
        
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
        """
        Create train/test split masks.
        
        Supports two modes:
        1. Random split: 80/20 split
        2. Biology-based split: Based on specified column values
        """
        if not self.biology_split:
            print("\nWarning: No biology_split provided, using random 80/20 split")
            self._create_random_split()
            return
        
        train_val_column = self.biology_split.get("train_val_column", "random")
        
        if train_val_column == "random":
            self._create_random_split()
        else:
            self._create_biology_split(train_val_column)
        
        print(f"\n✓ Train cells: {self.train_mask.sum()}")
        print(f"✓ Test cells: {self.test_mask.sum()}")
    
    def _create_random_split(self):
        """Create random 80/20 train/test split"""
        n_cells = self.adata_hvg.shape[0]
        indices = np.arange(n_cells)
        np.random.shuffle(indices)
        n_train = int(n_cells * 0.8)
        
        self.train_mask = np.zeros(n_cells, dtype=bool)
        self.train_mask[indices[:n_train]] = True
        self.test_mask = ~self.train_mask
    
    def _create_biology_split(self, train_val_column: str):
        """Create biology-based train/test split"""
        train_values = self.biology_split.get("train", [])
        test_values = self.biology_split.get("test", [])
        
        self.train_mask = self.adata_hvg.obs[train_val_column].isin(train_values).values
        self.test_mask = self.adata_hvg.obs[train_val_column].isin(test_values).values
    
    def validate_biology_split(self) -> bool:
        """
        Validate that train and test sets contain all time point categories.
        
        Returns:
            True if validation passes, False otherwise
        """
        print("\n" + "="*70)
        print("Biology Split Validation")
        print("="*70)
        
        if self.train_mask is None or self.test_mask is None:
            print("Error: Split masks not created yet. Call load_and_analyze() first.")
            return False
        
        # Get time labels in train and test
        train_times = set(
            self.adata_hvg.obs.loc[self.train_mask, self.obs_time_column].unique()
        )
        test_times = set(
            self.adata_hvg.obs.loc[self.test_mask, self.obs_time_column].unique()
        )
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
            train_count = (
                (self.adata_hvg.obs[self.obs_time_column] == time_label) & self.train_mask
            ).sum()
            test_count = (
                (self.adata_hvg.obs[self.obs_time_column] == time_label) & self.test_mask
            ).sum()
            print(f"{time_label:>5s}: Train={train_count:>5d}, Test={test_count:>5d}")
        
        print("="*70)
        return valid
    
    def get_data_for_setting(
        self,
        setting: int = 1,
        cells_per_timepoint: int = 2000,
        total_cells: Optional[int] = None,
        balance_strategy: str = 'per_timepoint',
        selected_time_points: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for specific experimental setting.
        
        Args:
            setting: 1 for boundary only, 2 for all timepoints
            cells_per_timepoint: Number of cells per timepoint
            total_cells: Total number of cells (only used when balance_strategy='total')
            balance_strategy: 'per_timepoint' (default) or 'total'
                - 'per_timepoint': Each timepoint has cells_per_timepoint samples
                - 'total': Total samples = total_cells, evenly distributed
            selected_time_points: List of time points to include (if None, uses all)
            
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
            return self._get_boundary_data(X, y, time_to_idx, cells_per_timepoint)
        else:
            # Setting 2: Selected timepoints (or all if not specified)
            return self._get_all_timepoints_data(
                X, y, time_to_idx, cells_per_timepoint, total_cells, balance_strategy,
                selected_time_points=selected_time_points
            )
    
    def _get_boundary_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time_to_idx: Dict[str, int],
        cells_per_timepoint: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract data for boundary timepoints only"""
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
                train_indices = np.random.choice(
                    train_indices, cells_per_timepoint, replace=False
                )
            X_train_list.append(X[train_indices])
            y_train_list.append(y[train_indices])
            
            # Test
            test_time_mask = time_mask & self.test_mask
            test_indices = np.where(test_time_mask)[0]
            if cells_per_timepoint and len(test_indices) > cells_per_timepoint:
                test_indices = np.random.choice(
                    test_indices, cells_per_timepoint, replace=False
                )
            X_test_list.append(X[test_indices])
            y_test_list.append(y[test_indices])
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)
        
        print(f"\nSetting 1 (boundary) data prepared:")
        print(f"  Train: {X_train.shape[0]} cells × {X_train.shape[1]} genes")
        print(f"  Test: {X_test.shape[0]} cells × {X_test.shape[1]} genes")
        
        return X_train, y_train, X_test, y_test
    
    def _get_all_timepoints_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time_to_idx: Dict[str, int],
        cells_per_timepoint: int,
        total_cells: Optional[int],
        balance_strategy: str,
        selected_time_points: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract data for selected timepoints
        
        Args:
            selected_time_points: List of time points to include. 
                                  If None, uses all time points in time_label_order.
        """
        # Use selected time points if provided, otherwise use all
        time_points_to_use = selected_time_points if selected_time_points else self.time_label_order
        n_timepoints = len(time_points_to_use)
        
        # Determine cells per timepoint based on balance strategy
        if balance_strategy == 'total' and total_cells:
            cells_per_tp = total_cells // n_timepoints
        else:
            cells_per_tp = cells_per_timepoint
        
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []
        
        for time_label in time_points_to_use:
            time_idx = time_to_idx[time_label]
            time_mask = (y == time_idx)
            
            # Train
            train_time_mask = time_mask & self.train_mask
            train_indices = np.where(train_time_mask)[0]
            if cells_per_tp and len(train_indices) > cells_per_tp:
                train_indices = np.random.choice(
                    train_indices, cells_per_tp, replace=False
                )
            X_train_list.append(X[train_indices])
            y_train_list.append(y[train_indices])
            
            # Test
            test_time_mask = time_mask & self.test_mask
            test_indices = np.where(test_time_mask)[0]
            if cells_per_tp and len(test_indices) > cells_per_tp:
                test_indices = np.random.choice(
                    test_indices, cells_per_tp, replace=False
                )
            X_test_list.append(X[test_indices])
            y_test_list.append(y[test_indices])
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)
        
        print(f"\nSetting 2 data prepared (using {n_timepoints} timepoints: {time_points_to_use}):")
        print(f"  Train: {X_train.shape[0]} cells × {X_train.shape[1]} genes")
        print(f"  Test: {X_test.shape[0]} cells × {X_test.shape[1]} genes")
        
        return X_train, y_train, X_test, y_test
    
    def get_data_info(self) -> Dict:
        """
        Get data loader information for logging and debugging.
        
        Returns:
            Dictionary containing data loader metadata
        """
        info = {
            'loader_class': self.__class__.__name__,
            'file_path': self.file_path,
            'n_hvg': self.n_hvg,
            'obs_time_column': self.obs_time_column,
            'time_labels': self.time_labels,
            'time_label_order': self.time_label_order,
            'random_seed': self.random_seed
        }
        
        if self.adata_hvg is not None:
            info.update({
                'n_cells': self.adata_hvg.shape[0],
                'n_genes': self.adata_hvg.shape[1],
                'n_train': self.train_mask.sum() if self.train_mask is not None else 0,
                'n_test': self.test_mask.sum() if self.test_mask is not None else 0
            })
        
        return info
