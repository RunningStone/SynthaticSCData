#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Manager - Data Loading and Management

Handles all data loading operations including:
- Loading h5ad files
- Loading generated PKL files
- Loading metrics JSON files
- Sampling data from datasets
- Managing data splits
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataManager:
    """
    Manages data loading and sampling operations.
    
    Responsibilities:
    - Load h5ad files with HVG selection
    - Load generated PKL files
    - Load metrics JSON files
    - Sample data by timepoint
    - Manage train/test splits
    """
    
    def __init__(self):
        """Initialize data manager"""
        pass
    
    def load_h5ad(
        self,
        file_path: str,
        n_hvg: int,
        time_labels: List[str],
        obs_time_column: str = 'Ground_truth',
        biology_split: Optional[Dict] = None
    ) -> Dict:
        """
        Load h5ad file and extract data.
        
        Args:
            file_path: Path to h5ad file
            n_hvg: Number of highly variable genes
            time_labels: List of time labels to use
            obs_time_column: Column name for time labels
            biology_split: Dictionary for train/test split
        
        Returns:
            Dictionary containing:
                - 'X': Expression matrix (n_cells, n_genes)
                - 'y': Time labels as integers (n_cells,)
                - 'time_labels': List of time label strings
                - 'adata': AnnData object
                - 'train_mask': Boolean mask for training samples
                - 'test_mask': Boolean mask for test samples
        """
        from Data import create_default_emt_data_loader
        
        # Create data loader
        loader = create_default_emt_data_loader(file_path=file_path, n_hvg=n_hvg)
        
        # Override time labels if provided
        if time_labels:
            loader.time_labels = time_labels
            loader.time_label_order = time_labels
        
        # Override biology split if provided
        if biology_split:
            loader.biology_split = biology_split
        
        # Load and analyze
        loader.load_and_analyze()
        loader.validate_biology_split()
        
        # Extract data
        X = loader.adata_hvg.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Convert time labels to integers
        time_to_idx = {label: idx for idx, label in enumerate(loader.time_label_order)}
        y = np.array([time_to_idx[t] for t in loader.adata_hvg.obs[obs_time_column]])
        
        return {
            'X': X,
            'y': y,
            'time_labels': loader.time_label_order,
            'adata': loader.adata_hvg,
            'train_mask': loader.train_mask,
            'test_mask': loader.test_mask
        }
    
    def load_generated_pkl(self, pkl_path: Path) -> Optional[Dict]:
        """
        Load generated data from PKL file.
        
        Args:
            pkl_path: Path to PKL file
        
        Returns:
            Dictionary with generated data or None if file doesn't exist
        """
        if not pkl_path.exists():
            return None
        
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    def load_metrics_json(self, json_path: Path) -> Dict:
        """
        Load evaluation metrics from JSON file.
        
        Args:
            json_path: Path to JSON file
        
        Returns:
            Dictionary with metrics or empty dict if file doesn't exist
        """
        if not json_path.exists():
            return {}
        
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def sample_by_timepoint(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray],
        n_samples_per_timepoint: int,
        time_labels: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample data by timepoint.
        
        Args:
            X: Expression matrix (n_cells, n_genes)
            y: Time labels as integers (n_cells,)
            mask: Boolean mask for filtering (e.g., test_mask)
            n_samples_per_timepoint: Number of samples per timepoint
            time_labels: List of time label strings
        
        Returns:
            Tuple of (X_sampled, y_sampled)
        """
        X_samples_list = []
        y_samples_list = []
        
        for time_idx, time_label in enumerate(time_labels):
            # Get samples for this timepoint
            time_mask = (y == time_idx)
            
            if mask is not None:
                time_mask = time_mask & mask
            
            indices = np.where(time_mask)[0]
            
            # Sample if we have more than requested
            if len(indices) > n_samples_per_timepoint:
                sampled_indices = np.random.choice(
                    indices, n_samples_per_timepoint, replace=False
                )
            else:
                sampled_indices = indices
            
            X_samples_list.append(X[sampled_indices])
            y_samples_list.append(y[sampled_indices])
        
        X_sampled = np.vstack(X_samples_list)
        y_sampled = np.concatenate(y_samples_list)
        
        return X_sampled, y_sampled
    
    def get_source_target_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        source_time_idx: int,
        target_time_idx: int,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get source and target samples for generation.
        
        Args:
            X: Expression matrix (n_cells, n_genes)
            y: Time labels as integers (n_cells,)
            source_time_idx: Source timepoint index
            target_time_idx: Target timepoint index
            n_samples: Number of samples to extract
        
        Returns:
            Tuple of (source_samples, target_samples)
        """
        # Get source samples
        source_mask = (y == source_time_idx)
        source_samples = X[source_mask]
        
        if len(source_samples) > n_samples:
            source_indices = np.random.choice(
                len(source_samples), n_samples, replace=False
            )
            source_samples = source_samples[source_indices]
        
        # Get target samples
        target_mask = (y == target_time_idx)
        target_samples = X[target_mask]
        
        if len(target_samples) > n_samples:
            target_indices = np.random.choice(
                len(target_samples), n_samples, replace=False
            )
            target_samples = target_samples[target_indices]
        
        return source_samples, target_samples
    
    def load_experiment_config(self, config_path: Path) -> Dict:
        """
        Load experiment configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file
        
        Returns:
            Dictionary with configuration
        """
        import yaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def aggregate_configs(self, config_paths: List[Path]) -> Dict[str, Dict]:
        """
        Load and aggregate multiple experiment configurations.
        
        Args:
            config_paths: List of paths to YAML config files
        
        Returns:
            Dictionary mapping setting names to configs
        """
        configs = {}
        
        for config_path in config_paths:
            config = self.load_experiment_config(config_path)
            
            # Extract setting name
            setting_name = config['experiment']['name'].split('_')[-1]
            if not setting_name.startswith('Setting'):
                setting_name = config_path.parent.name.replace('EMT_', '')
            
            configs[setting_name] = config
        
        return configs
