#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Builder for Continuous Time Single-Cell Data
Implements different sampling strategies for train/test splits
"""

import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class ContinuousTimeDataset(Dataset):
    """PyTorch Dataset for continuous time single-cell data"""
    
    def __init__(
        self,
        X: np.ndarray,
        time_labels: np.ndarray,
        time_slices: np.ndarray,
        is_real: np.ndarray,
        original_labels: np.ndarray
    ):
        """
        Args:
            X: Expression matrix (n_cells, n_genes)
            time_labels: Continuous time values
            time_slices: Time slice categorical labels
            is_real: Boolean array indicating real vs generated cells
            original_labels: Original discrete time labels (NaN for generated)
        """
        self.X = torch.FloatTensor(X)
        self.time_labels = torch.FloatTensor(time_labels)
        self.time_slices = time_slices
        self.is_real = is_real
        self.original_labels = original_labels
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return only the expression data for simple training
        # Metadata can be accessed via dataset attributes if needed
        return self.X[idx]


class DatasetBuilder:
    """
    Build train/test datasets with different sampling strategies
    """
    
    def __init__(
        self,
        continuous_data_path: str,
        sampling_strategy: Literal['all_time', 'specific_time', 'clustered_time'] = 'all_time',
        train_ratio: float = 0.8,
        random_seed: int = 42
    ):
        """
        Initialize dataset builder
        
        Args:
            continuous_data_path: Path to continuous time h5ad file
            sampling_strategy: Strategy for train/test split
                - 'all_time': Random split across all time slices
                - 'specific_time': Specify which time slices for train/test
                - 'clustered_time': Group time slices into clusters
            train_ratio: Ratio of training data (for 'all_time' strategy)
            random_seed: Random seed
        """
        self.continuous_data_path = continuous_data_path
        self.sampling_strategy = sampling_strategy
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.adata = None
        self.train_dataset = None
        self.test_dataset = None
        
    def load_data(self):
        """Load continuous time data"""
        print(f"Loading continuous time data from {self.continuous_data_path}...")
        self.adata = sc.read_h5ad(self.continuous_data_path)
        print(f"Loaded: {self.adata.shape[0]} cells × {self.adata.shape[1]} genes")
        print(f"  Real cells: {self.adata.obs['is_real'].sum()}")
        print(f"  Generated cells: {(~self.adata.obs['is_real']).sum()}")
        print(f"  Unique time slices: {self.adata.obs['time_slice'].nunique()}")
        
    def build_all_time_split(self):
        """
        Strategy 1: All time slices, random split within each slice
        """
        print(f"\nBuilding all_time split (train_ratio={self.train_ratio})...")
        
        train_indices = []
        test_indices = []
        
        # Split each time slice
        for time_slice in self.adata.obs['time_slice'].unique():
            slice_mask = self.adata.obs['time_slice'] == time_slice
            slice_indices = np.where(slice_mask)[0]
            
            n_train = int(len(slice_indices) * self.train_ratio)
            
            np.random.shuffle(slice_indices)
            train_indices.extend(slice_indices[:n_train])
            test_indices.extend(slice_indices[n_train:])
        
        print(f"  Train: {len(train_indices)} cells")
        print(f"  Test: {len(test_indices)} cells")
        
        self._create_datasets(train_indices, test_indices)
        
    def build_specific_time_split(
        self,
        train_time_slices: List[str],
        test_time_slices: Optional[List[str]] = None
    ):
        """
        Strategy 2: Specify which time slices for train and test
        
        Args:
            train_time_slices: List of time slice names for training
            test_time_slices: List of time slice names for testing (if None, use remaining)
        """
        print(f"\nBuilding specific_time split...")
        print(f"  Train time slices: {train_time_slices}")
        
        # Get train indices
        train_mask = self.adata.obs['time_slice'].isin(train_time_slices)
        train_indices = np.where(train_mask)[0]
        
        # Get test indices
        if test_time_slices is None:
            # Use all remaining time slices
            test_mask = ~train_mask
        else:
            test_mask = self.adata.obs['time_slice'].isin(test_time_slices)
        
        test_indices = np.where(test_mask)[0]
        
        print(f"  Test time slices: {self.adata.obs.loc[test_mask, 'time_slice'].unique().tolist()}")
        print(f"  Train: {len(train_indices)} cells")
        print(f"  Test: {len(test_indices)} cells")
        
        self._create_datasets(train_indices, test_indices)
        
    def build_clustered_time_split(
        self,
        time_clusters: Dict[str, List[str]],
        train_clusters: List[str]
    ):
        """
        Strategy 3: Group time slices into clusters, specify which clusters for train
        
        Args:
            time_clusters: Dict mapping cluster_name -> list of time_slice names
            train_clusters: List of cluster names for training
        """
        print(f"\nBuilding clustered_time split...")
        print(f"  Clusters: {list(time_clusters.keys())}")
        print(f"  Train clusters: {train_clusters}")
        
        # Get train time slices from train clusters
        train_time_slices = []
        for cluster in train_clusters:
            train_time_slices.extend(time_clusters[cluster])
        
        # Get test time slices from remaining clusters
        test_clusters = [c for c in time_clusters.keys() if c not in train_clusters]
        test_time_slices = []
        for cluster in test_clusters:
            test_time_slices.extend(time_clusters[cluster])
        
        print(f"  Test clusters: {test_clusters}")
        
        # Use specific_time_split logic
        self.build_specific_time_split(train_time_slices, test_time_slices)
        
    def _create_datasets(self, train_indices: List[int], test_indices: List[int]):
        """Create PyTorch datasets from indices"""
        # Get data
        X = self.adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        time_labels = self.adata.obs['continuous_time'].values
        time_slices = self.adata.obs['time_slice'].values
        is_real = self.adata.obs['is_real'].values
        original_labels = self.adata.obs['original_label'].values
        
        # Create train dataset
        self.train_dataset = ContinuousTimeDataset(
            X=X[train_indices],
            time_labels=time_labels[train_indices],
            time_slices=time_slices[train_indices],
            is_real=is_real[train_indices],
            original_labels=original_labels[train_indices]
        )
        
        # Create test dataset
        self.test_dataset = ContinuousTimeDataset(
            X=X[test_indices],
            time_labels=time_labels[test_indices],
            time_slices=time_slices[test_indices],
            is_real=is_real[test_indices],
            original_labels=original_labels[test_indices]
        )
        
    def get_dataloaders(
        self,
        batch_size: int = 128,
        num_workers: int = 4,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Get PyTorch DataLoaders
        
        Returns:
            (train_loader, test_loader)
        """
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("Datasets not built yet. Call build_*_split() first.")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if self.train_dataset is None or self.test_dataset is None:
            return {}
        
        stats = {
            'train_size': len(self.train_dataset),
            'test_size': len(self.test_dataset),
            'train_real_cells': self.train_dataset.is_real.sum(),
            'test_real_cells': self.test_dataset.is_real.sum(),
            'train_generated_cells': (~self.train_dataset.is_real).sum(),
            'test_generated_cells': (~self.test_dataset.is_real).sum(),
            'n_genes': self.train_dataset.X.shape[1],
            'time_range': (
                float(self.adata.obs['continuous_time'].min()),
                float(self.adata.obs['continuous_time'].max())
            )
        }
        
        return stats


def create_default_emt_dataset(
    continuous_data_path: str,
    sampling_strategy: Literal['all_time', 'specific_time', 'clustered_time'] = 'all_time',
    train_ratio: float = 0.8,
    batch_size: int = 128
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create default EMT dataset with specified sampling strategy
    
    Args:
        continuous_data_path: Path to continuous time h5ad file
        sampling_strategy: Sampling strategy
        train_ratio: Train ratio for 'all_time' strategy
        batch_size: Batch size for dataloaders
    
    Returns:
        (train_loader, test_loader, statistics)
    """
    builder = DatasetBuilder(
        continuous_data_path=continuous_data_path,
        sampling_strategy=sampling_strategy,
        train_ratio=train_ratio
    )
    
    builder.load_data()
    
    if sampling_strategy == 'all_time':
        builder.build_all_time_split()
    elif sampling_strategy == 'specific_time':
        # Example: train on early time points, test on late
        all_slices = sorted(builder.adata.obs['time_slice'].unique())
        n_train_slices = int(len(all_slices) * 0.6)
        train_slices = all_slices[:n_train_slices]
        test_slices = all_slices[n_train_slices:]
        builder.build_specific_time_split(train_slices, test_slices)
    elif sampling_strategy == 'clustered_time':
        # Example: cluster by early/mid/late phases
        all_slices = sorted(builder.adata.obs['time_slice'].unique())
        n = len(all_slices)
        clusters = {
            'early': all_slices[:n//3],
            'mid': all_slices[n//3:2*n//3],
            'late': all_slices[2*n//3:]
        }
        builder.build_clustered_time_split(clusters, train_clusters=['early', 'mid'])
    
    train_loader, test_loader = builder.get_dataloaders(batch_size=batch_size)
    stats = builder.get_statistics()
    
    return train_loader, test_loader, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build datasets from continuous time data')
    parser.add_argument('--input', type=str, required=True, help='Input continuous h5ad file')
    parser.add_argument('--strategy', type=str, default='all_time',
                       choices=['all_time', 'specific_time', 'clustered_time'])
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    
    train_loader, test_loader, stats = create_default_emt_dataset(
        continuous_data_path=args.input,
        sampling_strategy=args.strategy,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print("="*50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
