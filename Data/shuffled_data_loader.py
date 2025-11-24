#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shuffled Data Loader for Experiment 5
Extends RealDataLoader to support shuffled time series datasets
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .data_loader import RealDataLoader
from .shuffled_dataset import ShuffledTimeSeriesDataset
from .dataset_builder import TimeSeriesDataset


class ShuffledDataLoader(RealDataLoader):
    """
    Data loader that supports both normal and shuffled time series datasets
    
    Inherits from RealDataLoader and adds support for creating shuffled datasets
    where temporal causal relationships are broken while preserving time interval
    distribution.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with same parameters as RealDataLoader"""
        super().__init__(*args, **kwargs)
        
    def create_shuffled_datasets(
        self,
        time_points: List[str],
        time_intervals: Dict[str, float],
        n_samples: int = 8974,
        seed: int = 42
    ) -> Tuple[ShuffledTimeSeriesDataset, TimeSeriesDataset]:
        """
        Create shuffled training dataset and normal test dataset
        
        Args:
            time_points: List of time points to use
            time_intervals: Dict mapping interval names to hours
            n_samples: Number of training pairs to generate
            seed: Random seed
            
        Returns:
            (train_dataset, test_dataset): Shuffled train, normal test
        """
        if self.adata_hvg is None:
            raise ValueError("Data not loaded. Call load_and_analyze() first.")
        
        # Filter to specified time points
        mask = self.adata_hvg.obs[self.obs_time_column].isin(time_points)
        adata_filtered = self.adata_hvg[mask].copy()
        
        # Create shuffled training dataset
        train_mask_filtered = mask.values & self.train_mask
        adata_train = self.adata_hvg[train_mask_filtered].copy()
        
        train_dataset = ShuffledTimeSeriesDataset(
            adata=adata_train,
            time_labels=time_points,
            time_intervals=time_intervals,
            n_samples=n_samples,
            seed=seed,
            obs_time_column=self.obs_time_column
        )
        
        # Create normal test dataset (for evaluation on real data)
        test_mask_filtered = mask.values & self.test_mask
        adata_test = self.adata_hvg[test_mask_filtered].copy()
        
        # Get test data
        X_test = adata_test.X
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        
        # Convert time labels to indices
        time_to_idx = {label: idx for idx, label in enumerate(time_points)}
        y_test = np.array([
            time_to_idx[t] for t in adata_test.obs[self.obs_time_column]
        ])
        
        test_dataset = TimeSeriesDataset(
            X=X_test,
            y=y_test,
            time_labels=time_points
        )
        
        print(f"\n✓ Created shuffled train dataset: {len(train_dataset)} pairs")
        print(f"✓ Created normal test dataset: {len(test_dataset)} samples")
        
        return train_dataset, test_dataset
    
    def get_shuffled_dataloaders(
        self,
        time_points: List[str],
        time_intervals: Dict[str, float],
        n_samples: int = 8974,
        batch_size: int = 256,
        num_workers: int = 4,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        Create shuffled train and normal test dataloaders
        
        Args:
            time_points: List of time points to use
            time_intervals: Dict mapping interval names to hours
            n_samples: Number of training pairs
            batch_size: Batch size
            num_workers: Number of data loading workers
            seed: Random seed
            
        Returns:
            (train_loader, test_loader, statistics)
        """
        # Create datasets
        train_dataset, test_dataset = self.create_shuffled_datasets(
            time_points=time_points,
            time_intervals=time_intervals,
            n_samples=n_samples,
            seed=seed
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Get statistics
        stats = {
            'train_size': len(train_dataset),
            'test_size': len(test_dataset),
            'n_genes': self.adata_hvg.shape[1],
            'n_timepoints': len(time_points),
            'time_labels': time_points,
            'shuffled': True,
            'time_interval_stats': train_dataset.get_time_interval_statistics()
        }
        
        return train_loader, test_loader, stats


def create_shuffled_emt_data_loader(
    file_path: Optional[str] = None,
    n_hvg: int = 1000
) -> ShuffledDataLoader:
    """
    Create shuffled data loader with default EMT dataset configuration
    
    Args:
        file_path: Path to h5ad file (if None, uses default)
        n_hvg: Number of HVGs
        
    Returns:
        ShuffledDataLoader instance
    """
    if file_path is None:
        file_path = "/home/pan/Experiments/EXPs/2025_10_VCC_Exps/DATAs/EMT/2024_12_04_Cook_emt_dataset_with_removal.h5ad"
    
    loader = ShuffledDataLoader(
        file_path=file_path,
        n_hvg=n_hvg,
        obs_time_column='Ground_truth',
        time_labels=['0d', '8h', '1d', '3d', '7d', '8h_rm', '1d_rm', '3d_rm'],
        time_label_order=['0d', '8h', '1d', '3d', '7d', '8h_rm', '1d_rm', '3d_rm'],
        biology_split={
            "train_val_column": "batches",
            "train": ["Mix1", "Mix3", "Mix4"],
            "test": ["Mix2"]
        }
    )
    
    return loader


if __name__ == "__main__":
    """Test the shuffled data loader"""
    
    print("="*70)
    print("Testing ShuffledDataLoader")
    print("="*70)
    
    # Create loader
    loader = create_shuffled_emt_data_loader()
    loader.load_and_analyze()
    loader.validate_biology_split()
    
    # Define time points and intervals for forward EMT
    time_points = ['0d', '8h', '1d', '3d', '7d']
    time_intervals = {
        '0d-8h': 8,
        '8h-1d': 16,
        '1d-3d': 48,
        '3d-7d': 96
    }
    
    # Create shuffled dataloaders
    print("\n" + "="*70)
    print("Creating Shuffled Dataloaders")
    print("="*70)
    
    train_loader, test_loader, stats = loader.get_shuffled_dataloaders(
        time_points=time_points,
        time_intervals=time_intervals,
        n_samples=8974,
        batch_size=256
    )
    
    # Print statistics
    print("\n" + "="*70)
    print("Dataset Statistics")
    print("="*70)
    print(f"Train size: {stats['train_size']}")
    print(f"Test size: {stats['test_size']}")
    print(f"Number of genes: {stats['n_genes']}")
    print(f"Number of timepoints: {stats['n_timepoints']}")
    print(f"Time labels: {stats['time_labels']}")
    print(f"Shuffled: {stats['shuffled']}")
    
    print("\nTime interval statistics:")
    interval_stats = stats['time_interval_stats']
    print(f"Total pairs: {interval_stats['total_pairs']}")
    print("\nExpected distribution:")
    for dt, prob in interval_stats['expected_distribution'].items():
        print(f"  Δt={dt}h: {prob:.3f}")
    print("\nActual distribution:")
    for dt, prob in interval_stats['interval_distribution'].items():
        count = interval_stats['interval_counts'][dt]
        print(f"  Δt={dt}h: {prob:.3f} (count={count})")
    
    # Test batch loading
    print("\n" + "="*70)
    print("Testing Batch Loading")
    print("="*70)
    
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        print(f"Batch {batch_idx}: x_shape={x_batch.shape}, y_shape={y_batch.shape}")
        if batch_idx >= 2:
            break
    
    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)
