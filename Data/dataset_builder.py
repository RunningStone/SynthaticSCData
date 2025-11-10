#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Builder for Real Time Series Single-Cell Data
Implements Setting 1 (boundary) and Setting 2 (all timepoints) datasets
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


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series single-cell data"""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time_labels: List[str]
    ):
        """
        Args:
            X: Expression matrix (n_cells, n_genes)
            y: Time point labels as integers (n_cells,)
            time_labels: List of time label names
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.time_labels = time_labels
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DatasetBuilder:
    """
    Build train/test datasets for Setting 1 and Setting 2
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        time_labels: List[str],
        random_seed: int = 42
    ):
        """
        Initialize dataset builder
        
        Args:
            X_train: Training expression matrix
            y_train: Training time labels (integers)
            X_test: Test expression matrix
            y_test: Test time labels (integers)
            time_labels: List of time label names
            random_seed: Random seed
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.time_labels = time_labels
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.train_dataset = None
        self.test_dataset = None
        
    def build_datasets(self):
        """Build PyTorch datasets"""
        print("\nBuilding PyTorch datasets...")
        
        self.train_dataset = TimeSeriesDataset(
            X=self.X_train,
            y=self.y_train,
            time_labels=self.time_labels
        )
        
        self.test_dataset = TimeSeriesDataset(
            X=self.X_test,
            y=self.y_test,
            time_labels=self.time_labels
        )
        
        print(f"✓ Train dataset: {len(self.train_dataset)} samples")
        print(f"✓ Test dataset: {len(self.test_dataset)} samples")
        
    def get_dataloaders(
        self,
        batch_size: int = 256,
        num_workers: int = 4,
        shuffle_train: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Get PyTorch DataLoaders
        
        Returns:
            (train_loader, test_loader)
        """
        if self.train_dataset is None or self.test_dataset is None:
            raise ValueError("Datasets not built yet. Call build_datasets() first.")
        
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
        
        # Count cells per time point
        train_time_counts = {}
        test_time_counts = {}
        
        for i, label in enumerate(self.time_labels):
            train_time_counts[label] = (self.y_train == i).sum()
            test_time_counts[label] = (self.y_test == i).sum()
        
        stats = {
            'train_size': len(self.train_dataset),
            'test_size': len(self.test_dataset),
            'n_genes': self.X_train.shape[1],
            'n_timepoints': len(self.time_labels),
            'time_labels': self.time_labels,
            'train_time_counts': train_time_counts,
            'test_time_counts': test_time_counts
        }
        
        return stats


def create_dataloaders_from_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    time_labels: List[str],
    batch_size: int = 256
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create dataloaders from numpy arrays
    
    Args:
        X_train: Training expression matrix
        y_train: Training time labels
        X_test: Test expression matrix
        y_test: Test time labels
        time_labels: List of time label names
        batch_size: Batch size
        
    Returns:
        (train_loader, test_loader, statistics)
    """
    builder = DatasetBuilder(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        time_labels=time_labels
    )
    
    builder.build_datasets()
    train_loader, test_loader = builder.get_dataloaders(batch_size=batch_size)
    stats = builder.get_statistics()
    
    return train_loader, test_loader, stats


if __name__ == "__main__":
    from data_loader import create_default_emt_data_loader
    
    # Test with real data
    print("Testing DatasetBuilder with real EMT data")
    print("="*70)
    
    # Load data
    loader = create_default_emt_data_loader()
    loader.load_and_analyze()
    loader.validate_biology_split()
    
    # Setting 1
    print("\n" + "="*70)
    print("Setting 1: Boundary only")
    print("="*70)
    X_train_s1, y_train_s1, X_test_s1, y_test_s1 = loader.get_data_for_setting(
        setting=1, cells_per_timepoint=2000
    )
    train_loader_s1, test_loader_s1, stats_s1 = create_dataloaders_from_data(
        X_train_s1, y_train_s1, X_test_s1, y_test_s1,
        time_labels=loader.time_label_order,
        batch_size=256
    )
    
    print("\nStatistics:")
    for key, value in stats_s1.items():
        print(f"  {key}: {value}")
    
    # Setting 2
    print("\n" + "="*70)
    print("Setting 2: All timepoints")
    print("="*70)
    X_train_s2, y_train_s2, X_test_s2, y_test_s2 = loader.get_data_for_setting(
        setting=2, total_cells=X_train_s1.shape[0]
    )
    train_loader_s2, test_loader_s2, stats_s2 = create_dataloaders_from_data(
        X_train_s2, y_train_s2, X_test_s2, y_test_s2,
        time_labels=loader.time_label_order,
        batch_size=256
    )
    
    print("\nStatistics:")
    for key, value in stats_s2.items():
        print(f"  {key}: {value}")
