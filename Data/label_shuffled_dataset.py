#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label-Shuffled Time Series Dataset for Experiment 8 (Setting 8)

Purpose:
    Decouple absolute time labels from relative temporal positions by randomly
    shuffling time labels on intermediate timepoints (excluding start/end).
    
Design:
    - Remove boundary timepoints (0d, 7d) from training
    - Keep only intermediate timepoints (8h, 1d, 3d)
    - Randomly shuffle time labels among these intermediate points
    - Confound "relative position" vs "absolute time annotation"
    
Research Question:
    Does the model rely on absolute time labels, or can it learn true dynamics
    from data geometry even when time labels are randomized?
    
Expected Outcomes:
    - If performance drops: model depends on absolute time labels
    - If performance maintains: model learns from data geometry, not labels
    
Author: Shi Pan
Date: 2024-11-24
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class LabelShuffledDataset(Dataset):
    """
    Dataset with randomized time labels on intermediate timepoints
    
    This dataset:
    1. Excludes boundary timepoints (start and end)
    2. Keeps only intermediate timepoints
    3. Randomly shuffles time labels among intermediate points
    4. Maintains the same data structure as TimeSeriesDataset for compatibility
    
    Example:
        Original data:
            - 8h samples labeled as 8h (index 1)
            - 1d samples labeled as 1d (index 2)
            - 3d samples labeled as 3d (index 3)
        
        After shuffling:
            - 8h samples might be labeled as 1d (index 2)
            - 1d samples might be labeled as 3d (index 3)
            - 3d samples might be labeled as 8h (index 1)
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time_labels: List[str],
        intermediate_indices: List[int],
        seed: int = 42,
        shuffle_labels: bool = True
    ):
        """
        Initialize label-shuffled dataset
        
        Args:
            X: Expression matrix (n_cells, n_genes)
            y: Original time point labels as integers (n_cells,)
            time_labels: List of all time label names
            intermediate_indices: Indices of intermediate timepoints to keep
                                 (e.g., [1, 2, 3] for 8h, 1d, 3d if time_labels=['0d','8h','1d','3d','7d'])
            seed: Random seed for label shuffling
            shuffle_labels: If True, shuffle labels; if False, keep original (for testing)
        """
        self.time_labels = time_labels
        self.intermediate_indices = intermediate_indices
        self.seed = seed
        self.shuffle_labels = shuffle_labels
        
        # Filter to keep only intermediate timepoints
        mask = np.isin(y, intermediate_indices)
        self.X = X[mask]
        self.y_original = y[mask].copy()
        
        # Shuffle labels if requested
        if shuffle_labels:
            self.y_shuffled = self._shuffle_labels(self.y_original, seed)
        else:
            self.y_shuffled = self.y_original.copy()
        
        # Convert to tensors
        self.X_tensor = torch.FloatTensor(self.X)
        self.y_tensor = torch.LongTensor(self.y_shuffled)
        
        # Statistics
        self._compute_statistics()
        
    def _shuffle_labels(self, y: np.ndarray, seed: int) -> np.ndarray:
        """
        Randomly shuffle time labels among intermediate timepoints
        
        Strategy:
            For each intermediate timepoint, randomly assign it to another
            intermediate timepoint label. Ensures the permutation is NOT
            the identity (i.e., at least one label changes).
        
        Args:
            y: Original labels (n_cells,)
            seed: Random seed
            
        Returns:
            Shuffled labels (n_cells,)
        """
        np.random.seed(seed)
        y_shuffled = y.copy()
        
        # Create a random permutation of intermediate indices
        # Ensure it's not the identity permutation (no change)
        label_mapping = {}
        original_indices = list(self.intermediate_indices)
        
        # Keep trying until we get a non-identity permutation
        max_attempts = 100
        for attempt in range(max_attempts):
            shuffled_indices = np.random.permutation(self.intermediate_indices)
            # Check if it's not identity (not in ascending order)
            is_identity = np.array_equal(shuffled_indices, sorted(original_indices))
            if not is_identity:
                print(f"Permutation found after {attempt + 1} attempts")
                break
        else:
            # Fallback: use a simple rotation if random fails
            shuffled_indices = np.roll(sorted(original_indices), 1)
            print(f"Warning: Random permutation was identity after {max_attempts} attempts, using rotation instead")
        
        print(f"Original indices: {original_indices}")
        print(f"Shuffled indices: {list(shuffled_indices)}")
        
        for original_idx, shuffled_idx in zip(self.intermediate_indices, shuffled_indices):
            label_mapping[original_idx] = shuffled_idx
        
        print(f"Label mapping: {label_mapping}")
        
        # Apply mapping
        for i in range(len(y_shuffled)):
            y_shuffled[i] = label_mapping[y_shuffled[i]]
        
        return y_shuffled
    
    def _compute_statistics(self):
        """Compute and print dataset statistics"""
        print(f"\n{'='*70}")
        print("LabelShuffledDataset Statistics")
        print(f"{'='*70}")
        print(f"Total samples: {len(self.X)}")
        print(f"Number of genes: {self.X.shape[1]}")
        print(f"Intermediate timepoints: {[self.time_labels[i] for i in self.intermediate_indices]}")
        print(f"Label shuffling: {'Enabled' if self.shuffle_labels else 'Disabled'}")
        print(f"Random seed: {self.seed}")
        
        if self.shuffle_labels:
            print(f"\nLabel Distribution (Original → Shuffled):")
            for idx in self.intermediate_indices:
                original_count = (self.y_original == idx).sum()
                shuffled_count = (self.y_shuffled == idx).sum()
                label_name = self.time_labels[idx]
                print(f"  {label_name} (index {idx}): {original_count} → {shuffled_count} samples")
            
            # Check label mapping
            print(f"\nLabel Mapping:")
            for orig_idx in self.intermediate_indices:
                # Find what this label was mapped to
                orig_mask = self.y_original == orig_idx
                if orig_mask.sum() > 0:
                    # Get the most common shuffled label for this original label
                    shuffled_labels = self.y_shuffled[orig_mask]
                    unique, counts = np.unique(shuffled_labels, return_counts=True)
                    most_common = unique[np.argmax(counts)]
                    print(f"  {self.time_labels[orig_idx]} → {self.time_labels[most_common]}")
        else:
            print(f"\nLabel Distribution (No Shuffling):")
            for idx in self.intermediate_indices:
                count = (self.y_shuffled == idx).sum()
                label_name = self.time_labels[idx]
                print(f"  {label_name} (index {idx}): {count} samples")
        
        print(f"{'='*70}\n")
    
    def __len__(self):
        return len(self.X_tensor)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            (x, y): Expression vector and (shuffled) time label
        """
        return self.X_tensor[idx], self.y_tensor[idx]
    
    def get_original_labels(self):
        """Get original (true) labels for analysis"""
        return self.y_original
    
    def get_shuffled_labels(self):
        """Get shuffled labels used for training"""
        return self.y_shuffled
    
    def get_label_confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix between original and shuffled labels
        
        Returns:
            Confusion matrix (n_intermediate_timepoints, n_intermediate_timepoints)
            where entry [i, j] is the count of samples with original label i
            that were assigned shuffled label j
        """
        n_labels = len(self.intermediate_indices)
        confusion = np.zeros((n_labels, n_labels), dtype=int)
        
        for i, orig_idx in enumerate(self.intermediate_indices):
            for j, shuf_idx in enumerate(self.intermediate_indices):
                mask = (self.y_original == orig_idx) & (self.y_shuffled == shuf_idx)
                confusion[i, j] = mask.sum()
        
        return confusion


def create_label_shuffled_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    time_labels: List[str],
    start_timepoint: str,
    end_timepoint: str,
    seed: int = 42
) -> tuple:
    """
    Create label-shuffled train and test datasets
    
    Args:
        X_train: Training expression matrix
        y_train: Training time labels (integers)
        X_test: Test expression matrix
        y_test: Test time labels (integers)
        time_labels: List of all time label names
        start_timepoint: Name of start timepoint to exclude (e.g., '0d')
        end_timepoint: Name of end timepoint to exclude (e.g., '7d')
        seed: Random seed
        
    Returns:
        (train_dataset, test_dataset): LabelShuffledDataset objects
    """
    # Find indices of start and end timepoints
    start_idx = time_labels.index(start_timepoint)
    end_idx = time_labels.index(end_timepoint)
    
    # Get intermediate indices (exclude start and end)
    all_indices = list(range(len(time_labels)))
    intermediate_indices = [i for i in all_indices if i != start_idx and i != end_idx]
    
    print(f"\nCreating Label-Shuffled Datasets (Setting 8)")
    print(f"{'='*70}")
    print(f"All timepoints: {time_labels}")
    print(f"Excluded start: {start_timepoint} (index {start_idx})")
    print(f"Excluded end: {end_timepoint} (index {end_idx})")
    print(f"Intermediate timepoints: {[time_labels[i] for i in intermediate_indices]}")
    print(f"Random seed: {seed}")
    
    # Create training dataset with shuffled labels
    train_dataset = LabelShuffledDataset(
        X=X_train,
        y=y_train,
        time_labels=time_labels,
        intermediate_indices=intermediate_indices,
        seed=seed,
        shuffle_labels=True
    )
    
    # Create test dataset WITHOUT shuffling (evaluate on true labels)
    test_dataset = LabelShuffledDataset(
        X=X_test,
        y=y_test,
        time_labels=time_labels,
        intermediate_indices=intermediate_indices,
        seed=seed,
        shuffle_labels=False  # Keep original labels for evaluation
    )
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    # Test the dataset
    print("Testing LabelShuffledDataset")
    print("="*70)
    
    # Create dummy data
    n_cells = 1000
    n_genes = 100
    time_labels = ['0d', '8h', '1d', '3d', '7d']
    
    # Generate dummy expression data
    X = np.random.randn(n_cells, n_genes)
    
    # Generate time labels (200 cells per timepoint)
    y = np.repeat(np.arange(5), n_cells // 5)
    
    print(f"\nOriginal data:")
    print(f"  Shape: {X.shape}")
    print(f"  Time labels: {time_labels}")
    print(f"  Label distribution: {np.bincount(y)}")
    
    # Create label-shuffled dataset
    intermediate_indices = [1, 2, 3]  # 8h, 1d, 3d
    dataset = LabelShuffledDataset(
        X=X,
        y=y,
        time_labels=time_labels,
        intermediate_indices=intermediate_indices,
        seed=42,
        shuffle_labels=True
    )
    
    print(f"\nDataset length: {len(dataset)}")
    print(f"Expected: {(n_cells // 5) * 3} (only intermediate timepoints)")
    
    # Test __getitem__
    x, y_label = dataset[0]
    print(f"\nSample 0:")
    print(f"  Expression shape: {x.shape}")
    print(f"  Label: {y_label.item()} ({time_labels[y_label.item()]})")
    
    # Get confusion matrix
    confusion = dataset.get_label_confusion_matrix()
    print(f"\nLabel Confusion Matrix:")
    print(f"  Rows: Original labels")
    print(f"  Cols: Shuffled labels")
    print(confusion)
