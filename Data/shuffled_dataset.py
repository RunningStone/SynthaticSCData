#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shuffled Time Series Dataset for Experiment 5
Breaks temporal causal relationships while preserving time interval distribution
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from anndata import AnnData
import warnings
warnings.filterwarnings('ignore')


class ShuffledTimeSeriesDataset(Dataset):
    """
    Shuffled time series dataset that breaks causal relationships
    
    This dataset randomly pairs cells from different time points while maintaining
    the distribution of time intervals. This allows us to test whether models truly
    learn time-dependent dynamics or just memorize spatial mappings.
    
    Key features:
    - Preserves time interval distribution (Δt statistics)
    - Breaks cell-time correspondence (causal relationships)
    - Pre-generates all pairs for reproducibility
    """
    
    def __init__(
        self,
        adata: AnnData,
        time_labels: List[str],
        time_intervals: Dict[str, float],
        n_samples: int = 8974,
        seed: int = 42,
        obs_time_column: str = 'Ground_truth'
    ):
        """
        Initialize shuffled dataset
        
        Args:
            adata: AnnData object with expression data
            time_labels: List of time labels in order (e.g., ['0d', '8h', '1d', '3d', '7d'])
            time_intervals: Dict mapping interval names to hours (e.g., {'0d-8h': 8, '8h-1d': 16})
            n_samples: Total number of training pairs to generate
            seed: Random seed for reproducibility
            obs_time_column: Column name in adata.obs for time labels
        """
        self.adata = adata
        self.time_labels = time_labels
        self.time_intervals = time_intervals
        self.n_samples = n_samples
        self.seed = seed
        self.obs_time_column = obs_time_column
        
        np.random.seed(seed)
        
        # Build time interval distribution
        self.delta_t_values, self.delta_t_probs = self._build_delta_t_distribution()
        
        # Build cell pools for each time point
        self.cell_pools = self._build_cell_pools()
        
        # Pre-generate all shuffled pairs
        self.shuffled_pairs = self._generate_shuffled_pairs()
        
        print(f"\n✓ ShuffledTimeSeriesDataset created:")
        print(f"  - Total samples: {len(self.shuffled_pairs)}")
        print(f"  - Time points: {len(self.time_labels)}")
        print(f"  - Time interval distribution: {dict(zip(self.delta_t_values, self.delta_t_probs))}")
        
    def _build_delta_t_distribution(self):
        """
        Build time interval distribution from time_intervals dict
        
        Returns:
            (delta_t_values, probabilities): Lists of unique intervals and their probabilities
        """
        # Count occurrences of each time interval
        delta_t_counts = {}
        for interval_str, hours in self.time_intervals.items():
            delta_t_counts[hours] = delta_t_counts.get(hours, 0) + 1
        
        # Normalize to probabilities
        total = sum(delta_t_counts.values())
        delta_t_values = sorted(delta_t_counts.keys())
        probabilities = [delta_t_counts[dt] / total for dt in delta_t_values]
        
        return delta_t_values, probabilities
    
    def _build_cell_pools(self):
        """
        Build cell pools for each time point
        
        Returns:
            Dict mapping time_label -> expression matrix (n_cells, n_genes)
        """
        cell_pools = {}
        
        for time_label in self.time_labels:
            # Filter cells for this time point
            mask = self.adata.obs[self.obs_time_column] == time_label
            X = self.adata[mask].X
            
            # Convert to dense array if sparse
            if hasattr(X, 'toarray'):
                X = X.toarray()
            
            cell_pools[time_label] = X
            
        return cell_pools
    
    def _label_to_hours(self, label: str) -> float:
        """
        Convert time label to hours
        
        Args:
            label: Time label (e.g., '0d', '8h', '1d')
            
        Returns:
            Hours as float
        """
        if label == '0d':
            return 0.0
        elif label.endswith('h'):
            return float(label[:-1])
        elif label.endswith('d'):
            days = float(label[:-1])
            return days * 24.0
        elif label.endswith('h_rm'):
            # For reversal timepoints, calculate from 7d
            base_hours = 7 * 24.0  # 7d = 168h
            hours_after = float(label.split('h_rm')[0])
            return base_hours + hours_after
        elif label.endswith('d_rm'):
            # For reversal timepoints, calculate from 7d
            base_hours = 7 * 24.0  # 7d = 168h
            days_after = float(label.split('d_rm')[0])
            return base_hours + days_after * 24.0
        else:
            raise ValueError(f"Unknown time label format: {label}")
    
    def _hours_to_label(self, hours: float) -> Optional[str]:
        """
        Convert hours back to time label
        
        Args:
            hours: Hours as float
            
        Returns:
            Time label or None if not found
        """
        for label in self.time_labels:
            if abs(self._label_to_hours(label) - hours) < 0.1:
                return label
        return None
    
    def _generate_shuffled_pairs(self):
        """
        Generate n_samples shuffled (x_start, x_end, t_start, t_end) pairs
        
        Algorithm:
        1. Sample time interval Δt from distribution
        2. Sample valid start time t_start
        3. Calculate end time t_end = t_start + Δt
        4. Randomly sample cells from t_start and t_end pools
        
        Returns:
            List of (x_start, x_end, t_start_idx, t_end_idx) tuples
        """
        pairs = []
        
        # Create time to hours mapping
        time_to_hours = {t: self._label_to_hours(t) for t in self.time_labels}
        time_to_idx = {t: idx for idx, t in enumerate(self.time_labels)}
        max_hours = max(time_to_hours.values())
        
        attempts = 0
        max_attempts = self.n_samples * 10  # Prevent infinite loop
        
        while len(pairs) < self.n_samples and attempts < max_attempts:
            attempts += 1
            
            # 1. Sample time interval
            delta_t = np.random.choice(self.delta_t_values, p=self.delta_t_probs)
            
            # 2. Sample start time (must allow for delta_t)
            valid_starts = [t for t in self.time_labels 
                           if time_to_hours[t] + delta_t <= max_hours + 0.1]
            
            if not valid_starts:
                continue
                
            t_start = np.random.choice(valid_starts)
            
            # 3. Calculate end time
            t_end_hours = time_to_hours[t_start] + delta_t
            t_end = self._hours_to_label(t_end_hours)
            
            if t_end is None or t_end not in self.time_labels:
                continue
            
            # 4. Sample cells
            if len(self.cell_pools[t_start]) == 0 or len(self.cell_pools[t_end]) == 0:
                continue
                
            x_start = self.cell_pools[t_start][np.random.randint(len(self.cell_pools[t_start]))]
            x_end = self.cell_pools[t_end][np.random.randint(len(self.cell_pools[t_end]))]
            
            # Store as (x_start, x_end, t_start_idx, t_end_idx)
            pairs.append((
                x_start.copy(),
                x_end.copy(),
                time_to_idx[t_start],
                time_to_idx[t_end]
            ))
        
        if len(pairs) < self.n_samples:
            print(f"Warning: Only generated {len(pairs)} pairs out of {self.n_samples} requested")
        
        return pairs
    
    def __len__(self):
        return len(self.shuffled_pairs)
    
    def __getitem__(self, idx):
        """
        Get a shuffled training pair
        
        Returns:
            (x_start, x_end): Tuple of torch tensors
        """
        x_start, x_end, t_start_idx, t_end_idx = self.shuffled_pairs[idx]
        
        return (
            torch.tensor(x_start, dtype=torch.float32),
            torch.tensor(x_end, dtype=torch.float32)
        )
    
    def get_time_interval_statistics(self) -> Dict:
        """
        Get statistics about the generated time intervals
        
        Returns:
            Dict with interval distribution statistics
        """
        intervals = []
        time_to_hours = {t: self._label_to_hours(t) for t in self.time_labels}
        
        for _, _, t_start_idx, t_end_idx in self.shuffled_pairs:
            t_start = self.time_labels[t_start_idx]
            t_end = self.time_labels[t_end_idx]
            delta_t = time_to_hours[t_end] - time_to_hours[t_start]
            intervals.append(delta_t)
        
        intervals = np.array(intervals)
        
        # Count each interval
        interval_counts = {}
        for dt in self.delta_t_values:
            count = np.sum(np.abs(intervals - dt) < 0.1)
            interval_counts[dt] = count
        
        return {
            'interval_counts': interval_counts,
            'interval_distribution': {dt: count / len(intervals) 
                                     for dt, count in interval_counts.items()},
            'expected_distribution': dict(zip(self.delta_t_values, self.delta_t_probs)),
            'total_pairs': len(intervals)
        }
    
    def validate_distribution(self, alpha: float = 0.05) -> bool:
        """
        Validate that generated interval distribution matches expected distribution
        
        Uses Kolmogorov-Smirnov test
        
        Args:
            alpha: Significance level
            
        Returns:
            True if distribution is valid (p > alpha)
        """
        from scipy.stats import ks_2samp
        
        stats = self.get_time_interval_statistics()
        
        # Generate expected samples
        expected_intervals = np.random.choice(
            self.delta_t_values,
            size=len(self.shuffled_pairs),
            p=self.delta_t_probs
        )
        
        # Get actual intervals
        intervals = []
        time_to_hours = {t: self._label_to_hours(t) for t in self.time_labels}
        for _, _, t_start_idx, t_end_idx in self.shuffled_pairs:
            t_start = self.time_labels[t_start_idx]
            t_end = self.time_labels[t_end_idx]
            delta_t = time_to_hours[t_end] - time_to_hours[t_start]
            intervals.append(delta_t)
        
        # KS test
        statistic, pvalue = ks_2samp(intervals, expected_intervals)
        
        print(f"\n✓ Distribution validation:")
        print(f"  - KS statistic: {statistic:.4f}")
        print(f"  - p-value: {pvalue:.4f}")
        print(f"  - Valid (p > {alpha}): {pvalue > alpha}")
        
        return pvalue > alpha


if __name__ == "__main__":
    """Test the shuffled dataset"""
    import scanpy as sc
    
    # Load test data
    file_path = "/home/pan/Experiments/EXPs/2025_10_VCC_Exps/DATAs/EMT/2024_12_04_Cook_emt_dataset_with_removal.h5ad"
    adata = sc.read_h5ad(file_path)
    
    # Filter to forward EMT timepoints
    time_labels = ['0d', '8h', '1d', '3d', '7d']
    mask = adata.obs['Ground_truth'].isin(time_labels)
    adata = adata[mask].copy()
    
    # Define time intervals
    time_intervals = {
        '0d-8h': 8,
        '8h-1d': 16,
        '1d-3d': 48,
        '3d-7d': 96
    }
    
    # Create shuffled dataset
    print("="*70)
    print("Testing ShuffledTimeSeriesDataset")
    print("="*70)
    
    dataset = ShuffledTimeSeriesDataset(
        adata=adata,
        time_labels=time_labels,
        time_intervals=time_intervals,
        n_samples=8974,
        seed=42
    )
    
    # Test __getitem__
    x_start, x_end = dataset[0]
    print(f"\n✓ Sample pair shapes: {x_start.shape}, {x_end.shape}")
    
    # Get statistics
    print("\n" + "="*70)
    print("Time Interval Statistics")
    print("="*70)
    stats = dataset.get_time_interval_statistics()
    print("\nExpected distribution:")
    for dt, prob in stats['expected_distribution'].items():
        print(f"  Δt={dt}h: {prob:.3f}")
    
    print("\nActual distribution:")
    for dt, prob in stats['interval_distribution'].items():
        print(f"  Δt={dt}h: {prob:.3f} (count={stats['interval_counts'][dt]})")
    
    # Validate distribution
    print("\n" + "="*70)
    print("Distribution Validation")
    print("="*70)
    is_valid = dataset.validate_distribution()
    
    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)
