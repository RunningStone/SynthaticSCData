#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Loader for Time Series Single-Cell Data

Loads real h5ad files and provides analysis and validation.
Inherits all functionality from BaseDataLoader.
"""

from typing import Optional
from .base_data_loader import BaseDataLoader


class RealDataLoader(BaseDataLoader):
    """
    Load and analyze real single-cell time series data.
    
    This is the standard data loader that reads h5ad files directly
    without any transformation. All functionality is inherited from
    BaseDataLoader.
    
    Use this for:
    - Standard time series experiments
    - Real biological data without augmentation
    - Baseline comparisons
    """
    
    # All methods inherited from BaseDataLoader
    # No additional customization needed for real data loading
    pass


def create_default_emt_data_loader(
    file_path: Optional[str] = None,
    n_hvg: int = 100
) -> RealDataLoader:
    """
    Create data loader with default EMT dataset configuration.
    
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
