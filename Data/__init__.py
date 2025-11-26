"""
Data loading and dataset building modules for real time series data

Base Classes:
- BaseDataLoader: Abstract base class for all data loaders

Concrete Data Loaders:
- RealDataLoader: Load real h5ad data without transformation
- InterpolatedDataLoader: Load data with interpolated intermediate states

Dataset Builders:
- DatasetBuilder: Build PyTorch datasets from arrays
- TimeSeriesDataset: PyTorch dataset for time series data
- LabelShuffledDataset: Dataset with randomized time labels (Setting 8)

Configuration:
- ConfigLoader: Load and validate experiment configurations
"""

from .base_data_loader import BaseDataLoader
from .data_loader import RealDataLoader, create_default_emt_data_loader
from .interpolated_data_loader import InterpolatedDataLoader
from .dataset_builder import (
    DatasetBuilder, 
    TimeSeriesDataset, 
    create_dataloaders_from_data
)
from .label_shuffled_dataset import LabelShuffledDataset, create_label_shuffled_datasets
from .config_loader import (
    ConfigLoader,
    setup_logging,
    create_data_loader_from_config,
    validate_data_config,
    get_data_for_setting
)
from .entropy_utils import (
    estimate_entropy_knn,
    estimate_entropy_gaussian,
    estimate_entropy_both_methods,
    batch_estimate_entropy,
    compute_entropy_by_timepoint
)

__all__ = [
    # Base classes
    "BaseDataLoader",
    
    # Data loaders
    "RealDataLoader",
    "create_default_emt_data_loader",
    "InterpolatedDataLoader",
    
    # Dataset builders
    "DatasetBuilder",
    "TimeSeriesDataset",
    "LabelShuffledDataset",
    "create_dataloaders_from_data",
    "create_label_shuffled_datasets",
    
    # Configuration
    "ConfigLoader",
    "setup_logging",
    "create_data_loader_from_config",
    "validate_data_config",
    "get_data_for_setting",
    
    # Entropy utilities
    "estimate_entropy_knn",
    "estimate_entropy_gaussian",
    "estimate_entropy_both_methods",
    "batch_estimate_entropy",
    "compute_entropy_by_timepoint",
]
