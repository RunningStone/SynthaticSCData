#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Run Visualization for Multi-Setting Experiments
========================================================

This script loads evaluation results and generated data from multiple experiment settings
and creates comprehensive visualizations for comparison.

Features:
- Auto-discover Setting* folders in the experiment directory
- Load evaluation_results.json for metrics comparison
- Load generated_data/*.pkl for trajectory visualization
- Compute PHATE and LMNN+PCA embeddings
- Save visualizations and embeddings to vis/ folder

Usage:
    python step4_run_vis.py --experiment_dir /path/to/EMT_E2M
    
    # With custom output directory
    python step4_run_vis.py --experiment_dir /path/to/EMT_E2M --output_dir /path/to/vis

Output:
    - {experiment_dir}/vis/metrics_comparison.png/pdf/csv
    - {experiment_dir}/vis/generation_comparison_phate.png/pdf
    - {experiment_dir}/vis/generation_comparison_lmnn_pca.png/pdf
    - {experiment_dir}/vis/embeddings.pkl (cached embeddings for reuse)
"""

import argparse
import torch
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from Analyser import (
    DataManager,
    EmbeddingComputer,
    MetricsPlotter,
    GenerationPlotter
)
from Data import (
    create_data_loader_from_config,
    get_data_for_setting
)


def create_logger(output_dir: Path, log_name: str = "visualization") -> logging.Logger:
    """Create a logger for visualization"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('visualization')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_dir / f"{log_name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def discover_settings(experiment_dir: Path, logger: Optional[logging.Logger] = None) -> Dict[str, Path]:
    """
    Discover all Setting* folders in the experiment directory that have evaluation_results.json.
    
    Handles various naming patterns:
    - Setting1, Setting2, ...
    - Setting5_LabelShuffled
    - Setting4/Setting4_Ablation_Remove1d (nested ablation)
    
    Args:
        experiment_dir: Root experiment directory
        logger: Optional logger for warnings
        
    Returns:
        Dictionary mapping setting names to their paths (only those with evaluation_results.json)
    """
    settings = {}
    skipped = []
    
    for item in sorted(experiment_dir.iterdir()):
        if not item.is_dir():
            continue
            
        # Match any folder starting with "Setting" (e.g., Setting1, Setting5_LabelShuffled)
        if item.name.startswith('Setting'):
            # Must have evaluation_results.json
            has_eval_results = (item / 'evaluation_results.json').exists()
            
            if has_eval_results:
                settings[item.name] = item
            else:
                # Check for nested experiments (e.g., Setting4/Setting4_Ablation_Remove1d)
                has_nested = False
                for sub_item in sorted(item.iterdir()):
                    if sub_item.is_dir():
                        sub_has_eval_results = (sub_item / 'evaluation_results.json').exists()
                        if sub_has_eval_results:
                            settings[sub_item.name] = sub_item
                            has_nested = True
                
                # If no nested settings found, record as skipped
                if not has_nested:
                    skipped.append(item.name)
    
    # Log skipped settings
    if skipped and logger:
        for name in skipped:
            logger.warning(f"Skipping {name}: evaluation_results.json not found (run step3_run_inference.py first)")
    
    return settings


def load_setting_data(
    setting_path: Path,
    data_manager: DataManager,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Load evaluation results and generated data for a single setting.
    
    Args:
        setting_path: Path to the setting directory
        data_manager: DataManager instance
        logger: Logger instance
        
    Returns:
        Dictionary containing metrics and generated data for each model
    """
    result = {
        'metrics': {},
        'generated': {},
        'path': str(setting_path)
    }
    
    # Load evaluation_results.json
    results_path = setting_path / 'evaluation_results.json'
    if results_path.exists():
        with open(results_path, 'r') as f:
            eval_results = json.load(f)
        
        for model_name, model_data in eval_results.items():
            # Extract metrics (handle nested structure)
            if 'metrics' in model_data:
                result['metrics'][model_name] = model_data['metrics']
            else:
                result['metrics'][model_name] = model_data
        
        logger.debug(f"  Loaded metrics for models: {list(result['metrics'].keys())}")
    else:
        logger.warning(f"  evaluation_results.json not found in {setting_path}")
    
    # Load generated data from generated_data folder
    generated_dir = setting_path / 'generated_data'
    if generated_dir.exists():
        for pkl_file in generated_dir.glob('*.pkl'):
            model_name = pkl_file.stem
            gen_data = data_manager.load_generated_pkl(pkl_file)
            if gen_data is not None:
                result['generated'][model_name] = gen_data
        
        logger.debug(f"  Loaded generated data for models: {list(result['generated'].keys())}")
    else:
        logger.warning(f"  generated_data folder not found in {setting_path}")
    
    return result


def aggregate_all_data(
    settings_dict: Dict[str, Path],
    data_manager: DataManager,
    logger: logging.Logger
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Aggregate metrics and generated data from all settings.
    
    Args:
        settings_dict: Dictionary mapping setting names to paths
        data_manager: DataManager instance
        logger: Logger instance
        
    Returns:
        Tuple of (all_metrics, all_generated) dictionaries
    """
    all_metrics = {}
    all_generated = {}
    
    for setting_name, setting_path in settings_dict.items():
        logger.info(f"Loading data from {setting_name}...")
        
        data = load_setting_data(setting_path, data_manager, logger)
        
        # Aggregate metrics with setting prefix
        for model_name, metrics in data['metrics'].items():
            key = f"{setting_name}/{model_name}"
            all_metrics[key] = metrics
        
        # Aggregate generated data with setting prefix
        for model_name, gen_data in data['generated'].items():
            key = f"{setting_name}/{model_name}"
            all_generated[key] = gen_data
    
    return all_metrics, all_generated


def load_real_test_data(
    experiment_base: Path,
    settings_dict: Dict[str, Path],
    logger: logging.Logger,
    sample_ratio: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load real test data from Setting2 configuration.
    
    Args:
        experiment_base: Base experiment directory
        settings_dict: Dictionary of discovered settings
        logger: Logger instance
        sample_ratio: Ratio of test data to sample (default 0.2 = 20%)
        
    Returns:
        Tuple of (X_test, y_test, time_labels)
    """
    # Find Setting2 config (preferred) or any available setting config
    config_path = None
    for setting_name in ['Setting2', 'Setting1', 'Setting3']:
        if setting_name in settings_dict:
            candidate = settings_dict[setting_name] / 'experiment_config.yaml'
            if candidate.exists():
                config_path = candidate
                logger.info(f"Using config from {setting_name} for real data loading")
                break
    
    if config_path is None:
        # Try any available setting
        for setting_name, setting_path in settings_dict.items():
            candidate = setting_path / 'experiment_config.yaml'
            if candidate.exists():
                config_path = candidate
                logger.info(f"Using config from {setting_name} for real data loading")
                break
    
    if config_path is None:
        raise ValueError("No experiment_config.yaml found in any setting directory")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loader
    logger.info("Creating data loader from config...")
    data_loader = create_data_loader_from_config(config, logger)
    data_loader.load_and_analyze()
    
    # Get test data using Setting2 logic (all timepoints)
    logger.info("Loading test data (all timepoints)...")
    X_train, y_train, X_test, y_test = get_data_for_setting(data_loader, config, logger)
    
    # Get time labels
    time_labels = data_loader.time_label_order
    
    logger.info(f"Full test data: {X_test.shape[0]} samples")
    
    # Sample test data
    if sample_ratio < 1.0:
        n_samples = int(X_test.shape[0] * sample_ratio)
        logger.info(f"Sampling {sample_ratio*100:.0f}% of test data: {n_samples} samples")
        
        # Stratified sampling by time label
        sampled_indices = []
        unique_labels = np.unique(y_test)
        for label in unique_labels:
            label_indices = np.where(y_test == label)[0]
            n_label_samples = max(1, int(len(label_indices) * sample_ratio))
            sampled = np.random.choice(label_indices, size=n_label_samples, replace=False)
            sampled_indices.extend(sampled)
        
        sampled_indices = np.array(sampled_indices)
        X_test = X_test[sampled_indices]
        y_test = y_test[sampled_indices]
        
        logger.info(f"Sampled test data: {X_test.shape[0]} samples")
        for label in unique_labels:
            count = np.sum(y_test == label)
            logger.info(f"  {time_labels[label]}: {count} samples")
    
    return X_test, y_test, time_labels


def compute_embeddings(
    real_data: np.ndarray,
    real_labels: np.ndarray,
    time_labels: List[str],
    all_generated: Dict[str, Dict],
    embedding_computer: EmbeddingComputer,
    logger: logging.Logger,
    output_dir: Path,
    force_recompute: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute embeddings for real data and all generated data together.
    
    Args:
        real_data: Real test data array (n_samples, n_features)
        real_labels: Real test labels array (n_samples,)
        time_labels: List of time label strings
        all_generated: Dictionary of generated data from all settings/models
        embedding_computer: EmbeddingComputer instance
        logger: Logger instance
        output_dir: Output directory for caching
        force_recompute: Force recomputation even if cache exists
        
    Returns:
        Dictionary with 'phate' and 'lmnn_pca' embeddings
    """
    cache_path = output_dir / 'embeddings.pkl'
    
    # Try to load from cache
    if cache_path.exists() and not force_recompute:
        logger.info("Loading cached embeddings...")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        return cached['embeddings_dict']
    
    # Collect all generated data
    logger.info("Collecting generated data for embedding computation...")
    all_gen_data = []
    all_gen_labels = []  # Will be set to end_timepoint label
    gen_data_info = []  # Track which data belongs to which model
    
    end_timepoint_idx = len(time_labels) - 1  # Last timepoint index
    
    for model_key, data in all_generated.items():
        if data is not None and 'generated_data' in data:
            gen_data = data['generated_data']
            if gen_data is not None and len(gen_data) > 0:
                all_gen_data.append(gen_data)
                # All generated samples are for the end timepoint
                all_gen_labels.append(np.full(len(gen_data), end_timepoint_idx))
                gen_data_info.append({
                    'model_key': model_key,
                    'start_idx': sum(len(d) for d in all_gen_data[:-1]),
                    'end_idx': sum(len(d) for d in all_gen_data),
                    'n_samples': len(gen_data)
                })
                logger.debug(f"  {model_key}: {len(gen_data)} generated samples")
    
    # Combine real and generated data for joint embedding
    if all_gen_data:
        combined_data = np.vstack([real_data] + all_gen_data)
        combined_labels = np.concatenate([real_labels] + all_gen_labels)
    else:
        combined_data = real_data
        combined_labels = real_labels
    
    logger.info(f"Combined data shape: {combined_data.shape}")
    logger.info(f"  Real data: {real_data.shape[0]} samples")
    logger.info(f"  Generated data: {combined_data.shape[0] - real_data.shape[0]} samples")
    
    # Compute embeddings on combined data
    logger.info("Computing PHATE and LMNN+PCA embeddings on combined data...")
    embeddings_dict = embedding_computer.compute_all_embeddings(
        combined_data, combined_labels
    )
    
    # Cache embeddings with metadata
    logger.info(f"Caching embeddings to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'embeddings_dict': embeddings_dict,
            'real_data': real_data,
            'real_labels': real_labels,
            'time_labels': time_labels,
            'n_real_samples': len(real_data),
            'gen_data_info': gen_data_info
        }, f)
    
    return embeddings_dict


def plot_metrics_comparison(
    all_metrics: Dict[str, Dict],
    metrics_plotter: MetricsPlotter,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Plot metrics comparison across all settings and models.
    
    Args:
        all_metrics: Dictionary of all metrics
        metrics_plotter: MetricsPlotter instance
        output_dir: Output directory
        logger: Logger instance
    """
    if not all_metrics:
        logger.warning("No metrics data available for plotting")
        return
    
    logger.info("Plotting metrics comparison...")
    
    # Wrap metrics in expected format
    metrics_wrapped = {}
    for key, metrics in all_metrics.items():
        metrics_wrapped[key] = {'evaluation': metrics}
    
    saved_paths = metrics_plotter.plot_metrics_comparison(
        metrics_wrapped,
        title="Multi-Setting Metrics Comparison",
        save_prefix="metrics_comparison",
        output_dir=output_dir
    )
    
    logger.info(f"Metrics comparison saved: {[str(p) for p in saved_paths]}")


def plot_generation_comparison(
    all_generated: Dict[str, Dict],
    embeddings_dict: Dict[str, np.ndarray],
    real_labels: np.ndarray,
    n_real_samples: int,
    time_labels: List[str],
    generation_plotter: GenerationPlotter,
    output_dir: Path,
    logger: logging.Logger
):
    """
    Plot generation comparison across all settings and models.
    
    The embeddings_dict contains embeddings for combined data:
    - First n_real_samples rows are real test data (all timepoints)
    - Remaining rows are generated data (end timepoint only)
    
    Args:
        all_generated: Dictionary of all generated data
        embeddings_dict: Pre-computed embeddings for combined data
        real_labels: Real test data labels (n_real_samples,)
        n_real_samples: Number of real samples in embeddings
        time_labels: List of time label strings
        generation_plotter: GenerationPlotter instance
        output_dir: Output directory
        logger: Logger instance
    """
    if not all_generated:
        logger.warning("No generated data available for plotting")
        return
    
    # Build index mapping for generated data
    gen_start_idx = n_real_samples
    model_indices = {}
    for model_key, data in all_generated.items():
        if data is not None and 'generated_data' in data:
            gen_data = data['generated_data']
            if gen_data is not None and len(gen_data) > 0:
                model_indices[model_key] = {
                    'start': gen_start_idx,
                    'end': gen_start_idx + len(gen_data)
                }
                gen_start_idx += len(gen_data)
    
    for embedding_type in ['phate', 'lmnn_pca']:
        logger.info(f"Processing {embedding_type.upper()} embeddings...")
        
        # Get full embedding
        full_embedding = embeddings_dict[embedding_type]
        
        # Extract real embedding (first n_real_samples rows)
        real_embedding = full_embedding[:n_real_samples]
        
        # Extract generated embeddings for each model
        model_embeddings = {'original': real_embedding}
        
        for model_key, indices in model_indices.items():
            gen_emb = full_embedding[indices['start']:indices['end']]
            model_embeddings[model_key] = gen_emb
            logger.debug(f"  {model_key}: {len(gen_emb)} samples")
        
        # Plot comparison grid
        if len(model_embeddings) > 1:  # More than just 'original'
            saved_paths = generation_plotter.plot_comparison_grid(
                model_embeddings,
                real_embedding,
                real_labels,
                time_labels if time_labels else [str(i) for i in range(int(real_labels.max()) + 1)],
                embedding_type,
                "Multi-Setting Generation Comparison",
                "generation_comparison",
                output_dir
            )
            logger.info(f"{embedding_type.upper()} comparison saved: {[str(p) for p in saved_paths]}")


def run_visualization(
    experiment_dir: str,
    output_dir: Optional[str] = None,
    force_recompute: bool = False,
    sample_ratio: float = 1.0,
    seed: int = 42
) -> Dict:
    """
    Run visualization pipeline for multi-setting experiments.
    
    Args:
        experiment_dir: Root directory containing Setting* folders
        output_dir: Output directory for visualizations (default: {experiment_dir}/vis)
        force_recompute: Force recomputation of embeddings
        sample_ratio: Ratio of test data to sample for visualization (default 1.0 = 100%)
        seed: Random seed
        
    Returns:
        Dictionary containing visualization results
    """
    # Setup paths
    experiment_base = Path(experiment_dir)
    
    if not experiment_base.exists():
        raise ValueError(f"Experiment directory not found: {experiment_dir}")
    
    # Setup output directory
    if output_dir is None:
        vis_dir = experiment_base / 'vis'
    else:
        vis_dir = Path(output_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = create_logger(vis_dir, "visualization")
    
    logger.info("="*80)
    logger.info("VISUALIZATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Output directory: {vis_dir}")
    logger.info(f"Sample ratio: {sample_ratio*100:.0f}%")
    logger.info("")
    
    # Set random seed
    np.random.seed(seed)
    
    # Initialize components
    data_manager = DataManager()
    embedding_computer = EmbeddingComputer(random_seed=seed)
    metrics_plotter = MetricsPlotter()
    generation_plotter = GenerationPlotter()
    
    # Discover settings
    logger.info("Discovering experiment settings...")
    settings_dict = discover_settings(experiment_base, logger)
    
    if not settings_dict:
        logger.error("No Setting* folders found with evaluation results!")
        return {}
    
    logger.info(f"Found {len(settings_dict)} settings: {list(settings_dict.keys())}")
    
    # Aggregate all data
    logger.info("\nAggregating data from all settings...")
    all_metrics, all_generated = aggregate_all_data(settings_dict, data_manager, logger)
    
    logger.info(f"\nTotal metrics entries: {len(all_metrics)}")
    logger.info(f"Total generated data entries: {len(all_generated)}")
    
    # Load real test data from Setting2 config
    logger.info("\nLoading real test data from Setting2 config...")
    real_data, real_labels, time_labels = load_real_test_data(
        experiment_base, settings_dict, logger, sample_ratio
    )
    
    # Compute embeddings on combined real + generated data
    logger.info("\nComputing embeddings...")
    embeddings_dict = compute_embeddings(
        real_data, real_labels, time_labels, all_generated,
        embedding_computer, logger, vis_dir, force_recompute
    )
    
    # Plot metrics comparison
    logger.info("\nGenerating metrics visualization...")
    plot_metrics_comparison(all_metrics, metrics_plotter, vis_dir, logger)
    
    # Plot generation comparison
    logger.info("\nGenerating trajectory visualization...")
    plot_generation_comparison(
        all_generated, embeddings_dict, real_labels, len(real_data), time_labels,
        generation_plotter, vis_dir, logger
    )
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("VISUALIZATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nOutput files saved to: {vis_dir}")
    logger.info("\nGenerated files:")
    for f in sorted(vis_dir.glob('*')):
        if f.is_file() and f.name != 'visualization.log':
            logger.info(f"  - {f.name}")
    
    return {
        'output_dir': str(vis_dir),
        'settings': list(settings_dict.keys()),
        'n_metrics': len(all_metrics),
        'n_generated': len(all_generated),
        'n_real_samples': len(real_data)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run visualization for multi-setting experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run visualization for EMT_E2M experiments
  python step4_run_vis.py --experiment_dir /path/to/EMT_E2M

  # With custom output directory
  python step4_run_vis.py --experiment_dir /path/to/EMT_E2M --output_dir /path/to/vis

  # Force recompute embeddings
  python step4_run_vis.py --experiment_dir /path/to/EMT_E2M --force_recompute
        """
    )
    
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=True,
        help='Root experiment directory containing Setting* folders'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for visualizations (default: {experiment_dir}/vis)'
    )
    parser.add_argument(
        '--force_recompute',
        action='store_true',
        help='Force recomputation of embeddings even if cache exists'
    )
    parser.add_argument(
        '--sample_ratio',
        type=float,
        default=1.0,
        help='Ratio of test data to sample for visualization (default: 1.0 = 100%%)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_visualization(
            experiment_dir=args.experiment_dir,
            output_dir=args.output_dir,
            force_recompute=args.force_recompute,
            sample_ratio=args.sample_ratio,
            seed=args.seed
        )
        return results
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
