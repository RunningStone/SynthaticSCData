#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 - Figure 1: Core Performance Comparison
===============================================

This script generates Figure 1 for the paper:
- Figure 1.1: Performance radar chart comparing Setting1 vs Setting2
- Figure 1.2: PHATE 3x3 grid showing generation quality

Required Settings:
- Setting1: Boundary-only (0d, 7d)
- Setting2: Full trajectory (0d, 8h, 1d, 3d, 7d)

Usage:
    python step4_run_vis_fig1.py --experiment_dir /path/to/EMT_E2M

Output:
    - {experiment_dir}/vis/Fig1_1.pdf  (Radar chart)
    - {experiment_dir}/vis/Fig1_2.pdf  (PHATE 3x3 grid)
"""

import argparse
import json
import pickle
import logging
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Analyser import (
    DataManager,
    EmbeddingComputer,
    plot_performance_radar,
    plot_phate_dynamic_grid,
    extract_metrics_from_results,
    METRICS_CONFIG,
)
from Analyser.value_checker import ValueChecker
from Data import (
    create_data_loader_from_config,
    get_data_for_setting
)


def setup_logger(output_dir: Path) -> logging.Logger:
    """Setup logger for the script."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('fig1_visualization')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "fig1_visualization.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_evaluation_results(setting_path: Path) -> Dict:
    """Load evaluation_results.json from a setting directory."""
    results_path = setting_path / 'evaluation_results.json'
    if not results_path.exists():
        raise FileNotFoundError(f"evaluation_results.json not found in {setting_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def load_generated_data(setting_path: Path, model_name: str) -> Optional[Dict]:
    """Load generated data PKL file for a model."""
    pkl_path = setting_path / 'generated_data' / f'{model_name}.pkl'
    if not pkl_path.exists():
        return None
    
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def load_real_test_data(
    experiment_base: Path,
    setting_path: Path,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load real test data from Setting2 configuration.
    
    Returns:
        Tuple of (X_test, y_test, time_labels)
    """
    config_path = setting_path / 'experiment_config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"experiment_config.yaml not found in {setting_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Creating data loader from config...")
    data_loader = create_data_loader_from_config(config, logger)
    data_loader.load_and_analyze()
    
    logger.info("Loading test data...")
    X_train, y_train, X_test, y_test = get_data_for_setting(data_loader, config, logger)
    
    time_labels = data_loader.time_label_order
    
    logger.info(f"Test data: {X_test.shape[0]} samples, {len(time_labels)} timepoints")
    
    return X_test, y_test, time_labels


def compute_phate_embeddings(
    real_data: np.ndarray,
    real_labels: np.ndarray,
    generated_data_dict: Dict[str, np.ndarray],
    logger: logging.Logger,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    """
    Compute PHATE embeddings for real and generated data.
    
    Returns:
        Tuple of (combined_phate_coords, model_phate_coords_dict, n_real_samples)
    """
    # Check cache
    if cache_path and cache_path.exists() and not force_recompute:
        logger.info(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        return cached['phate_coords'], cached['model_coords'], cached['n_real']
    
    # Combine all data for joint embedding
    all_data = [real_data]
    model_indices = {}
    current_idx = len(real_data)
    
    for model_name, gen_data in generated_data_dict.items():
        if gen_data is not None and len(gen_data) > 0:
            model_indices[model_name] = (current_idx, current_idx + len(gen_data))
            all_data.append(gen_data)
            current_idx += len(gen_data)
    
    combined_data = np.vstack(all_data)
    logger.info(f"Combined data shape: {combined_data.shape}")
    
    # Check for NaN values and handle them
    nan_count = np.isnan(combined_data).sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in combined data, replacing with 0")
        combined_data = np.nan_to_num(combined_data, nan=0.0)
    
    # Check for infinite values
    inf_count = np.isinf(combined_data).sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} infinite values in combined data, clipping")
        combined_data = np.clip(combined_data, -1e10, 1e10)
    
    # Compute PHATE
    logger.info("Computing PHATE embeddings...")
    embedding_computer = EmbeddingComputer(random_seed=42)
    phate_coords = embedding_computer.fit_transform_phate(combined_data)
    
    # Extract model coordinates
    model_coords = {}
    for model_name, (start_idx, end_idx) in model_indices.items():
        model_coords[model_name] = phate_coords[start_idx:end_idx]
    
    # Cache results
    if cache_path:
        logger.info(f"Caching embeddings to {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'phate_coords': phate_coords,
                'model_coords': model_coords,
                'n_real': len(real_data)
            }, f)
    
    return phate_coords, model_coords, len(real_data)


def run_fig1_visualization(
    experiment_dir: str,
    force_recompute: bool = False,
    seed: int = 42
) -> Dict:
    """
    Run Figure 1 visualization pipeline.
    
    Args:
        experiment_dir: Root directory containing Setting* folders
        force_recompute: Force recomputation of embeddings
        seed: Random seed
    
    Returns:
        Dictionary containing paths to generated figures
    """
    experiment_base = Path(experiment_dir)
    vis_dir = experiment_base / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Figure1 subfolder
    fig1_dir = vis_dir / 'Figure1'
    fig1_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(vis_dir)
    
    logger.info("=" * 80)
    logger.info("FIGURE 1: Core Performance Comparison")
    logger.info("=" * 80)
    
    np.random.seed(seed)
    
    # Check required settings exist
    setting1_path = experiment_base / 'Setting1'
    setting2_path = experiment_base / 'Setting2'
    
    if not setting1_path.exists():
        raise FileNotFoundError(f"Setting1 not found at {setting1_path}")
    if not setting2_path.exists():
        raise FileNotFoundError(f"Setting2 not found at {setting2_path}")
    
    # =========================================================================
    # Load evaluation results
    # =========================================================================
    logger.info("\n--- Loading Evaluation Results ---")
    
    results_s1 = load_evaluation_results(setting1_path)
    results_s2 = load_evaluation_results(setting2_path)
    
    logger.info(f"Setting1 models: {list(results_s1.keys())}")
    logger.info(f"Setting2 models: {list(results_s2.keys())}")
    
    # Extract metrics
    metrics_s1 = {model: extract_metrics_from_results(results_s1, model) 
                  for model in results_s1.keys()}
    metrics_s2 = {model: extract_metrics_from_results(results_s2, model) 
                  for model in results_s2.keys()}
    
    # =========================================================================
    # Figure 1.1: Performance Radar Chart
    # =========================================================================
    logger.info("\n--- Generating Figure 1.1: Performance Radar Chart ---")
    
    fig1_1_path = fig1_dir / 'Fig1_1.pdf'
    plot_performance_radar(
        metrics_setting1=metrics_s1,
        metrics_setting2=metrics_s2,
        output_path=fig1_1_path,
        title="Performance Comparison: Setting1 (Boundary) vs Setting2 (Full Trajectory)",
        figsize=(12, 12),
        dpi=300
    )
    logger.info(f"Saved: {fig1_1_path}")
    
    # Also save PNG version
    fig1_1_png = fig1_dir / 'Fig1_1.png'
    plot_performance_radar(
        metrics_setting1=metrics_s1,
        metrics_setting2=metrics_s2,
        output_path=fig1_1_png,
        title="Performance Comparison: Setting1 (Boundary) vs Setting2 (Full Trajectory)",
        figsize=(12, 12),
        dpi=300
    )
    
    # =========================================================================
    # Figure 1.2: PHATE 3x3 Grid
    # =========================================================================
    logger.info("\n--- Generating Figure 1.2: PHATE 3x3 Grid ---")
    
    # Load real test data (use Setting2 config for all timepoints)
    logger.info("Loading real test data...")
    X_test, y_test, time_labels = load_real_test_data(experiment_base, setting2_path, logger)
    
    # Load generated data for all models
    logger.info("Loading generated data...")
    
    # Initialize value checker
    value_checker = ValueChecker(nan_threshold=0.1, logger=logger)
    
    # Setting1 models - dynamically load all available models from evaluation results
    gen_data_s1 = {}
    available_models_s1 = list(results_s1.keys())
    logger.info(f"Available models in Setting1: {available_models_s1}")
    for model_name in available_models_s1:
        gen_pkl = load_generated_data(setting1_path, model_name)
        if gen_pkl is not None and 'generated_data' in gen_pkl:
            gen_data_s1[model_name] = gen_pkl['generated_data']
            logger.info(f"  Setting1/{model_name}: {len(gen_pkl['generated_data'])} samples")
    
    # Validate Setting1 models (filter out NaN models)
    gen_data_s1, skipped_s1 = value_checker.filter_valid_models(gen_data_s1, results_s1)
    if skipped_s1:
        logger.warning(f"Skipped Setting1 models due to invalid data: {skipped_s1}")
    logger.info(f"Valid Setting1 models after filtering: {list(gen_data_s1.keys())}")
    
    # Setting2 models - dynamically load all available models from evaluation results
    gen_data_s2 = {}
    available_models_s2 = list(results_s2.keys())
    logger.info(f"Available models in Setting2: {available_models_s2}")
    for model_name in available_models_s2:
        gen_pkl = load_generated_data(setting2_path, model_name)
        if gen_pkl is not None and 'generated_data' in gen_pkl:
            gen_data_s2[model_name] = gen_pkl['generated_data']
            logger.info(f"  Setting2/{model_name}: {len(gen_pkl['generated_data'])} samples")
    
    # Validate Setting2 models (filter out NaN models)
    gen_data_s2, skipped_s2 = value_checker.filter_valid_models(gen_data_s2, results_s2)
    if skipped_s2:
        logger.warning(f"Skipped Setting2 models due to invalid data: {skipped_s2}")
    logger.info(f"Valid Setting2 models after filtering: {list(gen_data_s2.keys())}")
    
    # Check if we have any valid models to visualize
    if not gen_data_s1 and not gen_data_s2:
        logger.error("No valid models to visualize after filtering. Skipping PHATE visualization.")
        return {
            'fig1_1': str(fig1_1_path),
            'fig1_2': None,
            'skipped_models': {'Setting1': skipped_s1, 'Setting2': skipped_s2}
        }
    
    # Combine all generated data for joint embedding
    all_generated = {**{f's1_{k}': v for k, v in gen_data_s1.items()},
                     **{f's2_{k}': v for k, v in gen_data_s2.items()}}
    
    # Compute PHATE embeddings
    cache_path = fig1_dir / 'fig1_phate_cache.pkl'
    phate_coords, model_coords, n_real = compute_phate_embeddings(
        X_test, y_test, all_generated, logger, cache_path, force_recompute
    )
    
    # Separate model coordinates by setting
    gen_coords_s1 = {k.replace('s1_', ''): v for k, v in model_coords.items() if k.startswith('s1_')}
    gen_coords_s2 = {k.replace('s2_', ''): v for k, v in model_coords.items() if k.startswith('s2_')}
    
    # Determine start and end timepoints
    start_tp = time_labels[0] if time_labels else '0d'
    end_tp = time_labels[-1] if time_labels else '7d'
    
    # Prepare data for dynamic grid
    generated_coords_by_setting = {}
    if gen_coords_s1:
        generated_coords_by_setting['Setting1'] = gen_coords_s1
    if gen_coords_s2:
        generated_coords_by_setting['Setting2'] = gen_coords_s2
    
    logger.info(f"Generating PHATE grid with settings: {list(generated_coords_by_setting.keys())}")
    for setting_name, coords in generated_coords_by_setting.items():
        logger.info(f"  {setting_name} models: {list(coords.keys())}")
    
    # Generate PHATE grid using dynamic function
    fig1_2_path = fig1_dir / 'Fig1_2.pdf'
    plot_phate_dynamic_grid(
        phate_coords=phate_coords,
        real_labels=y_test,
        time_labels=time_labels,
        generated_coords_by_setting=generated_coords_by_setting,
        start_timepoint=start_tp,
        end_timepoint=end_tp,
        output_path=fig1_2_path,
        setting_colors={'Setting1': '#e74c3c', 'Setting2': '#3498db'},
        dpi=300
    )
    logger.info(f"Saved: {fig1_2_path}")
    
    # Also save PNG version
    fig1_2_png = fig1_dir / 'Fig1_2.png'
    plot_phate_dynamic_grid(
        phate_coords=phate_coords,
        real_labels=y_test,
        time_labels=time_labels,
        generated_coords_by_setting=generated_coords_by_setting,
        start_timepoint=start_tp,
        end_timepoint=end_tp,
        output_path=fig1_2_png,
        setting_colors={'Setting1': '#e74c3c', 'Setting2': '#3498db'},
        dpi=300
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FIGURE 1 GENERATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput files:")
    logger.info(f"  - {fig1_1_path}")
    logger.info(f"  - {fig1_2_path}")
    
    return {
        'fig1_1': str(fig1_1_path),
        'fig1_2': str(fig1_2_path),
        'vis_dir': str(vis_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 1: Core Performance Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step4_run_vis_fig1.py --experiment_dir /path/to/EMT_E2M
  python step4_run_vis_fig1.py --experiment_dir /path/to/EMT_E2M --force_recompute
        """
    )
    
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=True,
        help='Root experiment directory containing Setting* folders'
    )
    parser.add_argument(
        '--force_recompute',
        action='store_true',
        help='Force recomputation of embeddings even if cache exists'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_fig1_visualization(
            experiment_dir=args.experiment_dir,
            force_recompute=args.force_recompute,
            seed=args.seed
        )
        print(f"\nFigure 1 generated successfully!")
        print(f"Output directory: {results['vis_dir']}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
