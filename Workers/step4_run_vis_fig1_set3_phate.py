#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 - Figure 1 Extended: PHATE with Setting1, Setting2, and Setting3
========================================================================

This script generates an extended PHATE visualization including Setting3:
- Row 1: Real test data (3 subplots: all timepoints, start/end, intermediate)
- Row 2: Setting1 models (OT, VAE, SB) - boundary only
- Row 3: Setting2 models (Batch_OT, VAE, SB_MLPlus) - full trajectory
- Row 4: Setting3 models - key timepoints

Required Settings:
- Setting1: Boundary-only (0d, 7d)
- Setting2: Full trajectory (0d, 8h, 1d, 3d, 7d)
- Setting3: Key timepoints (e.g., 0d, 8h, 7d)

Usage:
    python step4_run_vis_fig1_set3_phate.py --experiment_dir /path/to/EMT_E2M

Output:
    - {experiment_dir}/vis/Figure1/Fig1_Set3_PHATE.pdf
"""

import argparse
import json
import pickle
import logging
import numpy as np
import yaml
import matplotlib.pyplot as plt
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
    TIMEPOINT_COLORS,
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
    
    logger = logging.getLogger('fig1_set3_phate_visualization')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "fig1_set3_phate_visualization.log")
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


def load_generated_data(setting_path: Path, model_name: str) -> Optional[Dict]:
    """Load generated data PKL file for a model."""
    pkl_path = setting_path / 'generated_data' / f'{model_name}.pkl'
    if not pkl_path.exists():
        return None
    
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def load_real_test_data(
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
    generated_data_dict: Dict[str, np.ndarray],
    logger: logging.Logger,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    """
    Compute PHATE embeddings for real and generated data.
    
    Returns:
        Tuple of (real_phate_coords, model_phate_coords_dict, n_real_samples)
    """
    # Check cache
    if cache_path and cache_path.exists() and not force_recompute:
        logger.info(f"Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        return cached['real_coords'], cached['model_coords'], cached['n_real']
    
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
    
    # Extract coordinates
    real_coords = phate_coords[:len(real_data)]
    model_coords = {}
    for model_name, (start_idx, end_idx) in model_indices.items():
        model_coords[model_name] = phate_coords[start_idx:end_idx]
    
    # Cache results
    if cache_path:
        logger.info(f"Caching embeddings to {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'real_coords': real_coords,
                'model_coords': model_coords,
                'n_real': len(real_data)
            }, f)
    
    return real_coords, model_coords, len(real_data)


def plot_phate_4row_grid(
    real_coords: np.ndarray,
    real_labels: np.ndarray,
    time_labels: List[str],
    generated_coords_s1: Dict[str, np.ndarray],
    generated_coords_s2: Dict[str, np.ndarray],
    generated_coords_s3: Dict[str, np.ndarray],
    start_timepoint: str,
    end_timepoint: str,
    output_path: Path,
    figsize: Tuple[int, int] = None,
    dpi: int = 300
) -> Path:
    """
    Create a dynamic PHATE embedding grid with Setting1, Setting2, and Setting3.
    
    Row 1: Real data views (all categories, start/end highlighted, intermediate highlighted)
    Row 2: Setting1 models - dynamically determined from available models
    Row 3: Setting2 models - dynamically determined from available models
    Row 4: Setting3 models - dynamically determined from available models
    
    Args:
        real_coords: PHATE coordinates for real data
        real_labels: Labels for real data points (integer indices)
        time_labels: List of time label strings
        generated_coords_s1: Dict mapping model_name -> generated PHATE coords for Setting1
        generated_coords_s2: Dict mapping model_name -> generated PHATE coords for Setting2
        generated_coords_s3: Dict mapping model_name -> generated PHATE coords for Setting3
        start_timepoint: Start timepoint label (e.g., '0d')
        end_timepoint: End timepoint label (e.g., '7d')
        output_path: Path to save the figure
        figsize: Figure size (auto-calculated if None)
        dpi: DPI for saving
    
    Returns:
        Path to saved figure
    """
    # Determine number of columns dynamically
    n_cols = max(3,  # At least 3 for real data row
                 len(generated_coords_s1) if generated_coords_s1 else 0,
                 len(generated_coords_s2) if generated_coords_s2 else 0,
                 len(generated_coords_s3) if generated_coords_s3 else 0)
    
    if figsize is None:
        figsize = (5 * n_cols, 20)
    
    fig, axes = plt.subplots(4, n_cols, figsize=figsize)
    
    # Convert integer labels to string labels
    unique_int_labels = sorted(set(real_labels))
    int_to_str = {i: time_labels[i] for i in unique_int_labels if i < len(time_labels)}
    real_labels_str = np.array([int_to_str.get(l, str(l)) for l in real_labels])
    
    # Define colors for each timepoint
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(time_labels)))
    tp_colors = {tp: colors[i] for i, tp in enumerate(time_labels)}
    
    # =========================================================================
    # Row 1: Real data views
    # =========================================================================
    
    # Row 1, Col 1: All categories with different colors
    ax = axes[0, 0]
    for time_idx, time_label in enumerate(time_labels):
        mask = (real_labels == time_idx)
        ax.scatter(real_coords[mask, 0], real_coords[mask, 1],
                  c=[tp_colors[time_label]], label=time_label,
                  alpha=0.6, s=15, edgecolors='none')
    ax.set_title('Real Data: All Categories', fontweight='bold', fontsize=11)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.5)
    ax.set_xlabel('PHATE 1', fontsize=10)
    ax.set_ylabel('PHATE 2', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Row 1, Col 2: Start/end highlighted, others gray
    ax = axes[0, 1]
    for time_idx, time_label in enumerate(time_labels):
        mask = (real_labels == time_idx)
        if time_label in [start_timepoint, end_timepoint]:
            ax.scatter(real_coords[mask, 0], real_coords[mask, 1],
                      c=[tp_colors[time_label]], label=time_label,
                      alpha=0.7, s=20, edgecolors='none')
        else:
            ax.scatter(real_coords[mask, 0], real_coords[mask, 1],
                      c='lightgray', alpha=0.3, s=10, edgecolors='none')
    ax.set_title(f'Highlighted: {start_timepoint} & {end_timepoint}', fontweight='bold', fontsize=11)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.5)
    ax.set_xlabel('PHATE 1', fontsize=10)
    ax.set_ylabel('PHATE 2', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Row 1, Col 3: Intermediate highlighted, start/end gray
    ax = axes[0, 2]
    for time_idx, time_label in enumerate(time_labels):
        mask = (real_labels == time_idx)
        if time_label not in [start_timepoint, end_timepoint]:
            ax.scatter(real_coords[mask, 0], real_coords[mask, 1],
                      c=[tp_colors[time_label]], label=time_label,
                      alpha=0.7, s=20, edgecolors='none')
        else:
            ax.scatter(real_coords[mask, 0], real_coords[mask, 1],
                      c='lightgray', alpha=0.3, s=10, edgecolors='none')
    ax.set_title('Highlighted: Intermediate', fontweight='bold', fontsize=11)
    ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.5)
    ax.set_xlabel('PHATE 1', fontsize=10)
    ax.set_ylabel('PHATE 2', fontsize=10)
    ax.grid(alpha=0.3)
    
    # =========================================================================
    # Row 2: Setting1 models
    # =========================================================================
    setting1_models = list(generated_coords_s1.keys()) if generated_coords_s1 else []
    for col_idx, model_name in enumerate(setting1_models):
        ax = axes[1, col_idx]
        
        # Plot real data in gray
        ax.scatter(real_coords[:, 0], real_coords[:, 1],
                  c='lightgray', alpha=0.3, s=10, edgecolors='none', label='Real')
        
        # Plot generated data in red
        if model_name in generated_coords_s1:
            gen_coords = generated_coords_s1[model_name]
            ax.scatter(gen_coords[:, 0], gen_coords[:, 1],
                      c='#e74c3c', alpha=0.7, s=25, edgecolors='darkred',
                      linewidths=0.5, label='Generated', marker='*')
        
        ax.set_title(f'Setting1 (Boundary): {model_name.upper()}', fontweight='bold', fontsize=11)
        ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.5)
        ax.set_xlabel('PHATE 1', fontsize=10)
        ax.set_ylabel('PHATE 2', fontsize=10)
        ax.grid(alpha=0.3)
    
    # =========================================================================
    # Row 3: Setting2 models
    # =========================================================================
    setting2_models = list(generated_coords_s2.keys()) if generated_coords_s2 else []
    for col_idx, model_name in enumerate(setting2_models):
        ax = axes[2, col_idx]
        
        # Plot real data in gray
        ax.scatter(real_coords[:, 0], real_coords[:, 1],
                  c='lightgray', alpha=0.3, s=10, edgecolors='none', label='Real')
        
        # Plot generated data in blue
        if model_name in generated_coords_s2:
            gen_coords = generated_coords_s2[model_name]
            ax.scatter(gen_coords[:, 0], gen_coords[:, 1],
                      c='#3498db', alpha=0.7, s=25, edgecolors='darkblue',
                      linewidths=0.5, label='Generated', marker='*')
        
        ax.set_title(f'Setting2 (Full): {model_name.upper()}', fontweight='bold', fontsize=11)
        ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.5)
        ax.set_xlabel('PHATE 1', fontsize=10)
        ax.set_ylabel('PHATE 2', fontsize=10)
        ax.grid(alpha=0.3)
    
    # =========================================================================
    # Row 4: Setting3 models
    # =========================================================================
    setting3_models = list(generated_coords_s3.keys()) if generated_coords_s3 else []
    for col_idx, model_name in enumerate(setting3_models):
        ax = axes[3, col_idx]
        
        # Plot real data in gray
        ax.scatter(real_coords[:, 0], real_coords[:, 1],
                  c='lightgray', alpha=0.3, s=10, edgecolors='none', label='Real')
        
        # Plot generated data in green
        if model_name in generated_coords_s3:
            gen_coords = generated_coords_s3[model_name]
            ax.scatter(gen_coords[:, 0], gen_coords[:, 1],
                      c='#2ecc71', alpha=0.7, s=25, edgecolors='darkgreen',
                      linewidths=0.5, label='Generated', marker='*')
        
        ax.set_title(f'Setting3 (Key): {model_name.upper()}', fontweight='bold', fontsize=11)
        ax.legend(loc='best', fontsize=8, framealpha=0.9, markerscale=1.5)
        ax.set_xlabel('PHATE 1', fontsize=10)
        ax.set_ylabel('PHATE 2', fontsize=10)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots in each row
    for col_idx in range(3, n_cols):
        axes[0, col_idx].axis('off')  # Row 1: Real data (only 3 columns used)
    for col_idx in range(len(setting1_models), n_cols):
        axes[1, col_idx].axis('off')  # Row 2: Setting1
    for col_idx in range(len(setting2_models), n_cols):
        axes[2, col_idx].axis('off')  # Row 3: Setting2
    for col_idx in range(len(setting3_models), n_cols):
        axes[3, col_idx].axis('off')  # Row 4: Setting3
    
    plt.suptitle('Generation Quality Visualization (PHATE)\nSetting1 vs Setting2 vs Setting3', 
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def run_fig1_set3_phate_visualization(
    experiment_dir: str,
    force_recompute: bool = False,
    seed: int = 42
) -> Dict:
    """
    Run Figure 1 extended PHATE visualization with Setting3.
    
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
    logger.info("FIGURE 1 EXTENDED: PHATE with Setting1, Setting2, and Setting3")
    logger.info("=" * 80)
    
    np.random.seed(seed)
    
    # Check required settings exist
    setting1_path = experiment_base / 'Setting1'
    setting2_path = experiment_base / 'Setting2'
    setting3_path = experiment_base / 'Setting3'
    
    missing_settings = []
    if not setting1_path.exists():
        missing_settings.append('Setting1')
    if not setting2_path.exists():
        missing_settings.append('Setting2')
    if not setting3_path.exists():
        missing_settings.append('Setting3')
    
    if missing_settings:
        raise FileNotFoundError(f"Missing required settings: {missing_settings}")
    
    # =========================================================================
    # Load real test data (use Setting2 config for all timepoints)
    # =========================================================================
    logger.info("\n--- Loading Real Test Data ---")
    X_test, y_test, time_labels = load_real_test_data(setting2_path, logger)
    
    start_tp = time_labels[0] if time_labels else '0d'
    end_tp = time_labels[-1] if time_labels else '7d'
    logger.info(f"Timepoints: {time_labels}")
    logger.info(f"Start: {start_tp}, End: {end_tp}")
    
    # =========================================================================
    # Load generated data from all settings
    # =========================================================================
    logger.info("\n--- Loading Generated Data ---")
    
    # Initialize value checker
    value_checker = ValueChecker(nan_threshold=0.1, logger=logger)
    
    # Load evaluation results for validation
    eval_results_s1 = ValueChecker.load_evaluation_results(setting1_path / 'evaluation_results.json')
    eval_results_s2 = ValueChecker.load_evaluation_results(setting2_path / 'evaluation_results.json')
    eval_results_s3 = ValueChecker.load_evaluation_results(setting3_path / 'evaluation_results.json')
    
    # Setting1 models - dynamically load all available models
    gen_data_s1 = {}
    available_models_s1 = list(eval_results_s1.keys()) if eval_results_s1 else []
    logger.info(f"Available models in Setting1: {available_models_s1}")
    for model_name in available_models_s1:
        gen_pkl = load_generated_data(setting1_path, model_name)
        if gen_pkl is not None and 'generated_data' in gen_pkl:
            gen_data_s1[model_name] = gen_pkl['generated_data']
            logger.info(f"  Setting1/{model_name}: {len(gen_pkl['generated_data'])} samples")
    
    # Validate Setting1 models (filter out NaN models)
    gen_data_s1, skipped_s1 = value_checker.filter_valid_models(gen_data_s1, eval_results_s1)
    if skipped_s1:
        logger.warning(f"Skipped Setting1 models due to invalid data: {skipped_s1}")
    logger.info(f"Valid Setting1 models after filtering: {list(gen_data_s1.keys())}")
    
    # Setting2 models - dynamically load all available models
    gen_data_s2 = {}
    available_models_s2 = list(eval_results_s2.keys()) if eval_results_s2 else []
    logger.info(f"Available models in Setting2: {available_models_s2}")
    for model_name in available_models_s2:
        gen_pkl = load_generated_data(setting2_path, model_name)
        if gen_pkl is not None and 'generated_data' in gen_pkl:
            gen_data_s2[model_name] = gen_pkl['generated_data']
            logger.info(f"  Setting2/{model_name}: {len(gen_pkl['generated_data'])} samples")
    
    # Validate Setting2 models (filter out NaN models)
    gen_data_s2, skipped_s2 = value_checker.filter_valid_models(gen_data_s2, eval_results_s2)
    if skipped_s2:
        logger.warning(f"Skipped Setting2 models due to invalid data: {skipped_s2}")
    logger.info(f"Valid Setting2 models after filtering: {list(gen_data_s2.keys())}")
    
    # Setting3 models - dynamically load all available models
    gen_data_s3 = {}
    available_models_s3 = list(eval_results_s3.keys()) if eval_results_s3 else []
    logger.info(f"Available models in Setting3: {available_models_s3}")
    for model_name in available_models_s3:
        gen_pkl = load_generated_data(setting3_path, model_name)
        if gen_pkl is not None and 'generated_data' in gen_pkl:
            gen_data_s3[model_name] = gen_pkl['generated_data']
            logger.info(f"  Setting3/{model_name}: {len(gen_pkl['generated_data'])} samples")
    
    # Validate Setting3 models (filter out NaN models)
    gen_data_s3, skipped_s3 = value_checker.filter_valid_models(gen_data_s3, eval_results_s3)
    if skipped_s3:
        logger.warning(f"Skipped Setting3 models due to invalid data: {skipped_s3}")
    logger.info(f"Valid Setting3 models after filtering: {list(gen_data_s3.keys())}")
    
    if not gen_data_s1 and not gen_data_s2 and not gen_data_s3:
        logger.error("No valid generated data found in any setting after filtering")
        return {
            'fig_path': None,
            'skipped_models': {'Setting1': skipped_s1, 'Setting2': skipped_s2, 'Setting3': skipped_s3}
        }
    
    # =========================================================================
    # Combine all generated data for joint PHATE embedding
    # =========================================================================
    logger.info("\n--- Preparing Data for PHATE ---")
    
    all_generated = {}
    for model_name, gen_data in gen_data_s1.items():
        all_generated[f's1_{model_name}'] = gen_data
    for model_name, gen_data in gen_data_s2.items():
        all_generated[f's2_{model_name}'] = gen_data
    for model_name, gen_data in gen_data_s3.items():
        all_generated[f's3_{model_name}'] = gen_data
    
    logger.info(f"Total variants: {len(all_generated)}")
    
    # =========================================================================
    # Compute PHATE embeddings
    # =========================================================================
    logger.info("\n--- Computing PHATE Embeddings ---")
    
    cache_path = fig1_dir / 'fig1_set3_phate_cache.pkl'
    real_coords, model_coords, n_real = compute_phate_embeddings(
        X_test, all_generated, logger, cache_path, force_recompute
    )
    
    logger.info(f"Real data PHATE coords: {real_coords.shape}")
    for name, coords in model_coords.items():
        logger.info(f"  {name}: {coords.shape}")
    
    # Separate model coordinates by setting
    gen_coords_s1 = {k.replace('s1_', ''): v for k, v in model_coords.items() if k.startswith('s1_')}
    gen_coords_s2 = {k.replace('s2_', ''): v for k, v in model_coords.items() if k.startswith('s2_')}
    gen_coords_s3 = {k.replace('s3_', ''): v for k, v in model_coords.items() if k.startswith('s3_')}
    
    # =========================================================================
    # Generate PHATE grid
    # =========================================================================
    logger.info("\n--- Generating PHATE Grid ---")
    
    fig_path = fig1_dir / 'Fig1_Set3_PHATE.pdf'
    plot_phate_4row_grid(
        real_coords=real_coords,
        real_labels=y_test,
        time_labels=time_labels,
        generated_coords_s1=gen_coords_s1,
        generated_coords_s2=gen_coords_s2,
        generated_coords_s3=gen_coords_s3,
        start_timepoint=start_tp,
        end_timepoint=end_tp,
        output_path=fig_path,
        figsize=(15, 20),
        dpi=300
    )
    logger.info(f"Saved: {fig_path}")
    
    # Also save PNG version
    fig_png = fig1_dir / 'Fig1_Set3_PHATE.png'
    plot_phate_4row_grid(
        real_coords=real_coords,
        real_labels=y_test,
        time_labels=time_labels,
        generated_coords_s1=gen_coords_s1,
        generated_coords_s2=gen_coords_s2,
        generated_coords_s3=gen_coords_s3,
        start_timepoint=start_tp,
        end_timepoint=end_tp,
        output_path=fig_png,
        figsize=(15, 20),
        dpi=300
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FIGURE 1 SET3 PHATE GENERATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput files:")
    logger.info(f"  - {fig_path}")
    logger.info(f"  - {fig_png}")
    
    return {
        'fig1_set3_phate': str(fig_path),
        'vis_dir': str(fig1_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 1 Extended: PHATE with Setting1, Setting2, and Setting3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step4_run_vis_fig1_set3_phate.py --experiment_dir /path/to/EMT_E2M
  python step4_run_vis_fig1_set3_phate.py --experiment_dir /path/to/EMT_E2M --force_recompute
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
        help='Force recomputation of PHATE embeddings'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_fig1_set3_phate_visualization(
            experiment_dir=args.experiment_dir,
            force_recompute=args.force_recompute,
            seed=args.seed
        )
        print(f"\nFigure 1 Set3 PHATE generated successfully!")
        print(f"Output directory: {results['vis_dir']}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
