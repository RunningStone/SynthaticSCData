#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 - Figure 2 PHATE: Ablation PHATE Visualization
======================================================

This script generates PHATE visualizations for Figure 2:
- Row 1: Real test data (3 subplots: all timepoints, start only, end only)
- Row 2+: Setting2 and Setting4 ablation variants comparison

Each subplot shows real data (colored by timepoint) overlaid with generated 
endpoint samples from sb_mlplus model.

Required Settings:
- Setting2: Full trajectory (0d, 8h, 1d, 3d, 7d) - baseline
- Setting4: Ablation variants (Setting4_Ablation_Remove*)

Usage:
    python step4_run_vis_fig2_phate.py --experiment_dir /path/to/EMT_E2M

Output:
    - {experiment_dir}/vis/Figure2/Fig2_PHATE.pdf
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
from Data import (
    create_data_loader_from_config,
    get_data_for_setting
)


def setup_logger(output_dir: Path) -> logging.Logger:
    """Setup logger for the script."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('fig2_phate_visualization')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "fig2_phate_visualization.log")
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


def discover_ablation_settings(setting4_path: Path) -> Dict[str, Path]:
    """
    Discover ablation variants in Setting4 directory.
    
    Returns:
        Dict mapping removed timepoint (e.g., '8h') to ablation path
    """
    ablations = {}
    
    for item in setting4_path.iterdir():
        if item.is_dir() and 'Ablation' in item.name:
            # Extract timepoint from name like "Setting4_Ablation_Remove8h"
            if 'Remove' in item.name:
                tp = item.name.split('Remove')[-1]
                # Check if generated_data exists
                if (item / 'generated_data').exists():
                    ablations[tp] = item
    
    return ablations


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


def plot_ablation_phate_grid(
    real_coords: np.ndarray,
    real_labels: np.ndarray,
    time_labels: List[str],
    model_coords_dict: Dict[str, np.ndarray],
    variant_names: List[str],
    output_path: Path,
    end_timepoint: str = '7d',
    figsize_per_subplot: Tuple[float, float] = (5, 5),
    dpi: int = 300
) -> Path:
    """
    Create PHATE grid visualization for ablation analysis.
    
    Layout:
    - Row 1: Real data only (3 subplots: all, start, end)
    - Row 2+: Setting2 and Setting4 variants (3 per row)
    
    Args:
        real_coords: PHATE coordinates for real data
        real_labels: Timepoint labels for real data
        time_labels: List of all timepoint labels
        model_coords_dict: Dict mapping variant_name -> generated PHATE coords
        variant_names: List of variant names in order
        output_path: Path to save figure
        end_timepoint: End timepoint label
        figsize_per_subplot: Size per subplot
        dpi: DPI for saving
    
    Returns:
        Path to saved figure
    """
    n_variants = len(variant_names)
    n_cols = 3
    n_rows = 1 + (n_variants + n_cols - 1) // n_cols  # Row 1 for real data + rows for variants
    
    figsize = (figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Convert integer labels to string labels if needed
    # real_labels may be integer indices into time_labels
    unique_int_labels = sorted(set(real_labels))
    
    # Create mapping from integer to string label
    int_to_str = {i: time_labels[i] for i in unique_int_labels if i < len(time_labels)}
    
    # Convert real_labels to string labels for display
    real_labels_str = np.array([int_to_str.get(l, str(l)) for l in real_labels])
    unique_labels = [int_to_str.get(l, str(l)) for l in unique_int_labels]
    
    # Get timepoint colors
    color_map = {}
    default_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, label in enumerate(unique_labels):
        if label in TIMEPOINT_COLORS:
            color_map[label] = TIMEPOINT_COLORS[label]
        else:
            color_map[label] = default_colors[i % 10]
    
    start_timepoint = time_labels[0] if time_labels else '0d'
    
    # =========================================================================
    # Row 1: Real data only
    # =========================================================================
    
    # Subplot 1.1: All timepoints
    ax = axes[0, 0]
    for label in unique_labels:
        mask = real_labels_str == label
        ax.scatter(real_coords[mask, 0], real_coords[mask, 1],
                  c=[color_map[label]], label=label, alpha=0.6, s=10)
    ax.set_title('Real Data: All Timepoints', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8, markerscale=1.5)
    ax.set_xlabel('PHATE 1', fontsize=10)
    ax.set_ylabel('PHATE 2', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Subplot 1.2: Start timepoint only
    ax = axes[0, 1]
    mask_start = real_labels_str == start_timepoint
    ax.scatter(real_coords[~mask_start, 0], real_coords[~mask_start, 1],
              c='lightgray', alpha=0.3, s=5, label='Other')
    ax.scatter(real_coords[mask_start, 0], real_coords[mask_start, 1],
              c=[color_map[start_timepoint]], alpha=0.8, s=15, label=start_timepoint)
    ax.set_title(f'Real Data: {start_timepoint} (Start)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8, markerscale=1.5)
    ax.set_xlabel('PHATE 1', fontsize=10)
    ax.set_ylabel('PHATE 2', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Subplot 1.3: End timepoint only
    ax = axes[0, 2]
    mask_end = real_labels_str == end_timepoint
    ax.scatter(real_coords[~mask_end, 0], real_coords[~mask_end, 1],
              c='lightgray', alpha=0.3, s=5, label='Other')
    ax.scatter(real_coords[mask_end, 0], real_coords[mask_end, 1],
              c=[color_map.get(end_timepoint, 'red')], alpha=0.8, s=15, label=end_timepoint)
    ax.set_title(f'Real Data: {end_timepoint} (End)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8, markerscale=1.5)
    ax.set_xlabel('PHATE 1', fontsize=10)
    ax.set_ylabel('PHATE 2', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # Row 2+: Setting2 and Setting4 variants
    # =========================================================================
    
    for idx, variant_name in enumerate(variant_names):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Plot real data as background (gray)
        ax.scatter(real_coords[:, 0], real_coords[:, 1],
                  c='lightgray', alpha=0.3, s=5, label='Real (all)')
        
        # Highlight real end timepoint
        mask_end = real_labels_str == end_timepoint
        ax.scatter(real_coords[mask_end, 0], real_coords[mask_end, 1],
                  c=[color_map.get(end_timepoint, 'blue')], alpha=0.5, s=10, 
                  label=f'Real {end_timepoint}')
        
        # Plot generated data
        if variant_name in model_coords_dict:
            gen_coords = model_coords_dict[variant_name]
            ax.scatter(gen_coords[:, 0], gen_coords[:, 1],
                      c='red', alpha=0.7, s=15, marker='x', label='Generated')
        
        # Format title
        if variant_name == 'Setting2':
            title = 'Setting2 (Full)'
        else:
            # Extract removed timepoint from variant name
            title = f'Setting4: -{variant_name}'
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=7, markerscale=1.2)
        ax.set_xlabel('PHATE 1', fontsize=9)
        ax.set_ylabel('PHATE 2', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    total_subplots = n_rows * n_cols
    used_subplots = 3 + n_variants  # 3 for row 1 + variants
    for idx in range(n_variants, (n_rows - 1) * n_cols):
        row = 1 + idx // n_cols
        col = idx % n_cols
        if row < n_rows and col < n_cols:
            if 3 + idx >= used_subplots:
                axes[row, col].axis('off')
    
    plt.suptitle('PHATE Visualization: Ablation Analysis\n(SB_MLPlus Model)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def run_fig2_phate_visualization(
    experiment_dir: str,
    model_name: str = 'sb_mlplus',
    force_recompute: bool = False,
    seed: int = 42
) -> Dict:
    """
    Run Figure 2 PHATE visualization pipeline.
    
    Args:
        experiment_dir: Root directory containing Setting* folders
        model_name: Model to visualize (default: sb_mlplus)
        force_recompute: Force recomputation of embeddings
        seed: Random seed
    
    Returns:
        Dictionary containing paths to generated figures
    """
    experiment_base = Path(experiment_dir)
    vis_dir = experiment_base / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Figure2 subfolder
    fig2_dir = vis_dir / 'Figure2'
    fig2_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(vis_dir)
    
    logger.info("=" * 80)
    logger.info("FIGURE 2 PHATE: Ablation PHATE Visualization")
    logger.info("=" * 80)
    
    np.random.seed(seed)
    
    # Check required settings exist
    setting2_path = experiment_base / 'Setting2'
    setting4_path = experiment_base / 'Setting4'
    
    if not setting2_path.exists():
        raise FileNotFoundError(f"Setting2 not found at {setting2_path}")
    if not setting4_path.exists():
        raise FileNotFoundError(f"Setting4 not found at {setting4_path}")
    
    # =========================================================================
    # Load real test data
    # =========================================================================
    logger.info("\n--- Loading Real Test Data ---")
    X_test, y_test, time_labels = load_real_test_data(setting2_path, logger)
    
    # Determine end timepoint
    end_timepoint = time_labels[-1] if time_labels else '7d'
    logger.info(f"End timepoint: {end_timepoint}")
    
    # =========================================================================
    # Load generated data from Setting2 and Setting4
    # =========================================================================
    logger.info("\n--- Loading Generated Data ---")
    
    # Setting2 generated data
    gen_data_s2 = load_generated_data(setting2_path, model_name)
    if gen_data_s2 is None:
        raise FileNotFoundError(f"No generated data found for {model_name} in Setting2")
    
    logger.info(f"Setting2/{model_name}: {len(gen_data_s2.get('generated_data', []))} samples")
    
    # Discover and load Setting4 ablation variants
    ablation_paths = discover_ablation_settings(setting4_path)
    logger.info(f"Found ablation variants: {list(ablation_paths.keys())}")
    
    gen_data_ablations = {}
    for tp, abl_path in ablation_paths.items():
        gen_pkl = load_generated_data(abl_path, model_name)
        if gen_pkl is not None and 'generated_data' in gen_pkl:
            gen_data_ablations[tp] = gen_pkl['generated_data']
            logger.info(f"  Setting4_Remove{tp}: {len(gen_pkl['generated_data'])} samples")
    
    if not gen_data_ablations:
        raise ValueError("No ablation generated data found in Setting4")
    
    # =========================================================================
    # Prepare data for PHATE
    # =========================================================================
    logger.info("\n--- Preparing Data for PHATE ---")
    
    # Collect all generated data
    all_generated = {}
    all_generated['Setting2'] = gen_data_s2['generated_data']
    for tp, gen_data in gen_data_ablations.items():
        all_generated[tp] = gen_data
    
    # Define variant order: Setting2 first, then ablations sorted
    variant_names = ['Setting2'] + sorted(gen_data_ablations.keys())
    logger.info(f"Variants to visualize: {variant_names}")
    
    # =========================================================================
    # Compute PHATE embeddings
    # =========================================================================
    logger.info("\n--- Computing PHATE Embeddings ---")
    
    cache_path = fig2_dir / 'fig2_phate_cache.pkl'
    real_coords, model_coords, n_real = compute_phate_embeddings(
        X_test, all_generated, logger, cache_path, force_recompute
    )
    
    logger.info(f"Real data PHATE coords: {real_coords.shape}")
    for name, coords in model_coords.items():
        logger.info(f"  {name}: {coords.shape}")
    
    # =========================================================================
    # Generate PHATE grid
    # =========================================================================
    logger.info("\n--- Generating PHATE Grid ---")
    
    fig2_phate_path = fig2_dir / 'Fig2_PHATE.pdf'
    plot_ablation_phate_grid(
        real_coords=real_coords,
        real_labels=y_test,
        time_labels=time_labels,
        model_coords_dict=model_coords,
        variant_names=variant_names,
        output_path=fig2_phate_path,
        end_timepoint=end_timepoint,
        figsize_per_subplot=(5, 5),
        dpi=300
    )
    logger.info(f"Saved: {fig2_phate_path}")
    
    # Also save PNG version
    fig2_phate_png = fig2_dir / 'Fig2_PHATE.png'
    plot_ablation_phate_grid(
        real_coords=real_coords,
        real_labels=y_test,
        time_labels=time_labels,
        model_coords_dict=model_coords,
        variant_names=variant_names,
        output_path=fig2_phate_png,
        end_timepoint=end_timepoint,
        figsize_per_subplot=(5, 5),
        dpi=300
    )
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FIGURE 2 PHATE GENERATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput files:")
    logger.info(f"  - {fig2_phate_path}")
    logger.info(f"  - {fig2_phate_png}")
    
    return {
        'fig2_phate': str(fig2_phate_path),
        'vis_dir': str(fig2_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 2 PHATE: Ablation PHATE Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step4_run_vis_fig2_phate.py --experiment_dir /path/to/EMT_E2M
  python step4_run_vis_fig2_phate.py --experiment_dir /path/to/EMT_E2M --force_recompute
        """
    )
    
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=True,
        help='Root experiment directory containing Setting* folders'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='sb_mlplus',
        help='Model to visualize (default: sb_mlplus)'
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
        results = run_fig2_phate_visualization(
            experiment_dir=args.experiment_dir,
            model_name=args.model,
            force_recompute=args.force_recompute,
            seed=args.seed
        )
        print(f"\nFigure 2 PHATE generated successfully!")
        print(f"Output directory: {results['vis_dir']}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
