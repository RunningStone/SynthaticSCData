#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Script for Experiment 7: Entropy Evolution Analysis

Compares entropy evolution across different experimental settings to test
the core hypothesis: boundary conditions are insufficient to constrain
non-monotonic entropy dynamics.

This script uses the modular architecture:
- Data module: ConfigLoader, create_data_loader_from_config
- Analyser module: EntropyAnalyzer, ModelManager

Usage:
    python Workers/step2_run_entropy_experiment.py \\
        experiment_EMT_Part1_setting7_entropy.yaml \\
        --config_dir configs/EMT_E2M \\
        --output_dir /path/to/output \\
        --setting1_checkpoint /path/to/Setting1/sb_mlplus_best.pt \\
        --setting2_checkpoint /path/to/Setting2/sb_mlplus_best.pt

Author: SynthaticSCData Project
Date: 2024-11
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Data import (
    ConfigLoader,
    create_data_loader_from_config,
    get_data_for_setting
)
from Analyser import EntropyAnalyzer, ModelManager


def load_model_from_checkpoint(
    checkpoint_path: str,
    dimension: int,
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    Load trained SB model from checkpoint.
    Automatically detects whether it's a standard SB or SB_MLPlus model.
    
    Args:
        checkpoint_path: Path to checkpoint file
        dimension: Input dimension (number of genes)
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    from Model.sb_model import SchrodingerBridgeModel
    from Model.sb_model_mlplus import MLPlus_SchrodingerBridgeModel
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Detect model type by checking state_dict keys
    is_mlplus = any('time_embed.freq_scales' in k or 'time_embed.mlp' in k for k in state_dict.keys())
    
    if is_mlplus:
        model = MLPlus_SchrodingerBridgeModel(
            dimension=dimension,
            hidden_dim=512,
            n_blocks=8,
            time_embedding_dim=64,
            n_time_frequencies=10,
            dropout=0.1,
            diffusion_coeff=0.1
        )
        print(f"    Detected: MLPlus_SchrodingerBridgeModel (hidden_dim=512, n_blocks=8)")
    else:
        model = SchrodingerBridgeModel(
            dimension=dimension,
            hidden_dims=[1024, 1024, 1024, 1024, 1024],
            time_embedding_dim=64,
            dropout=0.1,
            diffusion_coeff=0.1
        )
        print(f"    Detected: SchrodingerBridgeModel (standard, hidden_dims=[1024]*5)")
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Experiment 7: Entropy Evolution Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with config file
    python Workers/step2_run_entropy_experiment.py \\
        experiment_EMT_Part1_setting7_entropy.yaml \\
        --config_dir configs/EMT_E2M \\
        --output_dir ./outputs/entropy_analysis

    # With model checkpoints for comparison
    python Workers/step2_run_entropy_experiment.py \\
        experiment_EMT_Part1_setting7_entropy.yaml \\
        --config_dir configs/EMT_E2M \\
        --output_dir ./outputs/entropy_analysis \\
        --setting1_checkpoint ./outputs/Setting1/sb_mlplus_best.pt \\
        --setting2_checkpoint ./outputs/Setting2/sb_mlplus_best.pt
        """
    )
    
    # Required arguments
    parser.add_argument(
        'config_file',
        type=str,
        help='Experiment configuration file (e.g., experiment_EMT_Part1_setting7_entropy.yaml)'
    )
    
    # Configuration
    parser.add_argument(
        '--config_dir',
        type=str,
        default='configs',
        help='Directory containing configuration files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    # Model checkpoints
    parser.add_argument(
        '--setting1_checkpoint',
        type=str,
        default=None,
        help='Path to Setting1 (boundary) model checkpoint'
    )
    parser.add_argument(
        '--setting2_checkpoint',
        type=str,
        default=None,
        help='Path to Setting2 (full trajectory) model checkpoint'
    )
    parser.add_argument(
        '--setting3_checkpoint',
        type=str,
        default=None,
        help='Path to Setting3 (key points) model checkpoint'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--method',
        type=str,
        default='knn',
        choices=['knn', 'gaussian', 'both'],
        help='Entropy estimation method'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of nearest neighbors for KNN method'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1000,
        help='Number of cells to sample from t0 for trajectory generation'
    )
    parser.add_argument(
        '--cross_validate_methods',
        action='store_true',
        help='Run both KNN and Gaussian methods for cross-validation'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for entropy evolution analysis."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("="*70)
    print("Experiment 7: Entropy Evolution Analysis")
    print("="*70)
    
    # =========================================================================
    # Step 1: Load configuration
    # =========================================================================
    print("\n[1/7] Loading configuration...")
    
    config_loader = ConfigLoader(config_dir=args.config_dir)
    config = config_loader.load_experiment_config(args.config_file)
    
    # Setup output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device
    device_str = args.device if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    
    print(f"  Config: {args.config_file}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device}")
    
    # =========================================================================
    # Step 2: Load data using project infrastructure
    # =========================================================================
    print("\n[2/7] Loading data using project infrastructure...")
    
    data_loader = create_data_loader_from_config(config, logger)
    data_loader.load_and_analyze()
    
    # Get test data
    X_train, y_train, X_test, y_test = get_data_for_setting(data_loader, config, logger)
    
    dimension = X_test.shape[1]
    time_labels = data_loader.time_label_order
    
    # Get time points for this setting
    setting_time_points = config['data_setting'].get('time_points', time_labels)
    
    print(f"  Loaded {X_test.shape[0]} test cells × {dimension} genes")
    print(f"  Time labels: {time_labels}")
    print(f"  Setting time points: {setting_time_points}")
    
    # =========================================================================
    # Step 3: Create entropy analyzer
    # =========================================================================
    print("\n[3/7] Initializing entropy analyzer...")
    
    analyzer = EntropyAnalyzer(
        output_dir=output_dir,
        device=device_str,
        random_seed=42
    )
    
    # =========================================================================
    # Step 4: Compute real data entropy curve
    # =========================================================================
    print("\n[4/7] Computing real data entropy curve from test set...")
    
    # Filter test data to setting time points
    time_to_idx = {label: idx for idx, label in enumerate(time_labels)}
    
    # Create filtered arrays for setting time points
    X_filtered = []
    y_filtered = []
    
    for new_idx, time_label in enumerate(setting_time_points):
        if time_label in time_to_idx:
            old_idx = time_to_idx[time_label]
            mask = (y_test == old_idx)
            X_filtered.append(X_test[mask])
            y_filtered.append(np.full(mask.sum(), new_idx))
    
    X_filtered = np.vstack(X_filtered)
    y_filtered = np.concatenate(y_filtered)
    
    # Compute entropy curve
    real_entropy_curve, real_data_list, real_peak_analysis = analyzer.compute_entropy_curve_from_real_data(
        X=X_filtered,
        y=y_filtered,
        time_labels=setting_time_points,
        method=args.method,
        k=args.k,
        n_samples=args.n_samples,
        verbose=True
    )
    
    # =========================================================================
    # Step 5: Sample initial states from first time point
    # =========================================================================
    print(f"\n[5/7] Sampling initial states from {setting_time_points[0]}...")
    
    t0_mask = (y_filtered == 0)
    X_t0 = X_filtered[t0_mask]
    
    if len(X_t0) > args.n_samples:
        indices = np.random.choice(len(X_t0), args.n_samples, replace=False)
        X_t0 = X_t0[indices]
    
    initial_states = torch.tensor(X_t0, dtype=torch.float32)
    print(f"  Sampled {initial_states.shape[0]} cells from {setting_time_points[0]}")
    
    # =========================================================================
    # Step 6: Load models and compute entropy curves
    # =========================================================================
    print("\n[6/7] Loading trained models and computing entropy curves...")
    
    models_dict = {}
    
    # Load Setting1 checkpoint if provided
    if args.setting1_checkpoint and os.path.exists(args.setting1_checkpoint):
        print(f"  Loading Setting1 (boundary): {args.setting1_checkpoint}")
        models_dict['Setting1'] = load_model_from_checkpoint(
            args.setting1_checkpoint, dimension, device_str
        )
    else:
        print(f"  Skipping Setting1 (checkpoint not provided or not found)")
    
    # Load Setting2 checkpoint if provided
    if args.setting2_checkpoint and os.path.exists(args.setting2_checkpoint):
        print(f"  Loading Setting2 (full trajectory): {args.setting2_checkpoint}")
        models_dict['Setting2'] = load_model_from_checkpoint(
            args.setting2_checkpoint, dimension, device_str
        )
    else:
        print(f"  Skipping Setting2 (checkpoint not provided or not found)")
    
    # Load Setting3 checkpoint if provided
    if args.setting3_checkpoint and os.path.exists(args.setting3_checkpoint):
        print(f"  Loading Setting3 (key points): {args.setting3_checkpoint}")
        models_dict['Setting3'] = load_model_from_checkpoint(
            args.setting3_checkpoint, dimension, device_str
        )
    else:
        print(f"  Skipping Setting3 (checkpoint not provided or not found)")
    
    # Create time grid
    time_grid = torch.linspace(0, 1, len(setting_time_points))
    
    # Compare models
    results_dict = {}
    if models_dict:
        results_dict = analyzer.compare_multiple_models(
            models_dict=models_dict,
            initial_states=initial_states,
            time_grid=time_grid,
            time_labels=setting_time_points,
            real_entropy_curve=real_entropy_curve,
            method=args.method,
            k=args.k,
            verbose=True
        )
    
    # =========================================================================
    # Step 7: Generate visualizations and save results
    # =========================================================================
    print("\n[7/7] Generating visualizations and saving results...")
    
    # Plot entropy curves
    if results_dict:
        analyzer.plot_entropy_curves(
            results_dict=results_dict,
            real_curve=real_entropy_curve,
            time_labels=setting_time_points,
            method=args.method
        )
        
        analyzer.plot_peak_characteristics_comparison(
            results_dict=results_dict,
            real_peak_analysis=real_peak_analysis
        )
    
    # Optional: Cross-validate methods
    if args.cross_validate_methods and args.method != 'both' and models_dict:
        print("\n[Extra] Cross-validating entropy estimation methods...")
        
        alt_method = 'gaussian' if args.method == 'knn' else 'knn'
        
        results_dict_alt = analyzer.compare_multiple_models(
            models_dict=models_dict,
            initial_states=initial_states,
            time_grid=time_grid,
            time_labels=setting_time_points,
            real_entropy_curve=None,
            method=alt_method,
            k=args.k,
            verbose=False
        )
        
        if args.method == 'knn':
            analyzer.plot_method_cross_validation(
                results_dict_knn=results_dict,
                results_dict_gauss=results_dict_alt,
                time_labels=setting_time_points
            )
        else:
            analyzer.plot_method_cross_validation(
                results_dict_knn=results_dict_alt,
                results_dict_gauss=results_dict,
                time_labels=setting_time_points
            )
    
    # Save results
    analyzer.save_results(
        real_entropy_curve=real_entropy_curve,
        real_peak_analysis=real_peak_analysis,
        results_dict=results_dict,
        time_labels=setting_time_points,
        config=config,
        args=vars(args)
    )
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "="*70)
    print("✓ Experiment 7 completed successfully!")
    print(f"✓ Results saved to {output_dir}")
    print("="*70)
    
    print("\nFinal Summary:")
    print(f"  Real data peak: {real_peak_analysis['peak_time']} "
          f"(Non-monotonic: {real_peak_analysis['is_nonmonotonic']})")
    
    for setting_name in results_dict:
        peak = results_dict[setting_name]['peak_analysis']
        sim = results_dict[setting_name].get('similarity_to_real', 'N/A')
        sim_str = f"{sim:.4f}" if isinstance(sim, float) else str(sim)
        print(f"  {setting_name:10s} peak: {peak['peak_time']:5s} "
              f"(Non-monotonic: {peak['is_nonmonotonic']}, MSE: {sim_str})")


if __name__ == "__main__":
    main()
