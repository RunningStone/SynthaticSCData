#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Script for Experiment 7: Entropy Evolution Analysis

Compares entropy evolution across different experimental settings to test
the core hypothesis: boundary conditions are insufficient to constrain
non-monotonic entropy dynamics.

Usage:
    python run_entropy_analysis.py \\
        --data_path /path/to/test_data.h5ad \\
        --setting1_checkpoint /path/to/Setting1/sb_mlplus_best.pt \\
        --setting2_checkpoint /path/to/Setting2/sb_mlplus_best.pt \\
        --setting3_checkpoint /path/to/Setting3/sb_mlplus_best.pt \\
        --output_dir ./entropy_analysis_results

Author: Generated for Experiment 7
Date: 2024-11
"""

import argparse
import os
import json
import pickle
from pathlib import Path
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from Model.sb_model import SchrodingerBridgeModel
from Model.sb_model_mlplus import MLPlus_SchrodingerBridgeModel
from Data import ConfigLoader, create_data_loader_from_config, get_data_for_setting
from Experiments.exp7_entropy.entropy_estimators import (
    estimate_entropy_knn,
    estimate_entropy_gaussian,
    estimate_entropy_both_methods
)
from Experiments.exp7_entropy.analyze_entropy_evolution import (
    compute_entropy_curve,
    compute_entropy_curve_from_real_data,
    analyze_entropy_peak,
    compare_multiple_models
)


def load_model_from_checkpoint(checkpoint_path: str, dimension: int, device: str = 'cuda'):
    """
    Load trained SB model from checkpoint.
    Automatically detects whether it's a standard SB or SB_MLPlus model.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Detect model type by checking state_dict keys
    # SB_MLPlus has 'time_embed.freq_scales' and 'time_embed.mlp.X.weight'
    # Standard SB has 'time_embed.0.weight' and 'time_embed.2.weight'
    is_mlplus = any('time_embed.freq_scales' in k or 'time_embed.mlp' in k for k in state_dict.keys())
    
    if is_mlplus:
        # Load as MLPlus model (from models_default.yaml: hidden_dim=512, n_blocks=8)
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
        # Load as standard SB model (from models_default.yaml: hidden_dims=[1024, 1024, 1024, 1024, 1024])
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


def plot_entropy_curves(
    results_dict: dict,
    real_curve: np.ndarray,
    time_labels: list,
    output_dir: str,
    method: str = 'knn'
):
    """
    Create publication-quality entropy curve comparison plot.
    
    Args:
        results_dict: Dictionary mapping setting names to their results
        real_curve: Real data entropy curve
        time_labels: Time labels
        output_dir: Output directory
        method: Entropy estimation method for title
    """
    plt.figure(figsize=(12, 8))
    
    # Color scheme
    colors = {
        'Real': '#2E86AB',      # Blue
        'Setting1': '#A23B72',  # Purple (boundary only, should fail)
        'Setting2': '#F18F01',  # Orange (full trajectory, should succeed)
        'Setting3': '#C73E1D',  # Red (key points)
    }
    
    markers = {
        'Real': 'o',
        'Setting1': 's',
        'Setting2': '^',
        'Setting3': 'D'
    }
    
    linestyles = {
        'Real': '-',
        'Setting1': '--',
        'Setting2': '-',
        'Setting3': '-.'
    }
    
    # Plot real data
    plt.plot(
        range(len(time_labels)),
        real_curve,
        marker=markers['Real'],
        linestyle=linestyles['Real'],
        color=colors['Real'],
        linewidth=2.5,
        markersize=10,
        label='Real Data',
        zorder=10
    )
    
    # Plot each model
    for setting_name, results in results_dict.items():
        entropy_curve = results['entropy_curve']
        
        plt.plot(
            range(len(time_labels)),
            entropy_curve,
            marker=markers.get(setting_name, 'x'),
            linestyle=linestyles.get(setting_name, ':'),
            color=colors.get(setting_name, 'gray'),
            linewidth=2,
            markersize=8,
            label=setting_name,
            alpha=0.8
        )
    
    # Formatting
    plt.xlabel('Time Point', fontsize=14, fontweight='bold')
    plt.ylabel(f'Differential Entropy ({method.upper()} estimate)', fontsize=14, fontweight='bold')
    plt.title('Entropy Evolution: Real vs. Generated Trajectories', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.xticks(range(len(time_labels)), time_labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    
    # Highlight non-monotonic region for real data
    peak_idx = np.argmax(real_curve)
    plt.axvline(x=peak_idx, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    plt.text(
        peak_idx, plt.ylim()[1] * 0.95,
        f'Real peak\nat {time_labels[peak_idx]}',
        ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save
    output_path_png = os.path.join(output_dir, 'entropy_curves_comparison.png')
    output_path_pdf = os.path.join(output_dir, 'entropy_curves_comparison.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"Saved entropy curves to {output_path_png}")


def plot_peak_characteristics_comparison(
    results_dict: dict,
    real_peak_analysis: dict,
    output_dir: str
):
    """
    Create bar plot comparing peak characteristics across settings.
    
    Args:
        results_dict: Dictionary mapping setting names to their results
        real_peak_analysis: Peak analysis results for real data
        output_dir: Output directory
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    settings = list(results_dict.keys())
    
    # Metric 1: Peak amplitude
    ax = axes[0, 0]
    amplitudes = [real_peak_analysis['amplitude']] + \
                 [results_dict[s]['peak_analysis']['amplitude'] for s in settings]
    labels = ['Real'] + settings
    colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    ax.bar(labels, amplitudes, color=colors_list[:len(labels)])
    ax.set_ylabel('Peak Amplitude', fontsize=12, fontweight='bold')
    ax.set_title('Entropy Peak Amplitude', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Metric 2: Non-monotonicity
    ax = axes[0, 1]
    is_nonmono = [1 if real_peak_analysis['is_nonmonotonic'] else 0] + \
                 [1 if results_dict[s]['peak_analysis']['is_nonmonotonic'] else 0 for s in settings]
    
    ax.bar(labels, is_nonmono, color=colors_list[:len(labels)])
    ax.set_ylabel('Non-monotonic (1=Yes, 0=No)', fontsize=12, fontweight='bold')
    ax.set_title('Inverted-U Shape Detection', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.2])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Metric 3: Explore rate
    ax = axes[1, 0]
    explore_rates = [real_peak_analysis['explore_rate']] + \
                    [results_dict[s]['peak_analysis']['explore_rate'] for s in settings]
    
    ax.bar(labels, explore_rates, color=colors_list[:len(labels)])
    ax.set_ylabel('Entropy Increase Rate', fontsize=12, fontweight='bold')
    ax.set_title('Exploration Phase Rate', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Metric 4: MSE to real
    ax = axes[1, 1]
    mse_values = [results_dict[s].get('similarity_to_real', 0) for s in settings]
    
    ax.bar(settings, mse_values, color=colors_list[1:len(settings)+1])
    ax.set_ylabel('MSE to Real Entropy Curve', fontsize=12, fontweight='bold')
    ax.set_title('Curve Similarity (Lower is Better)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_path_png = os.path.join(output_dir, 'peak_characteristics_comparison.png')
    output_path_pdf = os.path.join(output_dir, 'peak_characteristics_comparison.pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"Saved peak characteristics to {output_path_png}")


def plot_method_cross_validation(
    results_dict_knn: dict,
    results_dict_gauss: dict,
    time_labels: list,
    output_dir: str
):
    """
    Cross-validate KNN and Gaussian entropy estimation methods.
    
    Args:
        results_dict_knn: Results using KNN method
        results_dict_gauss: Results using Gaussian method
        time_labels: Time labels
        output_dir: Output directory
    """
    fig, axes = plt.subplots(1, len(results_dict_knn), figsize=(5*len(results_dict_knn), 5))
    
    if len(results_dict_knn) == 1:
        axes = [axes]
    
    for idx, setting_name in enumerate(results_dict_knn.keys()):
        ax = axes[idx]
        
        H_knn = results_dict_knn[setting_name]['entropy_curve']
        H_gauss = results_dict_gauss[setting_name]['entropy_curve']
        
        ax.plot(range(len(time_labels)), H_knn, 'o-', label='KNN', linewidth=2, markersize=8)
        ax.plot(range(len(time_labels)), H_gauss, 's--', label='Gaussian', linewidth=2, markersize=8)
        
        ax.set_xlabel('Time Point', fontsize=12)
        ax.set_ylabel('Entropy', fontsize=12)
        ax.set_title(f'{setting_name}: Method Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(range(len(time_labels)))
        ax.set_xticklabels(time_labels)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Compute correlation
        corr = np.corrcoef(H_knn, H_gauss)[0, 1]
        ax.text(
            0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'method_cross_validation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved method cross-validation to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 7: Entropy Evolution Analysis'
    )
    
    # Configuration file (use existing project config system)
    parser.add_argument('--config', type=str, default='configs/experiment_EMT_Part1_setting7_entropy.yaml',
                       help='Experiment configuration file')
    parser.add_argument('--config_dir', type=str, default='configs',
                       help='Directory containing configuration files')
    
    # Model checkpoints
    parser.add_argument('--setting1_checkpoint', type=str, default=None,
                       help='Path to Setting1 (boundary) model checkpoint (optional)')
    parser.add_argument('--setting2_checkpoint', type=str, default=None,
                       help='Path to Setting2 (full trajectory) model checkpoint (optional, will use Setting7 if not provided)')
    parser.add_argument('--setting3_checkpoint', type=str, default=None,
                       help='Path to Setting3 (key points) model checkpoint (optional)')
    
    # Analysis parameters
    parser.add_argument('--method', type=str, default='knn', choices=['knn', 'gaussian', 'both'],
                       help='Entropy estimation method')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of nearest neighbors for KNN method')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of cells to sample from t0 for trajectory generation')
    parser.add_argument('--n_steps', type=int, default=50,
                       help='Number of integration steps between time points')
    parser.add_argument('--cross_validate_methods', action='store_true',
                       help='Run both KNN and Gaussian methods for cross-validation')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./entropy_analysis_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("="*70)
    print("Experiment 7: Entropy Evolution Analysis")
    print("="*70)
    
    # Step 1: Load configuration using existing project infrastructure
    print("\n[1/7] Loading configuration...")
    config_loader = ConfigLoader(config_dir=args.config_dir)
    
    # Determine which config to use
    if args.config:
        config_file = args.config
    else:
        config_file = 'experiment_EMT_Part1_setting7_entropy.yaml'
    
    config = config_loader.load_experiment_config(config_file)
    
    # Get output directory and device from config
    output_dir = config['settings']['output_dir']
    if args.output_dir:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    device_str = args.device if args.device else config['settings'].get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    
    print(f"  Config: {config_file}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {device}")
    
    # Step 2: Load data using existing project infrastructure
    print("\n[2/7] Loading data using project infrastructure...")
    data_loader = create_data_loader_from_config(config, logger)
    data_loader.load_and_analyze()
    
    # Get test data
    X_train, y_train, X_test, y_test = get_data_for_setting(data_loader, config, logger)
    
    dimension = X_test.shape[1]
    time_labels = data_loader.time_label_order
    
    # Filter to only the time points we need for entropy analysis
    setting_time_points = config['data_setting']['time_points']
    
    print(f"  Loaded {X_test.shape[0]} test cells × {dimension} genes")
    print(f"  Time labels: {time_labels}")
    print(f"  Setting time points: {setting_time_points}")
    
    # Step 3: Compute real data entropy curve
    print("\n[3/7] Computing real data entropy curve from test set...")
    
    # Prepare data for entropy calculation
    # Group test data by time labels
    time_point_data = {}
    for i, time_label in enumerate(time_labels):
        if time_label in setting_time_points:
            mask = y_test == i
            time_point_data[time_label] = X_test[mask]
    
    # Compute entropy for each time point
    real_entropy_curve = []
    for time_label in setting_time_points:
        if time_label in time_point_data:
            data = time_point_data[time_label]
            # Sample if too many cells
            if len(data) > args.n_samples:
                indices = np.random.choice(len(data), args.n_samples, replace=False)
                data = data[indices]
            
            entropy = estimate_entropy_knn(data, k=args.k)
            real_entropy_curve.append(entropy)
            print(f"  {time_label}: entropy = {entropy:.4f} ({len(data)} cells)")
    
    real_entropy_curve = np.array(real_entropy_curve)
    
    real_peak_analysis = analyze_entropy_peak(real_entropy_curve, setting_time_points)
    
    print("\nReal Data Entropy Peak Analysis:")
    print(f"  Peak at: {real_peak_analysis['peak_time']}")
    print(f"  Peak value: {real_peak_analysis['peak_value']:.4f}")
    print(f"  Amplitude: {real_peak_analysis['amplitude']:.4f}")
    print(f"  Non-monotonic: {real_peak_analysis['is_nonmonotonic']}")
    
    # Step 4: Sample initial states from first time point
    print(f"\n[4/7] Sampling initial states from {setting_time_points[0]}...")
    t0_idx = time_labels.index(setting_time_points[0])
    mask_t0 = y_test == t0_idx
    X_t0 = X_test[mask_t0]
    
    if len(X_t0) > args.n_samples:
        indices = np.random.choice(len(X_t0), args.n_samples, replace=False)
        X_t0 = X_t0[indices]
    
    initial_states = torch.tensor(X_t0, dtype=torch.float32)
    print(f"  Sampled {initial_states.shape[0]} cells from {setting_time_points[0]}")
    
    # Step 5: Load models
    print("\n[5/7] Loading trained models...")
    models_dict = {}
    
    if args.setting1_checkpoint and os.path.exists(args.setting1_checkpoint):
        print(f"  Loading Setting1 (boundary): {args.setting1_checkpoint}")
        models_dict['Setting1'] = load_model_from_checkpoint(
            args.setting1_checkpoint, dimension, device_str
        )
    else:
        print(f"  Skipping Setting1 (checkpoint not provided or not found)")
    
    if args.setting2_checkpoint and os.path.exists(args.setting2_checkpoint):
        print(f"  Loading Setting2 (full trajectory): {args.setting2_checkpoint}")
        models_dict['Setting2'] = load_model_from_checkpoint(
            args.setting2_checkpoint, dimension, device_str
        )
    else:
        print(f"  Skipping Setting2 (checkpoint not provided or not found)")
    
    if args.setting3_checkpoint and os.path.exists(args.setting3_checkpoint):
        print(f"  Loading Setting3 (key points): {args.setting3_checkpoint}")
        models_dict['Setting3'] = load_model_from_checkpoint(
            args.setting3_checkpoint, dimension, device_str
        )
    else:
        print(f"  Skipping Setting3 (checkpoint not provided or not found)")
    
    # Create time grid (normalized to [0, 1])
    time_grid = torch.linspace(0, 1, len(setting_time_points))
    
    # Step 6: Compute entropy curves for all models
    print("\n[6/7] Computing entropy curves for all models...")
    
    results_dict = compare_multiple_models(
        models_dict=models_dict,
        initial_states=initial_states,
        time_grid=time_grid,
        time_labels=setting_time_points,
        real_entropy_curve=real_entropy_curve,
        method=args.method,
        k=args.k,
        device=device_str,
        verbose=True
    )
    
    # Step 7: Generate visualizations
    print("\n[7/7] Generating visualizations...")
    
    plot_entropy_curves(
        results_dict=results_dict,
        real_curve=real_entropy_curve,
        time_labels=setting_time_points,
        output_dir=output_dir,
        method=args.method
    )
    
    plot_peak_characteristics_comparison(
        results_dict=results_dict,
        real_peak_analysis=real_peak_analysis,
        output_dir=output_dir
    )
    
    # Optional: Cross-validate methods
    if args.cross_validate_methods and args.method != 'both':
        print("\n[Extra] Cross-validating entropy estimation methods...")
        
        # Recompute with alternative method
        alt_method = 'gaussian' if args.method == 'knn' else 'knn'
        
        results_dict_alt = compare_multiple_models(
            models_dict=models_dict,
            initial_states=initial_states,
            time_grid=time_grid,
            time_labels=setting_time_points,
            real_entropy_curve=None,
            method=alt_method,
            k=args.k,
            device=device_str,
            verbose=False
        )
        
        if args.method == 'knn':
            plot_method_cross_validation(
                results_dict_knn=results_dict,
                results_dict_gauss=results_dict_alt,
                time_labels=setting_time_points,
                output_dir=output_dir
            )
        else:
            plot_method_cross_validation(
                results_dict_knn=results_dict_alt,
                results_dict_gauss=results_dict,
                time_labels=setting_time_points,
                output_dir=output_dir
            )
    
    # Save results
    print("\n[Final] Saving results...")
    
    # Helper function to convert numpy types to Python types for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, (int, np.integer)):
            return int(obj)
        elif isinstance(obj, (float, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Save numerical results
    summary = {
        'real_entropy_curve': real_entropy_curve.tolist(),
        'real_peak_analysis': convert_to_json_serializable(real_peak_analysis),
        'settings': {}
    }
    
    for setting_name, results in results_dict.items():
        summary['settings'][setting_name] = {
            'entropy_curve': results['entropy_curve'].tolist(),
            'peak_analysis': convert_to_json_serializable(results['peak_analysis']),
            'similarity_to_real': float(results.get('similarity_to_real', 0)) if results.get('similarity_to_real') is not None else None
        }
    
    summary_path = os.path.join(output_dir, 'entropy_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved summary to {summary_path}")
    
    # Save full results (including trajectories)
    full_results_path = os.path.join(output_dir, 'entropy_analysis_full_results.pkl')
    with open(full_results_path, 'wb') as f:
        pickle.dump({
            'real_entropy_curve': real_entropy_curve,
            'real_peak_analysis': real_peak_analysis,
            'settings_results': results_dict,
            'time_labels': setting_time_points,
            'config': config,
            'args': vars(args)
        }, f)
    
    print(f"  Saved full results to {full_results_path}")
    
    print("\n" + "="*70)
    print("✓ Experiment 7 completed successfully!")
    print(f"✓ Results saved to {output_dir}")
    print("="*70)
    
    # Print final summary
    print("\nFinal Summary:")
    print(f"  Real data peak: {real_peak_analysis['peak_time']} "
          f"(Non-monotonic: {real_peak_analysis['is_nonmonotonic']})")
    
    for setting_name in results_dict:
        peak = results_dict[setting_name]['peak_analysis']
        sim = results_dict[setting_name].get('similarity_to_real', 'N/A')
        print(f"  {setting_name:10s} peak: {peak['peak_time']:5s} "
              f"(Non-monotonic: {peak['is_nonmonotonic']}, MSE: {sim})")


if __name__ == "__main__":
    main()
