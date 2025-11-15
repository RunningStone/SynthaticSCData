#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Model Comparison Experiment

Setting 1 (Boundary Only): SB, OT, VAE models
Setting 2 (All Timepoints): MLPlus SB model

All models evaluated with comprehensive metrics: 
- Test Loss, Frechet Distance, MAE, PCC
- Wasserstein Distance, MMD, R² per gene, JS Divergence, Correlation Structure
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt

from Data import (
    create_default_emt_data_loader,
    create_dataloaders_from_data
)
from Model import (
    SchrodingerBridgeModel,
    MLPlus_SchrodingerBridgeModel,
    OptimalTransportModel,
    ConditionalVAEModel
)
from Trainer import SBTrainer, UnifiedTrainer, Evaluator


def run_unified_experiment(
    file_path: str = None,
    n_hvg: int = 100,
    cells_per_timepoint_s1: int = 2000,
    batch_size: int = 256,
    epochs: int = 100,
    learning_rate: float = 5e-4,
    device: str = 'cuda',
    output_dir: str = './outputs',
    random_seed: int = 42
):
    """
    Run unified experiment: Setting 1 (SB/OT/VAE) + Setting 2 (MLPlus SB)
    
    Args:
        file_path: Path to h5ad file (None for default EMT data)
        n_hvg: Number of highly variable genes
        cells_per_timepoint_s1: Cells per timepoint for Setting 1
        batch_size: Batch size for training
        epochs: Training epochs
        learning_rate: Learning rate
        device: Device for training
        output_dir: Output directory
        random_seed: Random seed
    """
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("UNIFIED MODEL COMPARISON EXPERIMENT")
    print("Setting 1 (Boundary): SB, OT, VAE")
    print("Setting 2 (All Timepoints): MLPlus SB")
    print("="*80)
    
    # ========== Step 1: Load Data ==========
    print("\n[Step 1/9] Loading and analyzing data...")
    
    loader = create_default_emt_data_loader(file_path=file_path, n_hvg=n_hvg)
    loader.load_and_analyze()
    
    # Validate biology split
    valid = loader.validate_biology_split()
    if not valid:
        print("\n⚠️  Warning: Biology split validation failed!")
    
    dimension = n_hvg
    
    # ========== Step 2: Prepare Setting 1 Data ==========
    print("\n[Step 2/9] Preparing Setting 1 (boundary only)...")
    
    X_train_s1, y_train_s1, X_test_s1, y_test_s1 = loader.get_data_for_setting(
        setting=1,
        cells_per_timepoint=cells_per_timepoint_s1
    )
    
    train_loader_s1, test_loader_s1, stats_s1 = create_dataloaders_from_data(
        X_train_s1, y_train_s1, X_test_s1, y_test_s1,
        time_labels=loader.time_label_order,
        batch_size=batch_size
    )
    
    print("\nSetting 1 Statistics:")
    for key, value in stats_s1.items():
        if key not in ['train_time_counts', 'test_time_counts']:
            print(f"  {key}: {value}")
    
    # ========== Step 3: Prepare Setting 2 Data ==========
    print("\n[Step 3/9] Preparing Setting 2 (all timepoints)...")
    
    X_train_s2, y_train_s2, X_test_s2, y_test_s2 = loader.get_data_for_setting(
        setting=2,
        cells_per_timepoint=cells_per_timepoint_s1
    )
    
    train_loader_s2, test_loader_s2, stats_s2 = create_dataloaders_from_data(
        X_train_s2, y_train_s2, X_test_s2, y_test_s2,
        time_labels=loader.time_label_order,
        batch_size=batch_size
    )
    
    print("\nSetting 2 Statistics:")
    for key, value in stats_s2.items():
        if key not in ['train_time_counts', 'test_time_counts']:
            print(f"  {key}: {value}")
    
    # ========== Step 4: Train Setting 1 - SB Model ==========
    print("\n[Step 4/9] Training Setting 1 - Schrödinger Bridge...")
    print("-" * 80)
    
    sb_s1_model = SchrodingerBridgeModel(
        dimension=dimension,
        hidden_dims=[512, 512, 512, 512],
        time_embedding_dim=64,
        dropout=0.1,
        diffusion_coeff=0.1
    ).to(device)
    
    sb_s1_trainer = SBTrainer(
        model=sb_s1_model,
        train_loader=train_loader_s1,
        test_loader=test_loader_s1,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_path / 'setting1' / 'sb_model',
        weight_decay=1e-5,
        grad_clip_norm=5.0
    )
    
    sb_s1_history = sb_s1_trainer.train(
        epochs=epochs,
        early_stopping_patience=30
    )
    
    # ========== Step 5: Train Setting 1 - OT Model ==========
    print("\n[Step 5/9] Training Setting 1 - Optimal Transport...")
    print("-" * 80)
    
    ot_model = OptimalTransportModel(
        dimension=dimension,
        hidden_dims=[512, 512, 512, 512],
        activation='relu',
        dropout=0.1,
        use_residual=True
    ).to(device)
    
    ot_trainer = UnifiedTrainer(
        model=ot_model,
        train_loader=train_loader_s1,
        test_loader=test_loader_s1,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_path / 'setting1' / 'ot_model',
        weight_decay=1e-5,
        grad_clip_norm=5.0,
        model_type='ot'
    )
    
    ot_history = ot_trainer.train(
        epochs=epochs,
        early_stopping_patience=30
    )
    
    # ========== Step 6: Train Setting 1 - VAE Model ==========
    print("\n[Step 6/9] Training Setting 1 - Conditional VAE...")
    print("-" * 80)
    
    vae_model = ConditionalVAEModel(
        dimension=dimension,
        latent_dim=128,
        hidden_dims=[512, 256],
        activation='relu',
        dropout=0.1,
        beta=1.0
    ).to(device)
    
    vae_trainer = UnifiedTrainer(
        model=vae_model,
        train_loader=train_loader_s1,
        test_loader=test_loader_s1,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_path / 'setting1' / 'vae_model',
        weight_decay=1e-5,
        grad_clip_norm=5.0,
        model_type='vae'
    )
    
    vae_history = vae_trainer.train(
        epochs=epochs,
        early_stopping_patience=30
    )
    
    # ========== Step 7: Train Setting 2 - MLPlus SB Model ==========
    print("\n[Step 7/9] Training Setting 2 - MLPlus Schrödinger Bridge...")
    print("-" * 80)
    
    sb_s2_model = MLPlus_SchrodingerBridgeModel(
        dimension=dimension,
        hidden_dim=512,
        n_blocks=4,
        time_embedding_dim=64,
        n_time_frequencies=10,
        dropout=0.1,
        diffusion_coeff=0.1
    ).to(device)
    
    sb_s2_trainer = SBTrainer(
        model=sb_s2_model,
        train_loader=train_loader_s2,
        test_loader=test_loader_s2,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_path / 'setting2' / 'sb_mlplus_model',
        weight_decay=1e-5,
        grad_clip_norm=5.0
    )
    
    sb_s2_history = sb_s2_trainer.train(
        epochs=epochs,
        early_stopping_patience=30
    )
    
    # ========== Step 8: Evaluate All Models ==========
    print("\n[Step 8/9] Evaluating all models...")
    print("="*80)
    
    evaluator = Evaluator(device=device)
    
    # Evaluate Setting 1 models
    print("\nEvaluating Setting 1 - Schrödinger Bridge...")
    sb_s1_results = evaluator.evaluate(
        model=sb_s1_model,
        test_loader=test_loader_s1,
        time_labels=loader.time_label_order
    )
    
    print("\nEvaluating Setting 1 - Optimal Transport...")
    ot_results = evaluator.evaluate(
        model=ot_model,
        test_loader=test_loader_s1,
        time_labels=loader.time_label_order
    )
    
    print("\nEvaluating Setting 1 - VAE...")
    vae_results = evaluator.evaluate(
        model=vae_model,
        test_loader=test_loader_s1,
        time_labels=loader.time_label_order
    )
    
    # Evaluate Setting 2 model
    print("\nEvaluating Setting 2 - MLPlus SB...")
    sb_s2_results = evaluator.evaluate(
        model=sb_s2_model,
        test_loader=test_loader_s2,
        time_labels=loader.time_label_order
    )
    
    # ========== Step 9: Compare and Visualize Results ==========
    print("\n[Step 9/9] Comparing results and generating visualizations...")
    print("="*80)
    
    # Print comparison table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    print("\n{:<25} {:<15} {:<15} {:<15} {:<15}".format(
        "Model", "Test Loss", "Frechet Dist", "MAE", "PCC"
    ))
    print("-" * 85)
    
    # Setting 1 models
    print("\nSetting 1 (Boundary Only):")
    print("-" * 85)
    
    models_s1 = ['SB', 'OT', 'VAE']
    results_s1 = [sb_s1_results, ot_results, vae_results]
    
    for model_name, results in zip(models_s1, results_s1):
        print("{:<25} {:<15.2f} {:<15.2f} {:<15.4f} {:<15.4f}".format(
            model_name,
            results['test_loss'],
            results.get('frechet_distance', float('nan')),
            results.get('mae', float('nan')),
            results.get('pcc', float('nan'))
        ))
    
    # Setting 2 model
    print("\nSetting 2 (All Timepoints):")
    print("-" * 85)
    print("{:<25} {:<15.2f} {:<15.2f} {:<15.4f} {:<15.4f}".format(
        'MLPlus SB',
        sb_s2_results['test_loss'],
        sb_s2_results.get('frechet_distance', float('nan')),
        sb_s2_results.get('mae', float('nan')),
        sb_s2_results.get('pcc', float('nan'))
    ))
    
    # Find best models
    print("\n" + "-" * 85)
    print("Best Performance:")
    
    all_models = models_s1 + ['MLPlus SB (S2)']
    all_results = results_s1 + [sb_s2_results]
    
    # Test Loss (lower is better)
    best_loss_idx = np.argmin([r['test_loss'] for r in all_results])
    print(f"  Test Loss: {all_models[best_loss_idx]}")
    
    # Frechet Distance (lower is better)
    fd_values = [r.get('frechet_distance', float('inf')) for r in all_results]
    best_fd_idx = np.argmin(fd_values)
    print(f"  Frechet Distance: {all_models[best_fd_idx]}")
    
    # MAE (lower is better)
    mae_values = [r.get('mae', float('inf')) for r in all_results]
    best_mae_idx = np.argmin(mae_values)
    print(f"  MAE: {all_models[best_mae_idx]}")
    
    # PCC (higher is better)
    pcc_values = [r.get('pcc', float('-inf')) for r in all_results]
    best_pcc_idx = np.argmax(pcc_values)
    print(f"  PCC: {all_models[best_pcc_idx]}")
    
    # ========== Save Results ==========
    results_dict = {
        'setting1': {
            'data_statistics': stats_s1,
            'models': {
                'sb': {
                    'training_history': sb_s1_history,
                    'evaluation': sb_s1_results
                },
                'ot': {
                    'training_history': ot_history,
                    'evaluation': ot_results
                },
                'vae': {
                    'training_history': vae_history,
                    'evaluation': vae_results
                }
            }
        },
        'setting2': {
            'data_statistics': stats_s2,
            'models': {
                'sb_mlplus': {
                    'training_history': sb_s2_history,
                    'evaluation': sb_s2_results
                }
            }
        },
        'best_models': {
            'test_loss': all_models[best_loss_idx],
            'frechet_distance': all_models[best_fd_idx],
            'mae': all_models[best_mae_idx],
            'pcc': all_models[best_pcc_idx]
        }
    }
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_dict = convert_to_serializable(results_dict)
    
    with open(output_path / 'unified_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path / 'unified_results.json'}")
    
    # ========== Generate Comprehensive Comparison Plot ==========
    print("\n[Step 9/9] Generating comprehensive comparison plots...")
    
    # Plot 1: All models comparison (basic metrics)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['test_loss', 'frechet_distance', 'mae', 'pcc']
    titles = ['Test Loss', 'Frechet Distance', 'MAE', 'Pearson Correlation']
    colors_s1 = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    color_s2 = '#FFA07A'
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        values_s1 = [r.get(metric, float('nan')) for r in results_s1]
        value_s2 = sb_s2_results.get(metric, float('nan'))
        
        if not all(np.isnan(values_s1 + [value_s2])):
            # Plot Setting 1 models
            x_pos = np.arange(len(models_s1))
            bars_s1 = ax.bar(x_pos, values_s1, color=colors_s1, alpha=0.8, label='Setting 1')
            
            # Plot Setting 2 model
            bar_s2 = ax.bar(len(models_s1), value_s2, color=color_s2, alpha=0.8, label='Setting 2')
            
            ax.set_xticks(list(range(len(all_models))))
            ax.set_xticklabels(all_models, rotation=15, ha='right')
            ax.set_ylabel(title)
            ax.set_title(title, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
            
            # Add value labels
            all_bars = list(bars_s1.patches) + [bar_s2.patches[0]]
            for i, (bar, v) in enumerate(zip(all_bars, values_s1 + [value_s2])):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2, v, 
                           f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'unified_comparison_basic.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Basic comparison plot saved to: {output_path / 'unified_comparison_basic.png'}")
    
    # Plot 2: Advanced metrics comparison (all models)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    advanced_metrics = ['wasserstein_distance', 'mmd', 'r2_mean', 'js_divergence', 
                       'correlation_structure_corr', 'correlation_frobenius_diff']
    advanced_titles = ['Wasserstein Distance', 'MMD', 'R² (mean)', 'JS Divergence',
                      'Correlation Structure', 'Correlation Frobenius Diff']
    
    for idx, (metric, title) in enumerate(zip(advanced_metrics, advanced_titles)):
        ax = axes[idx]
        
        values_s1 = [r.get(metric, float('nan')) for r in results_s1]
        value_s2 = sb_s2_results.get(metric, float('nan'))
        
        if not all(np.isnan(values_s1 + [value_s2])):
            x_pos = np.arange(len(models_s1))
            bars_s1 = ax.bar(x_pos, values_s1, color=colors_s1, alpha=0.8, label='Setting 1')
            bar_s2 = ax.bar(len(models_s1), value_s2, color=color_s2, alpha=0.8, label='Setting 2')
            
            ax.set_xticks(list(range(len(all_models))))
            ax.set_xticklabels(all_models, rotation=15, ha='right')
            ax.set_ylabel(title)
            ax.set_title(title, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.legend()
            
            all_bars = list(bars_s1.patches) + [bar_s2.patches[0]]
            for i, (bar, v) in enumerate(zip(all_bars, values_s1 + [value_s2])):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2, v, 
                           f'{v:.3f}', ha='center', va='bottom', fontsize=7)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'unified_comparison_advanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Advanced comparison plot saved to: {output_path / 'unified_comparison_advanced.png'}")
    
    # Plot 3: Best SB models comparison (Setting 1 SB vs Setting 2 MLPlus)
    evaluator = Evaluator(device=device)
    evaluator.plot_comparison(
        results_s1[0],  # Setting 1 SB results
        sb_s2_results,   # Setting 2 MLPlus results
        str(output_path / 'sb_setting_comparison.png')
    )
    
    print(f"✓ SB Setting comparison saved to: {output_path / 'sb_setting_comparison.png'}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    
    return results_dict


def main():
    parser = argparse.ArgumentParser(
        description='Unified model comparison: Setting 1 (SB/OT/VAE) + Setting 2 (MLPlus SB)'
    )
    parser.add_argument(
        '--file_path',
        type=str,
        default=None,
        help='Path to h5ad file (default: EMT dataset)'
    )
    parser.add_argument(
        '--n_hvg',
        type=int,
        default=100,
        help='Number of highly variable genes'
    )
    parser.add_argument(
        '--cells_per_timepoint',
        type=int,
        default=2000,
        help='Cells per timepoint'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=5e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for training'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    run_unified_experiment(
        file_path=args.file_path,
        n_hvg=args.n_hvg,
        cells_per_timepoint_s1=args.cells_per_timepoint,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
