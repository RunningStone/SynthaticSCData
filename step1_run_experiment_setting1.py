#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Model Comparison

Setting 1 (Boundary Only): SB, OT, VAE
Setting 2 (All Timepoints): MLPlus SB

All models evaluated with consistent metrics: Test Loss, Frechet Distance, MAE, PCC
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm

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


def run_comparison_experiment(
    file_path: str = None,
    n_hvg: int = 100,
    cells_per_timepoint: int = 2000,
    batch_size: int = 256,
    epochs: int = 100,
    learning_rate: float = 5e-4,
    device: str = 'cuda',
    output_dir: str = './outputs_compare',
    random_seed: int = 42
):
    """
    Run comparison experiment with SB, OT, and VAE on Setting 1
    
    Args:
        file_path: Path to h5ad file (None for default EMT data)
        n_hvg: Number of highly variable genes
        cells_per_timepoint: Cells per timepoint for Setting 1
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
    
    print("="*70)
    print("Model Comparison Experiment - Setting 1 (Boundary Only)")
    print("Models: Schrödinger Bridge, Optimal Transport, VAE")
    print("="*70)
    
    # ========== Step 1: Load and Prepare Data ==========
    print("\n[Step 1/4] Loading and preparing Setting 1 data...")
    
    loader = create_default_emt_data_loader(file_path=file_path, n_hvg=n_hvg)
    loader.load_and_analyze()
    
    # Validate biology split
    valid = loader.validate_biology_split()
    if not valid:
        print("\n⚠️  Warning: Biology split validation failed!")
        print("Continuing anyway, but results may not be reliable.")
    
    # Get Setting 1 data (boundary only)
    X_train, y_train, X_test, y_test = loader.get_data_for_setting(
        setting=1,
        cells_per_timepoint=cells_per_timepoint
    )
    
    train_loader, test_loader, stats = create_dataloaders_from_data(
        X_train, y_train, X_test, y_test,
        time_labels=loader.time_label_order,
        batch_size=batch_size
    )
    
    print("\nSetting 1 Statistics:")
    for key, value in stats.items():
        if key not in ['train_time_counts', 'test_time_counts']:
            print(f"  {key}: {value}")
    
    dimension = n_hvg
    
    # ========== Step 2: Train Schrödinger Bridge Model ==========
    print("\n[Step 2/4] Training Schrödinger Bridge Model...")
    print("-" * 70)
    
    sb_model = SchrodingerBridgeModel(
        dimension=dimension,
        hidden_dims=[512, 512, 512, 512],
        time_embedding_dim=64,
        dropout=0.1,
        diffusion_coeff=0.1
    ).to(device)
    
    sb_trainer = SBTrainer(
        model=sb_model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_path / 'sb_model',
        weight_decay=1e-5,
        grad_clip_norm=5.0
    )
    
    sb_history = sb_trainer.train(
        epochs=epochs,
        early_stopping_patience=30
    )
    
    # ========== Step 3: Train Optimal Transport Model ==========
    print("\n[Step 3/4] Training Optimal Transport Model...")
    print("-" * 70)
    
    ot_model = OptimalTransportModel(
        dimension=dimension,
        hidden_dims=[512, 512, 512, 512],
        activation='relu',
        dropout=0.1,
        use_residual=True
    ).to(device)
    
    ot_trainer = UnifiedTrainer(
        model=ot_model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_path / 'ot_model',
        weight_decay=1e-5,
        grad_clip_norm=5.0,
        model_type='ot'
    )
    
    ot_history = ot_trainer.train(
        epochs=epochs,
        early_stopping_patience=30
    )
    
    # ========== Step 4: Train VAE Model ==========
    print("\n[Step 4/4] Training Conditional VAE Model...")
    print("-" * 70)
    
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
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_path / 'vae_model',
        weight_decay=1e-5,
        grad_clip_norm=5.0,
        model_type='vae'
    )
    
    vae_history = vae_trainer.train(
        epochs=epochs,
        early_stopping_patience=30
    )
    
    # ========== Step 5: Evaluate and Compare ==========
    print("\n[Step 5/5] Evaluating and comparing all models...")
    print("="*70)
    
    evaluator = Evaluator(device=device)
    
    # Evaluate SB model
    print("\nEvaluating Schrödinger Bridge...")
    sb_results = evaluator.evaluate(
        model=sb_model,
        test_loader=test_loader,
        time_labels=loader.time_label_order
    )
    
    # Evaluate OT model
    print("\nEvaluating Optimal Transport...")
    ot_results = evaluator.evaluate(
        model=ot_model,
        test_loader=test_loader,
        time_labels=loader.time_label_order
    )
    
    # Evaluate VAE model
    print("\nEvaluating VAE...")
    vae_results = evaluator.evaluate(
        model=vae_model,
        test_loader=test_loader,
        time_labels=loader.time_label_order
    )
    
    # ========== Print Comparison Results ==========
    print("\n" + "="*70)
    print("COMPARISON RESULTS - Setting 1 (Boundary Only)")
    print("="*70)
    
    models = ['Schrödinger Bridge', 'Optimal Transport', 'VAE']
    results_list = [sb_results, ot_results, vae_results]
    
    print("\n{:<20} {:<15} {:<15} {:<15} {:<15}".format(
        "Model", "Test Loss", "Frechet Dist", "MAE", "PCC"
    ))
    print("-" * 80)
    
    for model_name, results in zip(models, results_list):
        print("{:<20} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(
            model_name,
            results['test_loss'],
            results.get('frechet_distance', float('nan')),
            results.get('mae', float('nan')),
            results.get('pcc', float('nan'))
        ))
    
    # Find best model for each metric
    print("\n" + "-" * 80)
    print("Best Performance:")
    
    # Test Loss (lower is better)
    best_loss_idx = np.argmin([r['test_loss'] for r in results_list])
    print(f"  Test Loss: {models[best_loss_idx]}")
    
    # Frechet Distance (lower is better)
    fd_values = [r.get('frechet_distance', float('inf')) for r in results_list]
    best_fd_idx = np.argmin(fd_values)
    print(f"  Frechet Distance: {models[best_fd_idx]}")
    
    # MAE (lower is better)
    mae_values = [r.get('mae', float('inf')) for r in results_list]
    best_mae_idx = np.argmin(mae_values)
    print(f"  MAE: {models[best_mae_idx]}")
    
    # PCC (higher is better)
    pcc_values = [r.get('pcc', float('-inf')) for r in results_list]
    best_pcc_idx = np.argmax(pcc_values)
    print(f"  PCC: {models[best_pcc_idx]}")
    
    # ========== Save Results ==========
    results = {
        'setting': 'Setting 1 (Boundary Only)',
        'data_statistics': stats,
        'models': {
            'schrodinger_bridge': {
                'training_history': sb_history,
                'evaluation': sb_results
            },
            'optimal_transport': {
                'training_history': ot_history,
                'evaluation': ot_results
            },
            'vae': {
                'training_history': vae_history,
                'evaluation': vae_results
            }
        },
        'best_models': {
            'test_loss': models[best_loss_idx],
            'frechet_distance': models[best_fd_idx],
            'mae': models[best_mae_idx],
            'pcc': models[best_pcc_idx]
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
    
    results = convert_to_serializable(results)
    
    with open(output_path / 'comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path / 'comparison_results.json'}")
    
    # Generate comparison plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['test_loss', 'frechet_distance', 'mae', 'pcc']
    titles = ['Test Loss', 'Frechet Distance', 'MAE', 'Pearson Correlation']
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        values = [r.get(metric, float('nan')) for r in results_list]
        
        if not all(np.isnan(values)):
            bars = ax.bar(models, values, color=colors)
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=15)
            
            # Add value labels
            for i, (bar, v) in enumerate(zip(bars, values)):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2, v, 
                           f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved to: {output_path / 'model_comparison.png'}")
    
    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compare SB, OT, and VAE models on Setting 1'
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
        help='Cells per timepoint for Setting 1'
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
        default='./outputs_compare',
        help='Output directory'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    run_comparison_experiment(
        file_path=args.file_path,
        n_hvg=args.n_hvg,
        cells_per_timepoint=args.cells_per_timepoint,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
