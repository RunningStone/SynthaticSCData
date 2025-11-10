#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Experiment Runner for Real Data Time Series Analysis

Compares Setting 1 (boundary only) vs Setting 2 (all timepoints)
using Schrödinger Bridge model
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
from Model import SchrodingerBridgeModel, MLPlus_SchrodingerBridgeModel
from Trainer import SBTrainer, Evaluator


def run_experiment(
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
    Run complete experiment comparing Setting 1 and Setting 2
    
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
    
    print("="*70)
    print("Real Data Time Series Experiment")
    print("Setting 1 (Boundary) vs Setting 2 (All Timepoints)")
    print("="*70)
    
    # ========== Step 1: Load and Analyze Data ==========
    print("\n[Step 1/5] Loading and analyzing data...")
    
    loader = create_default_emt_data_loader(file_path=file_path, n_hvg=n_hvg)
    loader.load_and_analyze()
    
    # Validate biology split
    valid = loader.validate_biology_split()
    if not valid:
        print("\n⚠️  Warning: Biology split validation failed!")
        print("Continuing anyway, but results may not be reliable.")
    
    # ========== Step 2: Prepare Setting 1 Data ==========
    print("\n[Step 2/5] Preparing Setting 1 (boundary only)...")
    
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
    print("\n[Step 3/5] Preparing Setting 2 (all timepoints)...")
    
    # Use per-timepoint balance: each timepoint has same cells as Setting 1
    # This increases total data for Setting 2 to leverage more temporal information
    X_train_s2, y_train_s2, X_test_s2, y_test_s2 = loader.get_data_for_setting(
        setting=2,
        cells_per_timepoint=cells_per_timepoint_s1,
        balance_strategy='per_timepoint'  # Default: each timepoint gets cells_per_timepoint_s1
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
    
    # ========== Step 4: Train SB Models ==========
    print("\n[Step 4/5] Training Schrödinger Bridge models...")
    
    dimension = n_hvg
    
    # Setting 1 model
    print("\n" + "-"*70)
    print("Training Setting 1 (Boundary Only)")
    print("-"*70)
    
    sb_model_s1 = SchrodingerBridgeModel(
        dimension=dimension,
        hidden_dims=[512, 512, 512, 512],
        time_embedding_dim=64,
        dropout=0.1,
        diffusion_coeff=0.1
    ).to(device)
    
    trainer_s1 = SBTrainer(
        model=sb_model_s1,
        train_loader=train_loader_s1,
        test_loader=test_loader_s1,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_path / 'setting1',
        weight_decay=1e-5,
        grad_clip_norm=5.0
    )
    
    history_s1 = trainer_s1.train(
        epochs=epochs,
        early_stopping_patience=30  # Increased for LR scheduler
    )
    
    # Setting 2 model - Use MLPlus for better multi-timepoint modeling
    print("\n" + "-"*70)
    print("Training Setting 2 (All Timepoints) - Using MLPlus Model")
    print("-"*70)
    
    sb_model_s2 = MLPlus_SchrodingerBridgeModel(
        dimension=dimension,
        hidden_dim=512,
        n_blocks=4,
        time_embedding_dim=64,
        n_time_frequencies=10,
        dropout=0.1,
        diffusion_coeff=0.1
    ).to(device)
    
    # Use higher learning rate for MLPlus model (deeper network needs larger LR)
    mlplus_lr = learning_rate * 2.0  # 2x learning rate for MLPlus
    
    trainer_s2 = SBTrainer(
        model=sb_model_s2,
        train_loader=train_loader_s2,
        test_loader=test_loader_s2,
        learning_rate=mlplus_lr,
        device=device,
        output_dir=output_path / 'setting2',
        weight_decay=1e-5,
        grad_clip_norm=10.0  # More relaxed clipping for deeper model
    )
    
    print(f"MLPlus Learning Rate: {mlplus_lr:.2e} (2x base rate)")
    print(f"Gradient Clipping: 10.0 (vs 5.0 for base model)")
    
    history_s2 = trainer_s2.train(
        epochs=epochs,
        early_stopping_patience=30  # Increased for LR scheduler
    )
    
    # ========== Step 5: Evaluate and Compare ==========
    print("\n[Step 5/5] Evaluating and comparing models...")
    
    evaluator = Evaluator(device=device)
    
    # Evaluate Setting 1
    print("\nEvaluating Setting 1...")
    results_s1 = evaluator.evaluate(
        model=sb_model_s1,
        test_loader=test_loader_s1,
        time_labels=loader.time_label_order
    )
    
    # Evaluate Setting 2
    print("\nEvaluating Setting 2...")
    results_s2 = evaluator.evaluate(
        model=sb_model_s2,
        test_loader=test_loader_s2,
        time_labels=loader.time_label_order
    )
    
    # Compare results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\nSetting 1 (Boundary Only):")
    print(f"  Test Loss: {results_s1['test_loss']:.6f}")
    print(f"  Frechet Distance: {results_s1.get('frechet_distance', 'N/A')}")
    print(f"  MAE: {results_s1.get('mae', 'N/A')}")
    print(f"  PCC: {results_s1.get('pcc', 'N/A')}")
    
    print("\nSetting 2 (All Timepoints):")
    print(f"  Test Loss: {results_s2['test_loss']:.6f}")
    print(f"  Frechet Distance: {results_s2.get('frechet_distance', 'N/A')}")
    print(f"  MAE: {results_s2.get('mae', 'N/A')}")
    print(f"  PCC: {results_s2.get('pcc', 'N/A')}")
    
    # Performance improvement
    if results_s1['test_loss'] > 0:
        improvement = (results_s1['test_loss'] - results_s2['test_loss']) / results_s1['test_loss'] * 100
        print(f"\nPerformance Improvement: {improvement:.2f}%")
        print(f"Setting 2 is {'BETTER' if improvement > 0 else 'WORSE'} than Setting 1")
    
    # Save results
    results = {
        'setting1': {
            'statistics': stats_s1,
            'training_history': history_s1,
            'evaluation': results_s1
        },
        'setting2': {
            'statistics': stats_s2,
            'training_history': history_s2,
            'evaluation': results_s2
        },
        'comparison': {
            'improvement_percentage': improvement if results_s1['test_loss'] > 0 else None
        }
    }
    
    # Save as JSON (convert numpy types)
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
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path / 'results.json'}")
    
    # Generate comparison plot
    evaluator.plot_comparison(
        results_s1=results_s1,
        results_s2=results_s2,
        save_path=output_path / 'comparison.png'
    )
    
    print(f"✓ Comparison plot saved to: {output_path / 'comparison.png'}")
    
    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run real data time series experiment'
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
    
    run_experiment(
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
