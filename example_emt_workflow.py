#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Workflow for EMT Dataset with Neural ODE-based Continuous Time Generation
Complete pipeline from data generation to model training and evaluation
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Data import create_neural_ode_emt_generator, create_default_emt_dataset
from Analyser.entropy_metrics import analyze_continuous_data_quality, calculate_entropy_timeline
from Analyser.real_data_metrics import evaluate_model_on_dataset
from Model.vae_model import VAEModel
from Trainer.trainer import Trainer


def step1_generate_continuous_data(
    output_path: str,
    n_hvg: int = 100,
    cells_per_label: int = 2000,
    time_granularity: float = 1.0,
    device: str = 'cuda',
    ode_epochs: int = 50
):
    """
    Step 1: Generate continuous time data using Neural ODE
    
    Args:
        output_path: Path to save generated h5ad file
        n_hvg: Number of highly variable genes
        cells_per_label: Number of cells to sample per time label
        time_granularity: Time step in hours
        device: Device for Neural ODE training
        ode_epochs: Training epochs for Neural ODE
    """
    print("\n" + "="*70)
    print("STEP 1: Generate Continuous Time Data with Neural ODE")
    print("="*70)
    
    generator = create_neural_ode_emt_generator(
        output_path=output_path,
        n_hvg=n_hvg,
        cells_per_label=cells_per_label,
        time_granularity=time_granularity,
        device=device,
        ode_epochs=ode_epochs
    )
    
    adata_continuous = generator.save_continuous_data(output_path)
    
    print(f"\n✓ Continuous time data saved to: {output_path}")
    print(f"  Shape: {adata_continuous.shape[0]} cells × {adata_continuous.shape[1]} genes")
    print(f"  Real cells: {adata_continuous.obs['is_real'].sum()}")
    print(f"  Generated cells: {(~adata_continuous.obs['is_real']).sum()}")
    
    return adata_continuous


def step2_analyze_data_quality(
    continuous_data_path: str,
    output_dir: str
):
    """
    Step 2: Analyze data quality with entropy metrics
    
    Args:
        continuous_data_path: Path to continuous time h5ad file
        output_dir: Directory to save analysis results
    """
    print("\n" + "="*70)
    print("STEP 2: Analyze Data Quality")
    print("="*70)
    
    analyze_continuous_data_quality(
        adata_path=continuous_data_path,
        output_dir=output_dir
    )
    
    print(f"\n✓ Quality analysis saved to: {output_dir}")


def step3_create_datasets(
    continuous_data_path: str,
    sampling_strategy: str = 'all_time',
    train_ratio: float = 0.8,
    batch_size: int = 128
):
    """
    Step 3: Create train/test datasets
    
    Args:
        continuous_data_path: Path to continuous time h5ad file
        sampling_strategy: 'all_time', 'specific_time', or 'clustered_time'
        train_ratio: Train ratio for 'all_time' strategy
        batch_size: Batch size for dataloaders
    
    Returns:
        (train_loader, test_loader, stats)
    """
    print("\n" + "="*70)
    print("STEP 3: Create Train/Test Datasets")
    print("="*70)
    
    train_loader, test_loader, stats = create_default_emt_dataset(
        continuous_data_path=continuous_data_path,
        sampling_strategy=sampling_strategy,
        train_ratio=train_ratio,
        batch_size=batch_size
    )
    
    print("\nDataset Statistics:")
    print("-" * 50)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n✓ Datasets created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, stats


def step4_train_model(
    train_loader,
    test_loader,
    n_genes: int,
    output_dir: str,
    n_epochs: int = 100,
    learning_rate: float = 1e-3,
    latent_dim: int = 32
):
    """
    Step 4: Train VAE model
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        n_genes: Number of genes (HVG dimension)
        output_dir: Directory to save model and results
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        latent_dim: Latent dimension for VAE
    
    Returns:
        Trained model
    """
    print("\n" + "="*70)
    print("STEP 4: Train VAE Model")
    print("="*70)
    
    # Create model
    model = VAEModel(
        dimension=n_genes,  # Use HVG dimension
        encoder_dims=[256, 128],
        latent_dim=latent_dim,
        decoder_dims=[128, 256],
        dropout=0.1,
        beta=1.0
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        learning_rate=learning_rate,
        device=device,
        output_dir=output_dir
    )
    
    # Train
    print(f"\nTraining for {n_epochs} epochs...")
    history = trainer.train(n_epochs=n_epochs)
    
    # Save model
    model_path = Path(output_dir) / "vae_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    return model, history


def step5_evaluate_model(
    model,
    test_loader,
    continuous_data_path: str,
    output_dir: str
):
    """
    Step 5: Evaluate model with Frechet distance, MAE, PCC, and Entropy
    
    Args:
        model: Trained model
        test_loader: Test data loader
        continuous_data_path: Path to continuous time h5ad file
        output_dir: Directory to save evaluation results
    """
    print("\n" + "="*70)
    print("STEP 5: Evaluate Model")
    print("="*70)
    
    import scanpy as sc
    
    # Load continuous data
    adata = sc.read_h5ad(continuous_data_path)
    
    # Get test indices (cells in test_loader)
    # For simplicity, we'll use all generated cells vs all real cells
    is_real = adata.obs['is_real'].values
    real_indices = np.where(is_real)[0]
    generated_indices = np.where(~is_real)[0]
    
    # Split real cells into train/test
    np.random.shuffle(real_indices)
    n_test = len(real_indices) // 5  # 20% for test
    test_indices = real_indices[:n_test]
    
    # Evaluate
    results = evaluate_model_on_dataset(
        adata_continuous=adata,
        train_indices=real_indices[n_test:],
        test_indices=test_indices,
        generated_indices=generated_indices,
        verbose=True
    )
    
    # Calculate entropy for test set
    print("\nCalculating entropy for test set...")
    test_adata = adata[test_indices]
    test_entropy_df = calculate_entropy_timeline(test_adata)
    test_entropy_mean = test_entropy_df['entropy'].mean()
    
    # Calculate entropy for generated set
    print("Calculating entropy for generated set...")
    gen_adata = adata[generated_indices]
    gen_entropy_df = calculate_entropy_timeline(gen_adata)
    gen_entropy_mean = gen_entropy_df['entropy'].mean()
    
    results['test_entropy_mean'] = test_entropy_mean
    results['generated_entropy_mean'] = gen_entropy_mean
    results['entropy_difference'] = abs(test_entropy_mean - gen_entropy_mean)
    
    print(f"\nTest set mean entropy: {test_entropy_mean:.4f}")
    print(f"Generated set mean entropy: {gen_entropy_mean:.4f}")
    print(f"Entropy difference: {results['entropy_difference']:.4f}")
    
    # Save results
    import json
    results_path = Path(output_dir) / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation results saved to: {results_path}")
    
    return results


def run_complete_workflow(
    base_output_dir: str = "/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/neural_ode_compare",
    n_hvg: int = 100,
    cells_per_label: int = 2000,
    time_granularity: float = 1.0,
    sampling_strategy: str = 'all_time',
    n_epochs: int = 100,
    ode_epochs: int = 50,
    device: str = 'cuda'
):
    """
    Run complete workflow from data generation to evaluation
    
    Args:
        base_output_dir: Base directory for all outputs
        n_hvg: Number of HVGs
        cells_per_label: Cells per time label
        time_granularity: Time step in hours
        sampling_strategy: Dataset sampling strategy
        n_epochs: Training epochs for VAE
        ode_epochs: Training epochs for Neural ODE
        device: Device for training
    """
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("EMT CONTINUOUS TIME DATA WORKFLOW (Neural ODE)")
    print("="*70)
    print(f"Output directory: {base_output_dir}")
    print(f"HVG count: {n_hvg}")
    print(f"Cells per label: {cells_per_label}")
    print(f"Time granularity: {time_granularity} hours")
    print(f"Sampling strategy: {sampling_strategy}")
    print(f"VAE training epochs: {n_epochs}")
    print(f"Neural ODE training epochs: {ode_epochs}")
    print(f"Device: {device}")
    
    # Paths
    continuous_data_path = str(base_path / "continuous_time_data_ode.h5ad")
    quality_dir = str(base_path / "quality_analysis")
    model_dir = str(base_path / "model")
    
    # Step 1: Generate data with Neural ODE
    adata_continuous = step1_generate_continuous_data(
        output_path=continuous_data_path,
        n_hvg=n_hvg,
        cells_per_label=cells_per_label,
        time_granularity=time_granularity,
        device=device,
        ode_epochs=ode_epochs
    )
    
    # Step 2: Analyze quality
    step2_analyze_data_quality(
        continuous_data_path=continuous_data_path,
        output_dir=quality_dir
    )
    
    # Step 3: Create datasets
    train_loader, test_loader, stats = step3_create_datasets(
        continuous_data_path=continuous_data_path,
        sampling_strategy=sampling_strategy,
        train_ratio=0.8,
        batch_size=128
    )
    
    # Step 4: Train model
    model, history = step4_train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        n_genes=stats['n_genes'],
        output_dir=model_dir,
        n_epochs=n_epochs,
        learning_rate=1e-3,
        latent_dim=32
    )
    
    # Step 5: Evaluate
    results = step5_evaluate_model(
        model=model,
        test_loader=test_loader,
        continuous_data_path=continuous_data_path,
        output_dir=model_dir
    )
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {base_output_dir}")
    print("\nFinal Evaluation Metrics:")
    print("-" * 50)
    print(f"  Frechet Distance: {results['frechet_distance']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"  Mean PCC: {results['mean_pcc']:.4f}")
    print(f"  Test Entropy: {results['test_entropy_mean']:.4f}")
    print(f"  Generated Entropy: {results['generated_entropy_mean']:.4f}")
    print(f"  Entropy Difference: {results['entropy_difference']:.4f}")
    print("\nComparison with Linear Interpolation:")
    print(f"  Linear interpolation results: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/naive_compare/")
    print(f"  Neural ODE results: {base_output_dir}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EMT Continuous Time Data Workflow with Neural ODE')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/neural_ode_compare',
                       help='Base output directory')
    parser.add_argument('--n_hvg', type=int, default=100,
                       help='Number of HVGs')
    parser.add_argument('--cells_per_label', type=int, default=2000,
                       help='Cells per time label')
    parser.add_argument('--granularity', type=float, default=1.0,
                       help='Time granularity (hours)')
    parser.add_argument('--strategy', type=str, default='all_time',
                       choices=['all_time', 'specific_time', 'clustered_time'],
                       help='Sampling strategy')
    parser.add_argument('--epochs', type=int, default=100,
                       help='VAE training epochs')
    parser.add_argument('--ode_epochs', type=int, default=50,
                       help='Neural ODE training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    run_complete_workflow(
        base_output_dir=args.output_dir,
        n_hvg=args.n_hvg,
        cells_per_label=args.cells_per_label,
        time_granularity=args.granularity,
        sampling_strategy=args.strategy,
        n_epochs=args.epochs,
        ode_epochs=args.ode_epochs,
        device=args.device
    )
