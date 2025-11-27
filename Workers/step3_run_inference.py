#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Run Inference on Test Set
=================================

This script loads trained models from checkpoints and runs inference on the test set only.
It computes evaluation metrics and saves generated data for downstream analysis.

Features:
- Load models from any Setting's checkpoints
- Automatically load experiment_config.yaml from the experiment folder
- Use only Test set data (based on biology split)
- Compute comprehensive evaluation metrics
- Save generated trajectories for visualization

Usage:
    python step3_run_inference.py --experiment_dir /path/to/Setting1
    
    # For Setting4 ablation experiments:
    python step3_run_inference.py --experiment_dir /path/to/Setting4/Setting4_Ablation_Remove1d

Output:
    - {experiment_dir}/evaluation_results.json : Evaluation metrics
    - {experiment_dir}/generated_data/{model}.pkl : Generated data for visualization
"""

import argparse
import torch
import numpy as np
import json
import logging
import pickle
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Data import ConfigLoader, setup_logging, create_dataloaders_from_data
from Data.data_loader import RealDataLoader
from Data.interpolated_data_loader import InterpolatedDataLoader
from Model import (
    SchrodingerBridgeModel,
    MLPlus_SchrodingerBridgeModel,
    OptimalTransportModel,
    BatchOTModel
)
from Model.c_vae_model import ConditionalVAEModel
from Trainer import Evaluator


def create_data_loader_from_config_direct(config: Dict[str, Any], logger: logging.Logger):
    """
    Create data loader directly from a merged experiment config (from experiment_config.yaml).
    This is different from create_data_loader_from_config which loads from separate config files.
    """
    data_source = config['data_source']
    biology_split = config['biology_split']
    setting_config = config['data_setting']
    
    logger.info("="*70)
    logger.info("Creating Data Loader from Experiment Config")
    logger.info("="*70)
    
    # Build biology_split parameters
    if biology_split.get('column_name') is None:
        logger.info("Using random train/test split")
        logger.info(f"  Train ratio: {biology_split.get('train_ratio', 0.8)}")
        split_config = {
            "train_val_column": "random",
            "train_ratio": biology_split.get('train_ratio', 0.8)
        }
    else:
        logger.info(f"Using biology-based split on column: {biology_split['column_name']}")
        logger.info(f"  Train batches: {biology_split.get('train_values', [])}")
        logger.info(f"  Test batches: {biology_split.get('test_values', [])}")
        split_config = {
            "train_val_column": biology_split['column_name'],
            "train": biology_split.get('train_values', []),
            "test": biology_split.get('test_values', [])
        }
    
    # Check if this is an interpolated data setting
    interpolation_params = setting_config.get('interpolation_params')
    seed = config.get('settings', {}).get('seed', 42)
    
    # Get time_labels_order from data_source
    time_labels_order = data_source.get('time_labels_order', [])
    
    if interpolation_params:
        logger.info("Using InterpolatedDataLoader for interpolated data generation")
        loader = InterpolatedDataLoader(
            file_path=data_source['file_path'],
            n_hvg=data_source['n_hvg'],
            obs_time_column=data_source['obs_time_column'],
            time_labels=time_labels_order,
            time_label_order=time_labels_order,
            biology_split=split_config,
            random_seed=seed,
            interpolation_params=interpolation_params
        )
    else:
        logger.info("Using RealDataLoader for standard data loading")
        loader = RealDataLoader(
            file_path=data_source['file_path'],
            n_hvg=data_source['n_hvg'],
            obs_time_column=data_source['obs_time_column'],
            time_labels=time_labels_order,
            time_label_order=time_labels_order,
            biology_split=split_config,
            random_seed=seed
        )
    
    return loader


def create_logger(output_dir: Path, log_name: str = "inference") -> logging.Logger:
    """Create a logger for inference"""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('inference')
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


def load_model_from_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    model_config: Dict[str, Any],
    dimension: int,
    time_labels: List[str],
    device: str,
    logger: logging.Logger
) -> torch.nn.Module:
    """Load model from checkpoint file
    
    Args:
        model_name: Name of the model (sb, sb_mlplus, ot, vae, batch_ot)
        checkpoint_path: Path to the checkpoint file
        model_config: Model configuration dictionary
        dimension: Feature dimension
        time_labels: List of time labels
        device: Device to load model on
        logger: Logger instance
        
    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading {model_name} from {checkpoint_path}")
    
    arch_config = model_config['architecture']
    
    # Create model architecture
    if model_name == 'sb':
        model = SchrodingerBridgeModel(
            dimension=dimension,
            hidden_dims=arch_config['hidden_dims'],
            time_embedding_dim=arch_config['time_embedding_dim'],
            dropout=arch_config['dropout'],
            diffusion_coeff=arch_config['diffusion_coeff']
        ).to(device)
        
    elif model_name == 'sb_mlplus':
        model = MLPlus_SchrodingerBridgeModel(
            dimension=dimension,
            hidden_dim=arch_config['hidden_dim'],
            n_blocks=arch_config['n_blocks'],
            time_embedding_dim=arch_config['time_embedding_dim'],
            n_time_frequencies=arch_config['n_time_frequencies'],
            dropout=arch_config['dropout'],
            diffusion_coeff=arch_config['diffusion_coeff']
        ).to(device)
        
    elif model_name == 'ot':
        model = OptimalTransportModel(
            dimension=dimension,
            hidden_dims=arch_config['hidden_dims'],
            activation=arch_config['activation'],
            dropout=arch_config['dropout'],
            use_residual=arch_config.get('use_residual', True)
        ).to(device)
        
    elif model_name == 'vae':
        n_timepoints = len(time_labels)
        model = ConditionalVAEModel(
            dimension=dimension,
            n_timepoints=n_timepoints,
            hidden_dims=arch_config['hidden_dims'],
            latent_dim=arch_config['latent_dim'],
            time_embedding_dim=arch_config.get('time_embedding_dim', 64),
            activation=arch_config['activation'],
            dropout=arch_config['dropout'],
            beta=arch_config['beta'],
            mmd_weight=arch_config.get('mmd_weight', 1.0),
            mmd_kernel=arch_config.get('mmd_kernel', 'rbf'),
            mmd_bandwidth=arch_config.get('mmd_bandwidth', 1.0)
        ).to(device)
        
    elif model_name == 'batch_ot':
        n_timepoints = len(time_labels)
        model = BatchOTModel(
            dimension=dimension,
            n_timepoints=n_timepoints,
            time_labels=time_labels,
            hidden_dims=arch_config['hidden_dims'],
            activation=arch_config['activation'],
            dropout=arch_config['dropout'],
            use_residual=arch_config.get('use_residual', True)
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle time labels for batch_ot and vae
    trained_time_labels = None
    
    if model_name in ['batch_ot', 'vae']:
        if 'time_labels' in checkpoint:
            trained_time_labels = checkpoint['time_labels']
        elif model_name == 'batch_ot' and 'time_pairs' in checkpoint:
            time_pairs = checkpoint['time_pairs']
            time_set = set()
            for start, end in time_pairs:
                time_set.add(start)
                time_set.add(end)
            trained_time_labels = []
            seen = set()
            for start, end in time_pairs:
                if start not in seen:
                    trained_time_labels.append(start)
                    seen.add(start)
                if end not in seen:
                    trained_time_labels.append(end)
                    seen.add(end)
        elif model_name == 'vae' and 'model_state_dict' in checkpoint:
            if 'time_embedding.weight' in checkpoint['model_state_dict']:
                n_trained_timepoints = checkpoint['model_state_dict']['time_embedding.weight'].shape[0]
                logger.info(f"VAE was trained with {n_trained_timepoints} timepoints")
                trained_time_labels = [f't{i}' for i in range(n_trained_timepoints)]
        
        if trained_time_labels:
            logger.info(f"Model was trained with time labels: {trained_time_labels}")
            
            # Reinitialize model with correct time configuration
            if model_name == 'batch_ot':
                model = BatchOTModel(
                    dimension=dimension,
                    n_timepoints=len(trained_time_labels),
                    time_labels=trained_time_labels,
                    hidden_dims=arch_config['hidden_dims'],
                    activation=arch_config['activation'],
                    dropout=arch_config['dropout'],
                    use_residual=arch_config.get('use_residual', True)
                ).to(device)
            elif model_name == 'vae':
                model = ConditionalVAEModel(
                    dimension=dimension,
                    n_timepoints=len(trained_time_labels),
                    hidden_dims=arch_config['hidden_dims'],
                    latent_dim=arch_config['latent_dim'],
                    time_embedding_dim=arch_config.get('time_embedding_dim', 64),
                    activation=arch_config['activation'],
                    dropout=arch_config['dropout'],
                    beta=arch_config['beta'],
                    mmd_weight=arch_config.get('mmd_weight', 1.0),
                    mmd_kernel=arch_config.get('mmd_kernel', 'rbf'),
                    mmd_bandwidth=arch_config.get('mmd_bandwidth', 1.0)
                ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✓ Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        logger.info(f"✓ Loaded model weights")
    
    model.eval()
    return model


def run_inference_on_model(
    model_name: str,
    model: torch.nn.Module,
    test_loader,
    time_labels: List[str],
    config: Dict[str, Any],
    logger: logging.Logger,
    output_dir: Path,
    start_timepoint: str = None,
    end_timepoint: str = None
) -> Dict:
    """Run inference on a single model and save results
    
    Args:
        model_name: Name of the model
        model: Loaded model
        test_loader: Test data loader
        time_labels: List of time labels
        config: Configuration dictionary
        logger: Logger instance
        output_dir: Output directory
        start_timepoint: Start timepoint for evaluation
        end_timepoint: End timepoint for evaluation
        
    Returns:
        Dictionary containing evaluation results and generated data info
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running Inference: {model_name.upper()}")
    logger.info(f"{'='*70}")
    
    device = config.get('device', 'cuda')
    
    # Create evaluator
    evaluator = Evaluator(
        device=device,
        model_name=model_name,
        start_timepoint=start_timepoint,
        end_timepoint=end_timepoint
    )
    
    # Run evaluation
    logger.info("Computing evaluation metrics...")
    results = evaluator.evaluate(
        model=model,
        test_loader=test_loader,
        time_labels=time_labels,
        model_name=model_name
    )
    
    logger.info(f"✓ {model_name.upper()} evaluation complete")
    logger.info(f"  Test Loss: {results.get('test_loss', float('nan')):.4f}")
    logger.info(f"  MAE: {results.get('mae', float('nan')):.4f}")
    logger.info(f"  PCC: {results.get('pcc', float('nan')):.4f}")
    logger.info(f"  Frechet Distance: {results.get('frechet_distance', float('nan')):.4f}")
    logger.info(f"  Wasserstein Distance: {results.get('wasserstein_distance', float('nan')):.4f}")
    
    # Generate samples for visualization
    logger.info("Generating samples for visualization...")
    generated_data = evaluator.generate_samples_for_visualization(
        model=model,
        test_loader=test_loader,
        time_labels=time_labels,
        model_name=model_name
    )
    
    # Save generated data to generated_data folder
    generated_dir = output_dir / 'generated_data'
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    pkl_path = generated_dir / f'{model_name}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(generated_data, f)
    
    logger.info(f"✓ Generated samples saved to: {pkl_path}")
    
    return {
        'metrics': results,
        'generated_data_path': str(pkl_path)
    }


def run_inference(
    experiment_dir: str,
    device: str = 'cuda',
    seed: int = 42,
    models_to_evaluate: Optional[List[str]] = None
) -> Dict:
    """Run inference on test set using trained models
    
    Args:
        experiment_dir: Directory containing experiment results (with experiment_config.yaml and checkpoints/)
        device: Device to run inference on
        seed: Random seed
        models_to_evaluate: Optional list of model names to evaluate (default: all found)
        
    Returns:
        Dictionary containing all inference results
    """
    # Setup paths
    experiment_base = Path(experiment_dir)
    checkpoint_base = experiment_base / 'checkpoints'
    
    if not experiment_base.exists():
        raise ValueError(f"Experiment directory not found: {experiment_dir}")
    
    if not checkpoint_base.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_base}")
    
    # Load experiment config
    experiment_config_path = experiment_base / 'experiment_config.yaml'
    if not experiment_config_path.exists():
        raise ValueError(f"experiment_config.yaml not found in {experiment_base}")
    
    with open(experiment_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Create logger
    logger = create_logger(experiment_base, "inference")
    
    logger.info("="*80)
    logger.info("INFERENCE MODE - Test Set Only")
    logger.info("="*80)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Config loaded from: {experiment_config_path}")
    logger.info("")
    
    # Set random seed
    seed = config.get('settings', {}).get('seed', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # Override device
    config['settings']['device'] = device
    
    # Load data
    logger.info("\nLoading test data...")
    data_loader = create_data_loader_from_config_direct(config, logger)
    data_loader.load_and_analyze()
    
    # Get evaluation timepoints from config
    eval_config = config.get('evaluation', {})
    start_timepoint = eval_config.get('start_timepoint', None)
    end_timepoint = eval_config.get('end_timepoint', None)
    
    # If not specified, use first and last from data setting
    if start_timepoint is None:
        start_timepoint = config['data_setting']['time_points'][0]
    if end_timepoint is None:
        end_timepoint = config['data_setting']['time_points'][-1]
    
    logger.info(f"Evaluation timepoints: {start_timepoint} → {end_timepoint}")
    
    # Get test set data only
    biology_split = config['biology_split']
    column_name = biology_split['column_name']
    test_values = biology_split['test_values']
    
    # Create test mask
    test_mask = data_loader.adata_hvg.obs[column_name].isin(test_values)
    
    # Filter by evaluation timepoints
    eval_timepoints = [start_timepoint, end_timepoint]
    time_mask = data_loader.adata_hvg.obs[data_loader.obs_time_column].isin(eval_timepoints)
    
    # Combine masks
    final_mask = test_mask & time_mask
    
    # Extract data
    X_test = data_loader.adata_hvg.X[final_mask]
    if hasattr(X_test, 'toarray'):
        X_test = X_test.toarray()
    y_test_labels = data_loader.adata_hvg.obs[data_loader.obs_time_column][final_mask].values
    
    # Convert labels to indices
    y_test = np.array([data_loader.time_label_order.index(label) for label in y_test_labels])
    
    logger.info(f"\nTest set prepared:")
    logger.info(f"  Total test samples: {len(X_test)}")
    for tp in eval_timepoints:
        count = np.sum(y_test_labels == tp)
        logger.info(f"    {tp}: {count} samples")
    
    # Create dummy train data (needed for interface but not used)
    X_train = X_test[:10]
    y_train = y_test[:10]
    
    # Get batch size from first model config
    batch_size = 256
    if config.get('models'):
        first_model = list(config['models'].keys())[0]
        batch_size = config['models'][first_model].get('training', {}).get('batch_size', 256)
    
    # Create DataLoader
    train_loader, test_loader, stats = create_dataloaders_from_data(
        X_train, y_train, X_test, y_test,
        time_labels=data_loader.time_label_order,
        batch_size=batch_size
    )
    
    logger.info(f"\nData Statistics:")
    logger.info(f"  Test samples: {stats['test_size']}")
    logger.info(f"  Feature dimension: {stats['n_genes']}")
    logger.info(f"  Time labels: {stats['time_labels']}")
    
    dimension = stats['n_genes']
    time_labels = stats['time_labels']
    
    # Output directory is the experiment directory
    output_base = experiment_base
    
    # Find and evaluate all models
    all_results = {}
    
    # Scan checkpoint directory for models
    for model_dir in sorted(checkpoint_base.iterdir()):
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Filter by models_to_evaluate if specified
        if models_to_evaluate and model_name not in models_to_evaluate:
            logger.info(f"Skipping {model_name} (not in models_to_evaluate)")
            continue
        
        # Check if model config exists
        if model_name not in config.get('models', {}):
            logger.warning(f"Skipping {model_name}: no model config found")
            continue
        
        # Find checkpoint file
        checkpoint_path = model_dir / 'best_model.pt'
        if not checkpoint_path.exists():
            checkpoint_path = model_dir / 'final_model.pt'
        
        if not checkpoint_path.exists():
            logger.warning(f"No checkpoint found for {model_name} in {model_dir}")
            continue
        
        try:
            # Load model
            model = load_model_from_checkpoint(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                model_config=config['models'][model_name],
                dimension=dimension,
                time_labels=time_labels,
                device=device,
                logger=logger
            )
            
            # Run inference
            results = run_inference_on_model(
                model_name=model_name,
                model=model,
                test_loader=test_loader,
                time_labels=time_labels,
                config={'device': device},
                logger=logger,
                output_dir=output_base,
                start_timepoint=start_timepoint,
                end_timepoint=end_timepoint
            )
            
            all_results[model_name] = {
                'checkpoint_path': str(checkpoint_path),
                **results
            }
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}", exc_info=True)
    
    # Save results to experiment directory root
    results_path = experiment_base / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n✓ Evaluation results saved to: {results_path}")
    
    logger.info("\n" + "="*80)
    logger.info("INFERENCE COMPLETE!")
    logger.info("="*80)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on test set using trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference for Setting1
  python step3_run_inference.py --experiment_dir /path/to/Setting1

  # Run inference for Setting4 ablation
  python step3_run_inference.py --experiment_dir /path/to/Setting4/Setting4_Ablation_Remove1d

  # Run inference for specific models only
  python step3_run_inference.py --experiment_dir /path/to/Setting1 --models sb_mlplus ot
        """
    )
    
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=True,
        help='Experiment directory containing experiment_config.yaml and checkpoints/'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run inference on (default: cuda)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Specific models to evaluate (default: all found in checkpoints/)'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_inference(
            experiment_dir=args.experiment_dir,
            device=args.device,
            models_to_evaluate=args.models
        )
        return results
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
