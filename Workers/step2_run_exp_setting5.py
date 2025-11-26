#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Setting 8 Experiment: Label-Shuffled Time Series

This script runs the Setting 8 experiment which tests whether models rely on
absolute time labels or can learn dynamics from data geometry.

Design:
    - Exclude boundary timepoints (0d, 7d)
    - Keep only intermediate timepoints (8h, 1d, 3d)
    - Randomly shuffle time labels among intermediate points
    - Train models (sb_mlplus, batch_ot, vae) as in Setting 2
    - Evaluate on true labels to see if model learned real dynamics

Research Question:
    Does the model depend on absolute time labels, or can it learn from
    data structure even when labels are randomized?

Author: Shi Pan
Date: 2024-11-24
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Data import ConfigLoader, setup_logging, RealDataLoader
from Data.label_shuffled_dataset import create_label_shuffled_datasets
from Data.dataset_builder import DatasetBuilder
from Trainer import train_model, Evaluator
import torch
from torch.utils.data import DataLoader
import json


def load_data_for_setting8(config: dict, logger):
    """
    Load and prepare data for Setting 8
    
    Args:
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        (train_loader, test_loader, data_info)
    """
    logger.info("="*80)
    logger.info("Loading Data for Setting 8 (Label-Shuffled)")
    logger.info("="*80)
    
    # Get data configuration from merged config
    # Note: ConfigLoader.load_experiment_config() returns a merged config with:
    #   - config['data_source'] - contains file_path, obs_time_column, etc.
    #   - config['data_setting'] - contains the selected setting (time_labels_order, etc.)
    #   - config['data_setting_name'] - the setting name
    #   - config['biology_split'] - train/test split configuration
    #   - config['data_sampling_override'] - experiment-specific sampling params
    
    data_source = config['data_source']
    data_setting = config['data_setting']
    data_setting_name = config['data_setting_name']
    biology_split = config.get('biology_split', {})
    
    # Get Setting 8 specific parameters
    setting8_params = config.get('data_sampling_override', {}) or {}
    total_cells = setting8_params.get('total_cells', 8974)
    start_timepoint = setting8_params.get('start_timepoint', '0d')
    end_timepoint = setting8_params.get('end_timepoint', '7d')
    shuffle_seed = setting8_params.get('shuffle_seed', 42)
    
    logger.info(f"Data setting: {data_setting_name}")
    logger.info(f"Total cells: {total_cells}")
    logger.info(f"Excluded start: {start_timepoint}")
    logger.info(f"Excluded end: {end_timepoint}")
    logger.info(f"Shuffle seed: {shuffle_seed}")
    
    # Get time labels from data_setting (time_points) or data_source (time_labels_order)
    # data_setting contains 'time_points' for the specific setting
    # data_source contains 'time_labels_order' as the global ordering
    time_points = data_setting.get('time_points', [])
    time_labels_order = data_source.get('time_labels_order', time_points)
    
    # Create data loader using merged config structure
    data_loader = RealDataLoader(
        file_path=data_source['file_path'],
        n_hvg=data_source.get('n_hvg', 100),
        obs_time_column=data_source.get('obs_time_column', 'Ground_truth'),
        time_labels=time_points,  # Filter to only these time points
        time_label_order=time_points,  # Use time_points as the order for this setting
        biology_split=biology_split,
        random_seed=data_source.get('random_seed', 42)
    )
    
    # Load and split data
    logger.info("\nLoading AnnData...")
    data_loader.load_and_analyze()
    
    logger.info("\nSplitting train/test...")
    data_loader.validate_biology_split()
    
    # Get data for the specified setting (use Setting 2 as base)
    logger.info(f"\nExtracting data for {data_setting_name}...")
    X_train, y_train, X_test, y_test = data_loader.get_data_for_setting(
        setting=2,  # Use Setting 2 as base (all timepoints)
        total_cells=total_cells
    )
    
    logger.info(f"Original data shapes:")
    logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Create label-shuffled datasets
    logger.info("\nCreating label-shuffled datasets...")
    train_dataset, test_dataset = create_label_shuffled_datasets(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        time_labels=data_loader.time_label_order,
        start_timepoint=start_timepoint,
        end_timepoint=end_timepoint,
        seed=shuffle_seed
    )
    
    # Create dataloaders
    batch_size = config.get('settings', {}).get('batch_size', 64)
    num_workers = config.get('settings', {}).get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Prepare data info
    data_info = {
        'n_genes': X_train.shape[1],
        'time_labels': data_loader.time_label_order,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'batch_size': batch_size,
        'setting': 'setting8_label_shuffled',
        'excluded_timepoints': [start_timepoint, end_timepoint],
        'shuffle_seed': shuffle_seed
    }
    
    logger.info("\n" + "="*80)
    logger.info("Data Loading Complete")
    logger.info("="*80)
    logger.info(f"Train samples: {data_info['train_size']}")
    logger.info(f"Test samples: {data_info['test_size']}")
    logger.info(f"Number of genes: {data_info['n_genes']}")
    logger.info(f"Time labels: {data_info['time_labels']}")
    logger.info(f"Excluded timepoints: {data_info['excluded_timepoints']}")
    logger.info("="*80 + "\n")
    
    return train_loader, test_loader, data_info


def run_setting8_experiment(config: dict, logger):
    """
    Run complete Setting 8 experiment
    
    Args:
        config: Experiment configuration
        logger: Logger instance
        
    Returns:
        results: Dictionary containing all experiment results
    """
    logger.info("\n" + "="*80)
    logger.info("SETTING 8 EXPERIMENT: Label-Shuffled Time Series")
    logger.info("="*80)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")
    logger.info("="*80 + "\n")
    
    # Load data
    train_loader, test_loader, data_info = load_data_for_setting8(config, logger)
    
    # Get models to train from merged config
    # ConfigLoader.load_experiment_config() processes models_to_train into config['models']
    # config['models'] is a dict: {model_name: model_config}
    models_config = config.get('models', {})
    if not models_config:
        raise ValueError("No models specified in 'models_to_train'")
    
    # Results storage
    results = {
        'experiment_name': config['experiment']['name'],
        'setting': 'setting8_label_shuffled',
        'data_info': data_info,
        'models': {}
    }
    
    # Train each model
    # models_config is a dict: {model_name: model_config} from the merged config
    for model_name, model_cfg in models_config.items():
        logger.info("\n" + "="*80)
        logger.info(f"Training Model: {model_name}")
        logger.info("="*80)
        
        # Make a copy to avoid modifying the original
        model_cfg = model_cfg.copy()
        
        # Add data info
        model_cfg['n_genes'] = data_info['n_genes']
        model_cfg['n_timepoints'] = len(data_info['time_labels'])
        model_cfg['time_labels'] = data_info['time_labels']
        
        # Train model
        try:
            trained_model = train_model(
                model_name=model_name,
                model_config=model_cfg,
                train_loader=train_loader,
                test_loader=test_loader,
                dimension=data_info['n_genes'],
                time_labels=data_info['time_labels'],
                config=config,
                logger=logger
            )
            
            # Evaluate model
            logger.info(f"\nEvaluating {model_name}...")
            evaluator = Evaluator(
                model=trained_model,
                model_name=model_name,
                test_loader=test_loader,
                time_labels=data_info['time_labels'],
                device=config['settings'].get('device', 'cuda'),
                output_dir=Path(config['settings']['output_dir']) / 'evaluation' / model_name
            )
            
            eval_results = evaluator.evaluate(
                start_timepoint=config['evaluation'].get('start_timepoint', '0d'),
                end_timepoint=config['evaluation'].get('end_timepoint', '7d'),
                n_samples=config['evaluation'].get('n_samples_per_model', 1000)
            )
            
            results['models'][model_name] = {
                'training': 'completed',
                'evaluation': eval_results
            }
            
            logger.info(f"✓ {model_name} training and evaluation completed")
            
        except Exception as e:
            logger.error(f"✗ {model_name} failed: {str(e)}", exc_info=True)
            results['models'][model_name] = {
                'training': 'failed',
                'error': str(e)
            }
    
    # Save results
    output_dir = Path(config['settings']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {results_file}")
    logger.info("="*80 + "\n")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run Setting 8 Experiment: Label-Shuffled Time Series',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config directory
  python Workers/step2_run_exp_setting8.py experiment_EMT_Part1_setting8.yaml
  
  # Specify custom config directory
  python Workers/step2_run_exp_setting8.py experiment_EMT_Part1_setting8.yaml --config_dir configs/EMT_E2M
  
  # Override output directory
  python Workers/step2_run_exp_setting8.py experiment_EMT_Part1_setting8.yaml --output_dir /custom/path

Research Question:
  Does the model rely on absolute time labels, or can it learn dynamics
  from data geometry even when time labels are randomized?
  
Design:
  - Exclude boundary timepoints (0d, 7d)
  - Keep intermediate timepoints (8h, 1d, 3d)
  - Randomly shuffle time labels
  - Train models as in Setting 2
  - Evaluate on true labels
        """
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Experiment configuration file (e.g., experiment_EMT_Part1_setting8.yaml)'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default='configs/EMT_E2M',
        help='Directory containing configuration files (default: configs/EMT_E2M)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory from config file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(config_dir=args.config_dir)
    config = config_loader.load_experiment_config(args.config_file)
    
    # Store config_dir in config for later use
    config['config_dir'] = args.config_dir
    
    # Override output directory if specified
    if args.output_dir is not None:
        config['settings']['output_dir'] = args.output_dir
    
    # Setup logging
    logger = setup_logging(config)
    
    # Run experiment
    try:
        results = run_setting8_experiment(config, logger)
        return results
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
