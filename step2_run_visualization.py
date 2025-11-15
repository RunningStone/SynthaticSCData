#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Generation Visualization

Command-line script to visualize model generation results based on YAML configs
"""

import argparse
import torch
import yaml
from pathlib import Path
from Analyser import GenerationVisualizer
from Data import ConfigLoader


def load_model_configs_from_experiment(experiment_config_path: str, config_dir: str = 'configs'):
    """
    Load model configurations from experiment YAML file
    
    Args:
        experiment_config_path: Path to experiment config file
        config_dir: Directory containing config files
        
    Returns:
        Dictionary of model configurations for visualization
    """
    # Load experiment configuration
    config_loader = ConfigLoader(config_dir=config_dir)
    config = config_loader.load_experiment_config(experiment_config_path)
    
    # Extract information
    output_dir = Path(config['settings']['output_dir'])
    checkpoint_subdir = config['settings']['subdirs']['checkpoints']
    models_config = config['models']
    data_config = config['data']
    n_hvg = data_config['data_source']['n_hvg']
    
    # Build model configs for visualization
    model_configs = {}
    
    for model_name in config['models'].keys():
        model_arch = models_config[model_name]['architecture']
        checkpoint_path = output_dir / checkpoint_subdir / model_name / 'best_model.pt'
        
        # Build model_kwargs based on model type
        model_kwargs = {'dimension': n_hvg}
        
        if model_name == 'sb':
            model_kwargs.update({
                'hidden_dims': model_arch['hidden_dims'],
                'time_embedding_dim': model_arch['time_embedding_dim'],
                'dropout': model_arch['dropout'],
                'diffusion_coeff': model_arch['diffusion_coeff']
            })
        elif model_name == 'sb_mlplus':
            model_kwargs.update({
                'hidden_dim': model_arch['hidden_dim'],
                'n_blocks': model_arch['n_blocks'],
                'time_embedding_dim': model_arch['time_embedding_dim'],
                'n_time_frequencies': model_arch['n_time_frequencies'],
                'dropout': model_arch['dropout'],
                'diffusion_coeff': model_arch['diffusion_coeff']
            })
        elif model_name == 'ot':
            model_kwargs.update({
                'hidden_dims': model_arch['hidden_dims'],
                'activation': model_arch['activation'],
                'dropout': model_arch['dropout'],
                'use_residual': model_arch.get('use_residual', True)
            })
        elif model_name == 'vae':
            model_kwargs.update({
                'hidden_dims': model_arch['hidden_dims'],
                'latent_dim': model_arch['latent_dim'],
                'activation': model_arch['activation'],
                'dropout': model_arch['dropout'],
                'beta': model_arch['beta']
            })
        
        # Only add if checkpoint exists
        if checkpoint_path.exists():
            model_configs[f"{model_name.upper()}"] = {
                'type': model_name,
                'checkpoint_path': checkpoint_path,
                'model_kwargs': model_kwargs
            }
    
    return model_configs, config


def main():
    parser = argparse.ArgumentParser(
        description='Visualize model generation results based on experiment YAML configs'
    )
    
    # Experiment config
    parser.add_argument(
        'experiment_config',
        type=str,
        help='Experiment configuration file (e.g., experiment_EMT_setting1.yaml)'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default='configs',
        help='Directory containing configuration files'
    )
    
    # Visualization arguments
    parser.add_argument(
        '--n_samples_per_timepoint',
        type=int,
        default=500,
        help='Number of samples per timepoint from test set'
    )
    parser.add_argument(
        '--n_generate_per_model',
        type=int,
        default=500,
        help='Number of samples to generate per model'
    )
    parser.add_argument(
        '--visualization_output_dir',
        type=str,
        default=None,
        help='Output directory for visualization results (default: use experiment output_dir/visualizations)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for model inference'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Loading Model Configurations from Experiment YAML")
    print("="*80)
    
    # Load model configurations from experiment YAML
    model_configs, config = load_model_configs_from_experiment(
        args.experiment_config,
        args.config_dir
    )
    
    print(f"\nExperiment: {config['experiment']['name']}")
    print(f"Dataset: {config['experiment']['dataset']}")
    print(f"Models found: {list(model_configs.keys())}")
    
    # Get data file path and n_hvg from config
    data_config = config['data']
    file_path = data_config['data_source']['file_path']
    n_hvg = data_config['data_source']['n_hvg']
    
    # Determine visualization output directory
    if args.visualization_output_dir is None:
        vis_output_dir = Path(config['settings']['output_dir']) / config['settings']['subdirs']['visualizations']
    else:
        vis_output_dir = Path(args.visualization_output_dir)
    
    print(f"\nData file: {file_path}")
    print(f"Number of HVGs: {n_hvg}")
    print(f"Visualization output: {vis_output_dir}")
    print()
    
    # Initialize visualizer
    visualizer = GenerationVisualizer(
        file_path=file_path,
        n_hvg=n_hvg,
        output_dir=str(vis_output_dir),
        device=args.device,
        random_seed=args.seed
    )
    
    # Run visualization pipeline
    print("="*80)
    print("Running Visualization Pipeline")
    print("="*80)
    print()
    
    visualizer.run_full_pipeline(
        model_configs=model_configs,
        n_samples_per_timepoint=args.n_samples_per_timepoint,
        n_generate_per_model=args.n_generate_per_model
    )
    
    print()
    print("="*80)
    print("Visualization Complete!")
    print(f"Results saved to: {vis_output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
