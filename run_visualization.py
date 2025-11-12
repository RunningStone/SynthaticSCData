#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Generation Visualization

Command-line script to visualize model generation results
"""

import argparse
import torch
from pathlib import Path
from Analyser import GenerationVisualizer


def main():
    parser = argparse.ArgumentParser(
        description='Visualize model generation results with PHATE and LMNN+PCA'
    )
    
    # Data arguments
    parser.add_argument(
        '--file_path',
        type=str,
        default=None,
        help='Path to h5ad file (default: EMT dataset)'
    )
    parser.add_argument(
        '--n_hvg',
        type=int,
        default=500,
        help='Number of highly variable genes'
    )
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
    
    # Model arguments
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='./outputs',
        help='Base directory containing trained models'
    )
    parser.add_argument(
        '--visualization_output_dir',
        type=str,
        default='./visualization_outputs',
        help='Output directory for visualization results'
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
    
    # Initialize visualizer
    visualizer = GenerationVisualizer(
        file_path=args.file_path,
        n_hvg=args.n_hvg,
        output_dir=args.visualization_output_dir,
        device=args.device,
        random_seed=args.seed
    )
    
    # Define model configurations based on output_base_dir
    output_base = Path(args.output_base_dir)
    
    model_configs = {
        'SB_S1': {
            'type': 'sb',
            'checkpoint_path': output_base / 'setting1' / 'sb_model' / 'best_model.pt',
            'model_kwargs': {
                'dimension': args.n_hvg,
                'hidden_dims': [512, 512, 512, 512],
                'time_embedding_dim': 64,
                'dropout': 0.1,
                'diffusion_coeff': 0.1
            }
        },
        'OT_S1': {
            'type': 'ot',
            'checkpoint_path': output_base / 'setting1' / 'ot_model' / 'best_model.pt',
            'model_kwargs': {
                'dimension': args.n_hvg,
                'hidden_dims': [512, 512, 512, 512],
                'activation': 'relu',
                'dropout': 0.1,
                'use_residual': True
            }
        },
        'VAE_S1': {
            'type': 'vae',
            'checkpoint_path': output_base / 'setting1' / 'vae_model' / 'best_model.pt',
            'model_kwargs': {
                'dimension': args.n_hvg,
                'latent_dim': 128,
                'hidden_dims': [512, 256],
                'activation': 'relu',
                'dropout': 0.1,
                'beta': 1.0
            }
        },
        'SB_MLPlus_S2': {
            'type': 'sb_mlplus',
            'checkpoint_path': output_base / 'setting2' / 'sb_mlplus_model' / 'best_model.pt',
            'model_kwargs': {
                'dimension': args.n_hvg,
                'hidden_dim': 512,
                'n_blocks': 4,
                'time_embedding_dim': 64,
                'n_time_frequencies': 10,
                'dropout': 0.1,
                'diffusion_coeff': 0.1
            }
        }
    }
    
    # Run visualization pipeline
    visualizer.run_full_pipeline(
        model_configs=model_configs,
        n_samples_per_timepoint=args.n_samples_per_timepoint,
        n_generate_per_model=args.n_generate_per_model
    )


if __name__ == '__main__':
    main()
