#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Setting Visualization Script

Run comprehensive visualization and comparison across multiple experimental settings.

Usage:
    python step2_multi_setting_visualization.py \
        --config_paths path1/experiment_config.yaml path2/experiment_config.yaml \
        --output_dir ./visualizations \
        --n_samples_per_timepoint 500 \
        --n_generate_per_model 500 \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the visualizer
from Analyser.multi_setting_visualizer import MultiSettingVisualizer
from Analyser.multi_setting_visualizer_methods import (
    load_and_sample_data,
    load_models_and_generate,
    compute_embeddings,
    create_metrics_comparison
)
from Analyser.multi_setting_visualizer_viz import (
    create_dynamic_visualization,
    run_full_pipeline
)

# Bind methods to class
MultiSettingVisualizer.load_and_sample_data = load_and_sample_data
MultiSettingVisualizer.load_models_and_generate = load_models_and_generate
MultiSettingVisualizer.compute_embeddings = compute_embeddings
MultiSettingVisualizer.create_metrics_comparison = create_metrics_comparison
MultiSettingVisualizer.create_dynamic_visualization = create_dynamic_visualization
MultiSettingVisualizer.run_full_pipeline = run_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description='Multi-setting visualization and comparison'
    )
    
    parser.add_argument(
        '--config_paths',
        type=str,
        nargs='+',
        required=True,
        help='Paths to experiment_config.yaml files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for visualizations (independent of settings)'
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
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
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
    print("Multi-Setting Visualization")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Config files: {len(args.config_paths)}")
    for cp in args.config_paths:
        print(f"    - {cp}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Samples per timepoint: {args.n_samples_per_timepoint}")
    print(f"  Generate per model: {args.n_generate_per_model}")
    print(f"  Device: {args.device}")
    print(f"  Seed: {args.seed}")
    print()
    
    # Initialize visualizer
    visualizer = MultiSettingVisualizer(
        output_dir=args.output_dir,
        device=args.device,
        random_seed=args.seed
    )
    
    # Run full pipeline
    visualizer.run_full_pipeline(
        config_paths=args.config_paths,
        n_samples_per_timepoint=args.n_samples_per_timepoint,
        n_generate_per_model=args.n_generate_per_model
    )


if __name__ == '__main__':
    main()
