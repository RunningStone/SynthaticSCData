#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 0: Pre-calculation Experiments
Unified script to run all pre-experiment analysis:
1. Data split parameter calculation
2. Model parameter and memory estimation

This script helps prepare optimal experimental configurations before training.

Author: Auto-generated
Date: 2024-11-24
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Analyser.data_split_analyzer import DataSplitAnalyzer
from Analyser.model_param_analyzer import ModelParamAnalyzer


def run_data_split_analysis(
    data_config_path: str,
    output_dir: str,
    settings_config: dict,
    group_definitions: dict = None,
    min_cells_per_category: int = 1000,
    bottleneck_percentage: float = 100.0
):
    """
    Run data split parameter analysis
    
    Args:
        data_config_path: Path to data configuration YAML
        output_dir: Output directory for results
        settings_config: Dict mapping setting names to time point lists
        group_definitions: Dict mapping group names to setting names
        min_cells_per_category: Minimum cells required per category
        bottleneck_percentage: Percentage of bottleneck to use
    """
    print("\n" + "="*80)
    print("PART 1: DATA SPLIT ANALYSIS")
    print("="*80)
    
    analyzer = DataSplitAnalyzer(output_dir=output_dir)
    
    results = analyzer.run_full_analysis(
        data_config_path=data_config_path,
        settings_config=settings_config,
        group_definitions=group_definitions,
        min_cells_per_category=min_cells_per_category,
        bottleneck_percentage=bottleneck_percentage
    )
    
    return results


def run_model_param_analysis(
    output_dir: str,
    model_configs: dict,
    batch_size: int = 256,
    input_dim: int = 100,
    optimizer_type: str = 'adam',
    mixed_precision: bool = False
):
    """
    Run model parameter and memory analysis
    
    Args:
        output_dir: Output directory for results
        model_configs: Dict mapping model names to (model_type, config) tuples
        batch_size: Training batch size
        input_dim: Input dimension
        optimizer_type: Optimizer type
        mixed_precision: Whether using mixed precision
    """
    print("\n" + "="*80)
    print("PART 2: MODEL PARAMETER ANALYSIS")
    print("="*80)
    
    analyzer = ModelParamAnalyzer(output_dir=output_dir)
    
    results = analyzer.run_full_analysis(
        model_configs=model_configs,
        batch_size=batch_size,
        input_dim=input_dim,
        optimizer_type=optimizer_type,
        mixed_precision=mixed_precision
    )
    
    return results


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run pre-experiment analysis for data split and model parameters"
    )
    
    # Common arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./precalc_results',
        help='Output directory for analysis results'
    )
    
    # Data split analysis arguments
    parser.add_argument(
        '--data_config',
        type=str,
        default='configs/data_EMT_Cook_with_label.yaml',
        help='Path to data configuration YAML'
    )
    parser.add_argument(
        '--min_cells',
        type=int,
        default=1000,
        help='Minimum cells per category'
    )
    parser.add_argument(
        '--bottleneck_pct',
        type=float,
        default=100.0,
        help='Percentage of bottleneck capacity to use (0-100)'
    )
    
    # Model analysis arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Training batch size for memory estimation'
    )
    parser.add_argument(
        '--input_dim',
        type=int,
        default=100,
        help='Input dimension (number of HVGs)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'adamw', 'sgd'],
        help='Optimizer type'
    )
    parser.add_argument(
        '--mixed_precision',
        action='store_true',
        help='Use mixed precision training'
    )
    
    # Control which analyses to run
    parser.add_argument(
        '--skip_data',
        action='store_true',
        help='Skip data split analysis'
    )
    parser.add_argument(
        '--skip_model',
        action='store_true',
        help='Skip model parameter analysis'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("STEP 0: PRE-CALCULATION EXPERIMENTS")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    
    # =========================================================================
    # PART 1: Data Split Analysis
    # =========================================================================
    if not args.skip_data:
        # Define settings configuration
        # These are the time points for each experimental setting
        settings_config = {
            'setting1': ['0d', '7d'],  # Boundary only
            'setting2': ['0d', '3d', '7d'],  # With one intermediate
            'setting3': ['0d', '8h', '1d', '3d', '7d'],  # Full forward trajectory
            'setting4': ['0d', '8h', '1d', '3d', '7d', '8h_rm', '1d_rm', '3d_rm'],  # With reversal
            'setting5': ['0d', '3d_rm'],  # Reversal boundary
            'setting6': ['0d', '7d', '3d_rm'],  # Forward peak + reversal endpoint
        }
        
        # Define experimental groups for fair comparison
        # Settings in the same group should use the same total number of cells
        group_definitions = {
            'forward_only': ['setting1', 'setting2', 'setting3'],
            'with_reversal': ['setting4', 'setting5', 'setting6']
        }
        
        data_results = run_data_split_analysis(
            data_config_path=args.data_config,
            output_dir=str(output_dir),
            settings_config=settings_config,
            group_definitions=group_definitions,
            min_cells_per_category=args.min_cells,
            bottleneck_percentage=args.bottleneck_pct
        )
    else:
        print("\n⏭️  Skipping data split analysis")
        data_results = None
    
    # =========================================================================
    # PART 2: Model Parameter Analysis
    # =========================================================================
    if not args.skip_model:
        # Define model configurations to analyze
        # Format: model_name -> (model_type, config_dict)
        # Note: All models use 'dimension' as the parameter name for input dimension
        model_configs = {
            'SB_Base': ('sb', {
                'dimension': args.input_dim,
                'hidden_dims': [256, 256, 256],
                'time_embedding_dim': 64
            }),
            'SB_MLPlus': ('sb_mlplus', {
                'dimension': args.input_dim,
                'hidden_dim': 256,
                'n_blocks': 3,
                'time_embedding_dim': 64,
                'n_time_frequencies': 10
            }),
            'OT': ('ot', {
                'dimension': args.input_dim,
                'hidden_dims': [256, 256, 256]
            }),
            'ConditionalVAE': ('vae', {
                'dimension': args.input_dim,
                'n_timepoints': 5,
                'latent_dim': 64,
                'hidden_dims': [256, 256]
            }),
            'BatchOT': ('batch_ot', {
                'dimension': args.input_dim,
                'n_timepoints': 4,
                'time_labels': ['0d', '1d', '3d', '7d'],  # 4 timepoints
                'hidden_dims': [256, 256, 256]
            }),
        }
        
        model_results = run_model_param_analysis(
            output_dir=str(output_dir),
            model_configs=model_configs,
            batch_size=args.batch_size,
            input_dim=args.input_dim,
            optimizer_type=args.optimizer,
            mixed_precision=args.mixed_precision
        )
    else:
        print("\n⏭️  Skipping model parameter analysis")
        model_results = None
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if data_results is not None:
        print("\n✓ Data Split Analysis Complete")
        print(f"  Analyzed {len(data_results)} settings")
        print(f"  Results saved to: {output_dir}/data_split_analysis_*")
    
    if model_results is not None:
        print("\n✓ Model Parameter Analysis Complete")
        print(f"  Analyzed {len(model_results)} models")
        print(f"  Results saved to: {output_dir}/model_param_analysis_*")
    
    print("\n" + "="*80)
    print("✓ ALL PRE-CALCULATION EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review the analysis results")
    print("2. Update your experiment configurations based on recommendations")
    print("3. Run step1_run_experiment.py to start training")
    print()


if __name__ == '__main__':
    main()
