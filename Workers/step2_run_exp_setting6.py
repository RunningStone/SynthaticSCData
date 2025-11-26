#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Run Experiment Setting 6 (Interpolated Data)

This script runs the complete Setting 6 experiment pipeline:
1. Generate linearly interpolated intermediate states from boundary data
2. Train models on the interpolated dataset
3. Run interpolation quality analysis

The script integrates with the modular configuration system and uses:
- Data/InterpolatedDataLoader for data generation
- Trainer/run_experiment_from_config for model training
- Analyser/InterpolationAnalyzer for quality analysis

Usage:
    python step2_run_exp_setting6.py CONFIG_FILE [OPTIONS]
    
Example:
    python step2_run_exp_setting6.py experiment_EMT_Part1_setting6_interpolated.yaml \\
        --config_dir configs/EMT_E2M \\
        --output_dir /path/to/output
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Data import ConfigLoader, setup_logging, create_data_loader_from_config
from Trainer import run_experiment_from_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Experiment Setting 6: Interpolated Data Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config directory
  python step2_run_exp_setting6.py experiment_EMT_Part1_setting6_interpolated.yaml
  
  # Specify custom config directory
  python step2_run_exp_setting6.py experiment_EMT_Part1_setting6_interpolated.yaml \\
      --config_dir configs/EMT_E2M
  
  # Override output directory
  python step2_run_exp_setting6.py experiment_EMT_Part1_setting6_interpolated.yaml \\
      --output_dir /custom/output/path
  
  # Skip training (only generate data)
  python step2_run_exp_setting6.py experiment_EMT_Part1_setting6_interpolated.yaml \\
      --skip_training
  
  # Skip data generation (use existing interpolated data)
  python step2_run_exp_setting6.py experiment_EMT_Part1_setting6_interpolated.yaml \\
      --skip_data_generation
        """
    )
    
    parser.add_argument(
        'config_file',
        type=str,
        help='Experiment configuration file (e.g., experiment_EMT_Part1_setting6_interpolated.yaml)'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default='configs',
        help='Directory containing configuration files (default: configs)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory from config file'
    )
    parser.add_argument(
        '--skip_data_generation',
        action='store_true',
        help='Skip interpolated data generation (use existing data)'
    )
    parser.add_argument(
        '--skip_training',
        action='store_true',
        help='Skip model training (only generate data)'
    )
    parser.add_argument(
        '--skip_analysis',
        action='store_true',
        help='Skip post-training analysis'
    )
    parser.add_argument(
        '--data_only',
        action='store_true',
        help='Only generate interpolated data, skip training and analysis'
    )
    
    return parser.parse_args()


def print_header():
    """Print experiment header."""
    print("=" * 70)
    print("EXPERIMENT SETTING 6: INTERPOLATED DATA TRAINING")
    print("=" * 70)
    print()
    print("This experiment tests whether geometric interpolation between")
    print("boundary states can replace real intermediate observations.")
    print()
    print("Pipeline:")
    print("  1. Generate linearly interpolated intermediate states")
    print("  2. Train models on boundary (real) + intermediate (interpolated)")
    print("  3. Evaluate and compare with Setting 1 and Setting 2")
    print("=" * 70)


def step1_verify_interpolation_config(config, logger):
    """
    Step 1: Verify interpolation configuration.
    
    Args:
        config: Loaded experiment configuration
        logger: Logger instance
        
    Returns:
        Dictionary with interpolation parameters
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Verifying Interpolation Configuration")
    logger.info("=" * 70)
    
    # Check for interpolation params in data setting
    setting_config = config['data_setting']
    interp_params = setting_config.get('interpolation_params')
    
    if not interp_params:
        logger.error("No interpolation_params found in data setting!")
        logger.error("This experiment requires interpolation_params to be defined.")
        raise ValueError("Missing interpolation_params in configuration")
    
    logger.info(f"✓ Interpolation parameters found:")
    logger.info(f"  - Boundary timepoints: {interp_params['boundary_timepoints']}")
    logger.info(f"  - Intermediate timepoints: {interp_params['intermediate_timepoints']}")
    logger.info(f"  - Samples per timepoint: {interp_params['n_samples_per_timepoint']}")
    
    return interp_params


def step2_generate_interpolated_data(config, logger, skip=False):
    """
    Step 2: Generate interpolated data using InterpolatedDataLoader.
    
    Args:
        config: Loaded experiment configuration
        logger: Logger instance
        skip: Whether to skip data generation
        
    Returns:
        DataLoader instance with interpolated data
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Generating Interpolated Data")
    logger.info("=" * 70)
    
    if skip:
        logger.info("⏭️  Skipping data generation (--skip_data_generation)")
        return None
    
    # Create data loader (InterpolatedDataLoader will be selected based on config)
    loader = create_data_loader_from_config(config, logger)
    
    # Load and generate interpolated data
    loader.load_and_analyze()
    
    # Validate the split
    loader.validate_biology_split()
    
    # Log summary
    logger.info("\n✓ Interpolated data generation complete")
    logger.info(f"  - Total cells: {loader.adata_hvg.shape[0]}")
    logger.info(f"  - Total genes: {loader.adata_hvg.shape[1]}")
    
    if 'data_source' in loader.adata_hvg.obs.columns:
        source_counts = loader.adata_hvg.obs['data_source'].value_counts()
        logger.info(f"  - Real cells: {source_counts.get('real', 0)}")
        logger.info(f"  - Interpolated cells: {source_counts.get('interpolated', 0)}")
    
    return loader


def step3_train_models(config, logger, skip=False):
    """
    Step 3: Train models using the standard training pipeline.
    
    Args:
        config: Loaded experiment configuration
        logger: Logger instance
        skip: Whether to skip training
        
    Returns:
        Training results dictionary
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Training Models")
    logger.info("=" * 70)
    
    if skip:
        logger.info("⏭️  Skipping model training (--skip_training)")
        return None
    
    # Run training using the standard pipeline
    results = run_experiment_from_config(config, logger)
    
    logger.info("\n✓ Model training complete")
    
    return results


def step4_run_analysis(config, logger, skip=False):
    """
    Step 4: Run interpolation quality analysis.
    
    Args:
        config: Loaded experiment configuration
        logger: Logger instance
        skip: Whether to skip analysis
        
    Returns:
        Analysis results dictionary
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Running Interpolation Quality Analysis")
    logger.info("=" * 70)
    
    if skip:
        logger.info("⏭️  Skipping analysis (--skip_analysis)")
        return None
    
    # Check if post_analysis is enabled
    post_analysis = config.get('post_analysis', {})
    if not post_analysis.get('enabled', False):
        logger.info("ℹ️  Post-analysis is disabled in configuration")
        return None
    
    # Import analyzer
    from Analyser import InterpolationAnalyzer
    
    # Create analyzer
    output_dir = Path(config['settings']['output_dir'])
    analysis_dir = output_dir / 'interpolation_analysis'
    
    analyzer = InterpolationAnalyzer(
        output_dir=analysis_dir,
        device=config['settings']['device'],
        random_seed=config['settings']['seed']
    )
    
    logger.info(f"✓ Analysis directory: {analysis_dir}")
    logger.info("ℹ️  Full analysis requires generated samples from trained models.")
    logger.info("   Run step3_multi_setting_visualization.py for complete comparison.")
    
    return {'analyzer': analyzer, 'analysis_dir': analysis_dir}


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print header
    print_header()
    
    # Load configuration
    print(f"\nLoading configuration: {args.config_file}")
    print(f"Config directory: {args.config_dir}")
    
    config_loader = ConfigLoader(config_dir=args.config_dir)
    config = config_loader.load_experiment_config(args.config_file)
    
    # Override output directory if specified
    if args.output_dir is not None:
        original_output = Path(config['settings']['output_dir'])
        base_output = Path(args.output_dir)
        
        # Preserve relative structure
        relative_parts = original_output.parts[-2:]
        new_output_dir = base_output / Path(*relative_parts)
        config['settings']['output_dir'] = str(new_output_dir)
        print(f"Output directory overridden: {new_output_dir}")
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")
    logger.info(f"Output directory: {config['settings']['output_dir']}")
    
    try:
        # Step 1: Verify configuration
        interp_params = step1_verify_interpolation_config(config, logger)
        
        # Step 2: Generate interpolated data
        skip_data = args.skip_data_generation or args.skip_training
        loader = step2_generate_interpolated_data(config, logger, skip=skip_data)
        
        # Check if data_only mode
        if args.data_only:
            logger.info("\n" + "=" * 70)
            logger.info("DATA GENERATION COMPLETE (--data_only mode)")
            logger.info("=" * 70)
            return
        
        # Step 3: Train models
        skip_training = args.skip_training
        results = step3_train_models(config, logger, skip=skip_training)
        
        # Step 4: Run analysis
        skip_analysis = args.skip_analysis or args.skip_training
        analysis_results = step4_run_analysis(config, logger, skip=skip_analysis)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("EXPERIMENT SETTING 6 COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nResults saved to: {config['settings']['output_dir']}")
        logger.info("\nNext steps:")
        logger.info("  1. Review training logs and metrics")
        logger.info("  2. Run step3_multi_setting_visualization.py for comparison")
        logger.info("  3. Analyze interpolation effectiveness vs real intermediate states")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
