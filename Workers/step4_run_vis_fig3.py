#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 - Figure 3: Causal and Interpolation Analysis
====================================================

This script generates Figure 3 for the paper:
- Figure 3.1: Sequential vs Shuffled intermediate states (Setting2 vs Setting5)
- Figure 3.2: Linear interpolation enhancement ladder (Setting1 -> Setting6 -> Setting2)

Required Settings:
- Setting1: Boundary-only (0d, 7d)
- Setting2: Full trajectory (0d, 8h, 1d, 3d, 7d)
- Setting5: Shuffled/cross-peak (0d, 3d_rm)
- Setting6: With peak point (0d, 7d, 3d_rm)

Usage:
    python step4_run_vis_fig3.py --experiment_dir /path/to/EMT_E2M

Output:
    - {experiment_dir}/vis/Fig3_1.pdf  (Causal ablation comparison)
    - {experiment_dir}/vis/Fig3_2.pdf  (Interpolation ladder)
"""

import argparse
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from Analyser import (
    plot_causal_ablation_comparison,
    plot_interpolation_ladder,
    extract_metrics_from_results,
    METRICS_CONFIG,
)


def setup_logger(output_dir: Path) -> logging.Logger:
    """Setup logger for the script."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger('fig3_visualization')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "fig3_visualization.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_evaluation_results(setting_path: Path) -> Dict:
    """Load evaluation_results.json from a setting directory."""
    results_path = setting_path / 'evaluation_results.json'
    if not results_path.exists():
        raise FileNotFoundError(f"evaluation_results.json not found in {setting_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def run_fig3_visualization(
    experiment_dir: str,
    seed: int = 42
) -> Dict:
    """
    Run Figure 3 visualization pipeline.
    
    Args:
        experiment_dir: Root directory containing Setting* folders
        seed: Random seed
    
    Returns:
        Dictionary containing paths to generated figures
    """
    experiment_base = Path(experiment_dir)
    vis_dir = experiment_base / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Figure3 subfolder
    fig3_dir = vis_dir / 'Figure3'
    fig3_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(vis_dir)
    
    logger.info("=" * 80)
    logger.info("FIGURE 3: Causal and Interpolation Analysis")
    logger.info("=" * 80)
    
    np.random.seed(seed)
    
    # Check required settings exist
    setting1_path = experiment_base / 'Setting1'
    setting2_path = experiment_base / 'Setting2'
    setting5_path = experiment_base / 'Setting5'
    setting6_path = experiment_base / 'Setting6'
    
    missing_settings = []
    if not setting1_path.exists():
        missing_settings.append('Setting1')
    if not setting2_path.exists():
        missing_settings.append('Setting2')
    if not setting5_path.exists():
        missing_settings.append('Setting5')
    if not setting6_path.exists():
        missing_settings.append('Setting6')
    
    if missing_settings:
        raise FileNotFoundError(f"Missing required settings: {missing_settings}")
    
    # =========================================================================
    # Load evaluation results
    # =========================================================================
    logger.info("\n--- Loading Evaluation Results ---")
    
    results_s1 = load_evaluation_results(setting1_path)
    results_s2 = load_evaluation_results(setting2_path)
    results_s5 = load_evaluation_results(setting5_path)
    results_s6 = load_evaluation_results(setting6_path)
    
    logger.info(f"Setting1 models: {list(results_s1.keys())}")
    logger.info(f"Setting2 models: {list(results_s2.keys())}")
    logger.info(f"Setting5 models: {list(results_s5.keys())}")
    logger.info(f"Setting6 models: {list(results_s6.keys())}")
    
    # Extract metrics for all models
    metrics_s1 = {model: extract_metrics_from_results(results_s1, model) 
                  for model in results_s1.keys()}
    metrics_s2 = {model: extract_metrics_from_results(results_s2, model) 
                  for model in results_s2.keys()}
    metrics_s5 = {model: extract_metrics_from_results(results_s5, model) 
                  for model in results_s5.keys()}
    metrics_s6 = {model: extract_metrics_from_results(results_s6, model) 
                  for model in results_s6.keys()}
    
    # Define metrics to plot
    metrics_keys = list(METRICS_CONFIG.keys())
    
    # =========================================================================
    # Figure 3.1: Causal Ablation Comparison (Setting2 vs Setting5)
    # =========================================================================
    logger.info("\n--- Generating Figure 3.1: Causal Ablation Comparison ---")
    
    # Models to compare: VAE, SB_MLPlus, Batch_OT
    models_to_compare = []
    for model in ['vae', 'sb_mlplus', 'batch_ot']:
        if model in metrics_s2 and model in metrics_s5:
            models_to_compare.append(model)
    
    if not models_to_compare:
        logger.warning("No common models found between Setting2 and Setting5")
        models_to_compare = list(set(metrics_s2.keys()) & set(metrics_s5.keys()))
    
    logger.info(f"Comparing models: {models_to_compare}")
    
    fig3_1_path = fig3_dir / 'Fig3_1.pdf'
    plot_causal_ablation_comparison(
        metrics_setting2=metrics_s2,
        metrics_setting5=metrics_s5,
        output_path=fig3_1_path,
        models_to_compare=models_to_compare,
        title="Sequential vs Shuffled Intermediate States (Setting2 vs Setting5)",
        metrics_keys=metrics_keys,
        figsize=(16, 8),
        dpi=300
    )
    logger.info(f"Saved: {fig3_1_path}")
    
    # Also save PNG version
    fig3_1_png = fig3_dir / 'Fig3_1.png'
    plot_causal_ablation_comparison(
        metrics_setting2=metrics_s2,
        metrics_setting5=metrics_s5,
        output_path=fig3_1_png,
        models_to_compare=models_to_compare,
        title="Sequential vs Shuffled Intermediate States (Setting2 vs Setting5)",
        metrics_keys=metrics_keys,
        figsize=(16, 8),
        dpi=300
    )
    
    # =========================================================================
    # Figure 3.2: Linear Interpolation Ladder (Setting1 -> Setting6 -> Setting2)
    # =========================================================================
    logger.info("\n--- Generating Figure 3.2: Interpolation Ladder ---")
    
    # Define model groups: (S1_model, S6_model, S2_model)
    # Map corresponding models across settings
    model_groups = []
    
    # VAE group
    if 'vae' in metrics_s1 and 'vae' in metrics_s6 and 'vae' in metrics_s2:
        model_groups.append(('vae', 'vae', 'vae'))
    
    # SB group: S1 has 'sb', S6 and S2 have 'sb_mlplus'
    if 'sb' in metrics_s1 and 'sb_mlplus' in metrics_s6 and 'sb_mlplus' in metrics_s2:
        model_groups.append(('sb', 'sb_mlplus', 'sb_mlplus'))
    
    # OT group: S1 has 'ot', S6 and S2 have 'batch_ot'
    if 'ot' in metrics_s1 and 'batch_ot' in metrics_s6 and 'batch_ot' in metrics_s2:
        model_groups.append(('ot', 'batch_ot', 'batch_ot'))
    
    if not model_groups:
        logger.warning("No complete model groups found for ladder chart")
        # Try to find any available groups
        for s1_model in metrics_s1.keys():
            for s6_model in metrics_s6.keys():
                for s2_model in metrics_s2.keys():
                    if s1_model == s6_model == s2_model:
                        model_groups.append((s1_model, s6_model, s2_model))
                        break
    
    logger.info(f"Model groups for ladder: {model_groups}")
    
    fig3_2_path = fig3_dir / 'Fig3_2.pdf'
    plot_interpolation_ladder(
        metrics_setting1=metrics_s1,
        metrics_setting2=metrics_s2,
        metrics_setting6=metrics_s6,
        output_path=fig3_2_path,
        model_groups=model_groups,
        title="Linear Interpolation Enhancement: Setting1 → Setting6 → Setting2",
        metrics_keys=metrics_keys,
        figsize=(18, 8),
        dpi=300
    )
    logger.info(f"Saved: {fig3_2_path}")
    
    # Also save PNG version
    fig3_2_png = fig3_dir / 'Fig3_2.png'
    plot_interpolation_ladder(
        metrics_setting1=metrics_s1,
        metrics_setting2=metrics_s2,
        metrics_setting6=metrics_s6,
        output_path=fig3_2_png,
        model_groups=model_groups,
        title="Linear Interpolation Enhancement: Setting1 → Setting6 → Setting2",
        metrics_keys=metrics_keys,
        figsize=(18, 8),
        dpi=300
    )
    
    # =========================================================================
    # Save analysis summary
    # =========================================================================
    summary = {
        'settings_compared': {
            'fig3_1': ['Setting2', 'Setting5'],
            'fig3_2': ['Setting1', 'Setting6', 'Setting2']
        },
        'models_compared': {
            'fig3_1': models_to_compare,
            'fig3_2': model_groups
        },
        'metrics_s1': metrics_s1,
        'metrics_s2': metrics_s2,
        'metrics_s5': metrics_s5,
        'metrics_s6': metrics_s6
    }
    
    summary_path = fig3_dir / 'fig3_analysis_summary.json'
    with open(summary_path, 'w') as f:
        # Convert tuples to lists for JSON serialization
        summary_json = summary.copy()
        summary_json['models_compared']['fig3_2'] = [list(g) for g in model_groups]
        json.dump(summary_json, f, indent=2)
    logger.info(f"Saved analysis summary: {summary_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FIGURE 3 GENERATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput files:")
    logger.info(f"  - {fig3_1_path}")
    logger.info(f"  - {fig3_2_path}")
    
    return {
        'fig3_1': str(fig3_1_path),
        'fig3_2': str(fig3_2_path),
        'vis_dir': str(vis_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 3: Causal and Interpolation Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step4_run_vis_fig3.py --experiment_dir /path/to/EMT_E2M
        """
    )
    
    parser.add_argument(
        '--experiment_dir',
        type=str,
        required=True,
        help='Root experiment directory containing Setting* folders'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_fig3_visualization(
            experiment_dir=args.experiment_dir,
            seed=args.seed
        )
        print(f"\nFigure 3 generated successfully!")
        print(f"Output directory: {results['vis_dir']}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
