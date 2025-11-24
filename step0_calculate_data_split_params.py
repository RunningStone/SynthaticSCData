#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate Optimal Data Split Parameters for Fair Model Comparison

This script analyzes the dataset and computes optimal sampling parameters
to ensure:
1. Fair comparison across settings (same total training samples)
2. Sufficient samples per category for model convergence (min 1000)
3. Balanced sampling respecting train/test split constraints

Author: Auto-generated
Date: 2024-11-16
"""

import argparse
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import json


def load_data_config(config_path: str) -> Dict:
    """Load data configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def analyze_data_distribution(
    adata_path: str,
    label_column: str,
    batch_column: str,
    train_batches: List[str],
    test_batches: List[str],
    time_labels_order: List[str]
) -> pd.DataFrame:
    """
    Analyze cell distribution across batches and time labels
    
    Returns:
        DataFrame with columns: [time_label, batch, count]
    """
    print("\n" + "="*80)
    print("Loading and Analyzing Data Distribution")
    print("="*80)
    
    # Load data
    print(f"\nLoading: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    print(f"Total cells: {adata.n_obs:,}")
    print(f"Total genes: {adata.n_vars:,}")
    
    # Check columns exist
    if label_column not in adata.obs.columns:
        raise ValueError(f"Label column '{label_column}' not found in adata.obs")
    if batch_column not in adata.obs.columns:
        raise ValueError(f"Batch column '{batch_column}' not found in adata.obs")
    
    # Filter to only include specified time labels
    valid_mask = adata.obs[label_column].isin(time_labels_order)
    adata_filtered = adata[valid_mask].copy()
    n_filtered = adata.n_obs - adata_filtered.n_obs
    if n_filtered > 0:
        print(f"\nFiltered out {n_filtered} cells with labels not in time_labels_order")
    
    # Create distribution table
    results = []
    
    print(f"\n{'Batch':<10} {'Time Label':<12} {'Count':>8}")
    print("-" * 35)
    
    for batch in sorted(adata_filtered.obs[batch_column].unique()):
        batch_mask = adata_filtered.obs[batch_column] == batch
        for time_label in time_labels_order:
            time_mask = adata_filtered.obs[label_column] == time_label
            count = (batch_mask & time_mask).sum()
            results.append({
                'batch': batch,
                'time_label': time_label,
                'count': count
            })
            print(f"{batch:<10} {time_label:<12} {count:>8,}")
    
    df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n" + "="*80)
    print("Train/Test Split Summary")
    print("="*80)
    
    train_mask = adata_filtered.obs[batch_column].isin(train_batches)
    test_mask = adata_filtered.obs[batch_column].isin(test_batches)
    
    print(f"\nTrain batches: {train_batches}")
    print(f"Test batches: {test_batches}")
    print(f"\nTrain cells: {train_mask.sum():,}")
    print(f"Test cells: {test_mask.sum():,}")
    
    print(f"\n{'Time Label':<12} {'Train Count':>12} {'Test Count':>12}")
    print("-" * 40)
    
    for time_label in time_labels_order:
        time_mask = adata_filtered.obs[label_column] == time_label
        train_count = (train_mask & time_mask).sum()
        test_count = (test_mask & time_mask).sum()
        print(f"{time_label:<12} {train_count:>12,} {test_count:>12,}")
    
    return df, adata_filtered.obs


def calculate_setting_params(
    df: pd.DataFrame,
    obs: pd.DataFrame,
    setting_name: str,
    time_points: List[str],
    train_batches: List[str],
    batch_column: str,
    label_column: str,
    min_cells_per_category: int = 1000
) -> Dict:
    """
    Calculate optimal sampling parameters for a specific setting
    
    Args:
        df: Distribution dataframe
        obs: AnnData obs dataframe
        setting_name: Name of the setting (e.g., "setting1")
        time_points: List of time points for this setting
        train_batches: List of training batch names
        batch_column: Column name for batch
        label_column: Column name for time label
        min_cells_per_category: Minimum cells required per category
        
    Returns:
        Dict with keys: max_cells_per_timepoint, bottleneck_timepoint, 
                       available_counts, recommended_cells_per_timepoint
    """
    print("\n" + "="*80)
    print(f"Calculating Parameters for {setting_name}")
    print("="*80)
    print(f"Time points: {time_points}")
    
    # Calculate available cells per timepoint in training set
    train_mask = obs[batch_column].isin(train_batches)
    available_counts = {}
    
    print(f"\n{'Time Point':<12} {'Available (Train)':>20}")
    print("-" * 35)
    
    for time_point in time_points:
        time_mask = obs[label_column] == time_point
        count = (train_mask & time_mask).sum()
        available_counts[time_point] = count
        print(f"{time_point:<12} {count:>20,}")
    
    # Find bottleneck (minimum available)
    bottleneck_timepoint = min(available_counts, key=available_counts.get)
    max_cells_per_timepoint = available_counts[bottleneck_timepoint]
    
    print(f"\n⚠️  Bottleneck: {bottleneck_timepoint} with {max_cells_per_timepoint:,} cells")
    
    # Check if bottleneck meets minimum requirement
    if max_cells_per_timepoint < min_cells_per_category:
        print(f"❌ ERROR: Bottleneck ({max_cells_per_timepoint:,}) < minimum required ({min_cells_per_category:,})")
        recommended = 0
    else:
        # Recommend using 90% of bottleneck to leave safety margin
        recommended = int(max_cells_per_timepoint * 0.9)
        # Round down to nearest 100
        recommended = (recommended // 100) * 100
        print(f"✓ Recommended cells_per_timepoint: {recommended:,} (90% of bottleneck)")
    
    return {
        'setting_name': setting_name,
        'time_points': time_points,
        'n_timepoints': len(time_points),
        'max_cells_per_timepoint': max_cells_per_timepoint,
        'bottleneck_timepoint': bottleneck_timepoint,
        'available_counts': available_counts,
        'recommended_cells_per_timepoint': recommended,
        'min_cells_required': min_cells_per_category
    }


def compute_fair_comparison_params(
    setting_results: List[Dict],
    min_cells_per_category: int = 1000,
    group_definitions: Dict[str, List[str]] = None,
    bottleneck_percentage: float = 100.0
) -> Dict:
    """
    Compute final parameters ensuring fair comparison across settings
    
    Strategy:
    1. Group settings into experimental groups (e.g., forward-only vs with-reversal)
    2. For each group, find the most restrictive setting (smallest max total)
    3. Apply bottleneck_percentage to adjust the target total
    4. Use that as the target total for all settings in the same group
    5. Distribute evenly across timepoints for each setting
    
    Args:
        setting_results: List of results from calculate_setting_params
        min_cells_per_category: Minimum cells per category (default for all groups)
        group_definitions: Dict mapping group names to list of setting names
                          e.g., {'group1': ['setting1', 'setting2', 'setting3'],
                                 'group2': ['setting4', 'setting5', 'setting6']}
        bottleneck_percentage: Percentage of bottleneck capacity to use (0-100)
                              Default 100.0 means use full bottleneck capacity
                              e.g., 90.0 means use 90% of bottleneck
        
    Returns:
        Dict with final parameters for each setting
    """
    print("\n" + "="*80)
    print("Computing Fair Comparison Parameters")
    print("="*80)
    
    # If no group definitions provided, treat all settings as one group
    if group_definitions is None:
        all_settings = [r['setting_name'] for r in setting_results]
        group_definitions = {'all': all_settings}
    
    # Calculate max possible total for each setting
    setting_max_totals = {}
    for result in setting_results:
        setting_name = result['setting_name']
        n_timepoints = result['n_timepoints']
        max_per_tp = result['max_cells_per_timepoint']
        max_total = n_timepoints * max_per_tp
        setting_max_totals[setting_name] = max_total
        print(f"\n{setting_name}:")
        print(f"  Time points: {n_timepoints}")
        print(f"  Max per timepoint: {max_per_tp:,}")
        print(f"  Max total: {max_total:,}")
    
    # Find target total for each group
    group_targets = {}
    for group_name, group_settings in group_definitions.items():
        # Filter to settings in this group
        group_totals = {s: setting_max_totals[s] for s in group_settings if s in setting_max_totals}
        if group_totals:
            bottleneck_setting = min(group_totals, key=group_totals.get)
            raw_target_total = group_totals[bottleneck_setting]
            
            # Apply bottleneck percentage
            adjusted_target_total = int(raw_target_total * (bottleneck_percentage / 100.0))
            
            group_targets[group_name] = {
                'target_total': adjusted_target_total,
                'raw_target_total': raw_target_total,
                'bottleneck_setting': bottleneck_setting,
                'settings': group_settings,
                'bottleneck_percentage': bottleneck_percentage
            }
            print(f"\n📊 {group_name}:")
            print(f"   Settings: {group_settings}")
            print(f"   Bottleneck: {bottleneck_setting}")
            print(f"   Raw target total: {raw_target_total:,}")
            if bottleneck_percentage != 100.0:
                print(f"   Bottleneck percentage: {bottleneck_percentage}%")
                print(f"   Adjusted target total: {adjusted_target_total:,} ({bottleneck_percentage}% of {raw_target_total:,})")
            else:
                print(f"   Target total: {adjusted_target_total:,}")
    
    # Compute final parameters for each setting
    final_params = {}
    
    print("\n" + "="*80)
    print("Final Recommended Parameters")
    print("="*80)
    
    for result in setting_results:
        setting_name = result['setting_name']
        n_timepoints = result['n_timepoints']
        
        # Find which group this setting belongs to
        target_total = None
        for group_name, group_info in group_targets.items():
            if setting_name in group_info['settings']:
                target_total = group_info['target_total']
                break
        
        if target_total is None:
            print(f"\n⚠️  Warning: {setting_name} not in any group, skipping...")
            continue
        
        # Calculate cells per timepoint to achieve target total
        cells_per_tp = target_total // n_timepoints
        
        # Check if this meets minimum requirement
        if cells_per_tp < min_cells_per_category:
            print(f"\n❌ {setting_name}: Cannot meet minimum requirement!")
            print(f"   Calculated: {cells_per_tp:,} < Required: {min_cells_per_category:,}")
            final_params[setting_name] = None
            continue
        
        # Check if this exceeds available capacity
        bottleneck_tp = result['bottleneck_timepoint']
        max_available = result['max_cells_per_timepoint']
        
        if cells_per_tp > max_available:
            print(f"\n⚠️  {setting_name}: Exceeds capacity!")
            print(f"   Calculated: {cells_per_tp:,} > Available: {max_available:,}")
            # Use maximum available instead
            cells_per_tp = max_available
            actual_total = cells_per_tp * n_timepoints
        else:
            actual_total = target_total
        
        # Get bottleneck percentage for this group
        group_bottleneck_pct = 100.0
        for group_name, group_info in group_targets.items():
            if setting_name in group_info['settings']:
                group_bottleneck_pct = group_info.get('bottleneck_percentage', 100.0)
                break
        
        final_params[setting_name] = {
            'cells_per_timepoint': cells_per_tp,
            'total_cells': actual_total,
            'n_timepoints': n_timepoints,
            'per_timepoint_usage': f"{cells_per_tp:,} / {max_available:,} ({100*cells_per_tp/max_available:.1f}%)",
            'bottleneck_timepoint': bottleneck_tp,
            'bottleneck_percentage': group_bottleneck_pct
        }
        
        print(f"\n✓ {setting_name}:")
        print(f"   cells_per_timepoint: {cells_per_tp:,}")
        print(f"   total_cells: {actual_total:,}")
        print(f"   ({n_timepoints} timepoints × {cells_per_tp:,} = {actual_total:,})")
        print(f"   Bottleneck usage: {cells_per_tp:,} / {max_available:,} ({100*cells_per_tp/max_available:.1f}%)")
    
    return final_params


def generate_yaml_snippets(final_params: Dict, output_dir: Path):
    """Generate YAML configuration snippets for each setting"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Generating YAML Configuration Snippets")
    print("="*80)
    
    for setting_name, params in final_params.items():
        if params is None:
            continue
        
        bottleneck_pct = params.get('bottleneck_percentage', 100.0)
        bottleneck_note = f"\n  # - Bottleneck percentage: {bottleneck_pct}%" if bottleneck_pct != 100.0 else ""
        
        yaml_content = f"""# Recommended parameters for {setting_name}
# Auto-generated by step0_calculate_data_split_params.py

{setting_name}:
  cells_per_timepoint: {params['cells_per_timepoint']}
  total_cells: {params['total_cells']}
  
  # Rationale:
  # - {params['n_timepoints']} timepoints × {params['cells_per_timepoint']:,} = {params['total_cells']:,} total
  # - Bottleneck: {params['bottleneck_timepoint']} ({params['per_timepoint_usage']}){bottleneck_note}
  # - Fair comparison: all settings use same total training samples
  # - Sufficient for convergence: each category has ≥1000 samples

# For experiment config override:
data_sampling_override:
  cells_per_timepoint: {params['cells_per_timepoint']}  # For setting1
  total_cells: {params['total_cells']}  # For setting2/setting3
"""
        
        output_file = output_dir / f"{setting_name}_params.yaml"
        with open(output_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate optimal data split parameters for fair model comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python step0_calculate_data_split_params.py \\
    --data_config configs/data_EMT_Cook_with_label.yaml \\
    --settings setting1 setting2 setting3 \\
    --min_cells 1000 \\
    --output_dir ./outputs/split_params
        """
    )
    
    parser.add_argument(
        '--data_config',
        type=str,
        required=True,
        help='Path to data configuration YAML file'
    )
    
    parser.add_argument(
        '--settings',
        type=str,
        nargs='+',
        required=True,
        help='List of setting names to analyze (e.g., setting1 setting2)'
    )
    
    parser.add_argument(
        '--min_cells',
        type=int,
        default=1000,
        help='Minimum cells per category for convergence (default: 1000)'
    )
    
    parser.add_argument(
        '--min_cells_group1',
        type=int,
        default=None,
        help='Minimum cells for experimental group 1 (if different from --min_cells)'
    )
    
    parser.add_argument(
        '--min_cells_group2',
        type=int,
        default=None,
        help='Minimum cells for experimental group 2 (if different from --min_cells)'
    )
    
    parser.add_argument(
        '--group1_settings',
        type=str,
        nargs='+',
        default=None,
        help='Settings in experimental group 1 (e.g., setting1 setting2 setting3)'
    )
    
    parser.add_argument(
        '--group2_settings',
        type=str,
        nargs='+',
        default=None,
        help='Settings in experimental group 2 (e.g., setting4 setting5 setting6)'
    )
    
    parser.add_argument(
        '--bottleneck_percentage',
        type=float,
        default=100.0,
        help='Percentage of bottleneck capacity to use (0-100, default: 100.0)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/split_params',
        help='Output directory for results (default: ./outputs/split_params)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.settings) < 2:
        raise ValueError("At least 2 settings must be specified for fair comparison")
    
    # Validate bottleneck_percentage
    if args.bottleneck_percentage <= 0 or args.bottleneck_percentage > 100:
        raise ValueError(f"bottleneck_percentage must be between 0 and 100, got {args.bottleneck_percentage}")
    
    if args.bottleneck_percentage != 100.0:
        print(f"\n⚙️  Using {args.bottleneck_percentage}% of bottleneck capacity")
    
    # Handle min_cells for different groups
    min_cells_group1 = args.min_cells_group1 if args.min_cells_group1 is not None else args.min_cells
    min_cells_group2 = args.min_cells_group2 if args.min_cells_group2 is not None else args.min_cells
    
    # Warn if only one group-specific min_cells is provided
    if args.min_cells_group1 is not None and args.min_cells_group2 is None:
        print(f"\n⚠️  Warning: Only min_cells_group1 specified, using min_cells={args.min_cells} for group2")
    if args.min_cells_group2 is not None and args.min_cells_group1 is None:
        print(f"\n⚠️  Warning: Only min_cells_group2 specified, using min_cells={args.min_cells} for group1")
    
    # Build group definitions
    group_definitions = None
    if args.group1_settings or args.group2_settings:
        group_definitions = {}
        if args.group1_settings:
            group_definitions['group1'] = args.group1_settings
        if args.group2_settings:
            group_definitions['group2'] = args.group2_settings
    
    # Load data configuration
    print("\n" + "="*80)
    print("Loading Data Configuration")
    print("="*80)
    print(f"Config file: {args.data_config}")
    
    config = load_data_config(args.data_config)
    
    # Extract parameters
    data_source = config['data_source']
    biology_split = config['biology_split']
    
    adata_path = data_source['file_path']
    label_column = data_source['obs_time_column']
    time_labels_order = data_source['time_labels_order']
    batch_column = biology_split['column_name']
    train_batches = biology_split['train_values']
    test_batches = biology_split['test_values']
    
    print(f"\nData file: {adata_path}")
    print(f"Label column: {label_column}")
    print(f"Batch column: {batch_column}")
    print(f"Time labels order: {time_labels_order}")
    print(f"Train batches: {train_batches}")
    print(f"Test batches: {test_batches}")
    
    # Analyze data distribution
    df, obs = analyze_data_distribution(
        adata_path=adata_path,
        label_column=label_column,
        batch_column=batch_column,
        train_batches=train_batches,
        test_batches=test_batches,
        time_labels_order=time_labels_order
    )
    
    # Calculate parameters for each setting
    setting_results = []
    
    for setting_name in args.settings:
        if setting_name not in config:
            print(f"\n⚠️  Warning: {setting_name} not found in config, skipping...")
            continue
        
        setting_config = config[setting_name]
        time_points = setting_config['time_points']
        
        # Determine which min_cells to use for this setting
        min_cells_for_setting = args.min_cells
        if group_definitions:
            if 'group1' in group_definitions and setting_name in group_definitions['group1']:
                min_cells_for_setting = min_cells_group1
            elif 'group2' in group_definitions and setting_name in group_definitions['group2']:
                min_cells_for_setting = min_cells_group2
        
        result = calculate_setting_params(
            df=df,
            obs=obs,
            setting_name=setting_name,
            time_points=time_points,
            train_batches=train_batches,
            batch_column=batch_column,
            label_column=label_column,
            min_cells_per_category=min_cells_for_setting
        )
        
        setting_results.append(result)
    
    # Compute fair comparison parameters
    final_params = compute_fair_comparison_params(
        setting_results=setting_results,
        min_cells_per_category=args.min_cells,
        group_definitions=group_definitions,
        bottleneck_percentage=args.bottleneck_percentage
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    results_json = {
        'input_config': args.data_config,
        'settings_analyzed': args.settings,
        'min_cells_per_category': args.min_cells,
        'setting_details': setting_results,
        'final_parameters': final_params
    }
    
    json_file = output_dir / 'split_params_analysis.json'
    with open(json_file, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    
    print(f"\n✓ Saved detailed results: {json_file}")
    
    # Generate YAML snippets
    generate_yaml_snippets(final_params, output_dir)
    
    # Save summary table
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Data Split Parameters Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Data config: {args.data_config}\n")
        f.write(f"Minimum cells per category: {args.min_cells:,}\n\n")
        
        f.write(f"{'Setting':<15} {'Timepoints':<12} {'Cells/TP':<12} {'Total':<12} {'Bottleneck':<15}\n")
        f.write("-" * 80 + "\n")
        
        for setting_name, params in final_params.items():
            if params is None:
                f.write(f"{setting_name:<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}\n")
            else:
                f.write(f"{setting_name:<15} "
                       f"{params['n_timepoints']:<12} "
                       f"{params['cells_per_timepoint']:>11,} "
                       f"{params['total_cells']:>11,} "
                       f"{params['bottleneck_timepoint']:<15}\n")
    
    print(f"\n✓ Saved summary: {summary_file}")
    
    print("\n" + "="*80)
    print("✓ Analysis Complete!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Review the generated YAML snippets")
    print("2. Copy recommended parameters to your experiment configs")
    print("3. Ensure all settings use the same total_cells for fair comparison")


if __name__ == '__main__':
    main()
