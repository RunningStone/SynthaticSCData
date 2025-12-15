#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Pre-calculation Results Reset to Config Files
从precalc_results中读取计算好的数据分割参数，更新到对应的YAML配置文件中。

Usage:
    python step1_precalc_reset.py \
        --output_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/GSE234181 \
        --config_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData/configs/GSE234181

Author: Auto-generated
Date: 2024-11-27
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml
import re
from copy import deepcopy


def load_precalc_results(output_dir: Path) -> Dict[str, Any]:
    """
    Load pre-calculation results from the output directory.
    
    Args:
        output_dir: Path to the experiment output directory
        
    Returns:
        Dictionary containing the pre-calculated parameters
    """
    precalc_dir = output_dir / "precalc_results"
    
    if not precalc_dir.exists():
        raise FileNotFoundError(f"precalc_results directory not found in {output_dir}")
    
    # Load the final parameters JSON
    final_params_file = precalc_dir / "data_split_analysis_final_params.json"
    
    if not final_params_file.exists():
        raise FileNotFoundError(f"data_split_analysis_final_params.json not found in {precalc_dir}")
    
    with open(final_params_file, 'r') as f:
        results = json.load(f)
    
    print(f"✓ Loaded pre-calculation results from: {final_params_file}")
    print(f"  Found {len(results)} settings: {list(results.keys())}")
    
    return results


def find_data_config(config_dir: Path) -> Optional[Path]:
    """
    Find the data configuration YAML file in the config directory.
    
    Args:
        config_dir: Path to the config directory
        
    Returns:
        Path to the data config file, or None if not found
    """
    # Look for data_*.yaml files
    data_configs = list(config_dir.glob("data_*.yaml"))
    
    if not data_configs:
        return None
    
    if len(data_configs) > 1:
        print(f"⚠️  Warning: Multiple data config files found: {[f.name for f in data_configs]}")
        print(f"   Using the first one: {data_configs[0].name}")
    
    return data_configs[0]


def find_experiment_configs(config_dir: Path) -> Dict[str, Path]:
    """
    Find all experiment configuration YAML files in the config directory.
    
    Args:
        config_dir: Path to the config directory
        
    Returns:
        Dictionary mapping setting names to experiment config file paths
    """
    experiment_configs = {}
    
    # Look for experiment_*.yaml files
    for exp_file in config_dir.glob("experiment_*.yaml"):
        # Extract setting name from filename
        # e.g., experiment_GSE234181_setting1.yaml -> setting1
        # e.g., experiment_EMT_E2M2E_setting4_ablation_remove_8h.yaml -> setting4_ablation_remove_8h
        
        filename = exp_file.stem  # Remove .yaml extension
        
        # Find the setting part in the filename
        # Pattern: experiment_{project}_{setting}.yaml
        parts = filename.split('_')
        
        # Find where 'setting' starts
        setting_start_idx = None
        for i, part in enumerate(parts):
            if part.startswith('setting'):
                setting_start_idx = i
                break
        
        if setting_start_idx is not None:
            setting_name = '_'.join(parts[setting_start_idx:])
            experiment_configs[setting_name] = exp_file
    
    return experiment_configs


def load_yaml_with_comments(file_path: Path) -> Tuple[Dict[str, Any], str]:
    """
    Load YAML file and preserve the original content for comment preservation.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Tuple of (parsed dict, original content string)
    """
    with open(file_path, 'r') as f:
        original_content = f.read()
    
    data = yaml.safe_load(original_content)
    return data, original_content


def update_yaml_value(content: str, key_path: str, new_value: Any, 
                      parent_key: Optional[str] = None) -> str:
    """
    Update a specific value in YAML content while preserving formatting.
    
    Args:
        content: Original YAML content string
        key_path: Key to update (e.g., 'cells_per_timepoint')
        new_value: New value to set
        parent_key: Optional parent key to scope the search (e.g., 'setting1')
        
    Returns:
        Updated YAML content string
    """
    lines = content.split('\n')
    updated_lines = []
    
    in_parent_section = parent_key is None
    parent_indent = -1
    
    for i, line in enumerate(lines):
        # Check if we're entering/leaving the parent section
        if parent_key is not None:
            # Check if this line starts the parent section
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
            
            if stripped.startswith(f"{parent_key}:"):
                in_parent_section = True
                parent_indent = current_indent
                updated_lines.append(line)
                continue
            
            # Check if we've left the parent section (same or lower indent level)
            if in_parent_section and parent_indent >= 0 and stripped and not stripped.startswith('#'):
                if current_indent <= parent_indent and not stripped.startswith(f"{parent_key}"):
                    in_parent_section = False
        
        # Update the value if we're in the right section
        if in_parent_section:
            # Match the key with various value formats
            # Pattern: key: value or key: null
            pattern = rf'^(\s*){key_path}:\s*(.*)$'
            match = re.match(pattern, line)
            
            if match:
                indent = match.group(1)
                # Format the new value appropriately
                if new_value is None:
                    formatted_value = "null"
                elif isinstance(new_value, bool):
                    formatted_value = str(new_value).lower()
                elif isinstance(new_value, (int, float)):
                    formatted_value = str(new_value)
                elif isinstance(new_value, str):
                    formatted_value = f'"{new_value}"' if ' ' in new_value else new_value
                else:
                    formatted_value = str(new_value)
                
                updated_lines.append(f"{indent}{key_path}: {formatted_value}")
                continue
        
        updated_lines.append(line)
    
    return '\n'.join(updated_lines)


def update_data_config(data_config_path: Path, precalc_results: Dict[str, Any], 
                       dry_run: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Update the data configuration YAML file with pre-calculated parameters.
    
    Args:
        data_config_path: Path to the data config YAML file
        precalc_results: Pre-calculated parameters dictionary
        dry_run: If True, don't write changes, just report what would be changed
        
    Returns:
        Dictionary of changes made
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Updating data config: {data_config_path.name}")
    
    data, original_content = load_yaml_with_comments(data_config_path)
    updated_content = original_content
    changes = {}
    
    for setting_name, params in precalc_results.items():
        if setting_name not in data:
            print(f"  ⚠️  Setting '{setting_name}' not found in data config, skipping")
            continue
        
        setting_changes = {}
        current_setting = data[setting_name]
        
        # Update cells_per_timepoint
        new_cells_per_timepoint = params.get('cells_per_timepoint')
        old_cells_per_timepoint = current_setting.get('cells_per_timepoint')
        
        if new_cells_per_timepoint is not None:
            if old_cells_per_timepoint != new_cells_per_timepoint:
                setting_changes['cells_per_timepoint'] = {
                    'old': old_cells_per_timepoint,
                    'new': new_cells_per_timepoint
                }
                updated_content = update_yaml_value(
                    updated_content, 'cells_per_timepoint', 
                    new_cells_per_timepoint, setting_name
                )
        
        # Update total_cells if present
        new_total_cells = params.get('actual_total_cells') or params.get('target_total_cells')
        old_total_cells = current_setting.get('total_cells')
        
        if 'total_cells' in current_setting and new_total_cells is not None:
            if old_total_cells != new_total_cells:
                setting_changes['total_cells'] = {
                    'old': old_total_cells,
                    'new': new_total_cells
                }
                updated_content = update_yaml_value(
                    updated_content, 'total_cells', 
                    new_total_cells, setting_name
                )
        
        if setting_changes:
            changes[setting_name] = setting_changes
            print(f"  ✓ {setting_name}:")
            for key, vals in setting_changes.items():
                print(f"      {key}: {vals['old']} → {vals['new']}")
    
    if not changes:
        print("  ℹ️  No changes needed for data config")
    elif not dry_run:
        with open(data_config_path, 'w') as f:
            f.write(updated_content)
        print(f"  ✓ Saved changes to {data_config_path.name}")
    
    return changes


def update_experiment_configs(config_dir: Path, precalc_results: Dict[str, Any],
                              dry_run: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Update experiment configuration YAML files with pre-calculated parameters.
    
    Args:
        config_dir: Path to the config directory
        precalc_results: Pre-calculated parameters dictionary
        dry_run: If True, don't write changes, just report what would be changed
        
    Returns:
        Dictionary of changes made per file
    """
    experiment_configs = find_experiment_configs(config_dir)
    
    if not experiment_configs:
        print("\n⚠️  No experiment config files found")
        return {}
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Found {len(experiment_configs)} experiment configs")
    
    all_changes = {}
    
    for setting_name, exp_config_path in experiment_configs.items():
        if setting_name not in precalc_results:
            print(f"  ⚠️  No precalc results for '{setting_name}', skipping {exp_config_path.name}")
            continue
        
        params = precalc_results[setting_name]
        exp_data, original_content = load_yaml_with_comments(exp_config_path)
        updated_content = original_content
        changes = {}
        
        # Update data_sampling_override.cells_per_timepoint
        new_cells_per_timepoint = params.get('cells_per_timepoint')
        
        if new_cells_per_timepoint is not None:
            # Check current value
            data_sampling = exp_data.get('data_sampling_override', {})
            old_cells_per_timepoint = data_sampling.get('cells_per_timepoint')
            
            if old_cells_per_timepoint != new_cells_per_timepoint:
                changes['data_sampling_override.cells_per_timepoint'] = {
                    'old': old_cells_per_timepoint,
                    'new': new_cells_per_timepoint
                }
                updated_content = update_yaml_value(
                    updated_content, 'cells_per_timepoint',
                    new_cells_per_timepoint, 'data_sampling_override'
                )
        
        if changes:
            all_changes[exp_config_path.name] = changes
            print(f"\n  ✓ {exp_config_path.name}:")
            for key, vals in changes.items():
                print(f"      {key}: {vals['old']} → {vals['new']}")
            
            if not dry_run:
                with open(exp_config_path, 'w') as f:
                    f.write(updated_content)
                print(f"      Saved changes")
        else:
            print(f"\n  ℹ️  {exp_config_path.name}: No changes needed")
    
    return all_changes


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Update YAML config files with pre-calculated data split parameters"
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to the experiment output directory containing precalc_results/'
    )
    
    parser.add_argument(
        '--config_dir',
        type=str,
        required=True,
        help='Path to the config directory containing data_*.yaml and experiment_*.yaml files'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be changed without actually modifying files'
    )
    
    parser.add_argument(
        '--skip_data_config',
        action='store_true',
        help='Skip updating the data configuration file'
    )
    
    parser.add_argument(
        '--skip_experiment_configs',
        action='store_true',
        help='Skip updating experiment configuration files'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    config_dir = Path(args.config_dir)
    
    # Validate paths
    if not output_dir.exists():
        print(f"❌ Error: Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    if not config_dir.exists():
        print(f"❌ Error: Config directory does not exist: {config_dir}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("STEP 1: PRE-CALCULATION RESULTS RESET TO CONFIG FILES")
    print("="*80)
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be modified")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Config directory: {config_dir}")
    
    # Load pre-calculation results
    try:
        precalc_results = load_precalc_results(output_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    all_changes = {}
    
    # Update data config
    if not args.skip_data_config:
        data_config = find_data_config(config_dir)
        if data_config:
            data_changes = update_data_config(data_config, precalc_results, args.dry_run)
            if data_changes:
                all_changes['data_config'] = data_changes
        else:
            print("\n⚠️  No data config file found (data_*.yaml)")
    
    # Update experiment configs
    if not args.skip_experiment_configs:
        exp_changes = update_experiment_configs(config_dir, precalc_results, args.dry_run)
        if exp_changes:
            all_changes['experiment_configs'] = exp_changes
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_changes:
        total_settings_updated = len(all_changes.get('data_config', {}))
        total_exp_files_updated = len(all_changes.get('experiment_configs', {}))
        
        print(f"\n✓ Data config: {total_settings_updated} settings updated")
        print(f"✓ Experiment configs: {total_exp_files_updated} files updated")
        
        if args.dry_run:
            print("\n🔍 This was a DRY RUN - no files were actually modified")
            print("   Run without --dry_run to apply changes")
    else:
        print("\nℹ️  No changes were needed - all configs are up to date")
    
    print("\n" + "="*80)
    print("✓ DONE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
