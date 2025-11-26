#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Workers utility functions

This module contains helper functions for experiment workflows, including:
- Configuration verification for Setting4 ablation experiments
- Other utility functions for experiment management
"""

import yaml
from pathlib import Path
from typing import Dict, List


def load_config(config_path: Path) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def verify_setting4_ablation_config(
    config: Dict,
    expected_removed: str,
    expected_setting_name: str
) -> List[str]:
    """
    Verify a Setting4 ablation configuration file.
    
    This function checks that an ablation configuration is correctly set up
    for the Setting4 timepoint ablation study.
    
    Args:
        config: Configuration dictionary loaded from YAML
        expected_removed: Expected timepoint to be removed (e.g., '8h', '1d', '3d')
        expected_setting_name: Expected data_setting name (e.g., 'setting4_ablation_remove_8h')
    
    Returns:
        List of error messages (empty list if no errors)
    
    Example:
        >>> config = load_config('experiment_EMT_Part1_setting4_ablation_remove_8h.yaml')
        >>> errors = verify_setting4_ablation_config(config, '8h', 'setting4_ablation_remove_8h')
        >>> if not errors:
        ...     print("Configuration is valid!")
    """
    errors = []
    
    # Check experiment metadata
    if 'experiment' not in config:
        errors.append("Missing 'experiment' section")
    else:
        exp = config['experiment']
        if 'ablation_target' not in exp:
            errors.append("Missing 'ablation_target' in experiment metadata")
        elif exp['ablation_target'] != expected_removed:
            errors.append(
                f"Expected ablation_target='{expected_removed}', "
                f"got '{exp['ablation_target']}'"
            )
    
    # Check data setting (should be a string reference)
    if 'data_setting' not in config:
        errors.append("Missing 'data_setting' section")
    else:
        data_setting = config['data_setting']
        
        if not isinstance(data_setting, str):
            errors.append(
                f"data_setting should be a string reference, "
                f"got {type(data_setting)}"
            )
        elif data_setting != expected_setting_name:
            errors.append(
                f"Expected data_setting='{expected_setting_name}', "
                f"got '{data_setting}'"
            )
    
    # Check data sampling override
    if 'data_sampling_override' not in config:
        errors.append("Missing 'data_sampling_override' section")
    else:
        override = config['data_sampling_override']
        if 'total_cells' not in override:
            errors.append("Missing 'total_cells' in data_sampling_override")
        elif override['total_cells'] != 8974:
            errors.append(
                f"Expected override total_cells=8974, "
                f"got {override['total_cells']}"
            )
    
    # Check evaluation endpoints
    if 'evaluation' not in config:
        errors.append("Missing 'evaluation' section")
    else:
        evaluation = config['evaluation']
        if 'start_timepoint' not in evaluation:
            errors.append("Missing 'start_timepoint' in evaluation")
        elif evaluation['start_timepoint'] != '0d':
            errors.append(
                f"Expected start_timepoint='0d', "
                f"got '{evaluation['start_timepoint']}'"
            )
        
        if 'end_timepoint' not in evaluation:
            errors.append("Missing 'end_timepoint' in evaluation")
        elif evaluation['end_timepoint'] != '7d':
            errors.append(
                f"Expected end_timepoint='7d', "
                f"got '{evaluation['end_timepoint']}'"
            )
    
    # Check models to train
    if 'models_to_train' not in config:
        errors.append("Missing 'models_to_train' section")
    else:
        models = config['models_to_train']
        if not isinstance(models, list) or len(models) != 1:
            errors.append("Expected exactly 1 model in models_to_train")
        elif models[0]['name'] != 'sb_mlplus':
            errors.append(
                f"Expected model 'sb_mlplus', "
                f"got '{models[0]['name']}'"
            )
    
    return errors


def verify_all_setting4_ablation_configs(configs_dir: Path) -> bool:
    """
    Verify all Setting4 ablation configuration files.
    
    Args:
        configs_dir: Directory containing configuration files
    
    Returns:
        True if all configurations are valid, False otherwise
    
    Example:
        >>> configs_dir = Path('configs/EMT_E2M')
        >>> all_valid = verify_all_setting4_ablation_configs(configs_dir)
        >>> if all_valid:
        ...     print("All configurations are valid!")
    """
    print("=" * 80)
    print("Verifying Setting4 Ablation Configuration Files")
    print("=" * 80)
    print()
    
    # Define expected configurations
    ablation_specs = [
        {
            'file': 'experiment_EMT_Part1_setting4_ablation_remove_8h.yaml',
            'removed': '8h',
            'setting_name': 'setting4_ablation_remove_8h'
        },
        {
            'file': 'experiment_EMT_Part1_setting4_ablation_remove_1d.yaml',
            'removed': '1d',
            'setting_name': 'setting4_ablation_remove_1d'
        },
        {
            'file': 'experiment_EMT_Part1_setting4_ablation_remove_3d.yaml',
            'removed': '3d',
            'setting_name': 'setting4_ablation_remove_3d'
        }
    ]
    
    all_passed = True
    
    for spec in ablation_specs:
        config_path = configs_dir / spec['file']
        print(f"Checking: {spec['file']}")
        print(f"  Removed timepoint: {spec['removed']}")
        print(f"  Expected data_setting: {spec['setting_name']}")
        
        if not config_path.exists():
            print(f"  ❌ ERROR: File not found!")
            all_passed = False
            print()
            continue
        
        try:
            config = load_config(config_path)
            errors = verify_setting4_ablation_config(
                config,
                spec['removed'],
                spec['setting_name']
            )
            
            if errors:
                print(f"  ❌ FAILED with {len(errors)} error(s):")
                for error in errors:
                    print(f"     - {error}")
                all_passed = False
            else:
                print(f"  ✅ PASSED")
        
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            all_passed = False
        
        print()
    
    print("=" * 80)
    if all_passed:
        print("✅ All configuration files are valid!")
        print()
        print("Next steps:")
        print("  1. Run the ablation experiments using:")
        print("     bash EXPs/EMTE2M/step2_run_ablations_setting4.sh")
        print()
        print("  2. After all experiments complete, run the analysis:")
        print("     bash EXPs/EMTE2M/step3_analyze_ablation.sh")
    else:
        print("❌ Some configuration files have errors. Please fix them before proceeding.")
    print("=" * 80)
    
    return all_passed
