#!/usr/bin/env python3
"""
Verify Experiment 4 configuration files.

This script checks that all ablation configs are correctly set up.
"""

import yaml
from pathlib import Path
from typing import Dict, List


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def verify_ablation_config(config: Dict, expected_removed: str, expected_setting_name: str) -> bool:
    """Verify a single ablation configuration."""
    
    errors = []
    
    # Check experiment metadata
    if 'experiment' not in config:
        errors.append("Missing 'experiment' section")
    else:
        exp = config['experiment']
        if 'ablation_target' not in exp:
            errors.append("Missing 'ablation_target' in experiment metadata")
        elif exp['ablation_target'] != expected_removed:
            errors.append(f"Expected ablation_target='{expected_removed}', got '{exp['ablation_target']}'")
    
    # Check data setting (should be a string reference)
    if 'data_setting' not in config:
        errors.append("Missing 'data_setting' section")
    else:
        data_setting = config['data_setting']
        
        if not isinstance(data_setting, str):
            errors.append(f"data_setting should be a string reference, got {type(data_setting)}")
        elif data_setting != expected_setting_name:
            errors.append(f"Expected data_setting='{expected_setting_name}', got '{data_setting}'")
    
    # Check data sampling override
    if 'data_sampling_override' not in config:
        errors.append("Missing 'data_sampling_override' section")
    else:
        override = config['data_sampling_override']
        if 'total_cells' not in override:
            errors.append("Missing 'total_cells' in data_sampling_override")
        elif override['total_cells'] != 8974:
            errors.append(f"Expected override total_cells=8974, got {override['total_cells']}")
    
    # Check evaluation endpoints
    if 'evaluation' not in config:
        errors.append("Missing 'evaluation' section")
    else:
        evaluation = config['evaluation']
        if 'start_timepoint' not in evaluation:
            errors.append("Missing 'start_timepoint' in evaluation")
        elif evaluation['start_timepoint'] != '0d':
            errors.append(f"Expected start_timepoint='0d', got '{evaluation['start_timepoint']}'")
        
        if 'end_timepoint' not in evaluation:
            errors.append("Missing 'end_timepoint' in evaluation")
        elif evaluation['end_timepoint'] != '7d':
            errors.append(f"Expected end_timepoint='7d', got '{evaluation['end_timepoint']}'")
    
    # Check models to train
    if 'models_to_train' not in config:
        errors.append("Missing 'models_to_train' section")
    else:
        models = config['models_to_train']
        if not isinstance(models, list) or len(models) != 1:
            errors.append("Expected exactly 1 model in models_to_train")
        elif models[0]['name'] != 'sb_mlplus':
            errors.append(f"Expected model 'sb_mlplus', got '{models[0]['name']}'")
    
    return errors


def main():
    print("=" * 80)
    print("Verifying Experiment 4 Configuration Files")
    print("=" * 80)
    print()
    
    configs_dir = Path('/home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData/configs')
    
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
            errors = verify_ablation_config(config, spec['removed'], spec['setting_name'])
            
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
        print("  1. Run the ablation experiments:")
        print("     bash step1_run_experiment_EMT.sh configs/experiment_EMT_Part1_setting4_ablation_remove_8h.yaml")
        print("     bash step1_run_experiment_EMT.sh configs/experiment_EMT_Part1_setting4_ablation_remove_1d.yaml")
        print("     bash step1_run_experiment_EMT.sh configs/experiment_EMT_Part1_setting4_ablation_remove_3d.yaml")
        print()
        print("  2. After all experiments complete, run the analysis:")
        print("     python Experiments/exp4_ablation/analyze_marginal_contribution.py")
    else:
        print("❌ Some configuration files have errors. Please fix them before proceeding.")
    print("=" * 80)


if __name__ == '__main__':
    main()
