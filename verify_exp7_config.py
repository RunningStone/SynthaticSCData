#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Experiment 7 Configuration

Checks that all configuration files are properly set up for Experiment 7.
"""

import yaml
from pathlib import Path
import sys


def check_data_config():
    """Check data configuration for setting7"""
    print("\n" + "="*70)
    print("Checking Data Configuration")
    print("="*70)
    
    data_config_path = Path("configs/data_EMT_Cook_with_label.yaml")
    
    if not data_config_path.exists():
        print(f"❌ ERROR: Data config not found: {data_config_path}")
        return False
    
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Check if setting7 exists
    if 'setting7' not in data_config:
        print("❌ ERROR: setting7 not found in data config")
        return False
    
    setting7 = data_config['setting7']
    
    print("✓ setting7 found in data config")
    print(f"  Name: {setting7['name']}")
    print(f"  Description: {setting7['description']}")
    print(f"  Time points: {setting7['time_points']}")
    print(f"  Total cells: {setting7['total_cells']}")
    print(f"  Balance strategy: {setting7['balance_strategy']}")
    
    # Check entropy analysis parameters
    if 'entropy_analysis' in setting7:
        entropy_params = setting7['entropy_analysis']
        print(f"\n  Entropy analysis parameters:")
        print(f"    Enabled: {entropy_params.get('enabled', False)}")
        print(f"    Method: {entropy_params.get('method', 'N/A')}")
        print(f"    K neighbors: {entropy_params.get('k_neighbors', 'N/A')}")
        print(f"    N samples: {entropy_params.get('n_samples_per_timepoint', 'N/A')}")
    
    return True


def check_experiment_config():
    """Check experiment configuration"""
    print("\n" + "="*70)
    print("Checking Experiment Configuration")
    print("="*70)
    
    exp_config_path = Path("configs/experiment_EMT_Part1_setting7_entropy.yaml")
    
    if not exp_config_path.exists():
        print(f"❌ ERROR: Experiment config not found: {exp_config_path}")
        return False
    
    with open(exp_config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    print("✓ Experiment config found")
    
    # Check required sections
    required_sections = ['experiment', 'configs', 'data_setting', 'models_to_train', 'settings']
    for section in required_sections:
        if section not in exp_config:
            print(f"❌ ERROR: Missing section '{section}'")
            return False
        print(f"  ✓ Section '{section}' present")
    
    # Check data_setting value
    data_setting = exp_config['data_setting']
    if data_setting != 'setting7':
        print(f"❌ ERROR: data_setting should be 'setting7', got '{data_setting}'")
        return False
    
    print(f"\n  Data setting: {data_setting}")
    
    # Check output directory
    output_dir = exp_config['settings']['output_dir']
    print(f"  Output directory: {output_dir}")
    
    if 'EMT_Part1_Setting7' not in output_dir:
        print(f"⚠️  WARNING: Output directory doesn't contain 'EMT_Part1_Setting7'")
        print(f"    Expected pattern: .../EMT_Part1_Setting7")
    else:
        print(f"  ✓ Output directory follows naming convention")
    
    # Check models to train
    print(f"\n  Models to train:")
    for model in exp_config['models_to_train']:
        status = "✓" if model.get('enabled', False) else "✗"
        print(f"    {status} {model['name']}")
    
    # Check experiment7 specific parameters
    if 'experiment7_params' in exp_config:
        print(f"\n  Experiment 7 specific parameters:")
        exp7_params = exp_config['experiment7_params']
        print(f"    Entropy method: {exp7_params.get('entropy_method', 'N/A')}")
        print(f"    K neighbors: {exp7_params.get('k_neighbors', 'N/A')}")
        print(f"    Time labels: {list(exp7_params.get('time_to_hours', {}).keys())}")
    
    return True


def check_entropy_module():
    """Check if entropy analysis module exists"""
    print("\n" + "="*70)
    print("Checking Entropy Analysis Module")
    print("="*70)
    
    exp7_dir = Path("Experiments/exp7_entropy")
    
    if not exp7_dir.exists():
        print(f"❌ ERROR: Experiment 7 directory not found: {exp7_dir}")
        return False
    
    print(f"✓ Experiment 7 directory found: {exp7_dir}")
    
    # Check required files
    required_files = [
        '__init__.py',
        'entropy_estimators.py',
        'analyze_entropy_evolution.py',
        'run_entropy_analysis.py',
        'run_exp7.sh',
        'README.md'
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = exp7_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ❌ {filename} (missing)")
            all_exist = False
    
    return all_exist


def check_run_script():
    """Check if run_experiment7.sh exists"""
    print("\n" + "="*70)
    print("Checking Run Script")
    print("="*70)
    
    run_script = Path("run_experiment7.sh")
    
    if not run_script.exists():
        print(f"❌ ERROR: Run script not found: {run_script}")
        return False
    
    print(f"✓ Run script found: {run_script}")
    
    # Check if executable
    if run_script.stat().st_mode & 0o111:
        print(f"  ✓ Script is executable")
    else:
        print(f"  ⚠️  Script is not executable (run: chmod +x {run_script})")
    
    return True


def main():
    """Run all checks"""
    print("="*70)
    print("EXPERIMENT 7 CONFIGURATION VERIFICATION")
    print("="*70)
    
    checks = [
        ("Data Configuration", check_data_config),
        ("Experiment Configuration", check_experiment_config),
        ("Entropy Analysis Module", check_entropy_module),
        ("Run Script", check_run_script)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n❌ ERROR in {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for check_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{check_name:30s} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    
    if all_passed:
        print("✓ All checks PASSED!")
        print("\nYou can now run Experiment 7:")
        print("  bash run_experiment7.sh")
        print("\nOr run individual components:")
        print("  1. Train models: python step1_run_experiment.py --config configs/experiment_EMT_Part1_setting7_entropy.yaml")
        print("  2. Analyze entropy: cd Experiments/exp7_entropy && bash run_exp7.sh")
        return 0
    else:
        print("❌ Some checks FAILED!")
        print("\nPlease fix the issues above before running Experiment 7.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
