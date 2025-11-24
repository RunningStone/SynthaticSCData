#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Experiment 6 Implementation
Quick checks to ensure all components are working correctly
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from generate_interpolated_data import generate_linear_interpolated_data
        print("  ✓ generate_interpolated_data imported")
    except Exception as e:
        print(f"  ✗ Failed to import generate_interpolated_data: {e}")
        return False
    
    try:
        from analyze_interpolation_quality import (
            compute_interpolation_effectiveness_index,
            compute_residual_structure_index,
            compute_per_timepoint_metrics
        )
        print("  ✓ analyze_interpolation_quality imported")
    except Exception as e:
        print(f"  ✗ Failed to import analyze_interpolation_quality: {e}")
        return False
    
    return True


def test_interpolation_logic():
    """Test interpolation weight calculation"""
    print("\nTesting interpolation logic...")
    
    # Time mapping
    time_to_hours = {
        "0d": 0.0,
        "8h": 8.0,
        "1d": 24.0,
        "3d": 72.0,
        "7d": 168.0
    }
    
    t0, tn = time_to_hours["0d"], time_to_hours["7d"]
    
    # Test weights
    expected_weights = {
        "8h": 0.952,
        "1d": 0.857,
        "3d": 0.571
    }
    
    all_correct = True
    for tp, expected in expected_weights.items():
        tk = time_to_hours[tp]
        lambda_k = (tn - tk) / (tn - t0)
        
        if abs(lambda_k - expected) < 0.001:
            print(f"  ✓ {tp}: λ={lambda_k:.3f} (expected {expected:.3f})")
        else:
            print(f"  ✗ {tp}: λ={lambda_k:.3f} (expected {expected:.3f})")
            all_correct = False
    
    return all_correct


def test_interpolation_function():
    """Test interpolation with synthetic data"""
    print("\nTesting interpolation function...")
    
    # Create synthetic data
    n_cells_t0 = 100
    n_cells_tn = 100
    n_genes = 50
    
    X_t0 = np.random.randn(n_cells_t0, n_genes)
    X_tn = np.random.randn(n_cells_tn, n_genes)
    
    # Test interpolation
    lambda_k = 0.5  # Midpoint
    n_samples = 10
    
    idx_t0 = np.random.choice(n_cells_t0, n_samples, replace=True)
    idx_tn = np.random.choice(n_cells_tn, n_samples, replace=True)
    
    X_interp = lambda_k * X_t0[idx_t0] + (1 - lambda_k) * X_tn[idx_tn]
    
    # Verify shape
    if X_interp.shape == (n_samples, n_genes):
        print(f"  ✓ Interpolated data shape: {X_interp.shape}")
    else:
        print(f"  ✗ Unexpected shape: {X_interp.shape}")
        return False
    
    # Verify values are between t0 and tn ranges
    min_val = min(X_t0.min(), X_tn.min())
    max_val = max(X_t0.max(), X_tn.max())
    
    if X_interp.min() >= min_val - 1e-6 and X_interp.max() <= max_val + 1e-6:
        print(f"  ✓ Interpolated values in valid range")
    else:
        print(f"  ✗ Interpolated values out of range")
        return False
    
    return True


def test_iei_calculation():
    """Test IEI calculation"""
    print("\nTesting IEI calculation...")
    
    from analyze_interpolation_quality import compute_interpolation_effectiveness_index
    
    # Create synthetic data
    real = np.random.randn(100, 50)
    gen_setting1 = real + np.random.randn(100, 50) * 2.0  # High error
    gen_interp = real + np.random.randn(100, 50) * 0.5    # Low error
    
    iei = compute_interpolation_effectiveness_index(real, gen_setting1, gen_interp)
    
    # IEI should be positive (interpolation better than setting1)
    if iei > 0:
        print(f"  ✓ IEI = {iei:.4f} (positive, as expected)")
    else:
        print(f"  ✗ IEI = {iei:.4f} (should be positive)")
        return False
    
    # Test edge case: perfect interpolation
    gen_perfect = real.copy()
    iei_perfect = compute_interpolation_effectiveness_index(real, gen_setting1, gen_perfect)
    
    if iei_perfect > 0.9:
        print(f"  ✓ Perfect interpolation IEI = {iei_perfect:.4f} (close to 1.0)")
    else:
        print(f"  ✗ Perfect interpolation IEI = {iei_perfect:.4f} (should be close to 1.0)")
        return False
    
    return True


def test_rsi_calculation():
    """Test RSI calculation"""
    print("\nTesting RSI calculation...")
    
    from analyze_interpolation_quality import compute_residual_structure_index
    
    # Create synthetic data with structured residuals
    real = np.random.randn(100, 50)
    
    # Add low-rank structure to interpolated data
    U = np.random.randn(100, 5)
    V = np.random.randn(5, 50)
    structured_error = U @ V
    
    interp_structured = real + structured_error + np.random.randn(100, 50) * 0.1
    
    rsi_results = compute_residual_structure_index(real, interp_structured, n_components=10)
    
    rsi = rsi_results['rsi']
    
    # RSI should be high (structured residuals)
    if rsi > 0.5:
        print(f"  ✓ RSI = {rsi:.4f} (high, as expected for structured residuals)")
    else:
        print(f"  ✗ RSI = {rsi:.4f} (should be high for structured residuals)")
        return False
    
    # Test with random residuals
    interp_random = real + np.random.randn(100, 50) * 0.5
    rsi_results_random = compute_residual_structure_index(real, interp_random, n_components=10)
    rsi_random = rsi_results_random['rsi']
    
    if rsi_random < rsi:
        print(f"  ✓ Random residual RSI = {rsi_random:.4f} (lower than structured)")
    else:
        print(f"  ✗ Random residual RSI = {rsi_random:.4f} (should be lower than structured)")
        return False
    
    return True


def test_config_file():
    """Test that config file exists and is valid"""
    print("\nTesting config file...")
    
    config_path = project_root / "configs" / "experiment_EMT_Part1_setting6_interpolated.yaml"
    
    if config_path.exists():
        print(f"  ✓ Config file exists: {config_path}")
    else:
        print(f"  ✗ Config file not found: {config_path}")
        return False
    
    # Try to load config
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check key sections
        required_sections = ['experiment', 'configs', 'data_setting', 'models_to_train', 'settings']
        for section in required_sections:
            if section in config:
                print(f"  ✓ Config section '{section}' present")
            else:
                print(f"  ✗ Config section '{section}' missing")
                return False
        
    except Exception as e:
        print(f"  ✗ Failed to load config: {e}")
        return False
    
    return True


def test_scripts_executable():
    """Test that scripts are executable"""
    print("\nTesting script permissions...")
    
    scripts = [
        "run_experiment6.sh",
        "run_experiment6.py",
        "generate_interpolated_data.py"
    ]
    
    exp_dir = Path(__file__).parent
    
    all_executable = True
    for script in scripts:
        script_path = exp_dir / script
        if script_path.exists():
            import os
            if os.access(script_path, os.X_OK):
                print(f"  ✓ {script} is executable")
            else:
                print(f"  ✗ {script} is not executable")
                all_executable = False
        else:
            print(f"  ✗ {script} not found")
            all_executable = False
    
    return all_executable


def main():
    """Run all verification tests"""
    print("="*70)
    print("EXPERIMENT 6: IMPLEMENTATION VERIFICATION")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Interpolation Logic", test_interpolation_logic),
        ("Interpolation Function", test_interpolation_function),
        ("IEI Calculation", test_iei_calculation),
        ("RSI Calculation", test_rsi_calculation),
        ("Config File", test_config_file),
        ("Script Permissions", test_scripts_executable)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Implementation is ready.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please fix before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
