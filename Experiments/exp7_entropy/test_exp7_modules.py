#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Tests for Experiment 7 Modules

Tests all components of the entropy evolution analysis pipeline:
1. Entropy estimators (KNN and Gaussian)
2. Entropy curve analysis functions
3. Integration with SB models

Run with:
    python test_exp7_modules.py
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from Experiments.exp7_entropy import (
    estimate_entropy_knn,
    estimate_entropy_gaussian,
    estimate_entropy_both_methods,
    analyze_entropy_peak,
    compute_entropy_curve_similarity
)


def test_entropy_estimators():
    """Test entropy estimators on known distributions."""
    print("\n" + "="*70)
    print("Test 1: Entropy Estimators on Known Distributions")
    print("="*70)
    
    # Test on standard Gaussian
    # True entropy: H = (d/2) * log(2πe) ≈ (d/2) * 2.838
    d = 50
    N = 1000
    X = np.random.randn(N, d)
    
    H_true = 0.5 * d * (1 + np.log(2 * np.pi))
    H_knn = estimate_entropy_knn(X, k=5)
    H_gauss = estimate_entropy_gaussian(X, shrinkage=True)
    
    print(f"\nStandard Gaussian (d={d}, N={N}):")
    print(f"  True entropy:     {H_true:.2f}")
    print(f"  KNN estimate:     {H_knn:.2f} (error: {abs(H_knn-H_true)/H_true*100:.1f}%)")
    print(f"  Gaussian estimate: {H_gauss:.2f} (error: {abs(H_gauss-H_true)/H_true*100:.1f}%)")
    
    # Check errors are reasonable (<20%)
    assert abs(H_knn - H_true) / H_true < 0.2, "KNN error too large"
    assert abs(H_gauss - H_true) / H_true < 0.2, "Gaussian error too large"
    
    print("\n✓ Entropy estimators passed!")
    return True


def test_both_methods():
    """Test cross-validation between methods."""
    print("\n" + "="*70)
    print("Test 2: Cross-Validation Between Methods")
    print("="*70)
    
    d = 100
    N = 1000
    X = np.random.randn(N, d)
    
    H_knn, H_gauss, H_avg = estimate_entropy_both_methods(X, k=5, shrinkage=True)
    
    print(f"\nStandard Gaussian (d={d}, N={N}):")
    print(f"  KNN:      {H_knn:.2f}")
    print(f"  Gaussian: {H_gauss:.2f}")
    print(f"  Average:  {H_avg:.2f}")
    print(f"  Relative difference: {abs(H_knn - H_gauss) / H_avg * 100:.1f}%")
    
    # Methods should roughly agree for Gaussian data
    assert abs(H_knn - H_gauss) / H_avg < 0.3, "Methods disagree too much"
    
    print("\n✓ Cross-validation passed!")
    return True


def test_peak_analysis():
    """Test entropy peak detection and analysis."""
    print("\n" + "="*70)
    print("Test 3: Entropy Peak Analysis")
    print("="*70)
    
    time_labels = ['t0', 't1', 't2', 't3', 't4']
    
    # Test case 1: Non-monotonic (inverted-U)
    print("\nTest Case 1: Non-monotonic entropy curve")
    entropy_nonmono = np.array([5.0, 7.0, 9.0, 7.5, 5.5])
    
    analysis = analyze_entropy_peak(entropy_nonmono, time_labels)
    
    print(f"  Entropy: {entropy_nonmono}")
    print(f"  Peak at: {analysis['peak_time']} (expected: t2)")
    print(f"  Peak value: {analysis['peak_value']:.2f} (expected: 9.0)")
    print(f"  Non-monotonic: {analysis['is_nonmonotonic']} (expected: True)")
    
    assert analysis['peak_time'] == 't2', "Peak position incorrect"
    assert analysis['peak_value'] == 9.0, "Peak value incorrect"
    assert analysis['is_nonmonotonic'] == True, "Should be non-monotonic"
    
    # Test case 2: Monotonic
    print("\nTest Case 2: Monotonic entropy curve")
    entropy_mono = np.array([5.0, 5.5, 6.0, 6.5, 7.0])
    
    analysis_mono = analyze_entropy_peak(entropy_mono, time_labels)
    
    print(f"  Entropy: {entropy_mono}")
    print(f"  Non-monotonic: {analysis_mono['is_nonmonotonic']} (expected: False)")
    
    assert analysis_mono['is_nonmonotonic'] == False, "Should be monotonic"
    
    print("\n✓ Peak analysis passed!")
    return True


def test_curve_similarity():
    """Test entropy curve similarity metrics."""
    print("\n" + "="*70)
    print("Test 4: Curve Similarity")
    print("="*70)
    
    real_curve = np.array([5.0, 7.0, 9.0, 7.5, 5.5])
    
    # Good match
    gen_curve_good = np.array([5.1, 6.9, 8.8, 7.6, 5.4])
    
    # Bad match (monotonic)
    gen_curve_bad = np.array([5.0, 5.5, 6.0, 6.5, 7.0])
    
    mse_good = compute_entropy_curve_similarity(real_curve, gen_curve_good, metric='mse')
    mse_bad = compute_entropy_curve_similarity(real_curve, gen_curve_bad, metric='mse')
    
    print(f"\nReal curve: {real_curve}")
    print(f"Good match: {gen_curve_good}")
    print(f"  MSE: {mse_good:.4f}")
    print(f"Bad match:  {gen_curve_bad}")
    print(f"  MSE: {mse_bad:.4f}")
    
    assert mse_good < mse_bad, "Good match should have lower MSE"
    assert mse_good < 0.1, "Good match MSE should be small"
    assert mse_bad > 1.0, "Bad match MSE should be large"
    
    print("\n✓ Curve similarity passed!")
    return True


def test_torch_integration():
    """Test that estimators work with PyTorch tensors."""
    print("\n" + "="*70)
    print("Test 5: PyTorch Tensor Integration")
    print("="*70)
    
    d = 50
    N = 500
    
    # Create PyTorch tensor
    X_torch = torch.randn(N, d)
    
    # Should automatically convert to numpy
    H_knn = estimate_entropy_knn(X_torch, k=5)
    H_gauss = estimate_entropy_gaussian(X_torch, shrinkage=True)
    
    print(f"\nPyTorch tensor (d={d}, N={N}):")
    print(f"  KNN estimate:      {H_knn:.2f}")
    print(f"  Gaussian estimate: {H_gauss:.2f}")
    
    assert isinstance(H_knn, float), "KNN should return float"
    assert isinstance(H_gauss, float), "Gaussian should return float"
    assert not np.isnan(H_knn), "KNN should not be NaN"
    assert not np.isnan(H_gauss), "Gaussian should not be NaN"
    
    print("\n✓ PyTorch integration passed!")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*70)
    print("Test 6: Edge Cases and Error Handling")
    print("="*70)
    
    # Test with small sample size
    print("\nTest: Small sample size (N=20, k=5)")
    X_small = np.random.randn(20, 10)
    
    try:
        H = estimate_entropy_knn(X_small, k=5)
        print(f"  KNN estimate: {H:.2f}")
        print("  ✓ Handled small sample size")
    except ValueError as e:
        print(f"  ✓ Correctly raised error: {e}")
    
    # Test with high-dimensional data
    print("\nTest: High-dimensional data (d=500, N=100)")
    X_highdim = np.random.randn(100, 500)
    
    try:
        H = estimate_entropy_gaussian(X_highdim, shrinkage=True)
        print(f"  Gaussian estimate: {H:.2f}")
        print("  ✓ Handled high dimensions with shrinkage")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    print("\n✓ Edge cases passed!")
    return True


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*70)
    print("Experiment 7: Unit Tests")
    print("="*70)
    
    tests = [
        ("Entropy Estimators", test_entropy_estimators),
        ("Cross-Validation", test_both_methods),
        ("Peak Analysis", test_peak_analysis),
        ("Curve Similarity", test_curve_similarity),
        ("PyTorch Integration", test_torch_integration),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
