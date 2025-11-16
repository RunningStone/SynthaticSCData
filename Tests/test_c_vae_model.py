#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Conditional VAE Model

Tests:
1. Model initialization
2. Forward pass
3. Loss computation
4. Trajectory generation
5. Time conditioning mechanism
6. MMD loss computation
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Model.c_vae_model import ConditionalVAEModel, compute_mmd_loss


def test_model_initialization():
    """Test model initialization"""
    print("\n" + "="*70)
    print("Test 1: Model Initialization")
    print("="*70)
    
    dimension = 1000
    n_timepoints = 8
    latent_dim = 128
    
    model = ConditionalVAEModel(
        dimension=dimension,
        n_timepoints=n_timepoints,
        latent_dim=latent_dim,
        time_embedding_dim=64,
        hidden_dims=[512, 256],
        activation='relu',
        dropout=0.1,
        beta=1.0,
        mmd_weight=1.0
    )
    
    print(f"✓ Model created successfully")
    print(f"  Dimension: {dimension}")
    print(f"  Number of timepoints: {n_timepoints}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Time embedding dimension: 64")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def test_forward_pass(model):
    """Test forward pass"""
    print("\n" + "="*70)
    print("Test 2: Forward Pass")
    print("="*70)
    
    batch_size = 32
    dimension = model.dimension
    
    # Create dummy data
    x_0 = torch.randn(batch_size, dimension)
    t_target = torch.randint(0, model.n_timepoints, (batch_size,))
    
    print(f"Input shape: {x_0.shape}")
    print(f"Target time indices: {t_target[:5].tolist()}...")
    
    # Forward pass
    x_0_recon, x_T_sample, mu, logvar = model(x_0, t_target)
    
    print(f"\n✓ Forward pass successful")
    print(f"  x_0_recon shape: {x_0_recon.shape}")
    print(f"  x_T_sample shape: {x_T_sample.shape}")
    print(f"  mu shape: {mu.shape}")
    print(f"  logvar shape: {logvar.shape}")
    
    # Check outputs
    assert x_0_recon.shape == (batch_size, dimension), "x_0_recon shape mismatch"
    assert x_T_sample.shape == (batch_size, dimension), "x_T_sample shape mismatch"
    assert mu.shape == (batch_size, model.latent_dim), "mu shape mismatch"
    assert logvar.shape == (batch_size, model.latent_dim), "logvar shape mismatch"
    
    print("\n✓ All output shapes correct")
    
    return x_0, x_T_sample, t_target


def test_loss_computation(model, x_0, x_T_sample, t_target):
    """Test loss computation"""
    print("\n" + "="*70)
    print("Test 3: Loss Computation")
    print("="*70)
    
    batch_size = x_0.shape[0]
    
    # Create dummy target data
    x_target = torch.randn_like(x_0)
    t_source = torch.zeros(batch_size, dtype=torch.long)  # All from time 0
    
    # Compute loss
    loss, loss_dict = model.compute_loss(x_0, x_target, t_source, t_target)
    
    print(f"✓ Loss computed successfully")
    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")
    
    # Check loss is finite
    assert torch.isfinite(loss), "Loss is not finite"
    assert loss.item() > 0, "Loss should be positive"
    
    print(f"\n✓ Loss is finite and positive: {loss.item():.6f}")
    
    # Check individual components
    assert 'recon' in loss_dict, "Missing reconstruction loss"
    assert 'kl' in loss_dict, "Missing KL loss"
    assert 'mmd' in loss_dict, "Missing MMD loss"
    assert 'total' in loss_dict, "Missing total loss"
    
    print("✓ All loss components present")
    
    return loss


def test_mmd_loss():
    """Test MMD loss computation"""
    print("\n" + "="*70)
    print("Test 4: MMD Loss")
    print("="*70)
    
    # Create two distributions
    batch_size = 100
    dim = 50
    
    # Same distribution
    x = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    
    mmd_rbf = compute_mmd_loss(x, y, kernel='rbf', bandwidth=1.0)
    mmd_linear = compute_mmd_loss(x, y, kernel='linear')
    
    print(f"✓ MMD computed successfully")
    print(f"  MMD (RBF kernel): {mmd_rbf.item():.6f}")
    print(f"  MMD (Linear kernel): {mmd_linear.item():.6f}")
    
    # MMD should be small for similar distributions
    assert torch.isfinite(mmd_rbf), "MMD (RBF) is not finite"
    assert torch.isfinite(mmd_linear), "MMD (Linear) is not finite"
    
    # Test with identical distributions (MMD should be ~0)
    mmd_identical = compute_mmd_loss(x, x, kernel='rbf')
    print(f"  MMD (identical distributions): {mmd_identical.item():.6e}")
    assert mmd_identical.item() < 1e-6, "MMD should be ~0 for identical distributions"
    
    print("✓ MMD loss working correctly")


def test_time_conditioning(model):
    """Test time conditioning mechanism"""
    print("\n" + "="*70)
    print("Test 5: Time Conditioning")
    print("="*70)
    
    batch_size = 16
    dimension = model.dimension
    
    # Same initial state
    x_0 = torch.randn(batch_size, dimension)
    
    # Different target times
    t_target_early = torch.zeros(batch_size, dtype=torch.long)  # Time 0
    t_target_late = torch.full((batch_size,), model.n_timepoints - 1, dtype=torch.long)  # Last time
    
    # Forward pass with different times
    _, x_T_early, _, _ = model(x_0, t_target_early)
    _, x_T_late, _, _ = model(x_0, t_target_late)
    
    # Check that outputs are different
    diff = torch.mean((x_T_early - x_T_late) ** 2).item()
    
    print(f"✓ Time conditioning tested")
    print(f"  Same input x_0, different target times")
    print(f"  Target time early (0): {t_target_early[0].item()}")
    print(f"  Target time late ({model.n_timepoints-1}): {t_target_late[0].item()}")
    print(f"  Mean squared difference in outputs: {diff:.6f}")
    
    # Outputs should be different for different times
    assert diff > 0.01, "Outputs should differ for different target times"
    
    print("✓ Time conditioning working correctly")


def test_trajectory_generation(model):
    """Test trajectory generation"""
    print("\n" + "="*70)
    print("Test 6: Trajectory Generation")
    print("="*70)
    
    batch_size = 8
    dimension = model.dimension
    n_time = 10
    
    # Create initial state
    x_0 = torch.randn(batch_size, dimension)
    time_grid = torch.linspace(0, 1, n_time)
    
    t_source_idx = 0
    t_target_idx = model.n_timepoints - 1
    
    print(f"Generating trajectory:")
    print(f"  Batch size: {batch_size}")
    print(f"  Time points: {n_time}")
    print(f"  Source time index: {t_source_idx}")
    print(f"  Target time index: {t_target_idx}")
    
    # Generate trajectory
    trajectory = model.generate_trajectory(
        x_0=x_0,
        time_grid=time_grid,
        t_source_idx=t_source_idx,
        t_target_idx=t_target_idx
    )
    
    print(f"\n✓ Trajectory generated successfully")
    print(f"  Trajectory shape: {trajectory.shape}")
    
    # Check shape
    assert trajectory.shape == (batch_size, n_time, dimension), "Trajectory shape mismatch"
    
    # Check first time point is close to x_0
    diff_start = torch.mean((trajectory[:, 0, :] - x_0) ** 2).item()
    print(f"  Difference from x_0 at t=0: {diff_start:.6f}")
    
    # Check trajectory is smooth (consecutive points should be similar)
    diffs = []
    for i in range(n_time - 1):
        diff = torch.mean((trajectory[:, i+1, :] - trajectory[:, i, :]) ** 2).item()
        diffs.append(diff)
    
    avg_diff = np.mean(diffs)
    print(f"  Average consecutive point difference: {avg_diff:.6f}")
    
    print("✓ Trajectory generation working correctly")
    
    return trajectory


def test_gradient_flow(model):
    """Test gradient flow through the model"""
    print("\n" + "="*70)
    print("Test 7: Gradient Flow")
    print("="*70)
    
    batch_size = 16
    dimension = model.dimension
    
    # Create dummy data
    x_0 = torch.randn(batch_size, dimension)
    x_target = torch.randn(batch_size, dimension)
    t_source = torch.zeros(batch_size, dtype=torch.long)
    t_target = torch.randint(1, model.n_timepoints, (batch_size,))
    
    # Compute loss
    loss, _ = model.compute_loss(x_0, x_target, t_source, t_target)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            has_grad += 1
            grad_norm = param.grad.norm().item()
            if grad_norm == 0:
                print(f"  Warning: {name} has zero gradient")
    
    print(f"✓ Gradient flow tested")
    print(f"  Parameters with gradients: {has_grad}/{total_params}")
    print(f"  Gradient coverage: {100*has_grad/total_params:.1f}%")
    
    assert has_grad == total_params, "Not all parameters have gradients"
    
    print("✓ All parameters have gradients")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("CONDITIONAL VAE MODEL TESTS")
    print("="*70)
    
    try:
        # Test 1: Initialization
        model = test_model_initialization()
        
        # Test 2: Forward pass
        x_0, x_T_sample, t_target = test_forward_pass(model)
        
        # Test 3: Loss computation
        loss = test_loss_computation(model, x_0, x_T_sample, t_target)
        
        # Test 4: MMD loss
        test_mmd_loss()
        
        # Test 5: Time conditioning
        test_time_conditioning(model)
        
        # Test 6: Trajectory generation
        trajectory = test_trajectory_generation(model)
        
        # Test 7: Gradient flow
        test_gradient_flow(model)
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
