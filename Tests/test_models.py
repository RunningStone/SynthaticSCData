"""
Tests for Model Module
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Model import OptimalTransportModel, SchrodingerBridgeModel, VAEModel


class TestOptimalTransportModel:
    """Test OT model"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        dim = 50
        model = OptimalTransportModel(
            dimension=dim,
            hidden_dims=[128, 128],
            activation='relu',
            dropout=0.1
        )
        
        assert model.dimension == dim
    
    def test_forward_pass(self):
        """Test forward pass"""
        dim = 50
        batch_size = 32
        
        model = OptimalTransportModel(dimension=dim, hidden_dims=[128, 128])
        x_0 = torch.randn(batch_size, dim)
        x_T = model(x_0)
        
        assert x_T.shape == (batch_size, dim)
    
    def test_interpolation(self):
        """Test trajectory interpolation"""
        dim = 50
        batch_size = 32
        n_time = 10
        
        model = OptimalTransportModel(dimension=dim, hidden_dims=[128, 128])
        x_0 = torch.randn(batch_size, dim)
        time_grid = torch.linspace(0, 1, n_time)
        
        trajectory = model.generate_trajectory(x_0, time_grid)
        
        assert trajectory.shape == (batch_size, n_time, dim)
        assert torch.allclose(trajectory[:, 0, :], x_0, atol=1e-5)
    
    def test_loss_computation(self):
        """Test loss computation"""
        dim = 50
        batch_size = 32
        
        model = OptimalTransportModel(dimension=dim, hidden_dims=[128, 128])
        x_0 = torch.randn(batch_size, dim)
        x_T = torch.randn(batch_size, dim)
        
        loss = model.compute_loss(x_0, x_T)
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0


class TestSchrodingerBridgeModel:
    """Test SB model"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        dim = 50
        model = SchrodingerBridgeModel(
            dimension=dim,
            hidden_dims=[256, 256],
            time_embedding_dim=64,
            diffusion_coeff=0.1
        )
        
        assert model.dimension == dim
        assert model.D == 0.1
    
    def test_drift_computation(self):
        """Test drift field computation"""
        dim = 50
        batch_size = 32
        
        model = SchrodingerBridgeModel(dimension=dim, hidden_dims=[256, 256])
        x = torch.randn(batch_size, dim)
        t = torch.tensor(0.5)
        
        drift = model.compute_drift(x, t)
        
        assert drift.shape == (batch_size, dim)
    
    def test_trajectory_generation(self):
        """Test trajectory generation"""
        dim = 50
        batch_size = 32
        n_time = 10
        
        model = SchrodingerBridgeModel(dimension=dim, hidden_dims=[256, 256])
        x_0 = torch.randn(batch_size, dim)
        time_grid = torch.linspace(0, 1, n_time)
        
        trajectory = model.generate_trajectory(x_0, time_grid, method='deterministic')
        
        assert trajectory.shape == (batch_size, n_time, dim)
        assert torch.allclose(trajectory[:, 0, :], x_0, atol=1e-5)


class TestVAEModel:
    """Test VAE model"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        dim = 50
        latent_dim = 32
        
        model = VAEModel(
            dimension=dim,
            encoder_dims=[128, 64],
            latent_dim=latent_dim,
            decoder_dims=[64, 128]
        )
        
        assert model.dimension == dim
        assert model.latent_dim == latent_dim
    
    def test_encode_decode(self):
        """Test encoding and decoding"""
        dim = 50
        batch_size = 32
        latent_dim = 32
        
        model = VAEModel(dimension=dim, latent_dim=latent_dim)
        x = torch.randn(batch_size, dim)
        
        # Encode
        mu, logvar = model.encode(x)
        assert mu.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)
        
        # Reparameterize
        z = model.reparameterize(mu, logvar)
        assert z.shape == (batch_size, latent_dim)
        
        # Decode
        x_recon = model.decode(z)
        assert x_recon.shape == (batch_size, dim)
    
    def test_forward_pass(self):
        """Test full forward pass"""
        dim = 50
        batch_size = 32
        
        model = VAEModel(dimension=dim, latent_dim=32)
        x = torch.randn(batch_size, dim)
        
        x_recon, mu, logvar = model(x)
        
        assert x_recon.shape == (batch_size, dim)
        assert mu.shape == (batch_size, 32)
        assert logvar.shape == (batch_size, 32)
    
    def test_loss_computation(self):
        """Test VAE loss computation"""
        dim = 50
        batch_size = 32
        
        model = VAEModel(dimension=dim, latent_dim=32, beta=1.0)
        x = torch.randn(batch_size, dim)
        
        x_recon, mu, logvar = model(x)
        total_loss, recon_loss, kl_loss = model.compute_loss(x, x_recon, mu, logvar)
        
        assert total_loss.ndim == 0
        assert recon_loss.ndim == 0
        assert kl_loss.ndim == 0
        assert total_loss.item() >= 0
        assert recon_loss.item() >= 0
        assert kl_loss.item() >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
