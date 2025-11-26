"""
MLPlus Schrödinger Bridge Model

Enhanced version with:
1. Multi-scale time embedding
2. Residual connections
3. Layer normalization

Inherits from base SchrodingerBridgeModel for maximum code reuse.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List

from .sb_model import SchrodingerBridgeModel


class TimeEmbeddingNetwork(nn.Module):
    """Enhanced time embedding with multiple frequency scales"""
    
    def __init__(self, embedding_dim: int = 64, n_frequencies: int = 10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_frequencies = n_frequencies
        
        # Learnable frequency scales
        self.freq_scales = nn.Parameter(torch.randn(n_frequencies))
        
        # MLP to process sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_frequencies, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch_size,) or scalar
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Multi-scale sinusoidal encoding
        t = t.unsqueeze(-1)  # (batch_size, 1)
        freqs = self.freq_scales.unsqueeze(0)  # (1, n_frequencies)
        angles = 2 * np.pi * t * torch.exp(freqs)  # (batch_size, n_frequencies)
        
        # Concatenate sin and cos
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)
        features = torch.cat([sin_features, cos_features], dim=-1)
        
        # Process through MLP
        return self.mlp(features)


class ResidualBlock(nn.Module):
    """Residual block with layer norm for stable gradients"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return x + self.dropout(self.net(x))


class EnhancedPotentialNetwork(nn.Module):
    """Enhanced potential network with residual connections"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(n_blocks)
        ])
        
        # Output projection to scalar
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class MLPlus_SchrodingerBridgeModel(SchrodingerBridgeModel):
    """
    Enhanced Schrödinger Bridge Model (MLPlus version)
    
    Inherits from base SchrodingerBridgeModel and overrides:
    - Time embedding network (multi-scale)
    - Potential networks (residual + layer norm)
    
    All other methods (compute_drift, compute_loss, generate_trajectory)
    are inherited without modification.
    """
    
    def __init__(
        self,
        dimension: int,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        time_embedding_dim: int = 64,
        n_time_frequencies: int = 10,
        dropout: float = 0.1,
        diffusion_coeff: float = 0.1
    ):
        """
        Args:
            dimension: State space dimension (number of genes)
            hidden_dim: Hidden dimension for potential networks
            n_blocks: Number of residual blocks
            time_embedding_dim: Dimension of time embedding
            n_time_frequencies: Number of frequency scales for time
            dropout: Dropout probability
            diffusion_coeff: Diffusion coefficient D
        """
        # Initialize parent class (but we'll override the networks)
        # We need to call nn.Module.__init__ directly to avoid parent's network creation
        # Initialize base class with minimal parameters
        # We'll override the networks anyway
        super().__init__(
            dimension=dimension,
            hidden_dims=[hidden_dim],  # Dummy, we override the networks
            time_embedding_dim=time_embedding_dim,
            dropout=dropout,
            diffusion_coeff=diffusion_coeff
        )
        
        # Store MLPlus-specific parameters
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.n_time_frequencies = n_time_frequencies
        
        # Override time embedding with enhanced version
        self.time_embed = TimeEmbeddingNetwork(
            embedding_dim=time_embedding_dim,
            n_frequencies=n_time_frequencies
        )
        
        # Enhanced forward potential network (replaces parent's MLP)
        self.forward_potential = EnhancedPotentialNetwork(
            input_dim=dimension + time_embedding_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout
        )
        
        # Enhanced backward potential network (replaces parent's MLP)
        self.backward_potential = EnhancedPotentialNetwork(
            input_dim=dimension + time_embedding_dim,
            hidden_dim=hidden_dim,
            n_blocks=n_blocks,
            dropout=dropout
        )
    
    def _get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Override parent's time embedding with enhanced version
        
        Args:
            t: Time (batch_size,) or scalar
        Returns:
            embedding: (batch_size, time_embedding_dim)
        """
        return self.time_embed(t)
    
    def forward_potential_value(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Override to use enhanced time embedding
        
        Args:
            x: State (batch_size, d)
            t: Time (batch_size,) or scalar
        Returns:
            potential: (batch_size,)
        """
        t_embed = self._get_time_embedding(t)
        
        # Broadcast time embedding to match batch size
        if t_embed.shape[0] == 1 and x.shape[0] > 1:
            t_embed = t_embed.expand(x.shape[0], -1)
        
        xt = torch.cat([x, t_embed], dim=-1)
        return self.forward_potential(xt).squeeze(-1)
    
    def backward_potential_value(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Override to use enhanced time embedding
        
        Args:
            x: State (batch_size, d)
            t: Time (batch_size,) or scalar
        Returns:
            potential: (batch_size,)
        """
        t_embed = self._get_time_embedding(t)
        
        # Broadcast time embedding to match batch size
        if t_embed.shape[0] == 1 and x.shape[0] > 1:
            t_embed = t_embed.expand(x.shape[0], -1)
        
        xt = torch.cat([x, t_embed], dim=-1)
        return self.backward_potential(xt).squeeze(-1)
    
    # All other methods (compute_drift, compute_loss, forward, generate_trajectory)
    # are inherited from SchrodingerBridgeModel without modification


if __name__ == "__main__":
    # Test the MLPlus model
    print("Testing MLPlus Schrödinger Bridge Model")
    print("="*70)
    
    model = MLPlus_SchrodingerBridgeModel(
        dimension=100,
        hidden_dim=512,
        n_blocks=4,
        time_embedding_dim=64,
        n_time_frequencies=10,
        dropout=0.1
    )
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    # Test forward pass
    x = torch.randn(32, 100)
    t = torch.tensor(0.5)
    
    drift = model.compute_drift(x, t)
    print(f"Drift shape: {drift.shape}")
    
    # Test loss
    x_next = torch.randn(32, 100)
    loss = model.compute_loss(x, x_next, t, dt=0.1)
    print(f"Loss: {loss.item():.6f}")
    
    # Test trajectory generation
    time_grid = torch.linspace(0, 1, 10)
    trajectory = model.generate_trajectory(x[:4], time_grid)
    print(f"Trajectory shape: {trajectory.shape}")
    
    print("\n✓ All tests passed!")
