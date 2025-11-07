"""
Schrödinger Bridge Model

Learns time-dependent drift field b(x, t) for the full trajectory.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class SchrodingerBridgeModel(nn.Module):
    """
    Neural network parameterized drift field for Schrödinger Bridge.
    
    Learns: b(x, t) = -D ∇[φ(x, t) + ψ(x, t)]
    where φ and ψ are forward and backward potentials.
    
    Training: Minimize ||∇φ + ∇ψ + (x_{t+Δt} - x_t)/Δt||²
    """
    
    def __init__(
        self,
        dimension: int,
        hidden_dims: List[int] = [512, 512, 512, 512],
        time_embedding_dim: int = 64,
        activation: str = 'relu',
        dropout: float = 0.1,
        diffusion_coeff: float = 0.1
    ):
        """
        Args:
            dimension: State space dimension
            hidden_dims: List of hidden layer dimensions
            time_embedding_dim: Dimension of time embedding
            activation: Activation function
            dropout: Dropout probability
            diffusion_coeff: Diffusion coefficient D
        """
        super().__init__()
        
        self.dimension = dimension
        self.time_embedding_dim = time_embedding_dim
        self.D = diffusion_coeff
        
        # Time embedding network (sinusoidal + learnable)
        self.time_embed = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Forward potential network φ(x, t)
        self.forward_potential = self._build_potential_network(
            dimension + time_embedding_dim, hidden_dims, activation, dropout
        )
        
        # Backward potential network ψ(x, t)
        self.backward_potential = self._build_potential_network(
            dimension + time_embedding_dim, hidden_dims, activation, dropout
        )
    
    def _build_potential_network(
        self,
        input_dim: int,
        hidden_dims: List[int],
        activation: str,
        dropout: float
    ) -> nn.Module:
        """Build potential network"""
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output scalar potential
        layers.append(nn.Linear(in_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal time embedding.
        
        Args:
            t: Time values (batch_size,) or scalar
            
        Returns:
            Time embedding (batch_size, time_embedding_dim)
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Sinusoidal encoding
        half_dim = self.time_embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Learnable transformation
        return self.time_embed(emb)
    
    def forward_potential_value(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute forward potential φ(x, t).
        
        Args:
            x: State (batch_size, d)
            t: Time (batch_size,) or scalar
            
        Returns:
            φ: Potential values (batch_size, 1)
        """
        t_emb = self._get_time_embedding(t)
        if t_emb.shape[0] == 1 and x.shape[0] > 1:
            t_emb = t_emb.repeat(x.shape[0], 1)
        
        xt = torch.cat([x, t_emb], dim=-1)
        return self.forward_potential(xt)
    
    def backward_potential_value(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute backward potential ψ(x, t).
        
        Args:
            x: State (batch_size, d)
            t: Time (batch_size,) or scalar
            
        Returns:
            ψ: Potential values (batch_size, 1)
        """
        t_emb = self._get_time_embedding(t)
        if t_emb.shape[0] == 1 and x.shape[0] > 1:
            t_emb = t_emb.repeat(x.shape[0], 1)
        
        xt = torch.cat([x, t_emb], dim=-1)
        return self.backward_potential(xt)
    
    def compute_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute drift field b(x, t) = -D ∇[φ(x, t) + ψ(x, t)].
        
        Args:
            x: State (batch_size, d)
            t: Time (batch_size,) or scalar
            
        Returns:
            drift: Drift field (batch_size, d)
        """
        x = x.requires_grad_(True)
        
        # Compute potentials
        phi = self.forward_potential_value(x, t)
        psi = self.backward_potential_value(x, t)
        total_potential = phi + psi
        
        # Compute gradient
        grad = torch.autograd.grad(
            total_potential.sum(),
            x,
            create_graph=True
        )[0]
        
        return -self.D * grad
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returns drift field.
        
        Args:
            x: State (batch_size, d)
            t: Time (batch_size,) or scalar
            
        Returns:
            drift: (batch_size, d)
        """
        return self.compute_drift(x, t)
    
    def generate_trajectory(
        self,
        x_0: torch.Tensor,
        time_grid: torch.Tensor,
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        Generate trajectory by solving SDE with learned drift.
        
        Args:
            x_0: Initial state (batch_size, d)
            time_grid: Time points (n_time,)
            method: Integration method ('euler' or 'deterministic')
            
        Returns:
            trajectory: (batch_size, n_time, d)
        """
        batch_size = x_0.shape[0]
        n_time = len(time_grid)
        
        trajectory = torch.zeros(
            batch_size, n_time, self.dimension, device=x_0.device
        )
        trajectory[:, 0, :] = x_0
        
        for i in range(n_time - 1):
            t = time_grid[i]
            dt = time_grid[i + 1] - time_grid[i]
            
            x_current = trajectory[:, i, :]
            drift = self.compute_drift(x_current, t)
            
            if method == 'euler':
                # Euler-Maruyama with noise
                noise = torch.randn_like(x_current) * torch.sqrt(2 * self.D * dt)
                trajectory[:, i + 1, :] = x_current + drift * dt + noise
            else:
                # Deterministic (drift only)
                trajectory[:, i + 1, :] = x_current + drift * dt
        
        return trajectory
    
    def compute_loss(
        self,
        x_t: torch.Tensor,
        x_next: torch.Tensor,
        t: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Compute Schrödinger Bridge loss.
        
        Loss = ||∇φ + ∇ψ + (x_{t+Δt} - x_t)/Δt||²
        
        Args:
            x_t: State at time t (batch_size, d)
            x_next: State at time t+dt (batch_size, d)
            t: Time values (batch_size,) or scalar
            dt: Time step
            
        Returns:
            loss: Scalar loss
        """
        # Compute drift
        drift = self.compute_drift(x_t, t)
        
        # Empirical velocity
        empirical_velocity = (x_next - x_t) / dt
        
        # Loss: drift should match empirical velocity
        return torch.mean((drift - empirical_velocity) ** 2)
