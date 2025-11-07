"""
Optimal Transport Model

Learns a transport map T: R^d -> R^d from initial to final state.
Generates intermediate states by linear interpolation.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class OptimalTransportModel(nn.Module):
    """
    Neural network parameterized transport map for Optimal Transport.
    
    Architecture: MLP that maps x_0 -> x_T
    Inference: Linear interpolation x_t = (1 - t/T) x_0 + (t/T) T(x_0)
    """
    
    def __init__(
        self,
        dimension: int,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = 'relu',
        dropout: float = 0.1
    ):
        """
        Args:
            dimension: State space dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dimension = dimension
        self.hidden_dims = hidden_dims
        
        # Build MLP
        layers = []
        in_dim = dimension
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, dimension))
        
        self.network = nn.Sequential(*layers)
    
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
    
    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        """
        Compute transport map T(x_0).
        
        Args:
            x_0: Initial state (batch_size, d)
            
        Returns:
            x_T: Final state (batch_size, d)
        """
        return self.network(x_0)
    
    def interpolate(
        self,
        x_0: torch.Tensor,
        t: float,
        T: float = 1.0
    ) -> torch.Tensor:
        """
        Interpolate to intermediate time t.
        
        Args:
            x_0: Initial state (batch_size, d)
            t: Current time
            T: Total time
            
        Returns:
            x_t: State at time t (batch_size, d)
        """
        x_T = self.forward(x_0)
        alpha = t / T
        return (1 - alpha) * x_0 + alpha * x_T
    
    def generate_trajectory(
        self,
        x_0: torch.Tensor,
        time_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate full trajectory by interpolation.
        
        Args:
            x_0: Initial state (batch_size, d)
            time_grid: Time points (n_time,)
            
        Returns:
            trajectory: (batch_size, n_time, d)
        """
        batch_size = x_0.shape[0]
        n_time = len(time_grid)
        T = time_grid[-1].item()
        
        trajectory = torch.zeros(batch_size, n_time, self.dimension, device=x_0.device)
        
        for i, t in enumerate(time_grid):
            trajectory[:, i, :] = self.interpolate(x_0, t.item(), T)
        
        return trajectory
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        x_T: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Wasserstein-2 loss (MSE between T(x_0) and x_T).
        
        Args:
            x_0: Initial state (batch_size, d)
            x_T: Final state (batch_size, d)
            
        Returns:
            loss: Scalar loss
        """
        x_T_pred = self.forward(x_0)
        return torch.mean((x_T_pred - x_T) ** 2)
