"""
Optimal Transport Model for Cell State Transition

Learns the optimal transport map between two distributions (start and end timepoints).
Uses neural network to parameterize the transport map T: X_0 -> X_T
"""

import torch
import torch.nn as nn
from typing import List, Optional


class OptimalTransportModel(nn.Module):
    """
    Neural network parameterized optimal transport map.
    
    Learns: T(x_0) ≈ x_T where T minimizes Wasserstein distance
    
    Training: Minimize ||T(x_0) - x_T||² (L2 loss)
    """
    
    def __init__(
        self,
        dimension: int,
        hidden_dims: List[int] = [512, 512, 512, 512],
        activation: str = 'relu',
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """
        Args:
            dimension: State space dimension (input and output)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            dropout: Dropout probability
            use_residual: Whether to use residual connection (T(x) = x + ΔT(x))
        """
        super().__init__()
        
        self.dimension = dimension
        self.use_residual = use_residual
        
        # Build transport network
        layers = []
        in_dim = dimension
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, dimension))
        
        self.transport_net = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transport map.
        
        Args:
            x: Input state (batch_size, d)
            
        Returns:
            Transported state (batch_size, d)
        """
        if self.use_residual:
            # Residual connection: T(x) = x + ΔT(x)
            delta = self.transport_net(x)
            return x + delta
        else:
            # Direct mapping: T(x)
            return self.transport_net(x)
    
    def compute_loss(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute OT loss (L2 distance).
        
        Args:
            x_source: Source distribution samples (batch_size, d)
            x_target: Target distribution samples (batch_size, d)
            
        Returns:
            loss: Scalar loss
        """
        # Apply transport map
        x_transported = self.forward(x_source)
        
        # L2 loss
        loss = torch.mean((x_transported - x_target) ** 2)
        
        return loss
    
    def generate_trajectory(
        self,
        x_0: torch.Tensor,
        time_grid: torch.Tensor,
        method: str = 'linear'
    ) -> torch.Tensor:
        """
        Generate trajectory by linear interpolation.
        
        For OT model, we only learn start -> end mapping,
        so intermediate points are linearly interpolated.
        
        Args:
            x_0: Initial state (batch_size, d)
            time_grid: Time points (n_time,), should be in [0, 1]
            method: Interpolation method ('linear' only for now)
            
        Returns:
            trajectory: (batch_size, n_time, d)
        """
        batch_size = x_0.shape[0]
        n_time = len(time_grid)
        
        # Get final state
        x_T = self.forward(x_0)
        
        # Linear interpolation
        trajectory = torch.zeros(
            batch_size, n_time, self.dimension, device=x_0.device
        )
        
        for i, t in enumerate(time_grid):
            # Interpolate: x(t) = (1-t) * x_0 + t * x_T
            trajectory[:, i, :] = (1 - t) * x_0 + t * x_T
        
        return trajectory


class RegularizedOTModel(OptimalTransportModel):
    """
    OT Model with regularization terms for smoother transport.
    
    Adds:
    - Gradient penalty for Lipschitz constraint
    - Cycle consistency loss (optional)
    """
    
    def __init__(
        self,
        dimension: int,
        hidden_dims: List[int] = [512, 512, 512, 512],
        activation: str = 'relu',
        dropout: float = 0.1,
        use_residual: bool = True,
        gradient_penalty_weight: float = 0.1
    ):
        """
        Args:
            dimension: State space dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
            dropout: Dropout probability
            use_residual: Whether to use residual connection
            gradient_penalty_weight: Weight for gradient penalty
        """
        super().__init__(dimension, hidden_dims, activation, dropout, use_residual)
        
        self.gradient_penalty_weight = gradient_penalty_weight
    
    def compute_gradient_penalty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient penalty to encourage Lipschitz continuity.
        
        Args:
            x: Input samples (batch_size, d)
            
        Returns:
            penalty: Scalar penalty
        """
        x = x.requires_grad_(True)
        
        # Forward pass
        y = self.forward(x)
        
        # Compute gradients
        grad_outputs = torch.ones_like(y)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Gradient norm penalty (encourage ||∇T|| ≈ 1)
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-8)
        penalty = torch.mean((gradient_norm - 1.0) ** 2)
        
        return penalty
    
    def compute_loss(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute regularized OT loss.
        
        Args:
            x_source: Source distribution samples (batch_size, d)
            x_target: Target distribution samples (batch_size, d)
            
        Returns:
            loss: Scalar loss
        """
        # Base OT loss
        ot_loss = super().compute_loss(x_source, x_target)
        
        # Gradient penalty
        if self.gradient_penalty_weight > 0:
            gp = self.compute_gradient_penalty(x_source)
            total_loss = ot_loss + self.gradient_penalty_weight * gp
        else:
            total_loss = ot_loss
        
        return total_loss
