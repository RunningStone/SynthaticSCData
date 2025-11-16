"""
Variational Autoencoder Model for Cell State Transition

Learns latent representations and generates trajectories through latent space interpolation.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class VAEModel(nn.Module):
    """
    Variational Autoencoder for cell state modeling.
    
    Architecture:
    - Encoder: x -> (μ, log_σ²) in latent space
    - Decoder: z -> x̂ reconstruction
    - Trajectory: Interpolate in latent space between start and end
    """
    
    def __init__(
        self,
        dimension: int,
        latent_dim: int = 128,
        hidden_dims: List[int] = [512, 256],
        activation: str = 'relu',
        dropout: float = 0.1,
        beta: float = 1.0
    ):
        """
        Args:
            dimension: Input/output dimension (gene expression space)
            latent_dim: Latent space dimension
            hidden_dims: List of hidden layer dimensions for encoder/decoder
            activation: Activation function
            dropout: Dropout probability
            beta: Weight for KL divergence term (β-VAE)
        """
        super().__init__()
        
        self.dimension = dimension
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Build encoder
        encoder_layers = []
        in_dim = dimension
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            encoder_layers.append(self._get_activation(activation))
            encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden_dim))
            decoder_layers.append(self._get_activation(activation))
            decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(in_dim, dimension))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
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
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input (batch_size, d)
            
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)
        
        Args:
            mu: Mean (batch_size, latent_dim)
            logvar: Log variance (batch_size, latent_dim)
            
        Returns:
            z: Sampled latent code (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent code to reconstruction.
        
        Args:
            z: Latent code (batch_size, latent_dim)
            
        Returns:
            x_recon: Reconstruction (batch_size, d)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input (batch_size, d)
            
        Returns:
            x_recon: Reconstruction (batch_size, d)
            mu: Latent mean (batch_size, latent_dim)
            logvar: Latent log variance (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def compute_loss(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VAE loss for paired source-target samples.
        
        Loss = Recon_source + Recon_target + β * (KL_source + KL_target)
        
        Args:
            x_source: Source samples (batch_size, d)
            x_target: Target samples (batch_size, d)
            
        Returns:
            loss: Scalar loss
        """
        # Reconstruct both source and target
        x_source_recon, mu_source, logvar_source = self.forward(x_source)
        x_target_recon, mu_target, logvar_target = self.forward(x_target)
        
        # Reconstruction loss (MSE)
        recon_loss_source = torch.mean((x_source_recon - x_source) ** 2)
        recon_loss_target = torch.mean((x_target_recon - x_target) ** 2)
        recon_loss = recon_loss_source + recon_loss_target
        
        # KL divergence loss (clamp logvar to prevent numerical instability)
        logvar_source = torch.clamp(logvar_source, min=-10, max=10)
        logvar_target = torch.clamp(logvar_target, min=-10, max=10)
        
        kl_loss_source = -0.5 * torch.mean(1 + logvar_source - mu_source.pow(2) - logvar_source.exp())
        kl_loss_target = -0.5 * torch.mean(1 + logvar_target - mu_target.pow(2) - logvar_target.exp())
        kl_loss = kl_loss_source + kl_loss_target
        
        # Total loss (check for NaN)
        total_loss = recon_loss + self.beta * kl_loss
        
        # Safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf detected in VAE loss!")
            print(f"  recon_loss: {recon_loss.item()}")
            print(f"  kl_loss: {kl_loss.item()}")
            # Return a large but finite loss with gradient
            return recon_loss.clamp(max=1e6) + self.beta * kl_loss.clamp(max=1e6)
        
        return total_loss
    
    def generate_trajectory(
        self,
        x_0: torch.Tensor,
        time_grid: torch.Tensor,
        method: str = 'latent_interpolation'
    ) -> torch.Tensor:
        """
        Generate trajectory through latent space interpolation.
        
        For VAE, we:
        1. Encode x_0 to get z_0 (using mean, no sampling)
        2. Predict z_T by learning the latent transition
        3. Interpolate in latent space: z(t) = (1-t) * z_0 + t * z_T
        4. Decode each z(t) to get x(t)
        
        Args:
            x_0: Initial state (batch_size, d)
            time_grid: Time points (n_time,), should be in [0, 1]
            method: Generation method ('latent_interpolation')
            
        Returns:
            trajectory: (batch_size, n_time, d)
        """
        batch_size = x_0.shape[0]
        n_time = len(time_grid)
        
        # Encode initial state (use mean, no sampling for deterministic trajectory)
        with torch.no_grad():
            mu_0, _ = self.encode(x_0)
        
        # For trajectory generation, we need to predict the final latent state
        # Here we use a simple approach: encode x_0, then linearly interpolate to a learned target
        # In practice, you might want to learn a separate transition model in latent space
        
        # For now, we'll use the latent mean as the trajectory
        # This is a simplified version - in full implementation, you'd learn z_T
        trajectory = torch.zeros(
            batch_size, n_time, self.dimension, device=x_0.device
        )
        
        # Decode at each timepoint (using same latent code for now)
        # This is a placeholder - proper implementation would interpolate to learned z_T
        for i, t in enumerate(time_grid):
            # Simple approach: decode the same latent code
            # In full implementation: z_t = (1-t) * z_0 + t * z_T
            z_t = mu_0  # Placeholder
            x_t = self.decode(z_t)
            trajectory[:, i, :] = x_t
        
        return trajectory


# Old ConditionalVAEModel removed - see c_vae_model.py for new time-conditional implementation
