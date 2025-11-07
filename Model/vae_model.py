"""
VAE Model

Variational Autoencoder as a simplified baseline.
Learns latent representation and generates trajectories through latent interpolation.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class VAEModel(nn.Module):
    """
    Variational Autoencoder for trajectory generation.
    
    Encodes initial and final states to latent space.
    Generates intermediate states by interpolating in latent space.
    """
    
    def __init__(
        self,
        dimension: int,
        encoder_dims: List[int] = [256, 128],
        latent_dim: int = 32,
        decoder_dims: List[int] = [128, 256],
        activation: str = 'relu',
        dropout: float = 0.1,
        beta: float = 1.0
    ):
        """
        Args:
            dimension: State space dimension
            encoder_dims: Encoder hidden dimensions
            latent_dim: Latent space dimension
            decoder_dims: Decoder hidden dimensions
            activation: Activation function
            dropout: Dropout probability
            beta: KL divergence weight (β-VAE)
        """
        super().__init__()
        
        self.dimension = dimension
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder: x -> z
        self.encoder = self._build_encoder(
            dimension, encoder_dims, latent_dim, activation, dropout
        )
        
        # Decoder: z -> x
        self.decoder = self._build_decoder(
            latent_dim, decoder_dims, dimension, activation, dropout
        )
    
    def _build_encoder(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        activation: str,
        dropout: float
    ) -> nn.Module:
        """Build encoder network"""
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Mean and log-variance heads
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
        
        return nn.Sequential(*layers)
    
    def _build_decoder(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str,
        dropout: float
    ) -> nn.Module:
        """Build decoder network"""
        layers = []
        in_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))
        
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
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input state (batch_size, d)
            
        Returns:
            mu: Mean (batch_size, latent_dim)
            logvar: Log-variance (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε
        
        Args:
            mu: Mean (batch_size, latent_dim)
            logvar: Log-variance (batch_size, latent_dim)
            
        Returns:
            z: Latent sample (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to state.
        
        Args:
            z: Latent code (batch_size, latent_dim)
            
        Returns:
            x: Reconstructed state (batch_size, d)
        """
        return self.decoder(z)
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input state (batch_size, d)
            
        Returns:
            x_recon: Reconstructed state (batch_size, d)
            mu: Latent mean (batch_size, latent_dim)
            logvar: Latent log-variance (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def generate_trajectory(
        self,
        x_0: torch.Tensor,
        x_T: torch.Tensor,
        time_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate trajectory by interpolating in latent space.
        
        Args:
            x_0: Initial state (batch_size, d)
            x_T: Final state (batch_size, d)
            time_grid: Time points (n_time,)
            
        Returns:
            trajectory: (batch_size, n_time, d)
        """
        batch_size = x_0.shape[0]
        n_time = len(time_grid)
        T = time_grid[-1].item()
        
        # Encode initial and final states
        mu_0, _ = self.encode(x_0)
        mu_T, _ = self.encode(x_T)
        
        trajectory = torch.zeros(
            batch_size, n_time, self.dimension, device=x_0.device
        )
        
        # Interpolate in latent space
        for i, t in enumerate(time_grid):
            alpha = t.item() / T
            z_t = (1 - alpha) * mu_0 + alpha * mu_T
            trajectory[:, i, :] = self.decode(z_t)
        
        return trajectory
    
    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss = Reconstruction + β * KL.
        
        Args:
            x: Original input (batch_size, d)
            x_recon: Reconstructed input (batch_size, d)
            mu: Latent mean (batch_size, latent_dim)
            logvar: Latent log-variance (batch_size, latent_dim)
            
        Returns:
            total_loss: Total loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence
        """
        # Reconstruction loss (MSE)
        recon_loss = torch.mean((x - x_recon) ** 2)
        
        # KL divergence: KL(N(μ, σ²) || N(0, 1))
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for a batch (used by simple Trainer).
        
        Args:
            x: Input batch (batch_size, d)
            
        Returns:
            Total loss
        """
        x_recon, mu, logvar = self.forward(x)
        total_loss, _, _ = self.compute_loss(x, x_recon, mu, logvar)
        return total_loss
