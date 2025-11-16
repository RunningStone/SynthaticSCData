"""
Conditional Variational Autoencoder Model for Cell State Transition

Time-conditioned VAE that learns to predict future cell states given:
- Initial state x_0
- Target time point t

Architecture:
- Encoder: x_0 -> (μ, log_σ²) in latent space
- Time Encoder: t -> time_condition (latent space offset)
- Latent Transition: z_T = z_0 + time_condition
- Decoder: z -> x̂ reconstruction
- Loss: VAE (reconstruction + KL) + MMD (distribution matching)
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


def compute_mmd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: str = 'rbf',
    bandwidth: float = 1.0
) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.
    
    MMD measures the distance between two distributions using kernel methods.
    
    Args:
        x: Samples from first distribution (batch_size, d)
        y: Samples from second distribution (batch_size, d)
        kernel: Kernel type ('rbf' or 'linear')
        bandwidth: Bandwidth for RBF kernel
        
    Returns:
        mmd: Scalar MMD loss
    """
    def rbf_kernel(x1, x2, bandwidth):
        """RBF (Gaussian) kernel"""
        # Compute pairwise squared distances
        x1_norm = (x1 ** 2).sum(dim=1, keepdim=True)
        x2_norm = (x2 ** 2).sum(dim=1, keepdim=True)
        dist = x1_norm + x2_norm.t() - 2.0 * torch.mm(x1, x2.t())
        return torch.exp(-dist / (2 * bandwidth ** 2))
    
    def linear_kernel(x1, x2):
        """Linear kernel"""
        return torch.mm(x1, x2.t())
    
    # Select kernel
    if kernel == 'rbf':
        k = lambda x1, x2: rbf_kernel(x1, x2, bandwidth)
    elif kernel == 'linear':
        k = linear_kernel
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # Compute kernel matrices
    k_xx = k(x, x)
    k_yy = k(y, y)
    k_xy = k(x, y)
    
    # MMD^2 = E[k(x,x)] + E[k(y,y)] - 2*E[k(x,y)]
    batch_size = x.shape[0]
    mmd = k_xx.sum() / (batch_size ** 2) + k_yy.sum() / (batch_size ** 2) - 2 * k_xy.sum() / (batch_size ** 2)
    
    return mmd


class ConditionalVAEModel(nn.Module):
    """
    Time-Conditional VAE for cell state transition.
    
    Key idea:
    - Encode initial state x_0 to latent z_0
    - Add time-dependent offset: z_T = z_0 + time_condition(t)
    - Decode z_T to predict target state x_T
    - Decode z_0 to reconstruct x_0 (for VAE regularization)
    
    Loss components:
    1. Reconstruction loss: ||x_0_recon - x_0||²
    2. KL divergence: KL(q(z_0|x_0) || p(z_0))
    3. MMD loss: MMD(x_T_sample, x_T_real) for distribution matching
    """
    
    def __init__(
        self,
        dimension: int,
        n_timepoints: int,
        latent_dim: int = 128,
        time_embedding_dim: int = 64,
        hidden_dims: List[int] = [512, 256],
        activation: str = 'relu',
        dropout: float = 0.1,
        beta: float = 1.0,
        mmd_weight: float = 1.0,
        mmd_kernel: str = 'rbf',
        mmd_bandwidth: float = 1.0
    ):
        """
        Args:
            dimension: Input/output dimension (gene expression space)
            n_timepoints: Number of time points in the dataset
            latent_dim: Latent space dimension
            time_embedding_dim: Dimension of time embedding
            hidden_dims: List of hidden layer dimensions for encoder/decoder
            activation: Activation function
            dropout: Dropout probability
            beta: Weight for KL divergence term (β-VAE)
            mmd_weight: Weight for MMD loss
            mmd_kernel: Kernel type for MMD ('rbf' or 'linear')
            mmd_bandwidth: Bandwidth for RBF kernel
        """
        super().__init__()
        
        self.dimension = dimension
        self.n_timepoints = n_timepoints
        self.latent_dim = latent_dim
        self.time_embedding_dim = time_embedding_dim
        self.beta = beta
        self.mmd_weight = mmd_weight
        self.mmd_kernel = mmd_kernel
        self.mmd_bandwidth = mmd_bandwidth
        
        # Time embedding: maps time index to embedding vector
        self.time_embedding = nn.Embedding(n_timepoints, time_embedding_dim)
        
        # Time encoder: maps time embedding to latent space offset
        self.time_encoder = nn.Sequential(
            nn.Linear(time_embedding_dim, latent_dim),
            self._get_activation(activation),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Build encoder: x_0 -> (μ, log_σ²)
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
        
        # Build decoder: z -> x̂
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
    
    def encode_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time index to latent space offset.
        
        Args:
            t: Time indices (batch_size,) as integers
            
        Returns:
            time_condition: Latent space offset (batch_size, latent_dim)
        """
        time_emb = self.time_embedding(t)  # (batch_size, time_embedding_dim)
        time_condition = self.time_encoder(time_emb)  # (batch_size, latent_dim)
        return time_condition
    
    def forward(
        self,
        x_0: torch.Tensor,
        t_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through conditional VAE.
        
        Args:
            x_0: Initial state (batch_size, d)
            t_target: Target time indices (batch_size,) as integers
            
        Returns:
            x_0_recon: Reconstruction of x_0 (batch_size, d)
            x_T_sample: Predicted target state (batch_size, d)
            mu: Latent mean (batch_size, latent_dim)
            logvar: Latent log variance (batch_size, latent_dim)
        """
        # Encode initial state
        mu, logvar = self.encode(x_0)
        z_0 = self.reparameterize(mu, logvar)
        
        # Reconstruct x_0 from z_0 (for VAE regularization)
        x_0_recon = self.decode(z_0)
        
        # Encode time condition
        time_condition = self.encode_time(t_target)
        
        # Latent transition: z_T = z_0 + time_condition
        z_T = z_0 + time_condition
        
        # Generate target state
        x_T_sample = self.decode(z_T)
        
        return x_0_recon, x_T_sample, mu, logvar
    
    def compute_loss(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        t_source: Optional[torch.Tensor] = None,
        t_target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute conditional VAE loss.
        
        Loss = Reconstruction + β * KL + λ * MMD
        
        Args:
            x_source: Source state (batch_size, d)
            x_target: Target state (batch_size, d)
            t_source: Source time indices (batch_size,) as integers, or None
            t_target: Target time indices (batch_size,) as integers, or None
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual loss components
        """
        batch_size = x_source.shape[0]
        device = x_source.device
        
        # Handle time indices
        if t_target is None:
            # Default: assume target is the last time point
            t_target = torch.full((batch_size,), self.n_timepoints - 1, dtype=torch.long, device=device)
        elif not isinstance(t_target, torch.Tensor):
            # Convert to tensor if needed
            t_target = torch.full((batch_size,), int(t_target), dtype=torch.long, device=device)
        elif t_target.dtype != torch.long:
            # Ensure correct dtype
            t_target = t_target.long()
        
        # Forward pass
        x_0_recon, x_T_sample, mu, logvar = self.forward(x_source, t_target)
        
        # 1. Reconstruction loss: ||x_0_recon - x_0||²
        recon_loss = torch.mean((x_0_recon - x_source) ** 2)
        
        # 2. KL divergence: KL(q(z_0|x_0) || N(0, I))
        # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        # 3. MMD loss: distribution matching between x_T_sample and x_target
        mmd_loss = compute_mmd_loss(
            x_T_sample,
            x_target,
            kernel=self.mmd_kernel,
            bandwidth=self.mmd_bandwidth
        )
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + self.mmd_weight * mmd_loss
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'mmd': mmd_loss.item()
        }
        
        return total_loss, loss_dict
    
    def generate_trajectory(
        self,
        x_0: torch.Tensor,
        time_grid: torch.Tensor,
        t_source_idx: int,
        t_target_idx: int,
        method: str = 'latent_interpolation'
    ) -> torch.Tensor:
        """
        Generate trajectory from x_0 to target time point.
        
        Args:
            x_0: Initial state (batch_size, d)
            time_grid: Time points (n_time,), should be in [0, 1]
            t_source_idx: Source time index (integer) - typically 0
            t_target_idx: Target time index (integer)
            method: Generation method ('latent_interpolation')
            
        Returns:
            trajectory: (batch_size, n_time, d)
        """
        batch_size = x_0.shape[0]
        n_time = len(time_grid)
        device = x_0.device
        
        with torch.no_grad():
            # Encode initial state (use mean for deterministic generation)
            mu_0, _ = self.encode(x_0)
            
            # Encode target time
            t_target = torch.full((batch_size,), t_target_idx, dtype=torch.long, device=device)
            time_condition = self.encode_time(t_target)
            
            # Target latent code
            mu_T = mu_0 + time_condition
            
            # Interpolate in latent space
            trajectory = torch.zeros(batch_size, n_time, self.dimension, device=device)
            
            for i, t in enumerate(time_grid):
                # Latent space interpolation: z(t) = (1-t) * z_0 + t * z_T
                z_t = (1 - t) * mu_0 + t * mu_T
                
                # Decode
                x_t = self.decode(z_t)
                trajectory[:, i, :] = x_t
        
        return trajectory
