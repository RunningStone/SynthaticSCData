"""
Batch Optimal Transport Model for Sequential Time Points

For Setting 2 with multiple time points (e.g., 0d, 8h, 1d, 3d, 7d),
this model trains separate OT models for each consecutive time pair.

During inference, it sequentially applies the OT models to generate
the final state from the initial state.

Architecture:
- OT_1: t0 -> t1 (e.g., 0d -> 8h)
- OT_2: t1 -> t2 (e.g., 8h -> 1d)
- OT_3: t2 -> t3 (e.g., 1d -> 3d)
- OT_4: t3 -> t4 (e.g., 3d -> 7d)

Inference: x_0 -> OT_1 -> x_1 -> OT_2 -> x_2 -> OT_3 -> x_3 -> OT_4 -> x_4
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple, Union
from .ot_model import OptimalTransportModel


class BatchOTModel(nn.Module):
    """
    Batch OT Model that trains separate OT models for consecutive time pairs.
    
    This model is designed for Setting 2 where we have multiple time points.
    Instead of learning a single direct mapping from start to end, we learn
    a sequence of mappings between consecutive time points.
    """
    
    def __init__(
        self,
        dimension: int,
        n_timepoints: int,
        time_labels: List[str],
        hidden_dims: List[int] = [512, 512, 512, 512],
        activation: str = 'relu',
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """
        Args:
            dimension: State space dimension (input and output)
            n_timepoints: Number of time points
            time_labels: List of time label names (e.g., ['0d', '8h', '1d', '3d', '7d'])
            hidden_dims: List of hidden layer dimensions for each OT model
            activation: Activation function
            dropout: Dropout probability
            use_residual: Whether to use residual connection in OT models
        """
        super().__init__()
        
        self.dimension = dimension
        self.n_timepoints = n_timepoints
        self.time_labels = time_labels
        self.n_transitions = n_timepoints - 1  # Number of consecutive pairs
        
        # Create separate OT models for each consecutive time pair
        self.ot_models = nn.ModuleList([
            OptimalTransportModel(
                dimension=dimension,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
                use_residual=use_residual
            )
            for _ in range(self.n_transitions)
        ])
        
        # Store time pair information for reference
        self.time_pairs = [
            (time_labels[i], time_labels[i+1]) 
            for i in range(self.n_transitions)
        ]
        
        print(f"\nBatchOTModel initialized with {self.n_transitions} OT models:")
        for i, (t_start, t_end) in enumerate(self.time_pairs):
            print(f"  OT_{i}: {t_start} -> {t_end}")
    
    def forward(
        self,
        x: torch.Tensor,
        t_source: Union[int, torch.Tensor],
        t_target: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply sequential OT mappings from source to target time.
        
        Args:
            x: Input state (batch_size, d)
            t_source: Source time index (0-indexed), can be int or tensor
            t_target: Target time index (0-indexed), can be int or tensor
            
        Returns:
            Transported state (batch_size, d)
        """
        # Convert tensors to int if needed
        if isinstance(t_source, torch.Tensor):
            t_source = t_source.item() if t_source.numel() == 1 else int(t_source[0].item())
        if isinstance(t_target, torch.Tensor):
            t_target = t_target.item() if t_target.numel() == 1 else int(t_target[0].item())
        
        if t_source >= t_target:
            raise ValueError(f"Source time {t_source} must be before target time {t_target}")
        
        if t_source < 0 or t_target >= self.n_timepoints:
            raise ValueError(f"Time indices must be in [0, {self.n_timepoints-1}]")
        
        # Sequentially apply OT models
        x_current = x
        for i in range(t_source, t_target):
            x_current = self.ot_models[i](x_current)
        
        return x_current
    
    def compute_loss(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        t_source: Union[int, torch.Tensor],
        t_target: Union[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for a specific time transition.
        
        For consecutive time pairs (t_target = t_source + 1), we directly compute
        the OT loss using the corresponding model.
        
        For non-consecutive pairs, we apply sequential mappings and compute the
        final loss, but this is mainly for evaluation purposes.
        
        Args:
            x_source: Source distribution samples (batch_size, d)
            x_target: Target distribution samples (batch_size, d)
            t_source: Source time index, can be int or tensor
            t_target: Target time index, can be int or tensor
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses for logging
        """
        # Convert tensors to int if needed
        t_src = t_source
        t_tgt = t_target
        if isinstance(t_source, torch.Tensor):
            t_src = t_source.item() if t_source.numel() == 1 else int(t_source[0].item())
        if isinstance(t_target, torch.Tensor):
            t_tgt = t_target.item() if t_target.numel() == 1 else int(t_target[0].item())
        
        # Apply sequential mappings
        x_transported = self.forward(x_source, t_src, t_tgt)
        
        # Compute L2 loss
        loss = torch.mean((x_transported - x_target) ** 2)
        
        # Create loss dictionary for logging
        loss_dict = {
            'total_loss': loss.item(),
            f'loss_{self.time_labels[t_src]}_to_{self.time_labels[t_tgt]}': loss.item()
        }
        
        return loss, loss_dict
    
    def compute_loss_for_transition(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        transition_idx: int
    ) -> torch.Tensor:
        """
        Compute loss for a specific transition (consecutive time pair).
        
        This is used during training to update individual OT models.
        
        Args:
            x_source: Source distribution samples (batch_size, d)
            x_target: Target distribution samples (batch_size, d)
            transition_idx: Index of the transition (0 to n_transitions-1)
            
        Returns:
            loss: Scalar loss for this transition
        """
        if transition_idx < 0 or transition_idx >= self.n_transitions:
            raise ValueError(f"Transition index must be in [0, {self.n_transitions-1}]")
        
        # Use the specific OT model for this transition
        return self.ot_models[transition_idx].compute_loss(x_source, x_target)
    
    def generate_trajectory(
        self,
        x_0: torch.Tensor,
        time_grid: torch.Tensor,
        method: str = 'sequential'
    ) -> torch.Tensor:
        """
        Generate trajectory by sequentially applying OT models.
        
        Args:
            x_0: Initial state (batch_size, d)
            time_grid: Time points (n_time,), should be in [0, 1]
            method: Generation method (any value will use sequential generation)
                   Accepts 'sequential', 'deterministic', or any other string
            
        Returns:
            trajectory: (batch_size, n_time, d)
        """
        batch_size = x_0.shape[0]
        n_time = len(time_grid)
        
        trajectory = torch.zeros(
            batch_size, n_time, self.dimension, device=x_0.device
        )
        
        # Always use sequential generation regardless of method parameter
        # This ensures compatibility with evaluators that use 'deterministic'
        # Generate states at each time point
        states = [x_0]
        x_current = x_0
        
        for i in range(self.n_transitions):
            x_current = self.ot_models[i](x_current)
            states.append(x_current)
        
        # Interpolate between discrete time points
        for i, t in enumerate(time_grid):
            # Find which segment this time falls into
            segment_idx = int(t * self.n_transitions)
            segment_idx = min(segment_idx, self.n_transitions - 1)
            
            # Local time within segment
            t_local = (t * self.n_transitions) - segment_idx
            t_local = max(0.0, min(1.0, t_local))
            
            # Linear interpolation within segment
            if segment_idx < self.n_transitions:
                x_start = states[segment_idx]
                x_end = states[segment_idx + 1]
                trajectory[:, i, :] = (1 - t_local) * x_start + t_local * x_end
            else:
                trajectory[:, i, :] = states[-1]
        
        return trajectory
    
    def get_model_for_transition(self, transition_idx: int) -> OptimalTransportModel:
        """
        Get the OT model for a specific transition.
        
        Args:
            transition_idx: Index of the transition (0 to n_transitions-1)
            
        Returns:
            The OT model for this transition
        """
        if transition_idx < 0 or transition_idx >= self.n_transitions:
            raise ValueError(f"Transition index must be in [0, {self.n_transitions-1}]")
        
        return self.ot_models[transition_idx]
    
    def get_transition_info(self) -> List[Tuple[str, str]]:
        """
        Get information about all transitions.
        
        Returns:
            List of (source_time, target_time) tuples
        """
        return self.time_pairs
    
    def save_models(self, save_dir: str):
        """
        Save all OT models separately.
        
        Args:
            save_dir: Directory to save models
        """
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, (t_start, t_end) in enumerate(self.time_pairs):
            model_path = save_path / f'ot_model_{i}_{t_start}_to_{t_end}.pt'
            torch.save(self.ot_models[i].state_dict(), model_path)
            print(f"Saved OT model {i}: {t_start} -> {t_end} to {model_path}")
    
    def load_models(self, save_dir: str):
        """
        Load all OT models from directory.
        
        Args:
            save_dir: Directory containing saved models
        """
        from pathlib import Path
        save_path = Path(save_dir)
        
        for i, (t_start, t_end) in enumerate(self.time_pairs):
            model_path = save_path / f'ot_model_{i}_{t_start}_to_{t_end}.pt'
            if model_path.exists():
                self.ot_models[i].load_state_dict(torch.load(model_path))
                print(f"Loaded OT model {i}: {t_start} -> {t_end} from {model_path}")
            else:
                print(f"Warning: Model file not found: {model_path}")
