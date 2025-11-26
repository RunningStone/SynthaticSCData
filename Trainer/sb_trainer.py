#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schrödinger Bridge Trainer

Refactored to inherit from BaseTrainer with customization for SB-specific logic.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
from .base_trainer import BaseTrainer


class SBTrainer(BaseTrainer):
    """
    Trainer for Schrödinger Bridge models.
    
    Customizations:
    - process_batch_data(): Extract consecutive time pairs
    - compute_batch_loss(): Compute SB loss with drift field
    - _needs_grad_for_eval(): Return True (SB needs gradients for drift computation)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 5e-4,
        device: str = 'cuda',
        output_dir: str = './outputs',
        weight_decay: float = 1e-5,
        grad_clip_norm: float = 5.0,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None
    ):
        """
        Args:
            model: SchrodingerBridgeModel instance
            train_loader: Training data loader
            test_loader: Test data loader
            learning_rate: Learning rate
            device: Device to train on
            output_dir: Output directory for checkpoints
            weight_decay: Weight decay for regularization
            grad_clip_norm: Gradient clipping norm
            optimizer_kwargs: Additional optimizer parameters
            scheduler_config: Scheduler configuration
        """
        super().__init__(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=learning_rate,
            device=device,
            output_dir=output_dir,
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_config=scheduler_config
        )
    
    def process_batch_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        """
        Extract consecutive time pairs for SB training.
        
        SB models learn local dynamics between consecutive timepoints,
        so we create pairs (x_t, x_{t+1}) from the batch.
        
        Args:
            X: Batch features (batch_size, d)
            y: Batch time labels (batch_size,)
            
        Returns:
            Dictionary containing pairs data, or None if no valid pairs
        """
        # Group by time labels
        unique_times = torch.unique(y)
        time_to_indices = {
            t.item(): (y == t).nonzero(as_tuple=True)[0]
            for t in unique_times
        }
        
        # Create consecutive pairs
        sorted_times = sorted(time_to_indices.keys())
        
        pairs_list = []
        for i in range(len(sorted_times) - 1):
            t_curr = sorted_times[i]
            t_next = sorted_times[i + 1]
            
            indices_curr = time_to_indices[t_curr]
            indices_next = time_to_indices[t_next]
            
            if len(indices_curr) == 0 or len(indices_next) == 0:
                continue
            
            # Match pairs (randomly if sizes differ)
            n_pairs = min(len(indices_curr), len(indices_next))
            if len(indices_curr) > n_pairs:
                indices_curr = indices_curr[torch.randperm(len(indices_curr))[:n_pairs]]
            if len(indices_next) > n_pairs:
                indices_next = indices_next[torch.randperm(len(indices_next))[:n_pairs]]
            
            x_t = X[indices_curr]
            x_next = X[indices_next]
            
            # Time values (normalized to [0, 1])
            t = torch.full((n_pairs,), float(t_curr) / len(sorted_times), device=self.device)
            dt = 1.0 / len(sorted_times)
            
            pairs_list.append({
                'x_t': x_t,
                'x_next': x_next,
                't': t,
                'dt': dt
            })
        
        if len(pairs_list) == 0:
            return None
        
        return {'pairs': pairs_list}
    
    def compute_batch_loss(
        self,
        batch_data: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute SB loss for all pairs in the batch.
        
        Args:
            batch_data: Dictionary containing 'pairs' list
            
        Returns:
            Average loss across all pairs
        """
        pairs_list = batch_data['pairs']
        losses = []
        
        for pair_data in pairs_list:
            loss = self.model.compute_loss(
                x_t=pair_data['x_t'],
                x_next=pair_data['x_next'],
                t=pair_data['t'],
                dt=pair_data['dt']
            )
            losses.append(loss)
        
        return torch.mean(torch.stack(losses))
    
    def _needs_grad_for_eval(self) -> bool:
        """
        SB models need gradients during evaluation for drift computation.
        
        The drift field is computed via autograd.grad on the input x,
        so we need gradients enabled even during evaluation.
        
        Returns:
            True
        """
        return True
