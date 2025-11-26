#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Trainer for OT and VAE models

Refactored to inherit from BaseTrainer with customization for direct mapping models.
"""

import torch
from torch.utils.data import DataLoader
import inspect
from typing import Dict, Optional, Any
from .base_trainer import BaseTrainer


class UnifiedTrainer(BaseTrainer):
    """
    Unified trainer for OT and VAE models.
    
    These models learn direct source->target mappings (unlike SB which learns
    time-dependent drift fields), so they use first and last timepoints.
    
    Customizations:
    - process_batch_data(): Extract source-target pairs (first and last timepoints)
    - compute_batch_loss(): Handle both conditional and non-conditional models
    - _pre_training_hook(): Fit normalizers for VAE models
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
        model_type: str = 'ot',  # 'ot' or 'vae'
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None
    ):
        """
        Args:
            model: Model instance (OT or VAE)
            train_loader: Training data loader
            test_loader: Test data loader
            learning_rate: Learning rate
            device: Device to train on
            output_dir: Output directory for checkpoints
            weight_decay: Weight decay for regularization
            grad_clip_norm: Gradient clipping norm
            model_type: Type of model ('ot' or 'vae')
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
        
        self.model_type = model_type
    
    def process_batch_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        """
        Extract source-target pairs (first and last timepoints).
        
        OT and VAE models learn direct mappings from source to target,
        so we use the first and last timepoints in the batch.
        
        Args:
            X: Batch features (batch_size, d)
            y: Batch time labels (batch_size,)
            
        Returns:
            Dictionary containing source-target pairs, or None if invalid
        """
        # Extract first and last timepoints
        unique_times = torch.unique(y)
        time_to_indices = {
            t.item(): (y == t).nonzero(as_tuple=True)[0]
            for t in unique_times
        }
        sorted_times = sorted(time_to_indices.keys())
        
        if len(sorted_times) < 2:
            return None
        
        # Use first and last timepoints
        t_start = sorted_times[0]
        t_end = sorted_times[-1]
        
        indices_start = time_to_indices[t_start]
        indices_end = time_to_indices[t_end]
        
        if len(indices_start) == 0 or len(indices_end) == 0:
            return None
        
        # Match pairs
        n_pairs = min(len(indices_start), len(indices_end))
        if len(indices_start) > n_pairs:
            indices_start = indices_start[torch.randperm(len(indices_start))[:n_pairs]]
        if len(indices_end) > n_pairs:
            indices_end = indices_end[torch.randperm(len(indices_end))[:n_pairs]]
        
        x_source = X[indices_start]
        x_target = X[indices_end]
        
        return {
            'x_source': x_source,
            'x_target': x_target,
            't_source': t_start,
            't_target': t_end,
            'n_pairs': n_pairs
        }
    
    def compute_batch_loss(
        self,
        batch_data: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute loss for OT/VAE models.
        
        Handles both conditional models (VAE) that need time indices
        and non-conditional models (OT) that don't.
        
        Args:
            batch_data: Dictionary containing source-target pairs
            
        Returns:
            Loss tensor
        """
        x_source = batch_data['x_source']
        x_target = batch_data['x_target']
        
        # Check if model needs time indices (conditional models)
        loss_signature = inspect.signature(self.model.compute_loss)
        
        if len(loss_signature.parameters) > 2:
            # Conditional model: pass time indices
            t_source = batch_data['t_source']
            t_target = batch_data['t_target']
            n_pairs = batch_data['n_pairs']
            
            t_source_tensor = torch.full(
                (n_pairs,), t_source, dtype=torch.long, device=self.device
            )
            t_target_tensor = torch.full(
                (n_pairs,), t_target, dtype=torch.long, device=self.device
            )
            
            loss_output = self.model.compute_loss(
                x_source, x_target, t_source_tensor, t_target_tensor
            )
            
            # Handle both single loss and (loss, loss_dict) returns
            if isinstance(loss_output, tuple):
                loss, loss_dict = loss_output
            else:
                loss = loss_output
        else:
            # Non-conditional model: standard interface
            loss = self.model.compute_loss(x_source, x_target)
        
        return loss
    
    def _pre_training_hook(self):
        """
        Pre-training hook for VAE models to fit normalizers.
        """
        if self.model_type == 'vae' and hasattr(self.model, 'fit_normalizer'):
            if not self.model.normalization_fitted:
                print("\n" + "="*70)
                self.model.fit_normalizer(self.train_loader)
                print("="*70)
