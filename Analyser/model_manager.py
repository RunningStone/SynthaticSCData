#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Manager - Model Loading and Inference

Handles model operations including:
- Loading model checkpoints
- Instantiating models from configs
- Generating trajectories
- Batch generation
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelManager:
    """
    Manages model loading and inference operations.
    
    Responsibilities:
    - Load model checkpoints
    - Instantiate models from configs
    - Generate trajectories
    - Batch generation across multiple models
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize model manager.
        
        Args:
            device: Device for computation ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
    
    def load_model(
        self,
        model_type: str,
        checkpoint_path: Path,
        model_kwargs: Dict,
        device: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Load model from checkpoint.
        
        Args:
            model_type: Type of model ('sb', 'sb_mlplus', 'ot', 'vae', 'batch_ot')
            checkpoint_path: Path to checkpoint file
            model_kwargs: Keyword arguments for model initialization
            device: Device to load model on (uses self.device if None)
        
        Returns:
            Loaded model
        """
        from Model import (
            SchrodingerBridgeModel,
            MLPlus_SchrodingerBridgeModel,
            OptimalTransportModel,
            ConditionalVAEModel,
            BatchOTModel
        )
        
        device = device or self.device
        
        # Instantiate model
        if model_type == 'sb':
            model = SchrodingerBridgeModel(**model_kwargs)
        elif model_type == 'sb_mlplus':
            model = MLPlus_SchrodingerBridgeModel(**model_kwargs)
        elif model_type == 'ot':
            model = OptimalTransportModel(**model_kwargs)
        elif model_type == 'vae':
            model = ConditionalVAEModel(**model_kwargs)
        elif model_type == 'batch_ot':
            model = BatchOTModel(**model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Move to device
        model = model.to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Special handling for VAE
        if model_type == 'vae' and hasattr(model, 'normalization_fitted'):
            model.normalization_fitted = True
        
        model.eval()
        
        return model
    
    def generate_trajectory(
        self,
        model: torch.nn.Module,
        x_source: torch.Tensor,
        time_grid: torch.Tensor,
        model_type: str,
        source_time_idx: Optional[int] = None,
        target_time_idx: Optional[int] = None,
        method: str = 'deterministic'
    ) -> np.ndarray:
        """
        Generate trajectory from source to target.
        
        Args:
            model: Model instance
            x_source: Source data (batch_size, dim)
            time_grid: Time grid for trajectory (n_time,)
            model_type: Type of model
            source_time_idx: Source time index (for VAE)
            target_time_idx: Target time index (for VAE)
            method: Generation method
        
        Returns:
            Generated trajectory (batch_size, n_time, dim)
        """
        with torch.no_grad():
            if model_type in ['sb', 'sb_mlplus']:
                trajectory = model.generate_trajectory(
                    x_source, time_grid, method=method
                )
            elif model_type == 'ot':
                trajectory = model.generate_trajectory(
                    x_source, time_grid, method=method
                )
            elif model_type == 'vae':
                if source_time_idx is None or target_time_idx is None:
                    raise ValueError("VAE requires source_time_idx and target_time_idx")
                trajectory = model.generate_trajectory(
                    x_source, time_grid, source_time_idx, target_time_idx, method=method
                )
            elif model_type == 'batch_ot':
                trajectory = model.generate_trajectory(
                    x_source, time_grid, method=method
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        return trajectory.cpu().numpy()
    
    def batch_generate(
        self,
        models_dict: Dict[str, Tuple[torch.nn.Module, str]],
        x_source: torch.Tensor,
        time_grid: torch.Tensor,
        source_time_idx: Optional[int] = None,
        target_time_idx: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate trajectories for multiple models.
        
        Args:
            models_dict: Dict mapping model names to (model, model_type) tuples
            x_source: Source data (batch_size, dim)
            time_grid: Time grid for trajectory (n_time,)
            source_time_idx: Source time index (for VAE)
            target_time_idx: Target time index (for VAE)
        
        Returns:
            Dict mapping model names to generated trajectories
        """
        results = {}
        
        for model_name, (model, model_type) in models_dict.items():
            trajectory = self.generate_trajectory(
                model, x_source, time_grid, model_type,
                source_time_idx, target_time_idx
            )
            results[model_name] = trajectory
        
        return results
    
    def extract_target_samples(
        self,
        trajectories_dict: Dict[str, np.ndarray],
        target_idx: int = -1
    ) -> Dict[str, np.ndarray]:
        """
        Extract target timepoint samples from trajectories.
        
        Args:
            trajectories_dict: Dict mapping model names to trajectories
            target_idx: Index of target timepoint in trajectory
        
        Returns:
            Dict mapping model names to target samples
        """
        results = {}
        
        for model_name, trajectory in trajectories_dict.items():
            # trajectory shape: (batch_size, n_time, dim)
            results[model_name] = trajectory[:, target_idx, :]
        
        return results
