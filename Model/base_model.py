#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Model for Cell State Transition

Defines the abstract interface that all trajectory generation models must implement.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple


class BaseTrajectoryModel(nn.Module, ABC):
    """
    Abstract base class for all trajectory generation models.
    
    All models learn to generate cell state trajectories given:
    - Initial state x_0
    - Complete time series (ordered list of timepoints)
    - Specific source and target timepoint indices
    
    This unified interface allows different model types (SB, OT, VAE, etc.)
    to be used interchangeably in the training and evaluation pipeline.
    """
    
    def __init__(self, dimension: int, **kwargs):
        """
        Args:
            dimension: State space dimension (number of genes/features)
            **kwargs: Model-specific parameters
        """
        super().__init__()
        self.dimension = dimension
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the model.
        
        The specific signature depends on the model type:
        - SB models: forward(x, t) -> drift field
        - OT models: forward(x) -> transported state
        - VAE models: forward(x_0, t_target) -> (x_0_recon, x_T_sample, mu, logvar)
        
        Returns:
            Model-specific output
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
        *args,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        """
        Compute training loss.
        
        Args:
            x_source: Source state samples (batch_size, d)
            x_target: Target state samples (batch_size, d)
            *args: Model-specific positional arguments (e.g., time indices, dt)
            **kwargs: Model-specific keyword arguments
            
        Returns:
            loss: Scalar loss tensor
            OR
            (loss, loss_dict): Loss and dictionary of loss components for logging
        """
        pass
    
    @abstractmethod
    def generate_trajectory(
        self,
        x_0: torch.Tensor,
        time_grid: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate trajectory from initial state.
        
        This is the core inference method that all models must implement.
        Given an initial state and a time grid, generate the predicted
        cell states at each time point.
        
        Args:
            x_0: Initial state (batch_size, d)
            time_grid: Time points (n_time,), typically normalized to [0, 1]
            **kwargs: Model-specific parameters:
                - For SB: method='euler'|'deterministic', create_graph=False
                - For OT: method='linear'
                - For VAE: t_source_idx, t_target_idx, method='latent_interpolation'
                - For BatchOT: method='sequential'|'deterministic'
            
        Returns:
            trajectory: Predicted states (batch_size, n_time, d)
        """
        pass
    
    def get_model_info(self) -> dict:
        """
        Get model information for logging and debugging.
        
        Returns:
            Dictionary containing model metadata
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_class': self.__class__.__name__,
            'dimension': self.dimension,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'memory_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
        }


class TimeConditionedModel(BaseTrajectoryModel):
    """
    Base class for models that require time conditioning.
    
    This includes models like VAE and BatchOT that need explicit
    time indices rather than continuous time values.
    """
    
    def __init__(self, dimension: int, n_timepoints: int, time_labels: List[str], **kwargs):
        """
        Args:
            dimension: State space dimension
            n_timepoints: Number of discrete time points
            time_labels: List of time point labels (e.g., ['0d', '8h', '1d', '3d', '7d'])
            **kwargs: Model-specific parameters
        """
        super().__init__(dimension, **kwargs)
        self.n_timepoints = n_timepoints
        self.time_labels = time_labels
    
    def get_model_info(self) -> dict:
        """Extended model info with time conditioning details"""
        info = super().get_model_info()
        info.update({
            'n_timepoints': self.n_timepoints,
            'time_labels': self.time_labels,
            'time_conditioned': True
        })
        return info


class ContinuousTimeModel(BaseTrajectoryModel):
    """
    Base class for models that operate in continuous time.
    
    This includes models like Schrödinger Bridge and OT that
    can generate trajectories at arbitrary time points.
    """
    
    def __init__(self, dimension: int, **kwargs):
        """
        Args:
            dimension: State space dimension
            **kwargs: Model-specific parameters
        """
        super().__init__(dimension, **kwargs)
    
    def get_model_info(self) -> dict:
        """Extended model info with continuous time flag"""
        info = super().get_model_info()
        info.update({
            'time_conditioned': False,
            'continuous_time': True
        })
        return info
