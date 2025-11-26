#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Parameter Analyzer for Resource Estimation

This module provides functionality to analyze model architectures and estimate
GPU memory requirements for training different generative models.

Author: Auto-generated
Date: 2024-11-24
"""

import torch
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json
from .base_visualizer import BaseVisualizer


class ModelParamAnalyzer(BaseVisualizer):
    """
    Analyzer for computing model parameters and memory requirements
    
    This class instantiates models from configurations, counts parameters,
    and estimates GPU memory usage for training.
    """
    
    def __init__(
        self,
        output_dir: str,
        device: str = 'cpu',
        seed: int = 42
    ):
        """
        Initialize ModelParamAnalyzer
        
        Args:
            output_dir: Directory to save analysis results
            device: Device for model instantiation
            seed: Random seed for reproducibility
        """
        super().__init__(output_dir, device, seed)
        self.log("ModelParamAnalyzer initialized")
    
    def count_parameters(self, model: torch.nn.Module) -> Tuple[int, int]:
        """
        Count total and trainable parameters
        
        Args:
            model: PyTorch model
            
        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    def estimate_memory_usage(
        self,
        model: torch.nn.Module,
        batch_size: int,
        input_dim: int,
        optimizer_type: str = 'adam',
        mixed_precision: bool = False
    ) -> Dict[str, float]:
        """
        Estimate GPU memory usage for training
        
        Args:
            model: PyTorch model
            batch_size: Training batch size
            input_dim: Input dimension
            optimizer_type: Optimizer type ('adam', 'sgd', 'adamw')
            mixed_precision: Whether using mixed precision training
            
        Returns:
            Dictionary with memory estimates in MB
        """
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        
        # Bytes per parameter (fp32 or fp16)
        bytes_per_param = 2 if mixed_precision else 4
        
        # 1. Model parameters memory
        model_memory = n_params * bytes_per_param / (1024 ** 2)
        
        # 2. Gradients memory (same size as parameters)
        gradient_memory = model_memory
        
        # 3. Optimizer state memory
        if optimizer_type.lower() in ['adam', 'adamw']:
            # Adam stores 2 states per parameter (momentum and variance)
            optimizer_memory = 2 * n_params * 4 / (1024 ** 2)  # Always fp32
        elif optimizer_type.lower() == 'sgd':
            # SGD with momentum stores 1 state per parameter
            optimizer_memory = n_params * 4 / (1024 ** 2)
        else:
            optimizer_memory = 0
        
        # 4. Activation memory (rough estimate)
        # Assume activations are ~2x the input size per layer
        # This is a rough heuristic
        activation_memory = batch_size * input_dim * bytes_per_param * 2 / (1024 ** 2)
        
        # 5. Workspace memory (for cuDNN, etc.)
        workspace_memory = 100  # MB, rough estimate
        
        # Total memory
        total_memory = (
            model_memory + 
            gradient_memory + 
            optimizer_memory + 
            activation_memory + 
            workspace_memory
        )
        
        return {
            'model_memory_mb': round(model_memory, 2),
            'gradient_memory_mb': round(gradient_memory, 2),
            'optimizer_memory_mb': round(optimizer_memory, 2),
            'activation_memory_mb': round(activation_memory, 2),
            'workspace_memory_mb': round(workspace_memory, 2),
            'total_memory_mb': round(total_memory, 2),
            'total_memory_gb': round(total_memory / 1024, 2)
        }
    
    def analyze_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        model_config: Dict,
        batch_size: int = 256,
        input_dim: int = 100,
        optimizer_type: str = 'adam',
        mixed_precision: bool = False
    ) -> Dict:
        """
        Perform complete analysis of a single model
        
        Args:
            model: PyTorch model instance
            model_name: Name of the model
            model_config: Configuration dict used to create the model
            batch_size: Training batch size
            input_dim: Input dimension
            optimizer_type: Optimizer type
            mixed_precision: Whether using mixed precision
            
        Returns:
            Dict with analysis results
        """
        self.log("\n" + "="*80)
        self.log(f"Analyzing Model: {model_name}")
        self.log("="*80)
        
        # Count parameters
        total_params, trainable_params = self.count_parameters(model)
        
        self.log(f"\nParameter Count:")
        self.log(f"  Total parameters: {total_params:,}")
        self.log(f"  Trainable parameters: {trainable_params:,}")
        self.log(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Estimate memory
        memory_est = self.estimate_memory_usage(
            model=model,
            batch_size=batch_size,
            input_dim=input_dim,
            optimizer_type=optimizer_type,
            mixed_precision=mixed_precision
        )
        
        self.log(f"\nMemory Estimation (batch_size={batch_size}):")
        self.log(f"  Model parameters: {memory_est['model_memory_mb']:.2f} MB")
        self.log(f"  Gradients: {memory_est['gradient_memory_mb']:.2f} MB")
        self.log(f"  Optimizer states: {memory_est['optimizer_memory_mb']:.2f} MB")
        self.log(f"  Activations: {memory_est['activation_memory_mb']:.2f} MB")
        self.log(f"  Workspace: {memory_est['workspace_memory_mb']:.2f} MB")
        self.log(f"  Total: {memory_est['total_memory_mb']:.2f} MB ({memory_est['total_memory_gb']:.2f} GB)")
        
        return {
            'model_name': model_name,
            'config': model_config,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params,
            'memory_estimation': memory_est,
            'training_config': {
                'batch_size': batch_size,
                'input_dim': input_dim,
                'optimizer_type': optimizer_type,
                'mixed_precision': mixed_precision
            }
        }
    
    def load_model_from_config(
        self,
        model_type: str,
        model_config: Dict
    ) -> torch.nn.Module:
        """
        Load model from configuration
        
        Args:
            model_type: Type of model ('sb', 'sb_mlplus', 'ot', 'vae', 'batch_ot')
            model_config: Model configuration dict
            
        Returns:
            Instantiated model
        """
        # Import here to avoid circular dependencies
        from Model import (
            SchrodingerBridgeModel,
            MLPlus_SchrodingerBridgeModel,
            OptimalTransportModel,
            ConditionalVAEModel,
            BatchOTModel
        )
        
        model_type = model_type.lower()
        
        if model_type == 'sb':
            model = SchrodingerBridgeModel(**model_config)
        elif model_type == 'sb_mlplus':
            model = MLPlus_SchrodingerBridgeModel(**model_config)
        elif model_type == 'ot':
            model = OptimalTransportModel(**model_config)
        elif model_type == 'vae':
            model = ConditionalVAEModel(**model_config)
        elif model_type == 'batch_ot':
            model = BatchOTModel(**model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def compare_models(
        self,
        model_configs: Dict[str, Dict],
        batch_size: int = 256,
        input_dim: int = 100,
        optimizer_type: str = 'adam',
        mixed_precision: bool = False
    ) -> Dict[str, Dict]:
        """
        Compare multiple models
        
        Args:
            model_configs: Dict mapping model names to (model_type, config) tuples
            batch_size: Training batch size
            input_dim: Input dimension
            optimizer_type: Optimizer type
            mixed_precision: Whether using mixed precision
            
        Returns:
            Dict mapping model names to analysis results
        """
        self.log("\n" + "="*80)
        self.log("Comparing Multiple Models")
        self.log("="*80)
        
        results = {}
        
        for model_name, (model_type, config) in model_configs.items():
            try:
                # Instantiate model
                model = self.load_model_from_config(model_type, config)
                model = model.to(self.device)
                
                # Analyze model
                result = self.analyze_model(
                    model=model,
                    model_name=model_name,
                    model_config=config,
                    batch_size=batch_size,
                    input_dim=input_dim,
                    optimizer_type=optimizer_type,
                    mixed_precision=mixed_precision
                )
                
                results[model_name] = result
                
                # Clean up
                del model
                if self.device.startswith('cuda'):
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.log(f"\n❌ Error analyzing {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def generate_comparison_table(
        self,
        results: Dict[str, Dict]
    ) -> str:
        """
        Generate a formatted comparison table
        
        Args:
            results: Dict of analysis results from compare_models
            
        Returns:
            Formatted table string
        """
        # Header
        table = "\n" + "="*120 + "\n"
        table += "Model Comparison Summary\n"
        table += "="*120 + "\n\n"
        
        # Column headers
        table += f"{'Model':<20} {'Total Params':>15} {'Trainable':>15} {'Memory (MB)':>15} {'Memory (GB)':>15}\n"
        table += "-"*120 + "\n"
        
        # Rows
        for model_name, result in results.items():
            if 'error' in result:
                table += f"{model_name:<20} {'ERROR':>15} {'ERROR':>15} {'ERROR':>15} {'ERROR':>15}\n"
            else:
                total_params = result['total_params']
                trainable = result['trainable_params']
                memory_mb = result['memory_estimation']['total_memory_mb']
                memory_gb = result['memory_estimation']['total_memory_gb']
                
                table += f"{model_name:<20} {total_params:>15,} {trainable:>15,} {memory_mb:>15.2f} {memory_gb:>15.2f}\n"
        
        table += "="*120 + "\n"
        
        return table
    
    def save_results(
        self,
        results: Dict[str, Dict],
        prefix: str = "model_param_analysis"
    ):
        """
        Save analysis results to files
        
        Args:
            results: Dict of analysis results
            prefix: Prefix for output filenames
        """
        # Save detailed JSON
        json_path = self.output_dir / f"{prefix}_detailed.json"
        
        # Convert to JSON-serializable format
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                k: v for k, v in result.items() 
                if k != 'model'  # Don't try to serialize the model object
            }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        self.log(f"\n✓ Saved detailed results: {json_path}")
        
        # Save comparison table
        table = self.generate_comparison_table(results)
        table_path = self.output_dir / f"{prefix}_comparison.txt"
        with open(table_path, 'w') as f:
            f.write(table)
        self.log(f"✓ Saved comparison table: {table_path}")
        
        # Print table to console
        self.log(table)
    
    def run_full_analysis(
        self,
        model_configs: Dict[str, Tuple[str, Dict]],
        batch_size: int = 256,
        input_dim: int = 100,
        optimizer_type: str = 'adam',
        mixed_precision: bool = False
    ) -> Dict[str, Dict]:
        """
        Run complete model parameter analysis pipeline
        
        Args:
            model_configs: Dict mapping model names to (model_type, config) tuples
            batch_size: Training batch size
            input_dim: Input dimension
            optimizer_type: Optimizer type
            mixed_precision: Whether using mixed precision
            
        Returns:
            Dict with analysis results for all models
        """
        self.log("\n" + "="*80)
        self.log("Starting Full Model Parameter Analysis")
        self.log("="*80)
        
        # Compare all models
        results = self.compare_models(
            model_configs=model_configs,
            batch_size=batch_size,
            input_dim=input_dim,
            optimizer_type=optimizer_type,
            mixed_precision=mixed_precision
        )
        
        # Save results
        self.save_results(results)
        
        self.log("\n" + "="*80)
        self.log("✓ Model Parameter Analysis Complete")
        self.log("="*80)
        
        return results
