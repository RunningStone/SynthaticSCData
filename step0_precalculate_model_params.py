#!/usr/bin/env python3
"""
Calculate and compare model parameters and GPU memory requirements
Reads configurations from YAML files and estimates training memory usage
"""

import torch
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from Model import (
    SchrodingerBridgeModel,
    MLPlus_SchrodingerBridgeModel,
    OptimalTransportModel,
    ConditionalVAEModel,
    BatchOTModel
)


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_memory_usage(
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
    
    # Bytes per parameter (float32 = 4 bytes, float16 = 2 bytes)
    param_bytes = 2 if mixed_precision else 4
    
    # 1. Model parameters
    model_memory = n_params * param_bytes / (1024 ** 2)  # MB
    
    # 2. Gradients (same size as parameters)
    gradient_memory = n_params * param_bytes / (1024 ** 2)  # MB
    
    # 3. Optimizer states
    if optimizer_type.lower() in ['adam', 'adamw']:
        # Adam stores 2 moments (m and v) for each parameter
        optimizer_memory = 2 * n_params * 4 / (1024 ** 2)  # Always float32
    elif optimizer_type.lower() == 'sgd':
        # SGD with momentum stores 1 momentum buffer
        optimizer_memory = n_params * 4 / (1024 ** 2)
    else:
        optimizer_memory = 0
    
    # 4. Activations (rough estimate)
    # Estimate based on typical activation memory for transformers/MLPs
    # Rule of thumb: ~2-4x model parameters for activations during forward pass
    activation_multiplier = 3.0  # Conservative estimate
    activation_memory = n_params * activation_multiplier * param_bytes / (1024 ** 2)
    
    # 5. Input/Output tensors
    # Input: (batch_size, input_dim)
    # Output: (batch_size, input_dim) for most models
    io_memory = 2 * batch_size * input_dim * param_bytes / (1024 ** 2)
    
    # 6. Temporary buffers (batch norm, dropout, etc.)
    # Rough estimate: 10-20% of model memory
    buffer_memory = model_memory * 0.15
    
    # Total memory
    total_memory = (
        model_memory + 
        gradient_memory + 
        optimizer_memory + 
        activation_memory + 
        io_memory + 
        buffer_memory
    )
    
    # Add 20% safety margin for PyTorch overhead
    total_with_overhead = total_memory * 1.2
    
    return {
        'model_params': model_memory,
        'gradients': gradient_memory,
        'optimizer_states': optimizer_memory,
        'activations': activation_memory,
        'io_tensors': io_memory,
        'buffers': buffer_memory,
        'subtotal': total_memory,
        'total_with_overhead': total_with_overhead,
        'n_parameters': n_params
    }


def format_memory(mb: float) -> str:
    """Format memory size in human-readable format"""
    if mb < 1024:
        return f"{mb:.1f} MB"
    else:
        return f"{mb/1024:.2f} GB"


def load_model_configs(config_file='configs/models_default.yaml', data_config_file='configs/data_EMT_Cook.yaml'):
    """Load model configurations from YAML file"""
    with open(config_file, 'r') as f:
        models_config = yaml.safe_load(f)
    
    with open(data_config_file, 'r') as f:
        data_config = yaml.safe_load(f)
    
    n_hvg = data_config['data_source']['n_hvg']
    return models_config, n_hvg


def main():
    # Load configurations from YAML
    models_config, dimension = load_model_configs()
    
    print("="*80)
    print("MODEL PARAMETER & MEMORY ANALYSIS (FROM YAML CONFIGS)")
    print("="*80)
    print(f"\nInput dimension (n_hvg): {dimension}")
    print()
    
    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {gpu_memory_gb:.2f} GB")
        print()
    
    # Build models from YAML configs
    sb_mlplus_arch = models_config['sb_mlplus']['architecture']
    sb_mlplus_train = models_config['sb_mlplus']['training']
    sb_arch = models_config['sb']['architecture']
    sb_train = models_config['sb']['training']
    ot_arch = models_config['ot']['architecture']
    ot_train = models_config['ot']['training']
    vae_arch = models_config['vae']['architecture']
    vae_train = models_config['vae']['training']
    batch_ot_arch = models_config['batch_ot']['architecture']
    batch_ot_train = models_config['batch_ot']['training']
    
    configs = {
        'sb_mlplus': {
            'name': 'SB MLPlus',
            'model': MLPlus_SchrodingerBridgeModel(
                dimension=dimension,
                hidden_dim=sb_mlplus_arch['hidden_dim'],
                n_blocks=sb_mlplus_arch['n_blocks'],
                time_embedding_dim=sb_mlplus_arch['time_embedding_dim'],
                n_time_frequencies=sb_mlplus_arch['n_time_frequencies'],
                dropout=sb_mlplus_arch['dropout'],
                diffusion_coeff=sb_mlplus_arch['diffusion_coeff']
            ),
            'config': f"hidden_dim={sb_mlplus_arch['hidden_dim']}, n_blocks={sb_mlplus_arch['n_blocks']}",
            'batch_size': sb_mlplus_train['batch_size'],
            'optimizer': sb_mlplus_train['optimizer']
        },
        'sb': {
            'name': 'SB',
            'model': SchrodingerBridgeModel(
                dimension=dimension,
                hidden_dims=sb_arch['hidden_dims'],
                time_embedding_dim=sb_arch['time_embedding_dim'],
                dropout=sb_arch['dropout'],
                diffusion_coeff=sb_arch['diffusion_coeff']
            ),
            'config': f"hidden_dims={sb_arch['hidden_dims']}",
            'batch_size': sb_train['batch_size'],
            'optimizer': sb_train['optimizer']
        },
        'ot': {
            'name': 'OT',
            'model': OptimalTransportModel(
                dimension=dimension,
                hidden_dims=ot_arch['hidden_dims'],
                dropout=ot_arch['dropout'],
                use_residual=ot_arch['use_residual']
            ),
            'config': f"hidden_dims={ot_arch['hidden_dims']}",
            'batch_size': ot_train['batch_size'],
            'optimizer': ot_train['optimizer']
        },
        'vae': {
            'name': 'VAE',
            'model': ConditionalVAEModel(
                dimension=dimension,
                n_timepoints=8,  # Maximum for setting2/4
                latent_dim=vae_arch['latent_dim'],
                time_embedding_dim=vae_arch['time_embedding_dim'],
                hidden_dims=vae_arch['hidden_dims'],
                dropout=vae_arch['dropout'],
                beta=vae_arch['beta'],
                mmd_weight=vae_arch['mmd_weight'],
                mmd_kernel=vae_arch['mmd_kernel'],
                mmd_bandwidth=vae_arch['mmd_bandwidth']
            ),
            'config': f"hidden_dims={vae_arch['hidden_dims']}, latent_dim={vae_arch['latent_dim']}",
            'batch_size': vae_train['batch_size'],
            'optimizer': vae_train['optimizer']
        },
        'batch_ot': {
            'name': 'Batch OT',
            'model': BatchOTModel(
                dimension=dimension,
                n_timepoints=8,  # Maximum for setting2/4
                time_labels=['0d', '8h', '1d', '3d', '7d', '8h_rm', '1d_rm', '3d_rm'],
                hidden_dims=batch_ot_arch['hidden_dims'],
                dropout=batch_ot_arch['dropout'],
                use_residual=batch_ot_arch['use_residual']
            ),
            'config': f"hidden_dims={batch_ot_arch['hidden_dims']} × 7 OT models",
            'batch_size': batch_ot_train['batch_size'],
            'optimizer': batch_ot_train['optimizer']
        }
    }
    
    # Calculate parameters and memory for current configs
    results = {}
    for model_key, info in configs.items():
        total, trainable = count_parameters(info['model'])
        memory_est = estimate_memory_usage(
            model=info['model'],
            batch_size=info['batch_size'],
            input_dim=dimension,
            optimizer_type=info['optimizer'],
            mixed_precision=False
        )
        results[model_key] = {
            'name': info['name'],
            'total': total,
            'trainable': trainable,
            'config': info['config'],
            'batch_size': info['batch_size'],
            'memory': memory_est
        }
    
    # Print current configurations from YAML
    print("\n" + "="*80)
    print("MODEL CONFIGURATIONS & MEMORY REQUIREMENTS")
    print("="*80)
    
    target_params = results['sb_mlplus']['total']
    
    for model_key, res in results.items():
        diff_pct = (res['total'] - target_params) / target_params * 100 if model_key != 'sb_mlplus' else 0
        mem = res['memory']
        
        print(f"\n{res['name']}:")
        print(f"  Config: {res['config']}")
        print(f"  Batch size: {res['batch_size']}")
        print(f"  Total params: {res['total']:,} ({res['total']/1e6:.2f}M)")
        if model_key == 'sb_mlplus':
            print(f"  → TARGET (baseline)")
        else:
            print(f"  → Difference: {diff_pct:+.1f}% from SB MLPlus")
        
        print(f"\n  Memory Breakdown:")
        print(f"    Model parameters:  {format_memory(mem['model_params'])}")
        print(f"    Gradients:         {format_memory(mem['gradients'])}")
        print(f"    Optimizer states:  {format_memory(mem['optimizer_states'])}")
        print(f"    Activations:       {format_memory(mem['activations'])}")
        print(f"    I/O tensors:       {format_memory(mem['io_tensors'])}")
        print(f"    Buffers:           {format_memory(mem['buffers'])}")
        print(f"    " + "-" * 50)
        print(f"    Subtotal:          {format_memory(mem['subtotal'])}")
        print(f"    With overhead:     {format_memory(mem['total_with_overhead'])}")
        
        # GPU fit check
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            if mem['total_with_overhead'] < gpu_memory_mb * 0.8:  # 80% threshold
                print(f"    ✅ Fits in GPU ({mem['total_with_overhead']/gpu_memory_mb*100:.1f}% usage)")
            else:
                print(f"    ⚠️  May not fit in GPU ({mem['total_with_overhead']/gpu_memory_mb*100:.1f}% usage)")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    param_values = [res['total'] for key, res in results.items()]
    memory_values = [res['memory']['total_with_overhead'] for key, res in results.items()]
    
    min_params = min(param_values)
    max_params = max(param_values)
    range_pct = (max_params - min_params) / min_params * 100
    
    min_memory = min(memory_values)
    max_memory = max(memory_values)
    
    print(f"\nParameter Statistics:")
    print(f"  Range: {min_params:,} - {max_params:,}")
    print(f"  Variation: {range_pct:.1f}%")
    
    if range_pct < 30:
        print("  ✅ GOOD: All models have similar parameter counts (within 30%)")
    elif range_pct < 50:
        print("  ⚠️  FAIR: Models have moderate parameter differences (30-50%)")
    else:
        print("  ❌ POOR: Models have large parameter differences (>50%)")
        print("     Consider adjusting hidden_dims to balance parameter counts")
    
    print(f"\nMemory Statistics:")
    print(f"  Range: {format_memory(min_memory)} - {format_memory(max_memory)}")
    print(f"  Peak usage: {format_memory(max_memory)}")
    
    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"  GPU capacity: {format_memory(gpu_memory_mb)}")
        print(f"  Peak/Capacity: {max_memory/gpu_memory_mb*100:.1f}%")
        
        if max_memory < gpu_memory_mb * 0.5:
            print("  ✅ All models fit comfortably in GPU (<50% usage)")
        elif max_memory < gpu_memory_mb * 0.8:
            print("  ✅ All models should fit in GPU (<80% usage)")
        else:
            print("  ⚠️  Some models may have memory issues (>80% usage)")
    
    # Model ranking by memory
    print(f"\nModels ranked by memory usage:")
    sorted_models = sorted(results.items(), key=lambda x: x[1]['memory']['total_with_overhead'])
    for i, (key, res) in enumerate(sorted_models, 1):
        mem = res['memory']['total_with_overhead']
        print(f"  {i}. {res['name']:<15} {format_memory(mem):>10} (batch_size={res['batch_size']})")
    
    print("\n" + "="*80)
    print("\nNote: Memory estimates are conservative and include 20% overhead.")
    print("Actual usage may vary based on PyTorch version and CUDA settings.")
    print("="*80)


if __name__ == '__main__':
    main()
