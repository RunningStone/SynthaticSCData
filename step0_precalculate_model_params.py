#!/usr/bin/env python3
"""
Calculate and compare model parameters for different architectures
Reads configurations from YAML files
"""

import torch
import yaml
from pathlib import Path
from Model import (
    SchrodingerBridgeModel,
    MLPlus_SchrodingerBridgeModel,
    OptimalTransportModel,
    ConditionalVAEModel
)


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


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
    print("MODEL PARAMETER COMPARISON (FROM YAML CONFIGS)")
    print("="*80)
    print(f"\nInput dimension (n_hvg): {dimension}")
    print()
    
    # Build models from YAML configs
    sb_mlplus_arch = models_config['sb_mlplus']['architecture']
    sb_arch = models_config['sb']['architecture']
    ot_arch = models_config['ot']['architecture']
    vae_arch = models_config['vae']['architecture']
    
    configs = {
        'SB MLPlus (YAML)': {
            'model': MLPlus_SchrodingerBridgeModel(
                dimension=dimension,
                hidden_dim=sb_mlplus_arch['hidden_dim'],
                n_blocks=sb_mlplus_arch['n_blocks'],
                time_embedding_dim=sb_mlplus_arch['time_embedding_dim'],
                n_time_frequencies=sb_mlplus_arch['n_time_frequencies'],
                dropout=sb_mlplus_arch['dropout'],
                diffusion_coeff=sb_mlplus_arch['diffusion_coeff']
            ),
            'config': f"hidden_dim={sb_mlplus_arch['hidden_dim']}, n_blocks={sb_mlplus_arch['n_blocks']}"
        },
        'SB (YAML)': {
            'model': SchrodingerBridgeModel(
                dimension=dimension,
                hidden_dims=sb_arch['hidden_dims'],
                time_embedding_dim=sb_arch['time_embedding_dim'],
                dropout=sb_arch['dropout'],
                diffusion_coeff=sb_arch['diffusion_coeff']
            ),
            'config': f"hidden_dims={sb_arch['hidden_dims']}"
        },
        'OT (YAML)': {
            'model': OptimalTransportModel(
                dimension=dimension,
                hidden_dims=ot_arch['hidden_dims'],
                dropout=ot_arch['dropout'],
                use_residual=ot_arch['use_residual']
            ),
            'config': f"hidden_dims={ot_arch['hidden_dims']}"
        },
        'VAE (YAML)': {
            'model': ConditionalVAEModel(
                dimension=dimension,
                latent_dim=vae_arch['latent_dim'],
                hidden_dims=vae_arch['hidden_dims'],
                dropout=vae_arch['dropout'],
                beta=vae_arch['beta']
            ),
            'config': f"hidden_dims={vae_arch['hidden_dims']}, latent_dim={vae_arch['latent_dim']}"
        }
    }
    
    # Calculate parameters for current configs
    results = {}
    for name, info in configs.items():
        total, trainable = count_parameters(info['model'])
        results[name] = {
            'total': total,
            'trainable': trainable,
            'config': info['config']
        }
    
    # Print current configurations from YAML
    print("\n" + "="*80)
    print("CURRENT CONFIGURATIONS FROM YAML FILES")
    print("="*80)
    
    target_params = results['SB MLPlus (YAML)']['total']
    
    for name, res in results.items():
        diff_pct = (res['total'] - target_params) / target_params * 100 if 'MLPlus' not in name else 0
        print(f"\n{name}:")
        print(f"  Config: {res['config']}")
        print(f"  Total params: {res['total']:,}")
        if 'MLPlus' in name:
            print(f"  → TARGET (baseline)")
        else:
            print(f"  → Difference: {diff_pct:+.1f}% from SB MLPlus")
    
    # Summary
    print("\n" + "="*80)
    print("PARAMETER BALANCE SUMMARY")
    print("="*80)
    
    param_values = [res['total'] for name, res in results.items()]
    min_params = min(param_values)
    max_params = max(param_values)
    range_pct = (max_params - min_params) / min_params * 100
    
    print(f"\nParameter range: {min_params:,} - {max_params:,}")
    print(f"Range variation: {range_pct:.1f}%")
    
    if range_pct < 30:
        print("\n✅ GOOD: All models have similar parameter counts (within 30%)")
    elif range_pct < 50:
        print("\n⚠️  FAIR: Models have moderate parameter differences (30-50%)")
    else:
        print("\n❌ POOR: Models have large parameter differences (>50%)")
        print("   Consider adjusting hidden_dims to balance parameter counts")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
