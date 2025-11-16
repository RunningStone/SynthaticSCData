#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Setting Visualization and Comparison

Comprehensive visualization and comparison across multiple experimental settings.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import warnings
import json
import yaml
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import phate
from metric_learn import LMNN

import sys
sys.path.append(str(Path(__file__).parent.parent))
from Data import create_default_emt_data_loader
from Model import (
    SchrodingerBridgeModel,
    MLPlus_SchrodingerBridgeModel,
    OptimalTransportModel,
    ConditionalVAEModel,
    BatchOTModel
)


class MultiSettingVisualizer:
    """Visualize and compare generation results across multiple experimental settings"""
    
    def __init__(self, output_dir: str = './visualization_outputs', device: str = 'cuda', random_seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        self.loader = None
        self.X_original = None
        self.y_original = None
        self.time_labels = None
        self.generated_data = {}  # {setting_name-model_name: samples}
        self.evaluation_metrics = {}  # {setting_name-model_name: metrics_dict}
        self.phate_op = None
        self.lmnn_pca_scaler = None
        self.lmnn_op = None
        self.pca_op = None
        self.phate_embeddings = {}
        self.lmnn_pca_embeddings = {}
    
    def load_experiment_configs(self, config_paths: List[str]) -> Dict[str, Dict]:
        """Load multiple experiment configurations"""
        print("="*80)
        print("Loading Experiment Configurations")
        print("="*80)
        
        configs = {}
        for config_path in config_paths:
            config_path = Path(config_path)
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            setting_name = config['experiment']['name'].split('_')[-1]
            if not setting_name.startswith('Setting'):
                setting_name = config_path.parent.name.replace('EMT_', '')
            
            configs[setting_name] = config
            print(f"  ✓ Loaded {setting_name}: {config['experiment']['name']}")
            print(f"    Models: {list(config['models'].keys())}")
        
        print("="*80)
        return configs
    
    def aggregate_model_configs(self, configs: Dict[str, Dict]) -> Dict[str, Dict]:
        """Aggregate model configurations from all settings"""
        print("\n" + "="*80)
        print("Aggregating Model Configurations")
        print("="*80)
        
        aggregated = {}
        
        for setting_name, config in configs.items():
            output_dir = Path(config['settings']['output_dir'])
            checkpoint_subdir = config['settings']['subdirs']['checkpoints']
            models_config = config['models']
            n_hvg = config['data_source']['n_hvg']
            
            for model_name in models_config.keys():
                model_arch = models_config[model_name]['architecture']
                
                # BatchOT uses final_model.pt instead of best_model.pt
                if model_name == 'batch_ot':
                    checkpoint_path = output_dir / checkpoint_subdir / model_name / 'final_model.pt'
                else:
                    checkpoint_path = output_dir / checkpoint_subdir / model_name / 'best_model.pt'
                
                model_kwargs = {'dimension': n_hvg}
                
                if model_name == 'sb':
                    model_kwargs.update({
                        'hidden_dims': model_arch['hidden_dims'],
                        'time_embedding_dim': model_arch['time_embedding_dim'],
                        'dropout': model_arch['dropout'],
                        'diffusion_coeff': model_arch['diffusion_coeff']
                    })
                elif model_name == 'sb_mlplus':
                    model_kwargs.update({
                        'hidden_dim': model_arch['hidden_dim'],
                        'n_blocks': model_arch['n_blocks'],
                        'time_embedding_dim': model_arch['time_embedding_dim'],
                        'n_time_frequencies': model_arch['n_time_frequencies'],
                        'dropout': model_arch['dropout'],
                        'diffusion_coeff': model_arch['diffusion_coeff']
                    })
                elif model_name == 'ot':
                    model_kwargs.update({
                        'hidden_dims': model_arch['hidden_dims'],
                        'activation': model_arch['activation'],
                        'dropout': model_arch['dropout'],
                        'use_residual': model_arch.get('use_residual', True)
                    })
                elif model_name == 'vae':
                    # ConditionalVAE needs n_timepoints parameter
                    n_timepoints = len(config['data_source']['time_labels_order'])
                    model_kwargs.update({
                        'n_timepoints': n_timepoints,
                        'hidden_dims': model_arch['hidden_dims'],
                        'latent_dim': model_arch['latent_dim'],
                        'activation': model_arch['activation'],
                        'dropout': model_arch['dropout'],
                        'beta': model_arch['beta'],
                        'time_embedding_dim': model_arch.get('time_embedding_dim', 64),
                        'mmd_weight': model_arch.get('mmd_weight', 1.0),
                        'mmd_kernel': model_arch.get('mmd_kernel', 'rbf'),
                        'mmd_bandwidth': model_arch.get('mmd_bandwidth', 1.0)
                    })
                elif model_name == 'batch_ot':
                    # BatchOT needs n_timepoints and time_labels
                    time_labels_order = config['data_source']['time_labels_order']
                    n_timepoints = len(time_labels_order)
                    model_kwargs.update({
                        'n_timepoints': n_timepoints,
                        'time_labels': time_labels_order,
                        'hidden_dims': model_arch['hidden_dims'],
                        'activation': model_arch['activation'],
                        'dropout': model_arch['dropout'],
                        'use_residual': model_arch.get('use_residual', True)
                    })
                
                if checkpoint_path.exists():
                    full_model_name = f"{setting_name}-{model_name}"
                    aggregated[full_model_name] = {
                        'type': model_name,
                        'checkpoint_path': checkpoint_path,
                        'model_kwargs': model_kwargs,
                        'setting_name': setting_name,
                        'model_name': model_name
                    }
                    print(f"  ✓ {full_model_name}")
                else:
                    print(f"  ⚠️  Skipping {setting_name}-{model_name}: checkpoint not found")
        
        print(f"\n  Total models: {len(aggregated)}")
        print("="*80)
        return aggregated
    
    def load_evaluation_metrics(self, configs: Dict[str, Dict]):
        """Load evaluation metrics from results.json files"""
        print("\n" + "="*80)
        print("Loading Evaluation Metrics")
        print("="*80)
        
        for setting_name, config in configs.items():
            output_dir = Path(config['settings']['output_dir'])
            results_path = output_dir / 'results.json'
            
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                for model_name, model_results in results.items():
                    if 'evaluation' in model_results:
                        full_model_name = f"{setting_name}-{model_name}"
                        self.evaluation_metrics[full_model_name] = model_results['evaluation']
                        print(f"  ✓ {full_model_name}: {len(model_results['evaluation'])} metrics")
            else:
                print(f"  ⚠️  No results.json found for {setting_name}")
        
        print("="*80)
