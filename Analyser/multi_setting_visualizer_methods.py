#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Setting Visualizer - Additional Methods

This file contains the remaining methods for MultiSettingVisualizer class.
Import and extend the main class with these methods.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import phate
from metric_learn import LMNN


def load_and_sample_data(self, file_path: str, n_hvg: int, n_samples_per_timepoint: int = 500):
    """Load data and sample from test set"""
    print("\n" + "="*80)
    print("Loading and Sampling Data")
    print("="*80)
    
    from Data import create_default_emt_data_loader
    self.loader = create_default_emt_data_loader(file_path=file_path, n_hvg=n_hvg)
    self.loader.load_and_analyze()
    self.loader.validate_biology_split()
    
    X = self.loader.adata_hvg.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    time_to_idx = {label: idx for idx, label in enumerate(self.loader.time_label_order)}
    y = np.array([time_to_idx[t] for t in self.loader.adata_hvg.obs[self.loader.obs_time_column]])
    
    X_samples_list = []
    y_samples_list = []
    
    for time_label in self.loader.time_label_order:
        time_idx = time_to_idx[time_label]
        time_mask = (y == time_idx)
        test_time_mask = time_mask & self.loader.test_mask
        test_indices = np.where(test_time_mask)[0]
        
        if len(test_indices) > n_samples_per_timepoint:
            sampled_indices = np.random.choice(test_indices, n_samples_per_timepoint, replace=False)
        else:
            sampled_indices = test_indices
        
        X_samples_list.append(X[sampled_indices])
        y_samples_list.append(y[sampled_indices])
        print(f"  {time_label}: sampled {len(sampled_indices)} cells from test set")
    
    self.X_original = np.vstack(X_samples_list)
    self.y_original = np.concatenate(y_samples_list)
    self.time_labels = self.loader.time_label_order
    
    print(f"\n✓ Total sampled: {self.X_original.shape[0]} cells × {self.X_original.shape[1]} genes")
    print("="*80)


def load_models_and_generate(self, model_configs: Dict[str, Dict], n_generate_per_model: int = 500):
    """Load trained models and generate target timepoint samples"""
    print("\n" + "="*80)
    print("Loading Models and Generating Samples")
    print("="*80)
    
    from Model import SchrodingerBridgeModel, MLPlus_SchrodingerBridgeModel, OptimalTransportModel, ConditionalVAEModel
    
    first_time_idx = 0
    last_time_idx = len(self.time_labels) - 1
    
    source_mask = (self.y_original == first_time_idx)
    source_samples = self.X_original[source_mask]
    
    if len(source_samples) > n_generate_per_model:
        source_indices = np.random.choice(len(source_samples), n_generate_per_model, replace=False)
        source_samples = source_samples[source_indices]
    
    print(f"\nSource samples ({self.time_labels[first_time_idx]}): {source_samples.shape[0]} cells")
    print(f"Target timepoint: {self.time_labels[last_time_idx]}\n")
    
    for full_model_name, config in model_configs.items():
        print(f"Processing {full_model_name}...")
        
        model_type = config['type']
        checkpoint_path = Path(config['checkpoint_path'])
        model_kwargs = config.get('model_kwargs', {})
        
        if model_type == 'sb':
            model = SchrodingerBridgeModel(**model_kwargs).to(self.device)
        elif model_type == 'sb_mlplus':
            model = MLPlus_SchrodingerBridgeModel(**model_kwargs).to(self.device)
        elif model_type == 'ot':
            model = OptimalTransportModel(**model_kwargs).to(self.device)
        elif model_type == 'vae':
            model = ConditionalVAEModel(**model_kwargs).to(self.device)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if model_type == 'vae' and hasattr(model, 'normalization_fitted'):
            model.normalization_fitted = True
        
        print(f"  ✓ Loaded checkpoint")
        
        model.eval()
        source_tensor = torch.FloatTensor(source_samples).to(self.device)
        time_grid = torch.tensor([0.0, 1.0], device=self.device)
        
        if model_type in ['sb', 'sb_mlplus']:
            trajectory = model.generate_trajectory(source_tensor, time_grid, method='deterministic')
            generated = trajectory[:, -1, :].detach()
        elif model_type in ['ot', 'vae']:
            with torch.no_grad():
                trajectory = model.generate_trajectory(source_tensor, time_grid, method='deterministic')
                generated = trajectory[:, -1, :]
        
        generated_np = generated.cpu().numpy()
        self.generated_data[full_model_name] = generated_np
        print(f"  ✓ Generated {generated_np.shape[0]} samples")
    
    print("\n" + "="*80)


def compute_embeddings(self):
    """Compute PHATE and LMNN+PCA embeddings"""
    print("\n" + "="*80)
    print("Computing Embeddings")
    print("="*80)
    
    print("\nComputing PHATE embeddings...")
    self.phate_op = phate.PHATE(n_components=2, knn=5, decay=40, n_jobs=-1, 
                                 random_state=self.random_seed, verbose=0)
    self.phate_embeddings['original'] = self.phate_op.fit_transform(self.X_original)
    print(f"  ✓ PHATE embeddings computed: {self.phate_embeddings['original'].shape}")
    
    for model_name, generated_data in self.generated_data.items():
        self.phate_embeddings[model_name] = self.phate_op.transform(generated_data)
        print(f"  ✓ Projected {model_name}")
    
    print("\nComputing LMNN+PCA embeddings...")
    self.lmnn_pca_scaler = StandardScaler()
    X_scaled = self.lmnn_pca_scaler.fit_transform(self.X_original)
    
    print("  Training LMNN...")
    self.lmnn_op = LMNN(n_components=min(50, self.X_original.shape[1]), k=5, 
                        learn_rate=1e-6, max_iter=100, verbose=False, random_state=self.random_seed)
    X_lmnn = self.lmnn_op.fit_transform(X_scaled, self.y_original)
    print(f"  ✓ LMNN transformation: {X_lmnn.shape}")
    
    self.pca_op = PCA(n_components=2, random_state=self.random_seed)
    self.lmnn_pca_embeddings['original'] = self.pca_op.fit_transform(X_lmnn)
    print(f"  ✓ LMNN+PCA embeddings computed")
    
    for model_name, generated_data in self.generated_data.items():
        gen_scaled = self.lmnn_pca_scaler.transform(generated_data)
        gen_lmnn = self.lmnn_op.transform(gen_scaled)
        self.lmnn_pca_embeddings[model_name] = self.pca_op.transform(gen_lmnn)
        print(f"  ✓ Projected {model_name}")
    
    print("\n" + "="*80)


def create_metrics_comparison(self):
    """Create comprehensive metrics comparison plots"""
    print("\n" + "="*80)
    print("Creating Metrics Comparison Plots")
    print("="*80)
    
    if not self.evaluation_metrics:
        print("  ⚠️  No evaluation metrics available")
        return
    
    metrics_to_plot = [
        ('test_loss', 'Test Loss', True),
        ('frechet_distance', 'Fréchet Distance', True),
        ('mae', 'MAE', True),
        ('pcc', 'Pearson Correlation', False),
        ('wasserstein_distance', 'Wasserstein Distance', True),
        ('mmd', 'MMD', True),
        ('r2_mean', 'R² (mean)', False),
        ('js_divergence', 'JS Divergence', True),
        ('correlation_frobenius_diff', 'Correlation Frobenius Diff', True),
        ('correlation_structure_corr', 'Correlation Structure Corr', False)
    ]
    
    model_names = sorted(self.evaluation_metrics.keys())
    n_metrics = len(metrics_to_plot)
    
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_label, lower_is_better) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        values = []
        labels = []
        for model_name in model_names:
            if metric_key in self.evaluation_metrics[model_name]:
                values.append(self.evaluation_metrics[model_name][metric_key])
                labels.append(model_name)
        
        if not values:
            ax.text(0.5, 0.5, f'No data for {metric_label}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric_label, fontweight='bold')
            continue
        
        colors = plt.cm.Set2(np.linspace(0, 0.9, len(values)))
        bars = ax.barh(range(len(values)), values, color=colors, alpha=0.7, edgecolor='black')
        
        best_idx = np.argmin(values) if lower_is_better else np.argmax(values)
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)
        
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(metric_label, fontweight='bold')
        ax.set_title(metric_label, fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)
    
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    output_path = self.output_dir / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved metrics comparison to: {output_path}")
    
    output_path_pdf = self.output_dir / 'metrics_comparison.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF to: {output_path_pdf}")
    
    plt.close()
    
    metrics_df = pd.DataFrame(self.evaluation_metrics).T
    csv_path = self.output_dir / 'metrics_comparison.csv'
    metrics_df.to_csv(csv_path)
    print(f"✓ Saved metrics CSV to: {csv_path}")
    
    print("="*80)
