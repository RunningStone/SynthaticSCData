#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Setting Visualizer - Visualization Methods

Dynamic visualization methods for multi-setting comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def create_dynamic_visualization(self, embedding_type: str = 'phate'):
    """
    Create dynamic visualization with adaptive subplot layout
    
    Layout:
    - Subplot 1: All original timepoints (for reference)
    - Subplots 2-N+1: Target timepoint (7d) + each model's generation
    
    Note: Models generate from 0d → 7d (first to last timepoint)
    """
    print("\n" + "="*80)
    print(f"Creating Dynamic Visualization ({embedding_type.upper()})")
    print("="*80)
    
    embeddings = self.phate_embeddings if embedding_type == 'phate' else self.lmnn_pca_embeddings
    
    original_emb = embeddings['original']
    model_names = sorted([k for k in embeddings.keys() if k != 'original'])
    n_models = len(model_names)
    
    # Dynamic layout: 1 (all original) + n_models (target+each model)
    total_subplots = 1 + n_models
    n_cols = min(4, total_subplots)
    n_rows = (total_subplots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if total_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    time_colors_dark = plt.cm.tab10(np.linspace(0, 0.9, len(self.time_labels)))
    model_colors = plt.cm.Set2(np.linspace(0, 0.9, n_models))
    
    # Get source and target timepoint indices
    first_time_idx = 0  # Source: 0d
    last_time_idx = len(self.time_labels) - 1  # Target: 7d
    
    subplot_idx = 0
    
    # ========== Subplot 1: All original data (for reference) ==========
    ax = axes[subplot_idx]
    for time_idx, time_label in enumerate(self.time_labels):
        mask = (self.y_original == time_idx)
        ax.scatter(original_emb[mask, 0], original_emb[mask, 1],
                  c=[time_colors_dark[time_idx]], label=time_label,
                  alpha=0.6, s=20, edgecolors='none')
    ax.set_title('Original Data\n(All Timepoints)', fontweight='bold', fontsize=12)
    ax.legend(loc='best', fontsize=8, framealpha=0.9)
    ax.set_xlabel(f'{embedding_type.upper()} 1')
    ax.set_ylabel(f'{embedding_type.upper()} 2')
    ax.grid(alpha=0.3)
    subplot_idx += 1
    
    # ========== Subplots 2-N+1: Target (7d) + each model's generation ==========
    target_time_mask = (self.y_original == last_time_idx)
    target_time_label = self.time_labels[last_time_idx]
    
    # 配色方案：粉红色表示真实数据，浅蓝色表示生成数据
    real_color = '#FF69B4'      # 粉红色 (HotPink)
    generated_color = '#87CEEB'  # 浅蓝色 (SkyBlue)
    
    for model_idx, model_name in enumerate(model_names):
        ax = axes[subplot_idx]
        
        # Plot target timepoint (7d) original data - 粉红色
        ax.scatter(original_emb[target_time_mask, 0], original_emb[target_time_mask, 1],
                  c=real_color, alpha=0.6, s=30, edgecolors='darkred', linewidths=0.8,
                  label=f'{target_time_label} (real)', marker='o')
        
        # Plot model's generated data - 浅蓝色
        gen_emb = embeddings[model_name]
        ax.scatter(gen_emb[:, 0], gen_emb[:, 1],
                  c=generated_color, alpha=0.7, s=45,
                  edgecolors='darkblue', linewidths=0.8, 
                  label=f'{model_name} (gen)', marker='*')
        
        ax.set_title(f'{target_time_label} Real vs {model_name}', fontweight='bold', fontsize=11)
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.set_xlabel(f'{embedding_type.upper()} 1')
        ax.set_ylabel(f'{embedding_type.upper()} 2')
        ax.grid(alpha=0.3)
        subplot_idx += 1
    
    # Hide unused subplots
    for idx in range(subplot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    output_path = self.output_dir / f'generation_comparison_{embedding_type}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    
    output_path_pdf = self.output_dir / f'generation_comparison_{embedding_type}.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF to: {output_path_pdf}")
    
    plt.close()
    
    print(f"\n  Layout: 1 (all timepoints) + {n_models} (target vs each model)")
    print(f"  Source: {self.time_labels[first_time_idx]} → Target: {target_time_label}")
    print("="*80)


def run_full_pipeline(self, config_paths: list, n_samples_per_timepoint: int = 500, n_generate_per_model: int = 500):
    """Run the full multi-setting visualization pipeline"""
    print("\n" + "="*80)
    print("MULTI-SETTING VISUALIZATION PIPELINE")
    print("="*80)
    
    # Step 1: Load experiment configurations
    configs = self.load_experiment_configs(config_paths)
    
    # Step 2: Aggregate model configurations
    model_configs = self.aggregate_model_configs(configs)
    
    # Step 3: Load evaluation metrics
    self.load_evaluation_metrics(configs)
    
    # Step 4: Load and sample data (use first config's data settings)
    first_config = list(configs.values())[0]
    file_path = first_config['data_source']['file_path']
    n_hvg = first_config['data_source']['n_hvg']
    self.load_and_sample_data(file_path, n_hvg, n_samples_per_timepoint)
    
    # Step 5: Load models and generate
    self.load_models_and_generate(model_configs, n_generate_per_model)
    
    # Step 6: Compute embeddings
    self.compute_embeddings()
    
    # Step 7: Create metrics comparison
    self.create_metrics_comparison()
    
    # Step 8: Create visualizations
    self.create_dynamic_visualization(embedding_type='phate')
    self.create_dynamic_visualization(embedding_type='lmnn_pca')
    
    print("\n" + "="*80)
    print("MULTI-SETTING VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {self.output_dir}")
    print(f"  - Metrics comparison: metrics_comparison.png/pdf/csv")
    print(f"  - PHATE visualization: generation_comparison_phate.png/pdf")
    print(f"  - LMNN+PCA visualization: generation_comparison_lmnn_pca.png/pdf")
    print()
