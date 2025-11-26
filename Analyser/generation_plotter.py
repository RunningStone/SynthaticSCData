#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation Plotter - Generation Comparison Visualization

Handles visualization of generated data comparison including:
- Embedding scatter plots
- Real vs generated comparison
- Multi-model comparison grids
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class GenerationPlotter:
    """
    Visualizes generation comparison for model evaluation.
    
    Responsibilities:
    - Plot embedding scatter plots
    - Create real vs generated comparisons
    - Generate multi-model comparison grids
    """
    
    def __init__(self):
        """Initialize generation plotter"""
        pass
    
    def plot_comparison_grid(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        real_embedding: np.ndarray,
        real_labels: np.ndarray,
        time_labels: List[str],
        embedding_type: str,
        title: str,
        save_prefix: str,
        output_dir: Path,
        formats: List[str] = ['png', 'pdf'],
        dpi: int = 300
    ) -> List[Path]:
        """
        Create comparison grid showing real data and all model generations.
        
        Args:
            embeddings_dict: Dict mapping model names to generated embeddings
            real_embedding: Real data embedding (n_samples, 2)
            real_labels: Real data time labels (n_samples,)
            time_labels: List of time label strings
            embedding_type: Type of embedding ('phate' or 'lmnn_pca')
            title: Plot title
            save_prefix: Filename prefix
            output_dir: Output directory
            formats: List of formats to save
            dpi: DPI for raster formats
        
        Returns:
            List of saved file paths
        """
        model_names = sorted([k for k in embeddings_dict.keys() if k != 'original'])
        n_models = len(model_names)
        
        # Dynamic layout
        total_subplots = 1 + n_models  # 1 for all real + n for each model
        n_cols = min(4, total_subplots)
        n_rows = (total_subplots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if total_subplots == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        # Color schemes
        time_colors = plt.cm.tab10(np.linspace(0, 0.9, len(time_labels)))
        real_color = '#FF69B4'  # Pink for real
        generated_color = '#87CEEB'  # Sky blue for generated
        
        # Get target timepoint (last one)
        last_time_idx = len(time_labels) - 1
        target_time_label = time_labels[last_time_idx]
        target_mask = (real_labels == last_time_idx)
        
        subplot_idx = 0
        
        # Subplot 1: All real data
        ax = axes[subplot_idx]
        for time_idx, time_label in enumerate(time_labels):
            mask = (real_labels == time_idx)
            ax.scatter(real_embedding[mask, 0], real_embedding[mask, 1],
                      c=[time_colors[time_idx]], label=time_label,
                      alpha=0.6, s=20, edgecolors='none')
        ax.set_title('Real Data\n(All Timepoints)', fontweight='bold', fontsize=12)
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.set_xlabel(f'{embedding_type.upper()} 1')
        ax.set_ylabel(f'{embedding_type.upper()} 2')
        ax.grid(alpha=0.3)
        subplot_idx += 1
        
        # Subplots 2-N+1: Target real vs each model's generation
        for model_name in model_names:
            ax = axes[subplot_idx]
            
            # Plot target timepoint real data
            ax.scatter(real_embedding[target_mask, 0], real_embedding[target_mask, 1],
                      c=real_color, alpha=0.6, s=30, edgecolors='darkred', linewidths=0.8,
                      label=f'{target_time_label} (real)', marker='o')
            
            # Plot model's generated data
            gen_emb = embeddings_dict[model_name]
            ax.scatter(gen_emb[:, 0], gen_emb[:, 1],
                      c=generated_color, alpha=0.7, s=45,
                      edgecolors='darkblue', linewidths=0.8,
                      label=f'{model_name} (gen)', marker='*')
            
            ax.set_title(f'{target_time_label} Real vs {model_name}',
                        fontweight='bold', fontsize=11)
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
            ax.set_xlabel(f'{embedding_type.upper()} 1')
            ax.set_ylabel(f'{embedding_type.upper()} 2')
            ax.grid(alpha=0.3)
            subplot_idx += 1
        
        # Hide unused subplots
        for idx in range(subplot_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{title} - {embedding_type.upper()}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        saved_paths = []
        for fmt in formats:
            output_path = output_dir / f'{save_prefix}_{embedding_type}.{fmt}'
            if fmt in ['png', 'jpg', 'jpeg']:
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            else:
                fig.savefig(output_path, bbox_inches='tight')
            saved_paths.append(output_path)
        
        plt.close()
        
        return saved_paths
