#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics Plotter - Evaluation Metrics Visualization

Handles visualization of evaluation metrics including:
- Bar charts for metrics comparison
- Heatmaps for metrics across models
- CSV export of metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MetricsPlotter:
    """
    Visualizes evaluation metrics for model comparison.
    
    Responsibilities:
    - Plot metrics comparison bar charts
    - Create metrics heatmaps
    - Save metrics as CSV
    - Highlight best models
    """
    
    def __init__(self):
        """Initialize metrics plotter"""
        pass
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict],
        title: str,
        save_prefix: str,
        output_dir: Path,
        formats: List[str] = ['png', 'pdf'],
        dpi: int = 300
    ) -> List[Path]:
        """
        Plot comprehensive metrics comparison.
        
        Args:
            metrics_dict: Dictionary mapping model names to metrics
            title: Plot title
            save_prefix: Filename prefix
            output_dir: Output directory
            formats: List of formats to save
            dpi: DPI for raster formats
        
        Returns:
            List of saved file paths
        """
        # Define metrics to plot
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
        
        model_names = sorted(metrics_dict.keys())
        n_metrics = len(metrics_to_plot)
        
        # Create subplot grid
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
        axes = axes.flatten()
        
        # Plot each metric
        for idx, (metric_key, metric_label, lower_is_better) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Extract values
            values = []
            labels = []
            for model_name in model_names:
                # Handle nested 'evaluation' key
                metrics = metrics_dict[model_name]
                if 'evaluation' in metrics:
                    metrics = metrics['evaluation']
                
                if metric_key in metrics:
                    values.append(metrics[metric_key])
                    labels.append(model_name)
            
            if not values:
                ax.text(0.5, 0.5, f'No data for {metric_label}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric_label, fontweight='bold')
                continue
            
            # Plot horizontal bar chart
            colors = plt.cm.Set2(np.linspace(0, 0.9, len(values)))
            bars = ax.barh(range(len(values)), values, color=colors, alpha=0.7, edgecolor='black')
            
            # Highlight best model
            best_idx = np.argmin(values) if lower_is_better else np.argmax(values)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
            
            # Configure axes
            ax.set_yticks(range(len(values)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel(metric_label, fontweight='bold')
            ax.set_title(metric_label, fontweight='bold', fontsize=12)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save figure
        saved_paths = []
        for fmt in formats:
            output_path = output_dir / f'{save_prefix}.{fmt}'
            if fmt in ['png', 'jpg', 'jpeg']:
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            else:
                fig.savefig(output_path, bbox_inches='tight')
            saved_paths.append(output_path)
        
        plt.close()
        
        # Save CSV
        csv_path = self.save_metrics_csv(metrics_dict, save_prefix, output_dir)
        saved_paths.append(csv_path)
        
        return saved_paths
    
    def save_metrics_csv(
        self,
        metrics_dict: Dict[str, Dict],
        save_prefix: str,
        output_dir: Path
    ) -> Path:
        """
        Save metrics to CSV file.
        
        Args:
            metrics_dict: Dictionary mapping model names to metrics
            save_prefix: Filename prefix
            output_dir: Output directory
        
        Returns:
            Path to saved CSV file
        """
        # Prepare data for CSV
        csv_data = []
        
        for model_name, metrics in metrics_dict.items():
            # Handle nested 'evaluation' key
            if 'evaluation' in metrics:
                metrics = metrics['evaluation']
            
            row = {'model': model_name}
            row.update(metrics)
            csv_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(csv_data)
        
        # Save to CSV
        csv_path = output_dir / f'{save_prefix}.csv'
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def plot_metrics_bar_simple(
        self,
        metrics_dict: Dict[str, Dict],
        metric_keys: List[str],
        title: str,
        save_prefix: str,
        output_dir: Path,
        formats: List[str] = ['png', 'pdf'],
        dpi: int = 300
    ) -> List[Path]:
        """
        Plot simple bar chart for selected metrics.
        
        Args:
            metrics_dict: Dictionary mapping model names to metrics
            metric_keys: List of metric keys to plot
            title: Plot title
            save_prefix: Filename prefix
            output_dir: Output directory
            formats: List of formats to save
            dpi: DPI for raster formats
        
        Returns:
            List of saved file paths
        """
        model_names = list(metrics_dict.keys())
        n_metrics = len(metric_keys)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric_key in enumerate(metric_keys):
            ax = axes[idx]
            
            # Extract values
            values = []
            for model_name in model_names:
                metrics = metrics_dict[model_name]
                if 'evaluation' in metrics:
                    metrics = metrics['evaluation']
                values.append(metrics.get(metric_key, np.nan))
            
            # Plot
            bars = ax.bar(range(len(values)), values)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel(metric_key)
            ax.set_title(metric_key, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values):
                if not np.isnan(v):
                    ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        saved_paths = []
        for fmt in formats:
            output_path = output_dir / f'{save_prefix}.{fmt}'
            if fmt in ['png', 'jpg', 'jpeg']:
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            else:
                fig.savefig(output_path, bbox_inches='tight')
            saved_paths.append(output_path)
        
        plt.close()
        
        return saved_paths
