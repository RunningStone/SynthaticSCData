#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluator for Schrödinger Bridge model
Computes metrics and generates comparison plots
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.stats import pearsonr
from .metrics import compute_all_metrics


class Evaluator:
    """
    Evaluator for SB models
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: Device for evaluation
        """
        self.device = device
    
    def evaluate(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        time_labels: List[str]
    ) -> Dict:
        """
        Evaluate model on test set
        
        Args:
            model: Trained SB model
            test_loader: Test data loader
            time_labels: List of time label names
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        
        # Collect all data (no torch.no_grad here, as generate_trajectory needs gradients)
        all_X = []
        all_y = []
        
        for X, y in test_loader:
            all_X.append(X)
            all_y.append(y)
        
        all_X = torch.cat(all_X, dim=0).to(self.device)
        all_y = torch.cat(all_y, dim=0).to(self.device)
        
        # Enable gradients for all computations (SB model needs gradients for drift)
        # But we won't update model parameters (model.eval() ensures this)
        all_X.requires_grad_(True)
        
        # Compute all metrics (all need gradients for SB model)
        test_loss = self._compute_test_loss(model, all_X, all_y)
        frechet_distance = self._compute_frechet_distance(model, all_X, all_y)
        mae = self._compute_mae(model, all_X, all_y)
        pcc = self._compute_pcc(model, all_X, all_y)
        
        # Compute advanced metrics (Wasserstein, MMD, R², JS, Correlation)
        advanced_metrics = self._compute_advanced_metrics(model, all_X, all_y)
        
        results = {
            'test_loss': test_loss,
            'frechet_distance': frechet_distance,
            'mae': mae,
            'pcc': pcc,
            'n_samples': len(all_X)
        }
        
        # Merge advanced metrics
        results.update(advanced_metrics)
        
        return results
    
    def _compute_test_loss(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """Compute test loss"""
        total_loss = 0.0
        n_pairs = 0
        
        unique_times = torch.unique(y)
        time_to_indices = {t.item(): (y == t).nonzero(as_tuple=True)[0] for t in unique_times}
        
        sorted_times = sorted(time_to_indices.keys())
        
        # Check if model is SB (has time-dependent loss) or OT/VAE (direct mapping)
        # SB models have compute_loss(x_t, x_next, t, dt)
        # OT/VAE models have compute_loss(x_source, x_target)
        import inspect
        loss_signature = inspect.signature(model.compute_loss)
        is_time_dependent = len(loss_signature.parameters) > 2
        
        for i in range(len(sorted_times) - 1):
            t_curr = sorted_times[i]
            t_next = sorted_times[i + 1]
            
            indices_curr = time_to_indices[t_curr]
            indices_next = time_to_indices[t_next]
            
            if len(indices_curr) == 0 or len(indices_next) == 0:
                continue
            
            n_pairs_curr = min(len(indices_curr), len(indices_next))
            x_t = X[indices_curr[:n_pairs_curr]]
            x_next = X[indices_next[:n_pairs_curr]]
            
            if is_time_dependent:
                # SB model: needs time parameters
                t = torch.full((n_pairs_curr,), float(t_curr) / len(sorted_times), device=self.device)
                dt = 1.0 / len(sorted_times)
                loss = model.compute_loss(x_t, x_next, t, dt)
            else:
                # OT/VAE model: direct mapping
                loss = model.compute_loss(x_t, x_next)
            
            total_loss += loss.detach().item() * n_pairs_curr
            n_pairs += n_pairs_curr
        
        return total_loss / n_pairs if n_pairs > 0 else 0.0
    
    def _compute_frechet_distance(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """
        Compute Frechet Distance between generated and real distributions
        FD = ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2(Σ1Σ2)^(1/2))
        """
        try:
            # Generate predictions for all samples
            unique_times = torch.unique(y)
            time_to_indices = {t.item(): (y == t).nonzero(as_tuple=True)[0] for t in unique_times}
            sorted_times = sorted(time_to_indices.keys())
            
            if len(sorted_times) < 2:
                return float('nan')
            
            # Use first and last timepoints
            t_start = sorted_times[0]
            t_end = sorted_times[-1]
            
            indices_start = time_to_indices[t_start]
            indices_end = time_to_indices[t_end]
            
            x_start = X[indices_start]
            x_real_end = X[indices_end]
            
            # Generate trajectory
            time_grid = torch.linspace(
                float(t_start) / len(sorted_times),
                float(t_end) / len(sorted_times),
                steps=10,
                device=self.device
            )
            
            trajectory = model.generate_trajectory(x_start, time_grid, method='deterministic')
            x_gen_end = trajectory[:, -1, :]
            
            # Compute statistics (detach tensors before converting to numpy)
            mu_real = x_real_end.mean(dim=0).detach().cpu().numpy()
            mu_gen = x_gen_end.mean(dim=0).detach().cpu().numpy()
            
            sigma_real = np.cov(x_real_end.detach().cpu().numpy(), rowvar=False)
            sigma_gen = np.cov(x_gen_end.detach().cpu().numpy(), rowvar=False)
            
            # Frechet distance
            diff = mu_real - mu_gen
            covmean = sqrtm(sigma_real @ sigma_gen)
            
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            fd = np.sum(diff ** 2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
            
            return float(fd)
        
        except Exception as e:
            print(f"Warning: Could not compute Frechet Distance: {e}")
            return float('nan')
    
    def _compute_mae(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """Compute Mean Absolute Error"""
        try:
            unique_times = torch.unique(y)
            time_to_indices = {t.item(): (y == t).nonzero(as_tuple=True)[0] for t in unique_times}
            sorted_times = sorted(time_to_indices.keys())
            
            if len(sorted_times) < 2:
                return float('nan')
            
            t_start = sorted_times[0]
            t_end = sorted_times[-1]
            
            indices_start = time_to_indices[t_start]
            indices_end = time_to_indices[t_end]
            
            x_start = X[indices_start]
            x_real_end = X[indices_end]
            
            time_grid = torch.linspace(
                float(t_start) / len(sorted_times),
                float(t_end) / len(sorted_times),
                steps=10,
                device=self.device
            )
            
            trajectory = model.generate_trajectory(x_start, time_grid, method='deterministic')
            x_gen_end = trajectory[:, -1, :]
            
            # Match sizes
            n_samples = min(len(x_real_end), len(x_gen_end))
            mae = torch.mean(torch.abs(x_real_end[:n_samples] - x_gen_end[:n_samples]))
            
            return float(mae.item())
        
        except Exception as e:
            print(f"Warning: Could not compute MAE: {e}")
            return float('nan')
    
    def _compute_pcc(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> float:
        """Compute Pearson Correlation Coefficient"""
        try:
            unique_times = torch.unique(y)
            time_to_indices = {t.item(): (y == t).nonzero(as_tuple=True)[0] for t in unique_times}
            sorted_times = sorted(time_to_indices.keys())
            
            if len(sorted_times) < 2:
                return float('nan')
            
            t_start = sorted_times[0]
            t_end = sorted_times[-1]
            
            indices_start = time_to_indices[t_start]
            indices_end = time_to_indices[t_end]
            
            x_start = X[indices_start]
            x_real_end = X[indices_end]
            
            time_grid = torch.linspace(
                float(t_start) / len(sorted_times),
                float(t_end) / len(sorted_times),
                steps=10,
                device=self.device
            )
            
            trajectory = model.generate_trajectory(x_start, time_grid, method='deterministic')
            x_gen_end = trajectory[:, -1, :]
            
            # Match sizes
            n_samples = min(len(x_real_end), len(x_gen_end))
            
            # Compute correlation across all genes (detach tensors)
            real_flat = x_real_end[:n_samples].detach().cpu().numpy().flatten()
            gen_flat = x_gen_end[:n_samples].detach().cpu().numpy().flatten()
            
            pcc, _ = pearsonr(real_flat, gen_flat)
            
            return float(pcc)
        
        except Exception as e:
            print(f"Warning: Could not compute PCC: {e}")
            return float('nan')
    
    def _compute_advanced_metrics(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict:
        """
        Compute advanced metrics: Wasserstein, MMD, R² per gene, JS divergence, 
        and correlation structure similarity.
        """
        try:
            unique_times = torch.unique(y)
            time_to_indices = {t.item(): (y == t).nonzero(as_tuple=True)[0] for t in unique_times}
            sorted_times = sorted(time_to_indices.keys())
            
            if len(sorted_times) < 2:
                return {
                    'wasserstein_distance': float('nan'),
                    'mmd': float('nan'),
                    'r2_mean': float('nan'),
                    'r2_median': float('nan'),
                    'r2_std': float('nan'),
                    'js_divergence': float('nan'),
                    'correlation_frobenius_diff': float('nan'),
                    'correlation_structure_corr': float('nan')
                }
            
            # Get start and end timepoints
            t_start = sorted_times[0]
            t_end = sorted_times[-1]
            
            indices_start = time_to_indices[t_start]
            indices_end = time_to_indices[t_end]
            
            x_start = X[indices_start]
            x_real_end = X[indices_end]
            
            # Generate trajectory
            time_grid = torch.linspace(
                float(t_start) / len(sorted_times),
                float(t_end) / len(sorted_times),
                steps=10,
                device=self.device
            )
            
            trajectory = model.generate_trajectory(x_start, time_grid, method='deterministic')
            x_gen_end = trajectory[:, -1, :]
            
            # Match sizes
            n_samples = min(len(x_real_end), len(x_gen_end))
            
            # Convert to numpy for metric computation
            real_np = x_real_end[:n_samples].detach().cpu().numpy()
            gen_np = x_gen_end[:n_samples].detach().cpu().numpy()
            
            # Compute all advanced metrics
            print("  Computing advanced metrics...")
            metrics = compute_all_metrics(real_np, gen_np, include_detailed=False)
            
            return metrics
        
        except Exception as e:
            print(f"Warning: Could not compute advanced metrics: {e}")
            import traceback
            traceback.print_exc()
            return {
                'wasserstein_distance': float('nan'),
                'mmd': float('nan'),
                'r2_mean': float('nan'),
                'r2_median': float('nan'),
                'r2_std': float('nan'),
                'js_divergence': float('nan'),
                'correlation_frobenius_diff': float('nan'),
                'correlation_structure_corr': float('nan')
            }
    
    def plot_comparison(
        self,
        results_s1: Dict,
        results_s2: Dict,
        save_path: str
    ):
        """
        Plot comprehensive comparison between Setting 1 and Setting 2
        
        Args:
            results_s1: Results from Setting 1
            results_s2: Results from Setting 2
            save_path: Path to save figure
        """
        # Create figure with more subplots for all metrics
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        # Define all metrics to plot
        metrics = [
            'test_loss', 'frechet_distance', 'mae', 'pcc',
            'wasserstein_distance', 'mmd', 'r2_mean', 'js_divergence',
            'correlation_structure_corr'
        ]
        titles = [
            'Test Loss', 'Frechet Distance', 'MAE', 'Pearson Correlation',
            'Wasserstein Distance', 'MMD', 'R² (mean)', 'JS Divergence',
            'Correlation Structure'
        ]
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            
            val_s1 = results_s1.get(metric, float('nan'))
            val_s2 = results_s2.get(metric, float('nan'))
            
            if not np.isnan(val_s1) and not np.isnan(val_s2):
                bars = ax.bar(['Setting 1\n(Boundary)', 'Setting 2\n(All Timepoints)'], 
                       [val_s1, val_s2],
                       color=['#FF6B6B', '#4ECDC4'])
                ax.set_ylabel(title)
                ax.set_title(title, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for i, v in enumerate([val_s1, val_s2]):
                    ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
                
                # Highlight better performance
                if metric in ['pcc', 'r2_mean', 'correlation_structure_corr']:
                    # Higher is better
                    better_idx = 0 if val_s1 > val_s2 else 1
                else:
                    # Lower is better
                    better_idx = 0 if val_s1 < val_s2 else 1
                bars[better_idx].set_edgecolor('gold')
                bars[better_idx].set_linewidth(3)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(title, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comprehensive comparison plot saved to: {save_path}")
