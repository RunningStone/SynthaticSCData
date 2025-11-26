#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entropy Evolution Analyzer for Cell State Transitions

Analyzes whether models can reproduce the non-monotonic entropy evolution
(entropy increase → entropy decrease) observed in real EMT trajectories.

This module inherits from BaseVisualizer for common infrastructure and
uses DataManager/ModelManager for data and model operations.

Usage:
    from Analyser import EntropyAnalyzer
    
    analyzer = EntropyAnalyzer(output_dir='./results', device='cuda')
    
    # Compute entropy curve from real data
    real_curve = analyzer.compute_entropy_curve_from_real_data(X_test, y_test, time_labels)
    
    # Compute entropy curve from model-generated trajectories
    gen_curve = analyzer.compute_entropy_curve_from_model(model, initial_states, time_grid)
    
    # Compare multiple models
    results = analyzer.compare_multiple_models(models_dict, initial_states, time_grid)
    
    # Generate visualizations
    analyzer.plot_entropy_curves(results, real_curve, time_labels)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

from .base_visualizer import BaseVisualizer
from .data_manager import DataManager
from .model_manager import ModelManager


class EntropyAnalyzer(BaseVisualizer):
    """
    Analyzes entropy evolution across generated trajectories.
    
    Inherits from BaseVisualizer for:
    - Output directory management
    - Device management
    - Figure saving utilities
    - Logging utilities
    
    Uses composition for:
    - DataManager: Data loading operations
    - ModelManager: Model loading and inference
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        device: str = 'cuda',
        random_seed: int = 42
    ):
        """
        Initialize entropy analyzer.
        
        Args:
            output_dir: Directory to save outputs
            device: Device for computation ('cuda' or 'cpu')
            random_seed: Random seed for reproducibility
        """
        super().__init__(output_dir, device, random_seed)
        
        # Composition: use existing managers
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        
        # Import entropy utilities from Data module
        from Data.entropy_utils import (
            estimate_entropy_knn,
            estimate_entropy_gaussian,
            estimate_entropy_both_methods
        )
        self._estimate_entropy_knn = estimate_entropy_knn
        self._estimate_entropy_gaussian = estimate_entropy_gaussian
        self._estimate_entropy_both_methods = estimate_entropy_both_methods
    
    def compute_entropy_curve_from_real_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        time_labels: List[str],
        method: str = 'knn',
        k: int = 5,
        n_samples: int = 1000,
        verbose: bool = True
    ) -> Tuple[np.ndarray, List[np.ndarray], Dict]:
        """
        Compute entropy curve from real data snapshots.
        
        This provides the ground truth entropy evolution for comparison.
        
        Args:
            X: Expression matrix (n_cells, n_genes)
            y: Time labels as integers (n_cells,)
            time_labels: Ordered list of time labels
            method: Entropy estimation method ('knn', 'gaussian', 'both')
            k: KNN parameter
            n_samples: Maximum samples per time point
            verbose: Show progress
        
        Returns:
            Tuple of:
                - entropy_curve: Array of entropy values
                - data_list: List of data matrices per time point
                - peak_analysis: Peak characteristics dictionary
        """
        if verbose:
            self._print_section("Computing Real Data Entropy Curve")
        
        data_list = []
        entropy_curve = []
        
        for time_idx, time_label in enumerate(time_labels):
            # Get data for this time point
            mask = (y == time_idx)
            X_t = X[mask]
            
            # Sample if too many cells
            if len(X_t) > n_samples:
                indices = np.random.choice(len(X_t), n_samples, replace=False)
                X_t = X_t[indices]
            
            data_list.append(X_t)
            
            # Compute entropy
            if method == 'knn':
                H_t = self._estimate_entropy_knn(X_t, k=k)
            elif method == 'gaussian':
                H_t = self._estimate_entropy_gaussian(X_t, shrinkage=True)
            elif method == 'both':
                _, _, H_t = self._estimate_entropy_both_methods(X_t, k=k)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            entropy_curve.append(H_t)
            
            if verbose:
                self._print_info(f"{time_label}: entropy = {H_t:.4f} ({len(X_t)} cells)")
        
        entropy_curve = np.array(entropy_curve)
        peak_analysis = self.analyze_entropy_peak(entropy_curve, time_labels)
        
        if verbose:
            self._print_subsection("Peak Analysis")
            self._print_info(f"Peak at: {peak_analysis['peak_time']}")
            self._print_info(f"Peak value: {peak_analysis['peak_value']:.4f}")
            self._print_info(f"Non-monotonic: {peak_analysis['is_nonmonotonic']}")
        
        return entropy_curve, data_list, peak_analysis
    
    def compute_entropy_curve_from_model(
        self,
        model: torch.nn.Module,
        initial_states: torch.Tensor,
        time_grid: torch.Tensor,
        time_labels: List[str],
        method: str = 'knn',
        k: int = 5,
        verbose: bool = True
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute entropy curve along model-generated trajectory.
        
        Args:
            model: Trained model with generate_trajectory method
            initial_states: Initial cell states (N, d)
            time_grid: Normalized time points [0, t1, ..., 1]
            time_labels: Human-readable labels
            method: Entropy estimation method
            k: KNN parameter
            verbose: Show progress
        
        Returns:
            Tuple of:
                - entropy_curve: Array of entropy values
                - trajectory_list: List of state arrays per time point
        """
        model.eval()
        model = model.to(self.device)
        initial_states = initial_states.to(self.device)
        time_grid = time_grid.to(self.device)
        
        N = initial_states.shape[0]
        K = len(time_grid)
        
        if verbose:
            self._print_info(f"Generating trajectory for {N} cells across {K} time points...")
        
        # Generate complete trajectory
        trajectory_tensor = model.generate_trajectory(
            initial_states,
            time_grid,
            method='deterministic'
        )
        
        # Convert to list of numpy arrays
        trajectory_list = []
        for j in range(K):
            X_t = trajectory_tensor[:, j, :].detach().cpu().numpy()
            trajectory_list.append(X_t)
        
        # Compute entropy at each time point
        entropy_curve = []
        
        iterator = enumerate(trajectory_list)
        if verbose:
            iterator = tqdm(iterator, total=K, desc="Computing entropy")
        
        for j, X_t in iterator:
            if method == 'knn':
                H_t = self._estimate_entropy_knn(X_t, k=k)
            elif method == 'gaussian':
                H_t = self._estimate_entropy_gaussian(X_t, shrinkage=True)
            elif method == 'both':
                _, _, H_t = self._estimate_entropy_both_methods(X_t, k=k)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            entropy_curve.append(H_t)
        
        return np.array(entropy_curve), trajectory_list
    
    def analyze_entropy_peak(
        self,
        entropy_curve: np.ndarray,
        time_labels: List[str]
    ) -> Dict:
        """
        Analyze entropy peak characteristics.
        
        Detects:
        1. Peak position and value
        2. Peak amplitude
        3. Non-monotonicity (inverted-U shape)
        4. Entropy change rates (explore vs. collapse)
        
        Args:
            entropy_curve: Array of entropy values
            time_labels: Time labels corresponding to entropy values
        
        Returns:
            Dictionary with analysis results
        """
        # Find peak
        peak_idx = int(np.argmax(entropy_curve))
        peak_time = time_labels[peak_idx]
        peak_value = float(entropy_curve[peak_idx])
        
        # Compute amplitude (peak height above boundary minimum)
        boundary_min = min(entropy_curve[0], entropy_curve[-1])
        amplitude = peak_value - boundary_min
        
        # Check non-monotonicity (inverted-U)
        is_nonmonotonic = (
            (peak_value > entropy_curve[0]) and 
            (peak_value > entropy_curve[-1])
        )
        
        # Compute entropy change rates
        if peak_idx > 0:
            explore_rate = (peak_value - entropy_curve[0]) / peak_idx
        else:
            explore_rate = 0.0
        
        if peak_idx < len(entropy_curve) - 1:
            collapse_rate = (entropy_curve[-1] - peak_value) / (len(entropy_curve) - peak_idx - 1)
        else:
            collapse_rate = 0.0
        
        # Asymmetry: ratio of rates
        if abs(collapse_rate) > 1e-10:
            asymmetry = abs(explore_rate) / abs(collapse_rate)
        else:
            asymmetry = float('inf')
        
        return {
            'peak_idx': peak_idx,
            'peak_time': peak_time,
            'peak_value': peak_value,
            'amplitude': amplitude,
            'is_nonmonotonic': is_nonmonotonic,
            'explore_rate': explore_rate,
            'collapse_rate': collapse_rate,
            'asymmetry': asymmetry
        }
    
    def compute_entropy_curve_similarity(
        self,
        H_real: np.ndarray,
        H_gen: np.ndarray,
        metric: str = 'mse'
    ) -> float:
        """
        Compute similarity between real and generated entropy curves.
        
        Args:
            H_real: Real entropy curve
            H_gen: Generated entropy curve
            metric: 'mse' (mean squared error) or 'dtw' (dynamic time warping)
        
        Returns:
            Similarity score (lower is better for MSE)
        """
        if metric == 'mse':
            return float(np.mean((H_real - H_gen) ** 2))
        
        elif metric == 'dtw':
            try:
                from dtaidistance import dtw
                distance = dtw.distance(H_real, H_gen)
                return float(distance)
            except ImportError:
                self._print_warning("dtaidistance not installed, falling back to MSE")
                return float(np.mean((H_real - H_gen) ** 2))
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compare_multiple_models(
        self,
        models_dict: Dict[str, torch.nn.Module],
        initial_states: torch.Tensor,
        time_grid: torch.Tensor,
        time_labels: List[str],
        real_entropy_curve: Optional[np.ndarray] = None,
        method: str = 'knn',
        k: int = 5,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Compare entropy evolution across multiple models.
        
        Args:
            models_dict: Dictionary mapping model names to model objects
            initial_states: Shared initial states for all models
            time_grid: Time grid for trajectory generation
            time_labels: Time labels
            real_entropy_curve: Optional ground truth for comparison
            method: Entropy estimation method
            k: KNN parameter
            verbose: Show progress
        
        Returns:
            Dictionary mapping model names to results
        """
        results = {}
        
        for model_name, model in models_dict.items():
            if verbose:
                self._print_section(f"Analyzing model: {model_name}")
            
            # Compute entropy curve
            entropy_curve, trajectory = self.compute_entropy_curve_from_model(
                model=model,
                initial_states=initial_states,
                time_grid=time_grid,
                time_labels=time_labels,
                method=method,
                k=k,
                verbose=verbose
            )
            
            # Analyze peak
            peak_analysis = self.analyze_entropy_peak(entropy_curve, time_labels)
            
            # Prepare results
            model_results = {
                'entropy_curve': entropy_curve,
                'peak_analysis': peak_analysis,
                'trajectory': trajectory
            }
            
            # Compare to real if provided
            if real_entropy_curve is not None:
                similarity = self.compute_entropy_curve_similarity(real_entropy_curve, entropy_curve)
                model_results['similarity_to_real'] = similarity
                
                if verbose:
                    self._print_info(f"Similarity to real data (MSE): {similarity:.4f}")
            
            # Print peak analysis
            if verbose:
                self._print_subsection("Peak Analysis")
                self._print_info(f"Peak time: {peak_analysis['peak_time']}")
                self._print_info(f"Peak value: {peak_analysis['peak_value']:.4f}")
                self._print_info(f"Amplitude: {peak_analysis['amplitude']:.4f}")
                self._print_info(f"Non-monotonic: {peak_analysis['is_nonmonotonic']}")
            
            results[model_name] = model_results
        
        return results
    
    def plot_entropy_curves(
        self,
        results_dict: Dict[str, Dict],
        real_curve: np.ndarray,
        time_labels: List[str],
        method: str = 'knn'
    ) -> List[Path]:
        """
        Create publication-quality entropy curve comparison plot.
        
        Args:
            results_dict: Dictionary mapping setting names to their results
            real_curve: Real data entropy curve
            time_labels: Time labels
            method: Entropy estimation method for title
        
        Returns:
            List of saved file paths
        """
        fig = plt.figure(figsize=(12, 8))
        
        # Color scheme
        colors = {
            'Real': '#2E86AB',      # Blue
            'Setting1': '#A23B72',  # Purple
            'Setting2': '#F18F01',  # Orange
            'Setting3': '#C73E1D',  # Red
        }
        
        markers = {
            'Real': 'o',
            'Setting1': 's',
            'Setting2': '^',
            'Setting3': 'D'
        }
        
        linestyles = {
            'Real': '-',
            'Setting1': '--',
            'Setting2': '-',
            'Setting3': '-.'
        }
        
        # Plot real data
        plt.plot(
            range(len(time_labels)),
            real_curve,
            marker=markers['Real'],
            linestyle=linestyles['Real'],
            color=colors['Real'],
            linewidth=2.5,
            markersize=10,
            label='Real Data',
            zorder=10
        )
        
        # Plot each model
        color_cycle = ['#A23B72', '#F18F01', '#C73E1D', '#2ECC71', '#9B59B6']
        for idx, (setting_name, results) in enumerate(results_dict.items()):
            entropy_curve = results['entropy_curve']
            
            plt.plot(
                range(len(time_labels)),
                entropy_curve,
                marker=markers.get(setting_name, 'x'),
                linestyle=linestyles.get(setting_name, ':'),
                color=colors.get(setting_name, color_cycle[idx % len(color_cycle)]),
                linewidth=2,
                markersize=8,
                label=setting_name,
                alpha=0.8
            )
        
        # Formatting
        plt.xlabel('Time Point', fontsize=14, fontweight='bold')
        plt.ylabel(f'Differential Entropy ({method.upper()} estimate)', fontsize=14, fontweight='bold')
        plt.title('Entropy Evolution: Real vs. Generated Trajectories', 
                  fontsize=16, fontweight='bold', pad=20)
        
        plt.xticks(range(len(time_labels)), time_labels, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, loc='best', framealpha=0.9)
        
        # Highlight non-monotonic region for real data
        peak_idx = np.argmax(real_curve)
        plt.axvline(x=peak_idx, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        plt.text(
            peak_idx, plt.ylim()[1] * 0.95,
            f'Real peak\nat {time_labels[peak_idx]}',
            ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        # Save using inherited method
        saved_paths = self._save_figure(fig, 'entropy_curves_comparison')
        plt.close()
        
        self._print_success(f"Saved entropy curves to {saved_paths[0]}")
        
        return saved_paths
    
    def plot_peak_characteristics_comparison(
        self,
        results_dict: Dict[str, Dict],
        real_peak_analysis: Dict
    ) -> List[Path]:
        """
        Create bar plot comparing peak characteristics across settings.
        
        Args:
            results_dict: Dictionary mapping setting names to their results
            real_peak_analysis: Peak analysis results for real data
        
        Returns:
            List of saved file paths
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        settings = list(results_dict.keys())
        labels = ['Real'] + settings
        colors_list = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#2ECC71']
        
        # Metric 1: Peak amplitude
        ax = axes[0, 0]
        amplitudes = [real_peak_analysis['amplitude']] + \
                     [results_dict[s]['peak_analysis']['amplitude'] for s in settings]
        
        ax.bar(labels, amplitudes, color=colors_list[:len(labels)])
        ax.set_ylabel('Peak Amplitude', fontsize=12, fontweight='bold')
        ax.set_title('Entropy Peak Amplitude', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Metric 2: Non-monotonicity
        ax = axes[0, 1]
        is_nonmono = [1 if real_peak_analysis['is_nonmonotonic'] else 0] + \
                     [1 if results_dict[s]['peak_analysis']['is_nonmonotonic'] else 0 for s in settings]
        
        ax.bar(labels, is_nonmono, color=colors_list[:len(labels)])
        ax.set_ylabel('Non-monotonic (1=Yes, 0=No)', fontsize=12, fontweight='bold')
        ax.set_title('Inverted-U Shape Detection', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.2])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Metric 3: Explore rate
        ax = axes[1, 0]
        explore_rates = [real_peak_analysis['explore_rate']] + \
                        [results_dict[s]['peak_analysis']['explore_rate'] for s in settings]
        
        ax.bar(labels, explore_rates, color=colors_list[:len(labels)])
        ax.set_ylabel('Entropy Increase Rate', fontsize=12, fontweight='bold')
        ax.set_title('Exploration Phase Rate', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Metric 4: MSE to real
        ax = axes[1, 1]
        mse_values = [results_dict[s].get('similarity_to_real', 0) for s in settings]
        
        ax.bar(settings, mse_values, color=colors_list[1:len(settings)+1])
        ax.set_ylabel('MSE to Real Entropy Curve', fontsize=12, fontweight='bold')
        ax.set_title('Curve Similarity (Lower is Better)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save using inherited method
        saved_paths = self._save_figure(fig, 'peak_characteristics_comparison')
        plt.close()
        
        self._print_success(f"Saved peak characteristics to {saved_paths[0]}")
        
        return saved_paths
    
    def plot_method_cross_validation(
        self,
        results_dict_knn: Dict[str, Dict],
        results_dict_gauss: Dict[str, Dict],
        time_labels: List[str]
    ) -> List[Path]:
        """
        Cross-validate KNN and Gaussian entropy estimation methods.
        
        Args:
            results_dict_knn: Results using KNN method
            results_dict_gauss: Results using Gaussian method
            time_labels: Time labels
        
        Returns:
            List of saved file paths
        """
        n_models = len(results_dict_knn)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, setting_name in enumerate(results_dict_knn.keys()):
            ax = axes[idx]
            
            H_knn = results_dict_knn[setting_name]['entropy_curve']
            H_gauss = results_dict_gauss[setting_name]['entropy_curve']
            
            ax.plot(range(len(time_labels)), H_knn, 'o-', label='KNN', linewidth=2, markersize=8)
            ax.plot(range(len(time_labels)), H_gauss, 's--', label='Gaussian', linewidth=2, markersize=8)
            
            ax.set_xlabel('Time Point', fontsize=12)
            ax.set_ylabel('Entropy', fontsize=12)
            ax.set_title(f'{setting_name}: Method Comparison', fontsize=13, fontweight='bold')
            ax.set_xticks(range(len(time_labels)))
            ax.set_xticklabels(time_labels)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Compute correlation
            corr = np.corrcoef(H_knn, H_gauss)[0, 1]
            ax.text(
                0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        plt.tight_layout()
        
        # Save using inherited method
        saved_paths = self._save_figure(fig, 'method_cross_validation')
        plt.close()
        
        self._print_success(f"Saved method cross-validation to {saved_paths[0]}")
        
        return saved_paths
    
    def save_results(
        self,
        real_entropy_curve: np.ndarray,
        real_peak_analysis: Dict,
        results_dict: Dict[str, Dict],
        time_labels: List[str],
        config: Optional[Dict] = None,
        args: Optional[Dict] = None
    ) -> Tuple[Path, Path]:
        """
        Save analysis results to JSON and pickle files.
        
        Args:
            real_entropy_curve: Real data entropy curve
            real_peak_analysis: Real data peak analysis
            results_dict: Model comparison results
            time_labels: Time labels
            config: Optional experiment configuration
            args: Optional command line arguments
        
        Returns:
            Tuple of (json_path, pkl_path)
        """
        import json
        import pickle
        
        # Helper function to convert numpy types to Python types
        def convert_to_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, (int, np.integer)):
                return int(obj)
            elif isinstance(obj, (float, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save JSON summary
        summary = {
            'real_entropy_curve': real_entropy_curve.tolist(),
            'real_peak_analysis': convert_to_json_serializable(real_peak_analysis),
            'settings': {}
        }
        
        for setting_name, results in results_dict.items():
            summary['settings'][setting_name] = {
                'entropy_curve': results['entropy_curve'].tolist(),
                'peak_analysis': convert_to_json_serializable(results['peak_analysis']),
                'similarity_to_real': float(results.get('similarity_to_real', 0)) 
                    if results.get('similarity_to_real') is not None else None
            }
        
        json_path = self._save_dict_to_json(summary, 'entropy_analysis_summary')
        self._print_success(f"Saved summary to {json_path}")
        
        # Save full pickle results
        full_results = {
            'real_entropy_curve': real_entropy_curve,
            'real_peak_analysis': real_peak_analysis,
            'settings_results': results_dict,
            'time_labels': time_labels,
            'config': config,
            'args': args
        }
        
        pkl_path = self.output_dir / 'entropy_analysis_full_results.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(full_results, f)
        
        self._print_success(f"Saved full results to {pkl_path}")
        
        return json_path, pkl_path
