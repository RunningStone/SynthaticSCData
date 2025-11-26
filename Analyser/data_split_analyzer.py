#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Split Analyzer for Fair Model Comparison

This module provides functionality to analyze dataset distribution and compute
optimal sampling parameters for fair comparison across different experimental settings.

Author: Auto-generated
Date: 2024-11-24
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
import json
from .base_visualizer import BaseVisualizer


class DataSplitAnalyzer(BaseVisualizer):
    """
    Analyzer for computing optimal data split parameters
    
    This class analyzes cell distribution across batches and time labels,
    computes bottlenecks, and recommends sampling parameters to ensure
    fair comparison across experimental settings.
    """
    
    def __init__(
        self,
        output_dir: str,
        device: str = 'cpu',
        seed: int = 42
    ):
        """
        Initialize DataSplitAnalyzer
        
        Args:
            output_dir: Directory to save analysis results
            device: Device for computation (not used, for API consistency)
            seed: Random seed for reproducibility
        """
        super().__init__(output_dir, device, seed)
        self.log("DataSplitAnalyzer initialized")
    
    def load_data_config(self, config_path: str) -> Dict:
        """Load data configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def analyze_data_distribution(
        self,
        adata_path: str,
        label_column: str,
        batch_column: str,
        train_batches: List[str],
        test_batches: List[str],
        time_labels_order: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze cell distribution across batches and time labels
        
        Args:
            adata_path: Path to h5ad file
            label_column: Column name for time labels
            batch_column: Column name for batch information
            train_batches: List of training batch names
            test_batches: List of test batch names
            time_labels_order: Ordered list of time labels to include
            
        Returns:
            Tuple of (distribution_df, obs_df)
            - distribution_df: DataFrame with columns [batch, time_label, count]
            - obs_df: AnnData obs dataframe
        """
        self.log("="*80)
        self.log("Loading and Analyzing Data Distribution")
        self.log("="*80)
        
        # Load data
        self.log(f"\nLoading: {adata_path}")
        adata = sc.read_h5ad(adata_path)
        self.log(f"Total cells: {adata.n_obs:,}")
        self.log(f"Total genes: {adata.n_vars:,}")
        
        # Check columns exist
        if label_column not in adata.obs.columns:
            raise ValueError(f"Label column '{label_column}' not found in adata.obs")
        if batch_column not in adata.obs.columns:
            raise ValueError(f"Batch column '{batch_column}' not found in adata.obs")
        
        # Filter to only include specified time labels
        valid_mask = adata.obs[label_column].isin(time_labels_order)
        adata_filtered = adata[valid_mask].copy()
        n_filtered = adata.n_obs - adata_filtered.n_obs
        if n_filtered > 0:
            self.log(f"\nFiltered out {n_filtered} cells with labels not in time_labels_order")
        
        # Create distribution table
        results = []
        
        self.log(f"\n{'Batch':<10} {'Time Label':<12} {'Count':>8}")
        self.log("-" * 35)
        
        for batch in sorted(adata_filtered.obs[batch_column].unique()):
            batch_mask = adata_filtered.obs[batch_column] == batch
            for time_label in time_labels_order:
                time_mask = adata_filtered.obs[label_column] == time_label
                count = (batch_mask & time_mask).sum()
                results.append({
                    'batch': batch,
                    'time_label': time_label,
                    'count': count
                })
                self.log(f"{batch:<10} {time_label:<12} {count:>8,}")
        
        df = pd.DataFrame(results)
        
        # Summary statistics
        self.log("\n" + "="*80)
        self.log("Train/Test Split Summary")
        self.log("="*80)
        
        train_mask = adata_filtered.obs[batch_column].isin(train_batches)
        test_mask = adata_filtered.obs[batch_column].isin(test_batches)
        
        self.log(f"\nTrain batches: {train_batches}")
        self.log(f"Test batches: {test_batches}")
        self.log(f"\nTrain cells: {train_mask.sum():,}")
        self.log(f"Test cells: {test_mask.sum():,}")
        
        self.log(f"\n{'Time Label':<12} {'Train Count':>12} {'Test Count':>12}")
        self.log("-" * 40)
        
        for time_label in time_labels_order:
            time_mask = adata_filtered.obs[label_column] == time_label
            train_count = (train_mask & time_mask).sum()
            test_count = (test_mask & time_mask).sum()
            self.log(f"{time_label:<12} {train_count:>12,} {test_count:>12,}")
        
        return df, adata_filtered.obs
    
    def calculate_setting_params(
        self,
        df: pd.DataFrame,
        obs: pd.DataFrame,
        setting_name: str,
        time_points: List[str],
        train_batches: List[str],
        batch_column: str,
        label_column: str,
        min_cells_per_category: int = 1000
    ) -> Dict:
        """
        Calculate optimal sampling parameters for a specific setting
        
        Args:
            df: Distribution dataframe
            obs: AnnData obs dataframe
            setting_name: Name of the setting (e.g., "setting1")
            time_points: List of time points for this setting
            train_batches: List of training batch names
            batch_column: Column name for batch
            label_column: Column name for time label
            min_cells_per_category: Minimum cells required per category
            
        Returns:
            Dict with keys: max_cells_per_timepoint, bottleneck_timepoint, 
                           available_counts, recommended_cells_per_timepoint
        """
        self.log("\n" + "="*80)
        self.log(f"Calculating Parameters for {setting_name}")
        self.log("="*80)
        self.log(f"Time points: {time_points}")
        
        # Calculate available cells per timepoint in training set
        train_mask = obs[batch_column].isin(train_batches)
        available_counts = {}
        
        self.log(f"\n{'Time Point':<12} {'Available (Train)':>20}")
        self.log("-" * 35)
        
        for time_point in time_points:
            time_mask = obs[label_column] == time_point
            count = (train_mask & time_mask).sum()
            available_counts[time_point] = count
            self.log(f"{time_point:<12} {count:>20,}")
        
        # Find bottleneck (minimum available)
        bottleneck_timepoint = min(available_counts, key=available_counts.get)
        max_cells_per_timepoint = available_counts[bottleneck_timepoint]
        
        self.log(f"\n⚠️  Bottleneck: {bottleneck_timepoint} with {max_cells_per_timepoint:,} cells")
        
        # Check if bottleneck meets minimum requirement
        if max_cells_per_timepoint < min_cells_per_category:
            self.log(f"❌ ERROR: Bottleneck ({max_cells_per_timepoint:,}) < minimum required ({min_cells_per_category:,})")
            recommended = 0
        else:
            # Recommend using 90% of bottleneck to leave safety margin
            recommended = int(max_cells_per_timepoint * 0.9)
            # Round down to nearest 100
            recommended = (recommended // 100) * 100
            self.log(f"✓ Recommended cells_per_timepoint: {recommended:,} (90% of bottleneck)")
        
        return {
            'setting_name': setting_name,
            'time_points': time_points,
            'n_timepoints': len(time_points),
            'max_cells_per_timepoint': max_cells_per_timepoint,
            'bottleneck_timepoint': bottleneck_timepoint,
            'available_counts': available_counts,
            'recommended_cells_per_timepoint': recommended,
            'min_cells_required': min_cells_per_category
        }
    
    def compute_fair_comparison_params(
        self,
        setting_results: List[Dict],
        min_cells_per_category: int = 1000,
        group_definitions: Optional[Dict[str, List[str]]] = None,
        bottleneck_percentage: float = 100.0
    ) -> Dict:
        """
        Compute final parameters ensuring fair comparison across settings
        
        Strategy:
        1. Group settings into experimental groups
        2. For each group, find the most restrictive setting
        3. Apply bottleneck_percentage to adjust the target total
        4. Use that as the target total for all settings in the same group
        5. Distribute evenly across timepoints for each setting
        
        Args:
            setting_results: List of results from calculate_setting_params
            min_cells_per_category: Minimum cells per category
            group_definitions: Dict mapping group names to list of setting names
            bottleneck_percentage: Percentage of bottleneck capacity to use (0-100)
            
        Returns:
            Dict with final parameters for each setting
        """
        self.log("\n" + "="*80)
        self.log("Computing Fair Comparison Parameters")
        self.log("="*80)
        
        # If no group definitions provided, treat all settings as one group
        if group_definitions is None:
            all_settings = [r['setting_name'] for r in setting_results]
            group_definitions = {'all': all_settings}
        
        # Calculate max possible total for each setting
        setting_max_totals = {}
        for result in setting_results:
            setting_name = result['setting_name']
            n_timepoints = result['n_timepoints']
            max_per_tp = result['max_cells_per_timepoint']
            max_total = n_timepoints * max_per_tp
            setting_max_totals[setting_name] = max_total
            self.log(f"\n{setting_name}:")
            self.log(f"  Time points: {n_timepoints}")
            self.log(f"  Max per timepoint: {max_per_tp:,}")
            self.log(f"  Max total: {max_total:,}")
        
        # Find target total for each group
        group_targets = {}
        for group_name, group_settings in group_definitions.items():
            group_max_totals = [setting_max_totals[s] for s in group_settings if s in setting_max_totals]
            if not group_max_totals:
                continue
            
            bottleneck_total = min(group_max_totals)
            target_total = int(bottleneck_total * (bottleneck_percentage / 100.0))
            group_targets[group_name] = target_total
            
            self.log(f"\n{'='*80}")
            self.log(f"Group: {group_name}")
            self.log(f"Settings: {group_settings}")
            self.log(f"Bottleneck total: {bottleneck_total:,}")
            self.log(f"Target total ({bottleneck_percentage}%): {target_total:,}")
        
        # Compute final parameters for each setting
        final_params = {}
        for result in setting_results:
            setting_name = result['setting_name']
            n_timepoints = result['n_timepoints']
            
            # Find which group this setting belongs to
            group_name = None
            for gname, gsettings in group_definitions.items():
                if setting_name in gsettings:
                    group_name = gname
                    break
            
            if group_name is None:
                self.log(f"\n⚠️  Warning: {setting_name} not in any group, skipping")
                continue
            
            target_total = group_targets[group_name]
            cells_per_timepoint = target_total // n_timepoints
            
            # Check if meets minimum requirement
            if cells_per_timepoint < min_cells_per_category:
                self.log(f"\n❌ ERROR: {setting_name} cannot meet minimum requirement")
                self.log(f"   Target per timepoint: {cells_per_timepoint:,}")
                self.log(f"   Minimum required: {min_cells_per_category:,}")
                cells_per_timepoint = 0
            
            final_params[setting_name] = {
                'group': group_name,
                'n_timepoints': n_timepoints,
                'target_total_cells': target_total,
                'cells_per_timepoint': cells_per_timepoint,
                'actual_total_cells': cells_per_timepoint * n_timepoints,
                'time_points': result['time_points'],
                'bottleneck_timepoint': result['bottleneck_timepoint'],
                'available_counts': result['available_counts']
            }
            
            self.log(f"\n{setting_name}:")
            self.log(f"  Group: {group_name}")
            self.log(f"  Time points: {n_timepoints}")
            self.log(f"  Cells per timepoint: {cells_per_timepoint:,}")
            self.log(f"  Total cells: {cells_per_timepoint * n_timepoints:,}")
        
        return final_params
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def save_results(
        self,
        distribution_df: pd.DataFrame,
        setting_results: List[Dict],
        final_params: Dict,
        prefix: str = "data_split_analysis"
    ):
        """
        Save analysis results to files
        
        Args:
            distribution_df: Cell distribution dataframe
            setting_results: List of per-setting analysis results
            final_params: Final parameters for all settings
            prefix: Prefix for output filenames
        """
        # Save distribution table
        dist_path = self.output_dir / f"{prefix}_distribution.csv"
        distribution_df.to_csv(dist_path, index=False)
        self.log(f"\n✓ Saved distribution table: {dist_path}")
        
        # Convert numpy types to native Python types for JSON serialization
        setting_results_serializable = self._convert_to_serializable(setting_results)
        final_params_serializable = self._convert_to_serializable(final_params)
        
        # Save setting results
        setting_path = self.output_dir / f"{prefix}_setting_results.json"
        with open(setting_path, 'w') as f:
            json.dump(setting_results_serializable, f, indent=2)
        self.log(f"✓ Saved setting results: {setting_path}")
        
        # Save final parameters
        final_path = self.output_dir / f"{prefix}_final_params.json"
        with open(final_path, 'w') as f:
            json.dump(final_params_serializable, f, indent=2)
        self.log(f"✓ Saved final parameters: {final_path}")
        
        # Generate summary report
        report_path = self.output_dir / f"{prefix}_summary.txt"
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Data Split Analysis Summary\n")
            f.write("="*80 + "\n\n")
            
            for setting_name, params in final_params.items():
                f.write(f"\n{setting_name}:\n")
                f.write(f"  Group: {params['group']}\n")
                f.write(f"  Time points: {params['time_points']}\n")
                f.write(f"  Cells per timepoint: {params['cells_per_timepoint']:,}\n")
                f.write(f"  Total cells: {params['actual_total_cells']:,}\n")
                f.write(f"  Bottleneck: {params['bottleneck_timepoint']}\n")
                f.write("\n")
        
        self.log(f"✓ Saved summary report: {report_path}")
    
    def run_full_analysis(
        self,
        data_config_path: str,
        settings_config: Dict[str, List[str]],
        group_definitions: Optional[Dict[str, List[str]]] = None,
        min_cells_per_category: int = 1000,
        bottleneck_percentage: float = 100.0
    ) -> Dict:
        """
        Run complete data split analysis pipeline
        
        Args:
            data_config_path: Path to data configuration YAML
            settings_config: Dict mapping setting names to time point lists
            group_definitions: Dict mapping group names to setting names
            min_cells_per_category: Minimum cells required per category
            bottleneck_percentage: Percentage of bottleneck to use
            
        Returns:
            Dict with final parameters for all settings
        """
        self.log("\n" + "="*80)
        self.log("Starting Full Data Split Analysis")
        self.log("="*80)
        
        # Load data configuration
        data_config = self.load_data_config(data_config_path)
        
        # Extract configuration parameters
        # Support both old and new config formats
        if 'adata_path' in data_config:
            # Old format
            adata_path = data_config['adata_path']
            label_column = data_config['label_column']
            batch_column = data_config['batch_column']
            train_batches = data_config['train_batches']
            test_batches = data_config['test_batches']
            time_labels_order = data_config['time_labels_order']
        else:
            # New format
            adata_path = data_config['data_source']['file_path']
            label_column = data_config['data_source']['obs_time_column']
            batch_column = data_config['biology_split']['column_name']
            train_batches = data_config['biology_split']['train_values']
            test_batches = data_config['biology_split']['test_values']
            time_labels_order = data_config['data_source']['time_labels_order']
        
        # Analyze data distribution
        distribution_df, obs = self.analyze_data_distribution(
            adata_path=adata_path,
            label_column=label_column,
            batch_column=batch_column,
            train_batches=train_batches,
            test_batches=test_batches,
            time_labels_order=time_labels_order
        )
        
        # Calculate parameters for each setting
        setting_results = []
        for setting_name, time_points in settings_config.items():
            result = self.calculate_setting_params(
                df=distribution_df,
                obs=obs,
                setting_name=setting_name,
                time_points=time_points,
                train_batches=train_batches,
                batch_column=batch_column,
                label_column=label_column,
                min_cells_per_category=min_cells_per_category
            )
            setting_results.append(result)
        
        # Compute fair comparison parameters
        final_params = self.compute_fair_comparison_params(
            setting_results=setting_results,
            min_cells_per_category=min_cells_per_category,
            group_definitions=group_definitions,
            bottleneck_percentage=bottleneck_percentage
        )
        
        # Save all results
        self.save_results(
            distribution_df=distribution_df,
            setting_results=setting_results,
            final_params=final_params
        )
        
        self.log("\n" + "="*80)
        self.log("✓ Data Split Analysis Complete")
        self.log("="*80)
        
        return final_params
