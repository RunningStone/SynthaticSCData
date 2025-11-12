#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization of Model Generation Results

This script visualizes the generation quality of different models by:
1. Loading original data and splitting it using biology_split
2. Sampling test set data from different timepoints
3. Loading trained models and generating target timepoint samples
4. Computing PHATE and LMNN+PCA embeddings on original data
5. Projecting generated samples into the same embedding space
6. Creating comprehensive comparison plots
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Dimensionality reduction
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import phate
from metric_learn import LMNN

# Project imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from Data import create_default_emt_data_loader
from Model import (
    SchrodingerBridgeModel,
    MLPlus_SchrodingerBridgeModel,
    OptimalTransportModel,
    ConditionalVAEModel
)


class GenerationVisualizer:
    """
    Visualize generation results from trained models
    """
    
    def __init__(
        self,
        file_path: str = None,
        n_hvg: int = 100,
        output_dir: str = './visualization_outputs',
        device: str = 'cuda',
        random_seed: int = 42
    ):
        """
        Initialize visualizer
        
        Args:
            file_path: Path to h5ad file
            n_hvg: Number of highly variable genes
            output_dir: Output directory for plots
            device: Device for model inference
            random_seed: Random seed
        """
        self.file_path = file_path
        self.n_hvg = n_hvg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        # Data loader
        self.loader = None
        
        # Original data samples
        self.X_original = None
        self.y_original = None
        self.time_labels = None
        
        # Generated data
        self.generated_data = {}  # model_name -> generated samples
        
        # Embedding transformers
        self.phate_op = None
        self.lmnn_pca_scaler = None
        self.lmnn_op = None
        self.pca_op = None
        
        # Embeddings
        self.phate_embeddings = {}  # 'original' and model names
        self.lmnn_pca_embeddings = {}  # 'original' and model names
        
    def load_and_sample_data(
        self,
        n_samples_per_timepoint: int = 500
    ):
        """
        Load data and sample from test set
        
        Args:
            n_samples_per_timepoint: Number of samples per timepoint from test set
        """
        print("="*80)
        print("Step 1: Loading and Sampling Data")
        print("="*80)
        
        # Load data with biology split
        self.loader = create_default_emt_data_loader(
            file_path=self.file_path,
            n_hvg=self.n_hvg
        )
        self.loader.load_and_analyze()
        self.loader.validate_biology_split()
        
        # Get HVG expression matrix
        X = self.loader.adata_hvg.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Get time labels as integers
        time_to_idx = {label: idx for idx, label in enumerate(self.loader.time_label_order)}
        y = np.array([time_to_idx[t] for t in self.loader.adata_hvg.obs[self.loader.obs_time_column]])
        
        # Sample from test set only
        X_samples_list = []
        y_samples_list = []
        
        for time_label in self.loader.time_label_order:
            time_idx = time_to_idx[time_label]
            time_mask = (y == time_idx)
            
            # Get test set samples for this timepoint
            test_time_mask = time_mask & self.loader.test_mask
            test_indices = np.where(test_time_mask)[0]
            
            # Sample n_samples_per_timepoint
            if len(test_indices) > n_samples_per_timepoint:
                sampled_indices = np.random.choice(
                    test_indices, 
                    n_samples_per_timepoint, 
                    replace=False
                )
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
        
    def load_models_and_generate(
        self,
        model_configs: Dict[str, Dict],
        n_generate_per_model: int = 500
    ):
        """
        Load trained models and generate target timepoint samples
        
        Args:
            model_configs: Dictionary with model configurations
                {
                    'model_name': {
                        'type': 'sb' | 'sb_mlplus' | 'ot' | 'vae',
                        'checkpoint_path': path to model checkpoint,
                        'model_kwargs': kwargs for model initialization
                    }
                }
            n_generate_per_model: Number of samples to generate per model
        """
        print("\n" + "="*80)
        print("Step 2: Loading Models and Generating Samples")
        print("="*80)
        
        # Get first and last timepoint indices
        first_time_idx = 0
        last_time_idx = len(self.time_labels) - 1
        
        # Get source samples (first timepoint from test set)
        source_mask = (self.y_original == first_time_idx)
        source_samples = self.X_original[source_mask]
        
        # Sample n_generate_per_model source samples
        if len(source_samples) > n_generate_per_model:
            source_indices = np.random.choice(
                len(source_samples),
                n_generate_per_model,
                replace=False
            )
            source_samples = source_samples[source_indices]
        
        print(f"\nSource samples ({self.time_labels[first_time_idx]}): {source_samples.shape[0]} cells")
        print(f"Target timepoint: {self.time_labels[last_time_idx]}")
        print()
        
        for model_name, config in model_configs.items():
            print(f"Processing {model_name}...")
            
            model_type = config['type']
            checkpoint_path = Path(config['checkpoint_path'])
            model_kwargs = config.get('model_kwargs', {})
            
            # Initialize model
            if model_type == 'sb':
                model = SchrodingerBridgeModel(**model_kwargs).to(self.device)
            elif model_type == 'sb_mlplus':
                model = MLPlus_SchrodingerBridgeModel(**model_kwargs).to(self.device)
            elif model_type == 'ot':
                model = OptimalTransportModel(**model_kwargs).to(self.device)
            elif model_type == 'vae':
                model = ConditionalVAEModel(**model_kwargs).to(self.device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load checkpoint if exists
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # For VAE models, set normalization_fitted flag after loading
                if model_type == 'vae' and hasattr(model, 'normalization_fitted'):
                    model.normalization_fitted = True
                
                print(f"  ✓ Loaded checkpoint from {checkpoint_path}")
            else:
                print(f"  ⚠️  Warning: Checkpoint not found at {checkpoint_path}")
                print(f"     Skipping {model_name}")
                continue
            
            # Generate samples
            model.eval()
            # Note: Don't use torch.no_grad() for SB models as they need gradients for drift computation
            source_tensor = torch.FloatTensor(source_samples).to(self.device)
            
            # Create time grid from source to target
            # For generation, we only need the final timepoint
            time_grid = torch.tensor([0.0, 1.0], device=self.device)
            
            if model_type in ['sb', 'sb_mlplus']:
                # SB models need gradients for drift computation
                trajectory = model.generate_trajectory(
                    source_tensor,
                    time_grid,
                    method='deterministic'  # Use deterministic for reproducibility
                )
                # Take the last timepoint (target)
                generated = trajectory[:, -1, :].detach()
            elif model_type in ['ot', 'vae']:
                # OT and VAE models can use no_grad
                with torch.no_grad():
                    trajectory = model.generate_trajectory(
                        source_tensor,
                        time_grid,
                        method='deterministic'
                    )
                    # Take the last timepoint (target)
                    generated = trajectory[:, -1, :]
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            generated_np = generated.cpu().numpy()
            
            self.generated_data[model_name] = generated_np
            print(f"  ✓ Generated {generated_np.shape[0]} samples")
        
        print("\n" + "="*80)
        
    def compute_embeddings(self):
        """
        Compute PHATE and Neural Contrastive embeddings on original data
        and project generated data
        """
        print("\n" + "="*80)
        print("Step 3: Computing Embeddings")
        print("="*80)
        
        # Compute PHATE on original data
        print("\nComputing PHATE embeddings...")
        self.phate_op = phate.PHATE(
            n_components=2,
            knn=5,
            decay=40,
            n_jobs=-1,
            random_state=self.random_seed,
            verbose=0
        )
        self.phate_embeddings['original'] = self.phate_op.fit_transform(self.X_original)
        print(f"  ✓ PHATE embeddings computed: {self.phate_embeddings['original'].shape}")
        
        # Project generated data using PHATE
        for model_name, generated_data in self.generated_data.items():
            self.phate_embeddings[model_name] = self.phate_op.transform(generated_data)
            print(f"  ✓ Projected {model_name}: {self.phate_embeddings[model_name].shape}")
        
        # Compute LMNN+PCA on original data
        print("\nComputing LMNN+PCA embeddings...")
        
        # Standardize data
        self.lmnn_pca_scaler = StandardScaler()
        X_scaled = self.lmnn_pca_scaler.fit_transform(self.X_original)
        
        # Apply LMNN (use time labels for metric learning)
        print("  Training LMNN...")
        self.lmnn_op = LMNN(
            n_components=min(50, self.X_original.shape[1]),
            k=5,
            learn_rate=1e-6,
            max_iter=100,
            verbose=False,
            random_state=self.random_seed
        )
        X_lmnn = self.lmnn_op.fit_transform(X_scaled, self.y_original)
        print(f"  ✓ LMNN transformation: {X_lmnn.shape}")
        
        # Apply PCA
        self.pca_op = PCA(n_components=2, random_state=self.random_seed)
        self.lmnn_pca_embeddings['original'] = self.pca_op.fit_transform(X_lmnn)
        print(f"  ✓ LMNN+PCA embeddings computed: {self.lmnn_pca_embeddings['original'].shape}")
        
        # Project generated data using LMNN+PCA
        for model_name, generated_data in self.generated_data.items():
            # Scale -> LMNN -> PCA
            gen_scaled = self.lmnn_pca_scaler.transform(generated_data)
            gen_lmnn = self.lmnn_op.transform(gen_scaled)
            self.lmnn_pca_embeddings[model_name] = self.pca_op.transform(gen_lmnn)
            print(f"  ✓ Projected {model_name}: {self.lmnn_pca_embeddings[model_name].shape}")
        
        print("\n" + "="*80)
        
    def create_visualization(
        self,
        embedding_type: str = 'phate',
        figsize: Tuple[int, int] = (24, 4)
    ):
        """
        Create comprehensive visualization plots
        
        Args:
            embedding_type: 'phate' or 'lmnn_pca'
            figsize: Figure size
        """
        print("\n" + "="*80)
        print(f"Step 4: Creating Visualization ({embedding_type.upper()})")
        print("="*80)
        
        # Select embeddings
        if embedding_type == 'phate':
            embeddings = self.phate_embeddings
        else:
            embeddings = self.lmnn_pca_embeddings
        
        original_emb = embeddings['original']
        model_names = list(self.generated_data.keys())
        n_models = len(model_names)
        
        # Create figure with 6 subplots
        fig, axes = plt.subplots(1, 6, figsize=figsize)
        
        # Define colors for timepoints (darker for original)
        time_colors_dark = plt.cm.tab10(np.linspace(0, 0.9, len(self.time_labels)))
        time_colors_bright = plt.cm.Set1(np.linspace(0, 0.9, len(self.time_labels)))
        
        # Define colors for models
        model_colors = plt.cm.Set2(np.linspace(0, 0.9, n_models))
        
        # Get last timepoint index
        last_time_idx = len(self.time_labels) - 1
        
        # ========== Subplot 1: All original data with timepoint colors ==========
        ax = axes[0]
        for time_idx, time_label in enumerate(self.time_labels):
            mask = (self.y_original == time_idx)
            ax.scatter(
                original_emb[mask, 0],
                original_emb[mask, 1],
                c=[time_colors_dark[time_idx]],
                label=time_label,
                alpha=0.6,
                s=20,
                edgecolors='none'
            )
        ax.set_title('Original Data\n(All Timepoints)', fontweight='bold', fontsize=12)
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.set_xlabel(f'{embedding_type.upper()} 1')
        ax.set_ylabel(f'{embedding_type.upper()} 2')
        ax.grid(alpha=0.3)
        
        # ========== Subplots 2-5: Original + each model's generated data ==========
        for model_idx, model_name in enumerate(model_names):
            ax = axes[1 + model_idx]
            
            # Plot original data (darker colors)
            for time_idx, time_label in enumerate(self.time_labels):
                mask = (self.y_original == time_idx)
                ax.scatter(
                    original_emb[mask, 0],
                    original_emb[mask, 1],
                    c=[time_colors_dark[time_idx]],
                    alpha=0.4,
                    s=15,
                    edgecolors='none',
                    label=f'{time_label} (orig)' if model_idx == 0 else None
                )
            
            # Plot generated data (brighter color)
            gen_emb = embeddings[model_name]
            ax.scatter(
                gen_emb[:, 0],
                gen_emb[:, 1],
                c=[time_colors_bright[last_time_idx]],
                alpha=0.8,
                s=30,
                edgecolors='black',
                linewidths=0.5,
                label=f'{model_name} (gen)',
                marker='*'
            )
            
            ax.set_title(f'Original + {model_name}\nGenerated', fontweight='bold', fontsize=12)
            ax.legend(loc='best', fontsize=7, framealpha=0.9)
            ax.set_xlabel(f'{embedding_type.upper()} 1')
            ax.set_ylabel(f'{embedding_type.upper()} 2')
            ax.grid(alpha=0.3)
        
        # ========== Subplot 6: Last timepoint original + all models generated ==========
        ax = axes[5]
        
        # Plot last timepoint original data (gray)
        last_time_mask = (self.y_original == last_time_idx)
        ax.scatter(
            original_emb[last_time_mask, 0],
            original_emb[last_time_mask, 1],
            c='gray',
            alpha=0.5,
            s=20,
            edgecolors='none',
            label=f'{self.time_labels[last_time_idx]} (orig)'
        )
        
        # Plot all models' generated data (different colors)
        for model_idx, model_name in enumerate(model_names):
            gen_emb = embeddings[model_name]
            ax.scatter(
                gen_emb[:, 0],
                gen_emb[:, 1],
                c=[model_colors[model_idx]],
                alpha=0.7,
                s=30,
                edgecolors='black',
                linewidths=0.5,
                label=model_name,
                marker='*'
            )
        
        ax.set_title(f'Target Timepoint\n(Original + All Models)', fontweight='bold', fontsize=12)
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.set_xlabel(f'{embedding_type.upper()} 1')
        ax.set_ylabel(f'{embedding_type.upper()} 2')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / f'generation_comparison_{embedding_type}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization to: {output_path}")
        
        # Also save as PDF
        output_path_pdf = self.output_dir / f'generation_comparison_{embedding_type}.pdf'
        plt.savefig(output_path_pdf, bbox_inches='tight')
        print(f"✓ Saved PDF to: {output_path_pdf}")
        
        plt.close()
        
        print("="*80)
        
    def run_full_pipeline(
        self,
        model_configs: Dict[str, Dict],
        n_samples_per_timepoint: int = 500,
        n_generate_per_model: int = 500
    ):
        """
        Run the full visualization pipeline
        
        Args:
            model_configs: Model configurations for loading and generation
            n_samples_per_timepoint: Number of samples per timepoint from test set
            n_generate_per_model: Number of samples to generate per model
        """
        print("\n" + "="*80)
        print("GENERATION VISUALIZATION PIPELINE")
        print("="*80)
        
        # Step 1: Load and sample data
        self.load_and_sample_data(n_samples_per_timepoint)
        
        # Step 2: Load models and generate
        self.load_models_and_generate(model_configs, n_generate_per_model)
        
        # Step 3: Compute embeddings
        self.compute_embeddings()
        
        # Step 4: Create visualizations
        self.create_visualization(embedding_type='phate')
        self.create_visualization(embedding_type='lmnn_pca')
        
        print("\n" + "="*80)
        print("VISUALIZATION PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {self.output_dir}")
        print()


if __name__ == "__main__":
    # Example usage
    visualizer = GenerationVisualizer(
        n_hvg=500,
        output_dir='./visualization_outputs',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Define model configurations
    model_configs = {
        'SB_S1': {
            'type': 'sb',
            'checkpoint_path': './outputs/setting1/sb_model/best_model.pt',
            'model_kwargs': {
                'dimension': 500,
                'hidden_dims': [512, 512, 512, 512],
                'time_embedding_dim': 64,
                'dropout': 0.1,
                'diffusion_coeff': 0.1
            }
        },
        'OT_S1': {
            'type': 'ot',
            'checkpoint_path': './outputs/setting1/ot_model/best_model.pt',
            'model_kwargs': {
                'dimension': 500,
                'hidden_dims': [512, 512, 512, 512],
                'activation': 'relu',
                'dropout': 0.1,
                'use_residual': True
            }
        },
        'VAE_S1': {
            'type': 'vae',
            'checkpoint_path': './outputs/setting1/vae_model/best_model.pt',
            'model_kwargs': {
                'dimension': 500,
                'latent_dim': 128,
                'hidden_dims': [512, 256],
                'activation': 'relu',
                'dropout': 0.1,
                'beta': 1.0
            }
        },
        'SB_MLPlus_S2': {
            'type': 'sb_mlplus',
            'checkpoint_path': './outputs/setting2/sb_mlplus_model/best_model.pt',
            'model_kwargs': {
                'dimension': 500,
                'hidden_dim': 512,
                'n_blocks': 4,
                'time_embedding_dim': 64,
                'n_time_frequencies': 10,
                'dropout': 0.1,
                'diffusion_coeff': 0.1
            }
        }
    }
    
    # Run pipeline
    visualizer.run_full_pipeline(
        model_configs=model_configs,
        n_samples_per_timepoint=500,
        n_generate_per_model=500
    )
