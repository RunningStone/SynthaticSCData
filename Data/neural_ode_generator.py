#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural ODE-based Continuous Time Data Generator
Uses Neural ODEs to learn smooth trajectories between time points
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ODEFunc(nn.Module):
    """
    Neural ODE function: dx/dt = f(x, t)
    """
    def __init__(self, dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        layers = []
        in_dim = dim + 1  # x + t
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, x):
        """
        Args:
            t: Time (scalar)
            x: State (batch_size, dim)
        Returns:
            dx/dt: (batch_size, dim)
        """
        batch_size = x.shape[0]
        t_vec = torch.ones(batch_size, 1, device=x.device) * t
        xt = torch.cat([x, t_vec], dim=1)
        return self.net(xt)


class NeuralODETrainer:
    """
    Train Neural ODE on real trajectory data
    """
    def __init__(
        self,
        dim: int,
        hidden_dims: List[int] = [256, 256],
        device: str = 'cuda'
    ):
        self.device = device
        self.ode_func = ODEFunc(dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.ode_func.parameters(), lr=1e-3)
    
    def train(
        self,
        X_pairs: List[Tuple[np.ndarray, np.ndarray, float, float]],
        n_epochs: int = 100,
        batch_size: int = 128
    ):
        """
        Train on pairs of (X_start, X_end, t_start, t_end)
        
        Args:
            X_pairs: List of (X_start, X_end, t_start, t_end)
            n_epochs: Training epochs
            batch_size: Batch size
        """
        print(f"\nTraining Neural ODE for {n_epochs} epochs...")
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0
            
            # Shuffle pairs
            np.random.shuffle(X_pairs)
            
            for i in range(0, len(X_pairs), batch_size):
                batch = X_pairs[i:i+batch_size]
                
                # Prepare batch
                X_starts = []
                X_ends = []
                t_spans = []
                
                for X_start, X_end, t_start, t_end in batch:
                    X_starts.append(X_start)
                    X_ends.append(X_end)
                    t_spans.append([t_start, t_end])
                
                X_starts = torch.FloatTensor(np.array(X_starts)).to(self.device)
                X_ends = torch.FloatTensor(np.array(X_ends)).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                losses = []
                for j, (t_start, t_end) in enumerate(t_spans):
                    t_eval = torch.FloatTensor([t_start, t_end]).to(self.device)
                    x0 = X_starts[j:j+1]
                    
                    # Solve ODE
                    pred_traj = odeint(self.ode_func, x0, t_eval, method='dopri5')
                    pred_end = pred_traj[-1]
                    
                    # Loss: match end point
                    loss = torch.mean((pred_end - X_ends[j:j+1]) ** 2)
                    losses.append(loss)
                
                batch_loss = torch.mean(torch.stack(losses))
                
                # Backward pass
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ode_func.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        print("✓ Neural ODE training completed")
    
    def generate_trajectory(
        self,
        X_start: np.ndarray,
        t_start: float,
        t_end: float,
        n_steps: int = 100
    ) -> np.ndarray:
        """
        Generate trajectory from X_start to t_end
        
        Args:
            X_start: Starting state (n_cells, dim)
            t_start: Start time
            t_end: End time
            n_steps: Number of time steps
        
        Returns:
            Trajectory (n_steps, n_cells, dim)
        """
        self.ode_func.eval()
        
        with torch.no_grad():
            X_start_t = torch.FloatTensor(X_start).to(self.device)
            t_eval = torch.linspace(t_start, t_end, n_steps).to(self.device)
            
            # Solve ODE
            trajectory = odeint(self.ode_func, X_start_t, t_eval, method='dopri5')
            
            return trajectory.cpu().numpy()


class NeuralODEDataGenerator:
    """
    Generate continuous time data using Neural ODE
    """
    def __init__(
        self,
        adata_path: str,
        n_hvg: int = 100,
        obs_time_column: str = 'Ground_truth',
        time_labels: List[str] = None,
        time_label_order: List[str] = None,
        time_intervals: Dict[Tuple[str, str], float] = None,
        target_time_range: Tuple[float, float] = (0.0, 168.0),
        time_granularity: float = 1.0,
        cells_per_label: int = 2000,
        device: str = 'cuda',
        ode_hidden_dims: List[int] = [256, 256],
        ode_epochs: int = 50
    ):
        """
        Initialize Neural ODE-based data generator
        
        Args:
            adata_path: Path to AnnData file
            n_hvg: Number of highly variable genes
            obs_time_column: Column name for time labels
            time_labels: List of discrete time labels
            time_label_order: Order of time labels
            time_intervals: Time intervals between labels (in hours)
            target_time_range: Target continuous time range
            time_granularity: Minimum time step (hours)
            cells_per_label: Number of cells to sample per label
            device: Device for Neural ODE training
            ode_hidden_dims: Hidden dimensions for ODE network
            ode_epochs: Training epochs for Neural ODE
        """
        self.adata_path = adata_path
        self.n_hvg = n_hvg
        self.obs_time_column = obs_time_column
        self.time_labels = time_labels
        self.time_label_order = time_label_order
        self.time_intervals = time_intervals
        self.target_time_range = target_time_range
        self.time_granularity = time_granularity
        self.cells_per_label = cells_per_label
        self.device = device
        self.ode_hidden_dims = ode_hidden_dims
        self.ode_epochs = ode_epochs
        
        self.adata_original = None
        self.adata_hvg = None
        self.hvg_genes = None
        self.time_mapping = None
        self.ode_trainer = None
    
    def load_and_preprocess(self):
        """Load data, sample cells, compute HVGs"""
        print(f"Loading data from {self.adata_path}...")
        self.adata_original = sc.read_h5ad(self.adata_path)
        print(f"Loaded: {self.adata_original.shape[0]} cells × {self.adata_original.shape[1]} genes")
        
        # Filter to specified time labels
        mask = self.adata_original.obs[self.obs_time_column].isin(self.time_labels)
        self.adata_original = self.adata_original[mask].copy()
        print(f"Filtered to {len(self.time_labels)} time labels: {self.adata_original.shape[0]} cells")
        
        # Sample cells per label
        sampled_indices = []
        for label in self.time_label_order:
            label_mask = self.adata_original.obs[self.obs_time_column] == label
            label_indices = np.where(label_mask)[0]
            
            if len(label_indices) > self.cells_per_label:
                sampled = np.random.choice(label_indices, self.cells_per_label, replace=False)
            else:
                sampled = label_indices
            
            sampled_indices.extend(sampled)
            print(f"  {label}: sampled {len(sampled)} cells")
        
        self.adata_original = self.adata_original[sampled_indices].copy()
        print(f"Total sampled: {self.adata_original.shape[0]} cells")
        
        # Clean data
        X = self.adata_original.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.nan_to_num(X, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
        self.adata_original.X = X
        
        # Compute HVGs
        if 'highly_variable' not in self.adata_original.var.columns:
            print(f"Computing top {self.n_hvg} HVGs...")
            try:
                sc.pp.highly_variable_genes(
                    self.adata_original,
                    n_top_genes=self.n_hvg,
                    flavor='seurat_v3',
                    subset=False
                )
            except (ImportError, ValueError) as e:
                print(f"  seurat_v3 failed, using variance-based selection...")
                variances = np.var(X, axis=0)
                top_indices = np.argsort(variances)[-self.n_hvg:]
                self.adata_original.var['highly_variable'] = False
                self.adata_original.var.iloc[top_indices, self.adata_original.var.columns.get_loc('highly_variable')] = True
        
        # Get HVG genes
        if 'highly_variable' in self.adata_original.var.columns:
            hvg_mask = self.adata_original.var['highly_variable']
            self.hvg_genes = self.adata_original.var_names[hvg_mask][:self.n_hvg].tolist()
        else:
            self.hvg_genes = self.adata_original.var_names[:self.n_hvg].tolist()
        
        print(f"Selected {len(self.hvg_genes)} HVGs")
        
        # Create HVG-filtered AnnData
        self.adata_hvg = self.adata_original[:, self.hvg_genes].copy()
        
        # Build time mapping
        self._build_time_mapping()
    
    def _build_time_mapping(self):
        """Build mapping from discrete labels to continuous time"""
        self.time_mapping = {}
        current_time = self.target_time_range[0]
        
        for i, label in enumerate(self.time_label_order):
            self.time_mapping[label] = current_time
            
            if i < len(self.time_label_order) - 1:
                next_label = self.time_label_order[i + 1]
                interval = self.time_intervals.get((label, next_label), 1.0)
                current_time += interval
        
        print(f"\nTime mapping: {self.time_mapping}")
    
    def train_neural_ode(self):
        """Train Neural ODE on real cell pairs"""
        print("\n" + "="*70)
        print("Training Neural ODE")
        print("="*70)
        
        # Prepare training pairs: consecutive time points
        X_pairs = []
        
        for i in range(len(self.time_label_order) - 1):
            label_start = self.time_label_order[i]
            label_end = self.time_label_order[i + 1]
            
            t_start = self.time_mapping[label_start]
            t_end = self.time_mapping[label_end]
            
            # Get cells
            mask_start = self.adata_hvg.obs[self.obs_time_column] == label_start
            mask_end = self.adata_hvg.obs[self.obs_time_column] == label_end
            
            X_start = self.adata_hvg[mask_start].X
            X_end = self.adata_hvg[mask_end].X
            
            if hasattr(X_start, 'toarray'):
                X_start = X_start.toarray()
            if hasattr(X_end, 'toarray'):
                X_end = X_end.toarray()
            
            # Create pairs (match cells randomly)
            n_pairs = min(len(X_start), len(X_end))
            for j in range(n_pairs):
                X_pairs.append((X_start[j], X_end[j], t_start, t_end))
        
        print(f"Created {len(X_pairs)} training pairs")
        
        # Initialize and train Neural ODE
        self.ode_trainer = NeuralODETrainer(
            dim=self.n_hvg,
            hidden_dims=self.ode_hidden_dims,
            device=self.device
        )
        
        self.ode_trainer.train(X_pairs, n_epochs=self.ode_epochs, batch_size=64)
    
    def generate_continuous_data(self) -> sc.AnnData:
        """Generate continuous time data using Neural ODE"""
        print("\n" + "="*70)
        print("Generating Continuous Time Data with Neural ODE")
        print("="*70)
        
        all_cells = []
        all_times = []
        all_time_slices = []
        all_is_real = []
        all_original_labels = []
        
        # Add real cells
        for label in self.time_label_order:
            mask = self.adata_hvg.obs[self.obs_time_column] == label
            X_real = self.adata_hvg[mask].X
            if hasattr(X_real, 'toarray'):
                X_real = X_real.toarray()
            
            t = self.time_mapping[label]
            n_cells = X_real.shape[0]
            
            all_cells.append(X_real)
            all_times.extend([t] * n_cells)
            all_time_slices.extend([f"T{t}"] * n_cells)
            all_is_real.extend([True] * n_cells)
            all_original_labels.extend([label] * n_cells)
        
        # Generate intermediate cells using Neural ODE
        for i in range(len(self.time_label_order) - 1):
            label_start = self.time_label_order[i]
            label_end = self.time_label_order[i + 1]
            
            t_start = self.time_mapping[label_start]
            t_end = self.time_mapping[label_end]
            
            # Get starting cells
            mask_start = self.adata_hvg.obs[self.obs_time_column] == label_start
            X_start = self.adata_hvg[mask_start].X
            if hasattr(X_start, 'toarray'):
                X_start = X_start.toarray()
            
            # Generate trajectory
            n_steps = int((t_end - t_start) / self.time_granularity)
            if n_steps <= 1:
                continue
            
            print(f"Generating {n_steps} steps from {label_start} ({t_start}h) to {label_end} ({t_end}h)...")
            
            trajectory = self.ode_trainer.generate_trajectory(
                X_start, t_start, t_end, n_steps
            )
            
            # Add intermediate points (skip first and last)
            for step_idx in range(1, n_steps - 1):
                X_gen = trajectory[step_idx]
                t_gen = t_start + step_idx * self.time_granularity
                
                all_cells.append(X_gen)
                all_times.extend([t_gen] * X_gen.shape[0])
                all_time_slices.extend([f"T{t_gen}"] * X_gen.shape[0])
                all_is_real.extend([False] * X_gen.shape[0])
                all_original_labels.extend([np.nan] * X_gen.shape[0])
        
        # Create AnnData
        X_all = np.vstack(all_cells)
        
        adata_continuous = sc.AnnData(
            X=X_all,
            var=pd.DataFrame(index=self.hvg_genes)
        )
        
        adata_continuous.obs['continuous_time'] = all_times
        adata_continuous.obs['time_slice'] = all_time_slices
        adata_continuous.obs['is_real'] = all_is_real
        adata_continuous.obs['original_label'] = all_original_labels
        
        print(f"\n✓ Generated continuous time data:")
        print(f"  Total cells: {adata_continuous.shape[0]}")
        print(f"  Real cells: {sum(all_is_real)}")
        print(f"  Generated cells: {sum(not x for x in all_is_real)}")
        print(f"  Genes (HVG): {adata_continuous.shape[1]}")
        
        return adata_continuous
    
    def save_continuous_data(self, output_path: str) -> sc.AnnData:
        """Complete pipeline: load, train ODE, generate, save"""
        self.load_and_preprocess()
        self.train_neural_ode()
        adata_continuous = self.generate_continuous_data()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        adata_continuous.write_h5ad(output_path)
        print(f"\n✓ Saved to: {output_path}")
        
        return adata_continuous


def create_neural_ode_emt_generator(
    output_path: str,
    n_hvg: int = 100,
    cells_per_label: int = 2000,
    time_granularity: float = 1.0,
    device: str = 'cuda',
    ode_epochs: int = 50
) -> NeuralODEDataGenerator:
    """
    Create Neural ODE generator with default EMT dataset configuration
    """
    generator = NeuralODEDataGenerator(
        adata_path="/home/pan/Experiments/EXPs/2024_EMT_LM_workspace/EMT-LM_Data/Step_0_data/preprocessed/GSE147405_Cook/2024_12_04_Cook_emt_dataset_with_removal_scBERT.h5ad",
        n_hvg=n_hvg,
        obs_time_column='Ground_truth',
        time_labels=['0d', '8h', '1d', '3d', '7d'],
        time_label_order=['0d', '8h', '1d', '3d', '7d'],
        time_intervals={
            ('0d', '8h'): 8.0,
            ('8h', '1d'): 16.0,
            ('1d', '3d'): 48.0,
            ('3d', '7d'): 96.0
        },
        target_time_range=(0.0, 168.0),
        time_granularity=time_granularity,
        cells_per_label=cells_per_label,
        device=device,
        ode_hidden_dims=[256, 256],
        ode_epochs=ode_epochs
    )
    
    return generator


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate continuous time data with Neural ODE')
    parser.add_argument('--output', type=str, required=True, help='Output h5ad file')
    parser.add_argument('--n_hvg', type=int, default=100, help='Number of HVGs')
    parser.add_argument('--cells_per_label', type=int, default=2000, help='Cells per label')
    parser.add_argument('--granularity', type=float, default=1.0, help='Time granularity (hours)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--ode_epochs', type=int, default=50, help='ODE training epochs')
    
    args = parser.parse_args()
    
    generator = create_neural_ode_emt_generator(
        output_path=args.output,
        n_hvg=args.n_hvg,
        cells_per_label=args.cells_per_label,
        time_granularity=args.granularity,
        device=args.device,
        ode_epochs=args.ode_epochs
    )
    
    generator.save_continuous_data(args.output)
