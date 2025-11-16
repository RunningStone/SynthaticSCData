#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Trainer for OT and VAE models

Handles training for models that learn direct source->target mappings
(unlike SB which learns time-dependent drift fields)
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import json


class UnifiedTrainer:
    """
    Unified trainer for OT and VAE models.
    
    These models learn direct mappings from source to target,
    so the training loop is simpler than SB models.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 5e-4,
        device: str = 'cuda',
        output_dir: str = './outputs',
        weight_decay: float = 1e-5,
        grad_clip_norm: float = 5.0,
        model_type: str = 'ot'  # 'ot' or 'vae'
    ):
        """
        Args:
            model: Model instance (OT or VAE)
            train_loader: Training data loader
            test_loader: Test data loader
            learning_rate: Learning rate
            device: Device to train on
            output_dir: Output directory for checkpoints
            weight_decay: Weight decay for regularization
            grad_clip_norm: Gradient clipping norm
            model_type: Type of model ('ot' or 'vae')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip_norm = grad_clip_norm
        self.model_type = model_type
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'learning_rate': []
        }
        
        self.best_test_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (X, y) in enumerate(pbar):
            X = X.to(self.device)
            y = y.to(self.device)
            
            # For OT/VAE, we need source-target pairs
            # Extract first and last timepoints
            unique_times = torch.unique(y)
            time_to_indices = {t.item(): (y == t).nonzero(as_tuple=True)[0] for t in unique_times}
            sorted_times = sorted(time_to_indices.keys())
            
            if len(sorted_times) < 2:
                continue
            
            # Use first and last timepoints
            t_start = sorted_times[0]
            t_end = sorted_times[-1]
            
            indices_start = time_to_indices[t_start]
            indices_end = time_to_indices[t_end]
            
            if len(indices_start) == 0 or len(indices_end) == 0:
                continue
            
            # Match pairs
            n_pairs = min(len(indices_start), len(indices_end))
            if len(indices_start) > n_pairs:
                indices_start = indices_start[torch.randperm(len(indices_start))[:n_pairs]]
            if len(indices_end) > n_pairs:
                indices_end = indices_end[torch.randperm(len(indices_end))[:n_pairs]]
            
            x_source = X[indices_start]
            x_target = X[indices_end]
            
            # Compute loss
            # Check if model needs time indices (conditional models)
            import inspect
            loss_signature = inspect.signature(self.model.compute_loss)
            if len(loss_signature.parameters) > 2:
                # Conditional model: pass time indices
                t_source_tensor = torch.full((n_pairs,), t_start, dtype=torch.long, device=self.device)
                t_target_tensor = torch.full((n_pairs,), t_end, dtype=torch.long, device=self.device)
                loss_output = self.model.compute_loss(x_source, x_target, t_source_tensor, t_target_tensor)
                
                # Handle both single loss and (loss, loss_dict) returns
                if isinstance(loss_output, tuple):
                    loss, loss_dict = loss_output
                else:
                    loss = loss_output
            else:
                # Non-conditional model: standard interface
                loss = self.model.compute_loss(x_source, x_target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def evaluate(self) -> float:
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Extract source-target pairs
                unique_times = torch.unique(y)
                time_to_indices = {t.item(): (y == t).nonzero(as_tuple=True)[0] for t in unique_times}
                sorted_times = sorted(time_to_indices.keys())
                
                if len(sorted_times) < 2:
                    continue
                
                t_start = sorted_times[0]
                t_end = sorted_times[-1]
                
                indices_start = time_to_indices[t_start]
                indices_end = time_to_indices[t_end]
                
                if len(indices_start) == 0 or len(indices_end) == 0:
                    continue
                
                n_pairs = min(len(indices_start), len(indices_end))
                x_source = X[indices_start[:n_pairs]]
                x_target = X[indices_end[:n_pairs]]
                
                # Compute loss
                # Check if model needs time indices (conditional models)
                import inspect
                loss_signature = inspect.signature(self.model.compute_loss)
                if len(loss_signature.parameters) > 2:
                    # Conditional model: pass time indices
                    t_source_tensor = torch.full((n_pairs,), t_start, dtype=torch.long, device=self.device)
                    t_target_tensor = torch.full((n_pairs,), t_end, dtype=torch.long, device=self.device)
                    loss_output = self.model.compute_loss(x_source, x_target, t_source_tensor, t_target_tensor)
                    
                    # Handle both single loss and (loss, loss_dict) returns
                    if isinstance(loss_output, tuple):
                        loss, _ = loss_output
                    else:
                        loss = loss_output
                else:
                    # Non-conditional model: standard interface
                    loss = self.model.compute_loss(x_source, x_target)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def train(
        self,
        epochs: int = 100,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Train the model
        
        Args:
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        print(f"\nTraining {self.model_type.upper()} model for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Output dir: {self.output_dir}")
        
        # For VAE models, fit normalization parameters before training
        if self.model_type == 'vae' and hasattr(self.model, 'fit_normalizer'):
            if not self.model.normalization_fitted:
                print("\n" + "="*70)
                self.model.fit_normalizer(self.train_loader)
                print("="*70)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Evaluate
            test_loss = self.evaluate()
            self.history['test_loss'].append(test_loss)
            
            # Learning rate scheduler step
            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(test_loss)
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Test Loss: {test_loss:.6f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Check if LR was reduced
            if current_lr < prev_lr:
                print(f"⚡ Learning rate reduced: {prev_lr:.2e} → {current_lr:.2e}")
            
            # Save best model
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"✓ New best model saved (test loss: {test_loss:.6f})")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        # Save history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n✓ Training complete!")
        print(f"Best test loss: {self.best_test_loss:.6f}")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_test_loss': self.best_test_loss,
            'model_type': self.model_type
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(self.output_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_test_loss = checkpoint['best_test_loss']
        print(f"✓ Loaded checkpoint from {filename}")
