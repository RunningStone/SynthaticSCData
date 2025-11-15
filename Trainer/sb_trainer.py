#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Schrödinger Bridge Trainer
Simplified trainer for SB model on real time series data
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import json


class SBTrainer:
    """
    Trainer for Schrödinger Bridge model
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
        grad_clip_norm: float = 5.0
    ):
        """
        Args:
            model: SchrodingerBridgeModel instance
            train_loader: Training data loader
            test_loader: Test data loader
            learning_rate: Learning rate
            device: Device to train on
            output_dir: Output directory for checkpoints
            weight_decay: Weight decay for regularization
            grad_clip_norm: Gradient clipping norm
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip_norm = grad_clip_norm
        
        # Setup optimizer with weight decay
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
            
            # For SB, we need pairs of consecutive timepoints
            # Create pairs from the batch
            losses = []
            
            # Group by time labels
            unique_times = torch.unique(y)
            time_to_indices = {t.item(): (y == t).nonzero(as_tuple=True)[0] for t in unique_times}
            
            # Create consecutive pairs
            sorted_times = sorted(time_to_indices.keys())
            for i in range(len(sorted_times) - 1):
                t_curr = sorted_times[i]
                t_next = sorted_times[i + 1]
                
                indices_curr = time_to_indices[t_curr]
                indices_next = time_to_indices[t_next]
                
                if len(indices_curr) == 0 or len(indices_next) == 0:
                    continue
                
                # Match pairs (randomly if sizes differ)
                n_pairs = min(len(indices_curr), len(indices_next))
                if len(indices_curr) > n_pairs:
                    indices_curr = indices_curr[torch.randperm(len(indices_curr))[:n_pairs]]
                if len(indices_next) > n_pairs:
                    indices_next = indices_next[torch.randperm(len(indices_next))[:n_pairs]]
                
                x_t = X[indices_curr]
                x_next = X[indices_next]
                
                # Time values (normalized to [0, 1])
                t = torch.full((n_pairs,), float(t_curr) / len(sorted_times), device=self.device)
                dt = 1.0 / len(sorted_times)
                
                # Compute loss
                loss = self.model.compute_loss(x_t, x_next, t, dt)
                losses.append(loss)
            
            if len(losses) == 0:
                continue
            
            batch_loss = torch.mean(torch.stack(losses))
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{batch_loss.item():.6f}'})
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def evaluate(self) -> float:
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        # Note: SB model needs gradients for drift computation (autograd.grad on x),
        # but we don't want to update model parameters, so we use torch.no_grad() 
        # only for model parameters, not for input x
        with torch.set_grad_enabled(True):
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Create pairs
                losses = []
                unique_times = torch.unique(y)
                time_to_indices = {t.item(): (y == t).nonzero(as_tuple=True)[0] for t in unique_times}
                
                sorted_times = sorted(time_to_indices.keys())
                for i in range(len(sorted_times) - 1):
                    t_curr = sorted_times[i]
                    t_next = sorted_times[i + 1]
                    
                    indices_curr = time_to_indices[t_curr]
                    indices_next = time_to_indices[t_next]
                    
                    if len(indices_curr) == 0 or len(indices_next) == 0:
                        continue
                    
                    n_pairs = min(len(indices_curr), len(indices_next))
                    if len(indices_curr) > n_pairs:
                        indices_curr = indices_curr[:n_pairs]
                    if len(indices_next) > n_pairs:
                        indices_next = indices_next[:n_pairs]
                    
                    x_t = X[indices_curr]
                    x_next = X[indices_next]
                    
                    t = torch.full((n_pairs,), float(t_curr) / len(sorted_times), device=self.device)
                    dt = 1.0 / len(sorted_times)
                    
                    # Compute loss (needs gradient for x, but not for model params)
                    loss = self.model.compute_loss(x_t, x_next, t, dt)
                    losses.append(loss.detach())  # Detach to avoid keeping computation graph
                
                if len(losses) > 0:
                    batch_loss = torch.mean(torch.stack(losses))
                    total_loss += batch_loss.item()
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
        print(f"\nTraining for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Output dir: {self.output_dir}")
        
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
                checkpoint_path = self.output_dir / 'best_model.pt'
                self.save_checkpoint('best_model.pt')
                print(f"✓ New best model saved (test loss: {test_loss:.6f})")
                print(f"  Saved to: {checkpoint_path}")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{early_stopping_patience}")
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_name = f'checkpoint_epoch_{epoch+1}.pt'
                checkpoint_path = self.output_dir / checkpoint_name
                self.save_checkpoint(checkpoint_name)
                print(f"  Checkpoint saved to: {checkpoint_path}")
        
        # Save final model
        final_model_path = self.output_dir / 'final_model.pt'
        self.save_checkpoint('final_model.pt')
        print(f"\n✓ Final model saved to: {final_model_path}")
        
        # Save history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved to: {history_path}")
        
        print(f"\n✓ Training complete!")
        print(f"Best test loss: {self.best_test_loss:.6f}")
        print(f"All outputs saved to: {self.output_dir}")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_test_loss': self.best_test_loss
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
