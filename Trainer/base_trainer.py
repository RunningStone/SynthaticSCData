#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Trainer for Trajectory Generation Models

Provides common training infrastructure with customization points
for different model types.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm
import json
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Abstract base trainer for all trajectory generation models.
    
    Provides:
    - Common training loop structure
    - Optimizer and scheduler setup
    - Checkpoint saving/loading
    - Training history tracking
    
    Customization points:
    - process_batch_data(): Extract source-target pairs from batch
    - compute_batch_loss(): Compute loss for a batch
    - Additional hooks for model-specific logic
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
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None
    ):
        """
        Args:
            model: Model instance
            train_loader: Training data loader
            test_loader: Test data loader
            learning_rate: Learning rate
            device: Device to train on
            output_dir: Output directory for checkpoints
            weight_decay: Weight decay for regularization
            grad_clip_norm: Gradient clipping norm
            optimizer_kwargs: Additional optimizer parameters
            scheduler_config: Scheduler configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grad_clip_norm = grad_clip_norm
        
        # Setup optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        betas = optimizer_kwargs.get('betas', [0.9, 0.999])
        eps = optimizer_kwargs.get('eps', 1e-8)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=tuple(betas),
            eps=eps
        )
        
        # Setup learning rate scheduler
        self.scheduler, self.scheduler_type = self._setup_scheduler(
            scheduler_config if scheduler_config is not None else {}
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'learning_rate': []
        }
        
        self.best_test_loss = float('inf')
        self.patience_counter = 0
    
    def _setup_scheduler(self, scheduler_config: Dict) -> Tuple[Optional[Any], str]:
        """
        Setup learning rate scheduler based on config.
        
        Args:
            scheduler_config: Scheduler configuration dictionary
            
        Returns:
            (scheduler, scheduler_type): Scheduler instance and type string
        """
        scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')
        
        if scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', 200)
            eta_min = scheduler_config.get('eta_min', 1e-6)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min
            )
            return scheduler, 'cosine'
        
        elif scheduler_type == 'reduce_on_plateau':
            patience = scheduler_config.get('patience', 10)
            factor = scheduler_config.get('factor', 0.5)
            min_lr = scheduler_config.get('min_lr', 1e-6)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=min_lr
            )
            return scheduler, 'plateau'
        
        else:
            return None, 'none'
    
    @abstractmethod
    def process_batch_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        """
        Process batch data to extract source-target pairs or other required format.
        
        This is the first customization point - different models need different
        data organization:
        - SB: consecutive time pairs
        - OT/VAE: source-target pairs (first and last)
        - BatchOT: multiple consecutive pairs
        
        Args:
            X: Batch features (batch_size, d)
            y: Batch time labels (batch_size,)
            
        Returns:
            Processed data dictionary, or None if batch should be skipped
            Format depends on model type, but typically contains:
            - 'x_source': source states
            - 'x_target': target states
            - Additional model-specific fields
        """
        pass
    
    @abstractmethod
    def compute_batch_loss(
        self,
        batch_data: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute loss for processed batch data.
        
        This is the second customization point - different models compute
        loss differently based on their training objectives.
        
        Args:
            batch_data: Processed batch data from process_batch_data()
            
        Returns:
            loss: Scalar loss tensor
        """
        pass
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        This method implements the common training loop structure.
        Customization is done through process_batch_data() and compute_batch_loss().
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (X, y) in enumerate(pbar):
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Process batch data (customization point 1)
            batch_data = self.process_batch_data(X, y)
            
            if batch_data is None:
                continue
            
            # Compute loss (customization point 2)
            loss = self.compute_batch_loss(batch_data)
            
            # Backward pass (common logic)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_norm
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def evaluate(self) -> float:
        """
        Evaluate on test set.
        
        Uses the same process_batch_data() and compute_batch_loss() methods
        as training, but with gradients disabled.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        # Some models (like SB) need gradients for forward pass even during eval
        # We use set_grad_enabled to handle this, but don't update model parameters
        with torch.set_grad_enabled(self._needs_grad_for_eval()):
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Process batch data
                batch_data = self.process_batch_data(X, y)
                
                if batch_data is None:
                    continue
                
                # Compute loss
                loss = self.compute_batch_loss(batch_data)
                
                # Detach to avoid keeping computation graph
                if isinstance(loss, torch.Tensor):
                    loss = loss.detach()
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def _needs_grad_for_eval(self) -> bool:
        """
        Whether gradients are needed during evaluation.
        
        Override this for models that need gradients for forward pass
        (e.g., SB models that compute drift via autograd).
        
        Returns:
            True if gradients needed, False otherwise
        """
        return False
    
    def _step_scheduler(self, test_loss: float):
        """
        Step the learning rate scheduler.
        
        Args:
            test_loss: Current test loss
        """
        if self.scheduler is not None:
            if self.scheduler_type == 'plateau':
                self.scheduler.step(test_loss)
            elif self.scheduler_type == 'cosine':
                self.scheduler.step()
    
    def train(
        self,
        epochs: int = 100,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        print(f"\nTraining {self.model.__class__.__name__} for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Output dir: {self.output_dir}")
        
        # Pre-training hook (for model-specific initialization)
        self._pre_training_hook()
        
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
            self._step_scheduler(test_loss)
            
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
    
    def _pre_training_hook(self):
        """
        Hook called before training starts.
        
        Override this for model-specific initialization
        (e.g., fitting normalizers for VAE models).
        """
        pass
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Override this to save additional model-specific state.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_test_loss': self.best_test_loss
        }
        
        # Add scheduler state if it exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """
        Load model checkpoint.
        
        Override this to load additional model-specific state.
        """
        checkpoint = torch.load(self.output_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_test_loss = checkpoint['best_test_loss']
        
        # Load scheduler state if it exists
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✓ Loaded checkpoint from {filename}")
