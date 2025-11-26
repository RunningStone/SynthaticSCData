#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch OT Trainer for Sequential Time Point Learning

Refactored to use composition pattern with BaseTrainer logic.
BatchOT is special because it trains multiple models, so we customize more heavily.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Any
from tqdm import tqdm
import json


class BatchOTTrainer:
    """
    Trainer for Batch OT Model.
    
    This trainer is different from others because it trains multiple OT models
    (one for each consecutive time pair) with separate optimizers and schedulers.
    
    We don't inherit from BaseTrainer because the multi-model structure
    requires significant customization, but we follow the same patterns.
    """
    
    def __init__(
        self,
        model,  # BatchOTModel
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
            model: BatchOTModel instance
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
        
        # Get optimizer kwargs
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        
        betas = optimizer_kwargs.get('betas', [0.9, 0.999])
        eps = optimizer_kwargs.get('eps', 1e-8)
        
        # Create separate optimizers for each OT model
        self.optimizers = []
        self.schedulers = []
        
        for i in range(self.model.n_transitions):
            optimizer = optim.AdamW(
                self.model.ot_models[i].parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=tuple(betas),
                eps=eps
            )
            self.optimizers.append(optimizer)
            
            # Setup scheduler
            scheduler_config = scheduler_config if scheduler_config is not None else {}
            scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')
            
            if scheduler_type == 'reduce_on_plateau':
                patience = scheduler_config.get('patience', 10)
                factor = scheduler_config.get('factor', 0.5)
                min_lr = scheduler_config.get('min_lr', 1e-6)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=factor,
                    patience=patience,
                    min_lr=min_lr
                )
            else:
                scheduler = None
            
            self.schedulers.append(scheduler)
        
        # Training history for each transition
        self.history = {
            'train_loss': [[] for _ in range(self.model.n_transitions)],
            'test_loss': [[] for _ in range(self.model.n_transitions)],
            'learning_rate': [[] for _ in range(self.model.n_transitions)],
            'overall_train_loss': [],
            'overall_test_loss': []
        }
        
        self.best_test_losses = [float('inf')] * self.model.n_transitions
        self.patience_counters = [0] * self.model.n_transitions
    
    def process_batch_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Extract data for each consecutive time pair from a batch.
        
        Args:
            X: Batch data (batch_size, d)
            y: Time labels (batch_size,)
            
        Returns:
            Dictionary mapping transition_idx to {source, target} tensors
        """
        unique_times = torch.unique(y)
        time_to_indices = {
            t.item(): (y == t).nonzero(as_tuple=True)[0]
            for t in unique_times
        }
        sorted_times = sorted(time_to_indices.keys())
        
        pairs_data = {}
        
        # Extract consecutive pairs
        for i in range(len(sorted_times) - 1):
            t_source = sorted_times[i]
            t_target = sorted_times[i + 1]
            
            indices_source = time_to_indices[t_source]
            indices_target = time_to_indices[t_target]
            
            if len(indices_source) > 0 and len(indices_target) > 0:
                # Match pairs
                n_pairs = min(len(indices_source), len(indices_target))
                if len(indices_source) > n_pairs:
                    indices_source = indices_source[torch.randperm(len(indices_source))[:n_pairs]]
                if len(indices_target) > n_pairs:
                    indices_target = indices_target[torch.randperm(len(indices_target))[:n_pairs]]
                
                # Use transition index (i) as key
                pairs_data[i] = {
                    'source': X[indices_source],
                    'target': X[indices_target]
                }
        
        return pairs_data
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Track losses for each transition
        transition_losses = [[] for _ in range(self.model.n_transitions)]
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (X, y) in enumerate(pbar):
            X = X.to(self.device)
            y = y.to(self.device)
            
            # Extract time pairs from batch
            pairs_data = self.process_batch_data(X, y)
            
            # Train each transition model
            batch_losses = []
            for transition_idx in range(self.model.n_transitions):
                # Check if this transition has data in current batch
                if transition_idx in pairs_data:
                    data = pairs_data[transition_idx]
                    x_source = data['source']
                    x_target = data['target']
                    
                    # Compute loss for this transition
                    loss = self.model.compute_loss_for_transition(
                        x_source, x_target, transition_idx
                    )
                    
                    # Backward pass for this specific model
                    self.optimizers[transition_idx].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.ot_models[transition_idx].parameters(),
                        max_norm=self.grad_clip_norm
                    )
                    self.optimizers[transition_idx].step()
                    
                    transition_losses[transition_idx].append(loss.item())
                    batch_losses.append(loss.item())
            
            # Update progress bar
            if batch_losses:
                avg_batch_loss = np.mean(batch_losses)
                pbar.set_postfix({'loss': f'{avg_batch_loss:.6f}'})
        
        # Compute average losses
        epoch_losses = {}
        for i in range(self.model.n_transitions):
            if transition_losses[i]:
                epoch_losses[f'transition_{i}'] = np.mean(transition_losses[i])
            else:
                epoch_losses[f'transition_{i}'] = 0.0
        
        epoch_losses['overall'] = np.mean([v for v in epoch_losses.values() if v > 0])
        
        return epoch_losses
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test set"""
        self.model.eval()
        
        # Track losses for each transition
        transition_losses = [[] for _ in range(self.model.n_transitions)]
        
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Extract time pairs from batch
                pairs_data = self.process_batch_data(X, y)
                
                # Evaluate each transition model
                for transition_idx in range(self.model.n_transitions):
                    if transition_idx in pairs_data:
                        data = pairs_data[transition_idx]
                        x_source = data['source']
                        x_target = data['target']
                        
                        # Compute loss
                        loss = self.model.compute_loss_for_transition(
                            x_source, x_target, transition_idx
                        )
                        transition_losses[transition_idx].append(loss.item())
        
        # Compute average losses
        eval_losses = {}
        for i in range(self.model.n_transitions):
            if transition_losses[i]:
                eval_losses[f'transition_{i}'] = np.mean(transition_losses[i])
            else:
                eval_losses[f'transition_{i}'] = 0.0
        
        eval_losses['overall'] = np.mean([v for v in eval_losses.values() if v > 0])
        
        return eval_losses
    
    def train(
        self,
        epochs: int = 100,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Train the batch OT model
        
        Args:
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        print(f"\nTraining Batch OT model for {epochs} epochs...")
        print(f"Number of transitions: {self.model.n_transitions}")
        print(f"Device: {self.device}")
        print(f"Output dir: {self.output_dir}")
        
        for transition_idx, (t_start, t_end) in enumerate(self.model.time_pairs):
            print(f"  Transition {transition_idx}: {t_start} -> {t_end}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 70)
            
            # Train
            train_losses = self.train_epoch()
            
            # Evaluate
            test_losses = self.evaluate()
            
            # Update history
            for i in range(self.model.n_transitions):
                self.history['train_loss'][i].append(train_losses[f'transition_{i}'])
                self.history['test_loss'][i].append(test_losses[f'transition_{i}'])
                
                # Update learning rate
                current_lr = self.optimizers[i].param_groups[0]['lr']
                self.history['learning_rate'][i].append(current_lr)
                
                # Step scheduler
                if test_losses[f'transition_{i}'] > 0 and self.schedulers[i] is not None:
                    self.schedulers[i].step(test_losses[f'transition_{i}'])
            
            self.history['overall_train_loss'].append(train_losses['overall'])
            self.history['overall_test_loss'].append(test_losses['overall'])
            
            # Print losses
            print(f"Overall Train Loss: {train_losses['overall']:.6f}")
            print(f"Overall Test Loss: {test_losses['overall']:.6f}")
            
            # Check for best models and early stopping
            all_converged = True
            for i in range(self.model.n_transitions):
                test_loss = test_losses[f'transition_{i}']
                if test_loss > 0:  # Only consider transitions with data
                    if test_loss < self.best_test_losses[i]:
                        self.best_test_losses[i] = test_loss
                        self.patience_counters[i] = 0
                        # Save individual model
                        t_start, t_end = self.model.time_pairs[i]
                        self.save_transition_model(i, f'best_model_{t_start}_to_{t_end}.pt')
                    else:
                        self.patience_counters[i] += 1
                    
                    if self.patience_counters[i] < early_stopping_patience:
                        all_converged = False
            
            # Early stopping if all transitions have converged
            if all_converged:
                print(f"\nEarly stopping: All transitions converged after {epoch+1} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Save final models
        self.save_checkpoint('final_model.pt')
        
        # Save history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n✓ Training complete!")
        
        return self.history
    
    def save_transition_model(self, transition_idx: int, filename: str):
        """Save a specific transition model"""
        model_state = self.model.ot_models[transition_idx].state_dict()
        torch.save(model_state, self.output_dir / filename)
    
    def save_checkpoint(self, filename: str):
        """Save complete checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dicts': [opt.state_dict() for opt in self.optimizers],
            'history': self.history,
            'best_test_losses': self.best_test_losses,
            'time_pairs': self.model.time_pairs
        }
        
        # Add scheduler states if they exist
        scheduler_states = []
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler_states.append(scheduler.state_dict())
            else:
                scheduler_states.append(None)
        checkpoint['scheduler_state_dicts'] = scheduler_states
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load checkpoint"""
        checkpoint = torch.load(self.output_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        for i, opt_state in enumerate(checkpoint['optimizer_state_dicts']):
            self.optimizers[i].load_state_dict(opt_state)
        
        # Load scheduler states if they exist
        if 'scheduler_state_dicts' in checkpoint:
            for i, sched_state in enumerate(checkpoint['scheduler_state_dicts']):
                if sched_state is not None and self.schedulers[i] is not None:
                    self.schedulers[i].load_state_dict(sched_state)
        
        self.history = checkpoint['history']
        self.best_test_losses = checkpoint['best_test_losses']
        print(f"✓ Loaded checkpoint from {filename}")
