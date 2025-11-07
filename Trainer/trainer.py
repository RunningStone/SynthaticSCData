"""
Model Trainer

Unified training interface for all model types.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm


class ModelTrainer:
    """
    Unified trainer for OT, SB, and VAE models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Model instance
            model_type: 'ot', 'sb', or 'vae'
            config: Configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.model_type = model_type
        self.config = config
        self.device = device
        
        # Get training config for this model type
        self.train_config = config['training'][model_type]
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        opt_type = self.train_config['optimizer']
        lr = self.train_config['learning_rate']
        weight_decay = self.train_config['weight_decay']
        
        if opt_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        sched_config = self.train_config.get('scheduler', None)
        if sched_config is None:
            return None
        
        sched_type = sched_config['type']
        
        if sched_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=sched_config['patience'],
                factor=sched_config['factor']
            )
        elif sched_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max']
            )
        else:
            return None
    
    def train(
        self,
        train_dataset: Dict,
        val_dataset: Optional[Dict] = None,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset dictionary
            val_dataset: Validation dataset dictionary (optional)
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        # Prepare data loaders
        train_loader = self._prepare_dataloader(train_dataset, shuffle=True)
        val_loader = self._prepare_dataloader(val_dataset, shuffle=False) if val_dataset else None
        
        epochs = self.train_config['epochs']
        log_interval = self.config['experiment'].get('log_interval', 10)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                
                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Save best model
                if save_dir is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(save_dir, epoch, is_best=True)
            else:
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        # ReduceLROnPlateau needs a metric even without validation
                        self.scheduler.step(train_loss)
                    else:
                        self.scheduler.step()
            
            # Logging
            if (epoch + 1) % log_interval == 0:
                log_str = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}"
                if val_loader is not None:
                    log_str += f", Val Loss: {val_loss:.6f}"
                print(log_str)
        
        # Save final model
        if save_dir is not None:
            self.save_checkpoint(save_dir, epochs, is_best=False)
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            loss = self._compute_batch_loss(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self._compute_batch_loss(batch)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def _compute_batch_loss(self, batch: Tuple) -> torch.Tensor:
        """Compute loss for a batch based on model type"""
        if self.model_type == 'ot':
            x_0, x_T = batch
            x_0 = x_0.to(self.device)
            x_T = x_T.to(self.device)
            return self.model.compute_loss(x_0, x_T)
        
        elif self.model_type == 'sb':
            x_t, x_next, t, dt = batch
            x_t = x_t.to(self.device)
            x_next = x_next.to(self.device)
            t = t.to(self.device)
            dt = dt.to(self.device)
            # Use mean dt for the batch
            return self.model.compute_loss(x_t, x_next, t, dt.mean().item())
        
        elif self.model_type == 'vae':
            x = batch[0].to(self.device)
            x_recon, mu, logvar = self.model(x)
            total_loss, _, _ = self.model.compute_loss(x, x_recon, mu, logvar)
            return total_loss
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _prepare_dataloader(
        self, dataset: Dict, shuffle: bool
    ) -> DataLoader:
        """Prepare DataLoader from dataset dictionary"""
        trajectories = dataset['trajectories']
        time_stamps = dataset['time_stamps']
        
        if self.model_type == 'ot':
            # Extract initial and final states
            x_0 = trajectories[:, :, 0, :].reshape(-1, trajectories.shape[-1])
            x_T = trajectories[:, :, -1, :].reshape(-1, trajectories.shape[-1])
            
            tensor_dataset = TensorDataset(
                torch.FloatTensor(x_0),
                torch.FloatTensor(x_T)
            )
        
        elif self.model_type == 'sb':
            # Extract consecutive time steps
            x_t_list = []
            x_next_list = []
            t_list = []
            dt_list = []
            
            n_traj, n_cells, n_time, d = trajectories.shape
            
            for traj_idx in range(n_traj):
                for t_idx in range(n_time - 1):
                    x_t = trajectories[traj_idx, :, t_idx, :]
                    x_next = trajectories[traj_idx, :, t_idx + 1, :]
                    t = time_stamps[t_idx]
                    dt = time_stamps[t_idx + 1] - time_stamps[t_idx]
                    
                    x_t_list.append(x_t)
                    x_next_list.append(x_next)
                    t_list.append(np.full(n_cells, t))
                    dt_list.append(np.full(n_cells, dt))
            
            x_t_all = np.concatenate(x_t_list, axis=0)
            x_next_all = np.concatenate(x_next_list, axis=0)
            t_all = np.concatenate(t_list, axis=0)
            dt_all = np.concatenate(dt_list, axis=0)
            
            tensor_dataset = TensorDataset(
                torch.FloatTensor(x_t_all),
                torch.FloatTensor(x_next_all),
                torch.FloatTensor(t_all),
                torch.FloatTensor(dt_all)
            )
        
        elif self.model_type == 'vae':
            # Use all states for VAE training
            x_all = trajectories.reshape(-1, trajectories.shape[-1])
            tensor_dataset = TensorDataset(torch.FloatTensor(x_all))
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return DataLoader(
            tensor_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=shuffle,
            num_workers=self.config['experiment'].get('num_workers', 0)
        )
    
    def save_checkpoint(
        self, save_dir: str, epoch: int, is_best: bool = False
    ):
        """Save model checkpoint"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        if is_best:
            save_path = save_dir / f'{self.model_type}_best.pt'
        else:
            save_path = save_dir / f'{self.model_type}_final.pt'
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {checkpoint_path}")


class Trainer:
    """
    Simple trainer wrapper for continuous time data.
    Provides an easy-to-use interface without requiring complex config dictionaries.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = 'cuda',
        output_dir: Optional[str] = None
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to train on
            output_dir: Directory to save outputs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=10,
            factor=0.5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
    
    def train(self, n_epochs: int = 100) -> Dict:
        """
        Train the model.
        
        Args:
            n_epochs: Number of epochs
            
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        
        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print("-" * 70)
        
        for epoch in range(n_epochs):
            # Training
            train_loss = self._train_epoch(epoch, n_epochs)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if self.val_loader is not None:
                val_loss = self._validate_epoch()
                self.history['val_loss'].append(val_loss)
                
                # Update scheduler
                self.scheduler.step(val_loss)
                
                # Log
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
                
                print(f"Epoch {epoch+1}/{n_epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"LR: {current_lr:.6f}")
                
                # Save best model
                if self.output_dir is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
                print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}")
        
        # Save final model
        if self.output_dir is not None:
            self._save_checkpoint(n_epochs - 1, is_best=False)
            self._save_history()
        
        print("\n" + "="*70)
        print("Training completed!")
        print("="*70)
        
        return self.history
    
    def _train_epoch(self, epoch: int, total_epochs: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data - now dataset returns tensors directly
            x = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Check if model has a loss method (like VAE)
            if hasattr(self.model, 'loss'):
                loss = self.model.loss(x)
            else:
                # Standard reconstruction loss
                output = self.model(x)
                loss = nn.MSELoss()(output, x)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / n_batches
    
    def _validate_epoch(self) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        n_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Get data - now dataset returns tensors directly
                x = batch.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'loss'):
                    loss = self.model.loss(x)
                else:
                    output = self.model(x)
                    loss = nn.MSELoss()(output, x)
                
                total_loss += loss.item()
        
        return total_loss / n_batches
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        if is_best:
            save_path = self.output_dir / 'model_best.pt'
            print(f"\n💾 Saving best model (val_loss improved)")
        else:
            save_path = self.output_dir / 'model_final.pt'
        
        torch.save(checkpoint, save_path)
    
    def _save_history(self):
        """Save training history"""
        import json
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {checkpoint_path}")
