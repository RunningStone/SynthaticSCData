#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network-based Embedding Learner with Contrastive Learning

Implements an autoencoder with contrastive learning for better time-point separation
in the embedding space, replacing LMNN for visualization purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm


class ContrastiveClassifier(nn.Module):
    """
    Classifier with contrastive learning for embedding learning.
    
    Architecture:
    - Encoder: Maps high-dimensional data to low-dimensional embedding
    - Classifier: Single linear layer for time-point classification
    
    Training:
    - Classification loss (CrossEntropy)
    - Contrastive loss (InfoNCE/SupCon) for time-point separation
    """
    
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        embedding_dim: int = 2,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = 'relu',
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Input dimension (number of features)
            n_classes: Number of time-point classes
            embedding_dim: Embedding dimension (typically 2 for visualization)
            hidden_dims: List of hidden layer dimensions for encoder
            activation: Activation function ('relu', 'elu', 'leaky_relu')
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        
        # Build encoder
        encoder_layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(self._get_activation(activation))
            encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Final embedding layer (no activation for embedding space)
        encoder_layers.append(nn.Linear(in_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Classifier: single linear layer from embedding to classes
        self.classifier = nn.Linear(embedding_dim, n_classes)
        
        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif activation == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to embedding space.
        
        Args:
            x: Input (batch_size, input_dim)
            
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        return self.encoder(x)
    
    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """
        Classify from embedding space.
        
        Args:
            z: Embedding (batch_size, embedding_dim)
            
        Returns:
            logits: (batch_size, n_classes)
        """
        return self.classifier(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through classifier.
        
        Args:
            x: Input (batch_size, input_dim)
            
        Returns:
            embedding: (batch_size, embedding_dim)
            logits: (batch_size, n_classes)
        """
        embedding = self.encode(x)
        logits = self.classify(embedding)
        return embedding, logits


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    
    Reference:
    Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for scaling
            base_temperature: Base temperature for normalization
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Normalized embeddings (batch_size, embedding_dim)
            labels: Ground truth labels (batch_size,)
            mask: Optional mask for valid pairs (batch_size, batch_size)
            
        Returns:
            loss: Scalar loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask_pos),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask_pos = mask_pos * logits_mask
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive
        mask_pos_sum = mask_pos.sum(1)
        mask_pos_sum = torch.where(mask_pos_sum == 0, torch.ones_like(mask_pos_sum), mask_pos_sum)
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / mask_pos_sum
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class EmbeddingLearner:
    """
    Trainer for ContrastiveClassifier.
    
    Combines classification loss and contrastive learning for better embeddings.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        embedding_dim: int = 2,
        hidden_dims: List[int] = [256, 128, 64],
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        cls_weight: float = 1.0,
        contrast_weight: float = 1.0,
        temperature: float = 0.07
    ):
        """
        Args:
            input_dim: Input dimension
            n_classes: Number of time-point classes
            embedding_dim: Embedding dimension
            hidden_dims: Hidden layer dimensions
            device: Device for training
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            cls_weight: Weight for classification loss
            contrast_weight: Weight for contrastive loss
            temperature: Temperature for contrastive loss
        """
        self.device = device
        self.n_classes = n_classes
        self.cls_weight = cls_weight
        self.contrast_weight = contrast_weight
        
        # Initialize model
        self.model = ContrastiveClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            activation='relu',
            dropout=0.1,
            use_batch_norm=True
        ).to(device)
        
        # Initialize loss functions
        self.cls_criterion = nn.CrossEntropyLoss()
        self.contrast_criterion = SupConLoss(temperature=temperature)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_cls_loss': [],
            'train_contrast_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_cls_loss': [],
            'val_contrast_loss': [],
            'val_accuracy': []
        }
    
    def compute_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Compute combined loss.
        
        Args:
            x: Input data (batch_size, input_dim)
            labels: Time labels (batch_size,)
            
        Returns:
            total_loss: Combined loss
            cls_loss: Classification loss
            contrast_loss: Contrastive loss
            accuracy: Classification accuracy
        """
        # Forward pass
        embedding, logits = self.model(x)
        
        # Classification loss
        cls_loss = self.cls_criterion(logits, labels)
        
        # Contrastive loss
        contrast_loss = self.contrast_criterion(embedding, labels)
        
        # Combined loss
        total_loss = (
            self.cls_weight * cls_loss +
            self.contrast_weight * contrast_loss
        )
        
        # Compute accuracy
        pred = logits.argmax(dim=1)
        accuracy = (pred == labels).float().mean().item()
        
        return total_loss, cls_loss, contrast_loss, accuracy
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float, float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            avg_loss: Average total loss
            avg_cls_loss: Average classification loss
            avg_contrast_loss: Average contrastive loss
            avg_accuracy: Average accuracy
        """
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_contrast_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0
        
        for x, labels in train_loader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss, cls_loss, contrast_loss, accuracy = self.compute_loss(x, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_contrast_loss += contrast_loss.item()
            total_accuracy += accuracy
            n_batches += 1
        
        return (
            total_loss / n_batches,
            total_cls_loss / n_batches,
            total_contrast_loss / n_batches,
            total_accuracy / n_batches
        )
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Tuple[float, float, float, float]:
        """
        Validate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            avg_loss: Average total loss
            avg_cls_loss: Average classification loss
            avg_contrast_loss: Average contrastive loss
            avg_accuracy: Average accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_contrast_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for x, labels in val_loader:
                x = x.to(self.device)
                labels = labels.to(self.device)
                
                # Compute loss
                loss, cls_loss, contrast_loss, accuracy = self.compute_loss(x, labels)
                
                # Accumulate losses
                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_contrast_loss += contrast_loss.item()
                total_accuracy += accuracy
                n_batches += 1
        
        return (
            total_loss / n_batches,
            total_cls_loss / n_batches,
            total_contrast_loss / n_batches,
            total_accuracy / n_batches
        )
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True
    ):
        """
        Train the embedding learner.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            verbose: Whether to print progress
        """
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        if verbose:
            print("="*80)
            print("Training Contrastive Classifier for Embedding Learning")
            print("="*80)
            print(f"Model: {sum(p.numel() for p in self.model.parameters())} parameters")
            print(f"Embedding dim: {self.model.embedding_dim}")
            print(f"Num classes: {self.n_classes}")
            print(f"Cls weight: {self.cls_weight}, Contrast weight: {self.contrast_weight}")
            print("="*80)
        
        for epoch in range(epochs):
            # Train
            train_loss, train_cls, train_contrast, train_acc = self.train_epoch(train_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_cls_loss'].append(train_cls)
            self.history['train_contrast_loss'].append(train_contrast)
            self.history['train_accuracy'].append(train_acc)
            
            # Validate
            if val_loader is not None:
                val_loss, val_cls, val_contrast, val_acc = self.validate(val_loader)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_cls_loss'].append(val_cls)
                self.history['val_contrast_loss'].append(val_contrast)
                self.history['val_accuracy'].append(val_acc)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if verbose and epoch % 10 == 0:
                        print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f} "
                              f"(Cls={train_cls:.4f}, Contrast={train_contrast:.4f}, Acc={train_acc:.3f}) | "
                              f"Val Loss={val_loss:.4f} "
                              f"(Cls={val_cls:.4f}, Contrast={val_contrast:.4f}, Acc={val_acc:.3f}) ✓")
                else:
                    patience_counter += 1
                    if verbose and epoch % 10 == 0:
                        print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f} "
                              f"(Cls={train_cls:.4f}, Contrast={train_contrast:.4f}, Acc={train_acc:.3f}) | "
                              f"Val Loss={val_loss:.4f} "
                              f"(Cls={val_cls:.4f}, Contrast={val_contrast:.4f}, Acc={val_acc:.3f})")
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f} "
                          f"(Cls={train_cls:.4f}, Contrast={train_contrast:.4f}, Acc={train_acc:.3f})")
        
        if verbose:
            print("="*80)
            print("Training complete!")
            print("="*80)
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform data to embedding space.
        
        Args:
            x: Input data (n_samples, input_dim)
            
        Returns:
            embedding: (n_samples, embedding_dim)
        """
        self.model.eval()
        
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            embedding = self.model.encode(x_tensor)
            return embedding.cpu().numpy()
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 256,
        epochs: int = 100,
        val_split: float = 0.2,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Fit the model and transform data.
        
        Args:
            X: Input data (n_samples, input_dim)
            y: Time labels (n_samples,)
            batch_size: Batch size for training
            epochs: Number of training epochs
            val_split: Validation split ratio
            verbose: Whether to print progress
            
        Returns:
            embedding: (n_samples, embedding_dim)
        """
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y)
        )
        
        # Split into train/val
        n_samples = len(dataset)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val]
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        ) if n_val > 0 else None
        
        # Train
        self.fit(train_loader, val_loader, epochs=epochs, verbose=verbose)
        
        # Transform
        return self.transform(X)
