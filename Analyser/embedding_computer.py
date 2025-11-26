#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Computer - Dimensionality Reduction

Computes low-dimensional embeddings for visualization:
- PHATE embeddings
- LMNN + PCA embeddings
"""

import numpy as np
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

import phate
from metric_learn import LMNN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class EmbeddingComputer:
    """
    Computes low-dimensional embeddings for visualization.
    
    Supports:
    - PHATE: Potential of Heat-diffusion for Affinity-based Trajectory Embedding
    - LMNN + PCA: Large Margin Nearest Neighbor + Principal Component Analysis
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize embedding computer.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        
        # PHATE components
        self.phate_op = None
        
        # LMNN + PCA components
        self.lmnn_scaler = None
        self.lmnn_op = None
        self.pca_op = None
    
    def fit_phate(
        self,
        X: np.ndarray,
        n_components: int = 2,
        knn: int = 5,
        decay: int = 40,
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Fit PHATE embedding.
        
        Args:
            X: Data matrix (n_samples, n_features)
            n_components: Number of embedding dimensions
            knn: Number of nearest neighbors
            decay: Decay parameter for heat diffusion
            n_jobs: Number of parallel jobs
            **kwargs: Additional arguments for PHATE
        
        Returns:
            self
        """
        self.phate_op = phate.PHATE(
            n_components=n_components,
            knn=knn,
            decay=decay,
            n_jobs=n_jobs,
            random_state=self.random_seed,
            verbose=0,
            **kwargs
        )
        
        self.phate_op.fit(X)
        
        return self
    
    def transform_phate(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PHATE.
        
        Args:
            X: Data matrix (n_samples, n_features)
        
        Returns:
            Embedded data (n_samples, n_components)
        """
        if self.phate_op is None:
            raise ValueError("PHATE not fitted. Call fit_phate() first.")
        
        return self.phate_op.transform(X)
    
    def fit_transform_phate(
        self,
        X: np.ndarray,
        n_components: int = 2,
        knn: int = 5,
        decay: int = 40,
        n_jobs: int = -1,
        **kwargs
    ) -> np.ndarray:
        """
        Fit and transform data with PHATE in one step.
        
        Args:
            X: Data matrix (n_samples, n_features)
            n_components: Number of embedding dimensions
            knn: Number of nearest neighbors
            decay: Decay parameter for heat diffusion
            n_jobs: Number of parallel jobs
            **kwargs: Additional arguments for PHATE
        
        Returns:
            Embedded data (n_samples, n_components)
        """
        self.fit_phate(X, n_components, knn, decay, n_jobs, **kwargs)
        return self.phate_op.fit_transform(X)
    
    def fit_lmnn_pca(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lmnn_components: Optional[int] = None,
        pca_components: int = 2,
        k: int = 5,
        learn_rate: float = 1e-6,
        max_iter: int = 100,
        **kwargs
    ):
        """
        Fit LMNN + PCA embedding.
        
        Args:
            X: Data matrix (n_samples, n_features)
            y: Labels (n_samples,)
            lmnn_components: Number of LMNN components (default: min(50, n_features))
            pca_components: Number of PCA components
            k: Number of neighbors for LMNN
            learn_rate: Learning rate for LMNN
            max_iter: Maximum iterations for LMNN
            **kwargs: Additional arguments for LMNN
        
        Returns:
            self
        """
        # Standardize data
        self.lmnn_scaler = StandardScaler()
        X_scaled = self.lmnn_scaler.fit_transform(X)
        
        # Fit LMNN
        if lmnn_components is None:
            lmnn_components = min(50, X.shape[1])
        
        self.lmnn_op = LMNN(
            n_components=lmnn_components,
            k=k,
            learn_rate=learn_rate,
            max_iter=max_iter,
            verbose=False,
            random_state=self.random_seed,
            **kwargs
        )
        
        X_lmnn = self.lmnn_op.fit_transform(X_scaled, y)
        
        # Fit PCA
        self.pca_op = PCA(n_components=pca_components, random_state=self.random_seed)
        self.pca_op.fit(X_lmnn)
        
        return self
    
    def transform_lmnn_pca(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted LMNN + PCA.
        
        Args:
            X: Data matrix (n_samples, n_features)
        
        Returns:
            Embedded data (n_samples, pca_components)
        """
        if self.lmnn_op is None or self.pca_op is None:
            raise ValueError("LMNN+PCA not fitted. Call fit_lmnn_pca() first.")
        
        X_scaled = self.lmnn_scaler.transform(X)
        X_lmnn = self.lmnn_op.transform(X_scaled)
        X_pca = self.pca_op.transform(X_lmnn)
        
        return X_pca
    
    def fit_transform_lmnn_pca(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lmnn_components: Optional[int] = None,
        pca_components: int = 2,
        k: int = 5,
        learn_rate: float = 1e-6,
        max_iter: int = 100,
        **kwargs
    ) -> np.ndarray:
        """
        Fit and transform data with LMNN + PCA in one step.
        
        Args:
            X: Data matrix (n_samples, n_features)
            y: Labels (n_samples,)
            lmnn_components: Number of LMNN components (default: min(50, n_features))
            pca_components: Number of PCA components
            k: Number of neighbors for LMNN
            learn_rate: Learning rate for LMNN
            max_iter: Maximum iterations for LMNN
            **kwargs: Additional arguments for LMNN
        
        Returns:
            Embedded data (n_samples, pca_components)
        """
        self.fit_lmnn_pca(X, y, lmnn_components, pca_components, k, learn_rate, max_iter, **kwargs)
        
        # Transform
        X_scaled = self.lmnn_scaler.transform(X)
        X_lmnn = self.lmnn_op.transform(X_scaled)
        X_pca = self.pca_op.transform(X_lmnn)
        
        return X_pca
    
    def compute_all_embeddings(
        self,
        X: np.ndarray,
        y: np.ndarray,
        phate_kwargs: Optional[Dict] = None,
        lmnn_pca_kwargs: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute both PHATE and LMNN+PCA embeddings.
        
        Args:
            X: Data matrix (n_samples, n_features)
            y: Labels (n_samples,)
            phate_kwargs: Keyword arguments for PHATE
            lmnn_pca_kwargs: Keyword arguments for LMNN+PCA
        
        Returns:
            Dictionary with 'phate' and 'lmnn_pca' embeddings
        """
        phate_kwargs = phate_kwargs or {}
        lmnn_pca_kwargs = lmnn_pca_kwargs or {}
        
        # Compute PHATE
        phate_emb = self.fit_transform_phate(X, **phate_kwargs)
        
        # Compute LMNN+PCA
        lmnn_pca_emb = self.fit_transform_lmnn_pca(X, y, **lmnn_pca_kwargs)
        
        return {
            'phate': phate_emb,
            'lmnn_pca': lmnn_pca_emb
        }
    
    def get_info(self) -> Dict:
        """
        Get embedding computer information.
        
        Returns:
            Dictionary with embedding method info
        """
        info = {
            'random_seed': self.random_seed,
            'phate_fitted': self.phate_op is not None,
            'lmnn_pca_fitted': (self.lmnn_op is not None and self.pca_op is not None)
        }
        
        if self.phate_op is not None:
            info['phate_n_components'] = self.phate_op.n_components
        
        if self.pca_op is not None:
            info['pca_n_components'] = self.pca_op.n_components
        
        return info
