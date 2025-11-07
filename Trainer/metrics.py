"""
Evaluation Metrics

Implements various metrics for assessing model performance.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from scipy import linalg
import ot  # Python Optimal Transport library


def compute_wasserstein_distance(
    samples1: np.ndarray,
    samples2: np.ndarray,
    p: int = 2
) -> float:
    """
    Compute Wasserstein-p distance between two empirical distributions.
    
    Args:
        samples1: Samples from first distribution (n1, d)
        samples2: Samples from second distribution (n2, d)
        p: Order of Wasserstein distance
        
    Returns:
        Wasserstein distance
    """
    n1, n2 = len(samples1), len(samples2)
    
    # Uniform weights
    a = np.ones(n1) / n1
    b = np.ones(n2) / n2
    
    # Compute cost matrix
    M = ot.dist(samples1, samples2, metric='euclidean')
    
    # Solve OT problem
    if p == 2:
        W = ot.emd2(a, b, M)
        return np.sqrt(W)
    else:
        W = ot.emd2(a, b, M ** p)
        return W ** (1.0 / p)


def compute_wasserstein_gaussian(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray
) -> float:
    """
    Compute Wasserstein-2 distance between two Gaussian distributions analytically.
    
    W_2²(N(μ1, Σ1), N(μ2, Σ2)) = ||μ1 - μ2||² + tr(Σ1 + Σ2 - 2(Σ2^{1/2} Σ1 Σ2^{1/2})^{1/2})
    
    Args:
        mu1: Mean of first Gaussian (d,)
        Sigma1: Covariance of first Gaussian (d, d)
        mu2: Mean of second Gaussian (d,)
        Sigma2: Covariance of second Gaussian (d, d)
        
    Returns:
        Wasserstein-2 distance
    """
    # Mean difference term
    mean_diff = np.sum((mu1 - mu2) ** 2)
    
    # Covariance term
    Sigma2_sqrt = linalg.sqrtm(Sigma2)
    M = Sigma2_sqrt @ Sigma1 @ Sigma2_sqrt
    M_sqrt = linalg.sqrtm(M)
    
    cov_term = np.trace(Sigma1 + Sigma2 - 2 * M_sqrt)
    
    return np.sqrt(mean_diff + cov_term)


def compute_entropy(samples: np.ndarray) -> float:
    """
    Compute differential entropy by fitting Gaussian.
    
    H = (d/2) log(2πe) + (1/2) log|Σ|
    
    Args:
        samples: Samples from distribution (n, d)
        
    Returns:
        Differential entropy
    """
    d = samples.shape[1]
    
    # Estimate covariance
    cov = np.cov(samples.T)
    
    # Add small regularization for numerical stability
    cov += 1e-6 * np.eye(d)
    
    # Compute log determinant
    sign, logdet = np.linalg.slogdet(cov)
    
    if sign <= 0:
        return 0.0
    
    entropy = 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * logdet
    
    return entropy


class BoundaryFidelityMetric:
    """
    Evaluate boundary fidelity: how well initial and final states match.
    """
    
    def __call__(
        self,
        pred_trajectory: np.ndarray,
        true_trajectory: np.ndarray
    ) -> Dict[str, float]:
        """
        Args:
            pred_trajectory: Predicted trajectory (n_cells, n_time, d)
            true_trajectory: True trajectory (n_cells, n_time, d)
            
        Returns:
            Dictionary with 'initial', 'final', and 'total' errors
        """
        # Initial state error
        W_init = compute_wasserstein_distance(
            pred_trajectory[:, 0, :],
            true_trajectory[:, 0, :]
        )
        
        # Final state error
        W_final = compute_wasserstein_distance(
            pred_trajectory[:, -1, :],
            true_trajectory[:, -1, :]
        )
        
        return {
            'initial': W_init,
            'final': W_final,
            'total': W_init + W_final
        }


class PathFidelityMetric:
    """
    Evaluate path fidelity: integrated Wasserstein distance over time.
    """
    
    def __call__(
        self,
        pred_trajectory: np.ndarray,
        true_trajectory: np.ndarray,
        time_grid: np.ndarray
    ) -> Dict[str, float]:
        """
        Args:
            pred_trajectory: Predicted trajectory (n_cells, n_time, d)
            true_trajectory: True trajectory (n_cells, n_time, d)
            time_grid: Time points (n_time,)
            
        Returns:
            Dictionary with 'path_error', 'time_resolved_errors'
        """
        n_time = len(time_grid)
        time_resolved_errors = np.zeros(n_time)
        
        for t_idx in range(n_time):
            W_t = compute_wasserstein_distance(
                pred_trajectory[:, t_idx, :],
                true_trajectory[:, t_idx, :]
            )
            time_resolved_errors[t_idx] = W_t
        
        # Integrate using trapezoidal rule
        path_error = np.trapz(time_resolved_errors, time_grid)
        
        return {
            'path_error': path_error,
            'time_resolved_errors': time_resolved_errors,
            'mean_error': np.mean(time_resolved_errors),
            'max_error': np.max(time_resolved_errors)
        }


class EntropyEvolutionMetric:
    """
    Evaluate entropy evolution: how well the model captures entropy dynamics.
    """
    
    def __call__(
        self,
        pred_trajectory: np.ndarray,
        true_trajectory: np.ndarray,
        time_grid: np.ndarray
    ) -> Dict[str, float]:
        """
        Args:
            pred_trajectory: Predicted trajectory (n_cells, n_time, d)
            true_trajectory: True trajectory (n_cells, n_time, d)
            time_grid: Time points (n_time,)
            
        Returns:
            Dictionary with entropy-related metrics
        """
        n_time = len(time_grid)
        
        pred_entropy = np.zeros(n_time)
        true_entropy = np.zeros(n_time)
        
        for t_idx in range(n_time):
            pred_entropy[t_idx] = compute_entropy(pred_trajectory[:, t_idx, :])
            true_entropy[t_idx] = compute_entropy(true_trajectory[:, t_idx, :])
        
        # Integrated error
        entropy_error = np.trapz(np.abs(pred_entropy - true_entropy), time_grid)
        
        # Peak detection
        pred_peak_idx = np.argmax(pred_entropy)
        true_peak_idx = np.argmax(true_entropy)
        
        peak_time_error = np.abs(time_grid[pred_peak_idx] - time_grid[true_peak_idx])
        peak_value_error = np.abs(pred_entropy[pred_peak_idx] - true_entropy[true_peak_idx])
        
        return {
            'entropy_error': entropy_error,
            'peak_time_error': peak_time_error,
            'peak_value_error': peak_value_error,
            'pred_entropy_curve': pred_entropy,
            'true_entropy_curve': true_entropy
        }


class GeometricStructureMetric:
    """
    Evaluate geometric structure: principal components of covariance at peak entropy.
    """
    
    def __call__(
        self,
        pred_trajectory: np.ndarray,
        true_trajectory: np.ndarray,
        time_grid: np.ndarray,
        n_components: int = 5
    ) -> Dict[str, float]:
        """
        Args:
            pred_trajectory: Predicted trajectory (n_cells, n_time, d)
            true_trajectory: True trajectory (n_cells, n_time, d)
            time_grid: Time points (n_time,)
            n_components: Number of principal components to compare
            
        Returns:
            Dictionary with structure-related metrics
        """
        # Find peak entropy time
        true_entropy = np.array([
            compute_entropy(true_trajectory[:, t, :])
            for t in range(len(time_grid))
        ])
        peak_idx = np.argmax(true_entropy)
        
        # Get samples at peak time
        pred_samples = pred_trajectory[:, peak_idx, :]
        true_samples = true_trajectory[:, peak_idx, :]
        
        # Compute covariances
        pred_cov = np.cov(pred_samples.T)
        true_cov = np.cov(true_samples.T)
        
        # Eigendecomposition
        pred_eigvals, pred_eigvecs = np.linalg.eigh(pred_cov)
        true_eigvals, true_eigvecs = np.linalg.eigh(true_cov)
        
        # Sort by eigenvalue (descending)
        pred_idx = np.argsort(pred_eigvals)[::-1]
        true_idx = np.argsort(true_eigvals)[::-1]
        
        pred_eigvals = pred_eigvals[pred_idx]
        pred_eigvecs = pred_eigvecs[:, pred_idx]
        true_eigvals = true_eigvals[true_idx]
        true_eigvecs = true_eigvecs[:, true_idx]
        
        # Compare top components
        n_comp = min(n_components, len(pred_eigvals))
        
        # Eigenvalue spectrum error
        spectrum_error = np.linalg.norm(
            pred_eigvals[:n_comp] - true_eigvals[:n_comp]
        )
        
        # Principal direction error (Frobenius norm, accounting for sign ambiguity)
        V_pred = pred_eigvecs[:, :n_comp]
        V_true = true_eigvecs[:, :n_comp]
        
        # Align signs
        for i in range(n_comp):
            if np.dot(V_pred[:, i], V_true[:, i]) < 0:
                V_pred[:, i] *= -1
        
        structure_error = np.linalg.norm(V_pred - V_true, 'fro')
        
        return {
            'spectrum_error': spectrum_error,
            'structure_error': structure_error,
            'pred_eigenvalues': pred_eigvals[:n_comp],
            'true_eigenvalues': true_eigvals[:n_comp]
        }


class GeneralizationMetric:
    """
    Aggregate generalization metrics across multiple test cases.
    """
    
    def __init__(self):
        self.boundary_metric = BoundaryFidelityMetric()
        self.path_metric = PathFidelityMetric()
        self.entropy_metric = EntropyEvolutionMetric()
        self.structure_metric = GeometricStructureMetric()
    
    def __call__(
        self,
        pred_trajectories: np.ndarray,
        true_trajectories: np.ndarray,
        time_grid: np.ndarray,
        extrapolation_type: str
    ) -> Dict[str, Dict]:
        """
        Compute all metrics for a set of test trajectories.
        
        Args:
            pred_trajectories: (n_test, n_cells, n_time, d)
            true_trajectories: (n_test, n_cells, n_time, d)
            time_grid: (n_time,)
            extrapolation_type: Type of extrapolation
            
        Returns:
            Dictionary of aggregated metrics
        """
        n_test = len(pred_trajectories)
        
        all_boundary = []
        all_path = []
        all_entropy = []
        all_structure = []
        
        for i in range(n_test):
            boundary = self.boundary_metric(
                pred_trajectories[i], true_trajectories[i]
            )
            path = self.path_metric(
                pred_trajectories[i], true_trajectories[i], time_grid
            )
            entropy = self.entropy_metric(
                pred_trajectories[i], true_trajectories[i], time_grid
            )
            structure = self.structure_metric(
                pred_trajectories[i], true_trajectories[i], time_grid
            )
            
            all_boundary.append(boundary)
            all_path.append(path)
            all_entropy.append(entropy)
            all_structure.append(structure)
        
        # Aggregate statistics
        results = {
            'extrapolation_type': extrapolation_type,
            'n_test': n_test,
            'boundary': {
                'mean_total': np.mean([b['total'] for b in all_boundary]),
                'std_total': np.std([b['total'] for b in all_boundary]),
                'mean_initial': np.mean([b['initial'] for b in all_boundary]),
                'mean_final': np.mean([b['final'] for b in all_boundary])
            },
            'path': {
                'mean_error': np.mean([p['path_error'] for p in all_path]),
                'std_error': np.std([p['path_error'] for p in all_path]),
                'mean_max_error': np.mean([p['max_error'] for p in all_path])
            },
            'entropy': {
                'mean_error': np.mean([e['entropy_error'] for e in all_entropy]),
                'std_error': np.std([e['entropy_error'] for e in all_entropy]),
                'mean_peak_time_error': np.mean([e['peak_time_error'] for e in all_entropy]),
                'mean_peak_value_error': np.mean([e['peak_value_error'] for e in all_entropy])
            },
            'structure': {
                'mean_spectrum_error': np.mean([s['spectrum_error'] for s in all_structure]),
                'mean_structure_error': np.mean([s['structure_error'] for s in all_structure])
            }
        }
        
        return results
