"""
Tests for Data Generation Module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Data import (
    DistributionParameterizer,
    PotentialFunction,
    TrajectorySampler,
    DatasetConstructor
)


class TestDistributionParameterizer:
    """Test distribution parameterization"""
    
    def test_initial_state_generation(self):
        """Test initial state is isotropic Gaussian"""
        dim = 50
        sigma = 1.0
        
        parameterizer = DistributionParameterizer(dim, seed=42)
        initial = parameterizer.generate_initial_state(sigma)
        
        assert initial['mu'].shape == (dim,)
        assert initial['Sigma'].shape == (dim, dim)
        assert np.allclose(initial['mu'], 0.0)
        assert np.allclose(initial['Sigma'], sigma**2 * np.eye(dim))
    
    def test_high_entropy_state_generation(self):
        """Test high entropy state has low-rank structure"""
        dim = 50
        effective_rank = 5
        
        parameterizer = DistributionParameterizer(dim, seed=42)
        high_entropy = parameterizer.generate_high_entropy_state(
            effective_rank=effective_rank,
            lambda_low=1.0,
            lambda_high=10.0,
            lambda_small=0.01,
            mu_peak_norm=2.0
        )
        
        assert high_entropy['mu'].shape == (dim,)
        assert high_entropy['Sigma'].shape == (dim, dim)
        assert high_entropy['eigenvalues'].shape == (dim,)
        
        # Check eigenvalue structure
        eigvals = high_entropy['eigenvalues']
        assert np.all(eigvals[:effective_rank] >= 1.0)
        assert np.all(eigvals[effective_rank:] == 0.01)
        
        # Check orthogonality of U
        U = high_entropy['U']
        assert np.allclose(U @ U.T, np.eye(dim), atol=1e-10)
    
    def test_final_state_generation(self):
        """Test final state mixture generation"""
        dim = 50
        n_modes = 2
        separation = 5.0
        sigma = 0.5
        
        parameterizer = DistributionParameterizer(dim, seed=42)
        final = parameterizer.generate_final_state(
            n_modes=n_modes,
            separation_distance=separation,
            sigma_final=sigma,
            mode_weights=None
        )
        
        assert final['n_modes'] == n_modes
        assert final['centers'].shape == (n_modes, dim)
        assert final['covariances'].shape == (n_modes, dim, dim)
        assert final['weights'].shape == (n_modes,)
        assert np.allclose(final['weights'].sum(), 1.0)
        
        # Check mode separation
        dist = np.linalg.norm(final['centers'][0] - final['centers'][1])
        assert dist >= separation * 0.9  # Allow some tolerance


class TestPotentialFunction:
    """Test potential function construction"""
    
    def test_potential_initialization(self):
        """Test potential function can be initialized"""
        dim = 50
        parameterizer = DistributionParameterizer(dim, seed=42)
        dist_params = parameterizer.generate_all_states(
            sigma_init=1.0,
            effective_rank=5,
            lambda_low=1.0,
            lambda_high=10.0,
            lambda_small=0.01,
            mu_peak_norm=2.0,
            n_modes=2,
            separation_distance=5.0,
            sigma_final=0.5,
            mode_weights=None
        )
        
        time_points = {'t1': 0.2, 't_peak': 0.5, 't2': 0.8}
        potential_params = {
            'alpha_init': 1.0,
            'alpha_decay': 2.0,
            'beta_init': 1.0,
            'beta_decay': 1.0,
            'gamma_init': 0.5,
            'gamma_growth': 3.0
        }
        
        potential = PotentialFunction(
            dim, dist_params, time_points, potential_params, total_time=1.0
        )
        
        assert potential.dimension == dim
    
    def test_potential_evaluation(self):
        """Test potential can be evaluated at different times"""
        dim = 10  # Smaller for faster test
        parameterizer = DistributionParameterizer(dim, seed=42)
        dist_params = parameterizer.generate_all_states(
            sigma_init=1.0,
            effective_rank=3,
            lambda_low=1.0,
            lambda_high=5.0,
            lambda_small=0.01,
            mu_peak_norm=1.0,
            n_modes=2,
            separation_distance=3.0,
            sigma_final=0.5,
            mode_weights=None
        )
        
        time_points = {'t1': 0.2, 't_peak': 0.5, 't2': 0.8}
        potential_params = {
            'alpha_init': 1.0,
            'alpha_decay': 2.0,
            'beta_init': 1.0,
            'beta_decay': 1.0,
            'gamma_init': 0.5,
            'gamma_growth': 3.0
        }
        
        potential = PotentialFunction(
            dim, dist_params, time_points, potential_params, total_time=1.0
        )
        
        # Test evaluation at different times
        x = np.random.randn(dim)
        
        U_0 = potential(x, 0.0)  # Phase 1
        U_mid = potential(x, 0.5)  # Phase 2
        U_end = potential(x, 0.9)  # Phase 3
        
        assert isinstance(U_0, (float, np.floating))
        assert isinstance(U_mid, (float, np.floating))
        assert isinstance(U_end, (float, np.floating))
    
    def test_gradient_computation(self):
        """Test gradient computation"""
        dim = 10
        parameterizer = DistributionParameterizer(dim, seed=42)
        dist_params = parameterizer.generate_all_states(
            sigma_init=1.0,
            effective_rank=3,
            lambda_low=1.0,
            lambda_high=5.0,
            lambda_small=0.01,
            mu_peak_norm=1.0,
            n_modes=2,
            separation_distance=3.0,
            sigma_final=0.5,
            mode_weights=None
        )
        
        time_points = {'t1': 0.2, 't_peak': 0.5, 't2': 0.8}
        potential_params = {
            'alpha_init': 1.0,
            'alpha_decay': 2.0,
            'beta_init': 1.0,
            'beta_decay': 1.0,
            'gamma_init': 0.5,
            'gamma_growth': 3.0
        }
        
        potential = PotentialFunction(
            dim, dist_params, time_points, potential_params, total_time=1.0
        )
        
        x = np.random.randn(dim)
        grad = potential.gradient(x, 0.5)
        
        assert grad.shape == (dim,)


class TestTrajectorySampler:
    """Test trajectory sampling"""
    
    def test_trajectory_sampling(self):
        """Test trajectory can be sampled"""
        dim = 10
        n_cells = 100
        n_time = 20
        
        # Setup
        parameterizer = DistributionParameterizer(dim, seed=42)
        dist_params = parameterizer.generate_all_states(
            sigma_init=1.0,
            effective_rank=3,
            lambda_low=1.0,
            lambda_high=5.0,
            lambda_small=0.01,
            mu_peak_norm=1.0,
            n_modes=2,
            separation_distance=3.0,
            sigma_final=0.5,
            mode_weights=None
        )
        
        time_points = {'t1': 0.2, 't_peak': 0.5, 't2': 0.8}
        potential_params = {
            'alpha_init': 1.0,
            'alpha_decay': 2.0,
            'beta_init': 1.0,
            'beta_decay': 1.0,
            'gamma_init': 0.5,
            'gamma_growth': 3.0
        }
        
        potential = PotentialFunction(
            dim, dist_params, time_points, potential_params, total_time=1.0
        )
        
        sampler = TrajectorySampler(potential, diffusion_coeff=0.1, seed=42)
        
        time_grid = np.linspace(0, 1.0, n_time)
        trajectory, time_stamps = sampler.sample_trajectory(
            n_cells, time_grid, dist_params['initial']
        )
        
        assert trajectory.shape == (n_cells, n_time, dim)
        assert np.array_equal(time_stamps, time_grid)


class TestDatasetConstructor:
    """Test dataset construction"""
    
    @pytest.fixture
    def minimal_config(self):
        """Minimal configuration for testing"""
        return {
            'data': {
                'dimension': 10,
                'n_cells': 50,
                'n_time_points': 15,
                'total_time': 1.0,
                'diffusion_coeff': 0.1,
                'initial_state': {'sigma_init': 1.0},
                'high_entropy_state': {
                    'effective_rank': 3,
                    'lambda_low': 1.0,
                    'lambda_high': 5.0,
                    'lambda_small': 0.01,
                    'mu_peak_norm': 1.0
                },
                'final_state': {
                    'n_modes': 2,
                    'separation_distance': 3.0,
                    'sigma_final': 0.5,
                    'mode_weights': [0.5, 0.5]
                },
                'time_points': {'t1': 0.2, 't_peak': 0.5, 't2': 0.8},
                'potential': {
                    'alpha_init': 1.0,
                    'alpha_decay': 2.0,
                    'beta_init': 1.0,
                    'beta_decay': 1.0,
                    'gamma_init': 0.5,
                    'gamma_growth': 3.0
                }
            },
            'dataset': {
                'train_size': 2,
                'test_size': 2,
                'train_variation': {
                    'separation_distance': [2.5, 3.5],
                    'lambda_high': [4.0, 6.0],
                    'entropy_duration': [0.5, 0.7]
                },
                'test_extrapolation': {
                    'type': 'geometric',
                    'geometric': {
                        'separation_distance': [4.0, 5.0],
                        'lambda_high': [7.0, 9.0]
                    }
                }
            }
        }
    
    def test_dataset_construction(self, minimal_config):
        """Test dataset can be constructed"""
        constructor = DatasetConstructor(minimal_config, seed=42)
        
        dataset = constructor.construct_dataset(dataset_type='train')
        
        assert 'trajectories' in dataset
        assert 'time_stamps' in dataset
        assert 'distribution_params' in dataset
        assert 'entropy_curves' in dataset
        
        n_traj = minimal_config['dataset']['train_size']
        n_cells = minimal_config['data']['n_cells']
        n_time = minimal_config['data']['n_time_points']
        dim = minimal_config['data']['dimension']
        
        assert dataset['trajectories'].shape == (n_traj, n_cells, n_time, dim)
        assert dataset['time_stamps'].shape == (n_time,)
        assert dataset['entropy_curves'].shape == (n_traj, n_time)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
