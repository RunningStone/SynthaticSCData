"""
Model Evaluator

Evaluates trained models on test datasets using various metrics.
Includes real data metrics: Frechet Distance, MAE, PCC, and Entropy.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from .metrics import (
    BoundaryFidelityMetric,
    PathFidelityMetric,
    EntropyEvolutionMetric,
    GeometricStructureMetric,
    GeneralizationMetric
)
from Analyser.real_data_metrics import (
    evaluate_generated_vs_test,
    calculate_frechet_distance,
    calculate_mae,
    calculate_pcc,
    calculate_statistics
)
from Analyser.entropy_metrics import estimate_entropy_knn


class ModelEvaluator:
    """
    Evaluates models and computes all metrics.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Trained model
            model_type: 'ot', 'sb', or 'vae'
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.model_type = model_type
        self.device = device
        
        # Initialize metrics
        self.boundary_metric = BoundaryFidelityMetric()
        self.path_metric = PathFidelityMetric()
        self.entropy_metric = EntropyEvolutionMetric()
        self.structure_metric = GeometricStructureMetric()
        self.generalization_metric = GeneralizationMetric()
    
    def evaluate(
        self,
        test_dataset: Dict,
        n_samples: int = 1000,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Evaluate model on test dataset.
        
        Args:
            test_dataset: Test dataset dictionary
            n_samples: Number of samples to generate per trajectory
            save_path: Path to save results (optional)
            
        Returns:
            Evaluation results dictionary
        """
        true_trajectories = test_dataset['trajectories']
        time_grid = test_dataset['time_stamps']
        extrapolation_type = test_dataset['metadata'].get('extrapolation_type', 'unknown')
        
        n_test = len(true_trajectories)
        
        # Generate predictions
        print(f"Generating predictions for {n_test} test trajectories...")
        pred_trajectories = []
        
        for i in range(n_test):
            pred_traj = self._generate_trajectory(
                true_trajectories[i],
                time_grid,
                n_samples
            )
            pred_trajectories.append(pred_traj)
        
        pred_trajectories = np.array(pred_trajectories)
        
        # Compute metrics
        print("Computing metrics...")
        results = self.generalization_metric(
            pred_trajectories,
            true_trajectories,
            time_grid,
            extrapolation_type
        )
        
        # Add detailed per-trajectory results
        detailed_results = []
        for i in range(n_test):
            traj_results = {
                'boundary': self.boundary_metric(
                    pred_trajectories[i], true_trajectories[i]
                ),
                'path': self.path_metric(
                    pred_trajectories[i], true_trajectories[i], time_grid
                ),
                'entropy': self.entropy_metric(
                    pred_trajectories[i], true_trajectories[i], time_grid
                ),
                'structure': self.structure_metric(
                    pred_trajectories[i], true_trajectories[i], time_grid
                )
            }
            detailed_results.append(traj_results)
        
        results['detailed_results'] = detailed_results
        results['model_type'] = self.model_type
        
        # Save results
        if save_path is not None:
            self.save_results(results, save_path)
        
        return results
    
    def _generate_trajectory(
        self,
        true_trajectory: np.ndarray,
        time_grid: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Generate predicted trajectory.
        
        Args:
            true_trajectory: True trajectory (n_cells, n_time, d)
            time_grid: Time points (n_time,)
            n_samples: Number of samples to generate
            
        Returns:
            Predicted trajectory (n_samples, n_time, d)
        """
        if self.model_type == 'ot':
            with torch.no_grad():
                # Sample from initial distribution
                x_0_samples = true_trajectory[:n_samples, 0, :]
                x_0 = torch.FloatTensor(x_0_samples).to(self.device)
                time_grid_torch = torch.FloatTensor(time_grid).to(self.device)
                
                pred_traj = self.model.generate_trajectory(x_0, time_grid_torch)
                return pred_traj.cpu().numpy()
        
        elif self.model_type == 'sb':
            # SB model needs gradients for drift computation
            # Sample from initial distribution
            x_0_samples = true_trajectory[:n_samples, 0, :]
            x_0 = torch.FloatTensor(x_0_samples).to(self.device)
            time_grid_torch = torch.FloatTensor(time_grid).to(self.device)
            
            pred_traj = self.model.generate_trajectory(
                x_0, time_grid_torch, method='deterministic'
            )
            return pred_traj.detach().cpu().numpy()
        
        elif self.model_type == 'vae':
            with torch.no_grad():
                # Sample from initial and final distributions
                x_0_samples = true_trajectory[:n_samples, 0, :]
                x_T_samples = true_trajectory[:n_samples, -1, :]
                
                x_0 = torch.FloatTensor(x_0_samples).to(self.device)
                x_T = torch.FloatTensor(x_T_samples).to(self.device)
                time_grid_torch = torch.FloatTensor(time_grid).to(self.device)
                
                pred_traj = self.model.generate_trajectory(x_0, x_T, time_grid_torch)
                return pred_traj.cpu().numpy()
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        test_dataset: Dict,
        n_samples: int = 1000
    ) -> Dict:
        """
        Compare multiple models on the same test dataset.
        
        Args:
            models: Dictionary of {model_name: model}
            test_dataset: Test dataset
            n_samples: Number of samples per trajectory
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            evaluator = ModelEvaluator(model, model_name, self.device)
            results = evaluator.evaluate(test_dataset, n_samples)
            comparison[model_name] = results
        
        # Compute path information gain: ΔL = L_OT - L_SB
        if 'ot' in comparison and 'sb' in comparison:
            path_info_gain = (
                comparison['ot']['path']['mean_error'] -
                comparison['sb']['path']['mean_error']
            )
            comparison['path_information_gain'] = path_info_gain
        
        return comparison
    
    def evaluate_real_data_metrics(
        self,
        X_test: np.ndarray,
        X_generated: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate using real data metrics: Frechet Distance, MAE, PCC, Entropy
        
        Args:
            X_test: Test data (n_test, n_features)
            X_generated: Generated data (n_gen, n_features)
            verbose: Print results
        
        Returns:
            Dictionary with real data metrics
        """
        results = evaluate_generated_vs_test(X_test, X_generated, verbose=verbose)
        
        # Add entropy metrics
        test_entropy = estimate_entropy_knn(X_test, k=10)
        gen_entropy = estimate_entropy_knn(X_generated, k=10)
        
        results['test_entropy'] = float(test_entropy)
        results['generated_entropy'] = float(gen_entropy)
        results['entropy_difference'] = float(abs(test_entropy - gen_entropy))
        
        if verbose:
            print(f"\nEntropy Metrics:")
            print(f"  Test entropy: {test_entropy:.4f}")
            print(f"  Generated entropy: {gen_entropy:.4f}")
            print(f"  Entropy difference: {results['entropy_difference']:.4f}")
        
        return results
    
    def save_results(self, results: Dict, save_path: str):
        """Save evaluation results to JSON"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj
        
        results_serializable = convert_to_serializable(results)
        
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"Results saved to {save_path}")
