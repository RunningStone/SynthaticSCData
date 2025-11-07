"""
Data Quality Monitor

Monitors generated synthetic data for quality control.
"""

import numpy as np
from typing import Dict, List
import json
from pathlib import Path


class DataQualityMonitor:
    """
    Monitors data quality by checking:
    - Entropy evolution pattern (increase then decrease)
    - Entropy peak timing
    - Final state mode separation
    - Covariance low-rank structure
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.dimension = config['data']['dimension']
    
    def check_dataset(self, dataset: Dict) -> Dict:
        """
        Check quality of entire dataset.
        
        Args:
            dataset: Dataset dictionary
            
        Returns:
            Quality report dictionary
        """
        trajectories = dataset['trajectories']
        time_grid = dataset['time_stamps']
        entropy_curves = dataset['entropy_curves']
        
        n_traj = len(trajectories)
        
        # Check each trajectory
        trajectory_checks = []
        for i in range(n_traj):
            check = self._check_trajectory(
                trajectories[i],
                time_grid,
                entropy_curves[i]
            )
            trajectory_checks.append(check)
        
        # Aggregate statistics
        n_pass = sum(1 for c in trajectory_checks if c['passes_all'])
        
        report = {
            'n_trajectories': n_traj,
            'n_pass': n_pass,
            'pass_rate': n_pass / n_traj,
            'trajectory_checks': trajectory_checks,
            'summary': self._summarize_checks(trajectory_checks)
        }
        
        return report
    
    def _check_trajectory(
        self,
        trajectory: np.ndarray,
        time_grid: np.ndarray,
        entropy_curve: np.ndarray
    ) -> Dict:
        """Check single trajectory quality"""
        checks = {}
        
        # Check 1: Entropy increases then decreases (允许小幅波动)
        peak_idx = np.argmax(entropy_curve)
        
        # 使用整体趋势而非严格单调性
        # 检查峰值前的平均斜率是否为正，峰值后是否为负
        if peak_idx > 0:
            before_peak_trend = (entropy_curve[peak_idx] - entropy_curve[0]) / peak_idx
            entropy_increases = before_peak_trend > 0
        else:
            entropy_increases = True
        
        if peak_idx < len(entropy_curve) - 1:
            after_peak_trend = (entropy_curve[-1] - entropy_curve[peak_idx]) / (len(entropy_curve) - peak_idx - 1)
            entropy_decreases = after_peak_trend < 0
        else:
            entropy_decreases = True
        
        checks['entropy_pattern'] = entropy_increases and entropy_decreases
        
        # Check 2: Peak timing in expected window
        t_peak = time_grid[peak_idx]
        expected_window = (
            self.config['data']['time_points']['t1'],
            self.config['data']['time_points']['t2']
        )
        checks['peak_timing'] = (
            expected_window[0] <= t_peak <= expected_window[1]
        )
        checks['peak_time'] = t_peak
        
        # Check 3: Final state mode separation
        final_states = trajectory[:, -1, :]
        separation = self._check_mode_separation(final_states)
        min_separation = self.config['data']['final_state']['separation_distance'] * 0.9
        checks['mode_separation'] = separation >= min_separation
        checks['actual_separation'] = separation
        
        # Check 4: High entropy state has low-rank structure
        peak_states = trajectory[:, peak_idx, :]
        rank_check = self._check_low_rank(peak_states)
        checks['low_rank_structure'] = rank_check
        
        # Overall pass
        checks['passes_all'] = all([
            checks['entropy_pattern'],
            checks['peak_timing'],
            checks['mode_separation'],
            checks['low_rank_structure']
        ])
        
        return checks
    
    def _check_mode_separation(self, samples: np.ndarray) -> float:
        """
        Check separation between modes in final state.
        
        Uses k-means clustering to find modes and measure separation.
        """
        from sklearn.cluster import KMeans
        
        n_modes = self.config['data']['final_state']['n_modes']
        
        if n_modes == 1:
            return 0.0
        
        # Cluster samples
        kmeans = KMeans(n_clusters=n_modes, random_state=42, n_init=10)
        kmeans.fit(samples)
        centers = kmeans.cluster_centers_
        
        # Compute minimum pairwise distance
        min_dist = float('inf')
        for i in range(n_modes):
            for j in range(i+1, n_modes):
                dist = np.linalg.norm(centers[i] - centers[j])
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _check_low_rank(self, samples: np.ndarray) -> bool:
        """
        Check if covariance has low-rank structure.
        
        Verifies that top r eigenvalues are much larger than the rest.
        """
        r = self.config['data']['high_entropy_state']['effective_rank']
        
        # Compute covariance
        cov = np.cov(samples.T)
        
        # Eigenvalues
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]  # Descending order
        
        # Check if top r eigenvalues are significantly larger
        if len(eigvals) < r + 1:
            return True
        
        ratio = eigvals[r-1] / eigvals[r]
        return ratio > 5.0  # Top eigenvalues should be 5x larger
    
    def _summarize_checks(self, trajectory_checks: List[Dict]) -> Dict:
        """Summarize checks across all trajectories"""
        summary = {
            'entropy_pattern_pass_rate': np.mean([
                c['entropy_pattern'] for c in trajectory_checks
            ]),
            'peak_timing_pass_rate': np.mean([
                c['peak_timing'] for c in trajectory_checks
            ]),
            'mode_separation_pass_rate': np.mean([
                c['mode_separation'] for c in trajectory_checks
            ]),
            'low_rank_pass_rate': np.mean([
                c['low_rank_structure'] for c in trajectory_checks
            ]),
            'mean_peak_time': np.mean([
                c['peak_time'] for c in trajectory_checks
            ]),
            'mean_separation': np.mean([
                c['actual_separation'] for c in trajectory_checks
            ])
        }
        return summary
    
    def save_report(self, report: Dict, save_path: str):
        """Save quality report to JSON"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj
        
        report_serializable = convert(report)
        
        with open(save_path, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        print(f"Quality report saved to {save_path}")
