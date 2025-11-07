"""Trajectory Visualizer - Creates PCA/UMAP plots of trajectories"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Dict, Optional

class TrajectoryVisualizer:
    def __init__(self, config: Dict):
        self.config = config
        self.n_components = config['visualization']['pca_components']
        self.dpi = config['visualization']['dpi']
    
    def plot_trajectories(self, dataset: Dict, save_path: Optional[str] = None, n_plot: int = 5):
        """Plot trajectories in PCA space"""
        trajectories = dataset['trajectories'][:n_plot]
        time_grid = dataset['time_stamps']
        entropy_curves = dataset['entropy_curves'][:n_plot]
        
        # Flatten for PCA
        all_states = trajectories.reshape(-1, trajectories.shape[-1])
        pca = PCA(n_components=min(3, self.n_components))
        states_pca = pca.fit_transform(all_states)
        states_pca = states_pca.reshape(trajectories.shape[0], trajectories.shape[1], trajectories.shape[2], -1)
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(131, projection='3d' if states_pca.shape[-1] >= 3 else None)
        for i in range(n_plot):
            traj_pca = states_pca[i].mean(axis=0)  # Average over cells
            if states_pca.shape[-1] >= 3:
                ax1.plot(traj_pca[:, 0], traj_pca[:, 1], traj_pca[:, 2], alpha=0.7)
            else:
                ax1.plot(traj_pca[:, 0], traj_pca[:, 1], alpha=0.7)
        ax1.set_title('Trajectories in PCA Space')
        
        # Entropy evolution
        ax2 = fig.add_subplot(132)
        for i in range(n_plot):
            ax2.plot(time_grid, entropy_curves[i], alpha=0.7)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Entropy')
        ax2.set_title('Entropy Evolution')
        ax2.grid(True, alpha=0.3)
        
        # Variance evolution
        ax3 = fig.add_subplot(133)
        for i in range(n_plot):
            variances = np.var(trajectories[i], axis=0).sum(axis=1)
            ax3.plot(time_grid, variances, alpha=0.7)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Total Variance')
        ax3.set_title('Variance Evolution')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Trajectory plot saved to {save_path}")
        plt.close()
