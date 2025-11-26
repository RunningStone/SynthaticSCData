#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Visualizer - Common Infrastructure

Provides common infrastructure for all visualizers including:
- Output directory management
- Device management (CPU/CUDA)
- Random seed setting
- File saving utilities
- Logging utilities
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class BaseVisualizer(ABC):
    """
    Abstract base class for all visualizers.
    
    Provides common infrastructure:
    - Output directory management
    - Device management
    - Random seed setting
    - File saving utilities
    - Logging utilities
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        device: str = 'cuda',
        random_seed: int = 42
    ):
        """
        Initialize base visualizer.
        
        Args:
            output_dir: Directory to save outputs
            device: Device for computation ('cuda' or 'cpu')
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.random_seed = random_seed
        
        # Ensure output directory exists
        self._ensure_output_dir()
        
        # Set random seeds
        self._set_random_seeds()
    
    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
    
    def _save_figure(
        self,
        fig: plt.Figure,
        prefix: str,
        formats: List[str] = ['png', 'pdf'],
        dpi: int = 300,
        **kwargs
    ):
        """
        Save figure in multiple formats.
        
        Args:
            fig: Matplotlib figure to save
            prefix: Filename prefix (without extension)
            formats: List of formats to save ('png', 'pdf', 'svg', etc.)
            dpi: DPI for raster formats
            **kwargs: Additional arguments for savefig
        
        Returns:
            List of saved file paths
        """
        saved_paths = []
        
        for fmt in formats:
            output_path = self.output_dir / f'{prefix}.{fmt}'
            
            if fmt in ['png', 'jpg', 'jpeg']:
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight', **kwargs)
            else:
                fig.savefig(output_path, bbox_inches='tight', **kwargs)
            
            saved_paths.append(output_path)
        
        return saved_paths
    
    def _save_dataframe(self, df, filename: str):
        """
        Save pandas DataFrame to CSV.
        
        Args:
            df: Pandas DataFrame
            filename: Output filename (with or without .csv extension)
        
        Returns:
            Path to saved file
        """
        if not filename.endswith('.csv'):
            filename = f'{filename}.csv'
        
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def _save_dict_to_json(self, data: Dict, filename: str):
        """
        Save dictionary to JSON file.
        
        Args:
            data: Dictionary to save
            filename: Output filename (with or without .json extension)
        
        Returns:
            Path to saved file
        """
        import json
        
        if not filename.endswith('.json'):
            filename = f'{filename}.json'
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def _print_section(self, title: str, width: int = 80):
        """Print a formatted section header"""
        print("\n" + "="*width)
        print(title)
        print("="*width)
    
    def _print_subsection(self, title: str, width: int = 80):
        """Print a formatted subsection header"""
        print("\n" + "-"*width)
        print(title)
        print("-"*width)
    
    def _print_info(self, message: str, indent: int = 2):
        """Print an info message with indentation"""
        print(" " * indent + message)
    
    def _print_success(self, message: str, indent: int = 2):
        """Print a success message with checkmark"""
        print(" " * indent + f"✓ {message}")
    
    def _print_warning(self, message: str, indent: int = 2):
        """Print a warning message"""
        print(" " * indent + f"⚠️  {message}")
    
    def _print_error(self, message: str, indent: int = 2):
        """Print an error message"""
        print(" " * indent + f"❌ {message}")
    
    def log(self, message: str):
        """
        Log a message to console.
        
        Args:
            message: Message to log
        """
        print(message)
    
    def get_info(self) -> Dict:
        """
        Get visualizer information.
        
        Returns:
            Dictionary containing visualizer metadata
        """
        return {
            'class': self.__class__.__name__,
            'output_dir': str(self.output_dir),
            'device': self.device,
            'random_seed': self.random_seed
        }
    
    def __repr__(self) -> str:
        """String representation"""
        info = self.get_info()
        return f"{info['class']}(output_dir='{info['output_dir']}', device='{info['device']}')"
