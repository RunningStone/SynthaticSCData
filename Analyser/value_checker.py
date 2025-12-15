#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Value Checker for Model Results Validation

This module provides utilities to validate model outputs before visualization.
It checks for:
1. NaN values in generated data (threshold: 10%)
2. NaN values in evaluation metrics
3. Infinite values in generated data

Usage:
    from Analyser.value_checker import ValueChecker
    
    checker = ValueChecker(nan_threshold=0.1)
    
    # Check generated data
    is_valid, reason = checker.check_generated_data(data)
    
    # Check evaluation results
    is_valid, reason = checker.check_evaluation_results(results_dict)
    
    # Filter valid models from a list
    valid_models = checker.filter_valid_models(model_data_dict, eval_results)
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)


class ValueChecker:
    """
    Validates model outputs for visualization readiness.
    
    Checks for NaN/Inf values in generated data and evaluation metrics
    to ensure models are suitable for visualization.
    """
    
    def __init__(
        self,
        nan_threshold: float = 0.1,
        inf_threshold: float = 0.01,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            nan_threshold: Maximum allowed fraction of NaN values (default: 0.1 = 10%)
            inf_threshold: Maximum allowed fraction of Inf values (default: 0.01 = 1%)
            logger: Optional logger instance
        """
        self.nan_threshold = nan_threshold
        self.inf_threshold = inf_threshold
        self.logger = logger or logging.getLogger(__name__)
    
    def check_generated_data(
        self,
        data: np.ndarray,
        model_name: str = "unknown"
    ) -> Tuple[bool, str]:
        """
        Check if generated data is valid for visualization.
        
        Args:
            data: Generated data array
            model_name: Name of the model (for logging)
            
        Returns:
            (is_valid, reason): Tuple of validity flag and reason string
        """
        if data is None:
            return False, "Data is None"
        
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception as e:
                return False, f"Cannot convert to numpy array: {e}"
        
        total_elements = data.size
        if total_elements == 0:
            return False, "Data is empty"
        
        # Check for NaN values
        nan_count = np.isnan(data).sum()
        nan_fraction = nan_count / total_elements
        
        if nan_fraction > self.nan_threshold:
            reason = f"NaN fraction ({nan_fraction:.1%}) exceeds threshold ({self.nan_threshold:.1%})"
            self.logger.warning(f"[{model_name}] {reason}")
            return False, reason
        
        # Check for Inf values
        inf_count = np.isinf(data).sum()
        inf_fraction = inf_count / total_elements
        
        if inf_fraction > self.inf_threshold:
            reason = f"Inf fraction ({inf_fraction:.1%}) exceeds threshold ({self.inf_threshold:.1%})"
            self.logger.warning(f"[{model_name}] {reason}")
            return False, reason
        
        # Log warning for any NaN/Inf (even if below threshold)
        if nan_count > 0 or inf_count > 0:
            self.logger.info(
                f"[{model_name}] Data has {nan_count} NaN ({nan_fraction:.2%}) "
                f"and {inf_count} Inf ({inf_fraction:.2%}) values (within threshold)"
            )
        
        return True, "OK"
    
    def check_evaluation_results(
        self,
        results: Dict[str, Any],
        model_name: str = "unknown"
    ) -> Tuple[bool, str]:
        """
        Check if evaluation results contain any NaN metrics.
        
        Args:
            results: Dictionary of evaluation metrics
            model_name: Name of the model (for logging)
            
        Returns:
            (is_valid, reason): Tuple of validity flag and reason string
        """
        if results is None:
            return False, "Results is None"
        
        if not isinstance(results, dict):
            return False, f"Results is not a dict: {type(results)}"
        
        nan_metrics = []
        
        for key, value in results.items():
            if value is None:
                nan_metrics.append(key)
            elif isinstance(value, float):
                if np.isnan(value) or np.isinf(value):
                    nan_metrics.append(key)
            elif isinstance(value, (list, np.ndarray)):
                arr = np.array(value)
                if np.isnan(arr).any() or np.isinf(arr).any():
                    nan_metrics.append(key)
        
        if nan_metrics:
            reason = f"NaN/Inf in metrics: {nan_metrics}"
            self.logger.warning(f"[{model_name}] {reason}")
            return False, reason
        
        return True, "OK"
    
    def check_model(
        self,
        generated_data: Optional[np.ndarray],
        eval_results: Optional[Dict[str, Any]],
        model_name: str = "unknown"
    ) -> Tuple[bool, str]:
        """
        Comprehensive check for a model's outputs.
        
        Args:
            generated_data: Generated data array (can be None)
            eval_results: Evaluation results dictionary (can be None)
            model_name: Name of the model
            
        Returns:
            (is_valid, reason): Tuple of validity flag and reason string
        """
        # Check generated data if provided
        if generated_data is not None:
            is_valid, reason = self.check_generated_data(generated_data, model_name)
            if not is_valid:
                return False, f"Generated data invalid: {reason}"
        
        # Check evaluation results if provided
        if eval_results is not None:
            is_valid, reason = self.check_evaluation_results(eval_results, model_name)
            if not is_valid:
                return False, f"Evaluation results invalid: {reason}"
        
        return True, "OK"
    
    def filter_valid_models(
        self,
        model_data: Dict[str, np.ndarray],
        eval_results: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Filter models to keep only those with valid outputs.
        
        Args:
            model_data: Dictionary mapping model names to generated data
            eval_results: Optional dictionary mapping model names to evaluation results
            
        Returns:
            (valid_data, skipped_models): Tuple of valid model data dict and list of skipped model names
        """
        valid_data = {}
        skipped_models = []
        
        for model_name, data in model_data.items():
            # Get evaluation results for this model if available
            model_eval = None
            if eval_results is not None and model_name in eval_results:
                model_eval = eval_results[model_name]
            
            is_valid, reason = self.check_model(data, model_eval, model_name)
            
            if is_valid:
                valid_data[model_name] = data
                self.logger.info(f"[{model_name}] ✓ Valid for visualization")
            else:
                skipped_models.append(model_name)
                self.logger.warning(f"[{model_name}] ✗ Skipped: {reason}")
        
        if skipped_models:
            self.logger.warning(
                f"Skipped {len(skipped_models)} models due to invalid data: {skipped_models}"
            )
        
        return valid_data, skipped_models
    
    @staticmethod
    def load_evaluation_results(eval_path: Union[str, Path]) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Load evaluation results from JSON file.
        
        Args:
            eval_path: Path to evaluation_results.json
            
        Returns:
            Dictionary of model evaluation results, or None if file doesn't exist
        """
        eval_path = Path(eval_path)
        if not eval_path.exists():
            return None
        
        try:
            with open(eval_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load evaluation results from {eval_path}: {e}")
            return None


def validate_models_for_visualization(
    generated_data_dict: Dict[str, np.ndarray],
    eval_results_path: Optional[Union[str, Path]] = None,
    nan_threshold: float = 0.1,
    logger: Optional[logging.Logger] = None
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Convenience function to validate models for visualization.
    
    Args:
        generated_data_dict: Dictionary mapping model names to generated data
        eval_results_path: Optional path to evaluation_results.json
        nan_threshold: Maximum allowed NaN fraction
        logger: Optional logger instance
        
    Returns:
        (valid_data, skipped_models): Tuple of valid model data and skipped model names
    """
    checker = ValueChecker(nan_threshold=nan_threshold, logger=logger)
    
    # Load evaluation results if path provided
    eval_results = None
    if eval_results_path is not None:
        eval_results = checker.load_evaluation_results(eval_results_path)
    
    return checker.filter_valid_models(generated_data_dict, eval_results)
