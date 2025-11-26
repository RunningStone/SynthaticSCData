#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration loader and data preparation utilities
"""

import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from .data_loader import RealDataLoader
from .interpolated_data_loader import InterpolatedDataLoader


class ConfigLoader:
    """加载和合并配置文件"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """加载单个YAML文件"""
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_experiment_config(self, experiment_file: str) -> Dict[str, Any]:
        """加载完整的实验配置（包含所有引用的配置）"""
        
        # 1. 加载实验配置
        exp_config = self.load_yaml(experiment_file)
        
        # 2. 加载引用的配置文件
        data_config = self.load_yaml(exp_config['configs']['data'])
        models_config = self.load_yaml(exp_config['configs']['models'])
        analyzer_config = self.load_yaml(exp_config['configs']['analyzer'])
        
        # 3. 提取指定的data setting
        data_setting_name = exp_config['data_setting']
        selected_setting = data_config[data_setting_name]
        
        # 4. 合并配置
        full_config = {
            'experiment': exp_config['experiment'],
            'data_source': data_config['data_source'],
            'biology_split': data_config['biology_split'],
            'data_setting': selected_setting,
            'data_setting_name': data_setting_name,
            'data_sampling_override': exp_config.get('data_sampling_override'),  # Add experiment-level sampling override
            'validation': data_config.get('validation', {}),
            'models': {},
            'analyzer': analyzer_config,
            'settings': exp_config['settings'],
            'evaluation': exp_config['evaluation'],
            'post_analysis': exp_config.get('post_analysis', {})
        }
        
        # 5. 处理每个要训练的模型
        for model_spec in exp_config['models_to_train']:
            if not model_spec['enabled']:
                continue
            
            model_name = model_spec['name']
            model_config = models_config[model_name].copy()
            
            # 应用override参数
            if model_spec.get('override_params'):
                model_config = self._deep_merge(
                    model_config, 
                    model_spec['override_params']
                )
            
            full_config['models'][model_name] = model_config
        
        return full_config
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """设置日志系统"""
    output_dir = Path(config['settings']['output_dir'])
    log_dir = output_dir / config['settings']['subdirs']['logs']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_config = config['settings']['logging']
    log_file = log_dir / log_config['log_file']
    
    # 创建logger
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.DEBUG)
    
    # 清除已有的handlers
    logger.handlers = []
    
    # 文件handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_config['file_level']))
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_config['console_level']))
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def create_data_loader_from_config(config: Dict[str, Any], logger: logging.Logger) -> RealDataLoader:
    """从配置创建数据加载器
    
    支持两种数据加载器：
    1. RealDataLoader: 标准数据加载（默认）
    2. InterpolatedDataLoader: 插值数据生成
    """
    
    data_source = config['data_source']
    biology_split = config['biology_split']
    setting_config = config['data_setting']
    data_sampling_override = config.get('data_sampling_override', {}) or {}  # 获取override配置
    
    logger.info("="*70)
    logger.info("Creating Data Loader from Configuration")
    logger.info("="*70)
    
    # 构建biology_split参数
    if biology_split['column_name'] is None:
        logger.info("Using random train/test split")
        logger.info(f"  Train ratio: {biology_split['train_ratio']}")
        split_config = {
            "train_val_column": "random",
            "train_ratio": biology_split['train_ratio']
        }
    else:
        logger.info(f"Using biology-based split on column: {biology_split['column_name']}")
        logger.info(f"  Train batches: {biology_split['train_values']}")
        logger.info(f"  Test batches: {biology_split['test_values']}")
        split_config = {
            "train_val_column": biology_split['column_name'],
            "train": biology_split['train_values'],
            "test": biology_split['test_values']
        }
    
    # Check if this is an interpolated data setting
    interpolation_params = setting_config.get('interpolation_params')
    
    if interpolation_params:
        logger.info("Using InterpolatedDataLoader for interpolated data generation")
        loader = InterpolatedDataLoader(
            file_path=data_source['file_path'],
            n_hvg=data_source['n_hvg'],
            obs_time_column=data_source['obs_time_column'],
            time_labels=data_source['time_labels_order'],
            time_label_order=data_source['time_labels_order'],
            biology_split=split_config,
            random_seed=config['settings']['seed'],
            interpolation_params=interpolation_params
        )
    else:
        logger.info("Using RealDataLoader for standard data loading")
        loader = RealDataLoader(
            file_path=data_source['file_path'],
            n_hvg=data_source['n_hvg'],
            obs_time_column=data_source['obs_time_column'],
            time_labels=data_source['time_labels_order'],
            time_label_order=data_source['time_labels_order'],
            biology_split=split_config,
            random_seed=config['settings']['seed']
        )
    
    return loader


def validate_data_config(loader: RealDataLoader, config: Dict[str, Any], logger: logging.Logger):
    """验证数据配置的有效性"""
    
    validation = config.get('validation', {})
    setting_config = config['data_setting']
    
    logger.info("\n" + "="*70)
    logger.info("Validating Data Configuration")
    logger.info("="*70)
    
    # 1. 检查时间点是否存在
    for time_point in setting_config['time_points']:
        if time_point not in loader.time_label_order:
            raise ValueError(
                f"Time point '{time_point}' not in time_labels_order: "
                f"{loader.time_label_order}"
            )
    
    logger.info(f"✓ All time points valid: {setting_config['time_points']}")
    
    # 2. 检查是否有足够的细胞
    if validation.get('check_sufficient_cells', True):
        min_required = setting_config.get('min_cells_required', 0)
        for time_point in setting_config['time_points']:
            time_idx = loader.time_label_order.index(time_point)
            n_cells = (loader.adata.obs[loader.obs_time_column] == time_point).sum()
            
            if n_cells < min_required:
                raise ValueError(
                    f"Time point '{time_point}' has only {n_cells} cells, "
                    f"but {min_required} required"
                )
        
        logger.info(f"✓ All time points have sufficient cells (min: {min_required})")
    
    # 3. 验证训练/测试集包含所有时间点
    if validation.get('check_all_timepoints_present', True):
        valid = loader.validate_biology_split()
        if not valid:
            logger.warning("⚠️  Train/test split validation failed!")
        else:
            logger.info("✓ Train/test split validation passed")


def get_data_for_setting(
    loader: RealDataLoader, 
    config: Dict[str, Any], 
    logger: logging.Logger
) -> tuple:
    """根据setting配置获取数据，支持experiment级别的覆盖"""
    
    setting_config = config['data_setting']
    time_points = setting_config['time_points']  # 配置中指定的时间点
    balance_strategy = setting_config['balance_strategy']
    
    logger.info(f"\nPreparing data for setting: {config['data_setting_name']}")
    logger.info(f"Time points to use: {time_points}")
    logger.info(f"Balance strategy: {balance_strategy}")
    
    # 确定setting类型
    if len(time_points) == 2 and time_points == [loader.time_label_order[0], loader.time_label_order[-1]]:
        setting = 1
        cells_per_timepoint = setting_config['cells_per_timepoint']
        total_cells = None
        
        # Check for experiment-level override
        if 'data_sampling_override' in config and config['data_sampling_override']:
            override = config['data_sampling_override']
            if 'cells_per_timepoint' in override:
                original_value = cells_per_timepoint
                cells_per_timepoint = override['cells_per_timepoint']
                logger.info(f"⚙️  Experiment override: cells_per_timepoint {original_value} → {cells_per_timepoint}")
        
        logger.info(f"Detected Setting 1 (boundary): {cells_per_timepoint} cells per timepoint")
        logger.info(f"  → Total samples: {cells_per_timepoint * len(time_points)}")
    else:
        setting = 2
        cells_per_timepoint = setting_config.get('cells_per_timepoint')
        total_cells = setting_config.get('total_cells')
        
        # Check for experiment-level override
        if 'data_sampling_override' in config and config['data_sampling_override']:
            override = config['data_sampling_override']
            if 'total_cells' in override:
                original_value = total_cells
                total_cells = override['total_cells']
                logger.info(f"⚙️  Experiment override: total_cells {original_value} → {total_cells}")
            if 'cells_per_timepoint' in override:
                original_value = cells_per_timepoint
                cells_per_timepoint = override['cells_per_timepoint']
                logger.info(f"⚙️  Experiment override: cells_per_timepoint {original_value} → {cells_per_timepoint}")
        
        if balance_strategy == 'total':
            logger.info(f"Detected Setting 2 (all timepoints): {total_cells} total cells")
            logger.info(f"  → Per timepoint: ~{total_cells // len(time_points)}")
        else:
            logger.info(f"Detected Setting 2 (all timepoints): {cells_per_timepoint} cells per timepoint")
            logger.info(f"  → Total samples: {cells_per_timepoint * len(time_points)}")
    
    return loader.get_data_for_setting(
        setting=setting,
        cells_per_timepoint=cells_per_timepoint,
        total_cells=total_cells,
        balance_strategy=balance_strategy,
        selected_time_points=time_points  # 传递配置中指定的时间点
    )
