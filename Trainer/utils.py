#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training utilities and experiment orchestration
"""

import torch
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from Model import (
    SchrodingerBridgeModel,
    MLPlus_SchrodingerBridgeModel,
    OptimalTransportModel,
    BatchOTModel
)
from Model.c_vae_model import ConditionalVAEModel
from .sb_trainer import SBTrainer
from .unified_trainer import UnifiedTrainer
from .batch_ot_trainer import BatchOTTrainer
from .sb_evaluator import Evaluator


def train_model(
    model_name: str,
    model_config: Dict[str, Any],
    train_loader,
    test_loader,
    dimension: int,
    time_labels: list,
    config: Dict[str, Any],
    logger: logging.Logger
):
    """训练单个模型"""
    
    logger.info("\n" + "="*70)
    logger.info(f"Training Model: {model_name.upper()}")
    logger.info("="*70)
    
    device = config['settings']['device']
    output_dir = Path(config['settings']['output_dir'])
    checkpoint_dir = output_dir / config['settings']['subdirs']['checkpoints'] / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    arch_config = model_config['architecture']
    train_config = model_config['training']
    
    # 创建模型
    if model_name == 'sb':
        model = SchrodingerBridgeModel(
            dimension=dimension,
            hidden_dims=arch_config['hidden_dims'],
            time_embedding_dim=arch_config['time_embedding_dim'],
            dropout=arch_config['dropout'],
            diffusion_coeff=arch_config['diffusion_coeff']
        ).to(device)
        
        # Get optimizer and training parameters from config
        optimizer_kwargs = train_config.get('optimizer_kwargs', {})
        scheduler_config = train_config.get('scheduler', {})
        grad_clip_config = train_config.get('gradient_clipping', {})
        
        trainer = SBTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=train_config['learning_rate'],
            device=device,
            output_dir=str(checkpoint_dir),
            weight_decay=optimizer_kwargs.get('weight_decay', 1e-5),
            grad_clip_norm=grad_clip_config.get('max_norm', 5.0),
            optimizer_kwargs=optimizer_kwargs,
            scheduler_config=scheduler_config
        )
        
    elif model_name == 'sb_mlplus':
        model = MLPlus_SchrodingerBridgeModel(
            dimension=dimension,
            hidden_dim=arch_config['hidden_dim'],
            n_blocks=arch_config['n_blocks'],
            time_embedding_dim=arch_config['time_embedding_dim'],
            n_time_frequencies=arch_config['n_time_frequencies'],
            dropout=arch_config['dropout'],
            diffusion_coeff=arch_config['diffusion_coeff']
        ).to(device)
        
        # Get optimizer and training parameters from config
        optimizer_kwargs = train_config.get('optimizer_kwargs', {})
        scheduler_config = train_config.get('scheduler', {})
        grad_clip_config = train_config.get('gradient_clipping', {})
        
        trainer = SBTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=train_config['learning_rate'],
            device=device,
            output_dir=str(checkpoint_dir),
            weight_decay=optimizer_kwargs.get('weight_decay', 1e-5),
            grad_clip_norm=grad_clip_config.get('max_norm', 5.0),
            optimizer_kwargs=optimizer_kwargs,
            scheduler_config=scheduler_config
        )
        
    elif model_name == 'ot':
        model = OptimalTransportModel(
            dimension=dimension,
            hidden_dims=arch_config['hidden_dims'],
            activation=arch_config['activation'],
            dropout=arch_config['dropout'],
            use_residual=arch_config.get('use_residual', True)
        ).to(device)
        
        trainer = UnifiedTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=train_config['learning_rate'],
            device=device,
            model_type='ot',
            output_dir=str(checkpoint_dir)
        )
        
    elif model_name == 'vae':
        # Get number of timepoints from time_labels
        n_timepoints = len(time_labels)
        
        model = ConditionalVAEModel(
            dimension=dimension,
            n_timepoints=n_timepoints,
            hidden_dims=arch_config['hidden_dims'],
            latent_dim=arch_config['latent_dim'],
            time_embedding_dim=arch_config.get('time_embedding_dim', 64),
            activation=arch_config['activation'],
            dropout=arch_config['dropout'],
            beta=arch_config['beta'],
            mmd_weight=arch_config.get('mmd_weight', 1.0),
            mmd_kernel=arch_config.get('mmd_kernel', 'rbf'),
            mmd_bandwidth=arch_config.get('mmd_bandwidth', 1.0)
        ).to(device)
        
        trainer = UnifiedTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=train_config['learning_rate'],
            device=device,
            model_type='vae',
            output_dir=str(checkpoint_dir)
        )
        
    elif model_name == 'batch_ot':
        # Get number of timepoints from time_labels
        n_timepoints = len(time_labels)
        
        model = BatchOTModel(
            dimension=dimension,
            n_timepoints=n_timepoints,
            time_labels=time_labels,
            hidden_dims=arch_config['hidden_dims'],
            activation=arch_config['activation'],
            dropout=arch_config['dropout'],
            use_residual=arch_config.get('use_residual', True)
        ).to(device)
        
        # Get optimizer and training parameters from config
        optimizer_kwargs = train_config.get('optimizer_kwargs', {})
        grad_clip_config = train_config.get('gradient_clipping', {})
        
        trainer = BatchOTTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            learning_rate=train_config['learning_rate'],
            device=device,
            output_dir=str(checkpoint_dir),
            weight_decay=optimizer_kwargs.get('weight_decay', 1e-5),
            grad_clip_norm=grad_clip_config.get('max_norm', 5.0)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 训练
    logger.info(f"Training for {train_config['epochs']} epochs...")
    
    # Get early stopping patience from config
    early_stopping_config = train_config.get('early_stopping', {})
    patience = early_stopping_config.get('patience', 10)  # Default to 10 if not specified
    
    logger.info(f"Early stopping patience: {patience}")
    
    history = trainer.train(
        epochs=train_config['epochs'],
        early_stopping_patience=patience
    )
    
    # 评估
    logger.info("Evaluating model...")
    evaluator = Evaluator(device=device, model_name=model_name)
    results = evaluator.evaluate(
        model=model,
        test_loader=test_loader,
        time_labels=time_labels,
        model_name=model_name
    )
    
    logger.info(f"✓ {model_name.upper()} training complete")
    logger.info(f"  Test Loss: {results.get('test_loss', 'N/A'):.4f}")
    logger.info(f"  MAE: {results.get('mae', 'N/A'):.4f}")
    logger.info(f"  PCC: {results.get('pcc', 'N/A'):.4f}")
    
    return model, history, results


def run_experiment_from_config(config: Dict[str, Any], logger: logging.Logger):
    """从配置运行完整实验"""
    
    from Data import create_dataloaders_from_data
    
    logger.info("="*80)
    logger.info(f"EXPERIMENT: {config['experiment']['name']}")
    logger.info(f"Description: {config['experiment']['description']}")
    logger.info("="*80)
    
    # 设置随机种子
    seed = config['settings']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # 创建输出目录
    output_dir = Path(config['settings']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # 保存配置
    config_save_path = output_dir / 'experiment_config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Configuration saved to: {config_save_path}")
    
    # 加载数据 (假设已经在外部创建好了data_loader)
    # 这部分需要从外部传入或在这里重新创建
    from Data.config_loader import create_data_loader_from_config, validate_data_config, get_data_for_setting
    
    data_loader = create_data_loader_from_config(config, logger)
    data_loader.load_and_analyze()
    
    # 验证配置
    validate_data_config(data_loader, config, logger)
    
    # 获取数据
    X_train, y_train, X_test, y_test = get_data_for_setting(data_loader, config, logger)
    
    # 创建DataLoader
    batch_size = config['models'][list(config['models'].keys())[0]]['training']['batch_size']
    train_loader, test_loader, stats = create_dataloaders_from_data(
        X_train, y_train, X_test, y_test,
        time_labels=data_loader.time_label_order,
        batch_size=batch_size
    )
    
    logger.info(f"\nData Statistics:")
    logger.info(f"  Train samples: {stats['train_size']}")
    logger.info(f"  Test samples: {stats['test_size']}")
    logger.info(f"  Feature dimension: {stats['n_genes']}")
    logger.info(f"  Total time points defined: {stats['n_timepoints']}")
    
    # Show actually used time points (non-zero counts)
    train_used_timepoints = [tp for tp, count in stats['train_time_counts'].items() if count > 0]
    test_used_timepoints = [tp for tp, count in stats['test_time_counts'].items() if count > 0]
    
    logger.info(f"  Actually used time points in train: {len(train_used_timepoints)} {train_used_timepoints}")
    logger.info(f"  Actually used time points in test: {len(test_used_timepoints)} {test_used_timepoints}")
    
    # Show detailed counts for used time points
    logger.info(f"  Train time point distribution:")
    for tp in train_used_timepoints:
        logger.info(f"    {tp}: {stats['train_time_counts'][tp]} cells")
    logger.info(f"  Test time point distribution:")
    for tp in test_used_timepoints:
        logger.info(f"    {tp}: {stats['test_time_counts'][tp]} cells")
    
    dimension = stats['n_genes']
    
    # 训练所有模型
    all_results = {}
    for model_name in config['models'].keys():
        try:
            model, history, results = train_model(
                model_name=model_name,
                model_config=config['models'][model_name],
                train_loader=train_loader,
                test_loader=test_loader,
                dimension=dimension,
                time_labels=stats['time_labels'],
                config=config,
                logger=logger
            )
            
            all_results[model_name] = {
                'history': history,
                'evaluation': results
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            if not config.get('error_handling', {}).get('continue_on_error', False):
                raise
    
    # 保存结果
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\n✓ Results saved to: {results_path}")
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE!")
    logger.info("="*80)
    
    return all_results
