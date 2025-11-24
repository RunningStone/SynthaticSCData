#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的Evaluation脚本
仅运行评估部分，不进行训练

用途：
  - 对已训练模型进行重新评估
  - 更改评估参数（如起点/终点）后重新评估
  - 修复OOM问题后重新运行evaluation
"""

import argparse
import torch
import numpy as np
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any

from Data import ConfigLoader, setup_logging, create_dataloaders_from_data
from Data.config_loader import create_data_loader_from_config, validate_data_config, get_data_for_setting
from Model import (
    SchrodingerBridgeModel,
    MLPlus_SchrodingerBridgeModel,
    OptimalTransportModel,
    BatchOTModel
)
from Model.c_vae_model import ConditionalVAEModel
from Trainer import Evaluator


def load_model_from_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    model_config: Dict[str, Any],
    dimension: int,
    time_labels: list,
    device: str,
    logger: logging.Logger
) -> torch.nn.Module:
    """从checkpoint加载模型"""
    
    logger.info(f"Loading {model_name} from {checkpoint_path}")
    
    arch_config = model_config['architecture']
    
    # 创建模型架构
    if model_name == 'sb':
        model = SchrodingerBridgeModel(
            dimension=dimension,
            hidden_dims=arch_config['hidden_dims'],
            time_embedding_dim=arch_config['time_embedding_dim'],
            dropout=arch_config['dropout'],
            diffusion_coeff=arch_config['diffusion_coeff']
        ).to(device)
        
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
        
    elif model_name == 'ot':
        model = OptimalTransportModel(
            dimension=dimension,
            hidden_dims=arch_config['hidden_dims'],
            activation=arch_config['activation'],
            dropout=arch_config['dropout'],
            use_residual=arch_config.get('use_residual', True)
        ).to(device)
        
    elif model_name == 'vae':
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
        
    elif model_name == 'batch_ot':
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
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 加载checkpoint (PyTorch 2.6+ requires weights_only=False for compatibility)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 对于batch_ot和vae，如果checkpoint中保存了训练时的时间点配置，使用它来重新初始化模型
    trained_time_labels = None
    
    if model_name in ['batch_ot', 'vae']:
        # 尝试从checkpoint中提取训练时的时间点
        if 'time_labels' in checkpoint:
            trained_time_labels = checkpoint['time_labels']
        elif model_name == 'batch_ot' and 'time_pairs' in checkpoint:
            # 从time_pairs提取所有唯一的时间点
            time_pairs = checkpoint['time_pairs']
            time_set = set()
            for start, end in time_pairs:
                time_set.add(start)
                time_set.add(end)
            # 保持原始顺序（假设time_pairs是按顺序的）
            trained_time_labels = []
            seen = set()
            for start, end in time_pairs:
                if start not in seen:
                    trained_time_labels.append(start)
                    seen.add(start)
                if end not in seen:
                    trained_time_labels.append(end)
                    seen.add(end)
        elif model_name == 'vae' and 'model_state_dict' in checkpoint:
            # 从time_embedding的shape提取时间点数量
            if 'time_embedding.weight' in checkpoint['model_state_dict']:
                n_trained_timepoints = checkpoint['model_state_dict']['time_embedding.weight'].shape[0]
                logger.info(f"VAE was trained with {n_trained_timepoints} timepoints")
                # 我们不知道具体的时间点标签，但可以使用数量来重新初始化
                # 这里我们需要一个占位符列表
                trained_time_labels = [f't{i}' for i in range(n_trained_timepoints)]
        
        if trained_time_labels:
            logger.info(f"Model was trained with time labels: {trained_time_labels}")
            
            # 重新初始化模型以匹配训练时的配置
            if model_name == 'batch_ot':
                model = BatchOTModel(
                    dimension=dimension,
                    n_timepoints=len(trained_time_labels),
                    time_labels=trained_time_labels,
                    hidden_dims=arch_config['hidden_dims'],
                    activation=arch_config['activation'],
                    dropout=arch_config['dropout'],
                    use_residual=arch_config.get('use_residual', True)
                ).to(device)
            elif model_name == 'vae':
                model = ConditionalVAEModel(
                    dimension=dimension,
                    n_timepoints=len(trained_time_labels),
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
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✓ Loaded model weights from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        logger.info(f"✓ Loaded model weights")
    
    model.eval()
    return model


def evaluate_model(
    model_name: str,
    model: torch.nn.Module,
    test_loader,
    time_labels: list,
    config: Dict[str, Any],
    logger: logging.Logger,
    output_dir: Path,
    config_dir: str = 'configs'
) -> Dict:
    """评估单个模型并保存生成的数据"""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating Model: {model_name.upper()}")
    logger.info(f"{'='*70}")
    
    device = config['settings']['device']
    
    # 统一评估框架：从Setting1配置中读取起点和终点
    # config已经是Setting1的配置，直接从数据配置中读取time_points
    data_config_name = config['configs']['data']
    data_config_path = Path(config_dir) / data_config_name
    
    # 加载数据配置文件
    import yaml
    with open(data_config_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 获取setting1的time_points
    setting1_config = data_config.get('setting1', {})
    time_points = setting1_config.get('time_points', ['0d', '7d'])
    
    # Setting1应该只有两个时间点：起点和终点
    if len(time_points) >= 2:
        start_timepoint = time_points[0]
        end_timepoint = time_points[-1]
    else:
        raise ValueError(f"Setting1 must have at least 2 timepoints, got: {time_points}")
    
    logger.info(f"Unified evaluation framework: {start_timepoint} → {end_timepoint}")
    
    evaluator = Evaluator(
        device=device,
        model_name=model_name,
        start_timepoint=start_timepoint,
        end_timepoint=end_timepoint
    )
    
    logger.info("Running evaluation...")
    results = evaluator.evaluate(
        model=model,
        test_loader=test_loader,
        time_labels=time_labels,
        model_name=model_name
    )
    
    logger.info(f"✓ {model_name.upper()} evaluation complete")
    logger.info(f"  Test Loss: {results.get('test_loss', float('nan')):.4f}")
    logger.info(f"  MAE: {results.get('mae', float('nan')):.4f}")
    logger.info(f"  PCC: {results.get('pcc', float('nan')):.4f}")
    logger.info(f"  Frechet Distance: {results.get('frechet_distance', float('nan')):.4f}")
    logger.info(f"  Wasserstein Distance: {results.get('wasserstein_distance', float('nan')):.4f}")
    
    # 保存生成的数据
    logger.info("Generating and saving samples for visualization...")
    generated_data = evaluator.generate_samples_for_visualization(
        model=model,
        test_loader=test_loader,
        time_labels=time_labels,
        model_name=model_name
    )
    
    # 创建generated文件夹
    generated_dir = output_dir / 'generated'
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为pkl文件
    pkl_path = generated_dir / f'{model_name}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(generated_data, f)
    
    logger.info(f"✓ Generated samples saved to: {pkl_path}")
    
    return results


def run_evaluation_only(
    setting1_config: str,
    checkpoint_dir: str,
    config_dir: str = 'configs',
    output_suffix: str = '_re_eval'
):
    """运行仅评估模式
    
    Args:
        setting1_config: Setting1配置文件，定义了统一评估的起点和终点
        checkpoint_dir: 包含模型checkpoints的目录
        config_dir: 配置文件目录
        output_suffix: 输出文件后缀
    """
    
    # 确定checkpoint目录和输出目录
    checkpoint_base = Path(checkpoint_dir)
    if not checkpoint_base.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # 输出目录是checkpoint目录的父目录
    output_dir = checkpoint_base.parent
    
    # 加载两个配置：
    # 1. Setting1配置 - 用于统一评估数据（起点和终点）
    # 2. 当前setting的experiment_config.yaml - 用于模型配置
    config_loader = ConfigLoader(config_dir=config_dir)
    
    # 首先加载Setting1的原始配置（未展开）以获取configs引用
    import yaml
    setting1_raw_path = Path(config_dir) / setting1_config
    with open(setting1_raw_path, 'r', encoding='utf-8') as f:
        setting1_raw = yaml.safe_load(f)
    
    # 然后加载展开后的Setting1配置
    setting1_eval_config = config_loader.load_experiment_config(setting1_config)
    
    # 从输出目录读取该setting的experiment_config.yaml
    experiment_config_path = output_dir / 'experiment_config.yaml'
    if not experiment_config_path.exists():
        raise ValueError(f"experiment_config.yaml not found in {output_dir}")
    
    # 加载模型配置
    with open(experiment_config_path, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    # 合并配置：使用Setting1的数据配置，但使用当前setting的模型配置
    config = setting1_eval_config.copy()
    config['models'] = model_config['models']  # 使用当前setting的模型配置
    config['configs'] = setting1_raw['configs']  # 保留configs引用信息
    
    # 设置日志
    # 使用原输出目录，但修改日志文件名
    original_log_file = config['settings']['logging']['log_file']
    config['settings']['logging']['log_file'] = f"evaluation_only{output_suffix}.log"
    logger = setup_logging(config)
    
    logger.info("="*80)
    logger.info("EVALUATION-ONLY MODE (Unified Evaluation Framework)")
    logger.info("="*80)
    logger.info(f"Setting1 config (evaluation data): {setting1_config}")
    logger.info(f"Model config (from experiment): {experiment_config_path}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Models to evaluate: {list(config['models'].keys())}")
    logger.info("")
    
    # 设置随机种子
    seed = config['settings']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # 加载数据
    logger.info("\nLoading data for unified evaluation...")
    data_loader = create_data_loader_from_config(config, logger)
    data_loader.load_and_analyze()
    
    # 统一评估：从Setting1配置中获取起点和终点
    # 读取数据配置文件
    data_config_name = config['configs']['data']
    data_config_path = Path(config_dir) / data_config_name
    
    # 加载数据配置文件
    import yaml
    with open(data_config_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 获取setting1的time_points作为评估的起点和终点
    setting1_config = data_config.get('setting1', {})
    eval_timepoints = setting1_config.get('time_points', ['0d', '7d'])
    
    logger.info(f"Unified evaluation framework (from Setting1):")
    logger.info(f"  Evaluation timepoints: {eval_timepoints}")
    
    # 获取测试集数据（使用biology split配置）
    biology_split_config = config['biology_split']
    column_name = biology_split_config['column_name']
    test_values = biology_split_config['test_values']
    
    test_mask = data_loader.adata_hvg.obs[column_name].isin(test_values)
    
    # 只保留Setting1定义的时间点
    time_mask = data_loader.adata_hvg.obs[data_loader.obs_time_column].isin(eval_timepoints)
    
    # 组合mask
    final_mask = test_mask & time_mask
    
    # 提取数据
    X_test = data_loader.adata_hvg.X[final_mask].toarray() if hasattr(data_loader.adata_hvg.X, 'toarray') else data_loader.adata_hvg.X[final_mask]
    y_test_labels = data_loader.adata_hvg.obs[data_loader.obs_time_column][final_mask].values
    
    # 将时间标签转换为索引
    y_test = np.array([data_loader.time_label_order.index(label) for label in y_test_labels])
    
    logger.info(f"Unified evaluation dataset prepared:")
    logger.info(f"  Time points: {eval_timepoints}")
    logger.info(f"  Total test samples: {len(X_test)}")
    for tp in eval_timepoints:
        count = np.sum(y_test_labels == tp)
        logger.info(f"    {tp}: {count} samples")
    
    # 创建虚拟的训练数据（评估时不使用，但需要保持接口一致）
    X_train = X_test[:10]  # 只取少量样本
    y_train = y_test[:10]
    
    # 创建DataLoader
    batch_size = config['models'][list(config['models'].keys())[0]]['training']['batch_size']
    train_loader, test_loader, stats = create_dataloaders_from_data(
        X_train, y_train, X_test, y_test,
        time_labels=data_loader.time_label_order,
        batch_size=batch_size
    )
    
    logger.info(f"\nData Statistics:")
    logger.info(f"  Test samples: {stats['test_size']}")
    logger.info(f"  Feature dimension: {stats['n_genes']}")
    logger.info(f"  Time points: {stats['time_labels']}")
    
    dimension = stats['n_genes']
    device = config['settings']['device']
    
    # 查找并评估所有模型
    all_results = {}
    
    # 遍历checkpoint目录找到所有模型
    for model_dir in sorted(checkpoint_base.iterdir()):
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        
        # 检查是否是支持的模型
        if model_name not in config['models']:
            logger.warning(f"Skipping unknown model: {model_name}")
            continue
        
        # 查找final_model.pt或best_model.pt
        checkpoint_path = model_dir / 'final_model.pt'
        if not checkpoint_path.exists():
            checkpoint_path = model_dir / 'best_model.pt'
        
        if not checkpoint_path.exists():
            logger.warning(f"No checkpoint found for {model_name} in {model_dir}")
            continue
        
        try:
            # 加载模型
            model = load_model_from_checkpoint(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                model_config=config['models'][model_name],
                dimension=dimension,
                time_labels=stats['time_labels'],
                device=device,
                logger=logger
            )
            
            # 评估模型
            results = evaluate_model(
                model_name=model_name,
                model=model,
                test_loader=test_loader,
                time_labels=stats['time_labels'],
                config=config,
                logger=logger,
                output_dir=output_dir,
                config_dir=config_dir
            )
            
            all_results[model_name] = {
                'checkpoint_path': str(checkpoint_path),
                'evaluation': results
            }
            
            # 清理内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}", exc_info=True)
            if not config.get('error_handling', {}).get('continue_on_error', False):
                raise
    
    # 保存结果
    output_dir = Path(config['settings']['output_dir'])
    results_path = output_dir / f'results{output_suffix}.json'
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n✓ Evaluation results saved to: {results_path}")
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("="*80)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run evaluation only (no training) on pre-trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Re-evaluate models from EMT_Part1_Setting3 experiment
  python step2_run_evaluation_only.py \\
      experiment_EMT_Part1_setting1.yaml \\
      /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting3/checkpoints
  
  # With custom suffix for output
  python step2_run_evaluation_only.py \\
      experiment_EMT_Part1_setting1.yaml \\
      /path/to/checkpoints --output_suffix _fixed_oom
        """
    )
    parser.add_argument(
        'setting1_config',
        type=str,
        help='Setting1 configuration file for unified evaluation (e.g., experiment_EMT_Part1_setting1.yaml)'
    )
    parser.add_argument(
        'checkpoint_dir',
        type=str,
        help='Directory containing model checkpoints (e.g., .../EMT_Part1_Setting3/checkpoints)'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default='configs',
        help='Directory containing configuration files (default: configs)'
    )
    parser.add_argument(
        '--output_suffix',
        type=str,
        default='_re_eval',
        help='Suffix for output files (default: _re_eval)'
    )
    
    args = parser.parse_args()
    
    try:
        results = run_evaluation_only(
            setting1_config=args.setting1_config,
            checkpoint_dir=args.checkpoint_dir,
            config_dir=args.config_dir,
            output_suffix=args.output_suffix
        )
        return results
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
