#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run experiment from YAML configuration files
Entry point for the modular configuration system
"""

import argparse
from pathlib import Path

from Data import ConfigLoader, setup_logging
from Trainer import run_experiment_from_config


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description='Run experiment from YAML configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Part1 (Forward EMT only)
  python step1_run_experiment.py experiment_EMT_Part1_setting1.yaml
  python step1_run_experiment.py experiment_EMT_Part1_setting2.yaml
  python step1_run_experiment.py experiment_EMT_Part1_setting3.yaml
  
  # Part2 (With Reversal)
  python step1_run_experiment.py experiment_EMT_Part2_setting1.yaml
  python step1_run_experiment.py experiment_EMT_Part2_setting2.yaml
  python step1_run_experiment.py experiment_EMT_Part2_setting3.yaml
  
  # Specify custom config directory
  python step1_run_experiment.py experiment_EMT_Part1_setting1.yaml --config_dir my_configs
  
  # Override output directory
  python step1_run_experiment.py experiment_EMT_Part1_setting1.yaml --output_dir /custom/path
        """
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Experiment configuration file (e.g., experiment_EMT_Part1_setting1.yaml)'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default='configs',
        help='Directory containing configuration files (default: configs)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory from config file'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config_loader = ConfigLoader(config_dir=args.config_dir)
    config = config_loader.load_experiment_config(args.config_file)
    
    # 覆盖输出目录（如果指定）
    # 直接使用传入的 output_dir，不追加任何子目录
    if args.output_dir is not None:
        config['settings']['output_dir'] = args.output_dir
    
    # 设置日志
    logger = setup_logging(config)
    
    # 运行实验
    try:
        results = run_experiment_from_config(config, logger)
        return results
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
