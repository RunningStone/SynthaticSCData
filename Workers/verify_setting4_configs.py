#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI tool to verify Setting4 ablation configuration files.

Usage:
    python Workers/verify_setting4_configs.py
    python Workers/verify_setting4_configs.py --config_dir configs/EMT_E2M

Author: Shi Pan
Date: 2024-11-24
"""

import argparse
from pathlib import Path
from utils import verify_all_setting4_ablation_configs


def main():
    """Main entry point for configuration verification."""
    parser = argparse.ArgumentParser(
        description='Verify Setting4 ablation configuration files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify configs in default directory
  python Workers/verify_setting4_configs.py
  
  # Verify configs in custom directory
  python Workers/verify_setting4_configs.py --config_dir my_configs/EMT_E2M
        """
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default='configs/EMT_E2M',
        help='Directory containing configuration files (default: configs/EMT_E2M)'
    )
    
    args = parser.parse_args()
    
    # Get absolute path
    configs_dir = Path(args.config_dir)
    if not configs_dir.is_absolute():
        # Assume relative to project root
        project_root = Path(__file__).parent.parent
        configs_dir = project_root / configs_dir
    
    # Verify all configurations
    all_valid = verify_all_setting4_ablation_configs(configs_dir)
    
    # Exit with appropriate code
    exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
