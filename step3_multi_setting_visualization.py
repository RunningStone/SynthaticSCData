#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Setting Visualization Script (Refactored)

基于新的模块化Analyser组件进行可视化对比。

使用新的模块化架构：
- DataManager: 数据加载
- ModelManager: 模型加载和推理
- EmbeddingComputer: 嵌入计算
- MetricsPlotter: 指标可视化
- GenerationPlotter: 生成对比可视化

Usage:
    python step3_multi_setting_visualization.py \
        --base_dir /path/to/OUTPUTs/SynthaticSCData \
        --output_dir ./visualizations
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# 导入新的模块化组件
from Analyser import (
    DataManager,
    ModelManager,
    EmbeddingComputer,
    MetricsPlotter,
    GenerationPlotter
)


class MultiSettingVisualizationPipeline:
    """
    多设置可视化管道（使用新的模块化组件）
    
    职责：
    - 协调各个模块化组件
    - 实现高层次的可视化工作流
    - 管理不同实验设置的对比
    """
    
    def __init__(
        self,
        base_dir: str,
        output_dir: str,
        device: str = 'cuda',
        random_seed: int = 42
    ):
        """
        初始化可视化管道
        
        Args:
            base_dir: 实验输出的基础目录
            output_dir: 可视化结果输出目录
            device: 计算设备
            random_seed: 随机种子
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.random_seed = random_seed
        
        # 初始化模块化组件
        self.data_manager = DataManager()
        self.model_manager = ModelManager(device=device)
        self.embedding_computer = EmbeddingComputer(random_seed=random_seed)
        self.metrics_plotter = MetricsPlotter()
        self.generation_plotter = GenerationPlotter()
        
        # 设置随机种子
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
    
    def _print_section(self, title: str, width: int = 80):
        """打印格式化的章节标题"""
        print("\n" + "="*width)
        print(title)
        print("="*width)
    
    def load_setting_data(
        self,
        setting_path: Path,
        model_names: List[str]
    ) -> Dict:
        """
        加载单个setting的数据
        
        Args:
            setting_path: Setting目录路径
            model_names: 要加载的模型名称列表
        
        Returns:
            包含metrics和generated data的字典
        """
        result = {
            'metrics': {},
            'generated': {}
        }
        
        # 加载metrics
        results_path = setting_path / 'results.json'
        if results_path.exists():
            all_metrics = self.data_manager.load_metrics_json(results_path)
            for model_name in model_names:
                if model_name in all_metrics:
                    result['metrics'][model_name] = all_metrics[model_name]
        
        # 加载generated data
        for model_name in model_names:
            pkl_path = setting_path / 'generated' / f'{model_name}.pkl'
            gen_data = self.data_manager.load_generated_pkl(pkl_path)
            if gen_data is not None:
                result['generated'][model_name] = gen_data
        
        return result
    
    def visualize_comparison(
        self,
        settings_dict: Dict[str, Path],
        model_names: List[str],
        title: str,
        save_prefix: str
    ):
        """
        可视化多个settings的对比
        
        Args:
            settings_dict: Setting名称到路径的映射
            model_names: 要对比的模型名称列表
            title: 图表标题
            save_prefix: 保存文件的前缀
        """
        self._print_section(f"Visualizing: {title}")
        
        # 收集所有metrics和generated data
        all_metrics = {}
        all_generated = {}
        
        for setting_name, setting_path in settings_dict.items():
            if not setting_path.exists():
                print(f"  ⚠️  {setting_name} not found: {setting_path}")
                continue
            
            data = self.load_setting_data(setting_path, model_names)
            
            # 合并metrics
            for model_name, metrics in data['metrics'].items():
                key = f"{setting_name}-{model_name}"
                all_metrics[key] = metrics
            
            # 合并generated data
            for model_name, gen_data in data['generated'].items():
                key = f"{setting_name}-{model_name}"
                all_generated[key] = gen_data
        
        if not all_metrics:
            print(f"  ⚠️  No data found for {title}")
            return
        
        # 绘制metrics对比
        print(f"\n  Plotting metrics comparison...")
        self.metrics_plotter.plot_metrics_comparison(
            all_metrics,
            title=f"{title} - Metrics Comparison",
            save_prefix=save_prefix,
            output_dir=self.output_dir
        )
        print(f"  ✓ Metrics saved: {save_prefix}.png/pdf/csv")
        
        # 绘制generation对比（如果有generated data）
        if all_generated:
            print(f"\n  Plotting generation comparison...")
            self._plot_generation_comparison(
                all_generated,
                title=f"{title} - Generation Comparison",
                save_prefix=save_prefix
            )
    
    def _plot_generation_comparison(
        self,
        generated_dict: Dict[str, Dict],
        title: str,
        save_prefix: str
    ):
        """
        绘制生成数据对比
        
        Args:
            generated_dict: 模型名称到生成数据的映射
            title: 图表标题
            save_prefix: 保存文件前缀
        """
        # 收集所有真实数据用于计算嵌入
        all_real_data = []
        all_real_labels = []
        time_labels = None
        
        for model_key, data in generated_dict.items():
            if data is not None and 'real_data' in data:
                all_real_data.append(data['real_data'])
                all_real_labels.append(data['real_labels'])
                if time_labels is None and 'time_labels' in data:
                    time_labels = data['time_labels']
        
        if not all_real_data:
            print("  ⚠️  No real data available for visualization")
            return
        
        # 合并真实数据
        all_real_data = np.vstack(all_real_data)
        all_real_labels = np.concatenate(all_real_labels)
        
        # 去重
        unique_indices = np.unique(all_real_data, axis=0, return_index=True)[1]
        all_real_data = all_real_data[unique_indices]
        all_real_labels = all_real_labels[unique_indices]
        
        # 计算嵌入
        print("    Computing embeddings...")
        embeddings_dict = self.embedding_computer.compute_all_embeddings(
            all_real_data, all_real_labels
        )
        
        # 为每个模型的生成数据计算嵌入
        for embedding_type in ['phate', 'lmnn_pca']:
            print(f"    Processing {embedding_type.upper()} embeddings...")
            
            model_embeddings = {'original': embeddings_dict[embedding_type]}
            
            for model_key, data in generated_dict.items():
                if data is not None and 'generated_data' in data:
                    gen_data = data['generated_data']
                    if gen_data is not None:
                        # Transform generated data
                        if embedding_type == 'phate':
                            gen_emb = self.embedding_computer.transform_phate(gen_data)
                        else:
                            gen_emb = self.embedding_computer.transform_lmnn_pca(gen_data)
                        model_embeddings[model_key] = gen_emb
            
            # 绘制对比图
            self.generation_plotter.plot_comparison_grid(
                model_embeddings,
                embeddings_dict[embedding_type],
                all_real_labels,
                time_labels or [str(i) for i in range(int(all_real_labels.max()) + 1)],
                embedding_type,
                title,
                save_prefix,
                self.output_dir
            )
            print(f"    ✓ {embedding_type.upper()} saved: {save_prefix}_{embedding_type}.png/pdf")
    
    def visualize_emt_process_comparison(self):
        """a) EMT过程建模对比：Setting1, Setting2, Setting3"""
        settings = {
            'Setting1': self.base_dir / 'EMT_Part1_Setting1',
            'Setting2': self.base_dir / 'EMT_Part1_Setting2',
            'Setting3': self.base_dir / 'EMT_Part1_Setting3'
        }
        
        model_names = ['sb_mlplus', 'ot', 'vae', 'batch_ot']
        
        self.visualize_comparison(
            settings,
            model_names,
            title="(a) EMT Process Modeling: Settings 1-3",
            save_prefix="a_emt_process"
        )
    
    def visualize_ablation_comparison(self):
        """b) 时间点消融：Setting2 vs Setting4"""
        settings = {
            'Setting2': self.base_dir / 'EMT_Part1_Setting2',
            'Setting4-Remove_8h': self.base_dir / 'EMT_Part1_Setting4' / 'experiment_EMT_Part1_setting4_ablation_remove_8h',
            'Setting4-Remove_1d': self.base_dir / 'EMT_Part1_Setting4' / 'experiment_EMT_Part1_setting4_ablation_remove_1d',
            'Setting4-Remove_3d': self.base_dir / 'EMT_Part1_Setting4' / 'experiment_EMT_Part1_setting4_ablation_remove_3d'
        }
        
        model_names = ['sb_mlplus']
        
        self.visualize_comparison(
            settings,
            model_names,
            title="(b) Timepoint Ablation: Setting2 vs Setting4",
            save_prefix="b_ablation"
        )
    
    def visualize_shuffle_comparison(self):
        """c) 时间点打乱：Setting2 vs Setting5"""
        settings = {
            'Setting2': self.base_dir / 'EMT_Part1_Setting2',
            'Setting5_Shuffled': self.base_dir / 'EMT_Part1_Setting5_Shuffled'
        }
        
        model_names = ['sb_mlplus']
        
        self.visualize_comparison(
            settings,
            model_names,
            title="(c) Timepoint Shuffle: Setting2 vs Setting5",
            save_prefix="c_shuffle"
        )
    
    def visualize_interpolation_comparison(self):
        """d) 线性插值：Setting2 vs Setting6"""
        settings = {
            'Setting2': self.base_dir / 'EMT_Part1_Setting2',
            'Setting6_Interpolated': self.base_dir / 'EMT_Part1_Setting6'
        }
        
        model_names = ['sb_mlplus', 'batch_ot']
        
        self.visualize_comparison(
            settings,
            model_names,
            title="(d) Linear Interpolation: Setting2 vs Setting6",
            save_prefix="d_interpolation"
        )
    
    def run_all_visualizations(self):
        """运行所有可视化"""
        self._print_section("Multi-Setting Visualization Pipeline")
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        
        # 运行各个对比
        self.visualize_emt_process_comparison()
        self.visualize_ablation_comparison()
        self.visualize_shuffle_comparison()
        self.visualize_interpolation_comparison()
        
        self._print_section("All Visualizations Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("\nGenerated files:")
        print("  - a_emt_process.png/pdf/csv")
        print("  - a_emt_process_phate.png/pdf")
        print("  - a_emt_process_lmnn_pca.png/pdf")
        print("  - b_ablation.png/pdf/csv")
        print("  - b_ablation_phate.png/pdf")
        print("  - b_ablation_lmnn_pca.png/pdf")
        print("  - c_shuffle.png/pdf/csv")
        print("  - c_shuffle_phate.png/pdf")
        print("  - c_shuffle_lmnn_pca.png/pdf")
        print("  - d_interpolation.png/pdf/csv")
        print("  - d_interpolation_phate.png/pdf")
        print("  - d_interpolation_lmnn_pca.png/pdf")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Multi-setting visualization using new modular components'
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        required=True,
        help='Base directory containing all experiment outputs'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for computation (cuda or cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # 创建并运行可视化管道
    pipeline = MultiSettingVisualizationPipeline(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        device=args.device,
        random_seed=args.seed
    )
    
    pipeline.run_all_visualizations()


if __name__ == '__main__':
    main()
