#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Setting Visualization Script

基于已保存的generated pkl文件进行可视化对比

Usage:
    python step2_multi_setting_visualization.py \
        --base_dir /path/to/OUTPUTs/SynthaticSCData \
        --output_dir ./visualizations
"""

import argparse
import sys
import pickle
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 可视化依赖
import phate
from metric_learn import LMNN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class VisualizationManager:
    """管理多个setting的可视化对比"""
    
    def __init__(self, base_dir: str, output_dir: str):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_generated_data(self, setting_path: Path, model_name: str) -> Dict:
        """加载生成的数据"""
        pkl_path = setting_path / 'generated' / f'{model_name}.pkl'
        if not pkl_path.exists():
            return None
        
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    def load_metrics(self, setting_path: Path) -> Dict:
        """加载评估指标"""
        results_path = setting_path / 'results.json'
        if not results_path.exists():
            return {}
        
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict], 
                               title: str, save_prefix: str):
        """绘制评估指标对比图"""
        metric_names = [
            'test_loss', 'frechet_distance', 'mae', 'pcc',
            'wasserstein_distance', 'mmd', 'js_divergence',
            'correlation_structure_corr', 'r2_mean', 'correlation_frobenius_diff'
        ]
        
        metric_titles = [
            'Test Loss', 'Fréchet Distance', 'MAE', 'PCC',
            'Wasserstein Distance', 'MMD', 'JS Divergence',
            'Correlation Structure', 'R² Mean', 'Correlation Frobenius Diff'
        ]
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, (metric_name, metric_title) in enumerate(zip(metric_names, metric_titles)):
            if idx >= len(axes):
                break
            
            ax = axes[idx]
            
            # 提取数据
            labels = []
            values = []
            
            for model_key, metrics in metrics_dict.items():
                if 'evaluation' in metrics:
                    metrics = metrics['evaluation']
                
                if metric_name in metrics:
                    labels.append(model_key)
                    values.append(metrics[metric_name])
            
            if values:
                bars = ax.bar(range(len(values)), values)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel(metric_title)
                ax.set_title(metric_title, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                # 标注数值
                for i, v in enumerate(values):
                    ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(metric_title, fontweight='bold')
        
        # 隐藏多余的子图
        for idx in range(len(metric_names), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # 保存
        png_path = self.output_dir / f'{save_prefix}_metrics.png'
        pdf_path = self.output_dir / f'{save_prefix}_metrics.pdf'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Metrics comparison saved: {png_path}")
        
        # 保存CSV
        csv_data = []
        for model_key, metrics in metrics_dict.items():
            if 'evaluation' in metrics:
                metrics = metrics['evaluation']
            row = {'model': model_key}
            for metric_name in metric_names:
                row[metric_name] = metrics.get(metric_name, np.nan)
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / f'{save_prefix}_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Metrics CSV saved: {csv_path}")
    
    def compute_embeddings(self, all_data: np.ndarray, all_labels: np.ndarray) -> Tuple:
        """计算PHATE和LMNN+PCA嵌入"""
        print("  Computing embeddings...")
        
        # PHATE
        phate_op = phate.PHATE(n_components=2, random_state=42, n_jobs=-1)
        phate_embedding = phate_op.fit_transform(all_data)
        
        # LMNN + PCA
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(all_data)
        
        lmnn = LMNN(n_components=min(50, all_data.shape[1]), random_state=42)
        data_lmnn = lmnn.fit_transform(data_scaled, all_labels)
        
        pca = PCA(n_components=2, random_state=42)
        lmnn_pca_embedding = pca.fit_transform(data_lmnn)
        
        return phate_embedding, lmnn_pca_embedding
    
    def plot_generation_comparison(self, data_dict: Dict[str, Dict],
                                  title: str, save_prefix: str):
        """绘制生成数据对比可视化"""
        print(f"\n  Plotting generation comparison: {title}")
        
        # 收集所有真实数据用于计算嵌入
        all_real_data = []
        all_real_labels = []
        
        for model_key, data in data_dict.items():
            if data is not None and 'real_data' in data:
                all_real_data.append(data['real_data'])
                all_real_labels.append(data['real_labels'])
        
        if not all_real_data:
            print("  ⚠️  No data available for visualization")
            return
        
        # 合并真实数据（去重）
        all_real_data = np.vstack(all_real_data)
        all_real_labels = np.concatenate(all_real_labels)
        
        # 去重
        unique_indices = np.unique(all_real_data, axis=0, return_index=True)[1]
        all_real_data = all_real_data[unique_indices]
        all_real_labels = all_real_labels[unique_indices]
        
        # 计算嵌入
        phate_emb, lmnn_pca_emb = self.compute_embeddings(all_real_data, all_real_labels)
        
        # 为每个模型绘制可视化
        for method_name, embedding in [('PHATE', phate_emb), ('LMNN-PCA', lmnn_pca_emb)]:
            n_models = len(data_dict)
            n_cols = min(3, n_models + 1)
            n_rows = (n_models + 1 + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            axes = axes.flatten()
            
            # 第一个子图：所有真实数据
            ax = axes[0]
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                               c=all_real_labels, cmap='viridis', 
                               s=10, alpha=0.6)
            ax.set_title('Real Data (All)', fontweight='bold', fontsize=12)
            ax.set_xlabel(f'{method_name} 1')
            ax.set_ylabel(f'{method_name} 2')
            plt.colorbar(scatter, ax=ax, label='Time')
            
            # 为每个模型绘制真实+生成数据
            for idx, (model_key, data) in enumerate(data_dict.items()):
                ax = axes[idx + 1]
                
                if data is None or 'generated_data' is None:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(model_key, fontweight='bold')
                    continue
                
                # 绘制真实数据
                ax.scatter(embedding[:, 0], embedding[:, 1],
                          c='lightgray', s=10, alpha=0.3, label='Real')
                
                # 生成数据需要转换到相同的嵌入空间
                if data['generated_data'] is not None:
                    gen_data = data['generated_data']
                    
                    # 将生成数据添加到总数据中进行转换
                    combined_data = np.vstack([all_real_data, gen_data])
                    
                    if method_name == 'PHATE':
                        phate_op_temp = phate.PHATE(n_components=2, random_state=42, n_jobs=-1)
                        combined_emb = phate_op_temp.fit_transform(combined_data)
                        gen_emb = combined_emb[-len(gen_data):]
                    else:
                        scaler_temp = StandardScaler()
                        combined_scaled = scaler_temp.fit_transform(combined_data)
                        lmnn_temp = LMNN(n_components=min(50, combined_data.shape[1]), random_state=42)
                        combined_lmnn = lmnn_temp.fit_transform(combined_scaled, 
                                                               np.concatenate([all_real_labels, 
                                                                             np.zeros(len(gen_data))]))
                        pca_temp = PCA(n_components=2, random_state=42)
                        combined_pca = pca_temp.fit_transform(combined_lmnn)
                        gen_emb = combined_pca[-len(gen_data):]
                    
                    ax.scatter(gen_emb[:, 0], gen_emb[:, 1],
                              c='red', s=20, alpha=0.6, label='Generated')
                
                ax.set_title(model_key, fontweight='bold', fontsize=10)
                ax.set_xlabel(f'{method_name} 1')
                ax.set_ylabel(f'{method_name} 2')
                ax.legend()
            
            # 隐藏多余的子图
            for idx in range(len(data_dict) + 1, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f'{title} - {method_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # 保存
            method_suffix = method_name.lower().replace('-', '_').replace('+', '_')
            png_path = self.output_dir / f'{save_prefix}_{method_suffix}.png'
            pdf_path = self.output_dir / f'{save_prefix}_{method_suffix}.pdf'
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_path, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ {method_name} visualization saved: {png_path}")
    
    def visualize_emt_process_comparison(self):
        """a) EMT过程建模对比：Setting1, Setting2, Setting3"""
        print("\n" + "="*80)
        print("(a) EMT Process Modeling Comparison")
        print("="*80)
        
        settings = {
            'Setting1': 'EMT_Part1_Setting1',
            'Setting2': 'EMT_Part1_Setting2',
            'Setting3': 'EMT_Part1_Setting3'
        }
        
        # 收集指标
        metrics_dict = {}
        data_dict = {}
        
        for setting_name, folder_name in settings.items():
            setting_path = self.base_dir / folder_name
            if not setting_path.exists():
                print(f"  ⚠️  {setting_name} not found: {setting_path}")
                continue
            
            metrics = self.load_metrics(setting_path)
            
            for model_name in metrics.keys():
                key = f"{setting_name}-{model_name}"
                metrics_dict[key] = metrics[model_name]
                
                # 加载生成数据
                gen_data = self.load_generated_data(setting_path, model_name)
                data_dict[key] = gen_data
        
        # 绘制指标对比
        self.plot_metrics_comparison(
            metrics_dict,
            "EMT Process Modeling: Settings 1-3 Comparison",
            "a_emt_process"
        )
        
        # 绘制生成数据可视化
        self.plot_generation_comparison(
            data_dict,
            "EMT Process Modeling: Generation Comparison",
            "a_emt_process"
        )
    
    def visualize_ablation_comparison(self):
        """b) 时间点消融：Setting2 vs Setting4"""
        print("\n" + "="*80)
        print("(b) Timepoint Ablation Comparison")
        print("="*80)
        
        # Setting2的SB_MLPlus
        setting2_path = self.base_dir / 'EMT_Part1_Setting2'
        
        # Setting4的三个消融实验
        setting4_paths = {
            'Remove_8h': self.base_dir / 'EMT_Part1_Setting4' / 'experiment_EMT_Part1_setting4_ablation_remove_8h',
            'Remove_1d': self.base_dir / 'EMT_Part1_Setting4' / 'experiment_EMT_Part1_setting4_ablation_remove_1d',
            'Remove_3d': self.base_dir / 'EMT_Part1_Setting4' / 'experiment_EMT_Part1_setting4_ablation_remove_3d'
        }
        
        metrics_dict = {}
        data_dict = {}
        
        # Setting2
        if setting2_path.exists():
            metrics = self.load_metrics(setting2_path)
            if 'sb_mlplus' in metrics:
                metrics_dict['Setting2-sb_mlplus'] = metrics['sb_mlplus']
                data_dict['Setting2-sb_mlplus'] = self.load_generated_data(setting2_path, 'sb_mlplus')
        
        # Setting4消融
        for ablation_name, ablation_path in setting4_paths.items():
            if ablation_path.exists():
                metrics = self.load_metrics(ablation_path)
                if 'sb_mlplus' in metrics:
                    key = f"Setting4-{ablation_name}"
                    metrics_dict[key] = metrics['sb_mlplus']
                    data_dict[key] = self.load_generated_data(ablation_path, 'sb_mlplus')
        
        # 绘制
        self.plot_metrics_comparison(
            metrics_dict,
            "Timepoint Ablation: Setting2 vs Setting4",
            "b_ablation"
        )
        
        self.plot_generation_comparison(
            data_dict,
            "Timepoint Ablation: Generation Comparison",
            "b_ablation"
        )
    
    def visualize_shuffle_comparison(self):
        """c) 时间点打乱：Setting2 vs Setting5"""
        print("\n" + "="*80)
        print("(c) Timepoint Shuffle Comparison")
        print("="*80)
        
        settings = {
            'Setting2': self.base_dir / 'EMT_Part1_Setting2',
            'Setting5_Shuffled': self.base_dir / 'EMT_Part1_Setting5_Shuffled'
        }
        
        metrics_dict = {}
        data_dict = {}
        
        for setting_name, setting_path in settings.items():
            if setting_path.exists():
                metrics = self.load_metrics(setting_path)
                if 'sb_mlplus' in metrics:
                    key = f"{setting_name}-sb_mlplus"
                    metrics_dict[key] = metrics['sb_mlplus']
                    data_dict[key] = self.load_generated_data(setting_path, 'sb_mlplus')
        
        # 绘制
        self.plot_metrics_comparison(
            metrics_dict,
            "Timepoint Shuffle: Setting2 vs Setting5",
            "c_shuffle"
        )
        
        self.plot_generation_comparison(
            data_dict,
            "Timepoint Shuffle: Generation Comparison",
            "c_shuffle"
        )
    
    def visualize_interpolation_comparison(self):
        """d) 线性插值：Setting2 vs Setting6"""
        print("\n" + "="*80)
        print("(d) Linear Interpolation Comparison")
        print("="*80)
        
        settings = {
            'Setting2': self.base_dir / 'EMT_Part1_Setting2',
            'Setting6_Interpolated': self.base_dir / 'EMT_Part1_Setting6'
        }
        
        models = ['sb_mlplus', 'batch_ot']
        
        metrics_dict = {}
        data_dict = {}
        
        for setting_name, setting_path in settings.items():
            if setting_path.exists():
                metrics = self.load_metrics(setting_path)
                for model_name in models:
                    if model_name in metrics:
                        key = f"{setting_name}-{model_name}"
                        metrics_dict[key] = metrics[model_name]
                        data_dict[key] = self.load_generated_data(setting_path, model_name)
        
        # 绘制
        self.plot_metrics_comparison(
            metrics_dict,
            "Linear Interpolation: Setting2 vs Setting6",
            "d_interpolation"
        )
        
        self.plot_generation_comparison(
            data_dict,
            "Linear Interpolation: Generation Comparison",
            "d_interpolation"
        )
    
    def run_all_visualizations(self):
        """运行所有可视化"""
        print("\n" + "="*80)
        print("Multi-Setting Visualization Pipeline")
        print("="*80)
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        
        self.visualize_emt_process_comparison()
        self.visualize_ablation_comparison()
        self.visualize_shuffle_comparison()
        self.visualize_interpolation_comparison()
        
        print("\n" + "="*80)
        print("All Visualizations Complete!")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-setting visualization based on saved generated data'
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
    
    args = parser.parse_args()
    
    manager = VisualizationManager(
        base_dir=args.base_dir,
        output_dir=args.output_dir
    )
    
    manager.run_all_visualizations()


if __name__ == '__main__':
    main()
