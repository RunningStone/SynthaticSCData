# Analyser Module

## 概述 / Overview

该模块用于可视化不同模型的生成质量，通过以下方式展示模型学习到的时间序列演化路径：

This module visualizes the generation quality of different models by demonstrating the learned temporal evolution paths through:

1. **数据加载与采样** / **Data Loading and Sampling**: 使用与训练相同的biology_split方法从测试集采样不同时间点的样本 / Sample different timepoints from test set using the same biology_split method as training
2. **模型生成** / **Model Generation**: 加载训练好的模型，从第一个时间点生成最后一个时间点的样本 / Load trained models and generate target timepoint samples from source timepoint
3. **降维嵌入** / **Dimensionality Reduction**: 在原始数据上计算PHATE和LMNN+PCA嵌入，并投影生成数据 / Compute PHATE and LMNN+PCA embeddings on original data and project generated samples
4. **可视化对比** / **Visualization Comparison**: 创建6个子图展示原始数据和各模型生成数据的分布 / Create 6 subplots showing distributions of original and generated data

## 安装依赖 / Installation

```bash
# 激活虚拟环境 / Activate virtual environment
source .venv/bin/activate

# 安装可视化依赖 / Install visualization dependencies
pip install phate metric-learn
```

## 使用方法 / Usage

### 方法1：使用Shell脚本 / Method 1: Using Shell Script

```bash
# 运行可视化 / Run visualization
bash run_visualization.sh
```

### 方法2：使用Python脚本 / Method 2: Using Python Script

```bash
python run_visualization.py \
    --n_hvg 500 \
    --n_samples_per_timepoint 500 \
    --n_generate_per_model 500 \
    --output_base_dir /path/to/trained/models \
    --visualization_output_dir /path/to/output \
    --device cuda
```

### 方法3：在代码中使用 / Method 3: Using in Code

```python
from Analyser import GenerationVisualizer

# 初始化可视化器 / Initialize visualizer
visualizer = GenerationVisualizer(
    file_path=None,  # None表示使用默认EMT数据集 / None for default EMT dataset
    n_hvg=500,
    output_dir='./visualization_outputs',
    device='cuda',
    random_seed=42
)

# 定义模型配置 / Define model configurations
model_configs = {
    'SB_S1': {
        'type': 'sb',
        'checkpoint_path': './outputs/setting1/sb_model/best_model.pt',
        'model_kwargs': {
            'dimension': 500,
            'hidden_dims': [512, 512, 512, 512],
            'time_embedding_dim': 64,
            'dropout': 0.1,
            'diffusion_coeff': 0.1
        }
    },
    'OT_S1': {
        'type': 'ot',
        'checkpoint_path': './outputs/setting1/ot_model/best_model.pt',
        'model_kwargs': {
            'dimension': 500,
            'hidden_dims': [512, 512, 512, 512],
            'activation': 'relu',
            'dropout': 0.1,
            'use_residual': True
        }
    },
    'VAE_S1': {
        'type': 'vae',
        'checkpoint_path': './outputs/setting1/vae_model/best_model.pt',
        'model_kwargs': {
            'dimension': 500,
            'latent_dim': 128,
            'hidden_dims': [512, 256],
            'activation': 'relu',
            'dropout': 0.1,
            'beta': 1.0
        }
    },
    'SB_MLPlus_S2': {
        'type': 'sb_mlplus',
        'checkpoint_path': './outputs/setting2/sb_mlplus_model/best_model.pt',
        'model_kwargs': {
            'dimension': 500,
            'hidden_dim': 512,
            'n_blocks': 4,
            'time_embedding_dim': 64,
            'n_time_frequencies': 10,
            'dropout': 0.1,
            'diffusion_coeff': 0.1
        }
    }
}

# 运行完整流程 / Run full pipeline
visualizer.run_full_pipeline(
    model_configs=model_configs,
    n_samples_per_timepoint=500,
    n_generate_per_model=500
)
```

## 输出文件 / Output Files

可视化结果将保存在指定的输出目录中：

Visualization results will be saved in the specified output directory:

- `generation_comparison_phate.png`: PHATE降维可视化 / PHATE dimensionality reduction visualization
- `generation_comparison_phate.pdf`: PHATE可视化PDF版本 / PHATE visualization PDF version
- `generation_comparison_lmnn_pca.png`: LMNN+PCA降维可视化 / LMNN+PCA dimensionality reduction visualization
- `generation_comparison_lmnn_pca.pdf`: LMNN+PCA可视化PDF版本 / LMNN+PCA visualization PDF version

## 可视化布局 / Visualization Layout

每个可视化图包含6个子图：

Each visualization contains 6 subplots:

1. **子图1** / **Subplot 1**: 所有原始数据点，不同时间点用不同颜色标注 / All original data points with different colors for different timepoints
2. **子图2-5** / **Subplots 2-5**: 原始数据（较暗颜色）+ 单个模型生成数据（较亮颜色，星形标记） / Original data (darker colors) + single model generated data (brighter color, star markers)
3. **子图6** / **Subplot 6**: 最后时间点原始数据（灰色）+ 所有模型生成数据（不同颜色，星形标记） / Last timepoint original data (gray) + all models generated data (different colors, star markers)

## 参数说明 / Parameters

- `file_path`: h5ad文件路径（None表示使用默认EMT数据集） / Path to h5ad file (None for default EMT dataset)
- `n_hvg`: 高变基因数量 / Number of highly variable genes
- `n_samples_per_timepoint`: 每个时间点从测试集采样的细胞数 / Number of cells to sample per timepoint from test set
- `n_generate_per_model`: 每个模型生成的样本数 / Number of samples to generate per model
- `output_base_dir`: 训练模型的基础目录 / Base directory containing trained models
- `visualization_output_dir`: 可视化结果输出目录 / Output directory for visualization results
- `device`: 推理设备（'cuda'或'cpu'） / Device for inference ('cuda' or 'cpu')
- `random_seed`: 随机种子 / Random seed

## 技术细节 / Technical Details

### PHATE降维 / PHATE Dimensionality Reduction

PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding) 是一种保持数据流形结构的降维方法，特别适合可视化生物学时间序列数据。

PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding) is a dimensionality reduction method that preserves the manifold structure of data, particularly suitable for visualizing biological time series data.

参数设置 / Parameters:
- `n_components=2`: 降至2维用于可视化 / Reduce to 2D for visualization
- `knn=5`: K近邻数量 / Number of k-nearest neighbors
- `decay=40`: 热扩散衰减参数 / Heat diffusion decay parameter

### LMNN+PCA降维 / LMNN+PCA Dimensionality Reduction

LMNN (Large Margin Nearest Neighbor) 是一种度量学习方法，学习一个距离度量使得同类样本更接近。结合PCA进行降维。

LMNN (Large Margin Nearest Neighbor) is a metric learning method that learns a distance metric to bring samples of the same class closer together. Combined with PCA for dimensionality reduction.

流程 / Pipeline:
1. 标准化数据 / Standardize data
2. LMNN度量学习（使用时间标签） / LMNN metric learning (using time labels)
3. PCA降至2维 / PCA reduction to 2D

参数设置 / Parameters:
- `n_components=50`: LMNN输出维度 / LMNN output dimensions
- `k=5`: K近邻数量 / Number of k-nearest neighbors
- `learn_rate=1e-6`: 学习率 / Learning rate
- `max_iter=100`: 最大迭代次数 / Maximum iterations

## 注意事项 / Notes

1. **数据切分一致性** / **Data Split Consistency**: 可视化使用与训练相同的biology_split方法，确保测试集的独立性 / Visualization uses the same biology_split method as training to ensure test set independence

2. **生成方法** / **Generation Method**: 所有模型使用`generate_trajectory`方法，从第一个时间点生成到最后一个时间点 / All models use `generate_trajectory` method to generate from first to last timepoint

3. **降维投影** / **Dimensionality Reduction Projection**: 降维变换在原始数据上拟合，然后应用到生成数据上，确保空间一致性 / Dimensionality reduction transformations are fitted on original data and then applied to generated data to ensure spatial consistency

4. **颜色编码** / **Color Encoding**: 
   - 原始数据使用较暗的颜色 / Original data uses darker colors
   - 生成数据使用较亮的颜色和星形标记 / Generated data uses brighter colors and star markers
   - 不同模型使用不同颜色便于区分 / Different models use different colors for easy distinction

## 故障排除 / Troubleshooting

### 问题1：找不到模型检查点 / Issue 1: Model checkpoint not found

**解决方案** / **Solution**: 确保`output_base_dir`参数指向正确的训练输出目录，并且模型已经训练完成 / Ensure `output_base_dir` parameter points to the correct training output directory and models are trained

### 问题2：内存不足 / Issue 2: Out of memory

**解决方案** / **Solution**: 减少`n_samples_per_timepoint`和`n_generate_per_model`参数 / Reduce `n_samples_per_timepoint` and `n_generate_per_model` parameters

### 问题3：LMNN训练时间过长 / Issue 3: LMNN training takes too long

**解决方案** / **Solution**: 
- 减少`max_iter`参数 / Reduce `max_iter` parameter
- 减少样本数量 / Reduce number of samples
- 只使用PHATE可视化 / Use only PHATE visualization

## 引用 / Citation

如果使用本可视化工具，请引用相关方法：

If you use this visualization tool, please cite the relevant methods:

- **PHATE**: Moon, K.R., van Dijk, D., Wang, Z. et al. Visualizing structure and transitions in high-dimensional biological data. Nat Biotechnol 37, 1482–1492 (2019).
- **LMNN**: Weinberger, K. Q., & Saul, L. K. (2009). Distance metric learning for large margin nearest neighbor classification. Journal of Machine Learning Research, 10(2).
