# 系统重构说明 / System Refactoring Summary

## 重构日期 / Refactoring Date
2024-11-06

## 重构原因 / Reason for Refactoring

原系统基于薛定谔桥理论，从分布采样生成合成数据。新需求是从真实单细胞数据通过调制插值生成连续时间轨迹数据，更适合实际应用场景。

The original system was based on Schrödinger Bridge theory, generating synthetic data from distribution sampling. The new requirement is to generate continuous time trajectory data from real single-cell data through modulation interpolation, which is more suitable for practical applications.

## 主要变更 / Major Changes

### 1. Data模块完全重写 / Complete Rewrite of Data Module

#### 删除的文件 / Deleted Files
- `Data/data_generator.py` (旧版本)
- `Data/dataset_builder.py` (旧版本)

#### 新增的文件 / New Files

**`Data/data_generator.py`**
- `ContinuousTimeDataGenerator`: 从真实数据生成连续时间数据
- `create_default_emt_generator()`: EMT数据集的默认配置
- 支持自定义时间间隔和颗粒度
- 调制函数: `(1-α)*X_t0 + α*X_t1 + noise`

**`Data/dataset_builder.py`**
- `ContinuousTimeDataset`: PyTorch数据集类
- `DatasetBuilder`: 支持三种采样策略
  - `all_time`: 全时间片随机采样
  - `specific_time`: 指定时间片采样
  - `clustered_time`: 聚类时间片采样
- `create_default_emt_dataset()`: 快速创建数据集

### 2. Analyser模块增强 / Analyser Module Enhancement

#### 新增的文件 / New Files

**`Analyser/entropy_metrics.py`**
- `estimate_entropy_knn()`: KNN方法估计熵
- `calculate_entropy_timeline()`: 计算时间线熵
- `plot_entropy_timeline()`: 绘制熵时间线
- `calculate_entropy_by_category()`: 按类别计算熵
- `analyze_continuous_data_quality()`: 完整质量分析

**`Analyser/real_data_metrics.py`** (重写)
- `calculate_frechet_distance()`: Frechet距离计算
- `calculate_mae()`: 平均绝对误差
- `calculate_pcc()`: Pearson相关系数
- `evaluate_generated_vs_test()`: 综合评估函数
- `evaluate_model_on_dataset()`: 数据集级别评估

### 3. Trainer模块更新 / Trainer Module Update

**`Trainer/evaluator.py`** (更新)
- 添加 `evaluate_real_data_metrics()` 方法
- 集成Frechet距离、MAE、PCC、Entropy评估
- 更新 `save_results()` 处理numpy类型

### 4. 新增示例和文档 / New Examples and Documentation

**`example_emt_workflow.py`**
- 完整的5步工作流示例
- 从数据生成到模型评估
- 支持命令行参数配置

**`README_CONTINUOUS_TIME.md`**
- 完整的系统文档（中英双语）
- 详细的API说明
- 使用示例和最佳实践

**`QUICKSTART.md`**
- 快速开始指南
- 分步教程
- 常见问题解答

## 核心功能对比 / Core Functionality Comparison

### 旧系统 / Old System
```python
# 从理论分布生成数据
distribution_params = {
    'dimension': 50,
    'init_variance': 1.0,
    'peak_params': {...},
    'terminal_params': {...}
}
trajectories = generate_from_distribution(distribution_params)
```

### 新系统 / New System
```python
# 从真实数据生成连续时间数据
generator = ContinuousTimeDataGenerator(
    adata_path="real_data.h5ad",
    n_hvg=100,
    time_labels=['0d', '8h', '1d', '3d', '7d'],
    time_intervals={('0d', '8h'): 8.0, ...}
)
adata_continuous = generator.generate_continuous_data()
```

## 数据格式变化 / Data Format Changes

### 输入数据 / Input Data

**旧系统**: 理论分布参数
```python
{
    'mu_0': np.array([...]),
    'sigma_0': np.array([[...]]),
    'mu_T': np.array([...]),
    'sigma_T': np.array([[...]])
}
```

**新系统**: 真实AnnData对象
```python
AnnData object with n_obs × n_vars = 53290 × 16906
    obs: 'Ground_truth', 'CellLine', 'stimulus', ...
    var: gene names
    X: expression matrix
```

### 输出数据 / Output Data

**旧系统**: 轨迹张量
```python
trajectories: (n_trajectories, n_timepoints, dimension)
time_stamps: (n_timepoints,)
```

**新系统**: 增强的AnnData对象
```python
AnnData object with n_obs × n_vars = N_cells × n_hvg
    obs:
        - original_label: 原始时间标签
        - is_real: 真实/生成标识
        - continuous_time: 连续时间值
        - time_slice: 时间片类别
    var: HVG gene names
    X: expression matrix
```

## 评估指标变化 / Evaluation Metrics Changes

### 旧系统指标 / Old System Metrics
- Boundary Fidelity (边界保真度)
- Path Fidelity (路径保真度)
- Entropy Evolution (熵演化)
- Geometric Structure (几何结构)
- Generalization (泛化能力)

### 新系统指标 / New System Metrics
- **Frechet Distance** (Frechet距离): 分布差异
- **MAE** (平均绝对误差): 点对点误差
- **PCC** (Pearson相关系数): 特征相关性
- **Entropy** (熵): 数据多样性
- 保留原有的泛化能力评估

## API变化 / API Changes

### 数据生成 / Data Generation

**旧API**:
```python
from Data import DistributionParameterizer, TrajectoryGenerator
params = DistributionParameterizer(dimension=50, ...)
trajectories = TrajectoryGenerator(params).generate()
```

**新API**:
```python
from Data import create_default_emt_generator
generator = create_default_emt_generator(output_path="data.h5ad")
adata = generator.save_continuous_data("data.h5ad")
```

### 数据集构建 / Dataset Building

**旧API**:
```python
from Data import DatasetConstructor
dataset = DatasetConstructor(trajectories, split_ratio=0.8)
train_data, test_data = dataset.split()
```

**新API**:
```python
from Data import create_default_emt_dataset
train_loader, test_loader, stats = create_default_emt_dataset(
    continuous_data_path="data.h5ad",
    sampling_strategy='all_time',
    train_ratio=0.8
)
```

### 模型评估 / Model Evaluation

**旧API**:
```python
from Trainer import ModelEvaluator
evaluator = ModelEvaluator(model, 'vae')
results = evaluator.evaluate(test_dataset)
```

**新API**:
```python
from Analyser.real_data_metrics import evaluate_model_on_dataset
results = evaluate_model_on_dataset(
    adata_continuous=adata,
    train_indices=train_idx,
    test_indices=test_idx,
    verbose=True
)
```

## 向后兼容性 / Backward Compatibility

⚠️ **不兼容 / Not Compatible**: 新系统与旧系统不兼容

### 迁移指南 / Migration Guide

如果需要使用旧的薛定谔桥方法，请参考 `SystemDesign.md`。新系统专注于真实数据的连续时间生成。

If you need to use the old Schrödinger Bridge method, please refer to `SystemDesign.md`. The new system focuses on continuous time generation from real data.

### 保留的模块 / Preserved Modules

以下模块保持兼容：
- `Model/vae_model.py` (更新了维度参数说明)
- `Model/ot_model.py` (保持不变)
- `Model/sb_model.py` (保持不变)
- `Trainer/trainer.py` (保持不变)

## 性能改进 / Performance Improvements

1. **内存效率**: 使用HVG降维，减少内存占用
2. **计算效率**: 调制插值比求解微分方程快
3. **可扩展性**: 支持大规模单细胞数据集

## 使用建议 / Usage Recommendations

### 推荐配置 / Recommended Configuration

**小规模实验** (快速测试):
```python
n_hvg = 50
cells_per_label = 500
time_granularity = 2.0
```

**中等规模** (标准使用):
```python
n_hvg = 100
cells_per_label = 2000
time_granularity = 1.0
```

**大规模** (完整分析):
```python
n_hvg = 200
cells_per_label = 5000
time_granularity = 0.5
```

### 参数调优建议 / Parameter Tuning Recommendations

1. **HVG数量** (`n_hvg`):
   - 太少 (< 50): 信息损失
   - 适中 (50-200): 平衡性能和信息
   - 太多 (> 500): 计算慢，可能过拟合

2. **时间颗粒度** (`time_granularity`):
   - 粗 (> 2.0h): 数据量小，可能丢失动态
   - 适中 (0.5-2.0h): 平衡
   - 细 (< 0.5h): 数据量大，计算慢

3. **噪声水平** (`noise_std`):
   - 低 (< 0.05): 过于平滑
   - 适中 (0.05-0.2): 自然变化
   - 高 (> 0.2): 可能破坏生物学意义

## 测试状态 / Testing Status

✅ 数据生成模块测试通过
✅ 数据集构建模块测试通过
✅ 熵计算模块测试通过
✅ 评估指标模块测试通过
✅ 完整工作流测试通过

## 已知问题 / Known Issues

1. 大数据集 (> 100k cells) 可能需要较长时间生成
2. Frechet距离在高维空间可能不稳定（已通过HVG缓解）
3. 内存峰值出现在数据生成阶段

## 未来改进 / Future Improvements

1. 支持GPU加速的数据生成
2. 增加更多插值方法（样条、贝塞尔曲线等）
3. 支持批量处理多个数据集
4. 添加交互式可视化界面

## 联系方式 / Contact

如有问题或建议，请参考项目文档或提交Issue。

For questions or suggestions, please refer to the project documentation or submit an Issue.
