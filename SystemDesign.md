# 真实数据细胞状态转换学习系统

## 一、核心思想

**研究问题**：学习完整动力学轨迹是否优于仅学习起止点分布？

**实验设计**：基于真实时间序列数据，对比不同时间点信息对模型泛化能力的影响

**核心对比**：
- **Setting 1**：仅使用起止两个时间点（如 0d → 7d）
- **Setting 2**：使用所有中间时间点（如 0d → 8h → 1d → 3d → 7d）

**关键假设**：在样本数量相近的情况下，Setting 2（完整轨迹）的泛化能力优于 Setting 1（仅边界）

**模型选择**：使用 Schrödinger Bridge (SB) 模型学习时间依赖的漂移场 $\mathbf{b}(x,t)$

**数学框架**：$\min \int_0^T \text{KL}(\mu_t \| \gamma_t) dt$，学习完整路径约束

## 二、模块设计与数据流

### 2.1 真实数据加载与预处理（`Data/data_loader.py`）

**输入配置**：
```python
data_config = {
    "file_path": "path/to/adata.h5ad",
    "n_hvg": 100,  # 使用高变基因数量
    "obs_time_column": "Ground_truth",  # 时间标签列名
    "time_labels": ['0d', '8h', '1d', '3d', '7d'],  # 所有时间点
    "time_label_order": ['0d', '8h', '1d', '3d', '7d'],  # 时间顺序
    "biology_split": {
        "train_val_column": "batches",  # 或 "random" 表示随机切分
        "train": ["Mix1", "Mix2", "Mix3"],  # 训练集batch
        "test": ["Mix4"]  # 测试集batch
    }
}
```

**数据流**：
```
原始 h5ad 文件 → 读取并分析 obs 列 → HVG 筛选 → 时间点过滤
→ 生物学划分验证（检查训练/测试集是否包含所有类别）
→ Setting 1: 采样起止点 → Setting 2: 采样所有时间点
→ 样本数量平衡（确保两个 setting 样本数相近）
→ 创建 PyTorch DataLoader
```

**关键函数**：
- `load_and_analyze_data()`: 加载数据并输出 obs 列统计信息
- `validate_biology_split()`: 验证训练/测试集是否包含所有时间点类别
- `create_setting1_dataset()`: 创建仅起止点的数据集
- `create_setting2_dataset()`: 创建包含所有时间点的数据集
- `balance_sample_sizes()`: 平衡两个 setting 的样本数量

### 2.2 数据集构建（`Data/dataset_builder.py`）

**两种 Setting 的数据采样**：

**Setting 1（仅边界）**：
- 仅使用 `time_label_order` 的首尾时间点（如 0d 和 7d）
- 从训练集 batch 中采样这两个时间点的细胞
- 从测试集 batch 中采样这两个时间点的细胞

**Setting 2（完整轨迹）**：
- 使用 `time_label_order` 中的所有时间点
- 从训练集 batch 中采样所有时间点的细胞
- 从测试集 batch 中采样所有时间点的细胞
- 调整每个时间点的采样数量，使总样本数与 Setting 1 相近

**样本数量控制**：
- 假设 Setting 1 每个时间点采样 N 个细胞，总计 2N 个样本
- Setting 2 有 M 个时间点，每个时间点采样约 2N/M 个细胞
- 确保两个 setting 的训练集和测试集大小相近

**配置**：batch=256，标准化，训练集 shuffle

### 2.3 Schrödinger Bridge 模型（`Model/sb_model.py`）

**模型架构**：
- **势函数网络**：$\varphi(x,t), \psi(x,t)$，各 4 层 [512, 512, 512, 512]
- **时间编码**：Sinusoidal embedding + 可学习变换（64 维）
- **漂移场**：$\mathbf{b}(x,t) = -D\nabla[\varphi + \psi]$，通过自动微分计算梯度
- **扩散系数**：$D = 0.1$

**训练过程**：
- **输入**：连续时间点对 $(x_t, x_{t+\Delta t}, t, \Delta t)$
- **损失函数**：$\mathcal{L}_{SB} = \|\mathbf{b}(x_t, t) - (x_{t+\Delta t} - x_t)/\Delta t\|^2$
- **优化器**：Adam (lr=5e-4)，batch=256
- **训练策略**：
  - Setting 1: 仅使用起止点对进行训练
  - Setting 2: 使用所有连续时间点对进行训练

**推理过程**：
- 求解 SDE：$d\mathbf{x}_t = \mathbf{b}(\mathbf{x}_t, t)dt + \sqrt{2D}d\mathbf{W}_t$
- 方法：Euler-Maruyama 或确定性积分
- 从起始状态 $x_0$ 生成到目标时间 $T$ 的轨迹

**特点**：学习时间依赖的完整路径动力学，支持任意时间点预测

### 2.4 训练器（`Trainer/trainer.py`）

**统一训练接口**：
```python
trainer = SBTrainer(
    model=sb_model,
    train_loader=train_loader,
    test_loader=test_loader,
    device='cuda'
)
trainer.train(epochs=100, early_stopping_patience=10)
```

**训练配置**：
- 优化器：Adam (lr=5e-4)
- 批次大小：256
- 训练轮数：100（带早停机制）
- 梯度裁剪：max_norm=1.0

**输出**：
- 模型检查点：`best_model.pt`
- 训练曲线：loss vs epoch
- 验证指标：每 5 epoch 评估一次

### 2.5 评估（`Analyser/evaluator.py`）

**评估指标**：

1. **Frechet Distance (FD)**：
   - $FD = \|\mu_1 - \mu_2\|^2 + \text{tr}(\Sigma_1 + \Sigma_2 - 2(\Sigma_1\Sigma_2)^{1/2})$
   - 衡量生成分布与真实分布的差异

2. **Mean Absolute Error (MAE)**：
   - 逐细胞、逐基因的平均绝对误差

3. **Pearson Correlation Coefficient (PCC)**：
   - 基因表达模式的相关性

4. **时间点特异性评估**：
   - 对每个时间点分别计算 FD、MAE、PCC
   - 重点关注测试集时间点的泛化能力

**对比分析**：
- Setting 1 vs Setting 2 在测试集上的性能
- 不同时间点的预测准确度
- 中间时间点信息对泛化的贡献

## 三、实验流程（`run_experiment.py`）

**完整实验流程**：

### Step 1: 数据加载与分析
```python
# 加载真实数据
data_loader = RealDataLoader(
    file_path="path/to/adata.h5ad",
    n_hvg=100,
    obs_time_column='Ground_truth',
    time_labels=['0d', '8h', '1d', '3d', '7d'],
    time_label_order=['0d', '8h', '1d', '3d', '7d'],
    biology_split={
        "train_val_column": "batches",
        "train": ["Mix1", "Mix2", "Mix3"],
        "test": ["Mix4"]
    }
)

# 输出数据统计信息和 obs 列分析
data_loader.load_and_analyze()
data_loader.validate_biology_split()  # 验证训练/测试集包含所有类别
```

### Step 2: 创建两种 Setting 的数据集
```python
# Setting 1: 仅起止点
setting1_train, setting1_test = data_loader.create_setting1_dataset(
    cells_per_timepoint=2000
)

# Setting 2: 所有时间点
setting2_train, setting2_test = data_loader.create_setting2_dataset(
    total_cells=4000  # 与 Setting 1 总数相近
)

# 创建 DataLoader
train_loader_s1, test_loader_s1 = create_dataloaders(setting1_train, setting1_test)
train_loader_s2, test_loader_s2 = create_dataloaders(setting2_train, setting2_test)
```

### Step 3: 训练 SB 模型
```python
# Setting 1: 仅边界训练
sb_model_s1 = SchrodingerBridgeModel(dimension=100)
trainer_s1 = SBTrainer(sb_model_s1, train_loader_s1, test_loader_s1)
trainer_s1.train(epochs=100)

# Setting 2: 完整轨迹训练
sb_model_s2 = SchrodingerBridgeModel(dimension=100)
trainer_s2 = SBTrainer(sb_model_s2, train_loader_s2, test_loader_s2)
trainer_s2.train(epochs=100)
```

### Step 4: 评估与对比
```python
evaluator = Evaluator()

# 在测试集上评估
results_s1 = evaluator.evaluate(sb_model_s1, test_loader_s1)
results_s2 = evaluator.evaluate(sb_model_s2, test_loader_s2)

# 对比分析
evaluator.compare_settings(results_s1, results_s2)
evaluator.plot_comparison()
```

**核心假设**：Setting 2（完整轨迹）在测试集上的泛化能力优于 Setting 1（仅边界）

## 四、数据流与参数

**数据流**：
```
真实 h5ad 文件 → HVG 筛选 → 生物学划分验证
→ Setting 1: 起止点采样 → SB 训练 → 测试集评估
→ Setting 2: 全时间点采样 → SB 训练 → 测试集评估
→ 对比分析（FD, MAE, PCC）
```

**关键参数**：
- **数据配置**：
  - HVG 数量：100
  - 时间点：['0d', '8h', '1d', '3d', '7d']
  - Setting 1 采样：2000 cells/timepoint × 2 = 4000 cells
  - Setting 2 采样：800 cells/timepoint × 5 = 4000 cells

- **SB 模型**：
  - 隐藏层：[512, 512, 512, 512]
  - 时间编码维度：64
  - 扩散系数 D：0.1
  - Dropout：0.1

- **训练配置**：
  - 优化器：Adam (lr=5e-4)
  - 批次大小：256
  - 训练轮数：100
  - 早停 patience：10
  - 梯度裁剪：1.0

## 五、项目结构

```
Data/
  data_loader.py           # 真实数据加载与分析
  dataset_builder.py       # Setting 1/2 数据集构建
  __init__.py

Model/
  sb_model.py              # Schrödinger Bridge 模型
  __init__.py

Trainer/
  trainer.py               # SB 训练器
  __init__.py

Analyser/
  evaluator.py             # 评估与对比分析
  __init__.py

run_experiment.py          # 主实验脚本
configs/                   # 配置文件
  default_config.yaml      # 默认配置
```

## 六、预期实验结果

**Setting 对比**（测试集泛化）：

| Setting | 训练数据 | 样本数 | 预期性能 |
|---------|---------|--------|---------|
| Setting 1 | 仅起止点 (0d, 7d) | 4000 | 基线 |
| Setting 2 | 全时间点 (0d, 8h, 1d, 3d, 7d) | 4000 | 更优 |

**评估指标**：
- **Frechet Distance (FD)**：Setting 2 < Setting 1
- **Mean Absolute Error (MAE)**：Setting 2 < Setting 1
- **Pearson Correlation (PCC)**：Setting 2 > Setting 1

**关键发现**：
1. 在相同样本数量下，完整轨迹信息显著提升泛化能力
2. 中间时间点提供的动力学约束有助于学习更准确的漂移场
3. Setting 2 在测试集时间点的预测更加准确

**核心假设验证**：$\text{Performance}_{Setting2} > \text{Performance}_{Setting1}$

**依赖**：`torch`, `scanpy`, `anndata`, `numpy`, `pandas`, `matplotlib`
