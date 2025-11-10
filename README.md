# 真实数据时间序列细胞状态转换学习

## 项目概述

本项目基于真实时间序列单细胞 RNA-seq 数据，使用 Schrödinger Bridge (SB) 模型对比研究**完整动力学轨迹**与**仅边界信息**对模型泛化能力的影响。

### 核心研究问题

**学习完整动力学轨迹是否优于仅学习起止点分布？**

### 实验设计

对比两种训练策略：
- **Setting 1（仅边界）**：仅使用起止两个时间点训练（如 0d → 7d）
  - 数据量: 2 个时间点 × 2000 cells = 4000 cells
  - 模型: 基础 SchrodingerBridgeModel
- **Setting 2（完整轨迹）**：使用所有中间时间点训练（如 0d → 8h → 1d → 3d → 7d）
  - 数据量: 5 个时间点 × 2000 cells = **10000 cells** ⬆️
  - 模型: **MLPlus_SchrodingerBridgeModel**（增强版）

**关键改进**（v2.0）:
- ✅ Setting 2 数据量增加 2.5 倍，充分利用多时间点信息
- ✅ MLPlus 模型：多尺度时间编码 + 残差连接 + LayerNorm
- ✅ 保留旧的总量平衡策略作为可选项

### 模型

- **SchrodingerBridgeModel**: 基础 SB 模型，适用于简单的边界学习
- **MLPlus_SchrodingerBridgeModel**: 增强版 SB 模型，专为多时间点设计
  - 多尺度时间嵌入（10 个可学习频率）
  - 4 个残差块（梯度稳定性）
  - Layer Normalization
  - 参数量: ~4.7M（基础版 ~2.7M）

## 项目结构

```
SynthaticSCData/
├── Data/                       # 数据加载和处理
│   ├── data_loader.py          # 真实数据加载、HVG选择、生物学划分
│   ├── dataset_builder.py      # PyTorch Dataset 和 DataLoader 构建
│   └── __init__.py
├── Model/                      # 模型定义
│   ├── sb_model.py             # Schrödinger Bridge 基础模型
│   ├── sb_model_mlplus.py      # MLPlus 增强模型（继承基础模型）
│   └── __init__.py
├── Trainer/                    # 训练和评估
│   ├── sb_trainer.py           # SB 模型训练器（早停、检查点保存）
│   ├── sb_evaluator.py         # 评估器（FD, MAE, PCC 等指标）
│   └── __init__.py
├── Analyser/                   # 分析工具（可选）
├── run_experiment.py           # 主实验脚本
├── test_pipeline.py            # 测试脚本
├── README.md                   # 本文档
└── SystemDesign.md             # 详细系统设计文档
```

## 快速开始

### 安装依赖

```bash
pip install torch scanpy anndata numpy pandas scipy matplotlib tqdm
```

### 1. 测试流程（推荐首次运行）

```bash
python test_pipeline.py
```

测试内容：
- ✅ 加载 EMT 数据集（~53K cells）
- ✅ HVG 选择和生物学划分验证
- ✅ Setting 1/2 数据集创建
- ✅ SB 模型创建和前向传播

### 2. 运行完整实验

**快速测试（~5分钟）**：
```bash
python run_experiment.py \
    --n_hvg 50 \
    --cells_per_timepoint 500 \
    --epochs 10 \
    --output_dir ./quick_test
```

**完整实验（~2-3小时）**：
```bash
python run_experiment.py \
    --n_hvg 100 \
    --cells_per_timepoint 2000 \
    --epochs 100 \
    --device cuda \
    --output_dir ./outputs
```

**命令行参数**：
- `--file_path`: h5ad 文件路径（默认使用 EMT 数据）
- `--n_hvg`: 高变基因数量（默认 100）
- `--cells_per_timepoint`: Setting 1 每个时间点的细胞数（默认 2000）
- `--batch_size`: 批次大小（默认 256）
- `--epochs`: 训练轮数（默认 100）
- `--lr`: 学习率（默认 5e-4）
- `--device`: 设备（cuda 或 cpu）
- `--output_dir`: 输出目录（默认 ./outputs）
- `--seed`: 随机种子（默认 42）

### 3. 查看结果

实验完成后，查看输出目录：
```bash
# 查看结果摘要
cat outputs/results.json

# 查看对比图表
# outputs/comparison.png
```

结果包含：
- **Test Loss**: 测试集损失（越小越好）
- **Frechet Distance**: 生成分布与真实分布的差异（越小越好）
- **MAE**: 平均绝对误差（越小越好）
- **PCC**: Pearson 相关系数（越大越好）

### 4. 使用自己的数据

修改 `Data/data_loader.py` 中的 `create_default_emt_data_loader()` 函数：

```python
loader = RealDataLoader(
    file_path="YOUR_DATA.h5ad",
    n_hvg=100,
    obs_time_column='YOUR_TIME_COLUMN',      # 如 'timepoint'
    time_labels=['t0', 't1', 't2', 't3'],    # 你的时间标签
    time_label_order=['t0', 't1', 't2', 't3'],
    biology_split={
        "train_val_column": "YOUR_BATCH_COLUMN",  # 如 'batch'
        "train": ["batch1", "batch2"],
        "test": ["batch3"]
    }
)
```

然后运行：
```bash
python run_experiment.py --file_path YOUR_DATA.h5ad
```

## 数据要求

**输入格式**：AnnData (.h5ad) 文件

**必需的 obs 列**：
- 时间标签列（如 'Ground_truth', 'timepoint'）
- 批次/分组列（如 'batches', 'batch'）用于训练/测试划分

**自动验证**：
- ✅ 训练集和测试集都包含所有时间点
- ✅ 每个时间点的样本数量统计
- ✅ HVG 选择和数据质量检查

## 实验流程

1. **数据加载** → 加载 h5ad，选择 HVG，创建训练/测试划分
2. **Setting 1** → 采样起止点（2N 样本）
3. **Setting 2** → 采样所有时间点（≈2N 样本，保持总数相近）
4. **训练** → 分别训练两个 SB 模型（相同架构和超参数）
5. **评估** → 计算 Test Loss, FD, MAE, PCC 并对比

## 技术细节

### 模型架构

**Schrödinger Bridge**：
- 势函数网络：φ(x,t) 和 ψ(x,t)，各 4 层 MLP [512, 512, 512, 512]
- 时间编码：Sinusoidal embedding + 可学习变换（64维）
- 漂移场：b(x,t) = -D∇[φ + ψ]，通过自动微分计算
- 扩散系数：D = 0.1

### 训练配置

- **优化器**：Adam (lr=5e-4)
- **批次大小**：256
- **梯度裁剪**：max_norm=1.0
- **早停**：patience=10
- **检查点**：自动保存最佳模型

### 评估指标

1. **Test Loss**：SB 模型的测试集损失
2. **Frechet Distance (FD)**：生成分布与真实分布的差异
3. **Mean Absolute Error (MAE)**：逐细胞、逐基因的平均绝对误差
4. **Pearson Correlation (PCC)**：基因表达模式的相关性

## 输出结果

```
outputs/
├── setting1/                      # Setting 1 结果
│   ├── best_model.pt
│   ├── final_model.pt
│   └── training_history.json
├── setting2/                      # Setting 2 结果
│   ├── best_model.pt
│   ├── final_model.pt
│   └── training_history.json
├── results.json                   # 完整结果和对比
└── comparison.png                 # 可视化对比图
```

**预期结果**：Setting 2（完整轨迹）的泛化能力优于 Setting 1（仅边界）
- Test Loss: Setting 2 < Setting 1
- Frechet Distance: Setting 2 < Setting 1
- MAE: Setting 2 < Setting 1
- PCC: Setting 2 > Setting 1

## 常见问题

**Q: 训练时间过长怎么办？**
- 减少 HVG：`--n_hvg 50`
- 减少样本：`--cells_per_timepoint 1000`
- 减少轮数：`--epochs 50`

**Q: 如何使用随机划分？**
- 设置 `biology_split["train_val_column"] = "random"`

**Q: 如何解释结果？**
- 如果 Setting 2 所有指标都优于 Setting 1，说明中间时间点信息有助于学习动力学和提升泛化能力

**Q: 内存不足怎么办？**
- 减小批次：`--batch_size 128`
- 减少 HVG 和样本数

## 更多信息

- 详细系统设计：参见 `SystemDesign.md`
- 测试脚本：`test_pipeline.py`
- 问题反馈：提交 Issue

## 许可证

MIT License
