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
## 实验流程

### Step 1: 训练模型

```bash
bash step1_run_experiment.sh
```

训练流程：
```
[Step 1/9] 加载和分析数据
    ↓
[Step 2/9] 准备 Setting 1 数据 (0d → 7d)
    ↓
[Step 3/9] 准备 Setting 2 数据 (0d → 8h → 1d → 3d → 7d)
    ↓
[Step 4/9] 训练 Setting 1 - Schrödinger Bridge
    ↓
[Step 5/9] 训练 Setting 1 - Optimal Transport
    ↓
[Step 6/9] 训练 Setting 1 - Conditional VAE
    ↓
[Step 7/9] 训练 Setting 2 - MLPlus Schrödinger Bridge
    ↓
[Step 8/9] 评估所有模型
    ↓
[Step 9/9] 保存结果到 results.json
```

### Step 2: 多设置可视化（新版）

```bash
bash step2_run_multi_setting_visualization.sh
```

**功能特性**：
- ✅ **跨Setting汇总**：自动聚合所有实验设置的模型（Setting1-sb, Setting1-ot, Setting1-vae, Setting2-sb_mlplus）
- ✅ **动态子图布局**：根据模型数量自动调整可视化布局（1+N+1个子图）
- ✅ **评估指标对比**：横向对比10个标准指标，红色边框高亮最佳模型
- ✅ **独立输出目录**：可视化结果保存在独立的`visualizations/`目录

**生成文件**：
```
visualizations/
├── metrics_comparison.png/pdf/csv      # 10个评估指标的横向对比
├── generation_comparison_phate.png/pdf # PHATE降维可视化
└── generation_comparison_lmnn_pca.png/pdf # LMNN+PCA监督降维可视化
```

**对比指标**：
1. Test Loss - 测试集损失
2. Fréchet Distance - 生成分布与真实分布的差异
3. MAE - 平均绝对误差
4. PCC - Pearson相关系数
5. Wasserstein Distance - Wasserstein距离
6. MMD - Maximum Mean Discrepancy
7. R² (mean) - R平方均值
8. JS Divergence - JS散度
9. Correlation Frobenius Diff - 相关矩阵Frobenius差异
10. Correlation Structure Corr - 相关结构相关性

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
├── Analyser/                   # 可视化和分析工具
│   ├── multi_setting_visualizer.py         # 多设置可视化主类
│   ├── multi_setting_visualizer_methods.py # 数据加载和模型生成
│   ├── multi_setting_visualizer_viz.py     # 可视化方法
│   └── embedding_learner.py                # 嵌入学习（可选）
├── step1_run_experiment.py     # Step 1: 训练脚本
├── step1_run_experiment.sh     # Step 1: Bash启动脚本
├── step2_multi_setting_visualization.py  # Step 2: 可视化脚本
├── step2_run_multi_setting_visualization.sh # Step 2: Bash启动脚本
├── README.md                   # 本文档
├── VISUALIZATION_GUIDE.md      # 可视化系统详细指南
└── SystemDesign.md             # 详细系统设计文档
```

## 快速开始

### 环境设置

```bash
# 创建虚拟环境
bash step0_setup_env.sh

# 激活环境
source .venv/bin/activate

# 安装依赖
pip install torch scanpy anndata numpy pandas scipy matplotlib tqdm phate metric-learn
```

### 完整实验流程

#### Step 1: 训练所有模型

```bash
bash step1_run_experiment.sh
```

这将训练4个模型：
- Setting1: SB, OT, VAE (仅边界点)
- Setting2: SB_MLPlus (完整轨迹)

输出目录：
```
OUTPUTs/SynthaticSCData/
├── EMT_Setting1/
│   ├── checkpoints/
│   │   ├── sb/best_model.pt
│   │   ├── ot/best_model.pt
│   │   └── vae/best_model.pt
│   ├── experiment_config.yaml
│   └── results.json
└── EMT_Setting2/
    ├── checkpoints/
    │   └── sb_mlplus/best_model.pt
    ├── experiment_config.yaml
    └── results.json
```

#### Step 2: 生成跨设置可视化

```bash
bash step2_run_multi_setting_visualization.sh
```

输出目录：
```
OUTPUTs/SynthaticSCData/visualizations/
├── metrics_comparison.png          # 10个指标横向对比
├── metrics_comparison.pdf
├── metrics_comparison.csv          # 可用于进一步分析
├── generation_comparison_phate.png # PHATE可视化
├── generation_comparison_phate.pdf
├── generation_comparison_lmnn_pca.png # LMNN+PCA可视化
└── generation_comparison_lmnn_pca.pdf
```

### 查看结果

```bash
# 查看Setting1的评估结果
cat OUTPUTs/SynthaticSCData/EMT_Setting1/results.json

# 查看Setting2的评估结果
cat OUTPUTs/SynthaticSCData/EMT_Setting2/results.json

# 查看跨设置指标对比
cat OUTPUTs/SynthaticSCData/visualizations/metrics_comparison.csv
```

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

### 训练结果 (Step 1)

每个setting的`results.json`包含所有模型的评估指标：

```json
{
  "sb": {
    "evaluation": {
      "test_loss": 2152.51,
      "frechet_distance": 290066.40,
      "mae": 10.20,
      "pcc": 0.59,
      "wasserstein_distance": 6.61,
      "mmd": 0.47,
      "r2_mean": -2.52,
      "js_divergence": 0.46,
      "correlation_frobenius_diff": 15.67,
      "correlation_structure_corr": -0.01
    }
  }
}
```

### 可视化结果 (Step 2)

**指标对比图** (`metrics_comparison.png/pdf`):
- 横向条形图显示所有模型在10个指标上的表现
- 红色边框标记每个指标的最佳模型
- CSV文件可用于进一步统计分析

**生成质量可视化** (`generation_comparison_*.png/pdf`):
- **PHATE嵌入**：基于流形学习的非线性降维，保留全局结构
- **LMNN+PCA嵌入**：基于度量学习的监督降维，强调类别分离

每个可视化包含：
1. **子图1**：所有原始数据的时间点分布（参考图）
2. **子图2-N+1**：目标时间点(7d)真实数据 vs 每个模型的生成数据

**验证逻辑**：模型从源时间点(0d)生成目标时间点(7d)的数据

**预期结果**：
- Setting2-sb_mlplus（完整轨迹）应在多数指标上优于Setting1模型（仅边界）
- 生成样本应与目标时间点的原始数据在嵌入空间中重叠

## 常见问题

### 训练相关

**Q: 训练时间过长怎么办？**
- 修改配置文件中的epochs参数
- 使用GPU加速：确保`--device cuda`
- 减少数据量（修改配置文件中的采样参数）

**Q: 如何使用自己的数据？**
- 准备h5ad格式的AnnData对象
- 修改配置文件中的`data_source`部分
- 确保包含时间标签和批次信息用于train/test划分

**Q: 如何调整模型架构？**
- 修改配置文件中的`models`部分
- 可调整hidden_dims, dropout, learning_rate等参数

### 可视化相关

**Q: 如何添加更多setting到可视化？**
```bash
python step2_multi_setting_visualization.py \
    --config_paths \
        path/to/Setting1/experiment_config.yaml \
        path/to/Setting2/experiment_config.yaml \
        path/to/Setting3/experiment_config.yaml \
    --output_dir ./visualizations
```

**Q: 可视化内存不足怎么办？**
- 减少`--n_samples_per_timepoint`参数（默认500）
- 减少`--n_generate_per_model`参数（默认500）

**Q: 如何解释可视化结果？**
- **指标对比图**：红色边框表示该指标的最佳模型
- **子图1（参考图）**：显示所有时间点的轨迹，了解整体动态过程
- **子图2-N+1（对比图）**：每个模型的生成数据（浅蓝色星形）应与7d真实数据（粉红色圆点）在嵌入空间中重叠
- **PHATE vs LMNN+PCA**：PHATE保留全局流形结构，LMNN+PCA强调时间点分离

**Q: 如何导出高分辨率图片？**
- PDF文件已自动生成，适合论文使用
- 可修改代码中的`dpi=300`参数提高分辨率

## 版本更新

### v2.1.1 (2024-11-15) - 可视化优化
- ✅ **优化**：简化可视化布局，移除冗余的"所有模型汇总"子图
- ✅ **优化**：对比图只显示目标时间点(7d)真实数据 vs 模型生成
- ✅ **改进**：更清晰的验证逻辑说明（0d → 7d）
- ✅ **改进**：配色优化 - 粉红色(真实数据) vs 浅蓝色(生成数据)，对比更清晰

### v2.1 (2024-11-15) - 多设置可视化系统
- ✅ **新增**：跨设置模型汇总和对比可视化
- ✅ **新增**：10个标准评估指标的横向对比图
- ✅ **新增**：动态子图布局（根据模型数量自动调整）
- ✅ **新增**：PHATE和LMNN+PCA两种降维可视化
- ✅ **新增**：CSV格式指标导出，便于进一步分析
- ✅ **改进**：独立的可视化输出目录
- ✅ **改进**：模型命名格式：`Setting名-模型名`
- ❌ **移除**：旧版单setting可视化系统

### v2.0 (2024-11-10) - 多模型对比
- ✅ Setting 2 数据量增加 2.5 倍
- ✅ MLPlus 模型增强
- ✅ 新增 OT 和 VAE 模型

### v1.0 (2024-11-01) - 初始版本
- ✅ 基础 Schrödinger Bridge 模型
- ✅ Setting 1/2 对比实验

## 更多信息

- **详细系统设计**：参见 `SystemDesign.md`
- **可视化系统指南**：参见 `VISUALIZATION_GUIDE.md`
- **配置文件示例**：参见 `configs/` 目录
- **问题反馈**：提交 Issue

## 许可证

MIT License
