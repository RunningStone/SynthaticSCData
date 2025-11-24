# 真实数据时间序列细胞状态转换学习

## 项目概述

本项目基于真实时间序列单细胞数据，研究细胞状态转换动力学建模中的一个核心问题：在学习细胞状态转换时，完整的时间轨迹信息是否比仅有起止点信息更有价值。项目使用EMT数据集，包含前向EMT过程和刺激移除后的逆转过程，通过配置化的实验框架系统地对比不同时间点采样策略对模型泛化能力的影响。

### 核心研究问题

在相同样本总量下，使用完整时间轨迹训练的模型是否能比仅使用边界点训练的模型学到更准确的状态转换规律？这个问题的答案将影响实验设计和数据收集策略。

### 实验设计理念

项目采用模块化配置系统，支持灵活定义不同的实验设置。每个设置通过YAML配置文件指定时间点选择、数据采样策略和模型配置。系统自动处理数据加载、模型训练、评估和可视化的完整流程。

当前实现包含六个预定义设置，分为两个实验部分（Part）：

**Part1 (仅正向EMT)**：
- Setting1: 边界点 (0d, 7d) - 低时间分辨率
- Setting2: 全部时间点 (0d→8h→1d→3d→7d) - 高时间分辨率
- Setting3: 关键时间点 (0d, 8h, 7d) - 中等分辨率

**Part2 (正向+反向)**：
- Setting1: 全部8个时间点 - 完整双向轨迹
- Setting2: 边界点 (0d, 3d_rm) - 长距离插值
- Setting3: 关键点 (0d, 7d, 3d_rm) - 峰值信息影响
## 实验流程

### Step 1: 训练模型

系统通过配置文件驱动的方式运行实验。用户指定实验配置文件后，系统自动完成数据加载、模型训练和评估的完整流程。每个实验配置引用三个基础配置文件：数据配置定义数据源和时间点选择，模型配置定义网络架构和训练参数，分析器配置定义评估指标和可视化方法。

运行单个实验设置的命令格式为：

```bash
# Part1 - 仅正向EMT
bash step1_run_experiment_EMT.sh experiment_EMT_Part1_setting1.yaml
bash step1_run_experiment_EMT.sh experiment_EMT_Part1_setting2.yaml
bash step1_run_experiment_EMT.sh experiment_EMT_Part1_setting3.yaml

# Part2 - 包含反向
bash step1_run_experiment_EMT.sh experiment_EMT_Part2_setting1.yaml
bash step1_run_experiment_EMT.sh experiment_EMT_Part2_setting2.yaml
bash step1_run_experiment_EMT.sh experiment_EMT_Part2_setting3.yaml
```

系统首先加载并验证配置，确保所有引用的时间点存在且有足够的细胞数量。然后根据配置中指定的采样策略准备训练和测试数据。对于每个启用的模型，系统创建相应的训练器实例，执行训练循环并保存最佳检查点。训练过程包含早停机制和学习率调度，自动优化收敛性能。最后系统在测试集上评估所有训练好的模型，计算十个标准指标并保存结果到JSON文件。

### Step 2: 跨设置可视化对比

可视化系统支持同时加载多个实验设置的结果，进行横向对比分析。系统接收多个实验配置文件路径作为输入，自动聚合所有训练好的模型，生成统一的对比图表。

运行可视化的命令格式为：

```bash
bash step2_run_multi_setting_visualization.sh
```

该脚本默认对比Setting1和Setting2的结果。用户也可以通过命令行参数指定任意组合的配置文件进行对比。

可视化流程包含四个主要步骤。首先从每个设置的测试集中采样数据，确保所有时间点都有足够的样本用于对比。然后加载所有训练好的模型检查点，使用源时间点的数据生成目标时间点的预测。接着计算两种降维嵌入：PHATE嵌入保留全局流形结构，LMNN+PCA嵌入强调时间点分离。最后生成三类输出文件：指标对比图以条形图形式展示所有模型在十个评估指标上的表现，PHATE和LMNN+PCA可视化图展示生成数据与真实数据在低维空间的分布对比，CSV文件记录详细的数值结果供进一步分析。

系统计算的十个标准指标包括测试集损失、Fréchet距离、平均绝对误差、Pearson相关系数、Wasserstein距离、最大均值差异、R平方均值、JS散度、相关矩阵Frobenius差异和相关结构相关性。这些指标从不同角度量化模型的预测质量和泛化能力。

## 模型架构

项目实现了五种生成模型用于细胞状态转换学习。Schrödinger Bridge模型是核心，通过学习时间依赖的漂移场来描述细胞状态演化。基础SB模型使用双势函数网络参数化漂移场，每个势函数由四层MLP实现，隐藏层维度为512。MLPlus增强版本针对多时间点场景进行了优化，引入多尺度时间编码使用10个可学习频率捕捉不同时间尺度的动力学，采用4个残差块提高梯度稳定性。

Optimal Transport模型学习从初始分布到目标分布的确定性映射，通过最小化Wasserstein-2距离实现。Conditional VAE模型采用变分自编码器框架，编码器将起始状态和时间条件映射到潜在空间，解码器重构目标状态。Batch OT模型针对多时间点场景设计，包含7个独立的OT模型（每个时间转换一个），推理时顺序应用这些映射并在离散状态间插值。

### 模型参数量平衡

所有模型的参数量已优化至相近水平，确保公平对比：

| 模型 | 参数量 | 显存占用 | 说明 |
|------|--------|---------|------|
| VAE | 5.95M | 196 MB | 最小模型 |
| SB MLPlus | 9.80M | 322 MB | 基准模型 |
| OT | 10.16M | 334 MB | +3.6% |
| SB | 10.59M | 347 MB | +8.0% |
| Batch OT | 10.86M | 357 MB | +10.7% (7个OT模型) |

**Batch OT优化**: 原始配置参数量达71M，通过减小每个OT模型的hidden_dims（从[1536,1536,1536,1536]降到[512,512,512]），使总参数量降至10.86M，与其他模型相当。

所有模型使用统一的训练框架，包含AdamW优化器、学习率调度、梯度裁剪和早停机制。训练过程自动保存最佳检查点，支持从中断点恢复训练。

## 项目结构

项目采用模块化设计，代码组织清晰且易于扩展。Data模块负责数据加载和预处理，包含RealDataLoader类处理h5ad文件读取、HVG筛选和训练测试划分，ConfigLoader类实现配置文件的加载和合并。Model模块定义所有生成模型的网络架构，每个模型类实现统一的接口包括forward方法和compute_loss方法。Trainer模块提供训练和评估功能，SBTrainer处理Schrödinger Bridge模型的训练，UnifiedTrainer处理OT和VAE模型，BatchOTTrainer处理批量OT模型，Evaluator类计算所有评估指标。Analyser模块实现跨设置的可视化对比，MultiSettingVisualizer类协调整个可视化流程，包含数据采样、模型加载、降维嵌入和图表生成。

configs目录包含所有配置文件，数据配置定义数据源和时间点选择，模型配置定义网络架构和训练参数，实验配置组合这些基础配置并指定要训练的模型。每个实验设置对应一个独立的实验配置文件。

step1_run_experiment.py是训练的主入口，通过命令行参数接收配置文件路径。step2_multi_setting_visualization.py是可视化的主入口，支持同时对比多个实验设置。两个bash脚本提供便捷的启动方式，自动激活虚拟环境并传递参数。

## 快速开始

### 环境设置

项目使用Python虚拟环境管理依赖。运行环境设置脚本创建虚拟环境并安装所有必需的包，包括PyTorch、Scanpy、AnnData和可视化库。

```bash
bash step0_setup_env.sh
source .venv/bin/activate
```

### 运行单个实验设置

每个实验设置对应一个配置文件。使用bash脚本指定配置文件名即可运行完整的训练和评估流程。系统自动处理数据加载、模型训练、指标计算和结果保存。

```bash
# Part1 - 仅正向EMT
bash step1_run_experiment_EMT.sh experiment_EMT_Part1_setting1.yaml  # 边界点
bash step1_run_experiment_EMT.sh experiment_EMT_Part1_setting2.yaml  # 全部时间点
bash step1_run_experiment_EMT.sh experiment_EMT_Part1_setting3.yaml  # 关键时间点

# Part2 - 包含反向
bash step1_run_experiment_EMT.sh experiment_EMT_Part2_setting1.yaml  # 全部8个时间点
bash step1_run_experiment_EMT.sh experiment_EMT_Part2_setting2.yaml  # 边界点
bash step1_run_experiment_EMT.sh experiment_EMT_Part2_setting3.yaml  # 关键点
```

每个设置的结果保存在独立的输出目录，包含模型检查点、训练历史、评估指标和日志文件。目录结构为OUTPUTs/SynthaticSCData/EMT_Part{X}_Setting{Y}，其中X是Part编号（1或2），Y是Setting编号（1、2或3）。

### 跨设置可视化对比

可视化脚本支持同时加载多个实验设置的结果进行横向对比。默认脚本对比Setting1和Setting2，用户也可以通过修改脚本指定任意组合的设置。

```bash
bash step2_run_multi_setting_visualization.sh
```

可视化结果保存在独立的visualizations目录，包含指标对比条形图、PHATE降维可视化、LMNN+PCA降维可视化和CSV格式的详细数值结果。这些文件可直接用于论文撰写和进一步分析。

### 查看和分析结果

每个实验设置的results.json文件包含所有训练好的模型的完整评估指标。可视化目录的CSV文件提供跨设置的指标汇总，便于进行统计分析和性能对比。

```bash
# 查看某个设置的结果
cat OUTPUTs/SynthaticSCData/EMT_Part1_Setting1/results.json

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

## 命名规范

### 配置文件命名

项目采用Part1/Part2命名规范，清晰区分实验类型：

```
experiment_EMT_Part{X}_setting{Y}.yaml
                   ↑           ↑
                   |           └─ 具体实验设计 (1,2,3)
                   └─ 实验部分 (1=仅正向, 2=含反向)
```

### 输出目录结构

```
/OUTPUTs/SynthaticSCData/
├── EMT_Part1_Setting1/  # Part1_setting1 (Forward EMT, Boundary)
├── EMT_Part1_Setting2/  # Part1_setting2 (Forward EMT, Full)
├── EMT_Part1_Setting3/  # Part1_setting3 (Forward EMT, Key points)
├── EMT_Part2_Setting1/  # Part2_setting1 (With Reversal, Full)
├── EMT_Part2_Setting2/  # Part2_setting2 (With Reversal, Boundary)
└── EMT_Part2_Setting3/  # Part2_setting3 (With Reversal, Key points)
```

### 数据采样参数

所有设置的采样参数已优化，确保公平对比：

| 设置 | 时间点数 | 采样策略 | 总样本量 |
|------|---------|---------|---------|
| Part1_setting1 | 2 | 4,487 cells/tp | 8,974 |
| Part1_setting2 | 5 | total | 8,974 |
| Part1_setting3 | 3 | total | 8,974 |
| Part2_setting1 | 8 | total | 8,974 |
| Part2_setting2 | 2 | 4,487 cells/tp | 8,974 |
| Part2_setting3 | 3 | total | 8,974 |

详细说明参见 `NAMING_CONVENTION_UPDATE.md` 和 `DATA_CONFIG_UPDATE_SUMMARY.md`。

## 更多信息

- **详细系统设计**：参见 `20251118_SystemDesign.md`
- **命名规范更新**：参见 `NAMING_CONVENTION_UPDATE.md`
- **数据配置更新**：参见 `DATA_CONFIG_UPDATE_SUMMARY.md`
- **模型参数平衡**：参见 `BATCH_OT_PARAMETER_BALANCE.md`
- **快速参考**：参见 `QUICK_REFERENCE.md`
- **配置文件示例**：参见 `configs/` 目录
- **问题反馈**：提交 Issue

## 许可证

MIT License
