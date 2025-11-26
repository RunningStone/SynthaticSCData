# Setting4 Ablation Study - 使用指南

## 实验目标

通过系统性地移除中间时间点（8h, 1d, 3d），量化每个时间点对模型性能的边际贡献，从而回答：
- 哪些时间点对学习EMT轨迹最关键？
- 中间状态是否包含不可从边界推断的信息？

## 实验设计

### 消融变体

基于Part1数据（0d, 8h, 1d, 3d, 7d），创建3个消融变体：

1. **Remove 8h**: 时间点 = {0d, 1d, 3d, 7d}
2. **Remove 1d**: 时间点 = {0d, 8h, 3d, 7d}
3. **Remove 3d**: 时间点 = {0d, 8h, 1d, 7d}

### 对照组

- **Setting2 (Full)**: 时间点 = {0d, 8h, 1d, 3d, 7d}

### 控制变量

- **总样本数**: 8974（与Setting2一致）
- **模型**: sb_mlplus（最优模型）
- **训练参数**: epochs=200, batch_size=64
- **评估端点**: 始终为0d→7d

## 文件结构

```
SynthaticSCData/
├── Workers/
│   ├── utils.py                              # Setting4配置验证函数
│   └── verify_setting4_configs.py            # CLI验证工具
├── Analyser/
│   ├── ablation_analyzer.py                  # 消融分析类（新）
│   └── __init__.py                           # 导出AblationAnalyzer
├── EXPs/EMTE2M/
│   ├── step2_run_ablations_setting4.sh       # 运行所有消融实验
│   ├── step3_analyze_ablation.sh             # 分析边际贡献
│   └── README_Setting4_Ablation.md           # 本文档
└── configs/EMT_E2M/
    ├── experiment_EMT_Part1_setting4_ablation_remove_8h.yaml
    ├── experiment_EMT_Part1_setting4_ablation_remove_1d.yaml
    └── experiment_EMT_Part1_setting4_ablation_remove_3d.yaml
```

## 使用流程

### 步骤0: 验证配置文件（可选）

```bash
# 在项目根目录执行
cd /home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData

# 验证所有Setting4配置文件
python Workers/verify_setting4_configs.py

# 或指定自定义配置目录
python Workers/verify_setting4_configs.py --config_dir configs/EMT_E2M
```

### 步骤1: 运行消融实验

```bash
# 在项目根目录执行
cd /home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData

# 顺序运行所有消融实验（推荐）
bash EXPs/EMTE2M/step2_run_ablations_setting4.sh sequential

# 或并行运行（需要3个GPU或足够的GPU内存）
bash EXPs/EMTE2M/step2_run_ablations_setting4.sh parallel
```

**预计时间**: 每个约3小时，总计9小时（顺序）或3小时（并行）

**输出目录**:
- `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting4_Ablation_Remove8h/`
- `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting4_Ablation_Remove1d/`
- `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting4_Ablation_Remove3d/`

### 步骤2: 分析边际贡献

**前置条件**:
1. Setting2（完整模型）已经训练完成
2. 所有三个消融实验已完成

```bash
# 在项目根目录执行
cd /home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData

# 运行分析
bash EXPs/EMTE2M/step3_analyze_ablation.sh
```

**输出目录**: `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting4_Ablation_Analysis/`

**生成文件**:
- `delta_P.csv`: 绝对边际贡献（ΔP）
- `I_margin.csv`: 相对边际贡献（%）
- `consistency.csv`: 跨指标一致性
- `critical_timepoints.json`: 关键时间点识别
- `marginal_contribution_absolute.png`: 各指标的绝对贡献柱状图
- `marginal_contribution_heatmap.png`: 相对贡献热力图
- `marginal_contribution_summary.png`: 平均贡献总结图
- `ablation_analysis_report.txt`: 综合分析报告

### 步骤3: 查看结果

```bash
# 查看文本报告
cat /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting4_Ablation_Analysis/ablation_analysis_report.txt

# 查看CSV结果
cd /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting4_Ablation_Analysis
cat I_margin.csv
cat consistency.csv

# 查看可视化（需要图像查看器）
# 或复制到本地查看
```

## 代码架构说明

### 模块化设计

1. **Workers/utils.py**: 配置验证工具函数
   - `load_config()`: 加载YAML配置
   - `verify_setting4_ablation_config()`: 验证单个配置
   - `verify_all_setting4_ablation_configs()`: 验证所有配置

2. **Analyser/ablation_analyzer.py**: 消融分析核心类
   - 继承自`BaseVisualizer`，复用基础设施
   - `load_experiment_results()`: 加载实验结果
   - `extract_metrics()`: 提取评估指标
   - `compute_marginal_contribution()`: 计算边际贡献
   - `identify_critical_timepoints()`: 识别关键时间点
   - `compute_consistency()`: 计算跨指标一致性
   - `plot_marginal_contribution()`: 生成可视化
   - `generate_report()`: 生成文本报告
   - `run_analysis()`: 运行完整分析流程

3. **Bash脚本**: 统一的实验运行接口
   - `step2_run_ablations_setting4.sh`: 批量运行消融实验
   - `step3_analyze_ablation.sh`: 运行分析并生成报告

### 设计原则

1. **最大化代码复用**: 
   - 复用`BaseVisualizer`的基础设施
   - 复用现有的`step2_run_experiment.py`训练入口
   - 不需要创建Setting4专用的训练脚本

2. **配置驱动**: 
   - 所有Setting4特性在配置文件中定义
   - 代码保持通用性

3. **模块化**: 
   - 验证、训练、分析功能分离
   - 每个模块职责单一

4. **统一接口**: 
   - Bash脚本提供一致的调用方式
   - 与其他Setting的脚本保持相同结构

## 预期结果模式

### 模式1: 均匀贡献
- 所有时间点贡献大致相等
- 削弱"熵峰值时刻特别重要"的假设

### 模式2: 晚期时间点敏感
- 3d的贡献显著高于8h和1d
- 支持"接近峰值的中间状态不可从边界推断"的假设

### 模式3: 阶段依赖
- 某个特定时间点（8h、1d或3d）显著更重要
- 说明EMT过程的不同阶段对建模难度有非对称贡献

## 关键指标

### 边际信息贡献率

$$
I_{\text{margin}}(t_i) = \frac{\Delta P(t_i)}{P_{\text{full}}} \times 100\%
$$

其中 $\Delta P(t_i) = P_{\text{full}} - P_{\text{ablation}(-t_i)}$

### 关键时间点阈值

$$
I_{\text{margin}}(t_i) > 1.5 \times \text{mean}(I_{\text{margin}})
$$

### 跨指标一致性

$$
\text{Consistency}(t_i) = \frac{1}{|\mathcal{M}|} \sum_{m \in \mathcal{M}} \mathbb{1}[\text{rank}_m(t_i) \leq 2]
$$

## 故障排查

### 问题1: 配置验证失败

```bash
# 运行验证工具查看详细错误
python Workers/verify_setting4_configs.py
```

### 问题2: 训练失败

```bash
# 检查日志文件
cat /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting4_Ablation_Remove8h/logs/experiment.log
```

### 问题3: 分析失败 - 缺少实验结果

确保以下实验都已完成：
1. Setting2 (完整模型)
2. Setting4_Ablation_Remove8h
3. Setting4_Ablation_Remove1d
4. Setting4_Ablation_Remove3d

检查results.json是否存在：
```bash
ls -l /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/*/results.json
```

## 与其他实验的关联

- **实验7（熵演化）**: 如果3d被识别为最关键点，预测熵峰值在3d或7d附近
- **实验5（时间打乱）**: 如果存在关键时间点，预测打乱后性能下降

## 作者

Shi Pan

## 更新日期

2024-11-24
