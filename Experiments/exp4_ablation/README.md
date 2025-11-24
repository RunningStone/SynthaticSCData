# Experiment 4: Timepoint Ablation Study

## 目标

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

## 使用方法

### 步骤1: 运行消融实验

```bash
# 在项目根目录执行
cd /home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData

# 消融1: 移除8h
bash step1_run_experiment_EMT.sh configs/experiment_EMT_Part1_setting4_ablation_remove_8h.yaml

# 消融2: 移除1d
bash step1_run_experiment_EMT.sh configs/experiment_EMT_Part1_setting4_ablation_remove_1d.yaml

# 消融3: 移除3d
bash step1_run_experiment_EMT.sh configs/experiment_EMT_Part1_setting4_ablation_remove_3d.yaml
```

**预计时间**: 每个约3小时，总计9小时（可并行）

### 步骤2: 分析边际贡献

确保Setting2（完整模型）已经训练完成，然后运行分析脚本：

```bash
python Experiments/exp4_ablation/analyze_marginal_contribution.py \
    --output_base /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData \
    --model sb_mlplus
```

**输出**:
- `delta_P.csv`: 绝对边际贡献（ΔP）
- `I_margin.csv`: 相对边际贡献（%）
- `consistency.csv`: 跨指标一致性
- `critical_timepoints.json`: 关键时间点识别
- `marginal_contribution_absolute.png`: 各指标的绝对贡献柱状图
- `marginal_contribution_heatmap.png`: 相对贡献热力图
- `marginal_contribution_summary.png`: 平均贡献总结图
- `ablation_analysis_report.txt`: 综合分析报告

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

## 与其他实验的关联

- **实验7（熵演化）**: 如果3d被识别为最关键点，预测熵峰值在3d或7d附近
- **实验5（时间打乱）**: 如果存在关键时间点，预测打乱后性能下降

## 文件结构

```
Experiments/exp4_ablation/
├── README.md                           # 本文件
├── analyze_marginal_contribution.py    # 分析脚本
└── [分析结果将保存在 OUTPUTs/SynthaticSCData/exp4_ablation_analysis/]
```

## 配置文件

```
configs/
├── experiment_EMT_Part1_setting4_ablation_remove_8h.yaml
├── experiment_EMT_Part1_setting4_ablation_remove_1d.yaml
└── experiment_EMT_Part1_setting4_ablation_remove_3d.yaml
```

## 依赖

- numpy
- pandas
- matplotlib
- seaborn
- json (标准库)

## 作者

Shi Pan

## 日期

2024-11-18
