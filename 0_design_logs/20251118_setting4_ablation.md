# 实验4：时间点系统消融实验 - 详细实现分析

## 实验目标回顾

通过系统性移除单个中间时间点，量化每个时间点对模型性能的边际贡献，识别对准确建模最关键的时间窗口。核心问题：**是否所有中间时间点同等重要，还是存在信息密度更高的关键时刻？**

---

## 现有系统支持度分析

### 完全支持的功能（无需修改）

1. **任意时间点子集选择**
   - 配置文件已支持：`time_points: ["0d", "1d", "3d", "7d"]`（移除8h）
   - 数据加载器自动过滤：`RealDataLoader`根据`time_points`筛选细胞
   - 验证机制完备：自动检查每个时间点的样本数量

2. **采样参数自动计算**
   - 工具脚本：`step0_calculate_data_split_params.py`
   - 输入：时间点数量、总样本量约束
   - 输出：每个时间点的采样数量（保证总量一致）

3. **模型训练和评估**
   - 统一训练接口：`train_model(config)`
   - 10个标准评估指标：已实现并验证
   - 检查点保存：自动保存最佳模型

### 需要补充的功能（轻量扩展）

1. **边际贡献分析脚本**（后处理，非训练流程）
   - 功能：聚合多个消融setting的结果，计算$\Delta P(t_i)$
   - 实现难度：低（约100行代码）
   - 依赖：仅需读取已保存的`results.json`文件

---

## 实验设计的数学形式化

### 消融变体定义

给定完整时间序列$\mathcal{T}_{\text{full}} = \{t_0, t_1, t_2, t_3, t_4\} = \{0d, 8h, 1d, 3d, 7d\}$，定义3个消融变体：

$$
\begin{aligned}
\mathcal{S}_{-1} &: \mathcal{T} = \{t_0, t_2, t_3, t_4\} = \{0d, 1d, 3d, 7d\} \quad \text{(移除8h)} \\
\mathcal{S}_{-2} &: \mathcal{T} = \{t_0, t_1, t_3, t_4\} = \{0d, 8h, 3d, 7d\} \quad \text{(移除1d)} \\
\mathcal{S}_{-3} &: \mathcal{T} = \{t_0, t_1, t_2, t_4\} = \{0d, 8h, 1d, 7d\} \quad \text{(移除3d)} 
\end{aligned}
$$

**注意**：
- 保留$t_0=0d$作为起点和$t_4=7d$作为终点（所有变体共享）
- 只移除**中间时间点**（8h, 1d, 3d），不移除边界点
- Part1没有reversal数据，所以只研究前向EMT过程

### 采样参数计算

总样本量约束：$M = 8974$（与Setting2一致）

对于消融变体$\mathcal{S}_{-i}$，包含$K_i = |\mathcal{T}_{-i}| = 4$个时间点，每点采样：

$$
N_i = \left\lfloor \frac{M}{K_i} \right\rfloor = \left\lfloor \frac{8974}{4} \right\rfloor = 2243 \text{ cells/timepoint}
$$

**实际采样策略**：使用`balance_strategy: "total"`，让系统自动均分。

### 性能下降度量

对于每个评估指标$P$（如MAE, FD, PCC等），定义边际贡献：

$$
\Delta P(t_i) = P_{\text{full}} - P_{-i}
$$

其中：
- $P_{\text{full}}$：Setting2（完整5个时间点）的性能
- $P_{-i}$：移除时间点$t_i$后的性能

**解释**：
- $\Delta P > 0$：移除该点导致性能下降（该点有正贡献）
- $\Delta P < 0$：移除该点反而性能提升（可能是噪声或冗余）
- $|\Delta P|$大：该点的边际重要性高

相对边际贡献（归一化）：

$$
I_{\text{margin}}(t_i) = \frac{\Delta P(t_i)}{P_{\text{full}}} \times 100\%
$$

---

## 配置文件设计

### 新增配置文件列表

在`configs/`目录下新增3个YAML文件：

1. `experiment_EMT_Part1_setting4_ablation_remove_8h.yaml`
2. `experiment_EMT_Part1_setting4_ablation_remove_1d.yaml`
3. `experiment_EMT_Part1_setting4_ablation_remove_3d.yaml`


### 配置文件模板（以移除8h为例）

```yaml
# experiment_EMT_Part1_ablation_remove_8h.yaml
experiment_name: "EMT_Part1_setting2_Ablation_Remove8h"
description: "Ablation study: remove 8h timepoint to assess its marginal contribution"

# Reference existing data and model configs
data_config: "data_EMT_Cook_with_label.yaml"
model_config: "models_default.yaml"
analyzer_config: "analyzer_default.yaml"

# Data setting (inline override)
data_setting:
  name: "ablation_remove_8h"
  time_points: ["0d", "1d", "3d", "7d"]  # 移除8h
  total_cells: 8974
  balance_strategy: "total"
  min_cells_required: 1000

# Training configuration
training:
  device: "cuda"
  batch_size: 256
  epochs: 200
  early_stopping_patience: 30

# Models to train (只训练SB_MLPlus以节省时间)
models_to_train:
  - sb_mlplus

# Evaluation
evaluation:
  start_timepoint: "0d"
  end_timepoint: "7d"  
```

**关键修改点**：
- `time_points`：移除对应的中间时间点（8h, 1d, 或 3d）
- `end_timepoint`：始终保持为"7d"（Part1的终点不变）
- `models_to_train`：仅训练`sb_mlplus`（最优模型），节省计算资源

---

## 实施步骤详解

### 步骤1：计算采样参数（可选）

虽然系统会自动均分，但建议先验证：

```bash
python step0_calculate_data_split_params.py \
    --data_config configs/data_EMT_Cook_with_label.yaml \
    --n_timepoints 4 \
    --total_cells 8974 \
    --bottleneck_ratio 0.8
```

**输出示例**：
```
Time points: 4
Total cells: 8974
Cells per timepoint: 2243
Bottleneck cells: 1794
```

### 步骤2：创建3个配置文件

手动或脚本生成上述3个YAML文件，确保：
- `time_points`正确移除对应的中间时间点
- `total_cells`统一为8974
- `end_timepoint`始终为"7d"（Part1终点不变）

### 步骤3：依次训练3个消融模型

```bash
# 消融1：移除8h
bash step1_run_experiment_EMT.sh experiment_EMT_Part1_setting4_ablation_remove_8h.yaml

# 消融2：移除1d
bash step1_run_experiment_EMT.sh experiment_EMT_Part1_setting4_ablation_remove_1d.yaml

# 消融3：移除3d
bash step1_run_experiment_EMT.sh experiment_EMT_Part1_setting4_ablation_remove_3d.yaml

```

**预计时间**：每个约3小时，总计9小时（可并行）

### 步骤4：聚合结果并分析边际贡献

新增脚本：`Experiments/exp4_ablation/analyze_marginal_contribution.py`

**功能**：
1. 读取4个setting的`results.json`：
   - Setting2（完整）：`OUTPUTs/.../EMT_Part1_Setting2/results.json`
   - 3个消融变体：`OUTPUTs/.../experiment_EMT_Part1_setting4_ablation_remove_*/results.json`
2. 提取每个setting的10个指标
3. 计算$\Delta P(t_i)$和$I_{\text{margin}}(t_i)$
4. 生成可视化：
   - 边际贡献条形图（每个指标一个子图）
   - 热力图（指标 × 时间点）
   - 关键时间点识别（$I_{\text{margin}} > \text{threshold}$）

**伪代码**：
```python
import json
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载数据
results = {
    'full': load_json('EMT_Part1_Setting2/results.json'),
    'remove_8h': load_json('experiment_EMT_Part1_setting4_ablation_remove_8h/results.json'),
    'remove_1d': load_json('experiment_EMT_Part1_setting4_ablation_remove_1d/results.json'),
    'remove_3d': load_json('experiment_EMT_Part1_setting4_ablation_remove_3d/results.json'),
}

# 2. 提取指标
metrics = ['mae', 'frechet_distance', 'pcc', 'wasserstein_distance', ...]
P_full = {m: results['full']['sb_mlplus']['evaluation'][m] for m in metrics}
P_ablations = {
    '8h': {m: results['remove_8h']['sb_mlplus']['evaluation'][m] for m in metrics},
    '1d': {m: results['remove_1d']['sb_mlplus']['evaluation'][m] for m in metrics},
    '3d': {m: results['remove_3d']['sb_mlplus']['evaluation'][m] for m in metrics},
}

# 3. 计算边际贡献（只计算中间点）
delta_P = {}
for tp in ['8h', '1d', '3d']:
    delta_P[tp] = {m: P_full[m] - P_ablations[tp][m] for m in metrics}

# 4. 可视化
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
for i, metric in enumerate(metrics):
    ax = axes.flatten()[i]
    values = [delta_P[tp][metric] for tp in ['8h', '1d', '3d']]
    ax.bar(['8h', '1d', '3d'], values)
    ax.set_title(f'Δ{metric}')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('marginal_contribution_by_metric.png')
```

---

## 预期结果的三种模式

### 模式1：均匀下降（削弱假设）

**数学特征**：
$$
\Delta P(t_i) \approx c, \quad \forall i \in \{1,2,3,4\}, \quad \text{std}(\Delta P) < 0.2 \cdot \text{mean}(\Delta P)
$$

**解释**：每个时间点的贡献大致相等，性能下降主要由数据量减少导致（从5点降到4点，减少20%数据）。

**生物学含义**：不存在特别关键的时间窗口，EMT过程的信息在时间上均匀分布。

**对核心假设的影响**：削弱"熵峰值时刻特别重要"的论点。

### 模式2：晚期时间点敏感（支持假设）

**数学特征**：
$$
\Delta P(t_3=3d) > 2 \cdot \text{mean}(\{\Delta P(t_1=8h), \Delta P(t_2=1d)\})
$$

其中$t_3=3d$是接近EMT峰值（7d）的晚期时间点。

**解释**：移除3d导致的性能损失显著大于移除早期时间点（8h, 1d），说明晚期中间状态包含不可替代的信息。

**生物学含义**：细胞在接近EMT峰值时的状态转换动力学更复杂，携带关于状态空间探索边界的关键信息。缺少这个锚点，模型无法正确学习从中期到峰值的转换规律。

**对核心假设的影响**：强烈支持"中间状态特别是接近峰值的时间点不可从边界推断"的论点。

### 模式3：阶段依赖（混合解释）

**数学特征**：
$$
\Delta P(t_1=8h) \gg \text{mean}(\{\Delta P(t_2=1d), \Delta P(t_3=3d)\})
$$
或
$$
\Delta P(t_2=1d) \gg \text{mean}(\{\Delta P(t_1=8h), \Delta P(t_3=3d)\})
$$

**解释**：某个特定中间时间点的重要性显著高于其他中间点，说明EMT过程的不同阶段对建模难度有非对称贡献。

**生物学含义**：
- 若8h最重要：细胞命运决定的关键事件发生在转换初期（TGF-β信号激活、转录因子级联）
- 若1d最重要：中期的基因表达重编程阶段是建模的关键
- 若3d最重要：接近峰值的表观遗传重塑和稳定化阶段更难建模（需要长时间尺度信息）

**对核心假设的影响**：部分支持，但需要结合实验7（熵演化）判断是否与熵峰值时刻一致。

---

## 定量分析指标

### 1. 边际信息贡献率

$$
I_{\text{margin}}(t_i) = \frac{\Delta P(t_i)}{P_{\text{full}}} \times 100\%
$$

这里的margin指的是目前所用的10个Evaluation metrics 再删除和不删除的差别。
**阈值定义**：关键时间点需满足
$$
I_{\text{margin}}(t_i) > 1.5 \times \text{mean}(I_{\text{margin}})
$$

**示例**：
- 若$I_{\text{margin}}(3d) = 25\%$，而其他中间点平均15%，则3d被识别为关键时间点。

### 2. 跨指标一致性

计算每个时间点在不同指标上的排名一致性：

$$
\text{Consistency}(t_i) = \frac{1}{|\mathcal{M}|} \sum_{m \in \mathcal{M}} \mathbb{1}[\text{rank}_m(t_i) \leq 2]
$$

其中$\mathcal{M}$是指标集合，$\text{rank}_m(t_i)$是时间点$t_i$在指标$m$下的重要性排名。

**解释**：若某时间点在多数指标上都排名前2，说明其重要性鲁棒。

### 3. 关键时间点集合

$$
\mathcal{T}_{\text{critical}} = \{t_i : I_{\text{margin}}(t_i) > \theta \text{ and } \text{Consistency}(t_i) > 0.6\}
$$

**应用**：在资源受限的实验设计中，优先采样$\mathcal{T}_{\text{critical}}$中的时间点。

---

## 实现复杂度评估

### 代码修改量
- **配置文件**：3个YAML（每个约50行）= 200行
- **分析脚本**：1个Python脚本（约150行）
- **总计**：约350行（其中200行是重复的配置模板）

---

## 与其他实验的关联

### 为实验7提供输入
实验7（熵演化）需要判断熵峰值时刻是否关键。实验4的结果可以：
- 如果3d被识别为最关键中间点，预测实验7会显示3d或7d附近是熵最大点
- 如果8h或1d更关键，说明早期动力学更重要，可能熵峰值在早期或中期

### 为实验5提供对照
实验5（时间打乱）测试时间因果信息的重要性。实验4的结果可以：
- 如果所有时间点同等重要（模式1），预测打乱后性能保持
- 如果存在关键时间点（模式2/3），预测打乱后性能下降

### 指导实验设计优化
如果发现某些时间点贡献极小（$I_{\text{margin}} < 5\%$），未来实验可以：
- 跳过这些时间点，节省数据收集成本
- 增加关键时间点的采样密度，提高建模精度

---
