# 实验5：时间信息解耦实验（时间打乱）- 详细实现分析

## 实验目标回顾

区分模型性能提升的两个可能来源：**时间的因果顺序信息**与**状态空间的几何覆盖**。通过破坏时间顺序但保持状态空间采样，测试模型是否真正学习了时间依赖的动力学，还是仅仅记忆了高维空间中的插值映射。

---

## 理论框架

### 信息分解

模型从训练数据中学习的信息可分解为两部分：

$$
I_{\text{total}} = I_{\text{causal}}(T) + I_{\text{spatial}}(X)
$$

其中：
- $I_{\text{causal}}(T)$：时间因果顺序提供的信息（细胞$x^{(t_j)}$在时间$t_j$后演化到$x^{(t_{j+1})}$）
- $I_{\text{spatial}}(X)$：状态空间几何结构提供的信息（细胞状态的分布和距离关系）

**正常时序组**：模型同时利用$I_{\text{causal}}$和$I_{\text{spatial}}$

**打乱时序组**：破坏因果关系，仅保留$I_{\text{spatial}}$

性能差异量化因果信息的贡献：

$$
\Delta P = P_{\text{ordered}} - P_{\text{shuffled}} \propto I_{\text{causal}}(T)
$$

### Schrödinger Bridge的理论优势检验

SB框架的核心价值主张是显式建模时间依赖的漂移场$b(x,t)$：

$$
dx = b(x,t)dt + \sigma dW_t
$$

如果时间$t$仅作为"条件标签"而非真正的动力学变量，SB退化为条件生成模型：

$$
x_{t_n} \sim p(x|x_{t_m}, \text{label}=t_n)
$$

**实验5的判别力**：
- 若打乱后性能崩溃：验证SB真正学习了时间依赖动力学
- 若打乱后性能保持：暴露SB在实际应用中可能退化为条件生成器

---

## 现有系统支持度分析

### 完全支持的功能

1. **时间编码输入**
   - SB模型已支持任意时间标签：`forward(x, t_start, t_end)`
   - 时间编码维度：64维（正弦位置编码 + MLP变换）
   - 时间标签与细胞状态解耦：可以给任意细胞分配任意时间标签

2. **模型训练流程**
   - 统一训练接口：`train_model(config)`
   - 支持自定义Dataset：可以在Dataset层实现打乱逻辑

3. **评估指标**
   - 10个标准指标已实现
   - 可扩展：轨迹连续性指标可添加到`Evaluator`类

### 需要新增的功能

1. **打乱数据集构造器**（核心新增）
   - 功能：随机配对细胞，但保持时间间隔分布
   - 实现位置：`Data/shuffled_dataset.py`
   - 代码量：约200行

2. **轨迹连续性指标**（评估扩展）
   - 功能：计算生成轨迹的平均跳跃距离
   - 实现位置：`Trainer/metrics.py`（新增函数）
   - 代码量：约50行

---

## 打乱数据集的数学定义

### 正常时序组（对照）

训练样本为连续时间点间的配对：

$$
\mathcal{D}_{\text{ordered}} = \{(x_i^{(t_j)}, x_i^{(t_{j+1})}, t_j, t_{j+1})\}_{i=1}^{N_j, j=0}^{6}
$$

其中$x_i^{(t_j)}$表示时间点$t_j$的第$i$个细胞，时间标签$(t_j, t_{j+1})$与细胞的真实时间身份一致。

### 打乱时序组（实验）

训练样本为随机配对，但约束时间间隔分布：

$$
\mathcal{D}_{\text{shuffled}} = \{(x_p^{(t_m)}, x_q^{(t_n)}, t_m, t_n)\}
$$

其中：
- $p, q$：随机采样的细胞索引（可能来自不同时间点）
- $t_m, t_n$：随机采样的时间标签（满足$\Delta t$分布约束）
- 关键：$x_p$的真实时间身份可能不是$t_m$，但模型被告知它在$t_m$

### 时间间隔分布匹配

定义真实数据的时间间隔分布：

$$
P_{\text{real}}(\Delta t) = \text{Histogram}(\{t_{j+1} - t_j : j=0,\ldots,6\})
$$

对于EMT数据：
- $\Delta t \in \{8h, 16h, 48h, 96h\}$
- 频率：$\{4/7, 1/7, 1/7, 1/7\}$（因为有4个8h间隔，1个16h，1个48h，1个96h）

打乱采样算法：
1. 从$P_{\text{real}}(\Delta t)$采样一个时间间隔$\delta$
2. 随机选择起点时间$t_m \in \{t_0, \ldots, t_6\}$
3. 计算终点时间$t_n = t_m + \delta$（若超出范围则重新采样）
4. 从时间点$t_m$的细胞池随机抽取$x_p$
5. 从时间点$t_n$的细胞池随机抽取$x_q$
6. 构造训练样本$(x_p, x_q, t_m, t_n)$

**验证**：打乱后的$\Delta t$分布应与真实分布无显著差异（KS检验$p > 0.05$）

---

## 实现设计

### 新增模块1：ShuffledDataset类

**文件**：`Data/shuffled_dataset.py`

**核心逻辑**：

```python
class ShuffledTimeSeriesDataset(Dataset):
    """
    打乱时间因果关系的数据集
    保持时间间隔分布，但破坏细胞-时间的真实对应关系
    """
    
    def __init__(
        self,
        adata: AnnData,
        time_labels: List[str],
        time_intervals: Dict[str, float],  # e.g., {"0d-8h": 8, "8h-1d": 16, ...}
        n_samples: int = 8974,
        seed: int = 42
    ):
        self.adata = adata
        self.time_labels = time_labels
        self.n_samples = n_samples
        
        # 1. 构建时间间隔分布
        self.delta_t_distribution = self._build_delta_t_distribution(time_intervals)
        
        # 2. 为每个时间点构建细胞池
        self.cell_pools = {
            t: adata[adata.obs['Ground_truth'] == t].X.toarray()
            for t in time_labels
        }
        
        # 3. 预生成所有打乱的配对
        self.shuffled_pairs = self._generate_shuffled_pairs(n_samples, seed)
    
    def _build_delta_t_distribution(self, time_intervals):
        """
        从时间间隔字典构建采样分布
        返回：(delta_t_values, probabilities)
        """
        # 统计每个间隔出现的次数
        delta_t_counts = {}
        for interval_str, hours in time_intervals.items():
            delta_t_counts[hours] = delta_t_counts.get(hours, 0) + 1
        
        # 归一化为概率
        total = sum(delta_t_counts.values())
        delta_t_values = list(delta_t_counts.keys())
        probabilities = [count / total for count in delta_t_counts.values()]
        
        return delta_t_values, probabilities
    
    def _generate_shuffled_pairs(self, n_samples, seed):
        """
        生成n_samples个打乱的(x_start, x_end, t_start, t_end)配对
        """
        np.random.seed(seed)
        pairs = []
        
        delta_t_values, probs = self.delta_t_distribution
        time_to_hours = {t: self._label_to_hours(t) for t in self.time_labels}
        
        for _ in range(n_samples):
            # 1. 采样时间间隔
            delta_t = np.random.choice(delta_t_values, p=probs)
            
            # 2. 采样起点时间
            valid_starts = [t for t in self.time_labels 
                           if time_to_hours[t] + delta_t <= max(time_to_hours.values())]
            if not valid_starts:
                continue
            t_start = np.random.choice(valid_starts)
            
            # 3. 计算终点时间
            t_end_hours = time_to_hours[t_start] + delta_t
            t_end = self._hours_to_label(t_end_hours)
            
            # 4. 随机采样细胞
            x_start = self.cell_pools[t_start][np.random.randint(len(self.cell_pools[t_start]))]
            x_end = self.cell_pools[t_end][np.random.randint(len(self.cell_pools[t_end]))]
            
            pairs.append((x_start, x_end, t_start, t_end))
        
        return pairs
    
    def __len__(self):
        return len(self.shuffled_pairs)
    
    def __getitem__(self, idx):
        x_start, x_end, t_start, t_end = self.shuffled_pairs[idx]
        return torch.tensor(x_start, dtype=torch.float32), \
               torch.tensor(x_end, dtype=torch.float32)
```

**关键设计点**：
1. 预生成所有配对（避免每次epoch重新采样，保证可复现性）
2. 时间间隔分布从配置文件读取（`time_intervals`字段）
3. 支持验证：可以导出打乱后的$\Delta t$分布与真实分布对比

### 新增模块2：轨迹连续性指标

**文件**：`Trainer/metrics.py`（新增函数）

**数学定义**：

$$
J = \frac{1}{K-1} \sum_{j=0}^{K-2} \mathbb{E}_{i=1}^{N} \left[ \|x_i^{(t_{j+1})} - x_i^{(t_j)}\|_2 \right]
$$

其中$K$是时间点数量，$N$是每个时间点的样本数。

**实现**：

```python
def compute_trajectory_continuity(
    model: nn.Module,
    initial_states: torch.Tensor,  # (N, d)
    time_grid: List[float],  # [t0, t1, ..., tK]
    device: str = 'cuda'
) -> float:
    """
    计算生成轨迹的平均跳跃距离
    
    Args:
        model: 训练好的SB模型
        initial_states: 初始状态 (N, d)
        time_grid: 时间网格点
        device: 计算设备
    
    Returns:
        average_jump_distance: 平均跳跃距离
    """
    model.eval()
    N = initial_states.shape[0]
    K = len(time_grid)
    
    # 生成完整轨迹
    trajectory = []  # List of (N, d) tensors
    x_current = initial_states.to(device)
    trajectory.append(x_current.cpu().numpy())
    
    for j in range(K - 1):
        t_start = time_grid[j]
        t_end = time_grid[j + 1]
        
        # 生成下一个时间点的状态
        with torch.no_grad():
            x_next = model.generate_trajectory(
                x_current, 
                t_start=t_start, 
                t_end=t_end,
                n_steps=10  # 少量步数即可
            )
        
        trajectory.append(x_next.cpu().numpy())
        x_current = x_next
    
    # 计算相邻时间点间的平均距离
    jump_distances = []
    for j in range(K - 1):
        diff = trajectory[j + 1] - trajectory[j]  # (N, d)
        distances = np.linalg.norm(diff, axis=1)  # (N,)
        jump_distances.append(distances.mean())
    
    return np.mean(jump_distances)
```

### 配置文件设计

**新增**：`experiment_EMT_Part1_shuffled.yaml`

```yaml
experiment_name: "EMT_Part1_Shuffled"
description: "Time-shuffled experiment: break causal order but preserve spatial coverage"

data_config: "data_EMT_Cook_with_label.yaml"
model_config: "models_default.yaml"
analyzer_config: "analyzer_default.yaml"

# 特殊标记：使用打乱数据集
data_setting:
  name: "shuffled_timeseries"
  use_shuffled_dataset: true  # 新增标志
  time_points: ["0d", "8h", "1d", "3d", "7d"]
  total_cells: 8974
  balance_strategy: "total"

# 训练配置（与Setting2相同）
training:
  device: "cuda"
  batch_size: 256
  epochs: 200
  early_stopping_patience: 30

# 只训练SB_MLPlus
models_to_train:
  - sb_mlplus

# 评估（新增轨迹连续性指标）
evaluation:
  start_timepoint: "0d"
  end_timepoint: "7d"
  compute_trajectory_continuity: true  # 新增标志
```

### 数据加载器修改

**文件**：`Data/data_loader.py`（轻量修改）

在`RealDataLoader.create_datasets()`中添加分支：

```python
def create_datasets(self, config):
    """创建训练和测试数据集"""
    
    # 检查是否使用打乱数据集
    if config.get('use_shuffled_dataset', False):
        from .shuffled_dataset import ShuffledTimeSeriesDataset
        
        train_dataset = ShuffledTimeSeriesDataset(
            adata=self.adata_train,
            time_labels=self.time_labels,
            time_intervals=self.time_intervals,
            n_samples=config['total_cells'],
            seed=42
        )
    else:
        # 原有逻辑：正常时序数据集
        train_dataset = TimeSeriesDataset(...)
    
    # 测试集始终使用正常时序（评估在真实数据上）
    test_dataset = TimeSeriesDataset(...)
    
    return train_dataset, test_dataset
```

---

## 实施步骤

### 步骤1：实现ShuffledDataset类（2小时）

1. 创建`Data/shuffled_dataset.py`
2. 实现时间间隔分布构建
3. 实现随机配对生成逻辑
4. 单元测试：验证$\Delta t$分布匹配

### 步骤2：实现轨迹连续性指标（1小时）

1. 在`Trainer/metrics.py`添加`compute_trajectory_continuity()`
2. 在`Evaluator.evaluate()`中集成该指标
3. 单元测试：验证计算正确性

### 步骤3：修改数据加载器（30分钟）

1. 在`RealDataLoader.create_datasets()`添加打乱分支
2. 从配置文件读取`use_shuffled_dataset`标志

### 步骤4：创建配置文件并训练（3小时）

```bash
# 训练打乱时序模型
bash step1_run_experiment_EMT.sh experiment_EMT_Part1_shuffled.yaml
```

### 步骤5：对比分析（1小时）

新增脚本：`Experiments/exp5_shuffle/analyze_causal_vs_spatial.py`

**功能**：
1. 加载Setting2（正常）和Shuffled的结果
2. 计算性能差异$\Delta P$
3. 分析轨迹连续性差异
4. 生成对比可视化

---

## 预期结果的三种情况

### 情况1：打乱后性能崩溃（支持假设）

**数学特征**：
$$
\frac{P_{\text{shuffled}}}{P_{\text{ordered}}} < 0.7 \quad \text{(主要指标下降超过30%)}
$$

**轨迹连续性**：
$$
J_{\text{shuffled}} > 2 \times J_{\text{ordered}}
$$

**解释**：模型高度依赖时间因果信息，打乱后无法学习平滑的状态转换轨迹。

**理论意义**：验证SB框架的核心价值——真正学习了时间依赖的动力学$b(x,t)$，而非简单的条件映射。

**生物学含义**：细胞状态转换不是静态的"起点→终点"映射，而是动态的、时间依赖的过程。缺少时间因果链，模型无法捕捉EMT的多阶段动力学（信号激活→转录响应→表观重塑）。

### 情况2：打乱后性能保持（削弱假设）

**数学特征**：
$$
\frac{P_{\text{shuffled}}}{P_{\text{ordered}}} > 0.9 \quad \text{(性能差异小于10%)}
$$

**轨迹连续性**：
$$
J_{\text{shuffled}} \approx J_{\text{ordered}}
$$

**解释**：模型主要利用状态空间的几何结构，时间标签$t$仅起到"选择映射强度"的作用。

**理论意义**：暴露SB在实际应用中可能退化为条件生成模型，其时间建模能力未被充分利用。

**对策**：需要改进模型架构或训练目标，强化时间依赖性的学习。例如：
- 增加时间正则化项：惩罚$\frac{\partial b}{\partial t}$过小
- 引入轨迹平滑性约束：最小化$\int \|\frac{dx}{dt}\|^2 dt$

### 情况3：部分指标敏感（混合解释）

**数学特征**：
- 点估计精度（MAE, PCC）：$\Delta P < 10\%$
- 分布质量（FD, Wasserstein）：$\Delta P > 30\%$
- 轨迹连续性：$J_{\text{shuffled}} \gg J_{\text{ordered}}$

**解释**：模型在空间平均意义上能找到正确的目标区域（"大致知道终点在哪"），但无法生成物理上合理的轨迹（"不知道怎么走过去"）。

**理论意义**：时间信息的作用更多体现在轨迹的全局结构而非局部预测。这仍然支持时间因果信息的必要性，但表明其贡献的层次不同。

**生物学含义**：细胞群体的平均行为（均值、方差）可能由边界状态约束，但个体细胞的演化路径需要时间信息指导。

---

## 定量分析指标

### 1. 因果信息贡献率

$$
C_{\text{causal}} = \frac{P_{\text{ordered}} - P_{\text{shuffled}}}{P_{\text{ordered}}} \times 100\%
$$

**阈值**：
- $C_{\text{causal}} > 30\%$：时间因果信息至关重要
- $10\% < C_{\text{causal}} < 30\%$：时间信息有贡献但非主导
- $C_{\text{causal}} < 10\%$：时间信息可忽略

### 2. 轨迹平滑度比

$$
R_{\text{smooth}} = \frac{J_{\text{shuffled}}}{J_{\text{ordered}}}
$$

**解释**：
- $R_{\text{smooth}} > 2$：打乱后轨迹显著不连续
- $1 < R_{\text{smooth}} < 2$：轻度影响
- $R_{\text{smooth}} \approx 1$：无影响

### 3. 时间间隔分布验证

Kolmogorov-Smirnov检验：

$$
D = \sup_x |F_{\text{real}}(x) - F_{\text{shuffled}}(x)|
$$

其中$F$是累积分布函数。

**要求**：$p\text{-value} > 0.05$，确保打乱操作仅破坏因果关系，不改变时间间隔统计。

---

## 实现复杂度评估

### 代码修改量
- **ShuffledDataset类**：约200行
- **轨迹连续性指标**：约50行
- **数据加载器修改**：约20行
- **配置文件**：约50行
- **分析脚本**：约100行
- **总计**：约420行

### 工作量估算
- ShuffledDataset开发和测试：2小时
- 轨迹连续性指标开发：1小时
- 数据加载器修改：30分钟
- 配置文件编写：30分钟
- 模型训练：3小时（GPU）
- 结果分析和可视化：1小时
- **总计**：约8小时人工时间 + 3小时计算时间

### 技术风险
- **风险等级**：低
- **主要风险点**：时间间隔分布匹配的正确性
- **缓解措施**：
  1. 单元测试验证$\Delta t$分布
  2. 可视化对比真实vs打乱的时间间隔直方图
  3. KS检验确保统计无显著差异

---

## 与其他实验的关联

### 与实验4的关系
如果实验4显示所有时间点同等重要（模式1），预测实验5中打乱后性能保持（情况2）。反之，如果实验4识别出关键时间点，预测打乱后性能下降（情况1）。

### 与实验7的关系
如果实验5显示打乱后轨迹连续性崩溃，预测实验7中打乱组无法重现熵的非单调演化（熵曲线变为单调）。

### 理论贡献
实验5是对Schrödinger Bridge框架本身的检验。如果通过（情况1），证明SB的时间建模能力确实被利用；如果失败（情况2），需要反思SB在单细胞数据上的适用性。

---

## 结论

**实施可行性**：⭐⭐⭐⭐（4/5星）

实验5需要新增约420行代码，但核心逻辑清晰，技术风险低。主要工作量在ShuffledDataset的实现和验证上。

**科学价值**：⭐⭐⭐⭐⭐（5/5星）

实验5具有最高的理论意义：
1. 区分时间因果信息与空间几何信息的贡献
2. 检验Schrödinger Bridge框架的核心价值主张
3. 为方法论改进提供明确方向

**建议**：在实验4完成后立即启动实验5，因为其结果对理解模型的学习机制至关重要。
