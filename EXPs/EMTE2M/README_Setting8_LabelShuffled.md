# Setting8: Label-Shuffled Time Series - 使用指南

## 实验目标

测试模型是否依赖**绝对时间标签**，还是能从**数据几何结构**中学习到真实的时间动力学。

### 与Setting5的区别

| 维度 | Setting 5 | Setting 8 |
|------|-----------|-----------|
| **操作对象** | 细胞-时间配对 | 时间标签 |
| **保留内容** | 时间间隔分布 | 数据几何结构 |
| **破坏内容** | 因果关系 | 绝对时间标注 |
| **评估方式** | 0d→7d（可能被质疑） | 0d→7d（更有说服力） |
| **回答问题** | 时间因果 vs 空间几何 | 绝对标签 vs 相对位置 |

### 为什么需要Setting8？

**Setting5的潜在质疑**：
> "你的模型在打乱的数据上训练，当然在打乱的数据上表现好。但这不能证明模型学到了真实的时间动力学，因为评估本身就是在乱序数据上进行的。"

**Setting8的回应**：
- 训练时：随机打乱时间标签（混淆绝对时间信息）
- 评估时：使用真实时间标签（0d→7d）
- 如果模型仍能表现良好 → 证明模型从**数据结构**中学到了真实的时间顺序，而非依赖标签

## 实验设计

### 数据处理流程

```
原始数据（5个时间点）:
├── 0d  (起点) ─────┐
├── 8h  (中间)      │
├── 1d  (中间)      ├─ 训练时排除
├── 3d  (中间)      │
└── 7d  (终点) ─────┘

训练数据（3个中间时间点）:
├── 8h 样本 → 随机标签 (可能是 1d, 3d, 或 8h)
├── 1d 样本 → 随机标签 (可能是 8h, 3d, 或 1d)
└── 3d 样本 → 随机标签 (可能是 8h, 1d, 或 3d)

评估数据:
├── 使用真实标签
└── 评估 0d → 7d 的生成能力
```

### 标签打乱策略

**随机排列（Permutation）**：
```python
原始标签: [8h, 1d, 3d]
打乱后:   [1d, 3d, 8h]  # 随机排列

示例映射:
- 真实8h的细胞 → 被标记为1d
- 真实1d的细胞 → 被标记为3d
- 真实3d的细胞 → 被标记为8h
```

**关键特性**：
1. 每个标签出现的次数大致相同（保持标签分布）
2. 标签与真实时间完全解耦
3. 模型无法从标签推断真实时间顺序

## 文件结构

```
SynthaticSCData/
├── Data/
│   └── label_shuffled_dataset.py         # Setting8数据集类（新）
├── Workers/
│   └── step2_run_exp_setting8.py         # Setting8运行脚本（新）
├── EXPs/EMTE2M/
│   ├── step2_train_setting8.sh           # Bash运行脚本
│   └── README_Setting8_LabelShuffled.md  # 本文档
└── configs/EMT_E2M/
    └── experiment_EMT_Part1_setting8_label_shuffled.yaml
```

## 使用流程

### 步骤1: 运行实验

```bash
# 在项目根目录执行
cd /home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData

# 运行Setting8实验
bash EXPs/EMTE2M/step2_train_setting8.sh
```

**训练模型**：
- `sb_mlplus` (Schrödinger Bridge)
- `batch_ot` (Batch Optimal Transport)
- `vae` (Conditional VAE)

**预计时间**：约6-9小时（3个模型，每个2-3小时）

**输出目录**：
```
/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting8_LabelShuffled/
├── results.json              # 实验结果
├── logs/
│   └── experiment.log        # 详细日志
├── checkpoints/              # 模型检查点
│   ├── sb_mlplus/
│   ├── batch_ot/
│   └── vae/
├── visualizations/           # 可视化结果
└── metrics/                  # 评估指标
```

### 步骤2: 与Setting2对比分析

```bash
# 对比Setting2（正常标签）和Setting8（打乱标签）
python -c "
import json
from pathlib import Path

# 加载结果
setting2_results = json.load(open('/path/to/Setting2/results.json'))
setting8_results = json.load(open('/path/to/Setting8_LabelShuffled/results.json'))

# 对比指标
for model in ['sb_mlplus', 'batch_ot', 'vae']:
    print(f'\n{model}:')
    s2_metrics = setting2_results['models'][model]['evaluation']
    s8_metrics = setting8_results['models'][model]['evaluation']
    
    for metric in ['test_loss', 'frechet_distance', 'mae', 'pcc']:
        s2_val = s2_metrics.get(metric, 'N/A')
        s8_val = s8_metrics.get(metric, 'N/A')
        if s2_val != 'N/A' and s8_val != 'N/A':
            degradation = ((s8_val - s2_val) / s2_val) * 100
            print(f'  {metric}: {s2_val:.4f} → {s8_val:.4f} ({degradation:+.1f}%)')
"
```

### 步骤3: 查看标签混淆矩阵

```bash
# 查看训练数据的标签映射
python Data/label_shuffled_dataset.py
```

## 代码架构

### 核心类：LabelShuffledDataset

```python
class LabelShuffledDataset(Dataset):
    """
    随机化时间标签的数据集
    
    功能：
    1. 排除边界时间点（0d, 7d）
    2. 保留中间时间点（8h, 1d, 3d）
    3. 随机打乱时间标签
    4. 保持数据结构与TimeSeriesDataset兼容
    """
    
    def __init__(
        self,
        X: np.ndarray,              # 表达矩阵
        y: np.ndarray,              # 原始时间标签
        time_labels: List[str],     # 时间标签列表
        intermediate_indices: List[int],  # 中间时间点索引
        seed: int = 42,
        shuffle_labels: bool = True
    ):
        # 1. 过滤：只保留中间时间点
        # 2. 打乱：随机排列时间标签
        # 3. 转换：转为PyTorch张量
```

### 核心函数：create_label_shuffled_datasets

```python
def create_label_shuffled_datasets(
    X_train, y_train, X_test, y_test,
    time_labels,
    start_timepoint='0d',
    end_timepoint='7d',
    seed=42
):
    """
    创建标签打乱的训练和测试数据集
    
    返回:
        - train_dataset: 标签已打乱
        - test_dataset: 标签未打乱（用于评估）
    """
```

### 运行脚本：step2_run_exp_setting8.py

```python
def run_setting8_experiment(config, logger):
    """
    运行完整的Setting8实验
    
    流程:
    1. 加载数据（Setting2作为基础）
    2. 创建标签打乱的数据集
    3. 训练3个模型
    4. 在真实标签上评估
    5. 保存结果
    """
```

## 预期结果模式

### 模式1：性能显著下降（依赖绝对标签）

**数学特征**：
```
Performance_Setting8 / Performance_Setting2 < 0.7
```

**解释**：
- 模型高度依赖绝对时间标签
- 无法从数据几何结构中学习时间顺序
- 标签打乱后失去了关键信息

**理论意义**：
- 模型的时间建模能力较弱
- 需要改进模型架构或训练策略

### 模式2：性能基本保持（学习数据几何）

**数学特征**：
```
Performance_Setting8 / Performance_Setting2 > 0.9
```

**解释**：
- 模型能从数据几何结构中学习时间顺序
- 绝对时间标签不是必需的
- 数据本身包含足够的时间信息

**理论意义**：
- 验证模型学到了真实的动力学
- 回应Setting5的质疑
- 支持"数据驱动"的时间建模

### 模式3：部分指标敏感（混合依赖）

**数学特征**：
- 点估计精度（MAE, PCC）：下降 < 10%
- 分布质量（FD, Wasserstein）：下降 > 30%

**解释**：
- 模型在局部预测上依赖数据几何
- 但在全局分布建模上需要时间标签
- 不同层次的时间信息作用不同

## 关键指标

### 1. 标签依赖度

$$
D_{\text{label}} = \frac{P_{\text{Setting2}} - P_{\text{Setting8}}}{P_{\text{Setting2}}} \times 100\%
$$

**阈值**：
- $D_{\text{label}} > 30\%$：高度依赖绝对标签
- $10\% < D_{\text{label}} < 30\%$：部分依赖
- $D_{\text{label}} < 10\%$：几乎不依赖

### 2. 标签混淆矩阵

```
         预测标签
        8h   1d   3d
真  8h  [a   b   c]
实  1d  [d   e   f]
标  3d  [g   h   i]
签
```

**理想情况**：对角线元素接近0（完全打乱）

### 3. 跨模型一致性

如果三个模型（sb_mlplus, batch_ot, vae）都表现出相同的模式（都下降或都保持），说明结论更可靠。

## 与其他实验的关联

### 与Setting5的关系

- **Setting5**：测试时间因果信息的必要性
- **Setting8**：测试绝对时间标签的必要性
- **组合解释**：
  - 如果Setting5性能下降 + Setting8性能保持 → 模型需要因果顺序，但不需要绝对标签
  - 如果Setting5性能保持 + Setting8性能下降 → 模型依赖标签，但不需要因果顺序
  - 如果两者都下降 → 模型同时依赖因果和标签
  - 如果两者都保持 → 模型主要依赖空间几何

### 与Setting2的关系

Setting2是对照组，提供正常训练的基线性能。

### 与Setting4的关系

如果Setting4显示某个时间点特别关键，而Setting8性能保持，说明模型能从数据中发现这个关键时间点，而不需要标签告诉它。

## 故障排查

### 问题1: 导入错误

```bash
# 确保Data模块已更新
python -c "from Data import LabelShuffledDataset; print('✓ Import successful')"
```

### 问题2: 训练失败

```bash
# 检查日志
cat /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting8_LabelShuffled/logs/experiment.log
```

### 问题3: 标签打乱验证

```bash
# 运行测试脚本
python Data/label_shuffled_dataset.py
```

## 理论贡献

Setting8的核心价值在于：

1. **回应质疑**：解决Setting5可能面临的"循环论证"质疑
2. **解耦分析**：区分"绝对时间标签"和"相对时间顺序"的作用
3. **验证能力**：测试模型是否真正理解数据的时间结构

## 作者

Shi Pan

## 更新日期

2024-11-24
