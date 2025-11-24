# 实验7：生成轨迹的熵演化保真度测试

## 实验目标

直接评估模型生成的时间轨迹是否重现真实数据的核心动力学特征：**熵的非单调演化（熵增-熵减过程）**。

这是对核心科学假设的最直接验证：边界条件不足以约束生成模型学习细胞状态转换的非单调动力学。

## 核心假设

**假设**：边界条件（起点和终点）不足以约束熵演化的非单调性

**预测**：
- **Setting1（仅边界）**：生成单调熵曲线，无峰值 ❌
- **Setting2（完整轨迹）**：准确重现熵峰值和非对称演化 ✓
- **Setting3（含峰值点）**：部分重现，但细节有偏差 ~

## 理论框架

### 熵作为动力学指标

细胞群体的微分熵量化状态分布的不确定性：

$$H(X) = -\int p(x)\log p(x)dx$$

EMT过程预期呈现**倒U型熵曲线**：

$$H(t_0) < H(t_{\text{peak}}) > H(t_n)$$

- **熵增阶段**：细胞探索状态空间，异质性增加
- **熵减阶段**：细胞收敛到稳定状态，异质性降低

## 实现设计

### 模块结构

```
exp7_entropy/
├── __init__.py                      # 模块初始化
├── entropy_estimators.py            # 熵估计器（KNN + 高斯）
├── analyze_entropy_evolution.py     # 熵曲线分析
├── run_entropy_analysis.py          # 主运行脚本
├── run_exp7.sh                      # Bash启动脚本
└── README.md                        # 本文档
```

### 1. 熵估计器 (`entropy_estimators.py`)

实现两种互补的非参数熵估计方法：

#### K近邻熵估计（KNN）

基于Kozachenko-Leonenko estimator：

$$\hat{H}_{\text{KNN}} = \frac{d}{N}\sum_{i=1}^{N}\log(\rho_k(x_i)) + \log(N-1) - \psi(k) + \log(c_d)$$

**优势**：
- 无需假设分布形式
- 适用于任意形状的高维分布
- 对非高斯数据鲁棒

**关键函数**：
```python
from Experiments.exp7_entropy import estimate_entropy_knn

H = estimate_entropy_knn(X, k=5)  # X: (N, d) numpy array
```

#### 多元高斯近似

使用Ledoit-Wolf收缩估计器改善高维协方差矩阵：

$$\hat{H}_{\text{Gauss}} = \frac{d}{2}\log(2\pi e) + \frac{1}{2}\log\det(\tilde{\Sigma})$$

其中：
$$\tilde{\Sigma} = (1-\alpha)\Sigma + \alpha\frac{\text{tr}(\Sigma)}{d}I$$

**优势**：
- 计算高效
- 在数据接近正态时精确
- 提供理论基准

**关键函数**：
```python
from Experiments.exp7_entropy import estimate_entropy_gaussian

H = estimate_entropy_gaussian(X, shrinkage=True)
```

#### 交叉验证

两种方法的结合提供鲁棒性：
- 若两者趋势一致：增强结论可信度
- 若不一致：说明分布的非高斯性显著

```python
from Experiments.exp7_entropy import estimate_entropy_both_methods

H_knn, H_gauss, H_avg = estimate_entropy_both_methods(X, k=5)
```

### 2. 熵曲线分析 (`analyze_entropy_evolution.py`)

#### 生成轨迹的熵演化

```python
from Experiments.exp7_entropy import compute_entropy_curve

# model: 训练好的SB_MLPlus模型
# initial_states: 从测试集t0采样的初始状态 (N, d)
# time_grid: 归一化时间网格 [0, t1, ..., 1]
# time_labels: 时间标签 ['0d', '8h', '1d', '3d', '7d']

entropy_curve, trajectory = compute_entropy_curve(
    model=model,
    initial_states=initial_states,
    time_grid=time_grid,
    time_labels=time_labels,
    method='knn',
    k=5,
    device='cuda'
)
```

**流程**：
1. 使用SB模型的`generate_trajectory()`方法生成完整轨迹
2. 对每个时间点的细胞群体计算熵
3. 返回熵曲线和轨迹快照

#### 真实数据的熵演化

```python
from Experiments.exp7_entropy import compute_entropy_curve_from_real_data

real_entropy, real_data = compute_entropy_curve_from_real_data(
    test_data=adata,  # AnnData对象
    time_column='Ground_truth',
    time_labels=['0d', '8h', '1d', '3d', '7d'],
    n_samples=1000,
    method='knn'
)
```

#### 峰值特征分析

```python
from Experiments.exp7_entropy import analyze_entropy_peak

peak_analysis = analyze_entropy_peak(entropy_curve, time_labels)

# 返回字典包含：
# - peak_time: 峰值时刻
# - peak_value: 峰值高度
# - amplitude: 峰值幅度（相对于边界最小值）
# - is_nonmonotonic: 是否非单调（倒U型）
# - explore_rate: 熵增速率（探索阶段）
# - collapse_rate: 熵减速率（收敛阶段）
# - asymmetry: 非对称性（探索/收敛速率比）
```

## 使用方法

### 快速开始（推荐）

使用提供的bash脚本：

```bash
# 进入实验目录
cd Experiments/exp7_entropy

# 运行分析（使用KNN方法）
bash run_exp7.sh knn cuda

# 或使用高斯方法
bash run_exp7.sh gaussian cuda

# 或同时使用两种方法
bash run_exp7.sh both cuda
```

**注意**：需要先修改`run_exp7.sh`中的路径配置：
- `DATA_PATH`: 测试数据路径
- `SETTING1_CKPT`: Setting1模型检查点
- `SETTING2_CKPT`: Setting2模型检查点
- `SETTING3_CKPT`: Setting3模型检查点（可选）

### 详细用法

直接运行Python脚本：

```bash
python run_entropy_analysis.py \
    --data_path /path/to/test_data.h5ad \
    --time_column Ground_truth \
    --time_labels 0d 8h 1d 3d 7d \
    --setting1_checkpoint /path/to/Setting1/sb_mlplus_best.pt \
    --setting2_checkpoint /path/to/Setting2/sb_mlplus_best.pt \
    --setting3_checkpoint /path/to/Setting3/sb_mlplus_best.pt \
    --method knn \
    --k 5 \
    --n_samples 1000 \
    --output_dir ./entropy_analysis_results \
    --device cuda \
    --cross_validate_methods
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_path` | 测试数据h5ad文件路径 | **必需** |
| `--time_column` | obs中时间标签列名 | `Ground_truth` |
| `--time_labels` | 有序时间标签列表 | `0d 8h 1d 3d 7d` |
| `--setting1_checkpoint` | Setting1模型检查点 | **必需** |
| `--setting2_checkpoint` | Setting2模型检查点 | **必需** |
| `--setting3_checkpoint` | Setting3模型检查点 | 可选 |
| `--method` | 熵估计方法 | `knn` |
| `--k` | KNN近邻数 | `5` |
| `--n_samples` | 从t0采样的细胞数 | `1000` |
| `--n_steps` | 时间点间积分步数 | `50` |
| `--output_dir` | 输出目录 | `./entropy_analysis_results` |
| `--device` | 计算设备 | `cuda` |
| `--cross_validate_methods` | 交叉验证KNN和高斯方法 | False |

## 输出文件

运行完成后，输出目录包含：

### 1. 可视化图表

#### `entropy_curves_comparison.png/pdf`
熵曲线对比图，显示：
- 真实数据的熵演化（蓝色实线）
- Setting1生成的熵演化（紫色虚线）
- Setting2生成的熵演化（橙色实线）
- Setting3生成的熵演化（红色点划线，如果提供）

**解读**：
- Setting1应显示单调或缺乏峰值
- Setting2应与真实曲线高度一致
- 峰值位置和幅度是关键指标

#### `peak_characteristics_comparison.png/pdf`
峰值特征对比图（2×2子图）：
1. **峰值幅度**：各setting的熵峰值高度
2. **非单调性检测**：是否呈倒U型（1=是，0=否）
3. **探索阶段速率**：熵增速率
4. **曲线相似度**：与真实曲线的MSE

#### `method_cross_validation.png`（如启用）
KNN vs 高斯方法的交叉验证：
- 每个setting一个子图
- 显示两种方法的相关性
- 高相关性（>0.9）表明估计可靠

### 2. 数值结果

#### `entropy_analysis_summary.json`
包含所有定量结果：
```json
{
  "real_entropy_curve": [5.2, 7.1, 9.3, 7.8, 5.6],
  "real_peak_analysis": {
    "peak_time": "1d",
    "peak_value": 9.3,
    "amplitude": 3.7,
    "is_nonmonotonic": true,
    "explore_rate": 2.05,
    "collapse_rate": -1.85,
    "asymmetry": 1.11
  },
  "settings": {
    "Setting1": { ... },
    "Setting2": { ... }
  }
}
```

#### `entropy_analysis_full_results.pkl`
完整结果的Python pickle文件，包含：
- 所有熵曲线
- 完整生成轨迹（每个时间点的细胞状态）
- 峰值分析详情
- 运行参数

**加载方式**：
```python
import pickle

with open('entropy_analysis_full_results.pkl', 'rb') as f:
    results = pickle.load(f)

real_curve = results['real_entropy_curve']
setting1_trajectory = results['settings_results']['Setting1']['trajectory']
```

## 预期结果

### 情况1：Setting1失败捕捉非单调性 ✓ 假设成立

**数学特征**：
$$H_{\text{S1}}(t) \text{ 单调变化}, \quad A_{\text{S1}} \approx 0$$

**表现**：
- 熵曲线单调增加或单调减少
- 无明显峰值
- `is_nonmonotonic = False`

**解释**：
仅用边界训练的模型学习到简单的线性或单调插值，无法捕捉中间的探索阶段。这直接验证了**边界条件不足以约束熵演化的非单调性**。

### 情况2：Setting3部分重现 ~ 中间信息有价值

**数学特征**：
$$t^*_{\text{S3}} \approx t^*_{\text{real}}, \quad \text{但} \quad A_{\text{S3}} < A_{\text{real}}$$

**表现**：
- 峰值位置接近真实
- 峰值幅度偏小
- `is_nonmonotonic = True`

**解释**：
给定熵最大点的信息，模型能学到探索-收敛框架，但缺少其他中间点导致细节拟合不足。

### 情况3：Setting2准确重现 ✓ 完整轨迹优越

**数学特征**：
$$\text{DTW}(H_{\text{real}}, H_{\text{S2}}) < \text{DTW}(H_{\text{real}}, H_{\text{S1/S3}})$$

**表现**：
- 熵曲线形状、峰值位置、幅度都接近真实
- MSE显著低于其他setting
- 探索/收敛速率非对称性被捕捉

**解释**：
完整轨迹训练使模型学习到真实动力学过程，不仅预测终点准确，而且**路径物理上合理**。

## 定量指标

### 1. 熵曲线相似度

**均方误差（MSE）**：
$$\text{MSE} = \frac{1}{K}\sum_{j=0}^{K-1}(H_{\text{real}}(t_j) - H_{\text{gen}}(t_j))^2$$

**动态时间规整距离（DTW）**：
对时间轴的轻微错位鲁棒，更符合生物学轨迹的不确定性。

### 2. 峰值位置误差

$$\Delta t^* = |t^*_{\text{gen}} - t^*_{\text{real}}|$$

理想情况：$\Delta t^* = 0$（峰值位置完全匹配）

### 3. 熵变速率非对称性

$$\text{Asymmetry} = \frac{|\dot{H}_{\text{explore}}|}{|\dot{H}_{\text{collapse}}|}$$

真实EMT数据通常显示$\text{Asymmetry} \neq 1$（探索和收敛速率不同）

## 生物学解释

### Setting1单调熵减的含义

若$H(t_0) > H(t_j) > H(t_7)$，模型错误假设细胞直接从异质的上皮状态收敛到更均一的间充质状态。

这**忽略了EMT的分子去稳定化阶段**：
- 上皮基因被抑制但间充质基因尚未充分激活
- 细胞处于"双低"状态，群体异质性增加
- 熵在中间阶段上升

### Setting3重现峰值的意义

峰值时刻的数据作为"**锚点**"指导模型理解状态空间的可达边界。

生物学上，7天时的细胞分布最分散，这些数据告诉模型"系统能探索到多远"。

### Setting2捕捉速率非对称性

若熵曲线在1-3天间显示快速上升随后缓慢下降，对应已知的EMT时间尺度：
- **早期转录响应快速但不协调**（熵增快）
- **晚期表观遗传重塑缓慢但确定性强**（熵减慢）

## 技术细节

### 高维熵估计的挑战

**问题**：1000维基因表达数据，熵估计可能有偏

**解决方案**：
1. 使用两种方法交叉验证
2. 可选：预先降维到50-100维（PCA或PHATE）
3. 参考文献中的超参数（$k=5$, Ledoit-Wolf收缩）

### 计算效率

**KNN复杂度**：$O(N \log N)$使用KD树优化

**优化策略**：
- 每个时间点采样$N=1000$个细胞（而非全部）
- 并行计算多个setting的熵曲线
- 使用`sklearn.neighbors.NearestNeighbors`的高效实现

### 数值稳定性

- 对数计算前添加小常数避免`log(0)`
- 协方差矩阵添加正则化项避免奇异
- 使用`slogdet`而非`det`避免数值溢出

## 与其他实验的关联

- **与实验4的关系**：如果实验4识别7d为最关键时间点，预测实验7会显示7d是熵最大点
- **与实验5的关系**：如果实验5显示打乱后轨迹连续性崩溃，预测实验7中打乱组无法重现非单调演化
- **与实验6的关系**：如果实验6显示插值失败，预测实验7中插值组的熵曲线与真实曲线差异显著

## 科学价值

实验7是**最直观、最有说服力**的证据：

1. **现象层面**：非专业读者能直观理解"熵曲线不匹配"意味着"模型没学对"
2. **理论层面**：直接验证核心假设（边界不足以约束非单调动力学）
3. **应用层面**：熵演化是细胞状态转换的通用特征，结论可推广到其他系统

## 工作量评估

- **代码实现**：~430行（已完成）
- **运行时间**：约1小时（3个setting，GPU）
- **分析时间**：约1小时（生成图表、解释结果）
- **总计**：约2小时

## 故障排查

### 问题1：ImportError

**症状**：`ModuleNotFoundError: No module named 'Experiments.exp7_entropy'`

**解决**：
```bash
# 确保在项目根目录
cd /path/to/SynthaticSCData

# 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 或者安装为可编辑包
pip install -e .
```

### 问题2：CUDA Out of Memory

**症状**：`RuntimeError: CUDA out of memory`

**解决**：
```bash
# 减少采样数量
python run_entropy_analysis.py ... --n_samples 500

# 或使用CPU
python run_entropy_analysis.py ... --device cpu
```

### 问题3：Checkpoint加载失败

**症状**：`KeyError: 'model_state_dict'`

**解决**：
检查点文件格式可能不同，脚本已处理三种常见格式：
- `checkpoint['model_state_dict']`
- `checkpoint['state_dict']`
- 直接的state_dict

如仍失败，手动检查检查点内容：
```python
import torch
ckpt = torch.load('path/to/checkpoint.pt')
print(ckpt.keys())
```

## 参考文献

1. Kozachenko, L. F. & Leonenko, N. N. (1987). Sample estimate of the entropy of a random vector. *Problems of Information Transmission*.

2. Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*.

3. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E*.

## 许可证

MIT License
