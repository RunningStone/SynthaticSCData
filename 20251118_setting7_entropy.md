# 实验7：生成轨迹的熵演化保真度测试 - 详细实现分析

## 实验目标

直接评估模型生成的时间轨迹是否重现真实数据的核心动力学特征：**熵的非单调演化（熵增-熵减过程）**。这是对核心科学假设的最直接验证。

---

## 理论框架

### 熵作为动力学指标

细胞群体的微分熵$H(X) = -\int p(x)\log p(x)dx$量化状态分布的不确定性：
- **熵增阶段**：细胞探索状态空间，异质性增加
- **熵减阶段**：细胞收敛到稳定状态，异质性降低

EMT过程预期呈现**倒U型熵曲线**：
$$
H(t_0) < H(t_{\text{peak}}) > H(t_n)
$$

### 核心假设检验

**假设**：边界条件不足以约束熵演化的非单调性

**预测**：
- Setting1（仅边界）：生成单调熵曲线，无峰值
- Setting2（完整轨迹）：准确重现熵峰值和非对称演化
- Setting3（含峰值点）：部分重现，但细节有偏差

---

## 现有系统支持度

### 完全支持
1. **轨迹生成**：SB模型的`generate_trajectory(x0, time_grid)`
2. **多时间点采样**：测试集包含所有时间点
3. **数值求解**：SDE求解器（Euler-Maruyama，步长0.01）

### 需新增
1. **KNN熵估计器**（约100行，可用`scikit-learn`）
2. **高斯熵估计器**（约80行，基于协方差矩阵）
3. **熵曲线分析和可视化**（约150行）

---

## 熵估计方法

### 方法1：K近邻熵估计

基于Kozachenko-Leonenko estimator：

$$
\hat{H}_{\text{KNN}} = \frac{d}{N}\sum_{i=1}^{N}\log(\rho_k(x_i)) + \log(N-1) - \psi(k) + \log(c_d)
$$

其中：
- $\rho_k(x_i)$：第$i$个样本到其第$k$近邻的欧氏距离
- $\psi$：Digamma函数
- $c_d = \pi^{d/2}/\Gamma(d/2+1)$：$d$维单位球体积
- 选择$k=5$平衡偏差-方差

**优势**：无需假设分布形式，适用于任意形状的高维分布

### 方法2：多元高斯近似

假设数据近似服从多元正态分布：

$$
\hat{H}_{\text{Gauss}} = \frac{d}{2}\log(2\pi e) + \frac{1}{2}\log\det(\tilde{\Sigma})
$$

其中$\tilde{\Sigma}$是Ledoit-Wolf收缩估计的协方差矩阵：

$$
\tilde{\Sigma} = (1-\alpha)\Sigma + \alpha\frac{\text{tr}(\Sigma)}{d}I
$$

**优势**：计算高效，在数据接近正态时精确

### 两种方法的互补

- 若两者趋势一致：增强结论可信度
- 若不一致：说明分布的非高斯性显著，需进一步分析

---

## 实现设计

### 新增模块1：熵估计器

**文件**：`Experiments/exp7_entropy/entropy_estimators.py`

**核心函数**：

```python
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
import numpy as np

def estimate_entropy_knn(X, k=5):
    """
    KNN熵估计
    Args:
        X: (N, d) 样本矩阵
        k: 近邻数
    Returns:
        entropy: 估计的微分熵
    """
    N, d = X.shape
    
    # 1. 计算k近邻距离
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    rho_k = distances[:, k]  # 第k个近邻距离
    
    # 2. 计算体积常数
    c_d = np.pi**(d/2) / np.exp(gammaln(d/2 + 1))
    
    # 3. KL估计
    H = d * np.mean(np.log(rho_k)) + np.log(N-1) - digamma(k) + np.log(c_d)
    
    return H

def estimate_entropy_gaussian(X):
    """
    高斯近似熵估计（带Ledoit-Wolf收缩）
    Args:
        X: (N, d) 样本矩阵
    Returns:
        entropy: 估计的微分熵
    """
    from sklearn.covariance import LedoitWolf
    
    N, d = X.shape
    
    # 1. Ledoit-Wolf协方差估计
    lw = LedoitWolf().fit(X)
    Sigma = lw.covariance_
    
    # 2. 计算行列式（数值稳定）
    sign, logdet = np.linalg.slogdet(Sigma)
    
    # 3. 高斯熵公式
    H = 0.5 * d * np.log(2 * np.pi * np.e) + 0.5 * logdet
    
    return H
```

### 新增模块2：熵曲线分析

**文件**：`Experiments/exp7_entropy/analyze_entropy_evolution.py`

**核心逻辑**：

```python
def compute_entropy_curve(
    model,
    initial_states,  # (N, d) 从测试集t0采样
    time_grid,  # [0, 8, 24, 72, 168] 小时
    time_labels,  # ['0d', '8h', '1d', '3d', '7d']
    method='knn'
):
    """
    计算生成轨迹的熵曲线
    """
    model.eval()
    N = initial_states.shape[0]
    K = len(time_grid)
    
    # 1. 生成完整轨迹
    trajectory = []
    x_current = initial_states
    trajectory.append(x_current.cpu().numpy())
    
    for j in range(K-1):
        with torch.no_grad():
            x_next = model.generate_trajectory(
                x_current,
                t_start=time_grid[j],
                t_end=time_grid[j+1],
                n_steps=50
            )
        trajectory.append(x_next.cpu().numpy())
        x_current = x_next
    
    # 2. 计算每个时间点的熵
    entropy_curve = []
    for X_t in trajectory:
        if method == 'knn':
            H_t = estimate_entropy_knn(X_t, k=5)
        elif method == 'gaussian':
            H_t = estimate_entropy_gaussian(X_t)
        entropy_curve.append(H_t)
    
    return np.array(entropy_curve)

def analyze_entropy_peak(entropy_curve, time_labels):
    """
    分析熵峰值特征
    """
    # 1. 检测峰值
    peak_idx = np.argmax(entropy_curve)
    peak_time = time_labels[peak_idx]
    peak_value = entropy_curve[peak_idx]
    
    # 2. 计算峰值幅度
    A = peak_value - min(entropy_curve[0], entropy_curve[-1])
    
    # 3. 检验非单调性
    is_nonmonotonic = (peak_value > entropy_curve[0]) and \
                      (peak_value > entropy_curve[-1])
    
    # 4. 计算熵变速率
    explore_rate = (peak_value - entropy_curve[0]) / peak_idx if peak_idx > 0 else 0
    collapse_rate = (entropy_curve[-1] - peak_value) / (len(entropy_curve) - peak_idx - 1)
    
    return {
        'peak_time': peak_time,
        'peak_value': peak_value,
        'amplitude': A,
        'is_nonmonotonic': is_nonmonotonic,
        'explore_rate': explore_rate,
        'collapse_rate': collapse_rate
    }
```

### 配置文件修改

在现有配置中添加熵分析标志：

```yaml
# experiment_EMT_Part1_setting*.yaml
evaluation:
  compute_entropy_evolution: true
  entropy_method: 'knn'  # 或 'gaussian'
  entropy_n_samples: 1000
```

---

## 实施步骤

### 步骤1：实现熵估计器

1. 创建`entropy_estimators.py`
2. 实现KNN和高斯两种方法
3. 单元测试：验证已知分布（如标准正态）的熵估计

### 步骤2：实现熵曲线分析

1. 创建`analyze_entropy_evolution.py`
2. 实现轨迹生成和熵计算
3. 实现峰值检测和速率分析

### 步骤3：运行分析

```bash
python Experiments/exp7_entropy/analyze_entropy_evolution.py \
    --setting1_checkpoint /path/to/Setting1/best_model.pt \
    --setting2_checkpoint /path/to/Setting2/best_model.pt \
    --setting3_checkpoint /path/to/Setting3/best_model.pt \
    --test_data /path/to/test_data.h5ad \
    --output_dir /path/to/entropy_analysis
```

### 步骤4：可视化和对比（1小时）

生成输出：
1. 熵曲线对比图（真实 vs Setting1/2/3）
2. 峰值特征对比表
3. 熵变速率对比图
4. 交叉验证图（KNN vs 高斯方法）

---

## 预期结果

### 结果1：Setting1失败捕捉非单调性

**数学特征**：
$$
H_{\text{S1}}(t) \text{ 单调变化}, \quad A_{\text{S1}} \approx 0
$$

**解释**：仅用边界训练的模型学习到简单的线性或单调插值，无法捕捉中间的探索阶段。

**验证核心假设**：边界条件不足以约束熵演化的非单调性 ✓

### 结果2：Setting3部分重现

**数学特征**：
$$
t^*_{\text{S3}} \approx t^*_{\text{real}}, \quad \text{但} \quad A_{\text{S3}} < A_{\text{real}}
$$

**解释**：给定熵最大点的信息，模型能学到探索-收敛框架，但缺少其他中间点导致细节拟合不足。

### 结果3：Setting2准确重现

**数学特征**：
$$
\text{DTW}(H_{\text{real}}, H_{\text{S2}}) < \text{DTW}(H_{\text{real}}, H_{\text{S1/S3}})
$$

**解释**：完整轨迹训练使模型学习到真实动力学过程，不仅预测终点准确，而且路径物理上合理。

---

## 定量指标

### 1. 熵曲线相似度

动态时间规整距离：
$$
\text{DTW}(H_{\text{real}}, H_{\text{gen}})
$$

或均方误差：
$$
\text{MSE} = \frac{1}{K}\sum_{j=0}^{K-1}(H_{\text{real}}(t_j) - H_{\text{gen}}(t_j))^2
$$

### 2. 峰值位置误差

$$
\Delta t^* = |t^*_{\text{gen}} - t^*_{\text{real}}|
$$

### 3. 熵变速率非对称性

$$
\text{Asymmetry} = \frac{|\dot{H}_{\text{explore}}| - |\dot{H}_{\text{collapse}}|}{|\dot{H}_{\text{explore}}| + |\dot{H}_{\text{collapse}}|}
$$

---

## 生物学解释

### Setting1单调熵减的含义

若$H(t_0) > H(t_j) > H(t_7)$，模型错误假设细胞直接从异质的上皮状态收敛到更均一的间充质状态。这忽略了EMT的分子去稳定化阶段：
- 上皮基因被抑制但间充质基因尚未充分激活
- 细胞处于"双低"状态，群体异质性增加

### Setting3重现峰值的意义

峰值时刻的数据作为"锚点"指导模型理解状态空间的可达边界。生物学上，7天时的细胞分布最分散，这些数据告诉模型"系统能探索到多远"。

### Setting2捕捉速率非对称性

若熵曲线在1-3天间显示快速上升随后缓慢下降，对应已知的EMT时间尺度：
- 早期转录响应快速但不协调（熵增快）
- 晚期表观遗传重塑缓慢但确定性强（熵减慢）

---

## 技术挑战与解决方案

### 挑战1：高维熵估计的准确性

**问题**：1000维基因表达数据，KNN和高斯方法都可能有偏

**解决方案**：
1. 使用两种方法交叉验证
2. 预先降维到50-100维（PCA或PHATE）
3. 参考文献中的超参数设置（$k=5$, Ledoit-Wolf收缩）

### 挑战2：计算效率

**问题**：KNN需要计算$N \times N$距离矩阵

**解决方案**：
1. 使用`sklearn.neighbors.NearestNeighbors`的高效实现（KD树）
2. 每个时间点采样$N=1000$个细胞（而非全部）
3. 并行计算多个setting的熵曲线

---

## 实现复杂度评估

### 代码修改量
- **熵估计器**：约180行（KNN + 高斯）
- **熵曲线分析**：约150行
- **可视化脚本**：约100行
- **总计**：约430行

### 工作量估算
- 熵估计器开发和测试：2小时
- 熵曲线分析开发：2小时
- 运行分析（3个setting）：1小时（CPU即可）
- 结果可视化和解释：1小时
- **总计**：约6小时人工时间 + 1小时计算时间

### 技术风险
- **风险等级**：中等
- **主要风险点**：高维下熵估计不稳定
- **缓解措施**：
  1. 两种方法交叉验证
  2. 预先降维
  3. 敏感性分析（不同$k$值）

---

## 与其他实验的关联

### 与实验4的关系
如果实验4识别7d为最关键时间点，预测实验7会显示7d是熵最大点。

### 与实验5的关系
如果实验5显示打乱后轨迹连续性崩溃，预测实验7中打乱组无法重现熵的非单调演化。

### 与实验6的关系
如果实验6显示插值失败，预测实验7中插值组的熵曲线与真实曲线差异显著。

---

## 科学价值

实验7是**最直观、最有说服力**的证据：
1. **现象层面**：非专业读者能直观理解"熵曲线不匹配"意味着"模型没学对"
2. **理论层面**：直接验证核心假设（边界不足以约束非单调动力学）
3. **应用层面**：熵演化是细胞状态转换的通用特征，结论可推广到其他系统

---

## 结论

**实施可行性**：⭐⭐⭐⭐（4/5星）

实验7需要新增约430行代码，主要工作量在熵估计器的实现和验证上。技术风险中等，但可通过交叉验证和降维缓解。

**科学价值**：⭐⭐⭐⭐⭐（5/5星）

实验7具有最高的科学价值和传播力：
1. 直接验证核心假设
2. 提供最直观的可视化证据
3. 连接动力学理论与生物学现象

**建议**：优先实施实验7，因为其结果对整个论证链条至关重要，且可以独立于其他实验运行（仅需现有Setting1/2/3的训练好的模型）。
