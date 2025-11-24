# 实验6：合成中间状态对照实验 - 详细实现分析

## 实验目标

测试边界数据通过几何插值能否达到真实中间状态的效果，区分"**状态空间采样密度**"与"**真实动力学信息**"的贡献。

---

## 理论框架

### 插值假设检验

**零假设$H_0$**：中间状态可从边界状态通过几何插值重构

$$
\tilde{x}^{(t_k)} = \lambda_k x^{(t_0)} + (1-\lambda_k) x^{(t_n)}, \quad \lambda_k = \frac{t_n - t_k}{t_n - t_0}
$$

**备择假设$H_1$**：真实中间状态包含不可插值的信息，残差$R_k = x_{\text{real}}^{(t_k)} - \tilde{x}^{(t_k)}$包含系统性结构。

### 判别力

- 若插值成功：支持OT几何观点，问题在于数据稀疏
- 若插值失败：支持动力学观点，真实轨迹包含探索性绕行

---

## 现有系统支持度

### 完全支持
1. 边界数据访问（Setting1）
2. 模型训练流程
3. 10个标准评估指标

### 需新增
1. **插值数据生成器**（约150行）
2. **中间时间点专项误差分析**（约100行）

---

## 线性插值定义

对每对边界细胞$(x_i^{(0d)}, x_i^{(7d)})$，在中间时间点生成：

$$
\tilde{x}_i^{(t_k)} = \lambda_k x_i^{(0d)} + (1-\lambda_k) x_i^{(7d)}
$$

**权重示例**（$t_0=0h$，$t_n=168h$）：
- 8h：$\lambda = 0.952$
- 1d (24h)：$\lambda = 0.857$
- 3d (72h)：$\lambda = 0.571$

**数据集构造**：
- 边界点：真实数据（0d, 7d）
- 中间点：插值数据（8h, 1d, 3d）
- 每点约750个细胞，总计约3750个（匹配Setting2）

---

## 实现设计

### 插值数据生成器

**文件**：`Experiments/exp6_interpolation/generate_interpolated_data.py`

**核心函数**：
```python
def generate_linear_interpolated_data(
    adata_boundary,  # Setting1数据（0d和7d）
    time_labels=["0d", "8h", "1d", "3d", "7d"],
    n_samples_per_timepoint=750
):
    # 1. 提取边界数据
    X_t0 = adata[adata.obs['Ground_truth']=='0d'].X
    X_tn = adata[adata.obs['Ground_truth']=='7d'].X
    
    # 2. 随机配对
    # 3. 线性插值生成中间点
    # 4. 构造新AnnData对象
    return adata_interpolated
```

### 配置文件

**新增**：`experiment_EMT_Part1_interpolated.yaml`

```yaml
experiment_name: "EMT_Part1_Interpolated"
data_setting:
  time_points: ["0d", "8h", "1d", "3d", "7d"]
  total_cells: 8974
  data_source_override: "/path/to/interpolated_data.h5ad"
models_to_train: [sb, sb_mlplus, ot, vae, batch_ot]
```

---

## 实施步骤

1. **生成插值数据**（1小时）
2. **创建配置文件**（30分钟）
3. **训练5个模型**（15小时GPU，可并行）
4. **专项误差分析**（2小时）

---

## 预期结果

### 情况1：插值有效（削弱假设）
- $P_{\text{interp}} / P_{\text{Setting2}} > 0.85$
- 残差结构化指数 < 0.3（接近随机噪声）
- **解释**：问题在于数据覆盖不足，非动力学复杂性

### 情况2：插值失败（支持假设）
- $P_{\text{interp}} / P_{\text{Setting2}} < 0.7$
- 残差结构化指数 > 0.7（系统性信息）
- **解释**：真实中间状态包含不可插值的探索动力学

### 情况3：部分有效（混合）
- 早期时间点插值效果好，晚期失败
- **解释**：EMT不同阶段对插值敏感度不同

---

## 定量指标

### 插值有效性指数
$$
\text{IEI}(t_k) = 1 - \frac{E_{\text{interp}}(t_k)}{E_{\text{Setting1}}(t_k)}
$$

### 残差结构化指数
$$
\text{RSI} = \frac{\sum_{i=1}^{10} \lambda_i}{\sum_{i=1}^{d} \lambda_i}
$$
（前10个PCA主成分解释的方差比例）

---

## 实现复杂度

- **代码量**：约430行
- **人工时间**：约5小时
- **计算时间**：15小时GPU（可并行）
- **技术风险**：低

---

## 理论洞察

若插值失败，说明EMT是**弱约束系统**：边界值问题有多个解，需中间条件唯一确定。数学上：

$$
\begin{cases}
dx/dt = f(x,t) \\
x(0) = x_0, \quad x(T) = x_T
\end{cases}
$$

在弱约束下可能有多解，需$x(t_k)$打破简并。

---

## 结论

**可行性**：⭐⭐⭐⭐（4/5星）  
**科学价值**：⭐⭐⭐⭐（4/5星）

实验6通过排除"简单几何就足够"的竞争解释，增强论证完整性。建议在实验4和7完成后实施。
