# Experiment 6: 合成中间状态对照实验

## 实验目标

测试边界数据通过几何插值能否达到真实中间状态的效果，区分"**状态空间采样密度**"与"**真实动力学信息**"的贡献。

## 核心假设

**零假设 H₀**: 中间状态可从边界状态通过几何插值重构

$$\tilde{x}^{(t_k)} = \lambda_k x^{(t_0)} + (1-\lambda_k) x^{(t_n)}, \quad \lambda_k = \frac{t_n - t_k}{t_n - t_0}$$

**备择假设 H₁**: 真实中间状态包含不可插值的信息

## 实验设计

### 数据构造

1. **边界点** (真实数据): 0d, 7d (各750个细胞)
2. **中间点** (插值数据): 8h, 1d, 3d (各750个细胞)
3. **总样本量**: 3,750个细胞 (匹配Setting2)

### 线性插值公式

对每对边界细胞 $(x_i^{(0d)}, x_i^{(7d)})$，在中间时间点生成：

$$\tilde{x}_i^{(t_k)} = \lambda_k x_i^{(0d)} + (1-\lambda_k) x_i^{(7d)}$$

**权重示例** (t₀=0h, tₙ=168h):
- 8h: λ = 0.952
- 1d (24h): λ = 0.857
- 3d (72h): λ = 0.571

## 评估指标

### 1. 插值有效性指数 (IEI)

$$\text{IEI}(t_k) = 1 - \frac{E_{\text{interp}}(t_k)}{E_{\text{Setting1}}(t_k)}$$

- **IEI > 0.85**: 插值有效，问题在于数据覆盖
- **IEI < 0.7**: 插值失败，真实状态包含不可插值信息

### 2. 残差结构化指数 (RSI)

$$\text{RSI} = \frac{\sum_{i=1}^{10} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

前10个PCA主成分解释的方差比例

- **RSI > 0.7**: 残差包含系统性结构，插值失败
- **RSI < 0.3**: 残差接近随机噪声，插值成功

## 文件结构

```
exp6_interpolation/
├── README.md                           # 本文件
├── generate_interpolated_data.py       # 生成插值数据
├── analyze_interpolation_quality.py    # 分析插值质量
├── run_experiment6.py                  # 完整实验流程
└── run_experiment6.sh                  # 便捷启动脚本
```

## 使用方法

### 方法1: 使用Shell脚本 (推荐)

```bash
# 运行完整流程
bash run_experiment6.sh

# 只生成数据
bash run_experiment6.sh --data-only

# 只训练模型
bash run_experiment6.sh --train-only

# 只分析结果
bash run_experiment6.sh --analysis-only
```

### 方法2: 使用Python脚本

```bash
# 完整流程
python run_experiment6.py \
    --input_data /path/to/EMT_data.h5ad \
    --output_dir /path/to/output \
    --n_samples 750

# 只生成数据
python generate_interpolated_data.py \
    --input /path/to/EMT_data.h5ad \
    --output /path/to/interpolated_data.h5ad \
    --n_samples 750

# 只训练模型 (使用已有的插值数据)
python run_experiment6.py \
    --input_data /path/to/EMT_data.h5ad \
    --skip_data_generation

# 只分析结果
python run_experiment6.py \
    --input_data /path/to/EMT_data.h5ad \
    --analysis_only
```

## 实验流程

### Step 1: 生成插值数据 (~1小时)

```bash
python generate_interpolated_data.py \
    --input /home/pan/Experiments/DATA/EMT_Cook/adata_hvg.h5ad \
    --output /path/to/output/interpolated_data.h5ad \
    --n_samples 750
```

**输出**:
- `interpolated_data.h5ad`: 包含真实边界点和插值中间点的数据集

### Step 2: 训练模型 (~15小时GPU)

使用标准训练流程:

```bash
bash ../../step1_run_experiment_EMT.sh experiment_EMT_Part1_setting6_interpolated.yaml
```

**训练模型**:
- SB MLPlus
- Batch OT
- VAE
- OT
- SB

### Step 3: 分析结果 (~2小时)

分析脚本会自动:
1. 计算IEI和RSI指标
2. 生成可视化图表
3. 创建综合报告

## 预期结果

### 情况1: 插值有效 (削弱假设)

- **IEI > 0.85**: 插值数据训练的模型接近Setting2性能
- **RSI < 0.3**: 残差接近随机噪声
- **解释**: 问题在于数据覆盖不足，非动力学复杂性

### 情况2: 插值失败 (支持假设)

- **IEI < 0.7**: 插值数据训练的模型仍接近Setting1水平
- **RSI > 0.7**: 残差包含系统性信息
- **解释**: 真实中间状态包含不可插值的探索动力学

### 情况3: 部分有效 (混合)

- 早期时间点插值效果好，晚期失败
- **解释**: EMT不同阶段对插值敏感度不同

## 输出文件

```
EMT_Part1_Setting6/
├── interpolated_data.h5ad                    # 插值数据集
├── checkpoints/                              # 模型检查点
│   ├── sb_mlplus_best.pt
│   ├── batch_ot_best.pt
│   └── ...
├── results.json                              # 评估指标
├── analysis/
│   ├── per_timepoint_metrics.png             # 每个时间点的指标
│   ├── residual_structure_analysis.png       # 残差结构分析
│   └── interpolation_quality_report.txt      # 综合报告
└── logs/
    └── experiment.log
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_samples_per_timepoint` | 750 | 每个时间点的细胞数 |
| `boundary_timepoints` | ["0d", "7d"] | 边界时间点 |
| `intermediate_timepoints` | ["8h", "1d", "3d"] | 中间时间点 |
| `rsi_n_components` | 10 | RSI计算的主成分数 |

## 理论洞察

若插值失败，说明EMT是**弱约束系统**：

$$\begin{cases}
dx/dt = f(x,t) \\
x(0) = x_0, \quad x(T) = x_T
\end{cases}$$

边界值问题有多个解，需中间条件 $x(t_k)$ 打破简并。

## 与其他实验的关系

- **实验4 (消融)**: 识别关键时间点
- **实验5 (打乱)**: 区分时间因果信息
- **实验6 (插值)**: 排除"简单几何就足够"的解释 ← **当前**
- **实验7 (熵演化)**: 直接验证动力学保真度

## 实现复杂度

- **代码量**: ~430行
- **人工时间**: ~5小时
- **计算时间**: 15小时GPU (可并行)
- **技术风险**: 低

## 可行性评分

- **可行性**: ⭐⭐⭐⭐ (4/5星)
- **科学价值**: ⭐⭐⭐⭐ (4/5星)

## 故障排除

### 问题1: 插值数据生成失败

**原因**: 边界时间点数据不足
**解决**: 检查原始数据中0d和7d的细胞数量

### 问题2: 训练过程中内存不足

**原因**: 批次大小过大
**解决**: 在配置文件中减小`batch_size`

### 问题3: 模型性能异常

**原因**: 插值权重计算错误
**解决**: 检查`time_to_hours`映射是否正确

## 参考文献

1. 线性插值理论
2. 残差分析方法
3. Schrödinger Bridge理论

## 联系方式

如有问题，请联系: Shi Pan
