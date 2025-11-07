# Neural ODE 迁移说明

## 问题背景

### 线性插值的问题
原始的调制插值方法：`X_gen(t) = (1-α)*X_t0 + α*X_t1 + noise`

**发现的问题**：
- 生成数据的熵在中间时间段（~100-130h）显著降低
- 从熵时间线图可以看到，生成细胞（红色）的熵值在中间阶段下降到~450，而真实细胞（蓝色）保持在~465-472
- 这表明线性插值导致**过度平滑**，丢失了数据的多样性

### 评估指标对比

**线性插值结果** (`/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/naive_compare/`):
```
Frechet Distance: 21070.5787
MAE: 44.9075
Mean PCC: -0.0036
Test Entropy: 489.0813
Generated Entropy: 456.2004
Entropy Difference: 32.8809  ← 显著差异
```

## 解决方案：Neural ODE

### 为什么使用 Neural ODE？

1. **学习真实动态**：Neural ODE 学习细胞状态转换的真实轨迹，而不是简单的线性插值
2. **保持多样性**：通过学习复杂的非线性动态，避免过度平滑
3. **物理意义**：ODE 描述了状态随时间的连续演化，更符合生物学过程

### Neural ODE 原理

```python
# 定义 ODE 函数: dx/dt = f(x, t)
class ODEFunc(nn.Module):
    def forward(self, t, x):
        # 神经网络学习 f(x, t)
        return self.net(concat([x, t]))

# 求解 ODE 得到轨迹
trajectory = odeint(ode_func, x_start, time_points)
```

**训练过程**：
1. 使用真实的细胞对 (X_start, X_end, t_start, t_end) 训练
2. 最小化终点预测误差：`loss = ||ODE(X_start, t_end) - X_end||²`
3. 学习到的 ODE 函数可以生成任意时间点的状态

## 代码重构

### 删除的文件
- ❌ `Data/data_generator.py` - 旧的线性插值生成器
- ❌ `example_neural_ode_workflow.py` - 临时测试文件
- ❌ `QUICKSTART.md` - 过时的快速开始指南
- ❌ `README_CONTINUOUS_TIME.md` - 过时的文档

### 新增的文件
- ✅ `Data/neural_ode_generator.py` - Neural ODE 数据生成器
  - `ODEFunc`: ODE 函数网络
  - `NeuralODETrainer`: ODE 训练器
  - `NeuralODEDataGenerator`: 完整的数据生成流程
  - `create_neural_ode_emt_generator`: EMT 数据集默认配置

### 更新的文件
- 📝 `Data/__init__.py` - 导出 Neural ODE 相关类
- 📝 `example_emt_workflow.py` - 完全重写，基于 Neural ODE
- 📝 `requirements.txt` - 添加 `torchdiffeq>=0.2.3`
- 📝 `pyproject.toml` - 添加依赖，更新版本到 0.2.0

### 保留的文件
- ✅ `Data/dataset_builder.py` - 数据集构建器（无需修改）
- ✅ `Model/vae_model.py` - VAE 模型
- ✅ `Trainer/trainer.py` - 训练器
- ✅ `Analyser/entropy_metrics.py` - 熵分析（已修复排序问题）
- ✅ `Analyser/real_data_metrics.py` - 评估指标

## 使用方法

### 环境配置

```bash
# 使用 uv 同步环境
cd /home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData
uv sync
```

### 运行完整工作流

```bash
# 使用 uv 运行
uv run python example_emt_workflow.py \
    --n_hvg 100 \
    --cells_per_label 2000 \
    --granularity 1.0 \
    --epochs 10 \
    --ode_epochs 50 \
    --device cuda
```

### 参数说明

- `--n_hvg`: HVG 数量（默认 100）
- `--cells_per_label`: 每个时间标签采样的细胞数（默认 2000）
- `--granularity`: 时间颗粒度，小时（默认 1.0）
- `--epochs`: VAE 训练轮数（默认 100）
- `--ode_epochs`: Neural ODE 训练轮数（默认 50）
- `--device`: 设备（cuda 或 cpu）
- `--output_dir`: 输出目录（默认 `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/neural_ode_compare`）

## 工作流程

### Step 1: 生成连续时间数据（Neural ODE）
```
1. 加载真实 AnnData 数据
2. 采样细胞，计算 HVG
3. 训练 Neural ODE（学习细胞状态转换动态）
4. 使用 ODE 生成中间时间点的细胞状态
5. 保存为 h5ad 文件
```

### Step 2: 数据质量分析
```
1. 计算每个时间片的熵
2. 绘制熵时间线图
3. 对比真实 vs 生成细胞的熵
```

### Step 3: 创建训练/测试数据集
```
1. 加载连续时间数据
2. 按策略划分训练/测试集
3. 创建 PyTorch DataLoader
```

### Step 4: 训练 VAE 模型
```
1. 初始化 VAE 模型
2. 训练模型
3. 保存最佳模型
```

### Step 5: 评估模型
```
1. 计算 Frechet Distance
2. 计算 MAE
3. 计算 PCC
4. 计算熵差异
5. 保存评估结果
```

## 预期改进

### 熵分布
- **线性插值**：中间时间段熵值显著下降（~450）
- **Neural ODE**：预期保持更平滑的熵分布，接近真实数据

### 数据质量
- **多样性**：Neural ODE 学习真实动态，避免过度平滑
- **生物学意义**：ODE 描述连续演化过程，更符合细胞分化
- **泛化能力**：可以生成训练时未见过的中间时间点

## 输出对比

### 线性插值输出
```
/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/naive_compare/
├── continuous_time_data.h5ad
├── quality_analysis_fixed/
│   ├── entropy_timeline.png  ← 显示中间时段熵值下降
│   └── entropy_comparison_real_vs_generated.png
└── model/
    └── evaluation_results.json
```

### Neural ODE 输出
```
/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/neural_ode_compare/
├── continuous_time_data_ode.h5ad
├── quality_analysis/
│   ├── entropy_timeline.png  ← 预期更平滑
│   └── entropy_comparison_real_vs_generated.png
└── model/
    └── evaluation_results.json
```

## 技术细节

### Neural ODE 网络结构
```python
ODEFunc:
  Input: [x (100-dim), t (1-dim)] → 101-dim
  Hidden: [256, 256] with Tanh activation
  Output: dx/dt (100-dim)
```

### 训练配置
- **优化器**: Adam (lr=1e-3)
- **批量大小**: 64
- **训练对**: 8000 对（相邻时间点的细胞对）
- **梯度裁剪**: max_norm=1.0
- **ODE 求解器**: dopri5（自适应步长 Runge-Kutta）

### 时间映射
```
0d   → 0h
8h   → 8h
1d   → 24h
3d   → 72h
7d   → 168h

总共生成 169 个时间点（0-168h，每小时一个）
```

## 依赖库

新增依赖：
- `torchdiffeq>=0.2.3` - Neural ODE 求解器
- `scanpy>=1.9.0` - 单细胞数据分析
- `anndata>=0.8.0` - 单细胞数据格式
- `pandas>=1.3.0` - 数据处理

## 常见问题

### Q1: Neural ODE 训练很慢？
A: 可以减少 `--ode_epochs` 或增加 `--granularity` 来加速。

### Q2: 内存不足？
A: 减少 `--cells_per_label` 或 `--n_hvg`。

### Q3: 如何调整 ODE 网络结构？
A: 修改 `neural_ode_generator.py` 中的 `ode_hidden_dims` 参数。

### Q4: 如何使用 CPU 训练？
A: 添加 `--device cpu` 参数（会很慢）。

## 下一步

1. ✅ 运行完整工作流，对比线性插值和 Neural ODE 的结果
2. ⏳ 分析熵时间线图，验证 Neural ODE 是否解决了熵下降问题
3. ⏳ 调整 ODE 训练参数，优化生成质量
4. ⏳ 扩展到其他单细胞数据集

## 参考文献

- Chen et al. "Neural Ordinary Differential Equations" (NeurIPS 2018)
- Grathwohl et al. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" (ICLR 2019)
