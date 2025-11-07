# 基于薛定谔桥的细胞状态转换合成数据集系统设计

## 一、核心思路与数学逻辑

本项目旨在验证：学习完整动力学轨迹比仅学习起止点分布具有更强泛化能力。最优传输（OT）只约束边界 $\min_{\pi \in \Pi(\mu_0, \mu_T)} \int c(\mathbf{x}, \mathbf{y}) d\pi$，而薛定谔桥（SB）最小化整个路径偏离 $\min_{\{\mu_t\}} \int_0^T \text{KL}(\mu_t \| \gamma_t) dt$。

我们在 $d=50$ 维空间建模三态演化：初态 $\mu_0 = \mathcal{N}(\mathbf{0}, \sigma_{init}^2 \mathbf{I})$，高熵态 $\mu_{peak} = \mathcal{N}(\boldsymbol{\mu}_{peak}, \mathbf{U}\Lambda_{peak}\mathbf{U}^\top)$ 具有低秩协方差（$r \ll 50$ 个大特征值），终态 $\mu_T = \sum_i \alpha_i \mathcal{N}(\boldsymbol{\mu}_i, \Sigma_i)$ 为多模混合。轨迹通过求解 Fokker-Planck 方程 $\partial_t p_t = \nabla \cdot [D \nabla p_t + p_t \nabla U(\mathbf{x}, t)]$ 生成，势能 $U(\mathbf{x}, t)$ 在三阶段（平衡、熵增、坍缩）具有不同形式。

泛化能力通过路径信息增益 $\Delta \mathcal{L} = \mathcal{L}_{test}^{OT} - \mathcal{L}_{test}^{SB}$ 量化，在几何外推（参数超出训练范围）、拓扑外推（模态数变化）、时间尺度外推三种场景下测试。

## 二、模块设计与数据流

### 2.1 数据生成模块

**分布参数化子模块**
- 输入：维度 $d$，初态方差 $\sigma_{init}^2$，高熵态参数 $(r, \lambda_{low}, \lambda_{high}, \boldsymbol{\mu}_{peak})$，终态参数 $(N_{modes}, \{\boldsymbol{\mu}_i, \alpha_i\}, \Delta)$
- 输出：三态参数字典 $\{\mu_0, \mu_{peak}, \mu_T\}$ 及其协方差矩阵
- 逻辑：初态各向同性，高熵态通过 $\Sigma_{peak} = \mathbf{U} \text{diag}(\lambda_1,\ldots,\lambda_r,\epsilon,\ldots,\epsilon) \mathbf{U}^\top$ 构造低秩结构，终态多模中心在超球面分布

**势能函数构造子模块**
- 输入：分布参数，时间节点 $(t_1, t_{peak}, t_2, T)$，强度参数 $(\alpha(t), \beta(t), \gamma(t))$
- 输出：势能函数对象 $U(\mathbf{x}, t)$ 及其梯度 $\nabla_{\mathbf{x}} U$
- 逻辑：$t \in [0,t_1]$ 调和势 $U = \frac{1}{2\sigma_{init}^2}\|\mathbf{x}\|^2$；$t \in [t_1,t_2]$ 多模吸引子 $U = -\alpha(t) \log[\sum_i \exp(-\|\mathbf{x}-\boldsymbol{\mu}_i(t)\|^2/2\sigma_i^2)] + \beta(t) \mathbf{x}^\top \mathbf{Q}(t) \mathbf{x}$ 其中 $\mathbf{Q}(t)$ 低秩；$t \in [t_2,T]$ 双井 $U = -\gamma(t) \sum_i \exp(-\|\mathbf{x}-\boldsymbol{\mu}_i\|^2/2\sigma_i^2)$

**轨迹采样子模块**
- 输入：势能函数 $U(\mathbf{x}, t)$，扩散系数 $D$，时间网格 $\{t_n\}_{n=0}^N$，细胞数 $N_{cells}$
- 输出：轨迹张量 $(N_{cells}, N_{time}, d)$，时间戳数组
- 逻辑：Euler-Maruyama 求解 $\mathbf{X}_{t+\Delta t} = \mathbf{X}_t - D \nabla U(\mathbf{X}_t, t) \Delta t + \sqrt{2D\Delta t} \boldsymbol{\xi}_t$，初值从 $\mu_0$ 采样

**数据集构造子模块**
- 输入：轨迹族配置（参数变化范围），训练集大小 $K_{train}$，测试集大小 $K_{test}$，外推类型
- 输出：训练集和测试集字典（轨迹、分布参数、元数据），统计信息（熵演化、Wasserstein 距离）
- 逻辑：训练集从配置空间均匀采样，测试集根据外推类型选择域外参数，计算每条轨迹的 $H(\mu_t)$ 和 $W_2(\mu_t, \mu_{t+\Delta t})$ 作为质控

### 2.2 模型设计模块

**OT 基线模型**
- 输入：初态样本 $\{\mathbf{x}_0^{(i)}\}$，终态样本 $\{\mathbf{x}_T^{(i)}\}$
- 输出：传输映射 $T_\theta: \mathbb{R}^d \to \mathbb{R}^d$，中间态预测 $\hat{\mu}_t$
- 架构：神经网络参数化 $T_\theta$，损失 $\min_\theta \mathbb{E}_{\mathbf{x}_0} \|T_\theta(\mathbf{x}_0) - \mathbf{x}_T\|^2$，插值 $\hat{\mathbf{x}}_t = (1-t/T)\mathbf{x}_0 + (t/T)T_\theta(\mathbf{x}_0)$

**薛定谔桥模型**
- 输入：完整轨迹 $\{\mathbf{x}_t^{(i)}\}_{t,i}$
- 输出：漂移场 $\mathbf{b}(\mathbf{x}, t)$，势函数 $\varphi_\theta(\mathbf{x}, t), \psi_\phi(\mathbf{x}, t)$
- 架构：神经网络参数化势函数，损失 $\mathcal{L} = \mathbb{E} \|\nabla \varphi + \nabla \psi + (\mathbf{x}_{t+\Delta t}-\mathbf{x}_t)/\Delta t\|^2$，漂移 $\mathbf{b} = -D\nabla(\varphi+\psi)$

**Diffusion 基线**
- 输入：终态样本 $\{\mathbf{x}_T^{(i)}\}$（可选初态条件）
- 输出：噪声预测网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$，生成轨迹
- 架构：U-Net/Transformer，去噪得分匹配 $\mathcal{L} = \mathbb{E} \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, t)\|^2$

### 2.3 训练与推理模块

**训练流程控制器**
- 输入：模型对象，训练数据集，超参数（学习率、批大小、轮数）
- 输出：模型检查点，训练曲线（损失、梯度范数、验证指标）
- 流程：初始化优化器，循环 batch 计算损失反向传播，定期验证集评估，早停机制

**推理引擎**
- 输入：训练好的模型，测试边界条件 $(\mu_0^{test}, \mu_T^{test})$，时间网格
- 输出：预测轨迹 $\{\hat{\mathbf{x}}_t\}$，预测分布 $\{\hat{\mu}_t\}$
- 流程：OT 模型通过插值生成，SB 模型求解 $d\mathbf{X}_t = \mathbf{b}(\mathbf{X}_t,t)dt + \sqrt{2D}d\mathbf{W}_t$，Diffusion 逐步去噪，拟合高斯得 $\hat{\mu}_t$

### 2.4 评估模块

**边界保真度评估器**
- 输入：预测和真实的初终态分布 $(\hat{\mu}_0, \hat{\mu}_T, \mu_0^{true}, \mu_T^{true})$
- 输出：$\mathcal{E}_{boundary} = W_2(\hat{\mu}_0, \mu_0^{true}) + W_2(\hat{\mu}_T, \mu_T^{true})$
- 计算：高斯用解析公式 $W_2^2 = \|\boldsymbol{\mu}_1-\boldsymbol{\mu}_2\|^2 + \text{tr}(\Sigma_1+\Sigma_2-2(\Sigma_2^{1/2}\Sigma_1\Sigma_2^{1/2})^{1/2})$

**路径保真度评估器**
- 输入：预测和真实的轨迹序列 $(\{\hat{\mu}_t\}, \{\mu_t^{true}\})$
- 输出：$\mathcal{E}_{path} = \int_0^T W_2(\hat{\mu}_t, \mu_t^{true}) dt$，时间分辨误差曲线
- 计算：逐时间点计算 Wasserstein 距离，梯形法则积分

**熵演化评估器**
- 输入：预测和真实的轨迹序列
- 输出：$\mathcal{E}_{entropy} = \int_0^T |H(\hat{\mu}_t) - H(\mu_t^{true})| dt$，峰值时刻和高度误差
- 计算：高斯熵 $H = \frac{1}{2}\log[(2\pi e)^d|\Sigma|]$，混合高斯用蒙特卡洛 $H \approx -\frac{1}{N}\sum_i \log p(\mathbf{x}_i)$

**几何结构评估器**
- 输入：预测和真实的高熵态协方差 $(\hat{\Sigma}_{peak}, \Sigma_{peak}^{true})$
- 输出：主成分方向误差 $\mathcal{E}_{structure} = \|\mathbf{V}_{pred}-\mathbf{V}_{true}\|_F$，特征值谱误差
- 计算：特征值分解提取前 $r$ 个主向量，处理符号不确定性

**泛化能力评估器**
- 输入：所有测试集评估结果，外推类型标签
- 输出：泛化报告（各类型平均误差、分布），路径信息增益 $\Delta \mathcal{L} = \mathcal{L}^{OT} - \mathcal{L}^{SB}$
- 计算：按外推类型分组统计，计算相对改进率，绘制误差随外推程度变化曲线

### 2.5 分析与可视化模块

**数据质量监控器**
- 输入：生成的轨迹数据集，设计参数
- 输出：质量报告（熵演化、Wasserstein 演化、参数对比），异常轨迹标记
- 监控：检查熵"先增后减"模式，峰值时间窗口，终态模态分离度，协方差低秩性

**轨迹可视化器**
- 输入：轨迹数据，时间戳
- 输出：PCA/UMAP 降维轨迹图，熵演化曲线，分布热图序列
- 方法：PCA 投影到前 2-3 维绘制轨迹，叠加熵颜色编码，动画展示分布演化

**模型对比可视化器**
- 输入：多个模型的预测结果，真实轨迹
- 输出：并排对比图（轨迹、熵曲线、误差热图），误差分解条形图
- 方法：统一坐标系绘制 OT/SB/Diffusion/Ground Truth，误差按时间段和空间区域分解

**泛化分析可视化器**
- 输入：泛化评估结果
- 输出：外推误差散点图（横轴外推程度，纵轴误差），雷达图（多指标对比），热图（参数空间误差分布）
- 方法：几何外推用 $\Delta/\Delta_{max}^{train}$ 归一化，拓扑外推按模态数分组，时间尺度外推用 $T/T_{max}^{train}$

**统计分析报告生成器**
- 输入：所有评估指标
- 输出：Markdown/LaTeX 格式报告，包含表格（均值±标准差）、显著性检验（配对 t 检验）、效应量（Cohen's d）
- 方法：计算 OT vs SB 的 $p$ 值，报告 $\Delta \mathcal{L}$ 的置信区间

## 三、实验流程设计

### 3.1 阶段一：简化验证

**目标**：验证基础假设，单一拓扑结构

**数据生成流程**：
1. 调用分布参数化子模块，固定 $d=50$, $N_{modes}=2$, $r=5$
2. 生成 $K_{train}=30$ 条轨迹，参数从 $\Delta \in [5, 10]$, $\lambda_1 \in [2, 5]$, $\tau_{entropy} \in [0.2T, 0.4T]$ 网格采样
3. 生成 $K_{test}=10$ 条轨迹，几何外推 $\Delta \in [12, 15]$, $\lambda_1 \in [6, 8]$
4. 调用数据质量监控器，检查熵峰值在 $t \in [0.3T, 0.5T]$，终态双井分离 $\|\boldsymbol{\mu}_A-\boldsymbol{\mu}_B\| \geq 0.9\Delta$
5. 保存数据集为 HDF5 格式，包含轨迹张量、分布参数、元数据

**模型训练流程**：
1. 初始化 OT 模型（3 层 MLP，隐藏维 256）和 SB 模型（时间条件 MLP，隐藏维 512）
2. OT 模型：提取训练集的 $(\mathbf{x}_0, \mathbf{x}_T)$ 对，训练 100 epoch，学习率 $10^{-3}$，Adam 优化器
3. SB 模型：使用完整轨迹，训练 200 epoch，学习率 $5 \times 10^{-4}$，每 10 epoch 验证
4. 保存检查点和训练曲线

**推理与评估流程**：
1. 对测试集的每个边界条件，OT 和 SB 模型各生成 1000 个细胞的轨迹
2. 调用路径保真度评估器，计算 $\mathcal{E}_{path}^{OT}$ 和 $\mathcal{E}_{path}^{SB}$
3. 调用熵演化评估器，检查峰值时刻预测误差
4. 调用泛化能力评估器，计算 $\Delta \mathcal{L} = \mathcal{E}_{path}^{OT} - \mathcal{E}_{path}^{SB}$
5. 预期结果：$\Delta \mathcal{L} > 0$ 且 $\mathcal{E}_{path}^{SB} < 0.5 \mathcal{E}_{path}^{OT}$

**可视化流程**：
1. 轨迹可视化器绘制 PCA 投影，对比 OT/SB/Ground Truth
2. 模型对比可视化器生成熵曲线并排图
3. 泛化分析可视化器绘制误差随 $\Delta$ 变化的散点图
4. 统计分析报告生成器输出 Markdown 报告

### 3.2 阶段二：拓扑多样化

**目标**：测试拓扑外推能力

**数据生成流程**：
1. 训练集包含三类：单模终态（$N_{modes}=1$，10 条），双模（$N_{modes}=2$，15 条），三模（$N_{modes}=3$，10 条）
2. 测试集包含四模终态（$N_{modes}=4$，5 条）和五模（$N_{modes}=5$，5 条）
3. 调整势能函数，熵增期的吸引子数量与终态模态数一致
4. 数据质量监控器额外检查：熵峰值 $H(\mu_{peak})$ 应随 $N_{modes}$ 增加

**模型训练流程**：
1. SB 模型架构升级为 Transformer（8 层，8 头，嵌入维 256），处理变长模态
2. 训练时随机采样不同模态数的轨迹组成 batch
3. 损失函数加入模态数条件：$\mathcal{L} = \mathbb{E}_{N_{modes}} [\text{SB\_loss} | N_{modes}]$

**推理与评估流程**：
1. 测试时给定 $N_{modes}=4$ 或 5，模型需推断高熵态的探索方向数
2. 几何结构评估器检查预测的 $\Sigma_{peak}$ 的有效秩是否接近 $N_{modes}$
3. 预期结果：SB 模型能推断"更多模态需要更高维探索"，OT 模型失败

### 3.3 阶段三：复杂动力学

**目标**：引入非单调熵演化

**数据生成流程**：
1. 设计双峰熵演化：$t \in [0, t_1]$ 第一次熵增，$t \in [t_1, t_2]$ 部分坍缩，$t \in [t_2, t_3]$ 第二次熵增，$t \in [t_3, T]$ 最终坍缩
2. 势能函数引入瞬态多模性：熵增期出现 3 个吸引子，但只有 2 个在终态存活
3. 训练集 20 条单峰轨迹 + 10 条双峰轨迹，测试集 10 条双峰轨迹（不同峰值时间）

**模型训练流程**：
1. SB 模型需学习"熵可以多次振荡"的模式
2. 增加时间编码的复杂度（正弦位置编码 + 可学习时间嵌入）

**推理与评估流程**：
1. 熵演化评估器检查预测轨迹是否出现双峰
2. 路径保真度评估器特别关注 $t \in [t_1, t_3]$ 的误差
3. 预期结果：SB 模型捕捉到非单调模式，OT 模型只能单调插值

### 3.4 阶段四：理论分析

**目标**：建立路径信息增益与几何量的定量关系

**分析流程**：
1. 对每条测试轨迹，计算薛定谔桥理论的精确解（高斯情况有解析解）
2. 计算路径的熵产生率：$\dot{S} = \int (\nabla \cdot \mathbf{b}) p_t d\mathbf{x}$
3. 计算信息几何曲率：Fisher 度量下的截面曲率
4. 计算与测地线的偏离度：$\int_0^T \|\mathbf{b}(\mathbf{x}_t, t) - \mathbf{b}^{geodesic}(\mathbf{x}_t)\|^2 dt$
5. 回归分析：$\Delta \mathcal{L}$ 对熵产生率、曲率、偏离度的依赖关系
6. 预期发现：$\Delta \mathcal{L} \propto \text{PathDeviation} \times \text{EntropyProduction}$

**可视化流程**：
1. 绘制 $\Delta \mathcal{L}$ vs 熵产生率的散点图，拟合幂律或线性关系
2. 绘制流形曲率的热图，叠加 OT 和 SB 模型的预测路径
3. 生成理论分析报告，包含推导和实验验证

## 四、模块间调用关系

**数据生成阶段**：
分布参数化 → 势能函数构造 → 轨迹采样 → 数据集构造 → 数据质量监控

**模型训练阶段**：
数据集构造 → 训练流程控制器 → (OT/SB/Diffusion 模型) → 检查点保存

**模型评估阶段**：
训练好的模型 + 测试集 → 推理引擎 → (边界/路径/熵/几何评估器) → 泛化能力评估器 → 统计分析报告

**可视化阶段**：
评估结果 → (轨迹/模型对比/泛化分析可视化器) → 图表保存

**完整实验流程**：
阶段一简化验证 → 阶段二拓扑多样化 → 阶段三复杂动力学 → 阶段四理论分析，每个阶段包含上述四个子阶段

## 五、预期输入输出总结

**数据生成模块**：配置文件（YAML）→ HDF5 数据集 + 质量报告（JSON）

**模型模块**：数据集 + 超参数 → 模型检查点（.pt）+ 训练日志（TensorBoard）

**评估模块**：模型 + 测试集 → 评估指标字典（JSON）+ 误差曲线（NumPy）

**可视化模块**：评估结果 → 图表（PNG/PDF）+ 分析报告（Markdown/LaTeX）

**端到端流程**：实验配置（YAML）→ 完整实验报告（包含数据、模型、评估、可视化）
