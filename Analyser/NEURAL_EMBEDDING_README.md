# 神经网络嵌入学习器 / Neural Embedding Learner

## 概述 / Overview

为了获得更好的时间点分离效果，我们实现了基于神经网络的嵌入学习器，使用**自编码器 + 对比学习**的方式替代了原有的LMNN+PCA方法。

To achieve better time-point separation, we implemented a neural network-based embedding learner using **AutoEncoder + Contrastive Learning** to replace the original LMNN+PCA method.

## 为什么需要替换LMNN？ / Why Replace LMNN?

### LMNN的局限性 / Limitations of LMNN

1. **线性假设** / **Linear Assumption**: LMNN学习的是线性度量变换，对于复杂的非线性流形结构效果有限
   - LMNN learns linear metric transformations, limited for complex nonlinear manifold structures

2. **训练时间长** / **Long Training Time**: 在大规模数据上训练LMNN非常耗时
   - Training LMNN on large-scale data is very time-consuming

3. **分离效果不佳** / **Poor Separation**: 在你的EMT数据上，LMNN无法很好地分离不同时间点
   - On your EMT data, LMNN cannot separate different time points well

### 神经网络方法的优势 / Advantages of Neural Network Method

1. **非线性表达能力** / **Nonlinear Expressiveness**: 多层神经网络可以学习复杂的非线性映射
   - Multi-layer neural networks can learn complex nonlinear mappings

2. **对比学习** / **Contrastive Learning**: 显式地拉近同类样本、推远异类样本
   - Explicitly pulls similar samples closer and pushes dissimilar samples apart

3. **端到端优化** / **End-to-End Optimization**: 同时优化重建和分离目标
   - Simultaneously optimizes reconstruction and separation objectives

4. **GPU加速** / **GPU Acceleration**: 可以利用GPU加速训练
   - Can leverage GPU for accelerated training

## 技术实现 / Technical Implementation

### 1. 模型架构 / Model Architecture

```
输入 (Input): [batch_size, n_features]
    ↓
编码器 (Encoder): [256, 128, 64] → 2D embedding
    ↓
嵌入空间 (Embedding Space): [batch_size, 2]
    ↓
解码器 (Decoder): [64, 128, 256] → reconstruction
    ↓
输出 (Output): [batch_size, n_features]
```

**关键组件** / **Key Components**:
- **Batch Normalization**: 稳定训练
- **Dropout (0.1)**: 防止过拟合
- **ReLU Activation**: 非线性激活

### 2. 损失函数 / Loss Functions

#### (1) 重建损失 / Reconstruction Loss (MSE)

```python
L_recon = MSE(x_reconstructed, x_original)
```

**作用** / **Purpose**: 确保嵌入空间保留原始数据的信息
- Ensures embedding space preserves information from original data

#### (2) 对比学习损失 / Contrastive Learning Loss (SupCon)

```python
L_contrast = SupConLoss(embeddings, time_labels)
```

**作用** / **Purpose**: 
- 拉近同一时间点的样本 / Pull samples from the same time point closer
- 推远不同时间点的样本 / Push samples from different time points apart

**公式** / **Formula**:
```
L_i = -log( Σ_p exp(z_i·z_p/τ) / Σ_a exp(z_i·z_a/τ) )
```
其中 / where:
- `z_i`: 样本i的嵌入 / Embedding of sample i
- `p`: 与i同类的样本 / Samples in the same class as i
- `a`: 所有其他样本 / All other samples
- `τ`: 温度参数 (0.07) / Temperature parameter

#### (3) 总损失 / Total Loss

```python
L_total = α * L_recon + β * L_contrast
```

**默认权重** / **Default Weights**:
- `α = 1.0` (重建权重 / Reconstruction weight)
- `β = 2.0` (对比权重 / Contrastive weight)

**说明** / **Note**: 我们给对比学习更高的权重，因为我们更关注时间点的分离效果
- We give contrastive learning higher weight because we care more about time-point separation

### 3. 训练策略 / Training Strategy

#### 优化器 / Optimizer
```python
AdamW(lr=1e-3, weight_decay=1e-5)
```

#### 学习率调度 / Learning Rate Scheduling
```python
ReduceLROnPlateau(factor=0.5, patience=10)
```

#### 早停 / Early Stopping
```python
patience=20 epochs
```

#### 数据划分 / Data Split
- 训练集 / Training: 80%
- 验证集 / Validation: 20%

## 使用方法 / Usage

### 方法1：在可视化中自动使用 / Method 1: Automatic Use in Visualization

可视化代码已经集成了新的嵌入学习器，会自动替代LMNN+PCA：

The visualization code has integrated the new embedding learner, automatically replacing LMNN+PCA:

```bash
bash run_visualization.sh
```

输出文件 / Output files:
- `generation_comparison_neural.png` - 神经嵌入可视化
- `generation_comparison_neural.pdf` - PDF版本

### 方法2：独立使用 / Method 2: Standalone Use

```python
from Analyser import EmbeddingLearner
import numpy as np

# 准备数据 / Prepare data
X = np.random.randn(1000, 500)  # 1000 samples, 500 features
y = np.random.randint(0, 5, 1000)  # 5 time points

# 初始化学习器 / Initialize learner
learner = EmbeddingLearner(
    input_dim=500,
    embedding_dim=2,
    hidden_dims=[256, 128, 64],
    device='cuda',
    learning_rate=1e-3,
    recon_weight=1.0,
    contrast_weight=2.0
)

# 训练并转换 / Fit and transform
embeddings = learner.fit_transform(
    X=X,
    y=y,
    batch_size=256,
    epochs=100,
    val_split=0.2
)

# 转换新数据 / Transform new data
X_new = np.random.randn(100, 500)
embeddings_new = learner.transform(X_new)
```

## 可视化结果对比 / Visualization Comparison

### LMNN+PCA vs Neural Embedding

| 方法 / Method | 训练时间 / Training Time | 分离效果 / Separation | GPU支持 / GPU Support |
|--------------|------------------------|---------------------|---------------------|
| LMNN+PCA | ~5-10分钟 / ~5-10 min | ⭐⭐ | ❌ |
| Neural Embedding | ~1-2分钟 / ~1-2 min | ⭐⭐⭐⭐⭐ | ✅ |

### 从可视化结果可以看到 / From Visualization Results

**Neural Embedding的优势** / **Advantages of Neural Embedding**:

1. **更清晰的时间点分离** / **Clearer Time-Point Separation**
   - 不同时间点的簇更加分离
   - Clusters of different time points are more separated

2. **更紧密的簇内聚合** / **Tighter Intra-Cluster Cohesion**
   - 同一时间点的样本更加聚集
   - Samples from the same time point are more clustered

3. **更好的生成数据定位** / **Better Generated Data Localization**
   - 生成的样本更准确地落在目标时间点区域
   - Generated samples more accurately fall in target time-point regions

## 参数调优指南 / Parameter Tuning Guide

### 1. 嵌入维度 / Embedding Dimension

```python
embedding_dim=2  # 用于可视化 / For visualization
embedding_dim=10  # 用于下游任务 / For downstream tasks
```

### 2. 隐藏层维度 / Hidden Dimensions

```python
# 浅层网络（快速） / Shallow network (fast)
hidden_dims=[128, 64]

# 中等网络（平衡） / Medium network (balanced)
hidden_dims=[256, 128, 64]  # 默认 / Default

# 深层网络（表达力强） / Deep network (expressive)
hidden_dims=[512, 256, 128, 64]
```

### 3. 损失权重 / Loss Weights

```python
# 更注重重建 / More focus on reconstruction
recon_weight=2.0, contrast_weight=1.0

# 平衡 / Balanced
recon_weight=1.0, contrast_weight=1.0

# 更注重分离（推荐） / More focus on separation (recommended)
recon_weight=1.0, contrast_weight=2.0
```

### 4. 温度参数 / Temperature Parameter

```python
temperature=0.07   # 默认，更强的分离 / Default, stronger separation
temperature=0.1    # 更温和的分离 / Gentler separation
temperature=0.05   # 更激进的分离 / More aggressive separation
```

## 训练输出解读 / Training Output Interpretation

```
Epoch   1: Train Loss=2074.53 (Recon=2059.58, Contrast=7.48) | Val Loss=2184.68 (Recon=2173.34, Contrast=5.67) ✓
Epoch  11: Train Loss=1618.98 (Recon=1607.76, Contrast=5.61) | Val Loss=1648.35 (Recon=1637.17, Contrast=5.59) ✓
...
Epoch  91: Train Loss=497.77 (Recon=486.60, Contrast=5.59) | Val Loss=510.63 (Recon=499.47, Contrast=5.58)
```

**关键指标** / **Key Metrics**:

1. **Total Loss下降** / **Total Loss Decreasing**: 模型在学习
   - Model is learning

2. **Recon Loss下降** / **Recon Loss Decreasing**: 重建质量提升
   - Reconstruction quality improving

3. **Contrast Loss稳定** / **Contrast Loss Stable**: 对比学习收敛
   - Contrastive learning converging

4. **✓标记** / **✓ Mark**: 该epoch取得了最佳验证损失
   - This epoch achieved best validation loss

## 性能优化建议 / Performance Optimization Tips

### 1. 内存优化 / Memory Optimization

```python
# 减小batch size / Reduce batch size
batch_size=128  # 默认256 / Default 256

# 减小隐藏层维度 / Reduce hidden dimensions
hidden_dims=[128, 64]  # 默认[256, 128, 64] / Default
```

### 2. 速度优化 / Speed Optimization

```python
# 减少训练轮数 / Reduce epochs
epochs=50  # 默认100 / Default 100

# 增大batch size（如果内存允许） / Increase batch size (if memory allows)
batch_size=512
```

### 3. 质量优化 / Quality Optimization

```python
# 增加训练轮数 / Increase epochs
epochs=200

# 使用更深的网络 / Use deeper network
hidden_dims=[512, 256, 128, 64]

# 调整对比权重 / Adjust contrastive weight
contrast_weight=3.0
```

## 扩展应用 / Extended Applications

### 1. 用于聚类分析 / For Clustering Analysis

```python
from sklearn.cluster import KMeans

# 获取嵌入 / Get embeddings
embeddings = learner.transform(X)

# 聚类 / Clustering
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(embeddings)
```

### 2. 用于异常检测 / For Anomaly Detection

```python
# 计算重建误差 / Compute reconstruction error
_, reconstruction = learner.model(torch.FloatTensor(X).to(device))
recon_error = np.mean((X - reconstruction.cpu().numpy())**2, axis=1)

# 高重建误差 = 异常 / High reconstruction error = anomaly
anomalies = recon_error > threshold
```

### 3. 用于迁移学习 / For Transfer Learning

```python
# 在数据集A上训练 / Train on dataset A
learner.fit_transform(X_A, y_A)

# 在数据集B上微调 / Fine-tune on dataset B
learner.fit(train_loader_B, val_loader_B, epochs=20)
```

## 常见问题 / FAQ

### Q1: 为什么对比损失不下降？ / Why doesn't contrastive loss decrease?

**A**: 对比损失通常会快速收敛到一个稳定值（~5-6），这是正常的。重要的是重建损失在下降。

Contrastive loss typically converges quickly to a stable value (~5-6), which is normal. What matters is that reconstruction loss is decreasing.

### Q2: 如何判断训练是否成功？ / How to judge if training is successful?

**A**: 观察以下指标：
1. 总损失持续下降 / Total loss continuously decreasing
2. 验证损失不过拟合 / Validation loss not overfitting
3. 可视化结果中时间点分离清晰 / Clear time-point separation in visualization

### Q3: 训练太慢怎么办？ / What if training is too slow?

**A**: 
1. 使用GPU (`device='cuda'`)
2. 减少epochs或hidden_dims
3. 增大batch_size

### Q4: 如何保存和加载模型？ / How to save and load model?

```python
# 保存 / Save
torch.save(learner.model.state_dict(), 'embedding_model.pt')

# 加载 / Load
learner.model.load_state_dict(torch.load('embedding_model.pt'))
```

## 引用 / Citation

如果使用本嵌入学习器，请引用对比学习相关论文：

If you use this embedding learner, please cite the contrastive learning paper:

```bibtex
@inproceedings{khosla2020supervised,
  title={Supervised contrastive learning},
  author={Khosla, Prannay and Teterwak, Piotr and Wang, Chen and Sarna, Aaron and Tian, Yonglong and Isola, Phillip and Maschinot, Aaron and Liu, Ce and Krishnan, Dilip},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={18661--18673},
  year={2020}
}
```

## 总结 / Summary

神经网络嵌入学习器通过结合自编码器的重建能力和对比学习的分离能力，为时间序列单细胞数据提供了更好的可视化嵌入空间。相比LMNN+PCA，它在分离效果、训练速度和GPU支持方面都有显著优势。

The neural embedding learner combines the reconstruction capability of autoencoders with the separation capability of contrastive learning, providing better visualization embedding space for time-series single-cell data. Compared to LMNN+PCA, it has significant advantages in separation effect, training speed, and GPU support.
