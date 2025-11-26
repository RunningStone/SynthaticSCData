# Setting2 训练超参数优化说明

## 问题诊断

### 旧版本 vs 新版本对比

| 指标 | 旧版本 | 新版本 | 差异 |
|------|--------|--------|------|
| 数据源 | scBERT预处理 | 原始数据 | 不同预处理 |
| 细胞数 | 4,000 | 8,974 | **2.24倍** |
| Train/Test split | Mix1,2,3 / Mix4 | Mix1,3,4 / Mix2 | 不同划分 |
| 训练epochs | 140 | 52 (提前停止) | **未充分训练** |
| Train loss | ~24,000 | ~55,000 | 2.3倍 |
| Test loss | ~1,709 | ~4,730 | 2.8倍 |
| Test loss波动 | 稳定 | 剧烈 (3805→6084) | **不稳定** |

### 核心问题

1. **数据规模翻倍**：8974 cells vs 4000 cells，需要更多训练时间
2. **Test loss波动大**：从3805跳到6084，触发early stopping (patience=50)
3. **学习率偏小**：1e-5对于更大数据集可能收敛过慢
4. **Batch size偏小**：64的batch size导致梯度估计不稳定

## 优化方案

### 1. 增加训练轮数
```yaml
epochs: 300  # 从200增加到300
```
**理由**：数据量翻倍，需要更多epoch才能充分学习

### 2. 增大Batch Size
```yaml
batch_size: 128  # 从64增加到128
```
**理由**：
- 更稳定的梯度估计，减少loss波动
- 加速训练（每epoch迭代次数减半）
- 8974 cells / 128 = 70 iterations/epoch（合理）

### 3. 提高学习率
```yaml
learning_rate: 3.0e-5  # 从1e-5提高到3e-5
```
**理由**：
- 更大的batch size允许使用更高学习率
- 加快收敛速度
- 3倍提升是保守的（线性scaling规则建议2倍）

### 4. 放宽Early Stopping
```yaml
early_stopping:
  patience: 80  # 从50增加到80
  min_delta: 5.0e-4  # 从1e-4增加到5e-4
```
**理由**：
- 更大的数据集需要更多时间稳定
- 容忍更大的loss波动（新数据集特性）
- 80 epochs patience ≈ 旧版本的50 epochs（相对数据量）

### 5. 调整Scheduler
```yaml
scheduler:
  type: "cosine"
  T_max: 300  # 匹配新的epochs
```
**理由**：Cosine scheduler在长训练中表现更好

## 预期效果

### 训练时间估算
- 每epoch时间：~旧版本的2倍（数据量2.24倍，batch size 2倍）
- 总训练时间：~旧版本的1.5倍（300 epochs vs 140 epochs）

### 收敛预期
- **前100 epochs**：快速下降期（学习率3e-5）
- **100-200 epochs**：稳定期（学习率逐渐衰减）
- **200-300 epochs**：精调期（学习率接近1e-6）

### Loss目标
- Train loss应该稳定在50,000-60,000范围（基于数据规模）
- Test loss应该稳定在4,000-5,000范围
- **关键**：Test loss波动应该减小到±500以内

## 监控指标

训练时重点关注：

1. **Test loss波动**：
   - 如果仍然>1000，考虑进一步增大batch size到256
   - 如果<500，说明优化成功

2. **收敛速度**：
   - 前50 epochs应该看到明显下降
   - 如果loss下降缓慢，可以尝试learning_rate=5e-5

3. **Early stopping触发时机**：
   - 理想情况：150-250 epochs之间触发
   - 如果<100 epochs触发：增加patience或min_delta
   - 如果>280 epochs仍未触发：可能过拟合，检查validation metrics

## 备选方案

如果上述优化仍不理想：

### 方案A：更激进的学习率
```yaml
learning_rate: 5.0e-5  # 5倍提升
batch_size: 256  # 进一步增大
```

### 方案B：Warmup + 更长训练
```yaml
epochs: 400
learning_rate: 5.0e-5
# 添加warmup（需要代码支持）
warmup_epochs: 20
```

### 方案C：降低模型复杂度
```yaml
architecture:
  hidden_dim: 384  # 从512降低
  n_blocks: 6  # 从8降低
```

## 实验记录

### 实验1：基础优化（当前配置）
- 日期：2024-11-24
- 配置：epochs=300, batch_size=128, lr=3e-5, patience=80
- 状态：待运行
- 结果：

### 实验2：（如需要）
- 日期：
- 配置：
- 状态：
- 结果：

## 参考

- 旧版本训练历史：`/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting2/checkpoints/sb_mlplus/training_history.json`
- 新版本训练历史：`/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting2/SynthaticSCData/EMT_Part1_Setting2/checkpoints/sb_mlplus/training_history.json`
- 配置文件：`configs/EMT_E2M/models_default.yaml`
