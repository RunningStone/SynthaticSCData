#!/usr/bin/env python3
"""
分析实验结果，回答三个问题：
1. 为什么 loss 值这么大？
2. MAE 的大小是否合理？
3. PCC 是否排除了样本量的影响？
"""

import json
import numpy as np
import torch
from scipy.stats import pearsonr

# 加载结果
results_path = "/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/sb_compare_mlp/results.json"
with open(results_path, 'r') as f:
    results = json.load(f)

print("="*80)
print("实验结果分析")
print("="*80)

# 提取关键数据
s1_eval = results['setting1']['evaluation']
s2_eval = results['setting2']['evaluation']

print(f"\nSetting 1 (Boundary Only):")
print(f"  Test Loss: {s1_eval['test_loss']:.2f}")
print(f"  MAE: {s1_eval['mae']:.4f}")
print(f"  PCC: {s1_eval['pcc']:.4f}")
print(f"  Samples: {s1_eval['n_samples']}")

print(f"\nSetting 2 (All Timepoints):")
print(f"  Test Loss: {s2_eval['test_loss']:.2f}")
print(f"  MAE: {s2_eval.get('mae', 'NaN')}")
print(f"  PCC: {s2_eval.get('pcc', 'NaN')}")
print(f"  Samples: {s2_eval['n_samples']}")

print("\n" + "="*80)
print("问题 1: 为什么 Loss 值这么大？")
print("="*80)

n_genes = 500
test_loss_s1 = s1_eval['test_loss']
test_loss_s2 = s2_eval['test_loss']

print(f"\nLoss 计算公式:")
print(f"  Loss = mean((drift - empirical_velocity)²)")
print(f"  其中 empirical_velocity = (x_next - x_t) / dt")
print(f"\nLoss 是在 {n_genes} 个基因维度上的均方误差 (MSE)")

print(f"\n每个基因的平均 loss:")
print(f"  Setting 1: {test_loss_s1/n_genes:.4f}")
print(f"  Setting 2: {test_loss_s2/n_genes:.4f}")

print(f"\nRMSE (每个基因的均方根误差):")
print(f"  Setting 1: {np.sqrt(test_loss_s1/n_genes):.4f}")
print(f"  Setting 2: {np.sqrt(test_loss_s2/n_genes):.4f}")

print(f"\n解释:")
print(f"  - Loss 是速度场预测的 MSE，不是直接的表达值误差")
print(f"  - 速度 = (x_next - x_t) / dt，其中 dt = 1/n_timepoints")
print(f"  - Setting 1: dt = 1/5 = 0.2")
print(f"  - Setting 2: dt = 1/5 = 0.2 (每个连续时间点对)")
print(f"  - 如果表达值变化 Δx，速度 = Δx/0.2 = 5*Δx")
print(f"  - 因此速度的误差会被放大 5 倍")

# 估算表达值变化
estimated_expr_change_s1 = np.sqrt(test_loss_s1/n_genes) * 0.2
estimated_expr_change_s2 = np.sqrt(test_loss_s2/n_genes) * 0.2
print(f"\n估算的表达值变化误差 (RMSE * dt):")
print(f"  Setting 1: {estimated_expr_change_s1:.4f}")
print(f"  Setting 2: {estimated_expr_change_s2:.4f}")

print("\n" + "="*80)
print("问题 2: MAE 的大小是否合理？")
print("="*80)

mae_s1 = s1_eval['mae']
print(f"\nMAE (Mean Absolute Error): {mae_s1:.4f}")
print(f"\n这是预测终点表达值与真实终点表达值的平均绝对误差")
print(f"需要与数据的实际尺度对比:")

# 从训练历史推断数据尺度
# 由于无法直接加载数据，我们从 loss 反推
print(f"\n从 Loss 反推数据尺度:")
print(f"  假设表达值的典型变化范围为 Δx")
print(f"  速度 v = Δx / dt = Δx / 0.2 = 5*Δx")
print(f"  Loss ≈ v² = 25*Δx²")
print(f"  因此 Δx ≈ sqrt(Loss/25) = sqrt({test_loss_s1}/25) = {np.sqrt(test_loss_s1/25):.4f}")

typical_change = np.sqrt(test_loss_s1/25)
print(f"\n相对误差分析:")
print(f"  MAE / 典型变化: {mae_s1/typical_change:.4f}x")
print(f"  这意味着预测误差约为典型变化的 {mae_s1/typical_change:.2f} 倍")

print(f"\n结论:")
if mae_s1/typical_change < 1.0:
    print(f"  ✓ MAE < 典型变化，模型预测较准确")
elif mae_s1/typical_change < 2.0:
    print(f"  ⚠ MAE ≈ 典型变化，模型预测中等")
else:
    print(f"  ✗ MAE > 典型变化，模型预测较差")

print("\n" + "="*80)
print("问题 3: PCC 是否排除了样本量的影响？")
print("="*80)

pcc_s1 = s1_eval['pcc']
n_samples_s1 = s1_eval['n_samples']
n_genes = 500

print(f"\nPCC 计算方式:")
print(f"  1. 生成轨迹: x_start → x_gen_end")
print(f"  2. 获取真实终点: x_real_end")
print(f"  3. 展平所有数据: real_flat = x_real_end.flatten()")
print(f"  4. 计算 Pearson 相关: pearsonr(real_flat, gen_flat)")

total_values = n_samples_s1 * n_genes
print(f"\n实际计算的数据点数:")
print(f"  样本数: {n_samples_s1}")
print(f"  基因数: {n_genes}")
print(f"  总数据点: {total_values:,}")

print(f"\nPearson 相关系数:")
print(f"  PCC = {pcc_s1:.4f}")

# 计算显著性
# 对于大样本，几乎任何相关性都是显著的
# 更重要的是效应量 (effect size)
print(f"\n效应量分析 (Cohen's guidelines):")
if abs(pcc_s1) < 0.1:
    effect = "可忽略 (negligible)"
elif abs(pcc_s1) < 0.3:
    effect = "小 (small)"
elif abs(pcc_s1) < 0.5:
    effect = "中等 (medium)"
else:
    effect = "大 (large)"
print(f"  |PCC| = {abs(pcc_s1):.4f} → {effect}")

print(f"\n决定系数 (R²):")
r_squared = pcc_s1 ** 2
print(f"  R² = {r_squared:.4f}")
print(f"  解释: 模型可以解释 {r_squared*100:.2f}% 的表达值变异")

print(f"\n关于样本量的影响:")
print(f"  - Pearson 相关系数本身不依赖于样本量")
print(f"  - 但样本量影响统计显著性")
print(f"  - 对于 n={total_values:,}，几乎任何 PCC > 0.01 都是显著的")
print(f"  - 因此应该关注效应量 (PCC 的大小) 而非 p 值")

# 计算 95% 置信区间 (Fisher's z-transformation)
z = 0.5 * np.log((1 + pcc_s1) / (1 - pcc_s1))
se_z = 1 / np.sqrt(total_values - 3)
z_lower = z - 1.96 * se_z
z_upper = z + 1.96 * se_z
pcc_lower = (np.exp(2*z_lower) - 1) / (np.exp(2*z_lower) + 1)
pcc_upper = (np.exp(2*z_upper) - 1) / (np.exp(2*z_upper) + 1)

print(f"\n95% 置信区间:")
print(f"  [{pcc_lower:.4f}, {pcc_upper:.4f}]")
print(f"  区间很窄，说明估计很精确（样本量大的结果）")

print("\n" + "="*80)
print("总结")
print("="*80)

print(f"\n1. Loss 值大的原因:")
print(f"   - Loss 是速度场的 MSE，不是表达值的 MSE")
print(f"   - 速度 = 表达值变化 / dt，放大了误差")
print(f"   - 实际表达值误差约为 {estimated_expr_change_s1:.2f}")

print(f"\n2. MAE 的合理性:")
print(f"   - MAE = {mae_s1:.2f}")
print(f"   - 相对于典型变化: {mae_s1/typical_change:.2f}x")
print(f"   - 评价: {'较好' if mae_s1/typical_change < 1.5 else '中等' if mae_s1/typical_change < 2.5 else '较差'}")

print(f"\n3. PCC 与样本量:")
print(f"   - PCC = {pcc_s1:.4f} (效应量: {effect})")
print(f"   - R² = {r_squared:.4f} (解释 {r_squared*100:.1f}% 变异)")
print(f"   - PCC 本身不受样本量影响，但大样本使估计更精确")
print(f"   - 应关注 PCC 的大小（效应量）而非显著性")

print("\n" + "="*80)
