# 实验7快速启动指南

## 一键运行（推荐）

从项目根目录运行：

```bash
cd /home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData

# 验证配置
python verify_exp7_config.py

# 运行完整实验（训练 + 熵分析）
bash run_experiment7.sh
```

## 分步运行

### 步骤1：训练模型

```bash
python step1_run_experiment.py \
    --config configs/experiment_EMT_Part1_setting7_entropy.yaml \
    --verbose
```

**输出位置**：`/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting7/`

**训练的模型**：
- `sb_mlplus` (推荐用于熵分析)
- `batch_ot`
- `vae`
- `ot`
- `sb`

**预计时间**：约2-3小时（GPU）

### 步骤2：熵演化分析

```bash
cd Experiments/exp7_entropy

# 方法1：使用bash脚本（需要先修改路径）
bash run_exp7.sh knn cuda

# 方法2：直接运行Python脚本
python run_entropy_analysis.py \
    --data_path /home/pan/Experiments/EXPs/2025_10_VCC_Exps/DATAs/EMT/2024_12_04_Cook_emt_dataset_with_removal.h5ad \
    --time_column Ground_truth \
    --time_labels 0d 8h 1d 3d 7d \
    --setting1_checkpoint /path/to/Setting1/sb_mlplus/best_model.pt \
    --setting2_checkpoint /path/to/Setting7/sb_mlplus/best_model.pt \
    --method knn \
    --k 5 \
    --n_samples 1000 \
    --output_dir ../../OUTPUTs/SynthaticSCData/EMT_Part1_Setting7/entropy_analysis \
    --device cuda \
    --cross_validate_methods
```

**预计时间**：约15-20分钟

## 配置说明

### 数据配置

**文件**：`configs/data_EMT_Cook_with_label.yaml`

**Setting7配置**：
```yaml
setting7:
  name: "Part1-Exp7: Entropy Evolution Analysis"
  time_points: ["0d", "8h", "1d", "3d", "7d"]
  total_cells: 8974
  balance_strategy: "total"
  
  entropy_analysis:
    enabled: true
    method: "knn"
    k_neighbors: 5
    n_samples_per_timepoint: 1000
```

### 实验配置

**文件**：`configs/experiment_EMT_Part1_setting7_entropy.yaml`

**关键参数**：
- `data_setting: "setting7"` - 使用setting7数据配置
- `output_dir: ".../EMT_Part1_Setting7"` - 输出目录遵循命名规范
- `experiment7_params` - 实验7特定参数

## 输出文件

### 训练输出

```
EMT_Part1_Setting7/
├── checkpoints/
│   ├── sb_mlplus/
│   │   ├── best_model.pt          # 最佳模型（用于熵分析）
│   │   └── final_model.pt
│   ├── batch_ot/
│   ├── vae/
│   ├── ot/
│   └── sb/
├── logs/
│   └── experiment.log
└── results.json
```

### 熵分析输出

```
EMT_Part1_Setting7/entropy_analysis/
├── entropy_curves_comparison.png       # 熵曲线对比图
├── entropy_curves_comparison.pdf
├── peak_characteristics_comparison.png # 峰值特征对比
├── peak_characteristics_comparison.pdf
├── method_cross_validation.png         # KNN vs 高斯交叉验证
├── entropy_analysis_summary.json       # 定量结果摘要
└── entropy_analysis_full_results.pkl   # 完整结果（含轨迹）
```

## 结果解读

### 1. 熵曲线对比图

**查看**：`entropy_curves_comparison.png`

**预期结果**：
- **真实数据**（蓝色实线）：显示倒U型曲线，峰值在1d或3d
- **Setting1**（紫色虚线）：单调或缺乏峰值 → 验证假设
- **Setting2/Setting7**（橙色实线）：与真实曲线接近 → 完整轨迹有效

### 2. 峰值特征对比

**查看**：`peak_characteristics_comparison.png`

**关键指标**：
- **峰值幅度**：Setting2应接近真实值
- **非单调性检测**：Setting1应为0（单调），Setting2应为1（非单调）
- **探索阶段速率**：熵增速率
- **MSE相似度**：Setting2应显著低于Setting1

### 3. 定量结果

**查看**：`entropy_analysis_summary.json`

```json
{
  "real_entropy_curve": [5.2, 7.1, 9.3, 7.8, 5.6],
  "real_peak_analysis": {
    "peak_time": "1d",
    "peak_value": 9.3,
    "is_nonmonotonic": true,
    "amplitude": 3.7
  },
  "settings": {
    "Setting1": {
      "peak_analysis": {
        "is_nonmonotonic": false,  // 预期：边界条件不足
        ...
      },
      "similarity_to_real": 2.5  // 高MSE
    },
    "Setting2": {
      "peak_analysis": {
        "is_nonmonotonic": true,   // 预期：捕捉到非单调性
        ...
      },
      "similarity_to_real": 0.3  // 低MSE
    }
  }
}
```

## 科学解释

### 假设验证

**核心假设**：边界条件不足以约束熵演化的非单调性

**验证标准**：
1. ✓ Setting1显示单调熵曲线 → 假设成立
2. ✓ Setting2/7重现非单调熵曲线 → 完整轨迹必要

### 生物学意义

**倒U型熵曲线**对应EMT过程的两个阶段：

1. **探索阶段**（熵增）：
   - 上皮基因被抑制
   - 间充质基因尚未激活
   - 细胞处于"双低"状态
   - 群体异质性增加

2. **收敛阶段**（熵减）：
   - 间充质程序稳定
   - 细胞收敛到新稳态
   - 群体异质性降低

## 故障排查

### 问题1：找不到训练好的模型

**症状**：
```
Warning: sb_mlplus checkpoint not found
```

**解决**：
1. 检查训练是否完成：`ls EMT_Part1_Setting7/checkpoints/sb_mlplus/`
2. 如果没有`best_model.pt`，使用`final_model.pt`
3. 重新训练：`python step1_run_experiment.py --config configs/experiment_EMT_Part1_setting7_entropy.yaml`

### 问题2：CUDA内存不足

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决**：
```bash
# 减少采样数量
python run_entropy_analysis.py ... --n_samples 500

# 或使用CPU
python run_entropy_analysis.py ... --device cpu
```

### 问题3：Setting1/2检查点不存在

**症状**：
```
Warning: Setting1 checkpoint not found, skipping comparison
```

**解决**：
这是正常的。如果没有Setting1/2的模型，脚本会跳过对比分析。

如需完整对比：
1. 先运行Setting1：`bash run_experiment1.sh`
2. 先运行Setting2：`bash run_experiment2.sh`
3. 再运行Setting7：`bash run_experiment7.sh`

## 下一步

### 扩展分析

1. **对比更多设置**：
   ```bash
   python run_entropy_analysis.py \
       --setting1_checkpoint .../Setting1/sb_mlplus/best_model.pt \
       --setting2_checkpoint .../Setting2/sb_mlplus/best_model.pt \
       --setting3_checkpoint .../Setting3/sb_mlplus/best_model.pt \
       ...
   ```

2. **尝试不同熵估计方法**：
   ```bash
   # 高斯方法
   bash run_exp7.sh gaussian cuda
   
   # 两种方法平均
   bash run_exp7.sh both cuda
   ```

3. **降维后分析**（如果高维估计不稳定）：
   ```python
   from sklearn.decomposition import PCA
   
   # 在entropy_estimators.py中添加降维
   pca = PCA(n_components=50)
   X_reduced = pca.fit_transform(X)
   H = estimate_entropy_knn(X_reduced, k=5)
   ```

### 论文撰写

**关键图表**：
- 图1：熵曲线对比（`entropy_curves_comparison.pdf`）
- 图2：峰值特征对比（`peak_characteristics_comparison.pdf`）
- 表1：定量指标（从`entropy_analysis_summary.json`提取）

**关键结论**：
1. Setting1无法重现非单调熵演化 → 边界条件不足
2. Setting2/7准确重现熵峰值 → 完整轨迹必要
3. 熵演化是细胞状态转换的关键动力学特征

## 参考

- **完整文档**：`README.md`
- **实现细节**：`IMPLEMENTATION_SUMMARY.md`
- **理论背景**：`/home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData/20251118_setting7_entropy.md`

## 联系

如有问题，请检查：
1. 配置验证：`python verify_exp7_config.py`
2. 单元测试：`python Experiments/exp7_entropy/test_exp7_modules.py`
3. 日志文件：`EMT_Part1_Setting7/logs/experiment.log`
