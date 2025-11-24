# 可视化系统使用指南

## 快速开始

### 1. 运行所有评估（生成PKL文件）

```bash
bash run_all_evaluations.sh
```

这会为所有实验生成：
- `results.json` - 评估指标
- `generated/{model}.pkl` - 可视化数据

### 2. 生成所有可视化

```bash
bash step2_run_multi_setting_visualization.sh
```

这会生成四组对比可视化：
- (a) EMT过程建模对比
- (b) 时间点消融对比
- (c) 时间点打乱对比
- (d) 线性插值对比

### 3. 测试系统

```bash
bash test_visualization_system.sh
```

检查PKL文件和输出文件是否正确生成。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Step 1: 评估阶段                          │
│                                                              │
│  输入: 训练好的模型checkpoint                                 │
│  处理: 加载模型 → 评估 → 生成样本                             │
│  输出: results.json + generated/*.pkl                        │
│                                                              │
│  脚本: step1_run_evaluation_only.py                          │
│  批量: run_all_evaluations.sh                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Step 2: 可视化阶段                          │
│                                                              │
│  输入: results.json + generated/*.pkl                        │
│  处理: 加载数据 → 计算嵌入 → 绘制对比图                       │
│  输出: PNG/PDF/CSV可视化文件                                  │
│                                                              │
│  脚本: step2_multi_setting_visualization.py                  │
│  快捷: step2_run_multi_setting_visualization.sh              │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构

```
SynthaticSCData/
├── step1_run_evaluation_only.py          # 评估脚本
├── step2_multi_setting_visualization.py  # 可视化脚本
├── step2_run_multi_setting_visualization.sh  # 可视化快捷脚本
├── run_all_evaluations.sh                # 批量评估脚本
├── test_visualization_system.sh          # 测试脚本
├── VISUALIZATION_WORKFLOW.md             # 详细工作流程
├── CHANGES_SUMMARY.md                    # 修改总结
└── README_VISUALIZATION.md               # 本文档

OUTPUTs/SynthaticSCData/
├── EMT_Part1_Setting1/
│   ├── checkpoints/
│   │   ├── sb/
│   │   ├── ot/
│   │   └── vae/
│   ├── generated/              # 新增
│   │   ├── sb.pkl
│   │   ├── ot.pkl
│   │   └── vae.pkl
│   └── results.json
├── EMT_Part1_Setting2/
│   ├── checkpoints/
│   ├── generated/
│   └── results.json
├── ... (其他settings)
└── visualizations/             # 可视化输出
    ├── a_emt_process_metrics.png/pdf/csv
    ├── a_emt_process_phate.png/pdf
    ├── a_emt_process_lmnn_pca.png/pdf
    ├── b_ablation_metrics.png/pdf/csv
    ├── b_ablation_phate.png/pdf
    ├── b_ablation_lmnn_pca.png/pdf
    ├── c_shuffle_metrics.png/pdf/csv
    ├── c_shuffle_phate.png/pdf
    ├── c_shuffle_lmnn_pca.png/pdf
    ├── d_interpolation_metrics.png/pdf/csv
    ├── d_interpolation_phate.png/pdf
    └── d_interpolation_lmnn_pca.png/pdf
```

## 详细使用说明

### Step 1: 评估单个实验

```bash
python step1_run_evaluation_only.py \
    experiment_EMT_Part1_setting1.yaml \
    /path/to/checkpoints \
    --config_dir configs
```

**输出**:
- `results.json`: 包含所有评估指标
- `generated/{model}.pkl`: 每个模型的可视化数据

**PKL文件内容**:
```python
{
    'real_data': np.ndarray,        # 真实测试数据
    'real_labels': np.ndarray,      # 时间标签
    'generated_data': np.ndarray,   # 生成的数据
    'start_indices': np.ndarray,    # 起始点索引
    'time_labels': list,            # 时间标签名称
    'start_timepoint': int,         # 起始时间点
    'end_timepoint': int            # 终止时间点
}
```

### Step 2: 生成可视化

#### 方法1: 使用shell脚本（推荐）

```bash
bash step2_run_multi_setting_visualization.sh
```

#### 方法2: 直接运行Python脚本

```bash
python step2_multi_setting_visualization.py \
    --base_dir /path/to/OUTPUTs/SynthaticSCData \
    --output_dir /path/to/visualizations
```

## 四种可视化对比

### (a) EMT过程建模对比

**目的**: 比较不同时间分辨率对模型性能的影响

**对比内容**:
- Setting1 (边界点: 0d, 7d): sb, ot, vae
- Setting2 (全部点: 0d, 8h, 1d, 3d, 7d): sb_mlplus, batch_ot, vae
- Setting3 (关键点: 0d, 8h, 7d): sb_mlplus, batch_ot, vae

**输出文件**:
- `a_emt_process_metrics.png/pdf/csv`
- `a_emt_process_phate.png/pdf`
- `a_emt_process_lmnn_pca.png/pdf`

### (b) 时间点消融对比

**目的**: 评估各时间点的边际贡献

**对比内容**:
- Setting2 (完整): sb_mlplus
- Setting4 (移除8h): sb_mlplus
- Setting4 (移除1d): sb_mlplus
- Setting4 (移除3d): sb_mlplus

**输出文件**:
- `b_ablation_metrics.png/pdf/csv`
- `b_ablation_phate.png/pdf`
- `b_ablation_lmnn_pca.png/pdf`

### (c) 时间点打乱对比

**目的**: 验证时间因果信息的重要性

**对比内容**:
- Setting2 (正常顺序): sb_mlplus
- Setting5 (打乱顺序): sb_mlplus

**输出文件**:
- `c_shuffle_metrics.png/pdf/csv`
- `c_shuffle_phate.png/pdf`
- `c_shuffle_lmnn_pca.png/pdf`

### (d) 线性插值对比

**目的**: 评估线性插值能否替代真实中间点

**对比内容**:
- Setting2 (真实中间点): sb_mlplus, batch_ot
- Setting6 (线性插值): sb_mlplus, batch_ot

**输出文件**:
- `d_interpolation_metrics.png/pdf/csv`
- `d_interpolation_phate.png/pdf`
- `d_interpolation_lmnn_pca.png/pdf`

## 可视化内容说明

### 评估指标对比图

包含10个子图，展示以下指标：

1. **Test Loss**: 测试损失
2. **Fréchet Distance**: 生成分布与真实分布的差异
3. **MAE**: 平均绝对误差
4. **PCC**: Pearson相关系数
5. **Wasserstein Distance**: Wasserstein距离
6. **MMD**: 最大均值差异
7. **JS Divergence**: JS散度
8. **Correlation Structure**: 相关结构相似度
9. **R² Mean**: R²均值
10. **Correlation Frobenius Diff**: 相关矩阵Frobenius差异

### 生成数据可视化

#### PHATE嵌入
- 保持数据的流形结构
- 适合可视化连续的生物学过程

#### LMNN+PCA嵌入
- 学习度量空间后降维
- 强调类别间的差异

每个可视化包含：
- **第一个子图**: 所有真实数据（按时间着色）
- **后续子图**: 每个模型的真实数据（灰色）+ 生成数据（红色）

## 常见问题

### Q1: PKL文件太大怎么办？

A: PKL文件大小取决于测试集大小。如果空间不足：
1. 可以在评估时减少测试集大小
2. 压缩PKL文件：`gzip generated/*.pkl`
3. 可视化后删除PKL文件（保留results.json）

### Q2: 可视化计算太慢怎么办？

A: PHATE和LMNN计算可能较慢。优化方法：
1. 减少数据点数量（在PKL生成时）
2. 使用更少的LMNN维度
3. 并行计算（修改n_jobs参数）

### Q3: 如何添加新的对比可视化？

A: 在`step2_multi_setting_visualization.py`中：

```python
def visualize_new_comparison(self):
    """新的对比可视化"""
    settings = {
        'SettingX': self.base_dir / 'EMT_Part1_SettingX',
        'SettingY': self.base_dir / 'EMT_Part1_SettingY'
    }
    
    metrics_dict = {}
    data_dict = {}
    
    for setting_name, setting_path in settings.items():
        if setting_path.exists():
            metrics = self.load_metrics(setting_path)
            for model_name in ['sb_mlplus', 'batch_ot']:
                if model_name in metrics:
                    key = f"{setting_name}-{model_name}"
                    metrics_dict[key] = metrics[model_name]
                    data_dict[key] = self.load_generated_data(setting_path, model_name)
    
    self.plot_metrics_comparison(
        metrics_dict,
        "New Comparison Title",
        "e_new_comparison"
    )
    
    self.plot_generation_comparison(
        data_dict,
        "New Comparison: Generation",
        "e_new_comparison"
    )
```

然后在`run_all_visualizations()`中调用：

```python
def run_all_visualizations(self):
    self.visualize_emt_process_comparison()
    self.visualize_ablation_comparison()
    self.visualize_shuffle_comparison()
    self.visualize_interpolation_comparison()
    self.visualize_new_comparison()  # 新增
```

### Q4: 如何只生成特定的可视化？

A: 修改`step2_multi_setting_visualization.py`的`run_all_visualizations()`方法，注释掉不需要的可视化：

```python
def run_all_visualizations(self):
    self.visualize_emt_process_comparison()
    # self.visualize_ablation_comparison()  # 注释掉
    # self.visualize_shuffle_comparison()   # 注释掉
    self.visualize_interpolation_comparison()
```

### Q5: 如何修改可视化样式？

A: 在`plot_metrics_comparison()`和`plot_generation_comparison()`方法中修改matplotlib参数：

```python
# 修改图片大小
fig, axes = plt.subplots(3, 4, figsize=(20, 12))  # 调整figsize

# 修改颜色
bars = ax.bar(range(len(values)), values, color='skyblue')  # 指定颜色

# 修改字体大小
ax.set_title(metric_title, fontweight='bold', fontsize=14)  # 调整fontsize
```

## 依赖安装

```bash
# 基础依赖
pip install numpy pandas matplotlib scikit-learn

# 可视化依赖
pip install phate metric-learn

# 可选：加速计算
pip install numba
```

## 性能优化建议

1. **减少数据量**: 在生成PKL时使用较小的测试集
2. **并行计算**: 设置`n_jobs=-1`使用所有CPU核心
3. **缓存嵌入**: 保存计算好的嵌入，避免重复计算
4. **分批处理**: 对大数据集分批计算嵌入

## 故障排除

### 问题: PKL文件不存在

**原因**: 评估脚本未成功运行

**解决**:
```bash
# 检查checkpoint是否存在
ls /path/to/checkpoints

# 重新运行评估
python step1_run_evaluation_only.py config.yaml /path/to/checkpoints
```

### 问题: 可视化脚本报错

**原因**: 数据格式不匹配或路径错误

**解决**:
```bash
# 检查PKL文件内容
python -c "import pickle; print(pickle.load(open('generated/sb.pkl', 'rb')).keys())"

# 检查路径
python step2_multi_setting_visualization.py --base_dir /correct/path --output_dir /output/path
```

### 问题: 内存不足

**原因**: 数据量太大

**解决**:
1. 减少测试集大小
2. 分批处理可视化
3. 使用更少的嵌入维度

## 更多信息

- **详细工作流程**: 参见 `VISUALIZATION_WORKFLOW.md`
- **修改总结**: 参见 `CHANGES_SUMMARY.md`
- **主README**: 参见 `README.md`

## 联系与支持

如有问题，请检查：
1. 日志输出
2. PKL文件是否正确生成
3. 路径配置是否正确
4. 依赖是否完整安装
