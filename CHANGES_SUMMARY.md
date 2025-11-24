# 可视化系统重构总结

## 修改概述

根据实验扩展的需求，重新实现了可视化系统，采用两阶段设计：
1. **评估阶段**: 保存生成数据为PKL文件
2. **可视化阶段**: 基于PKL文件进行多种对比可视化

## 修改的文件

### 1. `step1_run_evaluation_only.py`

**修改内容**:
- 添加`pickle`导入
- 修改`evaluate_model`函数，添加`output_dir`参数
- 在评估后调用`evaluator.generate_samples_for_visualization()`生成并保存数据
- 创建`generated/`文件夹，保存每个模型的PKL文件

**新增功能**:
```python
# 保存生成的数据
generated_data = evaluator.generate_samples_for_visualization(
    model=model,
    test_loader=test_loader,
    time_labels=time_labels,
    model_name=model_name
)

# 保存为pkl文件
pkl_path = generated_dir / f'{model_name}.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(generated_data, f)
```

### 2. `Trainer/sb_evaluator.py`

**新增方法**: `generate_samples_for_visualization`

**功能**:
- 收集所有测试数据
- 根据配置的起点/终点生成轨迹
- 返回包含真实数据和生成数据的字典

**返回数据结构**:
```python
{
    'real_data': np.ndarray,           # 所有真实测试数据
    'real_labels': np.ndarray,         # 时间标签
    'generated_data': np.ndarray,      # 模型生成的数据
    'start_indices': np.ndarray,       # 起始点索引
    'time_labels': list,               # 时间标签名称
    'start_timepoint': int,            # 起始时间点
    'end_timepoint': int               # 终止时间点
}
```

### 3. `step2_multi_setting_visualization.py`

**完全重写**，实现基于PKL文件的可视化系统。

**核心类**: `VisualizationManager`

**主要方法**:
- `load_generated_data()`: 加载PKL文件
- `load_metrics()`: 加载results.json
- `plot_metrics_comparison()`: 绘制评估指标对比
- `compute_embeddings()`: 计算PHATE和LMNN+PCA嵌入
- `plot_generation_comparison()`: 绘制生成数据可视化
- `visualize_emt_process_comparison()`: (a) EMT过程建模对比
- `visualize_ablation_comparison()`: (b) 时间点消融对比
- `visualize_shuffle_comparison()`: (c) 时间点打乱对比
- `visualize_interpolation_comparison()`: (d) 线性插值对比
- `run_all_visualizations()`: 运行所有可视化

**删除的内容**:
- 所有模型加载和推理相关代码
- 原有的`MultiSettingVisualizer`类及其方法
- 对`Analyser`模块的依赖

### 4. `step2_run_multi_setting_visualization.sh`

**完全重写**，简化为只需要两个参数：
- `--base_dir`: 包含所有实验输出的基础目录
- `--output_dir`: 可视化结果输出目录

**删除的参数**:
- `--config_paths`: 不再需要配置文件
- `--n_samples_per_timepoint`: 使用PKL中保存的数据
- `--n_generate_per_model`: 使用PKL中保存的数据
- `--device`: 不再需要模型推理

### 5. 新增文件

#### `VISUALIZATION_WORKFLOW.md`
详细的工作流程说明文档，包含：
- 两阶段设计概述
- 使用方法和示例
- 输出文件结构
- 技术细节

#### `CHANGES_SUMMARY.md`
本文档，总结所有修改

## 四种可视化对比

### (a) EMT过程建模对比
- **对比**: Setting1, Setting2, Setting3的所有模型
- **模型**: Setting1 (sb, ot, vae), Setting2 (sb_mlplus, batch_ot, vae), Setting3 (sb_mlplus, batch_ot, vae)
- **输出前缀**: `a_emt_process`

### (b) 时间点消融对比
- **对比**: Setting2的SB_MLPlus vs Setting4的三个消融实验
- **模型**: Setting2 (sb_mlplus), Setting4 (remove_8h, remove_1d, remove_3d)
- **输出前缀**: `b_ablation`

### (c) 时间点打乱对比
- **对比**: Setting2 vs Setting5的SB_MLPlus
- **模型**: Setting2 (sb_mlplus), Setting5 (sb_mlplus)
- **输出前缀**: `c_shuffle`

### (d) 线性插值对比
- **对比**: Setting2 vs Setting6的SB_MLPlus和Batch_OT
- **模型**: Setting2 (sb_mlplus, batch_ot), Setting6 (sb_mlplus, batch_ot)
- **输出前缀**: `d_interpolation`

## 输出文件

每种对比生成3类文件：

1. **评估指标对比**:
   - `{prefix}_metrics.png` - 高分辨率PNG
   - `{prefix}_metrics.pdf` - 矢量PDF
   - `{prefix}_metrics.csv` - 数据表格

2. **PHATE可视化**:
   - `{prefix}_phate.png`
   - `{prefix}_phate.pdf`

3. **LMNN+PCA可视化**:
   - `{prefix}_lmnn_pca.png`
   - `{prefix}_lmnn_pca.pdf`

## 评估指标对比图

包含10个子图，展示以下指标：
1. Test Loss
2. Fréchet Distance
3. MAE
4. PCC (Pearson Correlation Coefficient)
5. Wasserstein Distance
6. MMD (Maximum Mean Discrepancy)
7. JS Divergence
8. Correlation Structure
9. R² Mean
10. Correlation Frobenius Diff

## 生成数据可视化

每个可视化包含：
- **第一个子图**: 所有真实数据（按时间着色）
- **后续子图**: 每个模型的真实数据（灰色背景）+ 生成数据（红色）

使用两种降维方法：
- **PHATE**: 保持数据的流形结构
- **LMNN+PCA**: 学习度量空间后降维

## 技术优势

### 1. 效率提升
- 模型推理只需运行一次
- 可视化可以多次调整和重新生成
- 避免重复的模型加载和推理

### 2. 灵活性
- 易于添加新的可视化对比
- 可以独立调整可视化参数
- 支持部分实验的可视化

### 3. 可维护性
- 评估和可视化完全解耦
- 代码结构清晰，易于理解
- 删除了复杂的模型加载逻辑

### 4. 可复现性
- PKL文件保存完整的生成数据
- 可以在不同机器上生成相同的可视化
- 便于分享和存档

## 使用流程

### Step 1: 运行评估（一次性）

```bash
# 对每个实验运行评估
python step1_run_evaluation_only.py \
    experiment_config.yaml \
    /path/to/checkpoints
```

这会生成：
- `results.json` - 评估指标
- `generated/{model_name}.pkl` - 生成的数据

### Step 2: 生成可视化（可多次运行）

```bash
# 生成所有对比可视化
bash step2_run_multi_setting_visualization.sh
```

或者：

```bash
python step2_multi_setting_visualization.py \
    --base_dir /path/to/OUTPUTs/SynthaticSCData \
    --output_dir /path/to/visualizations
```

## 依赖要求

```bash
pip install phate metric-learn pandas matplotlib scikit-learn
```

## 向后兼容性

- 保留了`step1_run_evaluation_only.py`的原有功能
- 新增的PKL保存不影响原有的评估流程
- `results.json`格式保持不变

## 未来扩展

可以轻松添加新的可视化对比，只需：

1. 在`VisualizationManager`中添加新方法
2. 在`run_all_visualizations()`中调用
3. 定义要对比的settings和models

示例：

```python
def visualize_new_comparison(self):
    """新的对比可视化"""
    settings = {
        'SettingX': self.base_dir / 'EMT_Part1_SettingX',
        'SettingY': self.base_dir / 'EMT_Part1_SettingY'
    }
    
    metrics_dict = {}
    data_dict = {}
    
    # 收集数据...
    
    # 绘制
    self.plot_metrics_comparison(metrics_dict, "Title", "prefix")
    self.plot_generation_comparison(data_dict, "Title", "prefix")
```

## 注意事项

1. **磁盘空间**: PKL文件可能较大，确保有足够空间
2. **计算时间**: PHATE和LMNN计算可能需要较长时间
3. **内存使用**: 大数据集可能需要较多内存
4. **文件路径**: 确保base_dir结构与预期一致

## 测试建议

1. 先运行一个小的setting测试评估和PKL保存
2. 验证PKL文件内容正确
3. 运行可视化脚本测试单个对比
4. 最后运行完整的可视化流程

## 总结

这次重构实现了：
- ✅ 评估时保存生成数据为PKL
- ✅ 删除可视化中的模型加载和推理
- ✅ 实现四种对比可视化
- ✅ 复用现有的绘图逻辑
- ✅ 简化参数和使用流程
- ✅ 提供完整的文档

新系统更加高效、灵活和易于维护，完全满足实验扩展的需求。
