# Multi-Setting Visualization Update: Added CVAE and BatchOT Support

## 修改总结

为`step2_multi_setting_visualization.py`脚本添加了对**CVAE (vae)**和**BatchOT (batch_ot)**模型的完整支持。

## 修改文件

### 1. `Analyser/multi_setting_visualizer.py`

**修改内容**：
- 添加`BatchOTModel`导入
- 在`aggregate_model_configs()`方法中添加`batch_ot`模型配置处理

**新增代码**：
```python
# 导入部分
from Model import (
    SchrodingerBridgeModel,
    MLPlus_SchrodingerBridgeModel,
    OptimalTransportModel,
    ConditionalVAEModel,
    BatchOTModel  # 新增
)

# aggregate_model_configs方法中新增
elif model_name == 'batch_ot':
    # BatchOT needs n_timepoints and time_labels
    time_labels_order = config['data_source']['time_labels_order']
    n_timepoints = len(time_labels_order)
    model_kwargs.update({
        'n_timepoints': n_timepoints,
        'time_labels': time_labels_order,
        'hidden_dims': model_arch['hidden_dims'],
        'activation': model_arch['activation'],
        'dropout': model_arch['dropout'],
        'use_residual': model_arch.get('use_residual', True)
    })
```

### 2. `Analyser/multi_setting_visualizer_methods.py`

**修改内容**：
- 在`load_models_and_generate()`函数中添加`BatchOTModel`导入
- 添加`batch_ot`模型实例化逻辑
- 添加`batch_ot`轨迹生成逻辑

**新增代码**：
```python
# 导入部分
from Model import SchrodingerBridgeModel, MLPlus_SchrodingerBridgeModel, \
                  OptimalTransportModel, ConditionalVAEModel, BatchOTModel

# 模型实例化
elif model_type == 'batch_ot':
    model = BatchOTModel(**model_kwargs).to(self.device)

# 轨迹生成
elif model_type == 'batch_ot':
    # BatchOT uses sequential generation through multiple OT models
    with torch.no_grad():
        trajectory = model.generate_trajectory(source_tensor, time_grid, method='deterministic')
        generated = trajectory[:, -1, :]
```

## 支持的模型

现在可视化脚本支持以下所有模型：

1. **sb** - Schrödinger Bridge
2. **sb_mlplus** - Schrödinger Bridge with MLPlus架构
3. **ot** - Optimal Transport
4. **vae** - Conditional VAE (CVAE) ✅ **已支持**
5. **batch_ot** - Batch Optimal Transport ✅ **新增支持**

## 模型特性对比

| 模型 | 需要时间索引 | 生成方法 | 特殊参数 |
|------|------------|---------|---------|
| sb/sb_mlplus | ❌ | 连续轨迹 | diffusion_coeff |
| ot | ❌ | 直接映射 | use_residual |
| vae | ✅ | 潜空间插值 | n_timepoints, latent_dim, beta |
| batch_ot | ❌ | 顺序映射 | n_timepoints, time_labels |

## 使用方法

### 基本用法
```bash
python step2_multi_setting_visualization.py \
    --config_paths \
        /path/to/experiment_EMT_setting1.yaml \
        /path/to/experiment_EMT_setting2.yaml \
    --output_dir ./visualizations \
    --n_samples_per_timepoint 500 \
    --n_generate_per_model 500 \
    --device cuda
```

### 示例：比较Setting 2和Setting 3
```bash
python step2_multi_setting_visualization.py \
    --config_paths \
        configs/experiment_EMT_setting2.yaml \
        configs/experiment_EMT_setting3.yaml \
    --output_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/Visualizations/Setting2_vs_Setting3 \
    --n_samples_per_timepoint 500 \
    --n_generate_per_model 500 \
    --device cuda
```

## 输出文件

脚本会生成以下可视化文件：

1. **metrics_comparison.png/pdf/csv** - 所有模型的评估指标对比
2. **generation_comparison_phate.png/pdf** - PHATE降维可视化
3. **generation_comparison_lmnn_pca.png/pdf** - LMNN+PCA降维可视化

## 技术细节

### CVAE (vae)
- **特点**：使用潜空间插值生成轨迹
- **参数**：需要`n_timepoints`和时间索引
- **生成**：`model.generate_trajectory(x, time_grid, t_start_idx, t_end_idx)`

### BatchOT (batch_ot)
- **特点**：训练多个OT模型处理连续时间对
- **参数**：需要`n_timepoints`和`time_labels`列表
- **生成**：顺序应用所有OT模型，然后在离散状态间插值
- **优势**：适合多时间点设置，能捕捉局部转换动态

## 兼容性

- ✅ 完全兼容现有的Setting 1-6配置
- ✅ 支持所有实验配置文件格式
- ✅ 自动从checkpoint加载模型
- ✅ 自动读取evaluation metrics
- ✅ 支持多GPU和CPU模式
- ✅ 兼容PyTorch 2.6+（使用`weights_only=False`加载checkpoint）

## 注意事项

1. **Checkpoint要求**：
   - 大多数模型使用`best_model.pt`
   - **BatchOT特殊**：使用`final_model.pt`（因为它包含多个转换模型）
   - BatchOT还会保存单独的转换模型：`best_model_{t_start}_to_{t_end}.pt`
2. **配置一致性**：所有setting应使用相同的数据文件和n_hvg
3. **内存使用**：大量模型和样本可能需要较多GPU内存
4. **时间标签**：BatchOT需要正确的time_labels_order配置

## 测试建议

```bash
# 测试单个setting的所有模型
python step2_multi_setting_visualization.py \
    --config_paths configs/experiment_EMT_setting2.yaml \
    --output_dir ./test_viz \
    --n_samples_per_timepoint 100 \
    --n_generate_per_model 100 \
    --device cuda

# 测试多个setting对比
python step2_multi_setting_visualization.py \
    --config_paths \
        configs/experiment_EMT_setting2.yaml \
        configs/experiment_EMT_setting3.yaml \
        configs/experiment_EMT_setting4.yaml \
    --output_dir ./multi_setting_viz \
    --n_samples_per_timepoint 200 \
    --n_generate_per_model 200 \
    --device cuda
```
