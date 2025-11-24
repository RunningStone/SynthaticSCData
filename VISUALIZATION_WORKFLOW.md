# 可视化工作流程说明

## 概述

新的可视化系统分为两个步骤：
1. **Step 1**: 评估模型并保存生成的数据（PKL文件）
2. **Step 2**: 基于保存的PKL文件进行可视化对比

这种设计避免了重复的模型推理，提高了效率。

## Step 1: 评估并保存生成数据

### 功能
- 加载训练好的模型checkpoint
- 在测试集上进行评估
- 计算所有评估指标（Test Loss, Fréchet Distance, MAE, PCC等）
- **生成可视化用的样本数据并保存为PKL文件**

### 使用方法

```bash
python step1_run_evaluation_only.py \
    experiment_EMT_Part1_setting1.yaml \
    /path/to/checkpoints \
    --config_dir configs
```

### 输出文件结构

```
EMT_Part1_Setting1/
├── checkpoints/
│   ├── sb/
│   ├── ot/
│   └── vae/
├── generated/              # 新增：保存生成的数据
│   ├── sb.pkl
│   ├── ot.pkl
│   └── vae.pkl
└── results.json            # 评估指标
```

### PKL文件内容

每个PKL文件包含：
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

## Step 2: 多Setting可视化对比

### 功能

基于保存的PKL文件，生成四组对比可视化：

#### (a) EMT过程建模对比
- **对比**: Setting1, Setting2, Setting3的所有模型
- **目的**: 比较不同时间分辨率的影响
- **输出**: 
  - `a_emt_process_metrics.png/pdf/csv` - 评估指标对比
  - `a_emt_process_phate.png/pdf` - PHATE嵌入可视化
  - `a_emt_process_lmnn_pca.png/pdf` - LMNN+PCA可视化

#### (b) 时间点消融对比
- **对比**: Setting2的SB_MLPlus vs Setting4的三个消融实验
- **目的**: 评估各时间点的边际贡献
- **输出**: 
  - `b_ablation_metrics.png/pdf/csv`
  - `b_ablation_phate.png/pdf`
  - `b_ablation_lmnn_pca.png/pdf`

#### (c) 时间点打乱对比
- **对比**: Setting2 vs Setting5的SB_MLPlus
- **目的**: 验证时间因果信息的重要性
- **输出**: 
  - `c_shuffle_metrics.png/pdf/csv`
  - `c_shuffle_phate.png/pdf`
  - `c_shuffle_lmnn_pca.png/pdf`

#### (d) 线性插值对比
- **对比**: Setting2 vs Setting6的SB_MLPlus和Batch_OT
- **目的**: 评估线性插值能否替代真实中间点
- **输出**: 
  - `d_interpolation_metrics.png/pdf/csv`
  - `d_interpolation_phate.png/pdf`
  - `d_interpolation_lmnn_pca.png/pdf`

### 使用方法

```bash
# 方法1: 使用shell脚本（推荐）
bash step2_run_multi_setting_visualization.sh

# 方法2: 直接运行Python脚本
python step2_multi_setting_visualization.py \
    --base_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData \
    --output_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/visualizations
```

### 可视化内容

#### 评估指标对比图
- 包含10个子图，每个对应一个评估指标
- 柱状图展示不同模型/设置的性能
- 自动标注数值
- 保存为PNG、PDF和CSV格式

#### 生成数据可视化
- **PHATE嵌入**: 保持数据的流形结构
- **LMNN+PCA嵌入**: 学习度量空间后降维

每个可视化包含：
- 第一个子图：所有真实数据（按时间着色）
- 后续子图：每个模型的真实数据（灰色）+ 生成数据（红色）

## 完整工作流程示例

### 1. 运行所有实验的评估

```bash
# Setting 1
python step1_run_evaluation_only.py \
    experiment_EMT_Part1_setting1.yaml \
    /path/to/EMT_Part1_Setting1/checkpoints

# Setting 2
python step1_run_evaluation_only.py \
    experiment_EMT_Part1_setting2.yaml \
    /path/to/EMT_Part1_Setting2/checkpoints

# Setting 3
python step1_run_evaluation_only.py \
    experiment_EMT_Part1_setting3.yaml \
    /path/to/EMT_Part1_Setting3/checkpoints

# Setting 4 (三个消融实验)
python step1_run_evaluation_only.py \
    experiment_EMT_Part1_setting4_ablation_remove_8h.yaml \
    /path/to/experiment_EMT_Part1_setting4_ablation_remove_8h/checkpoints

python step1_run_evaluation_only.py \
    experiment_EMT_Part1_setting4_ablation_remove_1d.yaml \
    /path/to/experiment_EMT_Part1_setting4_ablation_remove_1d/checkpoints

python step1_run_evaluation_only.py \
    experiment_EMT_Part1_setting4_ablation_remove_3d.yaml \
    /path/to/experiment_EMT_Part1_setting4_ablation_remove_3d/checkpoints

# Setting 5
python step1_run_evaluation_only.py \
    experiment_EMT_Part1_setting5_shuffled.yaml \
    /path/to/EMT_Part1_Setting5_Shuffled/checkpoints

# Setting 6
python step1_run_evaluation_only.py \
    experiment_EMT_Part1_setting6_interpolated.yaml \
    /path/to/EMT_Part1_Setting6/checkpoints
```

### 2. 生成所有可视化

```bash
bash step2_run_multi_setting_visualization.sh
```

## 技术细节

### Evaluator类的新方法

在`Trainer/sb_evaluator.py`中添加了`generate_samples_for_visualization`方法：

```python
def generate_samples_for_visualization(
    self,
    model: torch.nn.Module,
    test_loader: DataLoader,
    time_labels: List[str],
    model_name: str = None
) -> Dict:
    """生成用于可视化的样本数据"""
```

该方法：
1. 收集所有测试数据
2. 根据配置的起点/终点生成轨迹
3. 返回包含真实数据和生成数据的字典

### 可视化管理器

`VisualizationManager`类负责：
1. 加载PKL文件和评估指标
2. 计算PHATE和LMNN+PCA嵌入
3. 生成对比图表
4. 保存多种格式（PNG, PDF, CSV）

### 嵌入计算策略

- **PHATE**: 直接在合并数据上计算，保持流形结构
- **LMNN+PCA**: 
  1. 标准化数据
  2. LMNN学习度量空间（最多50维）
  3. PCA降至2维

## 优势

1. **效率**: 模型推理只需运行一次，可视化可以多次调整
2. **灵活性**: 可以轻松添加新的可视化对比
3. **可复现**: PKL文件保存了完整的生成数据
4. **模块化**: 评估和可视化完全解耦

## 依赖

```bash
pip install phate metric-learn pandas matplotlib scikit-learn
```

## 注意事项

1. 确保所有实验都已运行评估并生成PKL文件
2. PKL文件可能较大，注意磁盘空间
3. PHATE和LMNN计算可能需要较长时间，特别是数据量大时
4. 可视化脚本会自动跳过不存在的设置/模型
