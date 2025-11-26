# Data Split Parameters Calculator

## 概述 / Overview

`step0_calculate_data_split_params.py` 是一个自动计算数据切分参数的工具，用于确保：

1. **公平对比**：不同setting使用相同的总训练样本数
2. **收敛保证**：每个category有足够的样本数（默认≥1000）
3. **类别均衡**：在训练/测试集约束下保持采样均衡

This script automatically calculates optimal data split parameters to ensure:

1. **Fair comparison**: All settings use the same total training samples
2. **Convergence guarantee**: Each category has sufficient samples (default ≥1000)
3. **Balanced sampling**: Maintains balance under train/test split constraints

---

## 使用方法 / Usage

### 基本用法 / Basic Usage

```bash
python step0_calculate_data_split_params.py \
    --data_config configs/data_EMT_Cook_with_label.yaml \
    --settings setting1 setting2 setting3 \
    --min_cells 1000 \
    --output_dir ./outputs/split_params
```

### 参数说明 / Arguments

- `--data_config`: 数据配置文件路径 / Path to data configuration YAML file
- `--settings`: 要分析的setting列表（至少2个）/ List of settings to analyze (minimum 2)
- `--min_cells`: 每个category最小样本数（默认1000）/ Minimum cells per category (default: 1000)
- `--output_dir`: 输出目录（默认`./outputs/split_params`）/ Output directory (default: `./outputs/split_params`)

---

## 计算逻辑 / Calculation Logic

### 1. 数据分布分析 / Data Distribution Analysis

脚本首先分析数据集中每个batch和时间点的细胞数量：

```
Batch      Time Label   Count
--------------------------------
Mix1       0d           1,234
Mix1       7d           1,456
...
```

### 2. 每个Setting的容量计算 / Capacity Calculation per Setting

对于每个setting，计算：

- **可用细胞数**：训练集中每个时间点的细胞数
- **瓶颈时间点**：可用细胞数最少的时间点
- **最大采样数**：瓶颈时间点的90%（留10%安全边际）

**Example for setting1** (0d, 3d_rm):
```
Time Point    Available (Train)
---------------------------------
0d            3,400
3d_rm         3,800

⚠️  Bottleneck: 0d with 3,400 cells
✓ Recommended cells_per_timepoint: 3,000 (90% of bottleneck)
```

### 3. 公平对比参数计算 / Fair Comparison Parameters

找出最严格的setting（最小总样本数），以此为基准：

```
setting1: 2 timepoints × 3,000 = 6,000 total
setting2: 8 timepoints × 750 = 6,000 total
setting3: 3 timepoints × 2,000 = 6,000 total
```

所有setting使用相同的总样本数（6,000），确保公平对比。

---

## 输出文件 / Output Files

### 1. `split_params_analysis.json`

完整的分析结果（JSON格式），包含：
- 每个setting的详细统计
- 瓶颈分析
- 推荐参数

### 2. `{setting_name}_params.yaml`

每个setting的YAML配置片段，可直接复制到实验配置文件：

```yaml
# Recommended parameters for setting1
setting1:
  cells_per_timepoint: 3000
  total_cells: 6000
  
  # Rationale:
  # - 2 timepoints × 3,000 = 6,000 total
  # - Bottleneck: 0d (3,000 / 3,400, 88.2%)
  # - Fair comparison: all settings use same total training samples
```

### 3. `summary.txt`

汇总表格：

```
Setting         Timepoints   Cells/TP     Total        Bottleneck
--------------------------------------------------------------------------------
setting1        2            3,000        6,000        0d
setting2        8            750          6,000        8h_rm
setting3        3            2,000        6,000        7d
```

---

## 工作流程 / Workflow

### Step 1: 准备数据配置 / Prepare Data Config

创建或修改数据配置文件（如 `data_EMT_Cook_with_label.yaml`），定义：

- `data_source`: 数据文件路径、时间标签列、时间顺序
- `biology_split`: 训练/测试集的batch划分
- `setting1`, `setting2`, `setting3`: 每个setting的时间点选择

### Step 2: 运行计算脚本 / Run Calculation Script

```bash
bash example_calculate_params.sh
```

或手动运行：

```bash
python step0_calculate_data_split_params.py \
    --data_config configs/data_EMT_Cook_with_label.yaml \
    --settings setting1 setting2 setting3 \
    --min_cells 1000 \
    --output_dir ./outputs/split_params_with_removal
```

### Step 3: 检查结果 / Review Results

查看输出目录中的文件：

```bash
ls -lh ./outputs/split_params_with_removal/
# split_params_analysis.json
# setting1_params.yaml
# setting2_params.yaml
# setting3_params.yaml
# summary.txt
```

### Step 4: 更新实验配置 / Update Experiment Configs

将推荐参数复制到实验配置文件中：

**For setting1** (`experiment_EMT_setting1.yaml`):
```yaml
data_sampling_override:
  cells_per_timepoint: 3000  # From calculation
```

**For setting2** (`experiment_EMT_setting2.yaml`):
```yaml
data_sampling_override:
  total_cells: 6000  # From calculation
```

**For setting3** (`experiment_EMT_setting3.yaml`):
```yaml
data_sampling_override:
  total_cells: 6000  # From calculation
```

---

## 示例：EMT数据集 / Example: EMT Dataset

### 场景 / Scenario

- **数据集**：EMT with removal labels
- **时间点**：0d, 8h, 1d, 3d, 7d, 8h_rm, 1d_rm, 3d_rm
- **训练集**：Mix1, Mix3, Mix4
- **测试集**：Mix2
- **最小样本数**：1000 cells/category

### Setting定义 / Setting Definitions

- **setting1**: 只用起点和终点 (0d, 3d_rm)
- **setting2**: 使用所有时间点 (8个)
- **setting3**: 起点-中间点-终点 (0d, 7d, 3d_rm)

### 预期输出 / Expected Output

假设数据分布如下：

| Time Point | Mix1+Mix3+Mix4 (Train) | Mix2 (Test) |
|------------|------------------------|-------------|
| 0d         | 3,400                  | 1,200       |
| 8h         | 2,800                  | 1,000       |
| 1d         | 3,200                  | 1,100       |
| 3d         | 3,600                  | 1,300       |
| 7d         | 4,200                  | 1,500       |
| 8h_rm      | 2,500                  | 900         |
| 1d_rm      | 2,900                  | 1,000       |
| 3d_rm      | 3,800                  | 1,400       |

**计算结果**：

- **setting1** (2 timepoints):
  - Bottleneck: 0d (3,400)
  - Recommended: 3,000 cells/timepoint
  - Total: 6,000

- **setting2** (8 timepoints):
  - Bottleneck: 8h_rm (2,500)
  - Max per timepoint: 2,250 (90% of 2,500)
  - But to match setting1 total: 6,000 / 8 = 750 cells/timepoint
  - Total: 6,000

- **setting3** (3 timepoints):
  - Bottleneck: 8h_rm if included, or 0d (3,400)
  - To match setting1 total: 6,000 / 3 = 2,000 cells/timepoint
  - Total: 6,000

---

## 注意事项 / Notes

### 1. 公平对比原则 / Fair Comparison Principle

所有setting必须使用**相同的总训练样本数**，这样性能差异才能归因于模型架构而非数据量。

All settings must use the **same total training samples** so that performance differences can be attributed to model architecture rather than data quantity.

### 2. 最小样本数要求 / Minimum Sample Requirement

每个category至少需要1000个样本以确保模型收敛。如果数据不足，脚本会报错。

Each category needs at least 1000 samples to ensure model convergence. The script will report an error if data is insufficient.

### 3. 安全边际 / Safety Margin

推荐值使用瓶颈的90%，留10%安全边际以应对数据过滤等情况。

Recommended values use 90% of the bottleneck, leaving a 10% safety margin for data filtering scenarios.

### 4. Setting检测逻辑 / Setting Detection Logic

- **setting=1**: 只有2个时间点且是首尾时间点
- **setting=2**: 其他所有情况（3个或更多时间点）

根据 `config_loader.py` 的逻辑，setting3会被识别为setting=2类型。

According to `config_loader.py` logic, setting3 will be recognized as setting=2 type.

---

## 故障排除 / Troubleshooting

### 错误：Bottleneck < minimum required

**问题**：某个时间点的可用细胞数少于最小要求（1000）

**解决方案**：
1. 降低 `--min_cells` 参数
2. 修改 `biology_split`，增加训练集batch
3. 从setting中移除该时间点

### 错误：Setting not found in config

**问题**：指定的setting在配置文件中不存在

**解决方案**：
1. 检查配置文件中的setting名称拼写
2. 确保配置文件包含所有指定的setting

### 警告：Exceeds capacity

**问题**：计算出的参数超过了某个时间点的可用容量

**解决方案**：
- 脚本会自动使用最大可用容量
- 检查是否需要调整训练/测试集划分

---

## 进阶用法 / Advanced Usage

### 自定义最小样本数 / Custom Minimum Samples

```bash
python step0_calculate_data_split_params.py \
    --data_config configs/data_EMT_Cook_with_label.yaml \
    --settings setting1 setting2 \
    --min_cells 500  # 降低到500
```

### 分析特定setting组合 / Analyze Specific Setting Combinations

```bash
# 只分析setting1和setting3
python step0_calculate_data_split_params.py \
    --data_config configs/data_EMT_Cook_with_label.yaml \
    --settings setting1 setting3 \
    --min_cells 1000
```

---

## 相关文件 / Related Files

- `configs/data_EMT_Cook.yaml`: 原始EMT数据配置（不含removal labels）
- `configs/data_EMT_Cook_with_label.yaml`: 新EMT数据配置（含removal labels）
- `Data/data_loader.py`: 数据加载器实现
- `Data/config_loader.py`: 配置加载器实现
- `example_calculate_params.sh`: 使用示例脚本

---

## 参考 / References

- **Setting1**: Boundary interpolation (首尾时间点)
- **Setting2**: Full trajectory learning (所有时间点)
- **Setting3**: Start-Middle-End (起点-中间点-终点)

每个setting的详细说明见对应的实验配置文件。

For detailed explanations of each setting, see the corresponding experiment configuration files.
