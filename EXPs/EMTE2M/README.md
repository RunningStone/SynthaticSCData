# EMT_E2M 实验脚本

这个目录包含 EMT_E2M 特定实验的 bash 脚本。所有脚本使用硬编码参数，无需命令行参数。

## 配置文件位置

所有 EMT_E2M 实验的配置文件位于：
```
configs/EMT_E2M/
├── data_EMT_Cook_with_label.yaml           # 数据配置
├── models_default.yaml                      # 模型配置
├── analyzer_default.yaml                    # 分析器配置
├── experiment_EMT_Part1_setting1.yaml       # Setting 1: 边界时间点 (0d, 7d)
├── experiment_EMT_Part1_setting2.yaml       # Setting 2: 包含中间点 (0d, 3d, 7d)
├── experiment_EMT_Part1_setting3.yaml       # Setting 3: 完整前向轨迹
└── ...
```

## 输出位置

所有输出保存到：
```
/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/
├── precalc_results/                         # 预计算结果
├── Setting1/                                # Setting 1 训练结果
├── Setting2/                                # Setting 2 训练结果
└── ...
```

## 使用方法

### Step 1: 预计算分析

运行数据划分和模型参数分析：

```bash
cd /path/to/SynthaticSCData/EXPs/EMTE2M
bash step1_run_precalc_EMT_E2M.sh
```

**硬编码参数**：
- 输出目录: `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/precalc_results`
- 数据配置: `configs/EMT_E2M/data_EMT_Cook_with_label.yaml`
- Batch size: 256
- Input dimension: 1000 (HVGs)
- Min cells: 1000
- Bottleneck percentage: 100.0

**输出文件**：
- `data_split_analysis_final_params.json` - 最终采样参数
- `data_split_analysis_summary.txt` - 分析摘要
- `model_param_analysis_comparison.txt` - 模型对比

### Step 2: 训练模型

#### Setting 1 (边界时间点: 0d, 7d)

```bash
bash step2_train_setting1.sh
```

**硬编码参数**：
- 配置文件: `experiment_EMT_Part1_setting1.yaml`
- 配置目录: `configs/EMT_E2M`
- 输出目录: `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting1`

**训练的模型**：
- Schrödinger Bridge (SB_Base)
- Optimal Transport (OT)
- Variational Autoencoder (VAE)

#### Setting 2 (所有时间点: 0d, 8h, 1d, 3d, 7d)

```bash
bash step2_train_setting2.sh
```

**硬编码参数**：
- 配置文件: `experiment_EMT_Part1_setting2.yaml`
- 配置目录: `configs/EMT_E2M`
- 输出目录: `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting2`

**训练的模型**：
- MLPlus Schrödinger Bridge (SB_MLPlus)
- Batch Optimal Transport (BatchOT)
- Conditional VAE (CVAE)

#### Setting 3 (起始-早期-峰值: 0d, 8h, 7d)

```bash
bash step2_train_setting3.sh
```

**硬编码参数**：
- 配置文件: `experiment_EMT_Part1_setting3.yaml`
- 配置目录: `configs/EMT_E2M`
- 输出目录: `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting3`

**训练的模型**：
- Schrödinger Bridge (SB_Base)
- Optimal Transport (OT)
- Variational Autoencoder (VAE)

## 脚本特点

### 1. 硬编码参数
- ✅ 无需记忆复杂的命令行参数
- ✅ 确保实验的可重复性
- ✅ 避免参数输入错误

### 2. 自动路径解析
- 脚本自动检测项目根目录
- 无论从哪里调用都能正常工作

### 3. 清晰的输出
- 详细的进度信息
- 明确的成功/失败提示
- 下一步操作建议

### 4. 错误检查
- 配置文件存在性检查
- 虚拟环境检测
- 输出目录自动创建

## 工作流程

完整的实验工作流：

```bash
# 1. 进入实验目录
cd /path/to/SynthaticSCData/EXPs/EMTE2M

# 2. 激活虚拟环境（如果需要）
source ../../.venv/bin/activate

# 3. 预计算分析
bash step1_run_precalc_EMT_E2M.sh

# 4. 查看预计算结果
cat /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/precalc_results/data_split_analysis_summary.txt

# 5. 训练 Setting 1
bash step2_train_setting1.sh

# 6. 检查训练结果
ls -lh /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting1/

# 7. 评估模型（待实现）
# bash step3_evaluate_setting1.sh

# 8. 可视化结果（待实现）
# bash step4_visualize_setting1.sh
```

## 目录结构

```
EXPs/EMTE2M/
├── README.md                           # 本文件
├── step1_run_precalc_EMT_E2M.sh       # 预计算脚本
├── step2_train_setting1.sh            # Setting 1 训练脚本
├── step2_train_setting2.sh            # Setting 2 训练脚本（待创建）
├── step2_train_setting3.sh            # Setting 3 训练脚本（待创建）
├── step3_evaluate_setting1.sh         # Setting 1 评估脚本（待创建）
└── step4_visualize_setting1.sh        # Setting 1 可视化脚本（待创建）
```

## 常见问题

### Q: 如何修改输出目录？
A: 直接编辑脚本中的 `OUTPUT_DIR` 变量。

### Q: 如何使用不同的数据配置？
A: 编辑脚本中的 `DATA_CONFIG` 或 `CONFIG_FILE` 变量。

### Q: 脚本提示找不到配置文件？
A: 检查 `configs/EMT_E2M/` 目录下是否有相应的配置文件。

### Q: 训练失败怎么办？
A: 查看输出目录中的日志文件：`$OUTPUT_DIR/logs/`

## 与通用脚本的区别

| 特性 | 通用脚本 (EXPs/) | 专用脚本 (EXPs/EMTE2M/) |
|------|------------------|------------------------|
| 参数方式 | 命令行参数 | 硬编码 |
| 灵活性 | 高 | 低 |
| 易用性 | 需要记忆参数 | 直接运行 |
| 适用场景 | 探索性实验 | 标准化实验 |
| 可重复性 | 依赖参数记录 | 内置保证 |

## 开发新的 Setting 脚本

如果需要为新的 Setting 创建训练脚本：

1. 复制 `step2_train_setting1.sh`
2. 修改以下变量：
   - `CONFIG_FILE`: 实验配置文件名
   - `OUTPUT_DIR`: 输出目录路径
   - 脚本标题和描述
3. 添加执行权限：`chmod +x step2_train_settingX.sh`
4. 测试运行

## 注意事项

1. **虚拟环境**：确保在运行前激活虚拟环境
2. **磁盘空间**：训练会生成大量数据，确保输出目录有足够空间
3. **GPU资源**：某些模型需要GPU，确保CUDA可用
4. **配置一致性**：确保实验配置文件引用的数据配置文件名正确
