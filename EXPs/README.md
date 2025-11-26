# EXPs 目录说明

这个目录包含所有实验相关的 bash 脚本。

## 目录结构

```
EXPs/
├── step0_setup_env.sh              # 环境设置脚本
├── step1_run_precalc.sh            # 预计算实验脚本
├── step1_run_experiment_EMT.sh     # EMT实验运行脚本
├── step2_run_evaluation_only.sh    # 评估脚本
├── step3_run_multi_setting_visualization.sh  # 可视化脚本
└── ...
```

## 使用方法

### 1. 环境设置

首次使用时，运行环境设置脚本：

```bash
cd /path/to/SynthaticSCData/EXPs
bash step0_setup_env.sh
```

该脚本会：
- 检查 uv 是否安装
- 在项目根目录创建虚拟环境
- 安装所有依赖

### 2. 运行预计算分析

在开始训练前，运行预计算分析以确定最优参数：

```bash
bash step1_run_precalc.sh
```

可选参数：
```bash
bash step1_run_precalc.sh \
    --output_dir ./my_results \
    --batch_size 512 \
    --input_dim 200 \
    --min_cells 2000

# 查看所有选项
bash step1_run_precalc.sh --help
```

### 3. 运行训练实验

```bash
bash step1_run_experiment_EMT.sh
```

### 4. 运行评估

```bash
bash step2_run_evaluation_only.sh
```

### 5. 运行可视化

```bash
bash step3_run_multi_setting_visualization.sh
```

## 脚本设计原则

所有 bash 脚本遵循以下原则：

1. **自动路径解析**：脚本自动检测项目根目录，无论从哪里调用都能正常工作
2. **参数化配置**：支持命令行参数覆盖默认配置
3. **清晰的输出**：提供详细的进度信息和错误提示
4. **帮助信息**：所有脚本都支持 `--help` 参数

## 与 Workers 的关系

- **EXPs/** (本目录): 存放 bash 脚本，用于启动和管理实验
- **Workers/**: 存放 Python 脚本，实现具体的功能逻辑

工作流程：
```
EXPs/step1_run_precalc.sh (bash)
    ↓ 调用
Workers/step1_precalc_exps.py (python)
    ↓ 使用
Analyser/data_split_analyzer.py (python class)
Analyser/model_param_analyzer.py (python class)
```

## 常见问题

### Q: 脚本提示找不到模块？
A: 确保已运行 `step0_setup_env.sh` 设置环境，并且从 EXPs 目录运行脚本。

### Q: 如何修改默认配置？
A: 可以通过命令行参数覆盖，或直接编辑脚本中的默认值部分。

### Q: 可以从其他目录运行这些脚本吗？
A: 可以，脚本会自动解析项目根目录。但建议从 EXPs 目录运行以保持一致性。

## 示例工作流

完整的实验工作流：

```bash
# 1. 设置环境（仅首次）
cd /path/to/SynthaticSCData/EXPs
bash step0_setup_env.sh

# 2. 预计算分析
bash step1_run_precalc.sh

# 3. 查看预计算结果，调整配置
cat ../precalc_results/data_split_analysis_summary.txt

# 4. 运行训练
bash step1_run_experiment_EMT.sh

# 5. 评估结果
bash step2_run_evaluation_only.sh

# 6. 可视化对比
bash step3_run_multi_setting_visualization.sh
```
