# Experiment 6: 快速开始指南

## 一键运行

```bash
cd /home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData/Experiments/exp6_interpolation
bash run_experiment6.sh
```

这将自动执行:
1. 训练所有模型 (插值数据自动生成) (~15小时GPU)
2. 分析结果 (~2小时)

## 核心改进

**数据生成已集成到训练流程中**:
- 不需要单独生成插值数据文件
- 数据路径从配置文件读取: `configs/data_EMT_Cook_with_label.yaml`
- 使用`InterpolatedDataLoader`自动生成插值数据

## 分步执行

### 训练模型

```bash
bash run_experiment6.sh
```

**训练模型**: SB MLPlus, Batch OT, VAE, OT, SB

**数据自动生成**: 
- 边界点 (0d, 7d): 从原始数据提取
- 中间点 (8h, 1d, 3d): 线性插值生成

### 分析结果

```bash
bash run_experiment6.sh --analysis-only
```

**输出**: 
- `per_timepoint_metrics.png`
- `residual_structure_analysis.png`
- `interpolation_quality_report.txt`

## 自定义参数

```bash
# 使用不同的输出目录
bash run_experiment6.sh --output /path/to/output

# 使用自定义配置文件
bash run_experiment6.sh --config my_custom_config.yaml
```

**修改数据参数**: 编辑 `configs/data_EMT_Cook_with_label.yaml` 中的 `setting6_interpolated` 部分

## 验证安装

```bash
python verify_implementation.py
```

应该看到: `✓ All tests passed! Implementation is ready.`

## 查看结果

```bash
# 查看训练日志
tail -f /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting6/logs/experiment.log

# 查看分析报告
cat /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting6/analysis/interpolation_quality_report.txt
```

## 常见问题

### Q1: 找不到输入数据文件

**A**: 数据路径在配置文件中定义，检查:
```bash
# 查看数据配置
cat /home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData/configs/data_EMT_Cook_with_label.yaml | grep file_path

# 验证文件存在
ls -lh /home/pan/Experiments/EXPs/2025_10_VCC_Exps/DATAs/EMT/2024_12_04_Cook_emt_dataset_with_removal.h5ad
```

### Q2: GPU内存不足

**A**: 减小批次大小，编辑配置文件:
```yaml
models_to_train:
  - name: "sb_mlplus"
    override_params:
      training:
        batch_size: 32  # 从64减小到32
```

### Q3: 训练时间过长

**A**: 减少训练轮数或只训练部分模型:
```yaml
models_to_train:
  - name: "sb_mlplus"
    enabled: true
    override_params:
      training:
        epochs: 100  # 从200减小到100
  - name: "vae"
    enabled: false  # 禁用某些模型
```

## 预期时间

| 步骤 | 时间 | 资源 |
|------|------|------|
| 数据生成 (自动) | ~5分钟 | CPU |
| 模型训练 | ~15小时 | GPU |
| 结果分析 | ~2小时 | CPU |
| **总计** | **~17小时** | - |

## 输出文件

```
EMT_Part1_Setting6/
├── checkpoints/                     # 模型检查点
│   ├── sb_mlplus_best.pt
│   ├── batch_ot_best.pt
│   ├── vae_best.pt
│   ├── ot_best.pt
│   └── sb_best.pt
├── results.json                     # 评估指标
├── analysis/                        # 分析结果
│   ├── per_timepoint_metrics.png
│   ├── residual_structure_analysis.png
│   └── interpolation_quality_report.txt
└── logs/
    └── experiment.log
```

**注意**: 插值数据在内存中生成，不保存为单独文件

## 下一步

1. 查看 `interpolation_quality_report.txt` 了解IEI和RSI指标
2. 对比Setting1和Setting2的结果
3. 准备论文图表和分析

## 获取帮助

```bash
bash run_experiment6.sh --help
python run_experiment6.py --help
```

## 文档

- `README.md`: 详细实验说明
- `IMPLEMENTATION_SUMMARY.md`: 实现细节
- `QUICKSTART.md`: 本文件
