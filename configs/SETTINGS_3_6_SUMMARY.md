# Setting 3-6 实验配置总结

## 配置文件创建

### 1. 数据配置 (`data_EMT_Cook_with_label.yaml`)
添加了4个新的setting定义：

- **Setting 3**: `["0d", "8h", "7d"]` - 前向EMT的3个关键时间点
- **Setting 4**: `["0d", "8h", "1d", "3d", "7d", "8h_rm", "1d_rm", "3d_rm"]` - 完整的8个时间点（包含reversal）
- **Setting 5**: `["0d", "3d_rm"]` - 起点到reversal终点的2个时间点
- **Setting 6**: `["0d", "7d", "3d_rm"]` - 起点、峰值、reversal终点的3个时间点

### 2. 实验配置文件
创建了4个新的实验YAML文件：
- `experiment_EMT_setting3.yaml`
- `experiment_EMT_setting4.yaml`
- `experiment_EMT_setting5.yaml`
- `experiment_EMT_setting6.yaml`

每个配置都训练3个模型：`sb_mlplus`, `batch_ot`, `vae`，epochs=200

## 代码修改

### 问题
原始代码在Evaluator中硬编码使用`sorted_times[0]`和`sorted_times[-1]`作为起点和终点。
这对于setting4-6不适用，因为：
- Setting 4/5/6的终点是`3d_rm`（索引7），而不是最后一个时间点
- 需要根据配置动态确定评估的起点和终点

### 解决方案

#### 1. Evaluator类增强 (`Trainer/sb_evaluator.py`)

**添加参数**：
```python
def __init__(self, device='cuda', model_name='sb', 
             start_timepoint=None, end_timepoint=None):
    """
    Args:
        start_timepoint: 起点标签 (e.g., '0d')
        end_timepoint: 终点标签 (e.g., '3d_rm')
    """
```

**存储time_labels**：
```python
def evaluate(self, model, test_loader, time_labels, model_name=None):
    # 存储time_labels供辅助方法使用
    self.time_labels = time_labels
```

**正确实现_find_timepoint_index()**：
```python
def _find_timepoint_index(self, sorted_times, time_to_indices, timepoint_label):
    """使用time_labels_order查找时间点索引"""
    # 在time_labels中查找timepoint_label的位置
    target_idx = self.time_labels.index(timepoint_label)
    
    # 检查该索引是否在当前数据中
    if target_idx not in sorted_times:
        raise ValueError(...)
    
    return target_idx
```

**修改的方法**：
- `_compute_frechet_distance()`
- `_compute_mae()`
- `_compute_pcc()`
- `_compute_advanced_metrics()`

所有方法现在使用：
```python
t_start = sorted_times[0] if self.start_timepoint is None else \
          self._find_timepoint_index(sorted_times, time_to_indices, self.start_timepoint)
t_end = sorted_times[-1] if self.end_timepoint is None else \
        self._find_timepoint_index(sorted_times, time_to_indices, self.end_timepoint)
```

#### 2. 训练工具更新 (`Trainer/utils.py`)

修改`train_model()`函数：
```python
# 从配置中读取起点和终点
eval_config = config.get('evaluation', {})
start_timepoint = eval_config.get('start_timepoint', None)
end_timepoint = eval_config.get('end_timepoint', None)

# 传递给Evaluator
evaluator = Evaluator(
    device=device, 
    model_name=model_name,
    start_timepoint=start_timepoint,
    end_timepoint=end_timepoint
)
```

## 配置示例

在实验YAML中添加evaluation配置（仅对需要非默认起点/终点的setting）：

```yaml
evaluation:
  eval_during_training: true
  eval_frequency: 5
  
  # 指定评估的起点和终点（可选）
  start_timepoint: "0d"      # 如果不指定，使用第一个时间点
  end_timepoint: "3d_rm"     # 如果不指定，使用最后一个时间点
```

## 运行实验

```bash
cd /home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData

# Setting 3
python step1_run_experiment.py configs/experiment_EMT_setting3.yaml

# Setting 4
python step1_run_experiment.py configs/experiment_EMT_setting4.yaml

# Setting 5
python step1_run_experiment.py configs/experiment_EMT_setting5.yaml

# Setting 6
python step1_run_experiment.py configs/experiment_EMT_setting6.yaml
```

## 关键设计原则

1. **利用现有顺序信息**：使用YAML中的`time_labels_order`提供的顺序
2. **索引映射**：`time_labels.index(label)` 直接给出数据集中的整数索引
3. **向后兼容**：如果不指定`start_timepoint`和`end_timepoint`，默认使用第一个和最后一个时间点
4. **错误检查**：验证指定的时间点标签存在且在当前数据中可用

## 输出目录

- Setting 3: `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Setting3`
- Setting 4: `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Setting4`
- Setting 5: `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Setting5`
- Setting 6: `/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Setting6`
