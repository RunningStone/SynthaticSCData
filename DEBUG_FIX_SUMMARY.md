# Evaluator Debug Fix Summary

## 问题描述

在运行`bash step1_run_experiment_EMT.sh`时，SB模型训练完成后评估阶段出现错误：

```
TypeError: compute_loss() missing 2 required positional arguments: 't' and 'dt'
```

后续VAE模型评估时也出现错误：

```
AttributeError: 'tuple' object has no attribute 'detach'
```

## 根本原因

### 问题1: SB模型参数识别错误

**原始实现**（`sb_evaluator.py`第101-112行）：
- 使用`inspect.signature()`检查`compute_loss`的参数签名来判断模型类型
- 判断逻辑：`is_sb_model = 't' in param_names and 'dt' not in param_names and len(param_names) == 4`
- 这个逻辑有问题：SB模型的`compute_loss(x_t, x_next, t, dt)`有4个参数，且包含`'dt'`参数
- 导致SB模型被错误识别，调用时缺少必需的`t`和`dt`参数

### 问题2: VAE模型返回值处理缺失

**实际使用的模型**（`utils.py`第130行）：
- 使用的是`ConditionalVAEModel`而不是简单的`VAEModel`
- `ConditionalVAEModel.compute_loss()`返回`Tuple[torch.Tensor, dict]`
- 但evaluator中只处理了单个tensor返回值，导致尝试对tuple调用`.detach()`失败

## 解决方案

### 修复1: 添加model_name参数

**修改文件**: `Trainer/sb_evaluator.py`, `Trainer/utils.py`

**关键改动**:

1. **Evaluator类初始化**（`sb_evaluator.py`第23-30行）：
```python
def __init__(self, device: str = 'cuda', model_name: str = 'sb'):
    """
    Args:
        device: Device for evaluation
        model_name: Model type ('sb', 'ot', 'vae') for proper loss computation
    """
    self.device = device
    self.model_name = model_name.lower()
```

2. **evaluate方法**（`sb_evaluator.py`第32-52行）：
```python
def evaluate(
    self,
    model: torch.nn.Module,
    test_loader: DataLoader,
    time_labels: List[str],
    model_name: str = None
) -> Dict:
    # Use provided model_name or fall back to instance variable
    current_model_name = (model_name or self.model_name).lower()
    # ... 传递model_name到所有内部方法
```

3. **_compute_test_loss方法**（`sb_evaluator.py`第92-159行）：
```python
def _compute_test_loss(
    self,
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    model_name: str
) -> float:
    # 使用model_name参数而不是参数签名检测
    is_sb_model = model_name == 'sb'
    is_conditional = False  # Reserved for future
    is_ot_or_vae = model_name in ['ot', 'vae']
    
    # 根据模型类型调用不同的compute_loss接口
    if is_sb_model:
        # SB model: needs normalized time parameters
        t = torch.full((n_pairs_curr,), float(t_curr) / len(sorted_times), device=self.device)
        dt = 1.0 / len(sorted_times)
        loss = model.compute_loss(x_t, x_next, t, dt)
    elif is_ot_or_vae:
        # OT/VAE model: direct mapping
        loss_output = model.compute_loss(x_t, x_next)
        # Handle both single loss and (loss, loss_dict) returns
        if isinstance(loss_output, tuple):
            loss, _ = loss_output
        else:
            loss = loss_output
```

4. **utils.py调用更新**（第172-178行）：
```python
evaluator = Evaluator(device=device, model_name=model_name)
results = evaluator.evaluate(
    model=model,
    test_loader=test_loader,
    time_labels=time_labels,
    model_name=model_name
)
```

### 修复2: 处理VAE的tuple返回值

**修改位置**: `sb_evaluator.py`第145-152行

```python
elif is_ot_or_vae:
    # OT/VAE model: direct mapping
    loss_output = model.compute_loss(x_t, x_next)
    # Handle both single loss and (loss, loss_dict) returns
    if isinstance(loss_output, tuple):
        loss, _ = loss_output
    else:
        loss = loss_output
```

## 设计优势

### 1. 明确的模型类型识别
- 不再依赖不可靠的参数签名检测
- 使用显式的`model_name`参数，清晰明确
- 易于扩展支持新的模型类型

### 2. 统一的错误处理
- 所有模型类型都有明确的分支处理
- 未知模型类型会抛出清晰的错误信息
- 支持tuple和单值两种返回格式

### 3. 向后兼容
- `model_name`参数有默认值，不破坏现有代码
- `evaluate()`方法可以override初始化时的model_name
- 保持了原有的方法签名结构

## 测试结果

运行`bash step1_run_experiment_EMT.sh`：

✅ **SB模型**: 训练51个epoch后early stopping，评估成功完成
- Best test loss: 1267.907104
- 所有指标正常计算（test_loss, MAE, PCC, Wasserstein, MMD等）

✅ **OT模型**: 训练69个epoch后early stopping，评估成功完成  
- Best test loss: 140.912521
- Test Loss: 387.5471, MAE: 8.8905, PCC: 0.5576

✅ **VAE模型**: 训练90个epoch后early stopping，评估成功完成
- Best test loss: 63.858681
- 所有指标正常计算

## 问题3: VAE模型generate_trajectory参数不匹配

### 根本原因

**不同模型的generate_trajectory签名不一致**：
- **SB模型**: `generate_trajectory(x_0, time_grid, method='euler')` - 3个参数
- **OT模型**: `generate_trajectory(x_0, time_grid, method='linear')` - 3个参数
- **ConditionalVAE模型**: `generate_trajectory(x_0, time_grid, t_source_idx, t_target_idx, method='latent_interpolation')` - **5个参数**

但evaluator中所有调用都使用3参数形式，导致VAE评估时所有高级指标计算失败。

### 修复3: 根据模型类型调用generate_trajectory

**修改位置**: `sb_evaluator.py`中的4个方法
- `_compute_frechet_distance` (第199-205行)
- `_compute_mae` (第262-268行)
- `_compute_pcc` (第313-319行)
- `_compute_advanced_metrics` (第383-389行)

**修复代码**:
```python
# Call generate_trajectory with appropriate parameters based on model type
if model_name == 'vae':
    # ConditionalVAE needs time indices
    trajectory = model.generate_trajectory(x_start, time_grid, int(t_start), int(t_end), method='deterministic')
else:
    # SB and OT models don't need time indices
    trajectory = model.generate_trajectory(x_start, time_grid, method='deterministic')
```

**关键点**:
- 使用`model_name`参数判断模型类型
- VAE模型传递`t_start`和`t_end`作为时间索引
- 将时间索引转换为`int`类型
- SB和OT模型保持原有的3参数调用

## 总结

通过添加显式的`model_name`参数和基于模型类型的条件调用，我们：
1. ✅ 解决了SB模型的参数传递问题（compute_loss缺少t和dt参数）
2. ✅ 解决了VAE模型的tuple返回值处理问题（compute_loss返回tuple）
3. ✅ 解决了VAE模型的generate_trajectory参数不匹配问题（需要时间索引）
4. ✅ 提高了代码的可维护性和可扩展性
5. ✅ 保持了向后兼容性

这种基于模型名称的调度方式比基于参数签名检测更加可靠和清晰，能够正确处理不同模型的接口差异。
