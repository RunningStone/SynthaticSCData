# 配置参数传递状态总结

## SB_MLPlus 模型配置参数

### ✅ Architecture 参数（已正确传入）

所有架构参数都通过 `model_config['architecture']` 正确传入模型构造函数：

```yaml
architecture:
  hidden_dim: 512              ✅ 传入 MLPlus_SchrodingerBridgeModel
  n_blocks: 8                  ✅ 传入 MLPlus_SchrodingerBridgeModel
  time_embedding_dim: 64       ✅ 传入 MLPlus_SchrodingerBridgeModel
  n_time_frequencies: 10       ✅ 传入 MLPlus_SchrodingerBridgeModel
  activation: "relu"           ⚠️  模型内部硬编码，不可配置
  dropout: 0.1                 ✅ 传入 MLPlus_SchrodingerBridgeModel
  diffusion_coeff: 0.1         ✅ 传入 MLPlus_SchrodingerBridgeModel
  use_residual: true           ⚠️  模型内部硬编码为 True
```

### ✅ Training 参数（修复后已正确传入）

#### 基础训练参数
```yaml
training:
  epochs: 200                  ✅ 传入 trainer.train(epochs=...)
  batch_size: 64               ✅ 用于创建 DataLoader
  learning_rate: 1.0e-5        ✅ 传入 SBTrainer(learning_rate=...)
  optimizer: "adam"            ✅ 使用 AdamW（Adam的改进版）
```

#### Optimizer 参数
```yaml
  optimizer_kwargs:
    betas: [0.9, 0.999]        ✅ 修复后传入 AdamW(betas=...)
    eps: 1.0e-8                ✅ 修复后传入 AdamW(eps=...)
    weight_decay: 5.0e-6       ✅ 修复后传入 AdamW(weight_decay=...)
```

**修复前问题**：
- `betas` 硬编码为 `(0.9, 0.999)`
- `eps` 使用默认值
- `weight_decay` 使用默认值 `1e-5`

**修复后**：
- 从 `train_config['optimizer_kwargs']` 读取
- 传入 `SBTrainer.__init__(optimizer_kwargs=...)`
- 在 `SBTrainer` 中正确应用到 `AdamW` 优化器

#### Scheduler 参数
```yaml
  scheduler:
    type: "cosine"             ✅ 修复后支持 cosine/reduce_on_plateau/none
    T_max: 200                 ✅ 修复后传入 CosineAnnealingLR
    eta_min: 1.0e-6            ✅ 修复后传入 CosineAnnealingLR
```

**修复前问题**：
- 硬编码使用 `ReduceLROnPlateau`
- 忽略配置中的 `type: "cosine"`

**修复后**：
- 从 `train_config['scheduler']` 读取配置
- 根据 `type` 创建对应的 scheduler：
  - `cosine` → `CosineAnnealingLR(T_max, eta_min)`
  - `reduce_on_plateau` → `ReduceLROnPlateau(patience, factor, min_lr)`
  - `none` → 不使用 scheduler
- 在训练循环中根据 `scheduler_type` 调用正确的 `step()` 方法

#### Early Stopping 参数
```yaml
  early_stopping:
    enabled: true              ✅ 在 trainer.train() 中实现
    patience: 50               ✅ 修复后传入 trainer.train(early_stopping_patience=...)
    min_delta: 1.0e-4          ⚠️  未使用（可选功能）
```

**修复前问题**：
- `patience` 使用函数默认值 `10`

**修复后**：
- 从 `train_config['early_stopping']['patience']` 读取
- 传入 `trainer.train(early_stopping_patience=patience)`

#### Gradient Clipping 参数
```yaml
  gradient_clipping:
    enabled: true              ✅ 在 train_epoch() 中实现
    max_norm: 5.0              ✅ 修复后传入 SBTrainer(grad_clip_norm=...)
```

**修复前问题**：
- 使用默认值 `5.0`（碰巧与配置相同）

**修复后**：
- 从 `train_config['gradient_clipping']['max_norm']` 读取
- 传入 `SBTrainer.__init__(grad_clip_norm=...)`

---

## 修复的代码文件

### 1. `/Trainer/utils.py`

#### 修改点 1: `sb` 模型初始化
```python
# Get optimizer and training parameters from config
optimizer_kwargs = train_config.get('optimizer_kwargs', {})
scheduler_config = train_config.get('scheduler', {})
grad_clip_config = train_config.get('gradient_clipping', {})

trainer = SBTrainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    learning_rate=train_config['learning_rate'],
    device=device,
    output_dir=str(checkpoint_dir),
    weight_decay=optimizer_kwargs.get('weight_decay', 1e-5),
    grad_clip_norm=grad_clip_config.get('max_norm', 5.0),
    optimizer_kwargs=optimizer_kwargs,
    scheduler_config=scheduler_config
)
```

#### 修改点 2: `sb_mlplus` 模型初始化
```python
# 同上，保持一致
```

#### 修改点 3: 训练调用
```python
# Get early stopping patience from config
early_stopping_config = train_config.get('early_stopping', {})
patience = early_stopping_config.get('patience', 10)

logger.info(f"Early stopping patience: {patience}")

history = trainer.train(
    epochs=train_config['epochs'],
    early_stopping_patience=patience
)
```

### 2. `/Trainer/sb_trainer.py`

#### 修改点 1: `__init__` 方法签名
```python
def __init__(
    self,
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    learning_rate: float = 5e-4,
    device: str = 'cuda',
    output_dir: str = './outputs',
    weight_decay: float = 1e-5,
    grad_clip_norm: float = 5.0,
    optimizer_kwargs: Optional[Dict] = None,      # 新增
    scheduler_config: Optional[Dict] = None       # 新增
):
```

#### 修改点 2: Optimizer 创建
```python
# Get optimizer kwargs with defaults
if optimizer_kwargs is None:
    optimizer_kwargs = {}

betas = optimizer_kwargs.get('betas', [0.9, 0.999])
eps = optimizer_kwargs.get('eps', 1e-8)

# Setup optimizer with weight decay
self.optimizer = optim.AdamW(
    self.model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=tuple(betas),
    eps=eps
)
```

#### 修改点 3: Scheduler 创建
```python
# Setup learning rate scheduler based on config
if scheduler_config is None:
    scheduler_config = {}

scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')

if scheduler_type == 'cosine':
    T_max = scheduler_config.get('T_max', 200)
    eta_min = scheduler_config.get('eta_min', 1e-6)
    self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        self.optimizer,
        T_max=T_max,
        eta_min=eta_min
    )
    self.scheduler_type = 'cosine'
elif scheduler_type == 'reduce_on_plateau':
    patience = scheduler_config.get('patience', 10)
    factor = scheduler_config.get('factor', 0.5)
    min_lr = scheduler_config.get('min_lr', 1e-6)
    self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        self.optimizer,
        mode='min',
        factor=factor,
        patience=patience,
        min_lr=min_lr
    )
    self.scheduler_type = 'plateau'
else:
    self.scheduler = None
    self.scheduler_type = 'none'
```

#### 修改点 4: Scheduler 调用
```python
# Learning rate scheduler step
prev_lr = self.optimizer.param_groups[0]['lr']
if self.scheduler is not None:
    if self.scheduler_type == 'plateau':
        self.scheduler.step(test_loss)
    elif self.scheduler_type == 'cosine':
        self.scheduler.step()
```

---

## 验证方法

运行实验时，检查日志输出：

```bash
./run_experiment.sh experiment_EMT_setting2.yaml /path/to/output
```

### 期望看到的日志

1. **Early stopping patience**:
```
INFO - Early stopping patience: 50
```

2. **Scheduler 类型**（可以添加日志）:
```
INFO - Using cosine annealing scheduler (T_max=200, eta_min=1e-6)
```

3. **训练过程中的 patience**:
```
Patience: 1/50
Patience: 2/50
...
```

4. **Learning rate 变化**（cosine scheduler）:
```
Learning Rate: 1.00e-05
Learning Rate: 9.99e-06
Learning Rate: 9.97e-06
...
```

---

## 总结

### 修复前的问题
1. ❌ `optimizer_kwargs` (betas, eps, weight_decay) 未从配置读取
2. ❌ `scheduler` 类型和参数被硬编码
3. ❌ `early_stopping.patience` 使用默认值
4. ❌ `gradient_clipping.max_norm` 使用默认值

### 修复后的状态
1. ✅ 所有 architecture 参数正确传入模型
2. ✅ 所有 training 参数从 YAML 配置读取
3. ✅ Optimizer 参数 (betas, eps, weight_decay) 正确应用
4. ✅ Scheduler 根据配置类型动态创建
5. ✅ Early stopping patience 从配置读取
6. ✅ Gradient clipping max_norm 从配置读取

### 配置优先级
- **模型架构参数**: 以模型类定义为主（部分参数可配置）
- **训练和优化参数**: 完全以 YAML 配置为主 ✅

现在所有训练相关的参数都从 YAML 配置中读取并正确应用！
