# Visualization Guide

## Overview

The visualization system has been refactored to work seamlessly with the YAML-based configuration system. Model parameters are now automatically loaded from experiment configuration files, eliminating hardcoded values.

## Key Features

✅ **YAML-driven**: Automatically loads model configurations from experiment YAML files  
✅ **No hardcoding**: Model architectures and parameters come from `models_*.yaml`  
✅ **Checkpoint auto-detection**: Automatically finds trained model checkpoints  
✅ **Flexible output**: Visualization results saved to experiment output directory or custom location  

## Usage

### Basic Usage

```bash
./run_visualization.sh experiment_EMT_setting1.yaml
```

### Using Python Directly

```bash
python step2_run_visualization.py experiment_EMT_setting1.yaml
```

### Advanced Options

```bash
python step2_run_visualization.py experiment_EMT_setting1.yaml \
    --config_dir configs \
    --n_samples_per_timepoint 500 \
    --n_generate_per_model 500 \
    --visualization_output_dir /custom/output/path \
    --device cuda \
    --seed 42
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `experiment_config` | Experiment YAML file (required) | - |
| `--config_dir` | Directory containing config files | `configs` |
| `--n_samples_per_timepoint` | Samples per timepoint from test set | 500 |
| `--n_generate_per_model` | Samples to generate per model | 500 |
| `--visualization_output_dir` | Custom output directory | `{experiment_output}/visualizations` |
| `--device` | Device for inference (cuda/cpu) | Auto-detect |
| `--seed` | Random seed | 42 |

## How It Works

### 1. Configuration Loading

The script loads the experiment YAML file and automatically:
- Reads data source information (file path, n_hvg)
- Extracts model architectures from referenced `models_*.yaml`
- Locates trained model checkpoints
- Determines output directories

### 2. Model Configuration Building

For each trained model, the script builds a configuration dictionary:

```python
{
    'SB': {
        'type': 'sb',
        'checkpoint_path': '/path/to/checkpoint/sb/best_model.pt',
        'model_kwargs': {
            'dimension': 500,
            'hidden_dims': [512, 512, 512, 512],
            'time_embedding_dim': 64,
            'dropout': 0.1,
            'diffusion_coeff': 0.1
        }
    },
    # ... other models
}
```

All parameters come from the YAML configuration files - **no hardcoding**!

### 3. Visualization Pipeline

The `GenerationVisualizer` then:
- Loads the trained models
- Generates synthetic samples
- Creates PHATE and LMNN+PCA visualizations
- Saves results to the output directory

## Examples

### Visualize EMT Setting 1 Results

```bash
./run_visualization.sh experiment_EMT_setting1.yaml
```

This will:
- Load models: SB, OT, VAE
- Use data from EMT dataset
- Save visualizations to: `OUTPUTs/SynthaticSCData/EMT_Setting1/visualizations/`

### Visualize EMT Setting 2 Results

```bash
./run_visualization.sh experiment_EMT_setting2.yaml
```

This will:
- Load model: SB MLPlus
- Use data from EMT dataset (all timepoints)
- Save visualizations to: `OUTPUTs/SynthaticSCData/EMT_Setting2/visualizations/`

### Visualize GSE234181 Results

```bash
# Setting 1 (boundary timepoints)
./run_visualization.sh experiment_GSE234181_setting1.yaml

# Setting 2 (all timepoints)
./run_visualization.sh experiment_GSE234181_setting2.yaml
```

## Output Structure

```
{experiment_output_dir}/
└── visualizations/
    ├── phate_visualization.png
    ├── lmnn_pca_visualization.png
    ├── generation_comparison.png
    └── metrics_summary.json
```

## Troubleshooting

### Checkpoint Not Found

If you see "Models found: []", ensure:
1. The experiment has been trained (`./run_experiment.sh`)
2. Checkpoints exist in `{output_dir}/checkpoints/{model_name}/best_model.pt`

### Configuration Loading Error

Ensure:
1. Experiment YAML file exists in `configs/` directory
2. Referenced data and model YAML files exist
3. YAML syntax is valid

### Memory Issues

If running out of memory:
- Reduce `--n_samples_per_timepoint`
- Reduce `--n_generate_per_model`
- Use CPU instead of GPU: `--device cpu`

## Migration from Old Script

### Old Way (Hardcoded)

```python
model_configs = {
    'SB_S1': {
        'type': 'sb',
        'checkpoint_path': output_base / 'setting1' / 'sb_model' / 'best_model.pt',
        'model_kwargs': {
            'dimension': args.n_hvg,
            'hidden_dims': [512, 512, 512, 512],  # Hardcoded!
            'time_embedding_dim': 64,              # Hardcoded!
            'dropout': 0.1,                        # Hardcoded!
            'diffusion_coeff': 0.1                 # Hardcoded!
        }
    }
}
```

### New Way (YAML-driven)

```bash
# Just specify the experiment config - everything else is automatic!
./run_visualization.sh experiment_EMT_setting1.yaml
```

All model parameters are loaded from `models_default.yaml` and merged with experiment-specific overrides.

## Benefits

1. **Consistency**: Visualization uses exact same model configs as training
2. **Maintainability**: Change model architecture in one place (YAML)
3. **Flexibility**: Easy to add new models or experiments
4. **Reproducibility**: Configuration is version-controlled
5. **No duplication**: Model parameters defined once in YAML files
