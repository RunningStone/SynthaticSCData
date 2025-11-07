# Schrödinger Bridge Synthetic Dataset

A modular implementation of synthetic cell state trajectory generation based on Schrödinger Bridge theory.

## Project Structure

```
SynthaticSCData/
├── Data/                    # Data generation modules
│   ├── distribution_params.py
│   ├── potential_function.py
│   ├── trajectory_sampler.py
│   └── dataset_constructor.py
├── Model/                   # Model architectures
│   ├── ot_model.py         # Optimal Transport baseline
│   ├── sb_model.py         # Schrödinger Bridge model
│   └── vae_model.py        # VAE baseline
├── Trainer/                 # Training and evaluation
│   ├── trainer.py
│   ├── evaluator.py
│   └── metrics.py
├── Analyser/                # Visualization and analysis
│   ├── data_quality.py
│   ├── trajectory_viz.py
│   ├── model_comparison.py
│   ├── generalization_viz.py
│   └── report_generator.py
├── Tests/                   # Test suite
│   └── test_data_generation.py
├── configs/                 # Configuration files
│   └── default_config.yaml
├── outputs/                 # Experiment outputs
├── run_experiment.py        # Main experiment runner
├── requirements.txt
└── SystemDesign.md         # Detailed system design
```

## Installation

### Option 1: Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run setup script
./setup_env.sh

# Or manually:
uv sync
```

### Option 2: Using pip

```bash
pip install -r requirements.txt
```

## Quick Start

### Quick Test

```bash
# Using uv
uv run python quick_test.py

# Or activate venv first
source .venv/bin/activate
python quick_test.py
```

### Run Phase 1 Experiment

```bash
# Using uv
uv run python run_experiment.py --config configs/default_config.yaml --output outputs/phase1

# Or with activated venv
python run_experiment.py --config configs/default_config.yaml --output outputs/phase1
```

## Running Tests

```bash
# Using uv
uv run pytest Tests/ -v

# Or with activated venv
pytest Tests/ -v
```

## Key Features

- **Modular Design**: Separate modules for data, models, training, and analysis
- **Three Model Baselines**: OT, SB, and VAE for comparison
- **Comprehensive Metrics**: Boundary fidelity, path fidelity, entropy evolution, geometric structure
- **Extrapolation Testing**: Geometric, topological, and temporal extrapolation
- **Quality Control**: Automated data quality monitoring

## Configuration

Edit `configs/default_config.yaml` to customize:
- Data generation parameters
- Model architectures
- Training hyperparameters
- Evaluation metrics
- Visualization settings

## Citation

See `SystemDesign.md` for detailed methodology and theoretical background.
用于通过给定gene list和默认或者给定GRN，简单的生成一组Single cell状态的数据
