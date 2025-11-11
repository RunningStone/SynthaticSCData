#!/bin/bash
# Run unified experiment with fixed VAE model

echo "========================================================================"
echo "Running Unified Experiment with Fixed VAE Model"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - HVG: 500"
echo "  - Cells per timepoint: 5000"
echo "  - Epochs: 300"
echo "  - Batch size: 256"
echo "  - Device: cuda"
echo "  - Output: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/uni_compare"
echo ""
echo "Expected improvements for VAE:"
echo "  - PCC: from -0.66 to > 0.5"
echo "  - Frechet Distance: from 2,172,022 to < 500,000"
echo "  - MAE: from 35.29 to < 20"
echo ""
echo "========================================================================"
echo ""

# Activate environment
source .venv/bin/activate

# Run experiment
python step1_run_experiment_unified.py \
    --n_hvg 500 \
    --cells_per_timepoint 5000 \
    --epochs 300 \
    --batch_size 256 \
    --output_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/uni_compare \
    --device cuda

echo ""
echo "========================================================================"
echo "Experiment complete!"
echo "Check results at: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/uni_compare"
echo "========================================================================"
