#!/bin/bash
# Run generation visualization

echo "========================================================================"
echo "Running Generation Visualization"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - HVG: 500"
echo "  - Samples per timepoint: 500"
echo "  - Generated samples per model: 500"
echo "  - Device: cuda"
echo "  - Models: SB_S1, OT_S1, VAE_S1, SB_MLPlus_S2"
echo "  - Output: ./visualization_outputs"
echo ""
echo "========================================================================"
echo ""

# Activate environment
source .venv/bin/activate

# Install additional dependencies if needed
echo "Installing visualization dependencies..."
pip install phate metric-learn -q

echo ""
echo "Running visualization..."
echo ""

# Run visualization
python run_visualization.py \
    --n_hvg 500 \
    --n_samples_per_timepoint 500 \
    --n_generate_per_model 500 \
    --output_base_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/uni_compare \
    --visualization_output_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/uni_compare/visualization \
    --device cuda

echo ""
echo "========================================================================"
echo "Visualization complete!"
echo "Check results at: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/uni_compare/visualization"
echo "========================================================================"
