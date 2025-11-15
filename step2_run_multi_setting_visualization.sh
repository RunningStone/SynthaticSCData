#!/bin/bash
# Run Multi-Setting Visualization
# This script aggregates and visualizes results from all experimental settings

echo "========================================================================"
echo "Multi-Setting Visualization"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Samples per timepoint: 500"
echo "  - Generated samples per model: 500"
echo "  - Device: cuda"
echo "  - Base directory: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData"
echo "  - Output directory: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/visualizations"
echo ""
echo "This will aggregate models from all settings:"
echo "  - Setting1: SB, OT, VAE"
echo "  - Setting2: SB_MLPlus"
echo ""
echo "========================================================================"
echo ""

# Activate environment
source .venv/bin/activate

# Install additional dependencies if needed
echo "Installing visualization dependencies..."
pip install phate metric-learn pandas -q

echo ""
echo "========================================================================"
echo "Running Multi-Setting Visualization..."
echo "========================================================================"
echo ""

# Run multi-setting visualization
python step2_multi_setting_visualization.py \
    --config_paths \
        /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Setting1/experiment_config.yaml \
        /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Setting2/experiment_config.yaml \
    --output_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/visualizations \
    --n_samples_per_timepoint 500 \
    --n_generate_per_model 500 \
    --device cuda

echo ""
echo "========================================================================"
echo "Multi-Setting Visualization Complete!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/visualizations"
echo ""
echo "Generated files:"
echo "  - metrics_comparison.png/pdf/csv    : Comparison of all evaluation metrics"
echo "  - generation_comparison_phate.png/pdf : PHATE embedding visualization"
echo "  - generation_comparison_lmnn_pca.png/pdf : LMNN+PCA visualization"
echo ""
echo "========================================================================"
