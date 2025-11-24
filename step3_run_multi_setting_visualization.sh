#!/bin/bash
# Run Multi-Setting Visualization
# 基于已保存的generated pkl文件进行可视化对比

echo "========================================================================"
echo "Multi-Setting Visualization"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Base directory: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData"
echo "  - Output directory: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/visualizations"
echo ""
echo "Visualization tasks:"
echo "  (a) EMT Process Modeling: Setting1, Setting2, Setting3"
echo "  (b) Timepoint Ablation: Setting2 vs Setting4"
echo "  (c) Timepoint Shuffle: Setting2 vs Setting5"
echo "  (d) Linear Interpolation: Setting2 vs Setting6"
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
    --base_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData \
    --output_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/visualizations

echo ""
echo "========================================================================"
echo "Multi-Setting Visualization Complete!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/visualizations"
echo ""
echo "Generated files:"
echo "  (a) a_emt_process_metrics.png/pdf/csv"
echo "      a_emt_process_phate.png/pdf"
echo "      a_emt_process_lmnn_pca.png/pdf"
echo ""
echo "  (b) b_ablation_metrics.png/pdf/csv"
echo "      b_ablation_phate.png/pdf"
echo "      b_ablation_lmnn_pca.png/pdf"
echo ""
echo "  (c) c_shuffle_metrics.png/pdf/csv"
echo "      c_shuffle_phate.png/pdf"
echo "      c_shuffle_lmnn_pca.png/pdf"
echo ""
echo "  (d) d_interpolation_metrics.png/pdf/csv"
echo "      d_interpolation_phate.png/pdf"
echo "      d_interpolation_lmnn_pca.png/pdf"
echo ""
echo "========================================================================"
