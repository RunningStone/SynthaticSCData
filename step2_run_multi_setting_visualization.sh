#!/bin/bash
# Run Multi-Setting Visualization (Streamlined Version)
# 基于已保存的generated pkl文件和results.json进行可视化对比
# 删除了所有模型加载和推理相关代码，专注于绘图

echo "========================================================================"
echo "Multi-Setting Visualization (Streamlined)"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Base directory: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData"
echo "  - Output directory: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/visualizations"
echo ""
echo "Visualization tasks:"
echo "  (a) EMT Process Modeling: Setting1 (3 models), Setting2 (3 models), Setting3 (3 models)"
echo "  (b) Timepoint Ablation: Setting2 (1 model) vs Setting4 (3 models)"
echo "  (c) Timepoint Shuffle: Setting2 (1 model) vs Setting5 (1 model)"
echo "  (d) Linear Interpolation: Setting2 (2 models) vs Setting6 (2 models)"
echo ""
echo "Output for each task:"
echo "  - Metrics comparison: bar charts for 10 evaluation metrics"
echo "  - Generation comparison: PHATE and LMNN-PCA visualizations"
echo ""
echo "========================================================================"
echo ""

# Activate environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "Warning: No virtual environment found (.venv or venv)"
fi

# Install additional dependencies if needed
echo ""
echo "Checking visualization dependencies..."
pip install phate metric-learn pandas matplotlib -q

echo ""
echo "========================================================================"
echo "Running Multi-Setting Visualization..."
echo "========================================================================"
echo ""

# Run multi-setting visualization
python step2_multi_setting_visualization.py \
    --base_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData \
    --output_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/visualizations

exit_code=$?

echo ""
echo "========================================================================"
if [ $exit_code -eq 0 ]; then
    echo "Multi-Setting Visualization Complete!"
    echo "========================================================================"
    echo ""
    echo "Results saved to:"
    echo "  /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/visualizations"
    echo ""
    echo "Generated files:"
    echo ""
    echo "  (a) EMT Process Modeling (9 models total):"
    echo "      - a_emt_process_metrics.png/pdf/csv"
    echo "      - a_emt_process_phate.png/pdf"
    echo "      - a_emt_process_lmnn_pca.png/pdf"
    echo ""
    echo "  (b) Timepoint Ablation (4 models total):"
    echo "      - b_ablation_metrics.png/pdf/csv"
    echo "      - b_ablation_phate.png/pdf"
    echo "      - b_ablation_lmnn_pca.png/pdf"
    echo ""
    echo "  (c) Timepoint Shuffle (2 models total):"
    echo "      - c_shuffle_metrics.png/pdf/csv"
    echo "      - c_shuffle_phate.png/pdf"
    echo "      - c_shuffle_lmnn_pca.png/pdf"
    echo ""
    echo "  (d) Linear Interpolation (4 models total):"
    echo "      - d_interpolation_metrics.png/pdf/csv"
    echo "      - d_interpolation_phate.png/pdf"
    echo "      - d_interpolation_lmnn_pca.png/pdf"
    echo ""
    echo "========================================================================"
else
    echo "Multi-Setting Visualization Failed!"
    echo "========================================================================"
    echo ""
    echo "Exit code: $exit_code"
    echo "Please check the error messages above."
    echo ""
    echo "========================================================================"
fi

exit $exit_code
