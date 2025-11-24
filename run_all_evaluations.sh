#!/bin/bash
# 为所有实验运行评估并生成PKL文件

echo "========================================================================"
echo "Running Evaluations for All Experiments"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Load trained models from checkpoints"
echo "  2. Evaluate on test sets"
echo "  3. Save evaluation metrics to results.json"
echo "  4. Generate and save visualization data to generated/*.pkl"
echo ""
echo "========================================================================"
echo ""

# 激活环境
source .venv/bin/activate

# 基础路径
BASE_OUTPUT="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData"

# 函数：运行单个实验的评估
run_evaluation() {
    local config_file=$1
    local checkpoint_dir=$2
    local experiment_name=$3
    
    echo ""
    echo "========================================================================"
    echo "Evaluating: ${experiment_name}"
    echo "========================================================================"
    echo "  Config: ${config_file}"
    echo "  Checkpoints: ${checkpoint_dir}"
    echo ""
    
    python step1_run_evaluation_only.py \
        "${config_file}" \
        "${checkpoint_dir}" \
        --config_dir configs
    
    if [ $? -eq 0 ]; then
        echo "  ✓ ${experiment_name} evaluation complete"
    else
        echo "  ✗ ${experiment_name} evaluation failed"
    fi
}

# ============================================================================
# Part 1: EMT Process Modeling
# ============================================================================

echo ""
echo "========================================================================"
echo "Part 1: EMT Process Modeling (Settings 1-3)"
echo "========================================================================"

# Setting 1: Boundary points (0d, 7d)
run_evaluation \
    "experiment_EMT_Part1_setting1.yaml" \
    "${BASE_OUTPUT}/EMT_Part1_Setting1/checkpoints" \
    "Setting 1 (Boundary Points)"

# Setting 2: All timepoints (0d, 8h, 1d, 3d, 7d)
run_evaluation \
    "experiment_EMT_Part1_setting2.yaml" \
    "${BASE_OUTPUT}/EMT_Part1_Setting2/checkpoints" \
    "Setting 2 (All Timepoints)"

# Setting 3: Key timepoints (0d, 8h, 7d)
run_evaluation \
    "experiment_EMT_Part1_setting3.yaml" \
    "${BASE_OUTPUT}/EMT_Part1_Setting3/checkpoints" \
    "Setting 3 (Key Timepoints)"

# ============================================================================
# Part 2: Ablation Studies (Setting 4)
# ============================================================================

echo ""
echo "========================================================================"
echo "Part 2: Timepoint Ablation (Setting 4)"
echo "========================================================================"

# Setting 4a: Remove 8h
run_evaluation \
    "experiment_EMT_Part1_setting4_ablation_remove_8h.yaml" \
    "${BASE_OUTPUT}/EMT_Part1_Setting4/experiment_EMT_Part1_setting4_ablation_remove_8h/checkpoints" \
    "Setting 4 - Remove 8h"

# Setting 4b: Remove 1d
run_evaluation \
    "experiment_EMT_Part1_setting4_ablation_remove_1d.yaml" \
    "${BASE_OUTPUT}/EMT_Part1_Setting4/experiment_EMT_Part1_setting4_ablation_remove_1d/checkpoints" \
    "Setting 4 - Remove 1d"

# Setting 4c: Remove 3d
run_evaluation \
    "experiment_EMT_Part1_setting4_ablation_remove_3d.yaml" \
    "${BASE_OUTPUT}/EMT_Part1_Setting4/experiment_EMT_Part1_setting4_ablation_remove_3d/checkpoints" \
    "Setting 4 - Remove 3d"

# ============================================================================
# Part 3: Shuffle Study (Setting 5)
# ============================================================================

echo ""
echo "========================================================================"
echo "Part 3: Timepoint Shuffle (Setting 5)"
echo "========================================================================"

# Setting 5: Shuffled timepoints
run_evaluation \
    "experiment_EMT_Part1_setting5_shuffled.yaml" \
    "${BASE_OUTPUT}/EMT_Part1_Setting5_Shuffled/checkpoints" \
    "Setting 5 (Shuffled)"

# ============================================================================
# Part 4: Interpolation Study (Setting 6)
# ============================================================================

echo ""
echo "========================================================================"
echo "Part 4: Linear Interpolation (Setting 6)"
echo "========================================================================"

# Setting 6: Linear interpolation
run_evaluation \
    "experiment_EMT_Part1_setting6_interpolated.yaml" \
    "${BASE_OUTPUT}/EMT_Part1_Setting6/checkpoints" \
    "Setting 6 (Interpolated)"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================================================"
echo "All Evaluations Complete!"
echo "========================================================================"
echo ""
echo "Generated files for each experiment:"
echo "  - results.json              : Evaluation metrics"
echo "  - generated/{model}.pkl     : Visualization data"
echo ""
echo "Next steps:"
echo "  1. Check that all PKL files were generated successfully"
echo "  2. Run visualization: bash step2_run_multi_setting_visualization.sh"
echo ""
echo "To check PKL files:"
echo "  bash test_visualization_system.sh"
echo ""
echo "========================================================================"
