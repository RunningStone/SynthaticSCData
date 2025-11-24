#!/bin/bash
# 测试新的可视化系统

echo "========================================================================"
echo "Testing New Visualization System"
echo "========================================================================"
echo ""

# 设置路径
BASE_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData"
OUTPUT_DIR="${BASE_DIR}/visualizations_test"

echo "Step 1: Checking if generated PKL files exist..."
echo "----------------------------------------------------------------------"

# 检查关键的PKL文件
check_pkl() {
    local setting=$1
    local model=$2
    local path="${BASE_DIR}/${setting}/generated/${model}.pkl"
    
    if [ -f "$path" ]; then
        echo "  ✓ Found: ${setting}/${model}.pkl"
        return 0
    else
        echo "  ✗ Missing: ${setting}/${model}.pkl"
        return 1
    fi
}

# 检查Setting1
echo ""
echo "Checking Setting1..."
check_pkl "EMT_Part1_Setting1" "sb"
check_pkl "EMT_Part1_Setting1" "ot"
check_pkl "EMT_Part1_Setting1" "vae"

# 检查Setting2
echo ""
echo "Checking Setting2..."
check_pkl "EMT_Part1_Setting2" "sb_mlplus"
check_pkl "EMT_Part1_Setting2" "batch_ot"
check_pkl "EMT_Part1_Setting2" "vae"

# 检查Setting3
echo ""
echo "Checking Setting3..."
check_pkl "EMT_Part1_Setting3" "sb_mlplus"
check_pkl "EMT_Part1_Setting3" "batch_ot"
check_pkl "EMT_Part1_Setting3" "vae"

echo ""
echo "========================================================================"
echo "Step 2: Testing visualization script..."
echo "========================================================================"
echo ""

# 激活环境
source .venv/bin/activate

# 运行可视化
python step2_multi_setting_visualization.py \
    --base_dir "${BASE_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "========================================================================"
echo "Step 3: Checking output files..."
echo "========================================================================"
echo ""

# 检查输出文件
check_output() {
    local prefix=$1
    local desc=$2
    
    echo "Checking ${desc}..."
    
    if [ -f "${OUTPUT_DIR}/${prefix}_metrics.png" ]; then
        echo "  ✓ ${prefix}_metrics.png"
    else
        echo "  ✗ ${prefix}_metrics.png"
    fi
    
    if [ -f "${OUTPUT_DIR}/${prefix}_metrics.pdf" ]; then
        echo "  ✓ ${prefix}_metrics.pdf"
    else
        echo "  ✗ ${prefix}_metrics.pdf"
    fi
    
    if [ -f "${OUTPUT_DIR}/${prefix}_metrics.csv" ]; then
        echo "  ✓ ${prefix}_metrics.csv"
    else
        echo "  ✗ ${prefix}_metrics.csv"
    fi
    
    if [ -f "${OUTPUT_DIR}/${prefix}_phate.png" ]; then
        echo "  ✓ ${prefix}_phate.png"
    else
        echo "  ✗ ${prefix}_phate.png"
    fi
    
    if [ -f "${OUTPUT_DIR}/${prefix}_lmnn_pca.png" ]; then
        echo "  ✓ ${prefix}_lmnn_pca.png"
    else
        echo "  ✗ ${prefix}_lmnn_pca.png"
    fi
    
    echo ""
}

check_output "a_emt_process" "(a) EMT Process Modeling"
check_output "b_ablation" "(b) Timepoint Ablation"
check_output "c_shuffle" "(c) Timepoint Shuffle"
check_output "d_interpolation" "(d) Linear Interpolation"

echo "========================================================================"
echo "Test Complete!"
echo "========================================================================"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "If any files are missing, check:"
echo "  1. Are the PKL files generated? (Run step1_run_evaluation_only.py)"
echo "  2. Are the paths correct in the visualization script?"
echo "  3. Check the log output for errors"
echo ""
