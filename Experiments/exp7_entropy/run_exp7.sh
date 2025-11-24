#!/bin/bash
# Run Experiment 7: Entropy Evolution Analysis
# 
# This script analyzes whether models can reproduce the non-monotonic 
# entropy evolution (entropy increase → entropy decrease) observed in
# real EMT trajectories.
#
# Usage:
#   bash run_exp7.sh [method] [device] [setting1_dir] [setting2_dir] [setting3_dir]
#
# Arguments:
#   method: knn, gaussian, or both (default: knn)
#   device: cuda or cpu (default: cuda)
#   setting1_dir: Path to Setting1 output directory (optional)
#   setting2_dir: Path to Setting2 output directory (optional)
#   setting3_dir: Path to Setting3 output directory (optional)

set -e  # Exit on error

# Parse arguments
METHOD=${1:-knn}
DEVICE=${2:-cuda}

# Get project root (two levels up from Experiments/exp7_entropy)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Base output directory
BASE_OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData"

# Model checkpoint directories (use arguments if provided, otherwise use defaults)
SETTING1_DIR="${3:-${BASE_OUTPUT_DIR}/EMT_Part1_Setting1}"
SETTING2_DIR="${4:-${BASE_OUTPUT_DIR}/EMT_Part1_Setting2}"
SETTING3_DIR="${5:-${BASE_OUTPUT_DIR}/EMT_Part1_Setting3}"

# Find best model checkpoints (try sb_mlplus first, then sb)
find_best_checkpoint() {
    local setting_dir=$1
    local ckpt_dir="${setting_dir}/checkpoints"
    
    # Try sb_mlplus first
    if [ -f "${ckpt_dir}/sb_mlplus/best_model.pt" ]; then
        echo "${ckpt_dir}/sb_mlplus/best_model.pt"
    elif [ -f "${ckpt_dir}/sb_mlplus/final_model.pt" ]; then
        echo "${ckpt_dir}/sb_mlplus/final_model.pt"
    # Fall back to sb
    elif [ -f "${ckpt_dir}/sb/best_model.pt" ]; then
        echo "${ckpt_dir}/sb/best_model.pt"
    elif [ -f "${ckpt_dir}/sb/final_model.pt" ]; then
        echo "${ckpt_dir}/sb/final_model.pt"
    else
        echo ""
    fi
}

SETTING1_CKPT=$(find_best_checkpoint "${SETTING1_DIR}")
SETTING2_CKPT=$(find_best_checkpoint "${SETTING2_DIR}")
SETTING3_CKPT=$(find_best_checkpoint "${SETTING3_DIR}")

# Output directory (save to base output directory)
OUTPUT_DIR="${BASE_OUTPUT_DIR}/EMT_Part1_Setting7/entropy_analysis_${METHOD}"
mkdir -p "${OUTPUT_DIR}"

# Sampling parameters
N_SAMPLES=1000
K_NEIGHBORS=5
N_STEPS=50

echo "========================================================================"
echo "Experiment 7: Entropy Evolution Analysis"
echo "========================================================================"
echo "Method: ${METHOD}"
echo "Device: ${DEVICE}"
echo "Project root: ${PROJECT_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Setting directories:"
echo "  Setting1: ${SETTING1_DIR}"
echo "  Setting2: ${SETTING2_DIR}"
echo "  Setting3: ${SETTING3_DIR}"
echo ""
echo "Model checkpoints:"
echo "  Setting1: ${SETTING1_CKPT:-NOT FOUND}"
echo "  Setting2: ${SETTING2_CKPT:-NOT FOUND}"
echo "  Setting3: ${SETTING3_CKPT:-NOT FOUND}"
echo "========================================================================"
echo ""

# Check if at least Setting2 checkpoint exists (required)
if [ -z "${SETTING2_CKPT}" ] || [ ! -f "${SETTING2_CKPT}" ]; then
    echo "Error: Setting2 checkpoint not found!"
    echo "Expected location: ${SETTING2_DIR}/checkpoints/sb_mlplus/best_model.pt"
    echo "                or ${SETTING2_DIR}/checkpoints/sb/best_model.pt"
    echo ""
    echo "Please train Setting2 model first:"
    echo "  python step1_run_experiment.py --config configs/experiment_EMT_Part1_setting2.yaml"
    exit 1
fi

# Check if Setting1 checkpoint exists (optional but recommended)
if [ -z "${SETTING1_CKPT}" ] || [ ! -f "${SETTING1_CKPT}" ]; then
    echo "Warning: Setting1 checkpoint not found. Comparison with Setting1 will be skipped."
    echo "Expected location: ${SETTING1_DIR}/checkpoints/sb_mlplus/best_model.pt"
    echo ""
fi

# Run analysis (using project's config system)
python run_entropy_analysis.py \
    --config "${PROJECT_ROOT}/configs/experiment_EMT_Part1_setting7_entropy.yaml" \
    --config_dir "${PROJECT_ROOT}/configs" \
    --setting1_checkpoint "${SETTING1_CKPT}" \
    --setting2_checkpoint "${SETTING2_CKPT}" \
    --setting3_checkpoint "${SETTING3_CKPT}" \
    --method "${METHOD}" \
    --k "${K_NEIGHBORS}" \
    --n_samples "${N_SAMPLES}" \
    --n_steps "${N_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --cross_validate_methods

echo ""
echo "========================================================================"
echo "✓ Analysis completed!"
echo "✓ Results saved to: ${OUTPUT_DIR}"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - entropy_curves_comparison.png/pdf"
echo "  - peak_characteristics_comparison.png/pdf"
echo "  - method_cross_validation.png (if cross-validation enabled)"
echo "  - entropy_analysis_summary.json"
echo "  - entropy_analysis_full_results.pkl"
echo ""
