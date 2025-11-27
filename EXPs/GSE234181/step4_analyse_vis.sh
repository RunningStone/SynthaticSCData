#!/bin/bash
# ============================================================================
# Step 4: Run Visualization and Analysis for GSE234181
# ============================================================================
#
# This script generates visualizations and analysis for the GSE234181 experiment.
#
# Usage:
#   bash step4_analyse_vis.sh
#
# ============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Configuration
BASE_OUTPUT="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/GSE234181"
VIS_OUTPUT="$BASE_OUTPUT/vis"

echo "========================================================================"
echo "Step 4: Visualization and Analysis (GSE234181)"
echo "========================================================================"
echo ""
echo "Base output directory: ${BASE_OUTPUT}"
echo "Visualization output: ${VIS_OUTPUT}"
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Create visualization output directory
mkdir -p "$VIS_OUTPUT"

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
fi

echo ""
echo "========================================================================"
echo "Running Visualizations..."
echo "========================================================================"
echo ""

# Run naive visualization (basic plots)
echo "----------------------------------------"
echo "1. Running Naive Visualization..."
echo "----------------------------------------"
python Workers/step4_run_vis_naive.py \
    --base_output "$BASE_OUTPUT" \
    --vis_output "$VIS_OUTPUT/naive" \
    --config_dir "$PROJECT_ROOT/configs/GSE234181" \
    2>&1 || echo "⚠️  Naive visualization had some issues (continuing...)"

echo ""
echo "========================================================================"
echo "Visualization Complete"
echo "========================================================================"
echo ""
echo "Output files saved to: $VIS_OUTPUT"
echo ""
echo "Generated visualizations:"
echo "  - $VIS_OUTPUT/naive/        : Basic trajectory plots"
echo ""
echo "Next steps:"
echo "  1. Review generated figures"
echo "  2. Run ablation analysis: bash step4_analyze_ablation.sh"
echo ""
