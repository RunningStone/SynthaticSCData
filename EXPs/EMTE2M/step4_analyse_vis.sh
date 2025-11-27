#!/bin/bash
################################################################################
# Step 4: Run Visualization for All Settings
################################################################################
# This script runs the visualization pipeline for all experiment settings,
# generating metrics comparison plots and trajectory visualizations.
#
# Usage:
#   bash step4_analyse_vis.sh
#
# Output:
#   - {OUTPUT_DIR}/vis/metrics_comparison.png/pdf/csv
#   - {OUTPUT_DIR}/vis/generation_comparison_phate.png/pdf
#   - {OUTPUT_DIR}/vis/generation_comparison_lmnn_pca.png/pdf
#   - {OUTPUT_DIR}/vis/embeddings.pkl (cached for reuse)
#
################################################################################

set -e  # Exit on error

# Get the directory where this script is located (EXPs/EMTE2M/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

echo "================================================================================"
echo "Step 4: Multi-Setting Visualization"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Configuration
OUTPUT_BASE="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M"
VIS_OUTPUT="$OUTPUT_BASE/vis"

echo "Configuration:"
echo "  Experiment base: $OUTPUT_BASE"
echo "  Visualization output: $VIS_OUTPUT"
echo ""

# Activate virtual environment if exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activating virtual environment (venv)..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Check if base directory exists
if [ ! -d "$OUTPUT_BASE" ]; then
    echo "❌ ERROR: Experiment directory not found: $OUTPUT_BASE"
    exit 1
fi

# Install visualization dependencies if needed
echo ""
echo "Checking visualization dependencies..."
pip install phate metric-learn pandas -q 2>/dev/null || true

echo ""
echo "================================================================================"
echo "Running Visualization Pipeline..."
echo "================================================================================"
echo ""

python "$PROJECT_ROOT/Workers/step4_run_vis.py" \
    --experiment_dir "$OUTPUT_BASE" \
    --output_dir "$VIS_OUTPUT" \
    --force_recompute

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Visualization Completed Successfully"
    echo "================================================================================"
    echo ""
    echo "Results saved to: $VIS_OUTPUT"
    echo ""
    echo "Generated files:"
    echo "  - metrics_comparison.png/pdf/csv  : Metrics comparison across settings"
    echo "  - generation_comparison_phate.png/pdf : PHATE trajectory visualization"
    echo "  - generation_comparison_lmnn_pca.png/pdf : LMNN+PCA trajectory visualization"
    echo "  - embeddings.pkl : Cached embeddings for reuse"
    echo ""
else
    echo "❌ Visualization Failed"
    echo "================================================================================"
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo "Check logs for details: $VIS_OUTPUT/logs/"
    echo ""
fi

exit $EXIT_CODE
