#!/bin/bash
################################################################################
# Step 4: Generate All Publication Figures
################################################################################
# This script runs all three figure generation scripts to create:
#   - Figure 1: Core Performance Comparison (Radar + PHATE grid)
#   - Figure 2: Timepoint Ablation Analysis (Bar + Heatmap + Scatter)
#   - Figure 3: Causal and Interpolation Analysis (Comparison + Ladder)
#
# Usage:
#   bash step4_generate_all_figures.sh
#
# Output:
#   All figures saved to: {OUTPUT_DIR}/vis/
#     - Fig1_1.pdf, Fig1_2.pdf
#     - Fig2_1.pdf, Fig2_2.pdf, Fig2_3.pdf
#     - Fig3_1.pdf, Fig3_2.pdf
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
echo "Step 4: Generate All Publication Figures"
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
pip install phate metric-learn pandas seaborn -q 2>/dev/null || true

# ================================================================================
# Figure 1: Core Performance Comparison
# ================================================================================
echo ""
echo "================================================================================"
echo "Generating Figure 1: Core Performance Comparison"
echo "================================================================================"
echo ""

python "$PROJECT_ROOT/Workers/step4_run_vis_fig1.py" \
    --experiment_dir "$OUTPUT_BASE" \
    --force_recompute

FIG1_EXIT=$?
if [ $FIG1_EXIT -ne 0 ]; then
    echo "❌ Figure 1 generation failed"
    exit $FIG1_EXIT
fi

# ================================================================================
# Figure 2: Timepoint Ablation Analysis
# ================================================================================
echo ""
echo "================================================================================"
echo "Generating Figure 2: Timepoint Ablation Analysis"
echo "================================================================================"
echo ""

python "$PROJECT_ROOT/Workers/step4_run_vis_fig2.py" \
    --experiment_dir "$OUTPUT_BASE" \
    --model sb_mlplus

FIG2_EXIT=$?
if [ $FIG2_EXIT -ne 0 ]; then
    echo "❌ Figure 2 generation failed"
    exit $FIG2_EXIT
fi

# ================================================================================
# Figure 3: Causal and Interpolation Analysis
# ================================================================================
echo ""
echo "================================================================================"
echo "Generating Figure 3: Causal and Interpolation Analysis"
echo "================================================================================"
echo ""

python "$PROJECT_ROOT/Workers/step4_run_vis_fig3.py" \
    --experiment_dir "$OUTPUT_BASE"

FIG3_EXIT=$?
if [ $FIG3_EXIT -ne 0 ]; then
    echo "❌ Figure 3 generation failed"
    exit $FIG3_EXIT
fi

# ================================================================================
# Summary
# ================================================================================
echo ""
echo "================================================================================"
echo "✅ All Figures Generated Successfully!"
echo "================================================================================"
echo ""
echo "Output directory: $VIS_OUTPUT"
echo ""
echo "Generated figures:"
echo "  Figure 1 (Core Performance):"
echo "    - Fig1_1.pdf  : Performance radar chart (Setting1 vs Setting2)"
echo "    - Fig1_2.pdf  : PHATE 3x3 grid (generation quality)"
echo ""
echo "  Figure 2 (Ablation Analysis):"
echo "    - Fig2_1.pdf  : Ablation bar chart (performance degradation)"
echo "    - Fig2_2.pdf  : Sensitivity heatmap (metric × ablation)"
echo "    - Fig2_3.pdf  : Entropy vs marginal contribution scatter"
echo ""
echo "  Figure 3 (Causal & Interpolation):"
echo "    - Fig3_1.pdf  : Sequential vs shuffled comparison"
echo "    - Fig3_2.pdf  : Interpolation ladder (S1 → S6 → S2)"
echo ""

exit 0
