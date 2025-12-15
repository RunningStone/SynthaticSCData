#!/bin/bash
################################################################################
# Step 4: Generate All Publication Figures - EMT_E2M2E Dataset
################################################################################
# This script runs all figure generation scripts to create:
#   - Figure 1: Core Performance Comparison (Radar + PHATE grid)
#   - Figure 2: Timepoint Ablation Analysis (Bar + Heatmap + Scatter)
#   - Figure 3: Causal and Interpolation Analysis (Comparison + Ladder)
#
# Dataset: EMT_E2M2E (Full EMT Trajectory with Reversal)
# Timepoints: 0d → 8h → 1d → 3d → 7d → 8h_rm → 1d_rm → 3d_rm
#
# Usage:
#   bash step4_generate_all_figures.sh
#
# Output:
#   All figures saved to: {OUTPUT_DIR}/vis/
#     - Figure1/Fig1_1.pdf, Fig1_2.pdf, Fig1_Set3_PHATE.pdf
#     - Figure2/Fig2_1.pdf, Fig2_2.pdf, Fig2_3.pdf, Fig2_PHATE.pdf
#     - Figure3/Fig3_1.pdf, Fig3_2.pdf
#
################################################################################

set -e  # Exit on error

# Get the directory where this script is located (EXPs/EMT_E2M2E/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

echo "================================================================================"
echo "Step 4: Generate All Publication Figures - EMT_E2M2E"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Configuration
OUTPUT_BASE="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M2E"
VIS_OUTPUT="$OUTPUT_BASE/vis"

echo "Configuration:"
echo "  Dataset: EMT_E2M2E (Full EMT with Reversal)"
echo "  Timepoints: 0d → 8h → 1d → 3d → 7d → 8h_rm → 1d_rm → 3d_rm"
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
    echo "Please run step2 and step3 first to generate experiment results."
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
# Figure 1 Extended: PHATE with Setting3
# ================================================================================
echo ""
echo "================================================================================"
echo "Generating Figure 1 Extended: PHATE with Setting1, Setting2, Setting3"
echo "================================================================================"
echo ""

# Check if Setting3 exists before running
if [ -d "$OUTPUT_BASE/Setting3" ]; then
    python "$PROJECT_ROOT/Workers/step4_run_vis_fig1_set3_phate.py" \
        --experiment_dir "$OUTPUT_BASE" \
        --force_recompute
    
    FIG1_SET3_EXIT=$?
    if [ $FIG1_SET3_EXIT -ne 0 ]; then
        echo "⚠️ Figure 1 Set3 PHATE generation failed (non-critical)"
    fi
else
    echo "⚠️ Setting3 not found, skipping Fig1_Set3_PHATE"
fi

# ================================================================================
# Figure 2: Timepoint Ablation Analysis
# ================================================================================
echo ""
echo "================================================================================"
echo "Generating Figure 2: Timepoint Ablation Analysis"
echo "================================================================================"
echo ""

# Check if Setting4 exists before running
if [ -d "$OUTPUT_BASE/Setting4" ]; then
    python "$PROJECT_ROOT/Workers/step4_run_vis_fig2.py" \
        --experiment_dir "$OUTPUT_BASE" \
        --model sb_mlplus
    
    FIG2_EXIT=$?
    if [ $FIG2_EXIT -ne 0 ]; then
        echo "❌ Figure 2 generation failed"
        exit $FIG2_EXIT
    fi
    
    # Figure 2 PHATE
    echo ""
    echo "Generating Figure 2 PHATE..."
    python "$PROJECT_ROOT/Workers/step4_run_vis_fig2_phate.py" \
        --experiment_dir "$OUTPUT_BASE" \
        --model sb_mlplus \
        --force_recompute
    
    FIG2_PHATE_EXIT=$?
    if [ $FIG2_PHATE_EXIT -ne 0 ]; then
        echo "⚠️ Figure 2 PHATE generation failed (non-critical)"
    fi
else
    echo "⚠️ Setting4 not found, skipping Figure 2 (Ablation Analysis)"
fi

# ================================================================================
# Figure 3: Causal and Interpolation Analysis
# ================================================================================
echo ""
echo "================================================================================"
echo "Generating Figure 3: Causal and Interpolation Analysis"
echo "================================================================================"
echo ""

# Check if required settings exist
MISSING_SETTINGS=""
[ ! -d "$OUTPUT_BASE/Setting5" ] && MISSING_SETTINGS="$MISSING_SETTINGS Setting5"
[ ! -d "$OUTPUT_BASE/Setting6" ] && MISSING_SETTINGS="$MISSING_SETTINGS Setting6"

if [ -z "$MISSING_SETTINGS" ]; then
    python "$PROJECT_ROOT/Workers/step4_run_vis_fig3.py" \
        --experiment_dir "$OUTPUT_BASE"
    
    FIG3_EXIT=$?
    if [ $FIG3_EXIT -ne 0 ]; then
        echo "❌ Figure 3 generation failed"
        exit $FIG3_EXIT
    fi
else
    echo "⚠️ Missing settings for Figure 3:$MISSING_SETTINGS"
    echo "   Skipping Figure 3 (Causal and Interpolation Analysis)"
fi

# ================================================================================
# Summary
# ================================================================================
echo ""
echo "================================================================================"
echo "✅ Figure Generation Complete!"
echo "================================================================================"
echo ""
echo "Dataset: EMT_E2M2E (Full EMT Trajectory with Reversal)"
echo "Timepoints: 0d → 8h → 1d → 3d → 7d → 8h_rm → 1d_rm → 3d_rm"
echo ""
echo "Output directory: $VIS_OUTPUT"
echo ""
echo "Generated figures:"
echo "  Figure 1 (Core Performance):"
echo "    - Fig1_1.pdf  : Performance radar chart (Setting1 vs Setting2)"
echo "    - Fig1_2.pdf  : PHATE 3x3 grid (generation quality)"
if [ -d "$OUTPUT_BASE/Setting3" ]; then
echo "    - Fig1_Set3_PHATE.pdf : Extended PHATE with Setting3"
fi
echo ""
if [ -d "$OUTPUT_BASE/Setting4" ]; then
echo "  Figure 2 (Ablation Analysis):"
echo "    - Fig2_1.pdf  : Ablation bar chart (performance degradation)"
echo "    - Fig2_2.pdf  : Sensitivity heatmap (metric × ablation)"
echo "    - Fig2_3.pdf  : Entropy vs marginal contribution scatter"
echo "    - Fig2_PHATE.pdf : Ablation PHATE visualization"
echo ""
fi
if [ -z "$MISSING_SETTINGS" ]; then
echo "  Figure 3 (Causal & Interpolation):"
echo "    - Fig3_1.pdf  : Sequential vs shuffled comparison"
echo "    - Fig3_2.pdf  : Interpolation ladder (S1 → S6 → S2)"
echo ""
fi

exit 0
