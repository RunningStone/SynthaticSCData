#!/bin/bash
################################################################################
# Analyze Setting4 Ablation Study Results
################################################################################
# This script analyzes the marginal contribution of each timepoint by comparing
# the performance of models trained with and without that timepoint.
#
# Prerequisites:
#   1. Setting2 (full model) must be trained
#   2. All three ablation experiments must be completed:
#      - Remove 8h
#      - Remove 1d
#      - Remove 3d
#
# Usage:
#   bash step3_analyze_ablation.sh
#
# Author: Shi Pan
# Date: 2024-11-24
################################################################################

set -e  # Exit on error

# Get the directory where this script is located (EXPs/EMTE2M/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the EXPs directory (parent of EMTE2M)
EXP_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$EXP_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

echo "================================================================================"
echo "Setting4 Ablation Study: Marginal Contribution Analysis"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# Configuration
OUTPUT_BASE="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M"
ANALYSIS_OUTPUT="$OUTPUT_BASE/Setting4_Ablation_Analysis"
MODEL_NAME="sb_mlplus"

# Experiment names
FULL_EXP="Setting2"
ABLATION_8H="Setting4_Ablation_Remove8h"
ABLATION_1D="Setting4_Ablation_Remove1d"
ABLATION_3D="Setting4_Ablation_Remove3d"

echo "Configuration:"
echo "  Output base: $OUTPUT_BASE"
echo "  Analysis output: $ANALYSIS_OUTPUT"
echo "  Model: $MODEL_NAME"
echo ""
echo "Experiments to analyze:"
echo "  Full model: $FULL_EXP"
echo "  Ablation (remove 8h): $ABLATION_8H"
echo "  Ablation (remove 1d): $ABLATION_1D"
echo "  Ablation (remove 3d): $ABLATION_3D"
echo ""

# Check if all required experiments exist
echo "Checking prerequisites..."
MISSING=0

for exp in "$FULL_EXP" "$ABLATION_8H" "$ABLATION_1D" "$ABLATION_3D"; do
    RESULTS_FILE="$OUTPUT_BASE/$exp/results.json"
    if [ ! -f "$RESULTS_FILE" ]; then
        echo "  ❌ Missing: $RESULTS_FILE"
        MISSING=1
    else
        echo "  ✓ Found: $exp"
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "❌ ERROR: Some required experiments are missing."
    echo ""
    echo "Please ensure the following experiments have been completed:"
    echo "  1. Setting2 (full model with all timepoints)"
    echo "  2. All three ablation experiments (run: bash step2_run_ablations_setting4.sh)"
    echo ""
    exit 1
fi

echo ""
echo "✓ All prerequisites satisfied"
echo ""

# Create analysis output directory
mkdir -p "$ANALYSIS_OUTPUT"

# Run analysis using Python
echo "================================================================================"
echo "Running Analysis..."
echo "================================================================================"
echo ""

python -c "
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, '$PROJECT_ROOT')

from Analyser import AblationAnalyzer

# Create analyzer
analyzer = AblationAnalyzer(
    output_dir='$ANALYSIS_OUTPUT',
    device='cpu',
    random_seed=42
)

# Run analysis
analyzer.run_analysis(
    output_base=Path('$OUTPUT_BASE'),
    full_exp_name='$FULL_EXP',
    ablation_exp_names={
        '8h': '$ABLATION_8H',
        '1d': '$ABLATION_1D',
        '3d': '$ABLATION_3D'
    },
    model_name='$MODEL_NAME'
)
"

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Analysis Complete"
    echo "================================================================================"
    echo ""
    echo "Results saved to: $ANALYSIS_OUTPUT"
    echo ""
    echo "Generated files:"
    echo "  - delta_P.csv                          (Absolute marginal contributions)"
    echo "  - I_margin.csv                         (Relative marginal contributions %)"
    echo "  - consistency.csv                      (Cross-metric consistency)"
    echo "  - critical_timepoints.json             (Critical timepoints identification)"
    echo "  - marginal_contribution_absolute.png   (Bar plots for each metric)"
    echo "  - marginal_contribution_heatmap.png    (Heatmap of relative contributions)"
    echo "  - marginal_contribution_summary.png    (Average contribution summary)"
    echo "  - ablation_analysis_report.txt         (Comprehensive text report)"
    echo ""
    echo "Next steps:"
    echo "  1. Review the report: cat $ANALYSIS_OUTPUT/ablation_analysis_report.txt"
    echo "  2. Examine visualizations in: $ANALYSIS_OUTPUT/"
    echo "  3. Check numerical results: $ANALYSIS_OUTPUT/*.csv"
    echo ""
else
    echo "❌ Analysis Failed"
    echo "================================================================================"
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo "Please check the error messages above for details."
    echo ""
fi

exit $EXIT_CODE
