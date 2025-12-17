#!/bin/bash
################################################################################
# Run Setting 5 Experiment: Label-Shuffled Time Series (Paul EMT)
################################################################################
# This script runs the Setting 5 experiment which tests whether models rely on
# absolute time labels or can learn dynamics from data geometry.
#
# Design:
#   - Keep boundary timepoints (T0, T7)
#   - Shuffle intermediate timepoints (T1, T2, T3, T4, T5, T6)
#   - Train models as in Setting 2
#   - Evaluate on true labels
#
# Research Question:
#   Does the model depend on absolute time labels, or can it learn from
#   data structure even when labels are randomized?
#
# Usage:
#   bash step2_train_setting5.sh
#
# Author: Shi Pan
# Date: 2024-12-16
################################################################################

set -e  # Exit on error

# Get the directory where this script is located (EXPs/Paul/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the EXPs directory (parent of Paul)
EXP_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$EXP_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

echo "================================================================================"
echo "Setting 5 Experiment: Label-Shuffled Time Series (Paul EMT)"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo ""
echo "Research Question:"
echo "  Does the model rely on absolute time labels, or can it learn dynamics"
echo "  from data geometry even when time labels are randomized?"
echo ""
echo "Design:"
echo "  1. Keep boundary timepoints (T0, T7)"
echo "  2. Shuffle intermediate timepoints (T1, T2, T3, T4, T5, T6)"
echo "  3. Train models: sb_mlplus, batch_ot, vae"
echo "  4. Evaluate on true labels (T0 → T7)"
echo ""
echo "================================================================================"
echo ""

# Configuration
CONFIG_FILE="experiment_Paul_setting5_label_shuffled.yaml"
CONFIG_DIR="$PROJECT_ROOT/configs/Paul"
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/Paul/Setting5"

# Check if config file exists
if [ ! -f "$CONFIG_DIR/$CONFIG_FILE" ]; then
    echo "❌ ERROR: Configuration file not found: $CONFIG_DIR/$CONFIG_FILE"
    exit 1
fi

echo "✓ Configuration file found: $CONFIG_FILE"
echo "✓ Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run experiment
echo "================================================================================"
echo "Starting Experiment..."
echo "================================================================================"
echo ""

python Workers/step2_run_exp_setting5.py \
    "$CONFIG_FILE" \
    --config_dir "$CONFIG_DIR" \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Experiment Completed Successfully"
    echo "================================================================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - results.json                    (Experiment results)"
    echo "  - logs/experiment.log             (Detailed logs)"
    echo "  - checkpoints/                    (Model checkpoints)"
    echo "  - visualizations/                 (Plots and figures)"
    echo "  - metrics/                        (Evaluation metrics)"
    echo ""
    echo "Next steps:"
    echo "  1. Compare with Setting 2 results to quantify label dependency"
    echo "  2. Analyze label confusion matrix"
    echo "  3. Check if model learned true temporal order from data geometry"
    echo ""
else
    echo "❌ Experiment Failed"
    echo "================================================================================"
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo "Check logs for details: $OUTPUT_DIR/logs/experiment.log"
    echo ""
fi

echo "================================================================================"

exit $EXIT_CODE
