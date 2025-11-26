#!/bin/bash
################################################################################
# Run all Setting4 ablation studies
################################################################################
# This script runs all three ablation experiments for Setting4 timepoint
# ablation study. It can run experiments sequentially or in parallel.
#
# Usage:
#   bash step2_run_ablations_setting4.sh [sequential|parallel]
#
# Default: sequential
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

# Mode: sequential or parallel
MODE="${1:-sequential}"

echo "================================================================================"
echo "Setting4 Ablation Study: Running All Experiments"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Mode: $MODE"
echo ""

# Configuration files
CONFIG_DIR="$PROJECT_ROOT/configs/EMT_E2M"
OUTPUT_BASE="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M"

CONFIG_REMOVE_8H="experiment_EMT_Part1_setting4_ablation_remove_8h.yaml"
CONFIG_REMOVE_1D="experiment_EMT_Part1_setting4_ablation_remove_1d.yaml"
CONFIG_REMOVE_3D="experiment_EMT_Part1_setting4_ablation_remove_3d.yaml"

OUTPUT_REMOVE_8H="$OUTPUT_BASE/Setting4_Ablation_Remove8h"
OUTPUT_REMOVE_1D="$OUTPUT_BASE/Setting4_Ablation_Remove1d"
OUTPUT_REMOVE_3D="$OUTPUT_BASE/Setting4_Ablation_Remove3d"

# Check if configs exist
for config in "$CONFIG_REMOVE_8H" "$CONFIG_REMOVE_1D" "$CONFIG_REMOVE_3D"; do
    if [ ! -f "$CONFIG_DIR/$config" ]; then
        echo "❌ ERROR: Configuration file not found: $CONFIG_DIR/$config"
        exit 1
    fi
done

echo "✓ All configuration files found"
echo ""

# Function to run a single experiment
run_experiment() {
    local config_file=$1
    local output_dir=$2
    local label=$3
    
    echo "----------------------------------------"
    echo "Running: $label"
    echo "----------------------------------------"
    echo "Config: $config_file"
    echo "Output: $output_dir"
    echo ""
    
    python Workers/step2_run_experiment.py \
        "$config_file" \
        --config_dir "$CONFIG_DIR" \
        --output_dir "$output_dir"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✅ $label completed successfully"
        echo ""
    else
        echo ""
        echo "❌ $label failed (exit code: $exit_code)"
        echo ""
        return $exit_code
    fi
}

if [ "$MODE" = "parallel" ]; then
    echo "Running experiments in PARALLEL mode..."
    echo "⚠️  WARNING: This requires 3 GPUs or sufficient GPU memory for concurrent training."
    echo ""
    
    # Run in background
    run_experiment "$CONFIG_REMOVE_8H" "$OUTPUT_REMOVE_8H" "Remove 8h" &
    PID_8H=$!
    echo "Started Remove 8h (PID: $PID_8H)"
    
    run_experiment "$CONFIG_REMOVE_1D" "$OUTPUT_REMOVE_1D" "Remove 1d" &
    PID_1D=$!
    echo "Started Remove 1d (PID: $PID_1D)"
    
    run_experiment "$CONFIG_REMOVE_3D" "$OUTPUT_REMOVE_3D" "Remove 3d" &
    PID_3D=$!
    echo "Started Remove 3d (PID: $PID_3D)"
    
    echo ""
    echo "Waiting for all experiments to complete..."
    echo ""
    
    # Wait for all background jobs
    wait $PID_8H
    EXIT_8H=$?
    echo "Remove 8h completed (exit code: $EXIT_8H)"
    
    wait $PID_1D
    EXIT_1D=$?
    echo "Remove 1d completed (exit code: $EXIT_1D)"
    
    wait $PID_3D
    EXIT_3D=$?
    echo "Remove 3d completed (exit code: $EXIT_3D)"
    
    # Check if all succeeded
    if [ $EXIT_8H -eq 0 ] && [ $EXIT_1D -eq 0 ] && [ $EXIT_3D -eq 0 ]; then
        echo ""
        echo "✅ All experiments completed successfully!"
    else
        echo ""
        echo "❌ Some experiments failed. Check logs for details."
        exit 1
    fi
    
else
    echo "Running experiments in SEQUENTIAL mode..."
    echo ""
    
    # Run sequentially
    echo "================================================================================"
    echo "1/3: Remove 8h Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_8H" "$OUTPUT_REMOVE_8H" "Remove 8h"
    
    echo "================================================================================"
    echo "2/3: Remove 1d Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_1D" "$OUTPUT_REMOVE_1D" "Remove 1d"
    
    echo "================================================================================"
    echo "3/3: Remove 3d Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_3D" "$OUTPUT_REMOVE_3D" "Remove 3d"
    
    echo "✅ All experiments completed successfully!"
fi

echo ""
echo "================================================================================"
echo "Next Step: Run Analysis"
echo "================================================================================"
echo ""
echo "To analyze the marginal contribution of each timepoint, run:"
echo ""
echo "  bash EXPs/EMTE2M/step3_analyze_ablation.sh"
echo ""
echo "Note: Make sure Setting2 (full model) has been trained before running analysis."
echo "================================================================================"
