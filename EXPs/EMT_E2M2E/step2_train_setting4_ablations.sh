#!/bin/bash
################################################################################
# Run all Setting4 ablation studies for EMT_E2M2E
################################################################################
# This script runs all six ablation experiments for Setting4 timepoint
# ablation study. It can run experiments sequentially or in parallel.
#
# Ablation targets:
#   - 8h: Remove 8h from forward EMT
#   - 1d: Remove 1d from forward EMT
#   - 3d: Remove 3d from forward EMT
#   - 7d: Remove 7d (peak EMT)
#   - 8h_rm: Remove 8h_rm from reversal
#   - 1d_rm: Remove 1d_rm from reversal
#
# Usage:
#   bash step2_train_setting4_ablations.sh [sequential|parallel]
#
# Default: sequential
#
# Author: Shi Pan
# Date: 2024-11-27
################################################################################

set -e  # Exit on error

# Get the directory where this script is located (EXPs/EMT_E2M2E/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the EXPs directory (parent of EMT_E2M2E)
EXP_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$EXP_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Mode: sequential or parallel
MODE="${1:-sequential}"

echo "================================================================================"
echo "Setting4 Ablation Study: Running All Experiments (EMT_E2M2E)"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Mode: $MODE"
echo ""

# Configuration files
CONFIG_DIR="$PROJECT_ROOT/configs/EMT_E2M2E"
OUTPUT_BASE="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M2E"

CONFIG_REMOVE_8H="experiment_EMT_E2M2E_setting4_ablation_remove_8h.yaml"
CONFIG_REMOVE_1D="experiment_EMT_E2M2E_setting4_ablation_remove_1d.yaml"
CONFIG_REMOVE_3D="experiment_EMT_E2M2E_setting4_ablation_remove_3d.yaml"
CONFIG_REMOVE_7D="experiment_EMT_E2M2E_setting4_ablation_remove_7d.yaml"
CONFIG_REMOVE_8H_RM="experiment_EMT_E2M2E_setting4_ablation_remove_8h_rm.yaml"
CONFIG_REMOVE_1D_RM="experiment_EMT_E2M2E_setting4_ablation_remove_1d_rm.yaml"

OUTPUT_REMOVE_8H="$OUTPUT_BASE/Setting4_Ablation_Remove8h"
OUTPUT_REMOVE_1D="$OUTPUT_BASE/Setting4_Ablation_Remove1d"
OUTPUT_REMOVE_3D="$OUTPUT_BASE/Setting4_Ablation_Remove3d"
OUTPUT_REMOVE_7D="$OUTPUT_BASE/Setting4_Ablation_Remove7d"
OUTPUT_REMOVE_8H_RM="$OUTPUT_BASE/Setting4_Ablation_Remove8h_rm"
OUTPUT_REMOVE_1D_RM="$OUTPUT_BASE/Setting4_Ablation_Remove1d_rm"

# Check if configs exist
for config in "$CONFIG_REMOVE_8H" "$CONFIG_REMOVE_1D" "$CONFIG_REMOVE_3D" \
              "$CONFIG_REMOVE_7D" "$CONFIG_REMOVE_8H_RM" "$CONFIG_REMOVE_1D_RM"; do
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
    echo "⚠️  WARNING: This requires 6 GPUs or sufficient GPU memory for concurrent training."
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
    
    run_experiment "$CONFIG_REMOVE_7D" "$OUTPUT_REMOVE_7D" "Remove 7d" &
    PID_7D=$!
    echo "Started Remove 7d (PID: $PID_7D)"
    
    run_experiment "$CONFIG_REMOVE_8H_RM" "$OUTPUT_REMOVE_8H_RM" "Remove 8h_rm" &
    PID_8H_RM=$!
    echo "Started Remove 8h_rm (PID: $PID_8H_RM)"
    
    run_experiment "$CONFIG_REMOVE_1D_RM" "$OUTPUT_REMOVE_1D_RM" "Remove 1d_rm" &
    PID_1D_RM=$!
    echo "Started Remove 1d_rm (PID: $PID_1D_RM)"
    
    echo ""
    echo "Waiting for all experiments to complete..."
    echo ""
    
    # Wait for all background jobs
    wait $PID_8H; EXIT_8H=$?
    wait $PID_1D; EXIT_1D=$?
    wait $PID_3D; EXIT_3D=$?
    wait $PID_7D; EXIT_7D=$?
    wait $PID_8H_RM; EXIT_8H_RM=$?
    wait $PID_1D_RM; EXIT_1D_RM=$?
    
    # Check if all succeeded
    if [ $EXIT_8H -eq 0 ] && [ $EXIT_1D -eq 0 ] && [ $EXIT_3D -eq 0 ] && \
       [ $EXIT_7D -eq 0 ] && [ $EXIT_8H_RM -eq 0 ] && [ $EXIT_1D_RM -eq 0 ]; then
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
    echo "1/6: Remove 8h Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_8H" "$OUTPUT_REMOVE_8H" "Remove 8h"
    
    echo "================================================================================"
    echo "2/6: Remove 1d Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_1D" "$OUTPUT_REMOVE_1D" "Remove 1d"
    
    echo "================================================================================"
    echo "3/6: Remove 3d Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_3D" "$OUTPUT_REMOVE_3D" "Remove 3d"
    
    echo "================================================================================"
    echo "4/6: Remove 7d Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_7D" "$OUTPUT_REMOVE_7D" "Remove 7d"
    
    echo "================================================================================"
    echo "5/6: Remove 8h_rm Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_8H_RM" "$OUTPUT_REMOVE_8H_RM" "Remove 8h_rm"
    
    echo "================================================================================"
    echo "6/6: Remove 1d_rm Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_1D_RM" "$OUTPUT_REMOVE_1D_RM" "Remove 1d_rm"
    
    echo "✅ All experiments completed successfully!"
fi

echo ""
echo "================================================================================"
echo "Next Step: Run Analysis"
echo "================================================================================"
echo ""
echo "To analyze the marginal contribution of each timepoint, run:"
echo ""
echo "  bash EXPs/EMT_E2M2E/step4_analyze_ablation.sh"
echo ""
echo "Note: Make sure Setting2 (full model) has been trained before running analysis."
echo "================================================================================"
