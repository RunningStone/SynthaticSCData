#!/bin/bash
################################################################################
# Run all Setting4 ablation studies for GSE234181
################################################################################
# This script runs all two ablation experiments for Setting4 timepoint
# ablation study. It can run experiments sequentially or in parallel.
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

# Get the directory where this script is located (EXPs/GSE234181/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the EXPs directory (parent of GSE234181)
EXP_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$EXP_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Mode: sequential or parallel
MODE="${1:-sequential}"

echo "================================================================================"
echo "Setting4 Ablation Study: Running All Experiments (GSE234181)"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Mode: $MODE"
echo ""

# Configuration files
CONFIG_DIR="$PROJECT_ROOT/configs/GSE234181"
OUTPUT_BASE="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/GSE234181"

CONFIG_REMOVE_T1="experiment_GSE234181_setting4_ablation_remove_T1.yaml"
CONFIG_REMOVE_T2="experiment_GSE234181_setting4_ablation_remove_T2.yaml"

OUTPUT_REMOVE_T1="$OUTPUT_BASE/Setting4_Ablation_RemoveT1"
OUTPUT_REMOVE_T2="$OUTPUT_BASE/Setting4_Ablation_RemoveT2"

# Check if configs exist
for config in "$CONFIG_REMOVE_T1" "$CONFIG_REMOVE_T2"; do
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
    echo "⚠️  WARNING: This requires 2 GPUs or sufficient GPU memory for concurrent training."
    echo ""
    
    # Run in background
    run_experiment "$CONFIG_REMOVE_T1" "$OUTPUT_REMOVE_T1" "Remove T1" &
    PID_T1=$!
    echo "Started Remove T1 (PID: $PID_T1)"
    
    run_experiment "$CONFIG_REMOVE_T2" "$OUTPUT_REMOVE_T2" "Remove T2" &
    PID_T2=$!
    echo "Started Remove T2 (PID: $PID_T2)"
    
    echo ""
    echo "Waiting for all experiments to complete..."
    echo ""
    
    # Wait for all background jobs
    wait $PID_T1
    EXIT_T1=$?
    echo "Remove T1 completed (exit code: $EXIT_T1)"
    
    wait $PID_T2
    EXIT_T2=$?
    echo "Remove T2 completed (exit code: $EXIT_T2)"
    
    # Check if all succeeded
    if [ $EXIT_T1 -eq 0 ] && [ $EXIT_T2 -eq 0 ]; then
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
    echo "1/2: Remove T1 Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_T1" "$OUTPUT_REMOVE_T1" "Remove T1"
    
    echo "================================================================================"
    echo "2/2: Remove T2 Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_T2" "$OUTPUT_REMOVE_T2" "Remove T2"
    
    echo "✅ All experiments completed successfully!"
fi

echo ""
echo "================================================================================"
echo "Next Step: Run Analysis"
echo "================================================================================"
echo ""
echo "To analyze the marginal contribution of each timepoint, run:"
echo ""
echo "  bash EXPs/GSE234181/step4_analyze_ablation.sh"
echo ""
echo "Note: Make sure Setting2 (full model) has been trained before running analysis."
echo "================================================================================"
