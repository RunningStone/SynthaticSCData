#!/bin/bash
################################################################################
# Run all Setting4 ablation studies for GSE213069
################################################################################
# This script runs all three ablation experiments for Setting4 timepoint
# ablation study. It can run experiments sequentially or in parallel.
#
# Usage:
#   bash step2_train_setting4_ablations.sh [sequential|parallel]
#
# Default: sequential
#
# Author: Shi Pan
# Date: 2024-12-15
################################################################################

set -e  # Exit on error

# Get the directory where this script is located (EXPs/GSE213069/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the EXPs directory (parent of GSE213069)
EXP_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$EXP_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Mode: sequential or parallel
MODE="${1:-sequential}"

echo "================================================================================"
echo "Setting4 Ablation Study: Running All Experiments (GSE213069)"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Mode: $MODE"
echo ""

# Configuration files
CONFIG_DIR="$PROJECT_ROOT/configs/GSE123069"
OUTPUT_BASE="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/GSE213069_V1"

CONFIG_REMOVE_DAY2="experiment_GSE213069_setting4_ablation_remove_day2.yaml"
CONFIG_REMOVE_DAY3="experiment_GSE213069_setting4_ablation_remove_day3.yaml"
CONFIG_REMOVE_DAY4="experiment_GSE213069_setting4_ablation_remove_day4.yaml"

OUTPUT_REMOVE_DAY2="$OUTPUT_BASE/Setting4_Ablation_RemoveDay2"
OUTPUT_REMOVE_DAY3="$OUTPUT_BASE/Setting4_Ablation_RemoveDay3"
OUTPUT_REMOVE_DAY4="$OUTPUT_BASE/Setting4_Ablation_RemoveDay4"

# Check if configs exist
for config in "$CONFIG_REMOVE_DAY2" "$CONFIG_REMOVE_DAY3" "$CONFIG_REMOVE_DAY4"; do
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
    run_experiment "$CONFIG_REMOVE_DAY2" "$OUTPUT_REMOVE_DAY2" "Remove day2" &
    PID_DAY2=$!
    echo "Started Remove day2 (PID: $PID_DAY2)"
    
    run_experiment "$CONFIG_REMOVE_DAY3" "$OUTPUT_REMOVE_DAY3" "Remove day3" &
    PID_DAY3=$!
    echo "Started Remove day3 (PID: $PID_DAY3)"
    
    run_experiment "$CONFIG_REMOVE_DAY4" "$OUTPUT_REMOVE_DAY4" "Remove day4" &
    PID_DAY4=$!
    echo "Started Remove day4 (PID: $PID_DAY4)"
    
    echo ""
    echo "Waiting for all experiments to complete..."
    echo ""
    
    # Wait for all background jobs
    wait $PID_DAY2
    EXIT_DAY2=$?
    echo "Remove day2 completed (exit code: $EXIT_DAY2)"
    
    wait $PID_DAY3
    EXIT_DAY3=$?
    echo "Remove day3 completed (exit code: $EXIT_DAY3)"
    
    wait $PID_DAY4
    EXIT_DAY4=$?
    echo "Remove day4 completed (exit code: $EXIT_DAY4)"
    
    # Check if all succeeded
    if [ $EXIT_DAY2 -eq 0 ] && [ $EXIT_DAY3 -eq 0 ] && [ $EXIT_DAY4 -eq 0 ]; then
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
    echo "1/3: Remove day2 Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_DAY2" "$OUTPUT_REMOVE_DAY2" "Remove day2"
    
    echo "================================================================================"
    echo "2/3: Remove day3 Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_DAY3" "$OUTPUT_REMOVE_DAY3" "Remove day3"
    
    echo "================================================================================"
    echo "3/3: Remove day4 Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_DAY4" "$OUTPUT_REMOVE_DAY4" "Remove day4"
    
    echo "✅ All experiments completed successfully!"
fi

echo ""
echo "================================================================================"
echo "Next Step: Run Analysis"
echo "================================================================================"
echo ""
echo "To analyze the marginal contribution of each timepoint, run:"
echo ""
echo "  bash EXPs/GSE213069/step4_analyze_ablation.sh"
echo ""
echo "Note: Make sure Setting2 (full model) has been trained before running analysis."
echo "================================================================================"
