#!/bin/bash
################################################################################
# Run all Setting4 ablation studies for Paul EMT
################################################################################
# This script runs all six ablation experiments for Setting4 timepoint
# ablation study. It can run experiments sequentially or in parallel.
#
# Usage:
#   bash step2_train_setting4_ablations.sh [sequential|parallel]
#
# Default: sequential
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

# Mode: sequential or parallel
MODE="${1:-sequential}"

echo "================================================================================"
echo "Setting4 Ablation Study: Running All Experiments (Paul EMT)"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Mode: $MODE"
echo ""

# Configuration files
CONFIG_DIR="$PROJECT_ROOT/configs/Paul"
OUTPUT_BASE="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/Paul"

CONFIG_REMOVE_T1="experiment_Paul_setting4_ablation_remove_T1.yaml"
CONFIG_REMOVE_T2="experiment_Paul_setting4_ablation_remove_T2.yaml"
CONFIG_REMOVE_T3="experiment_Paul_setting4_ablation_remove_T3.yaml"
CONFIG_REMOVE_T4="experiment_Paul_setting4_ablation_remove_T4.yaml"
CONFIG_REMOVE_T5="experiment_Paul_setting4_ablation_remove_T5.yaml"
CONFIG_REMOVE_T6="experiment_Paul_setting4_ablation_remove_T6.yaml"

OUTPUT_REMOVE_T1="$OUTPUT_BASE/Setting4_Ablation_RemoveT1"
OUTPUT_REMOVE_T2="$OUTPUT_BASE/Setting4_Ablation_RemoveT2"
OUTPUT_REMOVE_T3="$OUTPUT_BASE/Setting4_Ablation_RemoveT3"
OUTPUT_REMOVE_T4="$OUTPUT_BASE/Setting4_Ablation_RemoveT4"
OUTPUT_REMOVE_T5="$OUTPUT_BASE/Setting4_Ablation_RemoveT5"
OUTPUT_REMOVE_T6="$OUTPUT_BASE/Setting4_Ablation_RemoveT6"

# Check if configs exist
for config in "$CONFIG_REMOVE_T1" "$CONFIG_REMOVE_T2" "$CONFIG_REMOVE_T3" "$CONFIG_REMOVE_T4" "$CONFIG_REMOVE_T5" "$CONFIG_REMOVE_T6"; do
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
    echo "⚠️  WARNING: This requires multiple GPUs or sufficient GPU memory for concurrent training."
    echo ""
    
    # Run in background
    run_experiment "$CONFIG_REMOVE_T1" "$OUTPUT_REMOVE_T1" "Remove T1" &
    PID_T1=$!
    echo "Started Remove T1 (PID: $PID_T1)"
    
    run_experiment "$CONFIG_REMOVE_T2" "$OUTPUT_REMOVE_T2" "Remove T2" &
    PID_T2=$!
    echo "Started Remove T2 (PID: $PID_T2)"
    
    run_experiment "$CONFIG_REMOVE_T3" "$OUTPUT_REMOVE_T3" "Remove T3" &
    PID_T3=$!
    echo "Started Remove T3 (PID: $PID_T3)"
    
    run_experiment "$CONFIG_REMOVE_T4" "$OUTPUT_REMOVE_T4" "Remove T4" &
    PID_T4=$!
    echo "Started Remove T4 (PID: $PID_T4)"
    
    run_experiment "$CONFIG_REMOVE_T5" "$OUTPUT_REMOVE_T5" "Remove T5" &
    PID_T5=$!
    echo "Started Remove T5 (PID: $PID_T5)"
    
    run_experiment "$CONFIG_REMOVE_T6" "$OUTPUT_REMOVE_T6" "Remove T6" &
    PID_T6=$!
    echo "Started Remove T6 (PID: $PID_T6)"
    
    echo ""
    echo "Waiting for all experiments to complete..."
    echo ""
    
    # Wait for all background jobs
    wait $PID_T1; EXIT_T1=$?
    wait $PID_T2; EXIT_T2=$?
    wait $PID_T3; EXIT_T3=$?
    wait $PID_T4; EXIT_T4=$?
    wait $PID_T5; EXIT_T5=$?
    wait $PID_T6; EXIT_T6=$?
    
    # Check if all succeeded
    if [ $EXIT_T1 -eq 0 ] && [ $EXIT_T2 -eq 0 ] && [ $EXIT_T3 -eq 0 ] && [ $EXIT_T4 -eq 0 ] && [ $EXIT_T5 -eq 0 ] && [ $EXIT_T6 -eq 0 ]; then
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
    echo "1/6: Remove T1 Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_T1" "$OUTPUT_REMOVE_T1" "Remove T1"
    
    echo "================================================================================"
    echo "2/6: Remove T2 Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_T2" "$OUTPUT_REMOVE_T2" "Remove T2"
    
    echo "================================================================================"
    echo "3/6: Remove T3 Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_T3" "$OUTPUT_REMOVE_T3" "Remove T3"
    
    echo "================================================================================"
    echo "4/6: Remove T4 Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_T4" "$OUTPUT_REMOVE_T4" "Remove T4"
    
    echo "================================================================================"
    echo "5/6: Remove T5 Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_T5" "$OUTPUT_REMOVE_T5" "Remove T5"
    
    echo "================================================================================"
    echo "6/6: Remove T6 Ablation"
    echo "================================================================================"
    run_experiment "$CONFIG_REMOVE_T6" "$OUTPUT_REMOVE_T6" "Remove T6"
    
    echo "✅ All experiments completed successfully!"
fi

echo ""
echo "================================================================================"
echo "Next Step: Run Analysis"
echo "================================================================================"
echo ""
echo "To analyze the marginal contribution of each timepoint, run:"
echo ""
echo "  bash EXPs/Paul/step3_analyze_ablation.sh"
echo ""
echo "Note: Make sure Setting2 (full model) has been trained before running analysis."
echo "================================================================================"
