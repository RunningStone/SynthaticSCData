#!/bin/bash
################################################################################
# Run all Experiment 4 ablation studies
################################################################################
# This script runs all three ablation experiments sequentially or in parallel.
#
# Usage:
#   bash run_all_ablations.sh [sequential|parallel]
#
# Default: sequential
################################################################################

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Change to project root
cd "$PROJECT_ROOT"

# Mode: sequential or parallel
MODE="${1:-sequential}"

echo "================================================================================"
echo "Experiment 4: Running Ablation Studies"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Mode: $MODE"
echo ""

# Configuration files (without 'configs/' prefix - the script adds it)
CONFIG_REMOVE_8H="experiment_EMT_Part1_setting4_ablation_remove_8h.yaml"
CONFIG_REMOVE_1D="experiment_EMT_Part1_setting4_ablation_remove_1d.yaml"
CONFIG_REMOVE_3D="experiment_EMT_Part1_setting4_ablation_remove_3d.yaml"

# Check if configs exist (in configs/ directory)
for config in "$CONFIG_REMOVE_8H" "$CONFIG_REMOVE_1D" "$CONFIG_REMOVE_3D"; do
    if [ ! -f "configs/$config" ]; then
        echo "ERROR: Configuration file not found: configs/$config"
        exit 1
    fi
done

echo "✓ All configuration files found"
echo ""

if [ "$MODE" = "parallel" ]; then
    echo "Running experiments in PARALLEL mode..."
    echo "WARNING: This requires 3 GPUs or sufficient GPU memory for concurrent training."
    echo ""
    
    # Run in background
    bash step1_run_experiment_EMT.sh "$CONFIG_REMOVE_8H" &
    PID_8H=$!
    echo "Started Remove 8h (PID: $PID_8H)"
    
    bash step1_run_experiment_EMT.sh "$CONFIG_REMOVE_1D" &
    PID_1D=$!
    echo "Started Remove 1d (PID: $PID_1D)"
    
    bash step1_run_experiment_EMT.sh "$CONFIG_REMOVE_3D" &
    PID_3D=$!
    echo "Started Remove 3d (PID: $PID_3D)"
    
    echo ""
    echo "Waiting for all experiments to complete..."
    
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
    echo "----------------------------------------"
    echo "1/3: Running Remove 8h ablation..."
    echo "----------------------------------------"
    bash step1_run_experiment_EMT.sh "$CONFIG_REMOVE_8H"
    echo "✅ Remove 8h completed"
    echo ""
    
    echo "----------------------------------------"
    echo "2/3: Running Remove 1d ablation..."
    echo "----------------------------------------"
    bash step1_run_experiment_EMT.sh "$CONFIG_REMOVE_1D"
    echo "✅ Remove 1d completed"
    echo ""
    
    echo "----------------------------------------"
    echo "3/3: Running Remove 3d ablation..."
    echo "----------------------------------------"
    bash step1_run_experiment_EMT.sh "$CONFIG_REMOVE_3D"
    echo "✅ Remove 3d completed"
    echo ""
    
    echo "✅ All experiments completed successfully!"
fi

echo ""
echo "================================================================================"
echo "Next step: Run the analysis script"
echo "================================================================================"
echo ""
echo "Command:"
echo "  python Experiments/exp4_ablation/analyze_marginal_contribution.py \\"
echo "    --output_base /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData"
echo ""
echo "Note: Make sure Setting2 (full model) has been trained before running analysis."
echo "================================================================================"
