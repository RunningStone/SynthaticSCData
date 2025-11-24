#!/bin/bash
# 批量重新评估所有ablation实验

echo "========================================================================"
echo "Batch Re-evaluation of Ablation Experiments"
echo "========================================================================"
echo ""

# Define experiments
EXPERIMENTS=(
    "experiment_EMT_Part1_setting4_ablation_remove_1d"
    "experiment_EMT_Part1_setting4_ablation_remove_3d"
    "experiment_EMT_Part1_setting4_ablation_remove_8h"
)

BASE_OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData"
SUCCESS_COUNT=0
FAIL_COUNT=0

for EXP in "${EXPERIMENTS[@]}"; do
    echo "========================================================================"
    echo "Processing: $EXP"
    echo "========================================================================"
    
    CONFIG_FILE="${EXP}.yaml"
    CHECKPOINT_DIR="${BASE_OUTPUT_DIR}/${EXP}/checkpoints"
    
    # Check if checkpoint directory exists
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
        echo "   Skipping..."
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        continue
    fi
    
    # Run evaluation
    echo "Running evaluation for $EXP..."
    bash step1_run_evaluation_only.sh \
        "$CONFIG_FILE" \
        "$CHECKPOINT_DIR" \
        configs \
        _fixed
    
    if [ $? -eq 0 ]; then
        echo "✅ $EXP completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ $EXP failed"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    echo ""
done

echo "========================================================================"
echo "Batch Evaluation Summary"
echo "========================================================================"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "========================================================================"

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✅ All evaluations completed successfully!"
    exit 0
else
    echo "⚠️  Some evaluations failed. Check logs for details."
    exit 1
fi
