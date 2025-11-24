#!/bin/bash
# ============================================================================
# Run Experiment 7: Entropy Evolution Analysis
# ============================================================================
# This script trains models on full trajectory (Setting 7) and analyzes
# whether they can reproduce non-monotonic entropy evolution

set -e  # Exit on error

echo "========================================================================"
echo "EXPERIMENT 7: ENTROPY EVOLUTION ANALYSIS"
echo "========================================================================"
echo ""
echo "This experiment tests whether models can reproduce the non-monotonic"
echo "entropy evolution (entropy increase → entropy decrease) observed in"
echo "real EMT trajectories."
echo ""
echo "Hypothesis: Boundary conditions are insufficient to constrain"
echo "            non-monotonic entropy dynamics."
echo ""
echo "========================================================================"
echo ""

# Configuration
CONFIG_FILE="configs/experiment_EMT_Part1_setting7_entropy.yaml"
DATA_CONFIG="configs/data_EMT_Cook_with_label.yaml"

# Check if config files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$DATA_CONFIG" ]; then
    echo "Error: Data config file not found: $DATA_CONFIG"
    exit 1
fi

echo "Configuration:"
echo "  Experiment config: $CONFIG_FILE"
echo "  Data config:       $DATA_CONFIG"
echo ""

# ============================================================================
# STEP 1: TRAIN MODELS
# ============================================================================

echo "============================================================================"
echo "STEP 1: TRAINING MODELS ON FULL TRAJECTORY (Setting 7)"
echo "============================================================================"
echo ""
echo "Training models: sb_mlplus, batch_ot, vae, ot, sb"
echo "Data: Full forward EMT trajectory (0d → 8h → 1d → 3d → 7d)"
echo ""

python step1_run_experiment.py \
    --config "$CONFIG_FILE" \
    --verbose

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Model training failed!"
    exit 1
fi

echo ""
echo "✓ Model training completed successfully"
echo ""

# ============================================================================
# STEP 2: ENTROPY EVOLUTION ANALYSIS
# ============================================================================

echo "============================================================================"
echo "STEP 2: ENTROPY EVOLUTION ANALYSIS"
echo "============================================================================"
echo ""
echo "Analyzing entropy evolution in generated trajectories..."
echo ""

# Get output directory from config
OUTPUT_DIR=$(python -c "import yaml; config = yaml.safe_load(open('$CONFIG_FILE')); print(config['settings']['output_dir'])")

# Path to test data
DATA_PATH="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/DATAs/EMT/2024_12_04_Cook_emt_dataset_with_removal.h5ad"

# Check if trained models exist
SETTING7_CKPT_DIR="${OUTPUT_DIR}/checkpoints"

if [ ! -d "$SETTING7_CKPT_DIR" ]; then
    echo "Warning: Checkpoint directory not found: $SETTING7_CKPT_DIR"
    echo "Skipping entropy analysis. Please train models first."
else
    # Find model checkpoints
    SB_MLPLUS_CKPT="${SETTING7_CKPT_DIR}/sb_mlplus/best_model.pt"
    
    if [ ! -f "$SB_MLPLUS_CKPT" ]; then
        echo "Warning: sb_mlplus checkpoint not found: $SB_MLPLUS_CKPT"
        echo "Trying alternative checkpoint name..."
        SB_MLPLUS_CKPT="${SETTING7_CKPT_DIR}/sb_mlplus/final_model.pt"
    fi
    
    if [ -f "$SB_MLPLUS_CKPT" ]; then
        echo "Found checkpoint: $SB_MLPLUS_CKPT"
        echo ""
        
        # Create entropy analysis output directory
        ENTROPY_OUTPUT="${OUTPUT_DIR}/entropy_analysis"
        mkdir -p "$ENTROPY_OUTPUT"
        
        echo "Running entropy evolution analysis..."
        echo "  Output directory: $ENTROPY_OUTPUT"
        echo ""
        
        # Check if Setting1 and Setting2 checkpoints exist for comparison
        SETTING1_CKPT="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting1/checkpoints/sb_mlplus/best_model.pt"
        SETTING2_CKPT="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting2/checkpoints/sb_mlplus/best_model.pt"
        
        # Build command
        CMD="python Experiments/exp7_entropy/run_entropy_analysis.py \
            --data_path \"$DATA_PATH\" \
            --time_column Ground_truth \
            --time_labels 0d 8h 1d 3d 7d \
            --setting2_checkpoint \"$SB_MLPLUS_CKPT\" \
            --method knn \
            --k 5 \
            --n_samples 1000 \
            --output_dir \"$ENTROPY_OUTPUT\" \
            --device cuda \
            --cross_validate_methods"
        
        # Add Setting1 checkpoint if exists
        if [ -f "$SETTING1_CKPT" ]; then
            echo "  Found Setting1 checkpoint for comparison"
            CMD="$CMD --setting1_checkpoint \"$SETTING1_CKPT\""
        else
            echo "  Warning: Setting1 checkpoint not found, skipping comparison"
            # Use Setting7 as both Setting1 and Setting2 for now
            CMD="$CMD --setting1_checkpoint \"$SB_MLPLUS_CKPT\""
        fi
        
        # Add Setting3 checkpoint if exists (optional)
        SETTING3_CKPT="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting3/checkpoints/sb_mlplus/best_model.pt"
        if [ -f "$SETTING3_CKPT" ]; then
            echo "  Found Setting3 checkpoint for comparison"
            CMD="$CMD --setting3_checkpoint \"$SETTING3_CKPT\""
        fi
        
        echo ""
        
        # Run entropy analysis
        eval $CMD
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Entropy evolution analysis completed successfully"
            echo ""
            echo "Results saved to: $ENTROPY_OUTPUT"
            echo "  - entropy_curves_comparison.png/pdf"
            echo "  - peak_characteristics_comparison.png/pdf"
            echo "  - entropy_analysis_summary.json"
            echo "  - entropy_analysis_full_results.pkl"
            echo ""
        else
            echo ""
            echo "✗ Entropy analysis failed!"
            echo "Check logs for details."
        fi
    else
        echo "✗ No trained model checkpoint found!"
        echo "Please train models first using step1_run_experiment.py"
    fi
fi

# ============================================================================
# STEP 3: GENERATE SUMMARY REPORT
# ============================================================================

echo "============================================================================"
echo "STEP 3: GENERATING SUMMARY REPORT"
echo "============================================================================"
echo ""

# Check if entropy analysis results exist
SUMMARY_JSON="${OUTPUT_DIR}/entropy_analysis/entropy_analysis_summary.json"

if [ -f "$SUMMARY_JSON" ]; then
    echo "Entropy analysis results found. Generating summary..."
    echo ""
    
    # Print key findings
    python -c "
import json
import sys

try:
    with open('$SUMMARY_JSON', 'r') as f:
        results = json.load(f)
    
    print('='*70)
    print('KEY FINDINGS')
    print('='*70)
    print()
    
    # Real data
    real_peak = results['real_peak_analysis']
    print('Real Data:')
    print(f'  Peak time: {real_peak[\"peak_time\"]}')
    print(f'  Peak value: {real_peak[\"peak_value\"]:.4f}')
    print(f'  Non-monotonic: {real_peak[\"is_nonmonotonic\"]}')
    print(f'  Amplitude: {real_peak[\"amplitude\"]:.4f}')
    print()
    
    # Model results
    for setting_name, setting_results in results['settings'].items():
        peak = setting_results['peak_analysis']
        sim = setting_results.get('similarity_to_real', 'N/A')
        
        print(f'{setting_name}:')
        print(f'  Peak time: {peak[\"peak_time\"]}')
        print(f'  Non-monotonic: {peak[\"is_nonmonotonic\"]}')
        print(f'  Amplitude: {peak[\"amplitude\"]:.4f}')
        if sim != 'N/A':
            print(f'  MSE to real: {sim:.4f}')
        print()
    
    print('='*70)
    
except Exception as e:
    print(f'Error reading summary: {e}', file=sys.stderr)
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Summary report generated"
    fi
else
    echo "No entropy analysis results found. Skipping summary."
fi

echo ""

# ============================================================================
# EXPERIMENT COMPLETE
# ============================================================================

echo "============================================================================"
echo "EXPERIMENT 7 COMPLETE"
echo "============================================================================"
echo ""
echo "Results saved to:"
echo "  Output directory:  $OUTPUT_DIR"
echo "  Model checkpoints: ${OUTPUT_DIR}/checkpoints/"
echo "  Entropy analysis:  ${OUTPUT_DIR}/entropy_analysis/"
echo "  Logs:              ${OUTPUT_DIR}/logs/"
echo ""
echo "Next steps:"
echo "  1. Check entropy curves: ${OUTPUT_DIR}/entropy_analysis/entropy_curves_comparison.png"
echo "  2. Review peak analysis: ${OUTPUT_DIR}/entropy_analysis/peak_characteristics_comparison.png"
echo "  3. Read summary JSON:    ${OUTPUT_DIR}/entropy_analysis/entropy_analysis_summary.json"
echo ""
echo "Scientific interpretation:"
echo "  - If Setting1 shows monotonic entropy: ✓ Hypothesis confirmed"
echo "  - If Setting2 shows non-monotonic entropy: ✓ Full trajectory captures dynamics"
echo "  - Compare peak positions and amplitudes across settings"
echo ""
echo "============================================================================"
