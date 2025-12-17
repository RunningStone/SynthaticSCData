#!/bin/bash
# Run Step 1: Pre-calculation Experiments for Paul EMT
# This script runs both data split analysis and model parameter analysis
# Specific for Paul EMT experiment configurations
# 
# This script uses hardcoded parameters and requires no command-line arguments

set -e  # Exit on error

# Get the directory where this script is located (EXPs/Paul/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the EXPs directory (parent of Paul)
EXP_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$EXP_DIR/.." && pwd )"

# Hardcoded parameters - specific for Paul EMT experiment
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/Paul/precalc_results"
DATA_CONFIG="$PROJECT_ROOT/configs/Paul/data_Paul_EMT.yaml"
BATCH_SIZE=64  # Smaller batch size for smaller dataset
INPUT_DIM=500  # Using 500 HVGs as per config
MIN_CELLS=40   # Smaller min cells for smaller dataset
BOTTLENECK_PCT=100.0
OPTIMIZER="adam"

# Print configuration
echo "========================================"
echo "Step 1: Pre-calculation Experiments"
echo "Paul EMT Experiment"
echo "========================================"
echo ""
echo "Script location: $SCRIPT_DIR"
echo "EXPs directory: $EXP_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""
echo "Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Data config: $DATA_CONFIG"
echo "  Batch size: $BATCH_SIZE"
echo "  Input dimension: $INPUT_DIM"
echo "  Min cells per category: $MIN_CELLS"
echo "  Bottleneck percentage: $BOTTLENECK_PCT"
echo "  Optimizer: $OPTIMIZER"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to project root directory
cd "$PROJECT_ROOT"

# Run the analysis
python Workers/step1_precalc_exps.py \
    --output_dir "$OUTPUT_DIR" \
    --data_config "$DATA_CONFIG" \
    --batch_size "$BATCH_SIZE" \
    --input_dim "$INPUT_DIM" \
    --min_cells "$MIN_CELLS" \
    --bottleneck_pct "$BOTTLENECK_PCT" \
    --optimizer "$OPTIMIZER"

echo ""
echo "========================================"
echo "✓ Pre-calculation Complete (Paul EMT)"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key output files:"
echo "  - data_split_analysis_final_params.json"
echo "  - data_split_analysis_summary.txt"
echo "  - model_param_analysis_comparison.txt"
echo ""
echo "Next steps:"
echo "1. Review the analysis results:"
echo "   cat $OUTPUT_DIR/data_split_analysis_summary.txt"
echo "   cat $OUTPUT_DIR/model_param_analysis_comparison.txt"
echo ""
echo "2. Update your experiment configurations in configs/Paul/"
echo ""
echo "3. Run Paul EMT training experiments from EXPs/Paul/"
echo ""
