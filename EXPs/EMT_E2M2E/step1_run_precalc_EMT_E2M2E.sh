#!/bin/bash
# Run Step 1: Pre-calculation Experiments for EMT_E2M2E
# This script runs both data split analysis and model parameter analysis
# Specific for EMT_E2M2E experiment configurations (Full trajectory with reversal)
# 
# E2M2E: Epithelial → Mesenchymal → Epithelial
# Forward EMT: 0d → 8h → 1d → 3d → 7d
# Reversal: 7d → 8h_rm → 1d_rm → 3d_rm
#
# This script uses hardcoded parameters and requires no command-line arguments

set -e  # Exit on error

# Get the directory where this script is located (EXPs/EMT_E2M2E/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the EXPs directory (parent of EMT_E2M2E)
EXP_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$EXP_DIR/.." && pwd )"

# Hardcoded parameters - specific for EMT_E2M2E experiment
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M2E/precalc_results"
DATA_CONFIG="$PROJECT_ROOT/configs/EMT_E2M2E/data_EMT_E2M2E.yaml"
BATCH_SIZE=256
INPUT_DIM=1000  # Using 1000 HVGs as per config
MIN_CELLS=1000
BOTTLENECK_PCT=100.0
OPTIMIZER="adam"

# Print configuration
echo "========================================"
echo "Step 1: Pre-calculation Experiments"
echo "EMT_E2M2E Experiment (Full Trajectory with Reversal)"
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
echo "✓ Pre-calculation Complete (EMT_E2M2E)"
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
echo "2. Update your experiment configurations in configs/EMT_E2M2E/"
echo ""
echo "3. Run EMT_E2M2E training experiments from EXPs/EMT_E2M2E/"
echo ""
