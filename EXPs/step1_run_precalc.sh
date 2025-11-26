#!/bin/bash
# Run Step 1: Pre-calculation Experiments
# This script runs both data split analysis and model parameter analysis

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Default parameters
OUTPUT_DIR="$PROJECT_ROOT/precalc_results"
DATA_CONFIG="$PROJECT_ROOT/configs/data_EMT_Cook_with_label.yaml"
BATCH_SIZE=256
INPUT_DIM=100
MIN_CELLS=1000
BOTTLENECK_PCT=100.0
OPTIMIZER="adam"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data_config)
            DATA_CONFIG="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --input_dim)
            INPUT_DIM="$2"
            shift 2
            ;;
        --min_cells)
            MIN_CELLS="$2"
            shift 2
            ;;
        --bottleneck_pct)
            BOTTLENECK_PCT="$2"
            shift 2
            ;;
        --skip_data)
            SKIP_DATA="--skip_data"
            shift
            ;;
        --skip_model)
            SKIP_MODEL="--skip_model"
            shift
            ;;
        --mixed_precision)
            MIXED_PRECISION="--mixed_precision"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output_dir DIR        Output directory (default: <PROJECT_ROOT>/precalc_results)"
            echo "  --data_config PATH      Data config YAML path (default: <PROJECT_ROOT>/configs/data_EMT_Cook_with_label.yaml)"
            echo "  --batch_size N          Batch size for memory estimation (default: 256)"
            echo "  --input_dim N           Input dimension / HVGs (default: 100)"
            echo "  --min_cells N           Minimum cells per category (default: 1000)"
            echo "  --bottleneck_pct PCT    Bottleneck percentage (default: 100.0)"
            echo "  --skip_data             Skip data split analysis"
            echo "  --skip_model            Skip model parameter analysis"
            echo "  --mixed_precision       Use mixed precision training"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "========================================"
echo "Step 1: Pre-calculation Experiments"
echo "========================================"
echo ""
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
    --optimizer "$OPTIMIZER" \
    $SKIP_DATA \
    $SKIP_MODEL \
    $MIXED_PRECISION

echo ""
echo "========================================"
echo "✓ Pre-calculation Complete"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review the analysis results in $OUTPUT_DIR"
echo "2. Update your experiment configurations"
echo "3. Run step1_run_experiment.py to start training"
echo ""
