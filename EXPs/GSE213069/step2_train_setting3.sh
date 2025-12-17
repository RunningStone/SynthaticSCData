#!/bin/bash
# Run Step 2: Train GSE213069 Setting 3
# This script trains models for Setting 3 (key time points: T0, day2, day5)
# Specific for GSE213069 experiment configurations
# 
# This script uses hardcoded parameters and requires no command-line arguments

set -e  # Exit on error

# Get the directory where this script is located (EXPs/GSE213069/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the EXPs directory (parent of GSE213069)
EXP_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$EXP_DIR/.." && pwd )"

# Hardcoded parameters - specific for GSE213069 Setting 3
CONFIG_FILE="experiment_GSE213069_setting3.yaml"
CONFIG_DIR="$PROJECT_ROOT/configs/GSE123069"
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/GSE213069_V1/Setting3_KeyTimepoints"

# Print configuration
echo "========================================"
echo "Step 2: Train GSE213069 Setting 3"
echo "========================================"
echo ""
echo "Script location: $SCRIPT_DIR"
echo "EXPs directory: $EXP_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Config directory: $CONFIG_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo ""
echo "Setting 3 Details:"
echo "  Time points: T0, day2, day5 (key timepoints)"
echo "  Models: SB_MLPlus, Batch OT, Conditional VAE"
echo "  Total cells: 1,500 (across all timepoints)"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_DIR/$CONFIG_FILE" ]; then
    echo "❌ Error: Configuration file not found: $CONFIG_DIR/$CONFIG_FILE"
    echo ""
    echo "Available configuration files in $CONFIG_DIR:"
    ls -1 "$CONFIG_DIR"/experiment_*.yaml 2>/dev/null || echo "  No experiment configs found"
    echo ""
    exit 1
fi

echo "✓ Configuration file found: $CONFIG_DIR/$CONFIG_FILE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "✓ Output directory created: $OUTPUT_DIR"
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✓ Virtual environment found at $PROJECT_ROOT/.venv"
    echo "  (Make sure to activate it before running: source .venv/bin/activate)"
elif [ -d "venv" ]; then
    echo "✓ Virtual environment found at $PROJECT_ROOT/venv"
    echo "  (Make sure to activate it before running: source venv/bin/activate)"
else
    echo "⚠️  Warning: No virtual environment found (.venv or venv)"
    echo "  Please ensure dependencies are installed"
fi
echo ""

# Run experiment
echo "========================================"
echo "Starting Training..."
echo "========================================"
echo ""

python Workers/step2_run_experiment.py \
    "$CONFIG_FILE" \
    --config_dir "$CONFIG_DIR" \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training Complete (GSE213069 Setting 3)"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Check training logs in: $OUTPUT_DIR/logs/"
    echo "2. Review model checkpoints in: $OUTPUT_DIR/checkpoints/"
    echo "3. Run inference: bash step3_run_inference.sh"
    echo ""
else
    echo "❌ Training Failed (GSE213069 Setting 3)"
    echo "========================================"
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo "Check logs for details: $OUTPUT_DIR/logs/"
    echo ""
fi

exit $EXIT_CODE
