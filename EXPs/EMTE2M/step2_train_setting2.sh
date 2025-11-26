#!/bin/bash
# Run Step 2: Train EMT_E2M Setting 2
# This script trains models for Setting 2 (all time points: 0d, 8h, 1d, 3d, 7d)
# Specific for EMT_E2M experiment configurations
# 
# This script uses hardcoded parameters and requires no command-line arguments

set -e  # Exit on error

# Get the directory where this script is located (EXPs/EMTE2M/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the EXPs directory (parent of EMTE2M)
EXP_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$EXP_DIR/.." && pwd )"

# Hardcoded parameters - specific for EMT_E2M Setting 2
CONFIG_FILE="experiment_EMT_Part1_setting2.yaml"
CONFIG_DIR="$PROJECT_ROOT/configs/EMT_E2M"
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting2"

# Print configuration
echo "========================================"
echo "Step 2: Train EMT_E2M Setting 2"
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
echo "Setting 2 Details:"
echo "  Time points: 0d, 8h, 1d, 3d, 7d (all 5 timepoints)"
echo "  Models: MLPlus SB, Batch OT, Conditional VAE"
echo "  Total cells: 8,974 (across all timepoints)"
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
    echo "✓ Training Complete (EMT_E2M Setting 2)"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Check training logs in: $OUTPUT_DIR/logs/"
    echo "2. Review model checkpoints in: $OUTPUT_DIR/checkpoints/"
    echo "3. Run evaluation: bash step3_evaluate_setting2.sh"
    echo ""
else
    echo "❌ Training Failed (EMT_E2M Setting 2)"
    echo "========================================"
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo "Check logs for details: $OUTPUT_DIR/logs/"
    echo ""
fi

exit $EXIT_CODE
