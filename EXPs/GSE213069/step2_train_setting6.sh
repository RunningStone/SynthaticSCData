#!/bin/bash
# ============================================================================
# Run Step 2: Train GSE213069 Setting 6 (Interpolated Data)
# ============================================================================
# This script trains models for Setting 6 which uses:
# - Boundary time points (T0, day5): Real data
# - Intermediate time points (day2, day3, day4): Linearly interpolated data
#
# The experiment tests whether geometric interpolation between boundary states
# can replace real intermediate observations for trajectory learning.
#
# This script uses hardcoded parameters and requires no command-line arguments.
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Path Configuration
# ============================================================================

# Get the directory where this script is located (EXPs/GSE213069/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the EXPs directory (parent of GSE213069)
EXP_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$EXP_DIR/.." && pwd )"

# ============================================================================
# Hardcoded Parameters - Specific for GSE213069 Setting 6
# ============================================================================

# Configuration file
CONFIG_FILE="experiment_GSE213069_setting6_interpolated.yaml"
CONFIG_DIR="$PROJECT_ROOT/configs/GSE123069"

# Output directory
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/GSE213069_V1/Setting6_Interpolated"

# ============================================================================
# Print Configuration
# ============================================================================

echo "========================================"
echo "Step 2: Train GSE213069 Setting 6"
echo "        (Interpolated Data)"
echo "========================================"
echo ""
echo "Experiment Description:"
echo "  - Boundary points (T0, day5): Real data"
echo "  - Intermediate points (day2, day3, day4): Linearly interpolated"
echo "  - Tests geometric interpolation vs real observations"
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

# ============================================================================
# Validate Configuration
# ============================================================================

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

# ============================================================================
# Setup Environment
# ============================================================================

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

# ============================================================================
# Run Experiment
# ============================================================================

echo "========================================"
echo "Starting Training..."
echo "========================================"
echo ""
echo "Pipeline:"
echo "  1. Generate linearly interpolated intermediate states"
echo "  2. Train models on boundary (real) + intermediate (interpolated)"
echo "  3. Evaluate and save results"
echo ""

# Run the Setting 6 specific experiment script
python Workers/step2_run_exp_setting6.py \
    "$CONFIG_FILE" \
    --config_dir "$CONFIG_DIR" \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training Complete (GSE213069 Setting 6)"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Output structure:"
    echo "  $OUTPUT_DIR/"
    echo "  ├── checkpoints/     # Model checkpoints"
    echo "  ├── logs/            # Training logs"
    echo "  ├── metrics/         # Evaluation metrics"
    echo "  ├── generated_data/  # Generated samples"
    echo "  └── interpolation_analysis/  # Quality analysis"
    echo ""
    echo "Next steps:"
    echo "  1. Check training logs in: $OUTPUT_DIR/logs/"
    echo "  2. Review model checkpoints in: $OUTPUT_DIR/checkpoints/"
    echo "  3. Compare with Setting 1 and Setting 2 results"
    echo "  4. Run visualization: bash step4_generate_all_figures.sh"
    echo ""
else
    echo "❌ Training Failed (GSE213069 Setting 6)"
    echo "========================================"
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo "Check logs for details: $OUTPUT_DIR/logs/"
    echo ""
fi

exit $EXIT_CODE
