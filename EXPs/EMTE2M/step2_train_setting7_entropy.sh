#!/bin/bash
# Run Step 2: Entropy Evolution Analysis (Setting 7)
# This script analyzes entropy evolution across different experimental settings
# to test the core hypothesis: boundary conditions are insufficient to constrain
# non-monotonic entropy dynamics.
#
# This script uses hardcoded parameters and requires no command-line arguments

set -e  # Exit on error

# Get the directory where this script is located (EXPs/EMTE2M/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the EXPs directory (parent of EMTE2M)
EXP_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$EXP_DIR/.." && pwd )"

# Hardcoded parameters - specific for EMT_E2M Setting 7 (Entropy Analysis)
CONFIG_FILE="experiment_EMT_Part1_setting7_entropy.yaml"
CONFIG_DIR="$PROJECT_ROOT/configs/EMT_E2M"
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting7_Entropy"

# Model checkpoints from other settings (for comparison)
SETTING1_CHECKPOINT="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting1/checkpoints/sb_mlplus_best.pt"
SETTING2_CHECKPOINT="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting2/checkpoints/sb_mlplus_best.pt"
SETTING3_CHECKPOINT="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M/Setting3/checkpoints/sb_mlplus_best.pt"

# Analysis parameters
METHOD="knn"  # Entropy estimation method: knn, gaussian, or both
K=5           # Number of nearest neighbors for KNN method
N_SAMPLES=1000  # Number of cells to sample for analysis

# Print configuration
echo "========================================"
echo "Step 2: Entropy Evolution Analysis (Setting 7)"
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
echo "Model Checkpoints:"
echo "  Setting1: $SETTING1_CHECKPOINT"
echo "  Setting2: $SETTING2_CHECKPOINT"
echo "  Setting3: $SETTING3_CHECKPOINT"
echo ""
echo "Analysis Parameters:"
echo "  Method: $METHOD"
echo "  K (neighbors): $K"
echo "  N samples: $N_SAMPLES"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_DIR/$CONFIG_FILE" ]; then
    echo "❌ Error: Configuration file not found: $CONFIG_DIR/$CONFIG_FILE"
    echo ""
    echo "Available configuration files in $CONFIG_DIR:"
    ls -1 "$CONFIG_DIR"/experiment_*.yaml 2>/dev/null || echo "  No experiment configs found"
    echo ""
    echo "Note: You may need to create the entropy experiment config file."
    echo "You can copy from an existing setting and modify the time_points."
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

# Build checkpoint arguments
CHECKPOINT_ARGS=""
if [ -f "$SETTING1_CHECKPOINT" ]; then
    CHECKPOINT_ARGS="$CHECKPOINT_ARGS --setting1_checkpoint $SETTING1_CHECKPOINT"
    echo "✓ Setting1 checkpoint found"
else
    echo "⚠️  Setting1 checkpoint not found (will skip)"
fi

if [ -f "$SETTING2_CHECKPOINT" ]; then
    CHECKPOINT_ARGS="$CHECKPOINT_ARGS --setting2_checkpoint $SETTING2_CHECKPOINT"
    echo "✓ Setting2 checkpoint found"
else
    echo "⚠️  Setting2 checkpoint not found (will skip)"
fi

if [ -f "$SETTING3_CHECKPOINT" ]; then
    CHECKPOINT_ARGS="$CHECKPOINT_ARGS --setting3_checkpoint $SETTING3_CHECKPOINT"
    echo "✓ Setting3 checkpoint found"
else
    echo "⚠️  Setting3 checkpoint not found (will skip)"
fi
echo ""

# Run experiment
echo "========================================"
echo "Starting Entropy Analysis..."
echo "========================================"
echo ""

python Workers/step2_run_exp_setting7.py \
    "$CONFIG_FILE" \
    --config_dir "$CONFIG_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --method "$METHOD" \
    --k "$K" \
    --n_samples "$N_SAMPLES" \
    $CHECKPOINT_ARGS

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Entropy Analysis Complete (Setting 7)"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Output files:"
    echo "  - entropy_curves_comparison.png/pdf: Entropy evolution comparison"
    echo "  - peak_characteristics_comparison.png/pdf: Peak analysis"
    echo "  - entropy_analysis_summary.json: Numerical results"
    echo "  - entropy_analysis_full_results.pkl: Full results with trajectories"
    echo ""
    echo "Key findings to look for:"
    echo "  1. Does real data show non-monotonic (inverted-U) entropy evolution?"
    echo "  2. Can Setting1 (boundary only) reproduce this pattern?"
    echo "  3. Can Setting2 (full trajectory) better capture the dynamics?"
    echo ""
else
    echo "❌ Entropy Analysis Failed (Setting 7)"
    echo "========================================"
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo "Check the error messages above for details."
    echo ""
fi

exit $EXIT_CODE
