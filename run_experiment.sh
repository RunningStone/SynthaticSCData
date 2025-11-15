#!/bin/bash
# Run experiment from YAML configuration files
# New modular configuration system

# Parse arguments
CONFIG_FILE="${1:-experiment_EMT_setting1.yaml}"
OUTPUT_DIR="${2:-}"
CONFIG_DIR="configs"

echo "========================================================================"
echo "Running Experiment from Configuration"
echo "========================================================================"
echo ""
echo "Configuration file: $CONFIG_FILE"
echo "Configuration directory: $CONFIG_DIR"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output directory (override): $OUTPUT_DIR"
fi
echo ""
echo "========================================================================"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_DIR/$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_DIR/$CONFIG_FILE"
    echo ""
    echo "Available configuration files:"
    ls -1 $CONFIG_DIR/experiment_*.yaml 2>/dev/null || echo "  No experiment configs found"
    echo ""
    echo "Usage: $0 [config_file] [output_dir]"
    echo "Examples:"
    echo "  $0 experiment_EMT_setting1.yaml"
    echo "  $0 experiment_EMT_setting1.yaml /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs"
    exit 1
fi

# Activate environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: No virtual environment found (.venv or venv)"
fi

# Run experiment
echo "Starting experiment..."
echo ""
if [ -n "$OUTPUT_DIR" ]; then
    python step1_run_experiment.py "$CONFIG_FILE" --config_dir "$CONFIG_DIR" --output_dir "$OUTPUT_DIR"
else
    python step1_run_experiment.py "$CONFIG_FILE" --config_dir "$CONFIG_DIR"
fi

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Experiment complete!"
    echo "Check results in the output directory specified in the config file"
else
    echo "Experiment failed with exit code: $EXIT_CODE"
fi
echo "========================================================================"

exit $EXIT_CODE
