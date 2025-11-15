#!/bin/bash
# Run visualization from experiment YAML configuration

# Parse arguments
EXPERIMENT_CONFIG="${1:-experiment_EMT_setting1.yaml}"
CONFIG_DIR="configs"

echo "========================================================================"
echo "Running Visualization from Experiment Configuration"
echo "========================================================================"
echo ""
echo "Experiment config: $EXPERIMENT_CONFIG"
echo "Config directory: $CONFIG_DIR"
echo ""
echo "========================================================================"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_DIR/$EXPERIMENT_CONFIG" ]; then
    echo "Error: Experiment configuration file not found: $CONFIG_DIR/$EXPERIMENT_CONFIG"
    echo ""
    echo "Available experiment configuration files:"
    ls -1 $CONFIG_DIR/experiment_*.yaml 2>/dev/null || echo "  No experiment configs found"
    echo ""
    echo "Usage: $0 [experiment_config]"
    echo "Examples:"
    echo "  $0 experiment_EMT_setting1.yaml"
    echo "  $0 experiment_EMT_setting2.yaml"
    echo "  $0 experiment_GSE234181_setting1.yaml"
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

# Run visualization
echo "Starting visualization..."
echo ""
python step2_run_visualization.py "$EXPERIMENT_CONFIG" --config_dir "$CONFIG_DIR"

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Visualization complete!"
    echo "Check results in the visualization output directory"
else
    echo "Visualization failed with exit code: $EXIT_CODE"
fi
echo "========================================================================"

exit $EXIT_CODE
