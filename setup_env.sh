#!/bin/bash
# Setup script for Schrödinger Bridge project using uv

echo "=================================="
echo "Setting up Schrödinger Bridge Project"
echo "=================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv is installed"

# Create virtual environment and install dependencies
echo ""
echo "Creating virtual environment and installing dependencies..."
uv sync

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✓ Setup complete!"
    echo "=================================="
    echo ""
    echo "To activate the environment, run:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "Or use uv to run commands directly:"
    echo "  uv run python quick_test.py"
    echo "  uv run pytest Tests/ -v"
    echo "  uv run python run_experiment.py --config configs/default_config.yaml --output outputs/phase1"
    echo ""
else
    echo "✗ Setup failed"
    exit 1
fi
