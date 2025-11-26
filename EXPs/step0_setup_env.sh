#!/bin/bash
# Setup script for Schrödinger Bridge project using uv

echo "=================================="
echo "Setting up Schrödinger Bridge Project"
echo "=================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of EXPs)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Script location: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed"
    echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv is installed"

# Change to project root directory
cd "$PROJECT_ROOT"

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
    echo "Virtual environment created at: $PROJECT_ROOT/.venv"
    echo ""
    echo "To activate the environment, run:"
    echo "  source $PROJECT_ROOT/.venv/bin/activate"
    echo ""
    echo "Or use uv to run commands directly from project root:"
    echo "  cd $PROJECT_ROOT"
    echo "  uv run pytest Tests/ -v"
    echo ""
    echo "To run experiments from EXPs directory:"
    echo "  cd $SCRIPT_DIR"
    echo "  bash step1_run_precalc.sh"
    echo "  bash step1_run_experiment_EMT.sh"
    echo ""
    echo "To run workers directly:"
    echo "  cd $PROJECT_ROOT"
    echo "  uv run python Workers/step1_precalc_exps.py --help"
    echo ""
else
    echo "✗ Setup failed"
    exit 1
fi
