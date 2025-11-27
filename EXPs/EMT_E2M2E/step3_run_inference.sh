#!/bin/bash
# ============================================================================
# Step 3: Run Inference on Test Set for All Settings (EMT_E2M2E)
# ============================================================================
#
# This script automatically scans the experiment output directory for Setting
# folders and runs inference on each one.
#
# Usage:
#   bash step3_run_inference.sh [base_output_dir]
#
# Arguments:
#   base_output_dir : Base directory containing experiment outputs
#                     Default: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M2E
#
# The script will:
#   1. Scan for Setting* folders in the base directory
#   2. For each folder with experiment_config.yaml and checkpoints/, run inference
#   3. Handle nested folders (e.g., Setting4/Setting4_Ablation_Remove8h)
#
# Output for each experiment:
#   - {experiment_dir}/evaluation_results.json : Evaluation metrics
#   - {experiment_dir}/generated_data/{model}.pkl : Generated data
#
# ============================================================================

# Don't exit on error (we handle errors manually)
set +e

# Get script directory (for relative paths to python scripts)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default base output directory for EMT_E2M2E
DEFAULT_BASE_OUTPUT="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/EMT_E2M2E"

# Parse arguments
BASE_OUTPUT="${1:-$DEFAULT_BASE_OUTPUT}"

echo "========================================================================"
echo "Step 3: Run Inference on Test Set (EMT_E2M2E)"
echo "========================================================================"
echo ""
echo "Base output directory: ${BASE_OUTPUT}"
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Check if base directory exists
if [ ! -d "${BASE_OUTPUT}" ]; then
    echo "Error: Base output directory not found: ${BASE_OUTPUT}"
    exit 1
fi

# Activate virtual environment if exists
cd "${PROJECT_ROOT}"
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
fi

# Function to check if a directory is a valid experiment directory
is_valid_experiment_dir() {
    local dir=$1
    # Must have experiment_config.yaml and checkpoints/
    [ -f "${dir}/experiment_config.yaml" ] && [ -d "${dir}/checkpoints" ]
}

# Function to run inference for a single experiment directory
run_inference() {
    local experiment_dir=$1
    local experiment_name=$(basename "${experiment_dir}")
    
    echo ""
    echo "========================================================================"
    echo "Running Inference: ${experiment_name}"
    echo "========================================================================"
    echo "  Experiment dir: ${experiment_dir}"
    echo ""
    
    python "${PROJECT_ROOT}/Workers/step3_run_inference.py" \
        --experiment_dir "${experiment_dir}"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ ${experiment_name} inference complete"
        echo "    → evaluation_results.json"
        echo "    → generated_data/*.pkl"
    else
        echo "  ✗ ${experiment_name} inference failed (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Counters
TOTAL=0
SUCCESS=0
FAILED=0
SKIPPED=0

# Safe increment function
increment() {
    local var_name=$1
    eval "$var_name=\$(( $var_name + 1 ))"
}

# Arrays to store results
declare -a SUCCESSFUL_DIRS
declare -a FAILED_DIRS
declare -a SKIPPED_DIRS

echo ""
echo "========================================================================"
echo "Scanning for experiment directories..."
echo "========================================================================"

# Scan for experiment directories
for dir in "${BASE_OUTPUT}"/*; do
    if [ ! -d "${dir}" ]; then
        continue
    fi
    
    dir_name=$(basename "${dir}")
    
    # Skip non-Setting directories (e.g., precalc_results)
    if [[ ! "${dir_name}" =~ ^Setting ]]; then
        echo "  Skipping non-Setting directory: ${dir_name}"
        continue
    fi
    
    # Check if this is a direct experiment directory
    if is_valid_experiment_dir "${dir}"; then
        echo "  Found experiment: ${dir_name}"
        increment TOTAL
        
        if run_inference "${dir}"; then
            increment SUCCESS
            SUCCESSFUL_DIRS+=("${dir_name}")
        else
            increment FAILED
            FAILED_DIRS+=("${dir_name}")
        fi
    else
        # Check for nested experiment directories (e.g., Setting4/Setting4_Ablation_*)
        has_nested=false
        for subdir in "${dir}"/*; do
            if [ -d "${subdir}" ] && is_valid_experiment_dir "${subdir}"; then
                has_nested=true
                subdir_name=$(basename "${subdir}")
                echo "  Found nested experiment: ${dir_name}/${subdir_name}"
                increment TOTAL
                
                if run_inference "${subdir}"; then
                    increment SUCCESS
                    SUCCESSFUL_DIRS+=("${dir_name}/${subdir_name}")
                else
                    increment FAILED
                    FAILED_DIRS+=("${dir_name}/${subdir_name}")
                fi
            fi
        done
        
        if [ "$has_nested" = false ]; then
            echo "  Skipping ${dir_name}: no experiment_config.yaml or checkpoints/"
            increment SKIPPED
            SKIPPED_DIRS+=("${dir_name}")
        fi
    fi
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "========================================================================"
echo "Inference Summary (EMT_E2M2E)"
echo "========================================================================"
echo ""
echo "Total experiments processed: ${TOTAL}"
echo "  ✓ Successful: ${SUCCESS}"
echo "  ✗ Failed: ${FAILED}"
echo "  ⊘ Skipped: ${SKIPPED}"
echo ""

if [ ${#SUCCESSFUL_DIRS[@]} -gt 0 ]; then
    echo "Successful experiments:"
    for dir in "${SUCCESSFUL_DIRS[@]}"; do
        echo "  ✓ ${dir}"
    done
    echo ""
fi

if [ ${#FAILED_DIRS[@]} -gt 0 ]; then
    echo "Failed experiments:"
    for dir in "${FAILED_DIRS[@]}"; do
        echo "  ✗ ${dir}"
    done
    echo ""
fi

if [ ${#SKIPPED_DIRS[@]} -gt 0 ]; then
    echo "Skipped directories:"
    for dir in "${SKIPPED_DIRS[@]}"; do
        echo "  ⊘ ${dir}"
    done
    echo ""
fi

echo "Output files for each experiment:"
echo "  - evaluation_results.json    : Evaluation metrics"
echo "  - generated_data/{model}.pkl : Generated data for visualization"
echo ""
echo "========================================================================"

# Exit with error if any failed
if [ $FAILED -gt 0 ]; then
    echo "⚠️  Some inference runs failed. Check the logs for details."
    exit 1
else
    echo "✓ All inference runs completed successfully!"
    exit 0
fi
