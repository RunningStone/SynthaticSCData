#!/bin/bash
# Test script for run_exp7.sh
# This script tests the configuration reading and checkpoint finding logic

set -e

echo "Testing run_exp7.sh configuration..."
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: ${PROJECT_ROOT}"

# Test YAML reading
DATA_CONFIG="${PROJECT_ROOT}/configs/data_EMT_Cook_with_label.yaml"
echo ""
echo "Reading data config from: ${DATA_CONFIG}"

DATA_PATH=$(python3 -c "import yaml; config = yaml.safe_load(open('${DATA_CONFIG}')); print(config['data_source']['file_path'])")
TIME_COLUMN=$(python3 -c "import yaml; config = yaml.safe_load(open('${DATA_CONFIG}')); print(config['data_source']['obs_time_column'])")

echo "  Data path: ${DATA_PATH}"
echo "  Time column: ${TIME_COLUMN}"

# Test checkpoint finding
BASE_OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData"

echo ""
echo "Testing checkpoint finding..."

find_best_checkpoint() {
    local setting_dir=$1
    local ckpt_dir="${setting_dir}/checkpoints"
    
    # Try sb_mlplus first
    if [ -f "${ckpt_dir}/sb_mlplus/best_model.pt" ]; then
        echo "${ckpt_dir}/sb_mlplus/best_model.pt"
    elif [ -f "${ckpt_dir}/sb_mlplus/final_model.pt" ]; then
        echo "${ckpt_dir}/sb_mlplus/final_model.pt"
    # Fall back to sb
    elif [ -f "${ckpt_dir}/sb/best_model.pt" ]; then
        echo "${ckpt_dir}/sb/best_model.pt"
    elif [ -f "${ckpt_dir}/sb/final_model.pt" ]; then
        echo "${ckpt_dir}/sb/final_model.pt"
    else
        echo ""
    fi
}

for setting in Setting1 Setting2 Setting3; do
    setting_dir="${BASE_OUTPUT_DIR}/EMT_Part1_${setting}"
    echo ""
    echo "Checking ${setting}:"
    echo "  Directory: ${setting_dir}"
    
    if [ -d "${setting_dir}" ]; then
        echo "  ✓ Directory exists"
        
        ckpt=$(find_best_checkpoint "${setting_dir}")
        if [ -n "${ckpt}" ]; then
            echo "  ✓ Checkpoint found: ${ckpt}"
        else
            echo "  ✗ No checkpoint found"
        fi
    else
        echo "  ✗ Directory does not exist"
    fi
done

echo ""
echo "========================================================================"
echo "Test complete!"
echo "========================================================================"
