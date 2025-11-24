#!/bin/bash
# 批量Evaluation脚本 - 自动扫描并评估所有settings
# 用于重新评估已训练的模型

# Parse arguments
BASE_DIR="${1:-}"
PART="${2:-Part1}"
CONFIG_DIR="${3:-configs}"

# Function to show usage
show_usage() {
    echo "========================================================================"
    echo "Batch Evaluation for All Settings"
    echo "========================================================================"
    echo ""
    echo "Usage: $0 <base_dir> [part] [config_dir]"
    echo ""
    echo "Arguments:"
    echo "  base_dir    : Base directory containing experiment outputs"
    echo "  part        : Experiment part (Part1 or Part2, default: Part1)"
    echo "  config_dir  : Directory with config files (default: configs)"
    echo ""
    echo "Examples:"
    echo "  # Evaluate all Part1 settings"
    echo "  $0 /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData"
    echo ""
    echo "  # Evaluate all Part2 settings"
    echo "  $0 /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData Part2"
    echo ""
    echo "Directory Structure Expected:"
    echo "  base_dir/"
    echo "    ├── EMT_Part1_Setting1/"
    echo "    │   └── checkpoints/"
    echo "    │       ├── sb/"
    echo "    │       ├── ot/"
    echo "    │       └── vae/"
    echo "    ├── EMT_Part1_Setting2/"
    echo "    │   └── checkpoints/"
    echo "    ├── EMT_Part1_Setting4/"
    echo "    │   ├── experiment_EMT_Part1_setting4_ablation_remove_8h/"
    echo "    │   │   └── checkpoints/"
    echo "    │   ├── experiment_EMT_Part1_setting4_ablation_remove_1d/"
    echo "    │   │   └── checkpoints/"
    echo "    │   └── experiment_EMT_Part1_setting4_ablation_remove_3d/"
    echo "    │       └── checkpoints/"
    echo "    └── ..."
    echo ""
    echo "========================================================================"
}

# Check if required arguments are provided
if [ -z "$BASE_DIR" ]; then
    show_usage
    exit 1
fi

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory not found: $BASE_DIR"
    echo ""
    show_usage
    exit 1
fi

echo "========================================================================"
echo "Batch Evaluation for All Settings"
echo "========================================================================"
echo ""
echo "Base directory: $BASE_DIR"
echo "Part: $PART"
echo "Config directory: $CONFIG_DIR"
echo ""

# Activate environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
else
    echo "Warning: No virtual environment found (.venv or venv)"
fi

echo ""
echo "========================================================================"
echo "Scanning for available settings..."
echo "========================================================================"
echo ""

# Function to find config file for a setting
find_config_file() {
    local setting_name=$1
    local part_lower=$(echo "$PART" | tr '[:upper:]' '[:lower:]')
    local setting_lower=$(echo "$setting_name" | tr '[:upper:]' '[:lower:]')
    
    # Extract base setting number and suffix
    # e.g., Setting5_Shuffled -> setting5_shuffled
    # e.g., Setting1 -> setting1
    local base_setting=$(echo "$setting_lower" | sed 's/_/ /g' | awk '{print $1}')
    local suffix=$(echo "$setting_lower" | sed 's/^[^_]*_*//')
    
    # Try different config file naming patterns
    local patterns=(
        "experiment_EMT_${PART}_${setting_lower}.yaml"
        "experiment_EMT_${part_lower}_${setting_lower}.yaml"
        "experiment_EMT_${PART}_${base_setting}.yaml"
        "experiment_EMT_${part_lower}_${base_setting}.yaml"
    )
    
    # If there's a suffix, try with underscore separator
    if [ "$base_setting" != "$setting_lower" ]; then
        patterns+=(
            "experiment_EMT_${PART}_${base_setting}_${suffix}.yaml"
            "experiment_EMT_${part_lower}_${base_setting}_${suffix}.yaml"
        )
    fi
    
    for pattern in "${patterns[@]}"; do
        if [ -f "$CONFIG_DIR/$pattern" ]; then
            echo "$pattern"
            return 0
        fi
    done
    
    return 1
}

# Function to check if a directory contains model checkpoints
has_checkpoints() {
    local checkpoint_dir=$1
    
    if [ ! -d "$checkpoint_dir" ]; then
        return 1
    fi
    
    # Check for model subdirectories with checkpoints
    local model_count=0
    for model_dir in "$checkpoint_dir"/*; do
        if [ -d "$model_dir" ]; then
            if [ -f "$model_dir/best_model.pt" ] || [ -f "$model_dir/final_model.pt" ]; then
                ((model_count++))
            fi
        fi
    done
    
    [ $model_count -gt 0 ]
}

# Function to evaluate a single setting
evaluate_setting() {
    local setting_path=$1
    local checkpoint_dir=$2
    local setting_display_name=$3
    
    echo ""
    echo "========================================================================"
    echo "Evaluating: $setting_display_name"
    echo "========================================================================"
    echo "  Setting path: $setting_path"
    echo "  Checkpoint dir: $checkpoint_dir"
    echo "  Evaluation framework: $SETTING1_CONFIG"
    echo ""
    
    # Count available models
    local model_count=0
    local models=""
    for model_dir in "$checkpoint_dir"/*; do
        if [ -d "$model_dir" ]; then
            local model_name=$(basename "$model_dir")
            if [ -f "$model_dir/best_model.pt" ] || [ -f "$model_dir/final_model.pt" ]; then
                models="$models $model_name"
                ((model_count++))
            fi
        fi
    done
    
    echo "  Available models ($model_count):$models"
    echo ""
    
    # Run evaluation using Setting1 config for unified evaluation framework
    python step2_run_evaluation_only.py \
        "$SETTING1_CONFIG" \
        "$checkpoint_dir" \
        --config_dir "$CONFIG_DIR"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "  ✓ $setting_display_name evaluation complete"
    else
        echo "  ✗ $setting_display_name evaluation failed (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Find Setting1 config file for unified evaluation framework
SETTING1_CONFIG=$(find_config_file "Setting1")
if [ -z "$SETTING1_CONFIG" ]; then
    echo "ERROR: Cannot find Setting1 config file for $PART"
    echo "Setting1 config is required for unified evaluation framework"
    exit 1
fi

echo "Using Setting1 config for unified evaluation: $SETTING1_CONFIG"
echo ""

# Counters
TOTAL_SETTINGS=0
SUCCESSFUL_SETTINGS=0
FAILED_SETTINGS=0
SKIPPED_SETTINGS=0

# Scan for settings
for setting_dir in "$BASE_DIR"/EMT_${PART}_*; do
    if [ ! -d "$setting_dir" ]; then
        continue
    fi
    
    setting_name=$(basename "$setting_dir")
    
    # Extract setting number/name (e.g., Setting1, Setting2, Setting4)
    if [[ $setting_name =~ EMT_${PART}_(Setting[0-9]+.*) ]]; then
        setting_id="${BASH_REMATCH[1]}"
    else
        continue
    fi
    
    # Check if this is Setting4 with sub-experiments
    if [[ $setting_id == "Setting4" ]]; then
        # Handle Setting4 ablation experiments
        for sub_exp_dir in "$setting_dir"/experiment_*; do
            if [ ! -d "$sub_exp_dir" ]; then
                continue
            fi
            
            checkpoint_dir="$sub_exp_dir/checkpoints"
            
            if ! has_checkpoints "$checkpoint_dir"; then
                echo " Skipping $setting_name/$(basename $sub_exp_dir): No checkpoints found"
                ((SKIPPED_SETTINGS++))
                continue
            fi
            
            ((TOTAL_SETTINGS++))
            
            if evaluate_setting "$sub_exp_dir" "$checkpoint_dir" "$setting_name/$(basename $sub_exp_dir)"; then
                ((SUCCESSFUL_SETTINGS++))
            else
                ((FAILED_SETTINGS++))
            fi
        done
    else
        # Regular setting (Setting1, Setting2, Setting3, Setting5, Setting6, etc.)
        checkpoint_dir="$setting_dir/checkpoints"
        
        if ! has_checkpoints "$checkpoint_dir"; then
            echo "⊘ Skipping $setting_name: No checkpoints found"
            ((SKIPPED_SETTINGS++))
            continue
        fi
        
        ((TOTAL_SETTINGS++))
        
        if evaluate_setting "$setting_dir" "$checkpoint_dir" "$setting_name"; then
            ((SUCCESSFUL_SETTINGS++))
        else
            ((FAILED_SETTINGS++))
        fi
    fi
done

# Summary
echo ""
echo "========================================================================"
echo "Batch Evaluation Summary"
echo "========================================================================"
echo ""
echo "Total settings processed: $TOTAL_SETTINGS"
echo "  ✓ Successful: $SUCCESSFUL_SETTINGS"
echo "  ✗ Failed: $FAILED_SETTINGS"
echo "  ⊘ Skipped: $SKIPPED_SETTINGS"
echo ""

if [ $FAILED_SETTINGS -eq 0 ]; then
    echo "✓ All evaluations completed successfully!"
    echo ""
    echo "Generated files for each setting:"
    echo "  - results.json              : Evaluation metrics"
    echo "  - generated/{model}.pkl     : Visualization data"
else
    echo "⚠ Some evaluations failed. Check the logs above for details."
fi

echo ""
echo "========================================================================"

# Exit with error if any evaluation failed
[ $FAILED_SETTINGS -eq 0 ]
