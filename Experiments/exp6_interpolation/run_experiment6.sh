#!/bin/bash
# ============================================================================
# Experiment 6: Interpolation Quality Analysis - Runner Script
# ============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting6"
SETTING1_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting1"
SETTING2_DIR="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/SynthaticSCData/EMT_Part1_Setting2"
CONFIG_FILE="experiment_EMT_Part1_setting6_interpolated.yaml"

# Parse command line arguments
DATA_ONLY=false
TRAIN_ONLY=false
ANALYSIS_ONLY=false
SKIP_DATA=false
SKIP_TRAIN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-only)
            DATA_ONLY=true
            shift
            ;;
        --train-only)
            TRAIN_ONLY=true
            SKIP_DATA=true
            shift
            ;;
        --analysis-only)
            ANALYSIS_ONLY=true
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data-only          Only generate interpolated data (via training pipeline)"
            echo "  --train-only         Only train models"
            echo "  --analysis-only      Only run analysis"
            echo "  --output PATH        Output directory"
            echo "  --config FILE        Config file name (default: experiment_EMT_Part1_setting6_interpolated.yaml)"
            echo "  --help               Show this help message"
            echo ""
            echo "Note: Data generation is now integrated into the training pipeline."
            echo "      Input data path is specified in the config file."
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}EXPERIMENT 6: INTERPOLATION QUALITY ANALYSIS${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Project root:    $PROJECT_ROOT"
echo "  Config file:     $CONFIG_FILE"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Setting 1 dir:   $SETTING1_DIR"
echo "  Setting 2 dir:   $SETTING2_DIR"
echo ""
echo -e "${YELLOW}Note: Input data path is read from config file${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Check if config file exists
CONFIG_PATH="$PROJECT_ROOT/configs/$CONFIG_FILE"
if [ ! -f "$CONFIG_PATH" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_PATH${NC}"
    exit 1
fi

# Step 1 & 2: Train models (data generation is integrated)
if [ "$ANALYSIS_ONLY" = false ]; then
    echo ""
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}TRAINING MODELS (with integrated data generation)${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Data will be automatically generated during training pipeline${NC}"
    echo ""
    
    bash step1_run_experiment_EMT.sh "$CONFIG_FILE"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model training completed successfully${NC}"
    else
        echo -e "${RED}✗ Model training failed${NC}"
        exit 1
    fi
    
    if [ "$TRAIN_ONLY" = true ] || [ "$DATA_ONLY" = true ]; then
        echo ""
        echo -e "${GREEN}Training complete. Exiting.${NC}"
        exit 0
    fi
fi

# Step 3: Analyze results
if [ "$ANALYSIS_ONLY" = true ] || [ "$DATA_ONLY" = false ] && [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}STEP 3: ANALYZING RESULTS${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo ""
    
    # Check if results exist
    RESULTS_FILE="$OUTPUT_DIR/results.json"
    if [ ! -f "$RESULTS_FILE" ]; then
        echo -e "${YELLOW}Warning: Results file not found: $RESULTS_FILE${NC}"
        echo -e "${YELLOW}Make sure models have been trained before running analysis.${NC}"
    fi
    
    # Create analysis directory
    ANALYSIS_DIR="$OUTPUT_DIR/analysis"
    mkdir -p "$ANALYSIS_DIR"
    
    echo -e "${GREEN}✓ Analysis directory created: $ANALYSIS_DIR${NC}"
    echo ""
    echo -e "${YELLOW}Note: Full analysis requires trained models and generated samples.${NC}"
    echo -e "${YELLOW}Please implement the detailed analysis script as needed.${NC}"
fi

# Print summary
echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}EXPERIMENT 6 COMPLETE${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${GREEN}Results saved to:${NC}"
echo "  Output directory:  $OUTPUT_DIR"
echo "  Model checkpoints: $OUTPUT_DIR/checkpoints/"
echo "  Results:           $OUTPUT_DIR/results.json"
echo "  Analysis:          $OUTPUT_DIR/analysis/"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Check model training logs in $OUTPUT_DIR/logs/"
echo "  2. Analyze interpolation effectiveness metrics"
echo "  3. Compare with Setting 1 and Setting 2 results"
echo ""
echo -e "${GREEN}Note:${NC}"
echo "  Interpolated data was generated automatically during training"
echo "  and is embedded in the data loading pipeline."
echo ""
