#!/bin/bash
# Calculate optimal data split parameters for EMT dataset with removal labels
# New naming convention:
#   - Part1 (Forward EMT only): setting1, setting2, setting3
#   - Part2 (With Reversal): setting1, setting2, setting3 (maps to old setting4, setting5, setting6)

# Activate virtual environment if needed
# source .venv/bin/activate

echo "=================================="
echo "Data Split Parameter Calculation"
echo "=================================="
echo ""
echo "Experimental Parts:"
echo "  Part1 (Forward EMT): setting1, setting2, setting3"
echo "  Part2 (With Reversal): setting1, setting2, setting3"
echo ""

# Default: use 100% of bottleneck capacity
# To use a different percentage, add: --bottleneck_percentage 90.0
BOTTLENECK_PCT=${1:-100.0}

if [ "$BOTTLENECK_PCT" != "100.0" ]; then
    echo "Using ${BOTTLENECK_PCT}% of bottleneck capacity"
    echo ""
fi

# Run the calculation script with two experimental groups
python step0_calculate_data_split_params.py \
    --data_config configs/data_EMT_Cook_with_label.yaml \
    --settings setting1 setting2 setting3 setting4 setting5 setting6 \
    --min_cells 1000 \
    --group1_settings setting1 setting2 setting3 \
    --group2_settings setting4 setting5 setting6 \
    --bottleneck_percentage $BOTTLENECK_PCT \
    --output_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/split_params_EMT_with_removal

echo ""
echo "=================================="
echo "Results saved to: /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/split_params_EMT_with_removal"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Review the generated YAML snippets in the output directory"
echo "2. Update your experiment configs with the recommended parameters"
echo "3. Ensure settings within the same part use the same total_cells for fair comparison"
echo ""
echo "Note:"
echo "  - Part1 settings (setting1,2,3) will have the same total sample size"
echo "  - Part2 settings (setting1,2,3) will have the same total sample size"
echo "  - Part1 and Part2 may have different totals (based on their respective bottlenecks)"
echo ""
echo "Usage:"
echo "  bash step0_calculate_data_split_params.sh          # Use 100% of bottleneck (default)"
echo "  bash step0_calculate_data_split_params.sh 90.0     # Use 90% of bottleneck"
echo "  bash step0_calculate_data_split_params.sh 80.0     # Use 80% of bottleneck"
