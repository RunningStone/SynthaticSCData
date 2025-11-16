#!/bin/bash
# Example: Calculate optimal data split parameters for EMT dataset with removal labels

# Activate virtual environment if needed
# source .venv/bin/activate

# Run the calculation script
python step0_calculate_data_split_params.py \
    --data_config configs/data_EMT_Cook_with_label.yaml \
    --settings setting1 setting2 setting3 \
    --min_cells 1000 \
    --output_dir /home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/split_params_EMT_with_removal

echo ""
echo "Results saved to: ./outputs/split_params_with_removal"
echo ""
echo "Next steps:"
echo "1. Review the generated YAML snippets in ./outputs/split_params_with_removal/"
echo "2. Update your experiment configs with the recommended parameters"
echo "3. Ensure all settings use the same total_cells for fair comparison"
