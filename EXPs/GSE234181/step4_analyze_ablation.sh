#!/bin/bash
# ============================================================================
# Step 4: Analyze Ablation Study Results for GSE234181
# ============================================================================
#
# This script analyzes the ablation study results by comparing Setting4
# ablation experiments with the full Setting2 baseline.
#
# Ablation experiments for GSE234181:
#   - Setting4_Ablation_RemoveT1: Remove T1 timepoint
#   - Setting4_Ablation_RemoveT2: Remove T2 timepoint
#
# Usage:
#   bash step4_analyze_ablation.sh
#
# ============================================================================

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Configuration
BASE_OUTPUT="/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/GSE234181"
ANALYSIS_OUTPUT="$BASE_OUTPUT/ablation_analysis"

echo "========================================================================"
echo "Step 4: Ablation Study Analysis (GSE234181)"
echo "========================================================================"
echo ""
echo "Base output directory: ${BASE_OUTPUT}"
echo "Analysis output: ${ANALYSIS_OUTPUT}"
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Create analysis output directory
mkdir -p "$ANALYSIS_OUTPUT"

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
fi

echo ""
echo "========================================================================"
echo "Checking Required Experiments..."
echo "========================================================================"
echo ""

# Check if baseline (Setting2) exists
SETTING2_DIR="$BASE_OUTPUT/Setting2"
if [ ! -d "$SETTING2_DIR" ] || [ ! -f "$SETTING2_DIR/evaluation_results.json" ]; then
    echo "❌ ERROR: Setting2 (baseline) not found or not evaluated"
    echo "   Please run Setting2 training and inference first:"
    echo "   bash step2_train_setting2.sh"
    echo "   bash step3_run_inference.sh"
    exit 1
fi
echo "✓ Setting2 (baseline) found"

# Check ablation experiments
ABLATION_DIRS=(
    "Setting4_Ablation_RemoveT1"
    "Setting4_Ablation_RemoveT2"
)

MISSING_ABLATIONS=()
for ablation in "${ABLATION_DIRS[@]}"; do
    ablation_dir="$BASE_OUTPUT/$ablation"
    if [ ! -d "$ablation_dir" ] || [ ! -f "$ablation_dir/evaluation_results.json" ]; then
        MISSING_ABLATIONS+=("$ablation")
        echo "⚠️  $ablation not found or not evaluated"
    else
        echo "✓ $ablation found"
    fi
done

if [ ${#MISSING_ABLATIONS[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Warning: Some ablation experiments are missing."
    echo "   Missing: ${MISSING_ABLATIONS[*]}"
    echo "   Continuing with available experiments..."
fi

echo ""
echo "========================================================================"
echo "Running Ablation Analysis..."
echo "========================================================================"
echo ""

# Create a simple Python script to analyze ablation results
python << 'EOF'
import json
import os
from pathlib import Path

base_output = "/home/pan/Experiments/EXPs/2025_10_VCC_Exps/OUTPUTs/GSE234181"
analysis_output = f"{base_output}/ablation_analysis"

# Load baseline (Setting2) results
setting2_path = f"{base_output}/Setting2/evaluation_results.json"
if os.path.exists(setting2_path):
    with open(setting2_path, 'r') as f:
        baseline = json.load(f)
else:
    print("❌ Setting2 results not found")
    exit(1)

# Ablation experiments
ablations = {
    "Remove_T1": f"{base_output}/Setting4_Ablation_RemoveT1/evaluation_results.json",
    "Remove_T2": f"{base_output}/Setting4_Ablation_RemoveT2/evaluation_results.json",
}

# Collect results
results = {"baseline": baseline}
for name, path in ablations.items():
    if os.path.exists(path):
        with open(path, 'r') as f:
            results[name] = json.load(f)

# Generate analysis report
report = []
report.append("=" * 80)
report.append("GSE234181 Ablation Study Analysis")
report.append("=" * 80)
report.append("")
report.append("Baseline: Setting2 (All 4 timepoints: T0, T1, T2, T3)")
report.append("")

# Compare metrics
metrics_to_compare = ["test_loss", "frechet_distance", "mae", "pcc", "wasserstein_distance", "mmd"]

report.append("-" * 80)
report.append("Metric Comparison (vs Baseline)")
report.append("-" * 80)
report.append("")

for metric in metrics_to_compare:
    report.append(f"### {metric.upper()} ###")
    
    # Get baseline value
    baseline_val = None
    if "sb_mlplus" in baseline:
        baseline_val = baseline["sb_mlplus"].get(metric)
    
    if baseline_val is not None:
        report.append(f"  Baseline (Setting2): {baseline_val:.4f}")
        
        for name, data in results.items():
            if name == "baseline":
                continue
            if "sb_mlplus" in data:
                val = data["sb_mlplus"].get(metric)
                if val is not None:
                    diff = val - baseline_val
                    pct = (diff / baseline_val * 100) if baseline_val != 0 else 0
                    direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
                    report.append(f"  {name}: {val:.4f} ({direction} {abs(diff):.4f}, {pct:+.1f}%)")
    else:
        report.append(f"  (metric not available)")
    
    report.append("")

report.append("-" * 80)
report.append("Marginal Contribution Analysis")
report.append("-" * 80)
report.append("")
report.append("Interpretation:")
report.append("  - Larger performance drop when removing a timepoint = higher marginal contribution")
report.append("  - Smaller performance drop = timepoint is less critical for trajectory learning")
report.append("")

# Save report
os.makedirs(analysis_output, exist_ok=True)
report_path = f"{analysis_output}/ablation_analysis_report.txt"
with open(report_path, 'w') as f:
    f.write('\n'.join(report))

print('\n'.join(report))
print("")
print(f"Report saved to: {report_path}")

# Save JSON summary
summary = {
    "baseline": "Setting2",
    "ablations": list(ablations.keys()),
    "results": results
}
summary_path = f"{analysis_output}/ablation_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to: {summary_path}")
EOF

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Ablation Analysis Complete"
    echo "========================================================================"
    echo ""
    echo "Output files:"
    echo "  - $ANALYSIS_OUTPUT/ablation_analysis_report.txt"
    echo "  - $ANALYSIS_OUTPUT/ablation_summary.json"
    echo ""
else
    echo "❌ Ablation Analysis Failed"
    echo "========================================================================"
    echo ""
    echo "Exit code: $EXIT_CODE"
    echo ""
fi

exit $EXIT_CODE
