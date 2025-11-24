# Setting 5 Quick Start Guide

## 🎯 What is Setting 5?

**Experiment 5: Time Information Decoupling**

Tests whether your model truly learns time-dependent dynamics or just memorizes spatial mappings by **shuffling temporal relationships** while preserving time interval statistics.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Test the Implementation (5 minutes)

```bash
cd /home/pan/Experiments/EXPs/2025_10_VCC_Exps/other_repos/SynthaticSCData

# Test shuffled dataset
python Data/shuffled_dataset.py

# Test shuffled data loader
python Data/shuffled_data_loader.py
```

**Expected output**:
- ✅ 8,974 shuffled pairs generated
- ✅ KS test p-value > 0.05
- ✅ Δt distribution matches expected

### Step 2: Train the Model (3-4 hours on GPU)

```bash
# Activate environment
source .venv/bin/activate

# Train Setting 5 (shuffled)
bash step1_run_experiment_EMT.sh experiment_EMT_Part1_setting5_shuffled.yaml
```

**What happens**:
- Trains sb_mlplus model on shuffled time series
- 200 epochs, batch_size=64
- Evaluates on real ordered test data
- Computes trajectory continuity metric

### Step 3: Compare with Setting 2 (10 minutes)

```bash
# Generate comparison visualization
bash step2_run_multi_setting_visualization.sh \
    experiment_EMT_Part1_setting2.yaml \
    experiment_EMT_Part1_setting5_shuffled.yaml
```

**Outputs**:
- Metrics comparison chart
- PHATE/LMNN visualizations
- Performance degradation analysis

---

## 📊 Interpreting Results

### Key Metric: Causal Information Contribution

```
C_causal = (P_ordered - P_shuffled) / P_ordered × 100%
```

| C_causal | Interpretation | Action |
|----------|----------------|--------|
| **> 30%** | Time is critical | ✅ Your model learns dynamics! |
| **10-30%** | Time helps but not dominant | ⚠️ Consider temporal regularization |
| **< 10%** | Time is negligible | ❌ Model just memorizes spatial maps |

### Trajectory Smoothness Ratio

```
R_smooth = J_shuffled / J_ordered
```

| R_smooth | Interpretation |
|----------|----------------|
| **> 2** | Shuffled trajectories are very discontinuous |
| **1-2** | Moderate discontinuity |
| **≈ 1** | No effect on smoothness |

---

## 📁 Files Created

```
Data/
├── shuffled_dataset.py          # ShuffledTimeSeriesDataset class
└── shuffled_data_loader.py      # ShuffledDataLoader class

Trainer/
└── metrics.py                   # Added trajectory_continuity metric

configs/
└── experiment_EMT_Part1_setting5_shuffled.yaml  # Experiment config

Documentation/
├── SETTING5_IMPLEMENTATION_SUMMARY.md  # Full documentation
└── SETTING5_QUICK_START.md            # This file
```

---

## 🔧 Configuration Highlights

**Key flags in config**:
```yaml
data_sampling_override:
  use_shuffled_dataset: true     # Enable shuffling
  total_cells: 8974              # Same as Setting2
  shuffle_seed: 42               # Reproducibility
  validate_distribution: true    # KS test

evaluation:
  compute_trajectory_continuity: true  # New metric
  trajectory_continuity_samples: 500
```

---

## 🎓 Scientific Significance

### What This Experiment Reveals

1. **If performance drops significantly (C > 30%)**:
   - ✅ Model truly learns time-dependent dynamics
   - ✅ Validates Schrödinger Bridge framework
   - ✅ Temporal ordering is biologically meaningful

2. **If performance maintains (C < 10%)**:
   - ⚠️ Model mainly uses spatial information
   - ⚠️ Time acts as categorical label, not dynamics
   - ⚠️ Need stronger temporal regularization

3. **Mixed results (10% < C < 30%)**:
   - 🤔 Time affects global structure, not local prediction
   - 🤔 Different metrics capture different aspects
   - 🤔 Hierarchical role of temporal information

---

## 🐛 Troubleshooting

### Issue: KS test fails (p < 0.05)
**Solution**: Check time_intervals in config match data_EMT_Cook_with_label.yaml

### Issue: Cannot generate enough pairs
**Solution**: Reduce n_samples or check available cells per timepoint

### Issue: Training diverges
**Solution**: This is actually interesting! It means the model cannot learn from shuffled data.

---

## 📚 More Information

- **Full documentation**: `SETTING5_IMPLEMENTATION_SUMMARY.md`
- **Design document**: `20251118_setting5_shuffle.md`
- **Experiment overview**: `20251118_setting4567_overview.md`

---

## ✅ Checklist

Before training:
- [ ] Unit tests pass (`python Data/shuffled_dataset.py`)
- [ ] KS test p-value > 0.05
- [ ] 8,974 pairs generated successfully

After training:
- [ ] Model converged (check training loss)
- [ ] Evaluation completed
- [ ] Trajectory continuity computed
- [ ] Comparison with Setting 2 generated

---

## 🎯 Expected Timeline

| Task | Time | Status |
|------|------|--------|
| Test implementation | 5 min | ⏳ Ready |
| Train Setting 5 | 3-4 hours | ⏳ Ready |
| Compare with Setting 2 | 10 min | ⏳ Ready |
| Analyze results | 1 hour | ⏳ Pending |
| Write up findings | 2 hours | ⏳ Pending |

---

## 💡 Pro Tips

1. **Run Setting 2 first** if you haven't already - you need it for comparison
2. **Check GPU memory** - sb_mlplus needs ~4GB
3. **Save intermediate results** - training takes hours
4. **Document findings** - this is novel research!

---

## 🎉 You're Ready!

Setting 5 is fully implemented and ready to run. This experiment will provide crucial insights into whether your model truly learns temporal dynamics.

**Next command**:
```bash
python Data/shuffled_dataset.py  # Start here!
```

Good luck! 🚀
