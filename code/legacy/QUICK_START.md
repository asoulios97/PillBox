# 🚀 Quick Start Guide - K-Fold Cross-Validation with Experiment Tracking

## 📂 Directory Structure After Training

```
code/results/
├── experiment_20260117_1430/          ← Run 1 (Jan 17, 2026 at 14:30)
│   ├── experiment_config.json         (hyperparameters)
│   ├── models/                        (5 model checkpoints)
│   ├── metrics/                       (10 JSON files: results + history)
│   ├── splits/                        (5 split info files)
│   ├── predictions/                   (5 prediction files)
│   ├── plots/                         (16 PNG files)
│   ├── kfold_aggregated_results.json
│   └── summary_report.json
│
├── experiment_20260117_1645/          ← Run 2 (same day, 16:45)
│   └── ... (same structure)
│
└── experiment_20260118_0920/          ← Run 3 (next day, 09:20)
    └── ... (same structure)
```

## 🎯 How to Use

### Step 1: Run Training
```bash
python new_bottle_window_max.py
```

**Output:**
```
Experiment: experiment_20260117_1430
Results will be saved to: code/results/experiment_20260117_1430

FOLD 1/5
  Training: 12 users (1-3, 10-1, 11-1, ...)
  Validation: 3 users (5-1, 7-1, 8-1)
  ✓ No user overlap verified!
  
  Training... [50 epochs]
  [SAVED] Best model, results, history, predictions
  
FOLD 2/5
  ...
```

### Step 2: View Results
```bash
python view_results.py
```

**Output:**
```
Available experiments:
  [1] experiment_20260117_1430
  [2] experiment_20260117_1645
  [3] experiment_20260118_0920

✓ Using latest: experiment_20260118_0920

AGGREGATED METRICS (Mean ± Std)
================================
Accuracy:     0.9421 ± 0.0087
Class 1 F1:   0.8343 ± 0.0098
Segmental F1 (k=0.5): 0.7844 ± 0.0156

Creating visualizations...
  ✓ Saved: fold1_predictions.png
  ✓ Saved: fold1_segments.png
  ...
```

## 📊 What Gets Created

### For Each Fold (×5):

**models/fold{N}_best.pth**
- PyTorch checkpoint with best model
- Can load for inference or continued training

**metrics/fold{N}_results.json** (~5 KB)
```json
{
  "fold": 1,
  "best_val_f1": 0.8234,
  "frame_wise": { "frame_accuracy": 0.9345, ... },
  "segment_wise": {
    "k=0.1": { "TP": 45, "FP": 3, "FN": 2, "segmental_f1": 0.9345 },
    "k=0.25": { ... },
    "k=0.5": { ... }
  }
}
```

**metrics/fold{N}_history.json** (~3 KB)
```json
{
  "train_loss": [0.5, 0.4, 0.3, ...],
  "val_f1": [0.75, 0.78, 0.82, ...],
  "best_val_f1": 0.8234
}
```

**splits/fold{N}_split_info.json** (~4 KB)
```json
{
  "train_user_ids": ["10-1", "11-1", "12-1", ...],
  "val_user_ids": ["1-3", "5-1", "7-1"],
  "no_overlap_verified": true
}
```

**predictions/fold{N}_predictions.json** (~20-50 MB)
```json
{
  "predictions": [[0,0,0,1,1,1,0,...], ...],  ← 6 sessions
  "targets": [[0,0,0,1,1,1,0,...], ...],
  "val_sessions": ["1-3/Left", "1-3/Right", ...]
}
```

**plots/fold{N}_predictions.png**
- Timeline visualization (like your reference image)
- Shows predictions vs ground truth
- Gray shaded = drinking gestures

**plots/fold{N}_segments.png**
- TP/FP/FN visualization
- Top: Ground truth segments
- Bottom: Predicted segments colored by classification

### Aggregated Files:

**kfold_aggregated_results.json**
- Mean ± Std for all metrics
- All fold splits and results

**summary_report.json**
- Human-readable summary
- Quick reference for paper

**plots/** (6 images)
- Cross-fold comparisons
- Training curves
- Metric summaries with error bars

## 🔍 Inspecting Results

### View Predictions for Fold 2:
```python
import json
with open("results/experiment_20260117_1430/predictions/fold2_predictions.json") as f:
    data = json.load(f)
    
predictions = data['predictions']  # List of 6 sessions
targets = data['targets']
val_sessions = data['val_sessions']

print(f"Session 1 ({val_sessions[0]}): {len(predictions[0])} frames")
```

### Load Best Model from Fold 3:
```python
import torch
checkpoint = torch.load("results/experiment_20260117_1430/models/fold3_best.pth")
model.load_state_dict(checkpoint['model_state_dict'])
```

### Compare Two Experiments:
```python
# Load experiment 1 results
with open("results/experiment_20260117_1430/kfold_aggregated_results.json") as f:
    exp1 = json.load(f)

# Load experiment 2 results
with open("results/experiment_20260117_1645/kfold_aggregated_results.json") as f:
    exp2 = json.load(f)

# Compare
print(f"Exp1 F1: {exp1['frame_wise']['class_1_f1_mean']:.4f}")
print(f"Exp2 F1: {exp2['frame_wise']['class_1_f1_mean']:.4f}")
```

## 📊 Generated Visualizations

All plots saved to: `results/experiment_YYYYMMDD_HHMM/plots/`

1. **fold{N}_predictions.png** - Timeline view
   - Horizontal axis: time/frames
   - Shows prediction line vs shaded GT regions
   
2. **fold{N}_segments.png** - Segment analysis
   - Top: Ground truth drinking gestures
   - Bottom: Predicted segments (TP=teal, FP=gray)
   
3. **framewise_metrics_comparison.png**
   - Bar charts: Accuracy, Precision, Recall, F1
   - One bar per fold + mean line
   
4. **segmental_f1_comparison.png**
   - Grouped bars: IoU k=0.1, 0.25, 0.5
   - Shows how threshold affects F1
   
5. **training_curves.png**
   - 4 subplots: train/val loss, train/val acc
   - All 5 folds overlaid
   
6. **aggregated_metrics_summary.png**
   - Summary with error bars (Mean ± Std)

## ⚙️ Experiment Configuration

Every experiment folder contains `experiment_config.json` with:
- Model architecture settings
- Data preprocessing settings  
- Training hyperparameters
- Timestamp and experiment name

This ensures **complete reproducibility**!

## 🎓 For Your Thesis

Use these files:
- **Tables**: Data from kfold_aggregated_results.json
- **Figures**: Images from plots/ folder
- **Reproducibility**: experiment_config.json + split_info files
- **Detailed Analysis**: predictions/ for error analysis

