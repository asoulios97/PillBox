"""
view_results.py

Comprehensive visualization and analysis of K-Fold cross-validation results.
Loads all fold results and creates publication-quality visualizations.

Usage:
    python view_results.py                                      # Use latest experiment
    python view_results.py --experiment=experiment_20260117_1548
    python view_results.py --experiment experiment_20260117_1548
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import sys
import csv
from absl import app, flags

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'experiment',
    None,  # None = auto-detect latest experiment
    'Experiment folder name (e.g., experiment_20260117_1548). If not specified, uses the latest.'
)

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Paths
CODE_DIR = Path(__file__).parent
RESULTS_BASE = CODE_DIR / "results"


def plot_prediction_timeline(predictions, targets, title, save_path=None, max_frames=5000):
    """
    Create timeline visualization showing predictions vs ground truth.
    Highlights TP, FP, FN segments.
    """
    # Concatenate all sessions for this fold
    pred = np.concatenate(predictions)
    tgt = np.concatenate(targets)
    
    # Limit to first N frames for visibility
    if len(pred) > max_frames:
        pred = pred[:max_frames]
        tgt = tgt[:max_frames]
    
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Create time array
    time = np.arange(len(pred))
    
    # Plot predictions as thin black line
    ax.plot(time, pred * 0.4 + 0.5, 'k-', linewidth=1, alpha=0.8, label='Prediction')
    
    # Shade ground truth regions (gray)
    in_gt_segment = False
    start_gt = 0
    for i in range(len(tgt)):
        if tgt[i] == 1 and not in_gt_segment:
            start_gt = i
            in_gt_segment = True
        elif tgt[i] == 0 and in_gt_segment:
            ax.axvspan(start_gt, i, alpha=0.3, color='gray', label='Ground truth drinking gesture')
            in_gt_segment = False
    if in_gt_segment:
        ax.axvspan(start_gt, len(tgt), alpha=0.3, color='gray', label='Ground truth drinking gesture')
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    ax.set_ylim([0, 1.2])
    ax.set_xlabel('Frame Index (at 16Hz)', fontsize=11)
    ax.set_ylabel('Prediction', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['', 'Prediction', ''])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
    
    return fig


def plot_segments_with_iou(predictions, targets, title, iou_threshold=0.5, save_path=None, max_frames=5000):
    """
    Visualize segments with TP/FP/FN classification based on IoU.
    Similar to the second image in your reference.
    """
    pred = np.concatenate(predictions)
    tgt = np.concatenate(targets)
    
    if len(pred) > max_frames:
        pred = pred[:max_frames]
        tgt = tgt[:max_frames]
    
    # Find segments
    def get_segments(labels):
        segments = []
        in_segment = False
        start = 0
        for i, label in enumerate(labels):
            if label == 1 and not in_segment:
                start = i
                in_segment = True
            elif label != 1 and in_segment:
                segments.append((start, i))
                in_segment = False
        if in_segment:
            segments.append((start, len(labels)))
        return segments
    
    def compute_iou(seg1, seg2):
        start1, end1 = seg1
        start2, end2 = seg2
        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1, end2) - min(start1, start2)
        return intersection / union if union > 0 else 0.0
    
    pred_segments = get_segments(pred)
    gt_segments = get_segments(tgt)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    
    # Top plot: Ground truth segments (dark teal)
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('Ground Truth')
    for start, end in gt_segments:
        ax1.axvspan(start, end, alpha=0.6, color='darkslategray', 
                    label='Ground truth drinking gesture')
    ax1.set_yticks([])
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Bottom plot: Predicted segments colored by TP/FP/FN
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Predicted')
    ax2.set_xlabel('Frame Index (at 16Hz)', fontsize=11)
    
    # Classify each prediction
    matched_pred = set()
    matched_gt = set()
    
    # Find TPs
    for i, gt_seg in enumerate(gt_segments):
        best_iou = 0.0
        best_pred_idx = -1
        for j, pred_seg in enumerate(pred_segments):
            iou = compute_iou(gt_seg, pred_seg)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = j
        
        if best_iou >= iou_threshold and best_pred_idx >= 0:
            matched_gt.add(i)
            if best_pred_idx not in matched_pred:
                matched_pred.add(best_pred_idx)
                # True Positive (teal)
                start, end = pred_segments[best_pred_idx]
                ax2.axvspan(start, end, alpha=0.6, color='teal', label='TP (True Positive)')
    
    # Find FPs and FNs
    for j, pred_seg in enumerate(pred_segments):
        if j not in matched_pred:
            # False Positive (light gray)
            start, end = pred_seg
            ax2.axvspan(start, end, alpha=0.6, color='lightgray', label='FP (False Positive)')
    
    # FNs shown by showing GT that wasn't matched (on top plot with different color)
    for i, gt_seg in enumerate(gt_segments):
        if i not in matched_gt:
            start, end = gt_seg
            ax1.axvspan(start, end, alpha=0.8, color='darkblue', label='FN (False Negative)')
    
    ax2.set_yticks([])
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper right')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path.name}")
    
    return fig


def main(argv):
    """Main function for viewing results."""
    del argv  # Unused
    
    print("="*70)
    print("K-FOLD CROSS-VALIDATION RESULTS VIEWER")
    print("="*70)
    
    # ============================================================================
    # SELECT EXPERIMENT FOLDER
    # ============================================================================
    
    # Find all experiment folders
    if not RESULTS_BASE.exists():
        print(f"\n❌ Results folder not found: {RESULTS_BASE}")
        print("Run training first: python new_bottle_window_max.py")
        sys.exit(1)
    
    experiment_folders = sorted([d for d in RESULTS_BASE.iterdir() if d.is_dir() and d.name.startswith('experiment_')])
    
    if not experiment_folders:
        print(f"\n❌ No experiment folders found in: {RESULTS_BASE}")
        print("Run training first: python new_bottle_window_max.py")
        sys.exit(1)
    
    print(f"\nAvailable experiments:")
    for idx, exp_folder in enumerate(experiment_folders, 1):
        print(f"  [{idx}] {exp_folder.name}")
    
    # Select experiment based on user input or default to latest
    if FLAGS.experiment and FLAGS.experiment != 'None':
        # User specified an experiment
        RESULTS_DIR = RESULTS_BASE / FLAGS.experiment
        if not RESULTS_DIR.exists():
            print(f"\n❌ Specified experiment not found: {FLAGS.experiment}")
            print(f"\nAvailable experiments:")
            for exp_folder in experiment_folders:
                print(f"  - {exp_folder.name}")
            sys.exit(1)
        print(f"\n✓ Using specified experiment: {RESULTS_DIR.name}")
    else:
        # Use the most recent experiment
        RESULTS_DIR = experiment_folders[-1]
        print(f"\n✓ Using latest experiment: {RESULTS_DIR.name}")
        if len(experiment_folders) > 1:
            print(f"  (Use --experiment to specify a different one)")

    # Load experiment config if available
    try:
        with open(RESULTS_DIR / "experiment_config.json", 'r') as f:
            exp_config = json.load(f)
        print(f"\nExperiment Configuration:")
        print(f"  Timestamp: {exp_config['timestamp']}")
        print(f"  N-Folds: {exp_config['training_config']['n_folds']}")
        print(f"  Epochs: {exp_config['training_config']['num_epochs']}")
        print(f"  Window Size: {exp_config['data_config']['window_size']}")
    except:
        exp_config = None
        print("\n  (No config file found)")
    
    print("="*70)
    
    # ============================================================================
    # PART 1: LOAD ALL FOLD RESULTS
    # ============================================================================
    
    def load_fold_results(fold_idx: int) -> Dict:
        """Load all data for a specific fold."""
        fold_data = {}
        
        # Load metrics
        with open(RESULTS_DIR / "metrics" / f"fold{fold_idx}_results.json", 'r') as f:
            fold_data['results'] = json.load(f)
        
        # Load history
        with open(RESULTS_DIR / "metrics" / f"fold{fold_idx}_history.json", 'r') as f:
            fold_data['history'] = json.load(f)
        
        # Load split info
        with open(RESULTS_DIR / "splits" / f"fold{fold_idx}_split_info.json", 'r') as f:
            fold_data['split'] = json.load(f)
        
        # Load predictions
        with open(RESULTS_DIR / "predictions" / f"fold{fold_idx}_predictions.json", 'r') as f:
            pred_data = json.load(f)
            fold_data['predictions'] = [np.array(p) for p in pred_data['predictions']]
            fold_data['targets'] = [np.array(t) for t in pred_data['targets']]
        
        return fold_data
    
    print("\nLoading fold results...")
    n_folds = 5
    all_folds_data = []
    
    for fold_idx in range(1, n_folds + 1):
        try:
            fold_data = load_fold_results(fold_idx)
            all_folds_data.append(fold_data)
            print(f"  ✓ Loaded Fold {fold_idx}")
        except FileNotFoundError as e:
            print(f"  ✗ Fold {fold_idx} not found: {e}")
    
    if not all_folds_data:
        print("\n❌ No fold results found! Run training first.")
        sys.exit(1)
    
    print(f"\n✓ Loaded {len(all_folds_data)} folds successfully")

    # ============================================================================
    # PART 2: AGGREGATE METRICS ACROSS FOLDS
    # ============================================================================
    
    print("\n" + "="*70)
    print("AGGREGATED METRICS (Mean ± Std)")
    print("="*70)
    
    # Frame-wise metrics
    frame_accuracies = [f['results']['frame_wise']['frame_accuracy'] for f in all_folds_data]
    class1_precisions = [f['results']['frame_wise']['class_1_precision'] for f in all_folds_data]
    class1_recalls = [f['results']['frame_wise']['class_1_recall'] for f in all_folds_data]
    class1_f1s = [f['results']['frame_wise']['class_1_f1'] for f in all_folds_data]
    
    print("\n--- Frame-Wise Metrics ---")
    print(f"Accuracy:          {np.mean(frame_accuracies):.4f} ± {np.std(frame_accuracies):.4f}")
    print(f"\nClass 1 (Drinking):")
    print(f"  Precision:       {np.mean(class1_precisions):.4f} ± {np.std(class1_precisions):.4f}")
    print(f"  Recall:          {np.mean(class1_recalls):.4f} ± {np.std(class1_recalls):.4f}")
    print(f"  F1-score:        {np.mean(class1_f1s):.4f} ± {np.std(class1_f1s):.4f}")
    
    # Segment-wise metrics
    print(f"\n--- Segment-Wise Metrics (Segmental F1) ---")
    for iou_thresh in [0.1, 0.25, 0.5]:
        key = f'k={iou_thresh}'
        f1s = [f['results']['segment_wise'][key]['segmental_f1'] for f in all_folds_data]
        precs = [f['results']['segment_wise'][key]['precision'] for f in all_folds_data]
        recs = [f['results']['segment_wise'][key]['recall'] for f in all_folds_data]
        
        print(f"\nThreshold k={iou_thresh}:")
        print(f"  Precision:       {np.mean(precs):.4f} ± {np.std(precs):.4f}")
        print(f"  Recall:          {np.mean(recs):.4f} ± {np.std(recs):.4f}")
        print(f"  Segmental F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    # ============================================================================
    # PAPER-STYLE TABLE: Segment-wise Results
    # ============================================================================
    
    print("\n" + "="*100)
    print("PAPER-STYLE TABLE: Segment-wise Results at Different IoU Thresholds")
    print("="*100)
    
    # Calculate aggregated TP, FP, FN (total across all folds)
    tp_01 = sum([f['results']['segment_wise']['k=0.1']['TP'] for f in all_folds_data])
    fp_01 = sum([f['results']['segment_wise']['k=0.1']['FP'] for f in all_folds_data])
    fn_01 = sum([f['results']['segment_wise']['k=0.1']['FN'] for f in all_folds_data])
    f1_01 = [f['results']['segment_wise']['k=0.1']['segmental_f1'] for f in all_folds_data]
    
    tp_025 = sum([f['results']['segment_wise']['k=0.25']['TP'] for f in all_folds_data])
    fp_025 = sum([f['results']['segment_wise']['k=0.25']['FP'] for f in all_folds_data])
    fn_025 = sum([f['results']['segment_wise']['k=0.25']['FN'] for f in all_folds_data])
    f1_025 = [f['results']['segment_wise']['k=0.25']['segmental_f1'] for f in all_folds_data]
    
    tp_05 = sum([f['results']['segment_wise']['k=0.5']['TP'] for f in all_folds_data])
    fp_05 = sum([f['results']['segment_wise']['k=0.5']['FP'] for f in all_folds_data])
    fn_05 = sum([f['results']['segment_wise']['k=0.5']['FN'] for f in all_folds_data])
    f1_05 = [f['results']['segment_wise']['k=0.5']['segmental_f1'] for f in all_folds_data]
    
    # Print table header
    print(f"\n{'Dataset':<12} | {'k = 0.1':<35} | {'k = 0.25':<35} | {'k = 0.5':<35}")
    print(f"{'':12} | {'TP':>6} {'FP':>6} {'FN':>6} {'F1-score':>12} | "
          f"{'TP':>6} {'FP':>6} {'FN':>6} {'F1-score':>12} | "
          f"{'TP':>6} {'FP':>6} {'FN':>6} {'F1-score':>12}")
    print("-" * 125)
    
    # Print table row
    print(f"{'DX-I':<12} | "
          f"{tp_01:>6} {fp_01:>6} {fn_01:>6} {np.mean(f1_01):>12.3f} | "
          f"{tp_025:>6} {fp_025:>6} {fn_025:>6} {np.mean(f1_025):>12.3f} | "
          f"{tp_05:>6} {fp_05:>6} {fn_05:>6} {np.mean(f1_05):>12.3f}")
    
    print("\n" + "="*100)
    print("Note: TP/FP/FN are totals across all folds. F1-score is the mean across folds.")
    print("="*100)

    # Save table to text file for easy copy-paste
    table_text = f"""
Segment-wise Results Table (Paper Format)
==========================================

Dataset | k = 0.1                            | k = 0.25                           | k = 0.5
        | TP     FP     FN     F1-score     | TP     FP     FN     F1-score     | TP     FP     FN     F1-score
{'─'*125}
DX-I    | {tp_01:>6} {fp_01:>6} {fn_01:>6} {np.mean(f1_01):>12.3f} | {tp_025:>6} {fp_025:>6} {fn_025:>6} {np.mean(f1_025):>12.3f} | {tp_05:>6} {fp_05:>6} {fn_05:>6} {np.mean(f1_05):>12.3f}

Note: TP/FP/FN are totals across all {len(all_folds_data)} folds. F1-score is the mean ± std across folds.

Detailed F1-scores:
  k=0.1:  {np.mean(f1_01):.3f} ± {np.std(f1_01):.3f}
  k=0.25: {np.mean(f1_025):.3f} ± {np.std(f1_025):.3f}
  k=0.5:  {np.mean(f1_05):.3f} ± {np.std(f1_05):.3f}
"""
    
    with open(RESULTS_DIR / "paper_table.txt", 'w', encoding='utf-8') as f:
        f.write(table_text)
    
    print(f"\n✓ Table saved to: paper_table.txt")
    
    # Also save LaTeX version for thesis
    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Segment-wise Detection Results on DX-I Dataset at Different IoU Thresholds}}
\\label{{tab:results}}
\\begin{{tabular}}{{l|cccc|cccc|cccc}}
\\hline
\\multirow{{2}}{{*}}{{Dataset}} & \\multicolumn{{4}}{{c|}}{{$k = 0.1$}} & \\multicolumn{{4}}{{c|}}{{$k = 0.25$}} & \\multicolumn{{4}}{{c}}{{$k = 0.5$}} \\\\
& TP & FP & FN & F1-score & TP & FP & FN & F1-score & TP & FP & FN & F1-score \\\\
\\hline
DX-I & {tp_01} & {fp_01} & {fn_01} & {np.mean(f1_01):.3f} & {tp_025} & {fp_025} & {fn_025} & {np.mean(f1_025):.3f} & {tp_05} & {fp_05} & {fn_05} & {np.mean(f1_05):.3f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    
    with open(RESULTS_DIR / "paper_table_latex.tex", 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print(f"✓ LaTeX table saved to: paper_table_latex.tex")
    
    # Also save CSV version for Excel/spreadsheet tools
    with open(RESULTS_DIR / "paper_table.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 
                         'k=0.1_TP', 'k=0.1_FP', 'k=0.1_FN', 'k=0.1_F1',
                         'k=0.25_TP', 'k=0.25_FP', 'k=0.25_FN', 'k=0.25_F1',
                         'k=0.5_TP', 'k=0.5_FP', 'k=0.5_FN', 'k=0.5_F1'])
        writer.writerow(['DX-I',
                         tp_01, fp_01, fn_01, f"{np.mean(f1_01):.3f}",
                         tp_025, fp_025, fn_025, f"{np.mean(f1_025):.3f}",
                         tp_05, fp_05, fn_05, f"{np.mean(f1_05):.3f}"])
    
    print(f"✓ CSV table saved to: paper_table.csv")
    
    # ============================================================================
    # PART 4: CREATE VISUALIZATIONS FOR EACH FOLD
    # ============================================================================
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    for fold_idx, fold_data in enumerate(all_folds_data, 1):
        print(f"\nFold {fold_idx}:")
        
        # Prediction timeline
        plot_prediction_timeline(
            fold_data['predictions'],
            fold_data['targets'],
            title=f"Fold {fold_idx} - Predictions vs Ground Truth",
            save_path=RESULTS_DIR / "plots" / f"fold{fold_idx}_predictions.png"
        )
        plt.close()
        
        # Segment analysis with IoU
        plot_segments_with_iou(
            fold_data['predictions'],
            fold_data['targets'],
            title=f"Fold {fold_idx} - Segment-wise Analysis (IoU k=0.5)",
            iou_threshold=0.5,
            save_path=RESULTS_DIR / "plots" / f"fold{fold_idx}_segments.png"
        )
        plt.close()

    # ============================================================================
    # PART 5: CROSS-FOLD COMPARISON PLOTS
    # ============================================================================
    
    print("\nCreating cross-fold comparison plots...")
    
    # Plot 1: Frame-wise metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Frame-Wise Metrics Across Folds', fontsize=14, fontweight='bold')
    
    folds = list(range(1, len(all_folds_data) + 1))
    
    # Accuracy
    axes[0, 0].bar(folds, frame_accuracies, alpha=0.7, color='steelblue')
    axes[0, 0].axhline(np.mean(frame_accuracies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(frame_accuracies):.4f}')
    axes[0, 0].set_title('Frame Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Precision
    axes[0, 1].bar(folds, class1_precisions, alpha=0.7, color='green')
    axes[0, 1].axhline(np.mean(class1_precisions), color='red', linestyle='--',
                       label=f'Mean: {np.mean(class1_precisions):.4f}')
    axes[0, 1].set_title('Class 1 Precision', fontweight='bold')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Recall
    axes[1, 0].bar(folds, class1_recalls, alpha=0.7, color='orange')
    axes[1, 0].axhline(np.mean(class1_recalls), color='red', linestyle='--',
                       label=f'Mean: {np.mean(class1_recalls):.4f}')
    axes[1, 0].set_title('Class 1 Recall', fontweight='bold')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # F1-score
    axes[1, 1].bar(folds, class1_f1s, alpha=0.7, color='purple')
    axes[1, 1].axhline(np.mean(class1_f1s), color='red', linestyle='--',
                       label=f'Mean: {np.mean(class1_f1s):.4f}')
    axes[1, 1].set_title('Class 1 F1-Score', fontweight='bold')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots" / "framewise_metrics_comparison.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: framewise_metrics_comparison.png")
    plt.close()

    # Plot 2: Segment-wise F1 at different IoU thresholds
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iou_thresholds = [0.1, 0.25, 0.5]
    x = np.arange(len(folds))
    width = 0.25
    
    for idx, iou_thresh in enumerate(iou_thresholds):
        key = f'k={iou_thresh}'
        f1s = [f['results']['segment_wise'][key]['segmental_f1'] for f in all_folds_data]
        offset = width * (idx - 1)
        ax.bar(x + offset, f1s, width, alpha=0.7, label=f'IoU k={iou_thresh}')
    
    ax.set_title('Segmental F1-Score Across Folds at Different IoU Thresholds', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('Segmental F1-Score', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots" / "segmental_f1_comparison.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: segmental_f1_comparison.png")
    plt.close()
    
    # Plot 3: Training curves across folds
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves Across All Folds', fontsize=14, fontweight='bold')
    
    for fold_idx, fold_data in enumerate(all_folds_data, 1):
        history = fold_data['history']
        epochs = list(range(1, len(history['train_loss']) + 1))
        
        # Loss
        axes[0, 0].plot(epochs, history['train_loss'], label=f'Fold {fold_idx}', alpha=0.7)
        axes[0, 1].plot(epochs, history['val_loss'], label=f'Fold {fold_idx}', alpha=0.7)
        
        # Accuracy
        axes[1, 0].plot(epochs, history['train_acc'], label=f'Fold {fold_idx}', alpha=0.7)
        axes[1, 1].plot(epochs, history['val_acc'], label=f'Fold {fold_idx}', alpha=0.7)
    
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Training Accuracy', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Accuracy', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots" / "training_curves.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: training_curves.png")
    plt.close()

    # Plot 4: Summary metrics with error bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Aggregated Metrics with Standard Deviation', fontsize=14, fontweight='bold')
    
    # Frame-wise metrics
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_means = [
        np.mean(frame_accuracies),
        np.mean(class1_precisions),
        np.mean(class1_recalls),
        np.mean(class1_f1s)
    ]
    metrics_stds = [
        np.std(frame_accuracies),
        np.std(class1_precisions),
        np.std(class1_recalls),
        np.std(class1_f1s)
    ]
    
    axes[0].bar(metrics_names, metrics_means, yerr=metrics_stds, alpha=0.7, 
                color=['steelblue', 'green', 'orange', 'purple'], capsize=10)
    axes[0].set_title('Frame-Wise Metrics (Class 1)', fontweight='bold')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(metrics_means, metrics_stds)):
        axes[0].text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
                     ha='center', fontsize=9, fontweight='bold')
    
    # Segment-wise F1 at different IoU thresholds
    seg_f1_means = []
    seg_f1_stds = []
    for iou_thresh in iou_thresholds:
        key = f'k={iou_thresh}'
        f1s = [f['results']['segment_wise'][key]['segmental_f1'] for f in all_folds_data]
        seg_f1_means.append(np.mean(f1s))
        seg_f1_stds.append(np.std(f1s))
    
    iou_labels = [f'k={t}' for t in iou_thresholds]
    axes[1].bar(iou_labels, seg_f1_means, yerr=seg_f1_stds, alpha=0.7,
                color=['lightcoral', 'coral', 'indianred'], capsize=10)
    axes[1].set_title('Segmental F1-Score at Different IoU Thresholds', fontweight='bold')
    axes[1].set_ylabel('Segmental F1-Score')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(seg_f1_means, seg_f1_stds)):
        axes[1].text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
                     ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "plots" / "aggregated_metrics_summary.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: aggregated_metrics_summary.png")
    plt.close()
    
    # ============================================================================
    # PART 6: DETAILED PER-FOLD SUMMARY
    # ============================================================================
    
    print("\n" + "="*70)
    print("PER-FOLD DETAILED SUMMARY")
    print("="*70)
    
    for fold_idx, fold_data in enumerate(all_folds_data, 1):
        results = fold_data['results']
        split = fold_data['split']
        
        print(f"\n{'─'*70}")
        print(f"FOLD {fold_idx}")
        print(f"{'─'*70}")
        
        print(f"\nValidation Users ({len(split['val_user_ids'])} users):")
        print(f"  {', '.join(split['val_user_ids'])}")
        print(f"  Sessions: {', '.join(split['val_sessions'])}")
        
        print(f"\nFrame-Wise:")
        print(f"  Accuracy:  {results['frame_wise']['frame_accuracy']:.4f}")
        print(f"  Precision: {results['frame_wise']['class_1_precision']:.4f}")
        print(f"  Recall:    {results['frame_wise']['class_1_recall']:.4f}")
        print(f"  F1-Score:  {results['frame_wise']['class_1_f1']:.4f}")
        
        print(f"\nSegment-Wise:")
        for iou_thresh in [0.1, 0.25, 0.5]:
            key = f'k={iou_thresh}'
            seg = results['segment_wise'][key]
            print(f"  k={iou_thresh}: TP={seg['TP']}, FP={seg['FP']}, FN={seg['FN']}, "
                  f"F1={seg['segmental_f1']:.4f}")
    
    # ============================================================================
    # PART 7: SAVE SUMMARY REPORT
    # ============================================================================
    
    print("\n" + "="*70)
    print("SAVING SUMMARY REPORT")
    print("="*70)
    
    summary_report = {
        'n_folds': len(all_folds_data),
        'aggregated_metrics': {
            'frame_wise': {
                'accuracy': {'mean': float(np.mean(frame_accuracies)), 
                            'std': float(np.std(frame_accuracies))},
                'class_1_precision': {'mean': float(np.mean(class1_precisions)), 
                                     'std': float(np.std(class1_precisions))},
                'class_1_recall': {'mean': float(np.mean(class1_recalls)), 
                                  'std': float(np.std(class1_recalls))},
                'class_1_f1': {'mean': float(np.mean(class1_f1s)), 
                              'std': float(np.std(class1_f1s))}
            },
            'segment_wise': {}
        },
        'per_fold': []
    }
    
    for iou_thresh in iou_thresholds:
        key = f'k={iou_thresh}'
        f1s = [f['results']['segment_wise'][key]['segmental_f1'] for f in all_folds_data]
        precs = [f['results']['segment_wise'][key]['precision'] for f in all_folds_data]
        recs = [f['results']['segment_wise'][key]['recall'] for f in all_folds_data]
        
        summary_report['aggregated_metrics']['segment_wise'][key] = {
            'precision': {'mean': float(np.mean(precs)), 'std': float(np.std(precs))},
            'recall': {'mean': float(np.mean(recs)), 'std': float(np.std(recs))},
            'segmental_f1': {'mean': float(np.mean(f1s)), 'std': float(np.std(f1s))}
        }
    
    for fold_idx, fold_data in enumerate(all_folds_data, 1):
        summary_report['per_fold'].append({
            'fold': fold_idx,
            'val_users': fold_data['split']['val_user_ids'],
            'metrics': fold_data['results']
        })
    
    with open(RESULTS_DIR / "summary_report.json", 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"  ✓ Saved: summary_report.json")
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nGenerated files:")
    print("  - plots/fold{1-5}_predictions.png")
    print("  - plots/fold{1-5}_segments.png")
    print("  - plots/framewise_metrics_comparison.png")
    print("  - plots/segmental_f1_comparison.png")
    print("  - plots/training_curves.png")
    print("  - plots/aggregated_metrics_summary.png")
    print("  - summary_report.json")
    print("="*70)


if __name__ == '__main__':
    app.run(main)
