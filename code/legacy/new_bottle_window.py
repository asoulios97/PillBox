"""
hybrid_mstcn_complete.py

Complete implementation of Hybrid MS-TCN + Inception for drinking gesture detection.

Features:
✓ Inception-Dilated Temporal Blocks (multi-scale features)
✓ Bottleneck modules (efficiency + regularization)
✓ Multi-stage refinement (MS-TCN)
✓ Sliding window dataset (handles long sequences)
✓ 6-channel IMU input only (ax, ay, az, gx, gy, gz)
✓ Train/Val split with proper evaluation
✓ CUDA memory efficient

Author: [Your Name]
Date: 2025
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle as pkl
from pathlib import Path
from collections import Counter


# ============================================================================
#  PART 1: BOTTLENECK MODULE
# ============================================================================

class BottleneckBlock(nn.Module):
    """
    Bottleneck module για dimension reduction και feature refinement.
    
    Architecture:
        Input [B, C, T]
        ↓
        1×1 Conv: C → C//4 (compression)
        ↓
        BatchNorm + ReLU
        ↓
        1×1 Conv: C//4 → C (expansion)
        ↓
        BatchNorm
        ↓
        + Residual
        ↓
        ReLU + Dropout
        ↓
        Output [B, C, T]
    
    Benefits:
    - Reduces parameters by 75% in bottleneck layers
    - Non-linear feature transformation
    - Gradient flow via residual connection
    """
    
    def __init__(
        self,
        in_channels: int,
        bottleneck_ratio: float = 0.15, #0.25
        dropout: float = 0.5 #0.3
    ):
        super().__init__()
        
        bottleneck_ch = max(int(in_channels * bottleneck_ratio), 8)
        
        self.bottleneck = nn.Sequential(
            # Compression
            nn.Conv1d(in_channels, bottleneck_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(bottleneck_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # Expansion
            nn.Conv1d(bottleneck_ch, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.bottleneck(x)
        out = out + residual  # Residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


# ============================================================================
#  PART 2: INCEPTION-DILATED TEMPORAL BLOCK
# ============================================================================

class InceptionDilatedTemporalBlock(nn.Module):
    """
    Multi-scale temporal convolution με Inception-style parallel branches.
    
    Branches:
    1. 1×1 conv (local, pointwise patterns)
    2. 1×1 → 3×3 dilated → 1×1 (medium-range patterns)
    3. 1×1 → 5×5 dilated → 1×1 (long-range patterns)
    4. Residual skip connection
    
    All branches concatenated και projected πίσω σε out_channels.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [1, 3, 5],
        dilation: int = 1,
        dropout: float = 0.3,
        bottleneck_ratio: float = 0.25
    ):
        super().__init__()
        
        bottleneck_ch = max(int(in_channels * bottleneck_ratio), 8)
        
        self.branches = nn.ModuleList()
        
        for k in kernel_sizes:
            if k == 1:
                # Branch 1: Simple 1×1
                branch = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                # Branches 2,3: Bottleneck → Dilated Conv → Projection
                padding = ((k - 1) * dilation) // 2
                branch = nn.Sequential(
                    nn.Conv1d(in_channels, bottleneck_ch, kernel_size=1, bias=False),
                    nn.BatchNorm1d(bottleneck_ch),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv1d(bottleneck_ch, bottleneck_ch, kernel_size=k,
                             dilation=dilation, padding=padding, bias=False),
                    nn.BatchNorm1d(bottleneck_ch),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv1d(bottleneck_ch, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            self.branches.append(branch)
        
        # Residual
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual = nn.Identity()
        
        # Fusion
        total_channels = out_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Conv1d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Parallel branches
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate
        concat = torch.cat(branch_outputs, dim=1)
        
        # Fusion
        fused = self.fusion(concat)
        
        # Residual
        residual = self.residual(x)
        
        # Combine
        out = self.relu(fused + residual)
        out = self.dropout(out)
        
        return out


# ============================================================================
#  PART 3: MS-TCN STAGE με INCEPTION + BOTTLENECK
# ============================================================================

class HybridMSTCNStage(nn.Module):
    """
    Ένα stage του MS-TCN με:
    - Inception-Dilated blocks στον encoder
    - Inception-Dilated blocks στον decoder
    - Bottleneck modules κάθε N layers
    - Skip connections encoder → decoder
    """
    
    def __init__(
        self,
        num_layers: int = 6,#8, #10,
        num_features: int = 38, # 48, # 64,
        num_classes: int = 2,
        kernel_sizes: List[int] = [1, 3, 5],
        dropout: float = 0.6,
        use_bottleneck: bool = True,
        bottleneck_interval: int = 3
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_bottleneck = use_bottleneck
        
        # Exponential dilation schedule
        self.dilations = [2 ** i for i in range(num_layers)]
        
        # -------------------- ENCODER --------------------
        self.encoder_layers = nn.ModuleList()
        self.encoder_bottlenecks = nn.ModuleList()
        
        for i, dilation in enumerate(self.dilations):
            layer = InceptionDilatedTemporalBlock(
                in_channels=num_features,
                out_channels=num_features,
                kernel_sizes=kernel_sizes,
                dilation=dilation,
                dropout=dropout
            )
            self.encoder_layers.append(layer)
            
            # Add bottleneck every N layers
            if use_bottleneck and (i + 1) % bottleneck_interval == 0:
                bottleneck = BottleneckBlock(num_features, dropout=dropout)
            else:
                bottleneck = nn.Identity()
            self.encoder_bottlenecks.append(bottleneck)
        
        # -------------------- DECODER --------------------
        self.decoder_layers = nn.ModuleList()
        self.decoder_bottlenecks = nn.ModuleList()
        
        for i, dilation in enumerate(reversed(self.dilations)):
            in_ch = 2 * num_features if i > 0 else num_features
            
            layer = InceptionDilatedTemporalBlock(
                in_channels=in_ch,
                out_channels=num_features,
                kernel_sizes=kernel_sizes,
                dilation=dilation,
                dropout=dropout
            )
            self.decoder_layers.append(layer)
            
            if use_bottleneck and (i + 1) % bottleneck_interval == 0:
                bottleneck = BottleneckBlock(num_features, dropout=dropout)
            else:
                bottleneck = nn.Identity()
            self.decoder_bottlenecks.append(bottleneck)
        
        # -------------------- OUTPUT --------------------
        self.output_conv = nn.Conv1d(num_features, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder με bottlenecks
        encoder_features = []
        enc = x
        
        for layer, bottleneck in zip(self.encoder_layers, self.encoder_bottlenecks):
            enc = layer(enc)
            enc = bottleneck(enc)
            encoder_features.append(enc)
        
        # Decoder με skip connections
        dec = encoder_features[-1]
        
        for i, (layer, bottleneck) in enumerate(zip(self.decoder_layers, self.decoder_bottlenecks)):
            if i > 0:
                # Skip connection
                skip_idx = len(encoder_features) - i - 1
                dec = torch.cat([dec, encoder_features[skip_idx]], dim=1)
            
            dec = layer(dec)
            dec = bottleneck(dec)
        
        # Output
        out = self.output_conv(dec)
        return out


# ============================================================================
#  PART 4: COMPLETE HYBRID MS-TCN MODEL
# ============================================================================

class HybridMSTCN(nn.Module):
    """
    Complete Multi-Stage Temporal Convolutional Network με:
    - Inception blocks (multi-scale)
    - Bottleneck modules (efficiency)
    - Multi-stage refinement (accuracy)
    - 6-channel IMU input (ax,ay,az,gx,gy,gz)
    """
    
    def __init__(
        self,
        num_stages: int = 4,
        num_layers: int = 10,
        num_features: int = 64,
        num_classes: int = 2,
        input_channels: int = 6,
        kernel_sizes: List[int] = [1, 3, 5],
        dropout: float = 0.3,
        use_bottleneck: bool = True,
        bottleneck_interval: int = 3
    ):
        super().__init__()
        
        self.num_stages = num_stages
        self.num_classes = num_classes
        
        # Input projection: 6 IMU channels → feature space
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, num_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: Initial prediction
        self.stage1 = HybridMSTCNStage(
            num_layers=num_layers,
            num_features=num_features,
            num_classes=num_classes,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
            use_bottleneck=use_bottleneck,
            bottleneck_interval=bottleneck_interval
        )
        
        # Refinement stages 2-N
        self.refinement_stages = nn.ModuleList()
        
        for _ in range(num_stages - 1):
            # Projection: features + predictions → features
            proj = nn.Sequential(
                nn.Conv1d(num_features + num_classes, num_features, 
                         kernel_size=1, bias=False),
                nn.BatchNorm1d(num_features),
                nn.ReLU(inplace=True)
            )
            
            # Inter-stage bottleneck
            if use_bottleneck:
                inter_bottleneck = BottleneckBlock(num_features, dropout=dropout)
            else:
                inter_bottleneck = nn.Identity()
            
            # Stage
            stage = HybridMSTCNStage(
                num_layers=num_layers,
                num_features=num_features,
                num_classes=num_classes,
                kernel_sizes=kernel_sizes,
                dropout=dropout,
                use_bottleneck=use_bottleneck,
                bottleneck_interval=bottleneck_interval
            )
            
            self.refinement_stages.append(nn.ModuleDict({
                'proj': proj,
                'bottleneck': inter_bottleneck,
                'stage': stage
            }))
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 6, T] - 6-channel IMU input
        Returns:
            List of [B, num_classes, T] predictions (one per stage)
        """
        # Input projection
        features = self.input_proj(x)
        
        # Stage 1
        stage1_out = self.stage1(features)
        outputs = [stage1_out]
        
        # Refinement stages
        prev_pred = stage1_out
        for stage_dict in self.refinement_stages:
            # Combine features + predictions
            combined = torch.cat([features, prev_pred], dim=1)
            
            # Project and refine
            stage_input = stage_dict['proj'](combined)
            stage_input = stage_dict['bottleneck'](stage_input)
            stage_out = stage_dict['stage'](stage_input)
            
            outputs.append(stage_out)
            prev_pred = stage_out
        
        return outputs


# ============================================================================
#  PART 5: LOSS FUNCTION
# ============================================================================

class HybridMSTCNLoss(nn.Module):
    """
    Combined loss: Classification + Temporal Smoothing
    
    L_total = Σ_stages [L_cls + λ_smooth * L_smooth]
    
    L_cls: Cross-entropy για frame-wise classification
    L_smooth: Penalizes rapid transitions between frames
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        class_weights: Optional[torch.Tensor] = None,
        smooth_weight: float = 0.25, #0.15
        ignore_index: int = -100
    ):
        super().__init__()
        
        self.smooth_weight = smooth_weight
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
    
    def temporal_smoothing_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """L_smooth = Σ_t ||p_t - p_{t+1}||^2"""
        probs = F.softmax(predictions, dim=1)  # [B, C, T]
        diff = probs[:, :, 1:] - probs[:, :, :-1]  # Temporal differences
        return torch.mean(diff ** 2)
    
    def forward(
        self,
        stage_outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            stage_outputs: List of [B, num_classes, T]
            targets: [B, T]
        Returns:
            total_loss, loss_dict
        """
        total_loss = 0.0
        loss_dict = {}
        
        for stage_idx, pred in enumerate(stage_outputs):
            cls_loss = self.ce_loss(pred, targets)
            smooth_loss = self.temporal_smoothing_loss(pred)
            stage_loss = cls_loss + self.smooth_weight * smooth_loss
            
            total_loss += stage_loss
            loss_dict[f'stage{stage_idx+1}_cls'] = cls_loss.item()
            loss_dict[f'stage{stage_idx+1}_smooth'] = smooth_loss.item()
        
        total_loss = total_loss / len(stage_outputs)
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# ============================================================================
#  PART 6: SLIDING WINDOW DATASET
# ============================================================================

def create_sliding_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 128,
    stride: int = 64
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split long sequence into fixed-size windows.
    
    Args:
        X: [T, C] features
        y: [T] labels
        window_size: Window length
        stride: Step size
    
    Returns:
        X_windows: [num_windows, window_size, C]
        y_windows: [num_windows, window_size]
        valid_lengths: [num_windows]
    """
    T, C = X.shape
    num_windows = (T - window_size + stride) // stride
    if (T - window_size) % stride != 0:
        num_windows += 1
    
    X_windows = []
    y_windows = []
    valid_lengths = []
    
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        
        if end <= T:
            X_win = X[start:end]
            y_win = y[start:end]
            valid_len = window_size
        else:
            # Pad last window
            actual_len = T - start
            X_win = np.zeros((window_size, C), dtype=X.dtype)
            y_win = np.full((window_size,), -100, dtype=y.dtype)
            
            X_win[:actual_len] = X[start:T]
            y_win[:actual_len] = y[start:T]
            valid_len = actual_len
        
        X_windows.append(X_win)
        y_windows.append(y_win)
        valid_lengths.append(valid_len)
    
    return (np.stack(X_windows, axis=0),
            np.stack(y_windows, axis=0),
            np.array(valid_lengths, dtype=np.int32))


class WindowedDrinkingDataset(Dataset):
    """
    PyTorch Dataset με sliding windows.
    
    Converts variable-length sessions → fixed-size windows.
    Enables batching and efficient CUDA memory usage.
    """
    
    def __init__(
        self,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
        meta_list: List[Dict],
        window_size: int = 128,
        stride: int = 64
    ):
        print(f"Creating windowed dataset (window={window_size}, stride={stride})...")
        
        all_X_windows = []
        all_y_windows = []
        all_valid_lengths = []
        
        for X, y, meta in zip(X_list, y_list, meta_list):
            X_win, y_win, valid_len = create_sliding_windows(X, y, window_size, stride)
            all_X_windows.append(X_win)
            all_y_windows.append(y_win)
            all_valid_lengths.append(valid_len)
        
        self.X_windows = np.concatenate(all_X_windows, axis=0)
        self.y_windows = np.concatenate(all_y_windows, axis=0)
        self.valid_lengths = np.concatenate(all_valid_lengths, axis=0)
        
        print(f"  Sessions: {len(X_list)}")
        print(f"  Total windows: {len(self.X_windows)}")
        print(f"  Window shape: {self.X_windows[0].shape}")
        
        # Label distribution
        all_labels = self.y_windows[self.y_windows != -100]
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"  Label distribution:")
        for label, count in zip(unique, counts):
            print(f"    Class {label}: {count} ({count/len(all_labels)*100:.2f}%)")
    
    def __len__(self) -> int:
        return len(self.X_windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.X_windows[idx]  # [W, C]
        y = self.y_windows[idx]  # [W]
        
        # Transpose για Conv1d: [W, C] → [C, W]
        X = torch.from_numpy(X).float().T
        y = torch.from_numpy(y).long()
        
        return X, y


def create_dataloaders(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    meta_list: List[Dict],
    window_size: int = 128,
    stride: int = 64,
    batch_size: int = 32,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders με session-level split.
    """
    # Session-level split (avoid data leakage)
    num_sessions = len(X_list)
    num_train = int(num_sessions * train_ratio)
    
    indices = np.random.permutation(num_sessions)
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]
    
    print(f"\n=== Data Split ===")
    print(f"Train sessions: {len(train_idx)}")
    print(f"Val sessions: {len(val_idx)}")
    
    # Split data
    X_train = [X_list[i] for i in train_idx]
    y_train = [y_list[i] for i in train_idx]
    meta_train = [meta_list[i] for i in train_idx]
    
    X_val = [X_list[i] for i in val_idx]
    y_val = [y_list[i] for i in val_idx]
    meta_val = [meta_list[i] for i in val_idx]
    
    # Create datasets
    print("\n--- Training Dataset ---")
    train_dataset = WindowedDrinkingDataset(X_train, y_train, meta_train, window_size, stride)
    
    print("\n--- Validation Dataset ---")
    val_dataset = WindowedDrinkingDataset(X_val, y_val, meta_val, window_size, stride)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    return train_loader, val_loader


# ============================================================================
#  PART 7: ADVANCED EVALUATION METRICS
# ============================================================================

def compute_segment_wise_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, int]:
    """
    Compute segment-wise TP, FP, FN για drinking gesture detection.
    
    Args:
        predictions: [T] predicted labels
        targets: [T] ground truth labels
        iou_threshold: Minimum IoU για TP
    
    Returns:
        Dict με TP, FP, FN, precision, recall, f1
    """
    
    def get_segments(labels):
        """Βρίσκει συνεχόμενα segments."""
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
        """IoU για δύο segments."""
        start1, end1 = seg1
        start2, end2 = seg2
        
        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1, end2) - min(start1, start2)
        
        return intersection / union if union > 0 else 0.0
    
    pred_segments = get_segments(predictions)
    gt_segments = get_segments(targets)
    
    matched_gt = set()
    matched_pred = set()
    
    TP = 0
    for i, gt_seg in enumerate(gt_segments):
        best_iou = 0.0
        best_pred_idx = -1
        
        for j, pred_seg in enumerate(pred_segments):
            iou = compute_iou(gt_seg, pred_seg)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = j
        
        if best_iou >= iou_threshold:
            TP += 1
            matched_gt.add(i)
            if best_pred_idx >= 0:
                matched_pred.add(best_pred_idx)
    
    FP = len(pred_segments) - len(matched_pred)
    FN = len(gt_segments) - len(matched_gt)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


@torch.no_grad()
def enhanced_validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Enhanced validation με segment-wise metrics."""
    model.eval()
    
    losses = []
    all_predictions = []
    all_targets = []
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        stage_outputs = model(X_batch)
        loss, _ = criterion(stage_outputs, y_batch)
        
        final_pred = stage_outputs[-1].argmax(dim=1)
        
        mask = y_batch != -100
        pred_valid = final_pred[mask].cpu().numpy()
        y_valid = y_batch[mask].cpu().numpy()
        
        losses.append(loss.item())
        all_predictions.append(pred_valid)
        all_targets.append(y_valid)
    
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    frame_acc = (all_predictions == all_targets).mean()
    
    seg_metrics = compute_segment_wise_metrics(all_predictions, all_targets, iou_threshold=0.5)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, labels=[1], average='binary', zero_division=0
    )
    
    return {
        'val_loss': np.mean(losses),
        'val_acc': float(frame_acc),
        'val_precision': float(precision),
        'val_recall': float(recall),
        'val_f1': float(f1),
        'seg_tp': seg_metrics['TP'],
        'seg_fp': seg_metrics['FP'],
        'seg_fn': seg_metrics['FN'],
        'seg_f1': seg_metrics['f1']
    }


def print_evaluation_summary(results: Dict, iou_thresholds: List[float] = [0.1, 0.25, 0.5]):
    """Print formatted evaluation report."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    fm = results['frame_wise']
    print(f"\n--- FRAME-WISE METRICS ---")
    print(f"Overall Accuracy: {fm['frame_accuracy']:.4f} ({fm['frame_accuracy']*100:.2f}%)")
    print(f"\nClass 1 (Drinking):")
    print(f"  Precision: {fm['class_1_precision']:.4f}")
    print(f"  Recall:    {fm['class_1_recall']:.4f}")
    print(f"  F1-score:  {fm['class_1_f1']:.4f}")
    
    print(f"\n--- SEGMENT-WISE METRICS ---")
    print(f"{'Threshold':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'F1-score':<12}")
    print("-" * 50)
    
    for thresh_name, metrics in results['segment_wise'].items():
        print(f"{thresh_name:<12} "
              f"{metrics['TP']:<6} "
              f"{metrics['FP']:<6} "
              f"{metrics['FN']:<6} "
              f"{metrics['f1']:<12.4f}")
    
    print("="*70)


# ============================================================================
#  PART 8: TRAINING & VALIDATION LOOPS
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Training loop για ένα epoch."""
    model.train()
    
    losses = []
    accs = []
    
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)  # [B, C, W]
        y_batch = y_batch.to(device)  # [B, W]
        
        # Forward
        optimizer.zero_grad()
        stage_outputs = model(X_batch)
        
        # Loss
        loss, _ = criterion(stage_outputs, y_batch)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        final_pred = stage_outputs[-1].argmax(dim=1)
        mask = y_batch != -100
        acc = (final_pred[mask] == y_batch[mask]).float().mean().item()
        
        losses.append(loss.item())
        accs.append(acc)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}: "
                  f"Loss={loss.item():.4f}, Acc={acc:.4f}")
    
    return {
        'train_loss': np.mean(losses),
        'train_acc': np.mean(accs)
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Simple validation (deprecated - use enhanced_validate)."""
    model.eval()
    
    losses = []
    accs = []
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        stage_outputs = model(X_batch)
        loss, _ = criterion(stage_outputs, y_batch)
        
        final_pred = stage_outputs[-1].argmax(dim=1)
        mask = y_batch != -100
        acc = (final_pred[mask] == y_batch[mask]).float().mean().item()
        
        losses.append(loss.item())
        accs.append(acc)
    
    return {
        'val_loss': np.mean(losses),
        'val_acc': np.mean(accs)
    }


@torch.no_grad()
def evaluate_full_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    iou_thresholds: List[float] = [0.1, 0.25, 0.5]
) -> Dict:
    """
    Comprehensive evaluation for thesis.
    Computes frame-wise and segment-wise metrics.
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("\nRunning comprehensive evaluation...")
    
    for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        stage_outputs = model(X_batch)
        final_pred = stage_outputs[-1].argmax(dim=1)
        
        mask = y_batch != -100
        pred_valid = final_pred[mask].cpu().numpy()
        y_valid = y_batch[mask].cpu().numpy()
        
        all_predictions.append(pred_valid)
        all_targets.append(y_valid)
    
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Frame-wise metrics
    frame_accuracy = (all_predictions == all_targets).mean()
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, labels=[0, 1], average=None, zero_division=0
    )
    cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1])
    
    frame_metrics = {
        'frame_accuracy': float(frame_accuracy),
        'class_0_precision': float(precision[0]),
        'class_0_recall': float(recall[0]),
        'class_0_f1': float(f1[0]),
        'class_1_precision': float(precision[1]),
        'class_1_recall': float(recall[1]),
        'class_1_f1': float(f1[1]),
        'confusion_matrix': cm.tolist(),
        'support': support.tolist()
    }
    
    # Segment-wise metrics at multiple thresholds
    segment_metrics = {}
    for iou_thresh in iou_thresholds:
        seg_result = compute_segment_wise_metrics(
            all_predictions, all_targets, iou_threshold=iou_thresh
        )
        segment_metrics[f'k={iou_thresh}'] = seg_result
    
    return {
        'frame_wise': frame_metrics,
        'segment_wise': segment_metrics
    }


# ============================================================================
#  PART 8: MAIN EXECUTION
# ============================================================================

def main():
    """Main training script."""
    
    # ==================== CONFIGURATION ====================
    
    # Paths
    DATA_DIR = r"C:\Users\Apost\Desktop\Διπλωματική\others\open datasets\drinking gesture identifications.open\code"
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    
    # Model hyperparameters
    NUM_STAGES = 4
    NUM_LAYERS = 10
    NUM_FEATURES = 64
    NUM_CLASSES = 2
    INPUT_CHANNELS = 6  # Only IMU (ax,ay,az,gx,gy,gz)
    USE_BOTTLENECK = True
    BOTTLENECK_INTERVAL = 3
    
    # Data hyperparameters
    WINDOW_SIZE = 128    # ~8 seconds at 16Hz
    STRIDE = 64          # 50% overlap
    BATCH_SIZE = 32
    TRAIN_RATIO = 0.8
    
    # Training hyperparameters
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    SMOOTH_WEIGHT = 0.15
    
    # ==================== LOAD DATA ====================
    
    print("Loading data...")
    with open(f"{DATA_DIR}/DX_I_X_16Hz_with_jerk.pkl", 'rb') as f:
        X_list = pkl.load(f)
    with open(f"{DATA_DIR}/DX_I_Y_16Hz_with_jerk.pkl", 'rb') as f:
        y_list = pkl.load(f)
    with open(f"{DATA_DIR}/DX_I_meta_16Hz_with_jerk.pkl", 'rb') as f:
        meta_list = pkl.load(f)
    
    # Extract only first 6 channels (ax,ay,az,gx,gy,gz)
    print(f"Original shape: {X_list[0].shape}")
    X_list = [X[:, :6] for X in X_list]
    print(f"Reduced to 6 IMU channels: {X_list[0].shape}")
    
    print(f"Loaded {len(X_list)} sessions")
    print(f"Sequence lengths: min={min(len(x) for x in X_list)}, "
          f"max={max(len(x) for x in X_list)}, "
          f"mean={np.mean([len(x) for x in X_list]):.0f}")
    
    # ==================== CREATE DATALOADERS ====================
    
    train_loader, val_loader = create_dataloaders(
        X_list, y_list, meta_list,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO
    )
    
    # ==================== INITIALIZE MODEL ====================
    
    print("\n=== Initializing Hybrid MS-TCN ===")
    model = HybridMSTCN(
        num_stages=NUM_STAGES,
        num_layers=NUM_LAYERS,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        input_channels=INPUT_CHANNELS,
        kernel_sizes=[1, 3, 5],
        dropout=0.3,
        use_bottleneck=USE_BOTTLENECK,
        bottleneck_interval=BOTTLENECK_INTERVAL
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==================== LOSS & OPTIMIZER ====================
    
    # Calculate class weights
    all_labels = []
    for _, y_batch in train_loader:
        all_labels.append(y_batch[y_batch != -100].numpy())
    all_labels = np.concatenate(all_labels)
    
    class_counts = np.bincount(all_labels)
    class_weights = torch.tensor(
        [1.0 / c if c > 0 else 0.0 for c in class_counts],
        dtype=torch.float32,
        device=DEVICE
    )
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"\nClass weights: {class_weights}")
    print(f"Class distribution: {class_counts}")
    
    criterion = HybridMSTCNLoss(
        num_classes=NUM_CLASSES,
        class_weights=class_weights,
        smooth_weight=SMOOTH_WEIGHT
    )
    
    optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=5e-4  # L2 regularization to reduce overfitting
    )

    # = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      #
      #  optimizer, mode='min', factor=0.5, patience=5, verbose=True
     #)
    # NEW:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
)
    # ==================== TRAINING LOOP ====================
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    
    # Track metrics for plotting
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    val_f1s_history = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate (με enhanced metrics)
        val_metrics = enhanced_validate(model, val_loader, criterion, DEVICE)
        
        # Store history
        train_losses_history.append(train_metrics['train_loss'])
        val_losses_history.append(val_metrics['val_loss'])
        train_accs_history.append(train_metrics['train_acc'])
        val_accs_history.append(val_metrics['val_acc'])
        val_f1s_history.append(val_metrics['val_f1'])
        
        # Print detailed metrics
        print(f"\nResults:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_acc']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_metrics['val_acc']:.4f}")
        print(f"  Val F1: {val_metrics['val_f1']:.4f}, Precision: {val_metrics['val_precision']:.4f}, Recall: {val_metrics['val_recall']:.4f}")
        print(f"  Segment (k=0.5): TP={val_metrics['seg_tp']}, FP={val_metrics['seg_fp']}, FN={val_metrics['seg_fn']}, F1={val_metrics['seg_f1']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['val_loss'])
        
        # Save best model (based on F1 score - more robust for imbalanced data)
        if val_metrics['val_f1'] > best_val_f1:
            best_val_f1 = val_metrics['val_f1']
            best_val_acc = val_metrics['val_acc']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_f1': best_val_f1,
                'val_loss': val_metrics['val_loss'],
                'val_precision': val_metrics['val_precision'],
                'val_recall': val_metrics['val_recall'],
                'seg_f1': val_metrics['seg_f1'],
                'config': {
                    'num_stages': NUM_STAGES,
                    'num_layers': NUM_LAYERS,
                    'num_features': NUM_FEATURES,
                    'input_channels': INPUT_CHANNELS,
                    'use_bottleneck': USE_BOTTLENECK,
                    'window_size': WINDOW_SIZE
                }
            }, f"{DATA_DIR}/hybrid_mstcn_best.pth")
            print(f"  [SAVED] Best model (val_f1={best_val_f1:.4f}, val_acc={best_val_acc:.4f})")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}\n")
    
    # ==================== FINAL COMPREHENSIVE EVALUATION ====================
    
    print("\n" + "="*70)
    print("LOADING BEST MODEL FOR FINAL EVALUATION")
    print("="*70)
    
    # Load best checkpoint
    checkpoint = torch.load(f"{DATA_DIR}/hybrid_mstcn_best.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")
    
    # Comprehensive evaluation
    results = evaluate_full_metrics(
        model, val_loader, DEVICE,
        iou_thresholds=[0.1, 0.25, 0.5]
    )
    
    # Print formatted report
    print_evaluation_summary(results)
    
    # Save results to JSON
    import json
    with open(f"{DATA_DIR}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Detailed results: evaluation_results.json")
    
    # Save training history
    history = {
        'train_loss': train_losses_history,
        'val_loss': val_losses_history,
        'train_acc': train_accs_history,
        'val_acc': val_accs_history,
        'val_f1': val_f1s_history
    }
    with open(f"{DATA_DIR}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[SAVED] Training history: training_history.json")
    
    print("\n" + "="*70)
    print("ALL EVALUATIONS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()