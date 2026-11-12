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
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle as pkl
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime


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
        smooth_weight: float = 0.75, #0.15
        ignore_index: int = -100,
        truncate_threshold: float = 3.0 #4gt # MS-TCN truncation threshold
    ):
        super().__init__()
        
        self.smooth_weight = smooth_weight
        self.truncate_threshold = truncate_threshold
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
    
    def temporal_smoothing_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        MS-TCN Truncated Log-Probability Smoothing Loss
        
        Original MS-TCN formula:
        L_smooth = Σ_t Σ_c max(0, -log(p_{t,c}) - log(p_{t+1,c}) - τ)
        
        where τ is a truncation threshold (typically 4.0)
        
        Intuition:
        - Penalizes over-segmentation (jittery predictions within same class)
        - Truncation allows legitimate transitions (different classes)
        - Only penalizes when both consecutive frames are confident
        """
        # Get probabilities: [B, C, T]
        probs = F.softmax(predictions, dim=1)
        
        # Clamp probabilities to avoid log(0)
        probs = torch.clamp(probs, min=1e-7, max=1.0)
        
        # Log probabilities: [B, C, T]
        log_probs = torch.log(probs)
        
        # Sum of log probs at consecutive time steps: [B, C, T-1]
        log_prob_t = log_probs[:, :, :-1]   # t
        log_prob_t1 = log_probs[:, :, 1:]   # t+1
        
        # MS-TCN smoothing with truncation
        # -log(p_t) - log(p_t+1) measures "consistency" between frames
        # Truncation: only penalize if above threshold (both frames confident)
        smooth_values = -log_prob_t - log_prob_t1 - self.truncate_threshold
        
        # Apply truncation: max(0, value)
        # Values below threshold → 0 penalty (legitimate transitions)
        # Values above threshold → penalty (unwanted jitter)
        smooth_loss = torch.mean(torch.clamp(smooth_values, min=0.0))
        
        return smooth_loss
    
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
#  PART 6: STRATIFIED K-FOLD AT USER LEVEL
# ============================================================================

def create_stratified_kfold_splits(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    meta_list: List[Dict],
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Create Stratified K-Fold splits at userId level (NOT userId+hand).
    
    CRITICAL: All sessions from the same userId (both Left and Right hands)
    must be in the SAME fold to prevent data leakage.
    
    Stratification is based on the average positive ratio per userId.
    
    Args:
        X_list: List of session features
        y_list: List of session labels  
        meta_list: List of metadata (contains userId and hand info)
        n_splits: Number of folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Group sessions by userId
    user_to_sessions = defaultdict(list)
    
    for idx, meta in enumerate(meta_list):
        user_id = meta.get('user_id', f'session_{idx}')
        user_to_sessions[user_id].append(idx)
    
    unique_users = sorted(user_to_sessions.keys())
    print(f"\n=== Stratified K-Fold Setup (User-Level) ===")
    print(f"Number of folds: {n_splits}")
    print(f"Total sessions: {len(X_list)}")
    print(f"Unique users: {len(unique_users)}")
    print(f"Sessions per user: {[len(user_to_sessions[u]) for u in unique_users]}")
    
    # Calculate average positive ratio per USER (across both hands)
    user_positive_ratios = []
    for user_id in unique_users:
        session_indices = user_to_sessions[user_id]
        # Average positive ratio across all sessions for this user
        total_pos = sum((y_list[i] == 1).sum() for i in session_indices)
        total_frames = sum(len(y_list[i]) for i in session_indices)
        avg_ratio = total_pos / total_frames if total_frames > 0 else 0.0
        user_positive_ratios.append(avg_ratio)
    
    # Convert to stratification bins
    user_positive_ratios = np.array(user_positive_ratios)
    stratify_labels = np.digitize(user_positive_ratios, bins=[0.01, 0.05, 0.10, 0.15, 0.20])
    
    print(f"\nStratification bins (positive ratio per user):")
    for i in range(stratify_labels.max() + 1):
        count = (stratify_labels == i).sum()
        print(f"  Bin {i}: {count} users")
    
    # Create stratified K-fold splits at USER level
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for fold_idx, (train_user_idx, val_user_idx) in enumerate(skf.split(unique_users, stratify_labels)):
        # Get userIds for train and val
        train_users = [unique_users[i] for i in train_user_idx]
        val_users = [unique_users[i] for i in val_user_idx]
        
        # Expand to ALL sessions (both hands) for each user
        train_session_indices = []
        for user_id in train_users:
            train_session_indices.extend(user_to_sessions[user_id])
        
        val_session_indices = []
        for user_id in val_users:
            val_session_indices.extend(user_to_sessions[user_id])
        
        # Sort for consistency
        train_session_indices = sorted(train_session_indices)
        val_session_indices = sorted(val_session_indices)
        
        splits.append((train_session_indices, val_session_indices))
        
        # Verify no overlap
        train_users_set = set(train_users)
        val_users_set = set(val_users)
        overlap = train_users_set & val_users_set
        if overlap:
            raise ValueError(f"CRITICAL ERROR: Users {overlap} appear in both train and val!")
        
        # Print fold statistics
        train_pos = sum((y_list[i] == 1).sum() for i in train_session_indices)
        train_total = sum(len(y_list[i]) for i in train_session_indices)
        val_pos = sum((y_list[i] == 1).sum() for i in val_session_indices)
        val_total = sum(len(y_list[i]) for i in val_session_indices)
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: {len(train_users)} users, {len(train_session_indices)} sessions, {train_total:,} frames, "
              f"positive: {train_pos:,} ({train_pos/train_total*100:.2f}%)")
        print(f"  Val:   {len(val_users)} users, {len(val_session_indices)} sessions, {val_total:,} frames, "
              f"positive: {val_pos:,} ({val_pos/val_total*100:.2f}%)")
        print(f"  Train users: {', '.join(sorted(train_users))}")
        print(f"  Val users:   {', '.join(sorted(val_users))}")
    
    return splits


def compute_class_weights(y_list: List[np.ndarray], indices: List[int]) -> torch.Tensor:
    """
    Compute class weights from training data only.
    
    Uses inverse frequency: weight = n_samples / (n_classes * n_samples_per_class)
    
    Args:
        y_list: List of all label arrays
        indices: Indices of training sessions
        
    Returns:
        Tensor of class weights [class_0_weight, class_1_weight]
    """
    # Concatenate labels from training sessions only
    train_labels = np.concatenate([y_list[i] for i in indices])
    
    # Count samples per class
    n_samples = len(train_labels)
    n_class_0 = (train_labels == 0).sum()
    n_class_1 = (train_labels == 1).sum()
    
    # Inverse frequency weights
    weight_0 = n_samples / (2 * n_class_0) if n_class_0 > 0 else 1.0
    weight_1 = n_samples / (2 * n_class_1) if n_class_1 > 0 else 1.0
    
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32)
    
    print(f"\n  Class weights (from training data):")
    print(f"    Class 0: {weight_0:.4f} ({n_class_0:,} samples)")
    print(f"    Class 1: {weight_1:.4f} ({n_class_1:,} samples)")
    print(f"    Ratio (1/0): {weight_1/weight_0:.2f}x")
    
    return class_weights


# ============================================================================
#  PART 7: SLIDING WINDOW DATASET
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
    train_indices: List[int],
    val_indices: List[int],
    window_size: int = 128,
    stride: int = 64,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Create train/val dataloaders from specific indices (for K-Fold CV).
    
    Args:
        X_list: Full list of session features
        y_list: Full list of session labels
        meta_list: Full list of metadata
        train_indices: Indices for training sessions
        val_indices: Indices for validation sessions
        window_size: Window size for sliding windows
        stride: Stride for sliding windows
        batch_size: Batch size for dataloaders
        
    Returns:
        train_loader, val_loader, (X_val, y_val)
    """
    print(f"\n=== Data Split ===")
    print(f"Train sessions: {len(train_indices)}")
    print(f"Val sessions: {len(val_indices)}")
    
    # Split data using provided indices
    X_train = [X_list[i] for i in train_indices]
    y_train = [y_list[i] for i in train_indices]
    meta_train = [meta_list[i] for i in train_indices]
    
    X_val = [X_list[i] for i in val_indices]
    y_val = [y_list[i] for i in val_indices]
    meta_val = [meta_list[i] for i in val_indices]
    
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
    
    return train_loader, val_loader, (X_val, y_val)


# ============================================================================
#  PART 7: OVERLAP-MERGE RECONSTRUCTION FOR VALIDATION
# ============================================================================

def reconstruct_from_windows(
    windowed_predictions: List[np.ndarray],
    windowed_targets: List[np.ndarray],
    windowed_valid_lengths: List[int],
    window_size: int,
    stride: int,
    original_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct continuous timeline from overlapping windows using voting.
    
    This solves the "false cuts" problem by properly merging overlapping regions.
    
    Args:
        windowed_predictions: List of prediction arrays [window_size]
        windowed_targets: List of target arrays [window_size]
        windowed_valid_lengths: Valid length for each window
        window_size: Size of each window
        stride: Stride between windows
        original_length: Original sequence length
        
    Returns:
        reconstructed_preds: [original_length] continuous predictions
        reconstructed_targets: [original_length] continuous targets
    """
    # Initialize vote accumulators
    pred_votes = np.zeros((original_length, 2), dtype=np.float32)  # [T, 2 classes]
    target_votes = np.zeros(original_length, dtype=np.float32)
    vote_counts = np.zeros(original_length, dtype=np.int32)
    
    for window_idx, (pred, target, valid_len) in enumerate(
        zip(windowed_predictions, windowed_targets, windowed_valid_lengths)
    ):
        # Calculate window position in original timeline
        start = window_idx * stride
        end = min(start + valid_len, original_length)
        actual_len = end - start
        
        # Accumulate votes for predictions (class probabilities)
        # pred is [window_size] with class indices, convert to one-hot
        for i in range(actual_len):
            if target[i] != -100:  # Skip padding
                pred_votes[start + i, pred[i]] += 1.0
                target_votes[start + i] = target[i]
                vote_counts[start + i] += 1
    
    # Majority voting for final predictions
    reconstructed_preds = np.argmax(pred_votes, axis=1)
    
    # For targets, use the most common value (should be consistent)
    reconstructed_targets = target_votes.astype(np.int64)
    
    # Only keep frames that were actually predicted
    valid_mask = vote_counts > 0
    
    return reconstructed_preds[valid_mask], reconstructed_targets[valid_mask]


def reconstruct_per_session(
    model: nn.Module,
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    window_size: int,
    stride: int,
    device: torch.device
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Process each session completely and reconstruct continuous predictions.
    
    This properly handles overlapping windows for each session independently.
    
    Returns:
        all_predictions_per_session: List of continuous prediction arrays
        all_targets_per_session: List of continuous target arrays
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    print(f"Reconstructing {len(X_list)} sessions with overlap-merge...")
    
    # Batch size for efficient GPU processing
    batch_size = 64
    
    for session_idx, (X_session, y_session) in enumerate(zip(X_list, y_list)):
        print(f"  Session {session_idx+1}/{len(X_list)}: ", end='', flush=True)
        
        # Create windows for this session
        X_windows, y_windows, valid_lengths = create_sliding_windows(
            X_session, y_session, window_size, stride
        )
        
        num_windows = len(X_windows)
        print(f"{num_windows} windows...", end='', flush=True)
        
        # Predict on all windows in batches
        window_preds = []
        window_targets = []
        window_valids = []
        
        # Process windows in batches for efficiency
        for batch_start in range(0, num_windows, batch_size):
            batch_end = min(batch_start + batch_size, num_windows)
            
            # Prepare batch: [batch_size, window_size, channels]
            batch_X = X_windows[batch_start:batch_end]  # [B, 128, 6]
            batch_y = y_windows[batch_start:batch_end]
            batch_valid = valid_lengths[batch_start:batch_end]
            
            # Convert to tensor and transpose: [B, 128, 6] -> [B, 6, 128]
            X_batch = torch.from_numpy(batch_X).float()
            X_batch = X_batch.transpose(1, 2).to(device)  # [B, 6, 128]
            
            with torch.no_grad():
                stage_outputs = model(X_batch)
                batch_preds = stage_outputs[-1].argmax(dim=1).cpu().numpy()  # [B, 128]
            
            # Store predictions
            for i in range(len(batch_preds)):
                window_preds.append(batch_preds[i])
                window_targets.append(batch_y[i])
                window_valids.append(batch_valid[i])
            
            # Progress indicator
            if (batch_end % 500 == 0) or (batch_end == num_windows):
                print(f"\r  Session {session_idx+1}/{len(X_list)}: {batch_end}/{num_windows} windows...", 
                      end='', flush=True)
        
        print(" ✓", flush=True)
        
        # Reconstruct continuous timeline with voting
        session_pred, session_target = reconstruct_from_windows(
            window_preds,
            window_targets,
            window_valids,
            window_size,
            stride,
            len(X_session)
        )
        
        all_predictions.append(session_pred)
        all_targets.append(session_target)
    
    print(f"\n  ✓ All {len(X_list)} sessions reconstructed successfully!")
    return all_predictions, all_targets


# ============================================================================
#  PART 8: ADVANCED SEGMENT-WISE EVALUATION METRICS
# ============================================================================

def compute_segment_wise_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, int]:
    """
    Segment-wise evaluation following the paper's schema.
    
    Paper's Evaluation Logic:
    1. Calculate IoU for each predicted segment vs ground truth
    2. If IoU >= k (threshold): True Positive (TP)
    3. If IoU < k:
       - If GT length < Pred length → False Positive (over-segmentation)
       - If GT length > Pred length → False Negative (under-segmentation)
    4. Handle many-to-one cases:
       - Multiple predictions → 1 GT: Only 1 TP, rest are FP
       - 1 prediction → Multiple GTs: Only 1 TP, rest are FN
    
    Args:
        predictions: [T] predicted labels (0 or 1)
        targets: [T] ground truth labels (0 or 1)
        iou_threshold: IoU threshold k (0.1, 0.25, or 0.5)
    
    Returns:
        Dict with TP, FP, FN, precision, recall, segmental_f1
    """
    
    def get_segments(labels):
        """Find continuous segments of class=1."""
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
        """Intersection over Union for two segments."""
        start1, end1 = seg1
        start2, end2 = seg2
        
        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1, end2) - min(start1, start2)
        
        return intersection / union if union > 0 else 0.0
    
    def segment_length(seg):
        """Length of a segment."""
        return seg[1] - seg[0]
    
    # Extract segments
    pred_segments = get_segments(predictions)
    gt_segments = get_segments(targets)
    
    # Special case: no gestures
    if len(gt_segments) == 0 and len(pred_segments) == 0:
        return {'TP': 0, 'FP': 0, 'FN': 0, 'precision': 1.0, 'recall': 1.0, 'segmental_f1': 1.0}
    if len(gt_segments) == 0:
        return {'TP': 0, 'FP': len(pred_segments), 'FN': 0, 'precision': 0.0, 'recall': 1.0, 'segmental_f1': 0.0}
    if len(pred_segments) == 0:
        return {'TP': 0, 'FP': 0, 'FN': len(gt_segments), 'precision': 1.0, 'recall': 0.0, 'segmental_f1': 0.0}
    
    # Build IoU matrix: [num_gt, num_pred]
    iou_matrix = np.zeros((len(gt_segments), len(pred_segments)))
    for i, gt_seg in enumerate(gt_segments):
        for j, pred_seg in enumerate(pred_segments):
            iou_matrix[i, j] = compute_iou(gt_seg, pred_seg)
    
    # Track matches
    gt_matched = set()
    pred_matched = set()
    TP = 0
    FP = 0
    FN = 0
    
    # Paper's logic: Process each GT segment
    for i, gt_seg in enumerate(gt_segments):
        # Find all predictions that overlap with this GT
        overlapping_preds = [(j, iou_matrix[i, j]) for j in range(len(pred_segments)) 
                              if iou_matrix[i, j] > 0]
        
        if not overlapping_preds:
            # No prediction overlaps this GT → False Negative
            FN += 1
            continue
        
        # Find best matching prediction
        best_pred_idx, best_iou = max(overlapping_preds, key=lambda x: x[1])
        
        if best_iou >= iou_threshold:
            # Good match → True Positive
            if best_pred_idx not in pred_matched:
                TP += 1
                gt_matched.add(i)
                pred_matched.add(best_pred_idx)
            else:
                # Multiple GTs matched to same prediction → FN for this GT
                FN += 1
        else:
            # IoU below threshold → classify by length
            gt_len = segment_length(gt_seg)
            pred_len = segment_length(pred_segments[best_pred_idx])
            
            if pred_len > gt_len:
                # Over-segmentation → False Positive
                FP += 1
            else:
                # Under-segmentation → False Negative
                FN += 1
    
    # Handle unmatched predictions (multiple predictions → 1 GT case)
    for j in range(len(pred_segments)):
        if j not in pred_matched:
            # This prediction didn't match any GT → False Positive
            FP += 1
    
    # Calculate metrics using paper's formula: F1 = 2*TP / (2*TP + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    segmental_f1 = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    
    return {
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'precision': float(precision),
        'recall': float(recall),
        'segmental_f1': float(segmental_f1)
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
    
    # Compute segment-wise metrics at all three thresholds
    seg_metrics_01 = compute_segment_wise_metrics(all_predictions, all_targets, iou_threshold=0.1)
    seg_metrics_025 = compute_segment_wise_metrics(all_predictions, all_targets, iou_threshold=0.25)
    seg_metrics_05 = compute_segment_wise_metrics(all_predictions, all_targets, iou_threshold=0.5)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, labels=[1], average='binary', zero_division=0
    )
    
    return {
        'val_loss': np.mean(losses),
        'val_acc': float(frame_acc),
        'val_precision': float(precision),
        'val_recall': float(recall),
        'val_f1': float(f1),
        # k=0.1
        'seg_tp_01': seg_metrics_01['TP'],
        'seg_fp_01': seg_metrics_01['FP'],
        'seg_fn_01': seg_metrics_01['FN'],
        'seg_f1_01': seg_metrics_01['segmental_f1'],
        # k=0.25
        'seg_tp_025': seg_metrics_025['TP'],
        'seg_fp_025': seg_metrics_025['FP'],
        'seg_fn_025': seg_metrics_025['FN'],
        'seg_f1_025': seg_metrics_025['segmental_f1'],
        # k=0.5
        'seg_tp': seg_metrics_05['TP'],
        'seg_fp': seg_metrics_05['FP'],
        'seg_fn': seg_metrics_05['FN'],
        'seg_f1': seg_metrics_05['segmental_f1']
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
              f"{metrics['segmental_f1']:<12.4f}")
    
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
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    window_size: int,
    stride: int,
    device: torch.device,
    iou_thresholds: List[float] = [0.1, 0.25, 0.5]
) -> Dict:
    """
    Comprehensive evaluation with proper overlap-merge reconstruction.
    
    This fixes the "false cuts" problem by:
    1. Processing each session completely
    2. Reconstructing continuous predictions with voting
    3. Computing metrics on the reconstructed timeline
    
    Args:
        model: Trained model
        X_list: List of session features
        y_list: List of session labels
        window_size: Window size used during training
        stride: Stride used during training
        device: torch device
        iou_thresholds: List of IoU thresholds for segment-wise evaluation
    
    Returns:
        Dict with frame_wise and segment_wise metrics
    """
    model.eval()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION WITH OVERLAP-MERGE")
    print("="*60)
    
    # Reconstruct continuous predictions per session
    all_pred_sessions, all_target_sessions = reconstruct_per_session(
        model, X_list, y_list, window_size, stride, device
    )
    
    # Concatenate all sessions for frame-wise metrics
    all_predictions = np.concatenate(all_pred_sessions)
    all_targets = np.concatenate(all_target_sessions)
    
    print(f"\n  Total frames evaluated: {len(all_predictions):,}")
    print(f"  Positive frames (anno=1): {(all_targets == 1).sum():,} ({(all_targets == 1).mean()*100:.2f}%)")
    
    # Frame-wise metrics
    print("\n  Computing frame-wise metrics...")
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
    print("\n  Computing segment-wise metrics (paper's schema)...")
    segment_metrics = {}
    
    for iou_thresh in iou_thresholds:
        print(f"    Threshold k={iou_thresh}...")
        seg_result = compute_segment_wise_metrics(
            all_predictions, all_targets, iou_threshold=iou_thresh
        )
        segment_metrics[f'k={iou_thresh}'] = seg_result
    
    print("\n" + "="*60)
    
    return {
        'frame_wise': frame_metrics,
        'segment_wise': segment_metrics,
        'num_sessions': len(X_list),
        'total_frames': len(all_predictions),
        'predictions': all_pred_sessions,  # Save per-session predictions
        'targets': all_target_sessions      # Save per-session targets
    }


# ============================================================================
#  PART 8: MAIN EXECUTION
# ============================================================================

def main():
    """Main training script."""
    
    # ==================== CONFIGURATION ====================
    
    # Paths
    DATA_DIR = r"C:\Users\Apost\Desktop\Διπλωματική\others\open datasets\drinking gesture identifications.open\code"
    
    # Create experiment folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = f"experiment_{timestamp}"
    RESULTS_DIR = Path(DATA_DIR) / "results" / experiment_name
    
    # Create results folder structure
    (RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "splits").mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"{'='*60}")
    
    # Save experiment configuration
    experiment_config = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'model_config': {
            'num_stages': 4,
            'num_layers': 10,
            'num_features': 64,
            'num_classes': 2,
            'input_channels': 6,
            'use_bottleneck': True,
            'bottleneck_interval': 3
        },
        'data_config': {
            'window_size': 128,
            'stride': 64,
            'batch_size': 32
        },
        'training_config': {
            'n_folds': 5,
            'num_epochs': 50,
            'learning_rate': 1e-3,
            'smooth_weight': 0.75,
            'truncate_threshold': 4.0
        }
    }
    
    import json
    with open(RESULTS_DIR / "experiment_config.json", 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
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
    BATCH_SIZE = 128
    
    # K-Fold Cross-Validation
    N_FOLDS = 5
    RANDOM_STATE = 42
    
    # Training hyperparameters
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-4
    SMOOTH_WEIGHT = 0.75
    
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
    
    print(f"\nLoaded {len(X_list)} sessions")
    print(f"Sequence lengths: min={min(len(x) for x in X_list)}, "
          f"max={max(len(x) for x in X_list)}, "
          f"mean={np.mean([len(x) for x in X_list]):.0f}")
    
    # Display meta_list information
    print("\n" + "="*70)
    print("META_LIST INFORMATION")
    print("="*70)
    print(f"\nSample meta entry (first session):")
    print(f"  Keys: {list(meta_list[0].keys())}")
    for key, value in meta_list[0].items():
        if key not in ['zscore_stats', 'features']:  # Skip large nested data
            print(f"  {key}: {value}")
    
    print(f"\nAll sessions:")
    for idx, meta in enumerate(meta_list):
        user_id = meta.get('user_id', f'session_{idx}')
        hand = meta.get('hand', 'unknown')
        pos_samples = (y_list[idx] == 1).sum()
        total_samples = len(y_list[idx])
        print(f"  [{idx}] {user_id:8s} {hand:5s} - {len(X_list[idx]):7,} frames, "
              f"anno=1: {pos_samples:5,} ({pos_samples/total_samples*100:5.2f}%)")
    print("="*70)
    
    # ==================== CREATE STRATIFIED K-FOLD SPLITS ====================
    
    kfold_splits = create_stratified_kfold_splits(
        X_list, y_list, meta_list,
        n_splits=N_FOLDS,
        random_state=RANDOM_STATE
    )
    
    # Store results for each fold
    all_fold_results = []
    all_fold_splits = []  # Track userId/hand splits for each fold
    
    # ==================== K-FOLD CROSS-VALIDATION LOOP ====================
    
    for fold_idx, (train_indices, val_indices) in enumerate(kfold_splits):
        print(f"\n\n")
        print(f"{'='*70}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'='*70}")
        
        # Display userId/hand for this fold
        print(f"\n--- Training Set ({len(train_indices)} sessions) ---")
        train_users = []
        train_user_ids = set()
        for idx in train_indices:
            user_id = meta_list[idx].get('user_id', f'session_{idx}')
            hand = meta_list[idx].get('hand', 'unknown')
            train_users.append(f"{user_id}/{hand}")
            train_user_ids.add(user_id)
        print(f"  Unique users: {len(train_user_ids)} - {', '.join(sorted(train_user_ids))}")
        print(f"  Sessions: {', '.join(sorted(train_users))}")
        
        print(f"\n--- Validation Set ({len(val_indices)} sessions) ---")
        val_users = []
        val_user_ids = set()
        for idx in val_indices:
            user_id = meta_list[idx].get('user_id', f'session_{idx}')
            hand = meta_list[idx].get('hand', 'unknown')
            val_users.append(f"{user_id}/{hand}")
            val_user_ids.add(user_id)
            print(f"  [{idx}] {user_id}/{hand}")
        print(f"  Unique users: {len(val_user_ids)} - {', '.join(sorted(val_user_ids))}")
        
        # Verify no user overlap
        user_overlap = train_user_ids & val_user_ids
        if user_overlap:
            raise ValueError(f"CRITICAL ERROR: Users {user_overlap} in BOTH train and val!")
        else:
            print(f"  ✓ No user overlap verified!")
        
        # Save fold split information
        fold_split_info = {
            'fold': fold_idx + 1,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'train_user_ids': sorted(list(train_user_ids)),
            'val_user_ids': sorted(list(val_user_ids)),
            'train_sessions': sorted(train_users),
            'val_sessions': sorted(val_users),
            'num_train_users': len(train_user_ids),
            'num_val_users': len(val_user_ids),
            'no_overlap_verified': len(user_overlap) == 0
        }
        
        # Create dataloaders for this fold
        train_loader, val_loader, (X_val, y_val) = create_dataloaders(
            X_list, y_list, meta_list,
            train_indices=train_indices,
            val_indices=val_indices,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            batch_size=BATCH_SIZE
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
        
        # Calculate class weights from TRAINING DATA ONLY
        class_weights = compute_class_weights(y_list, train_indices).to(DEVICE)
        
        criterion = HybridMSTCNLoss(
            num_classes=NUM_CLASSES,
            class_weights=class_weights,
            smooth_weight=SMOOTH_WEIGHT
        )
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=LEARNING_RATE,
            weight_decay=5e-4  # L2 regularization
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        # ==================== TRAINING LOOP ====================
        
        print(f"\n{'='*60}")
        print(f"Starting Training - Fold {fold_idx + 1}")
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
            print(f"  Segment (k=0.1):  TP={val_metrics['seg_tp_01']:3d}, FP={val_metrics['seg_fp_01']:4d}, FN={val_metrics['seg_fn_01']:3d}, F1={val_metrics['seg_f1_01']:.4f}")
            print(f"  Segment (k=0.25): TP={val_metrics['seg_tp_025']:3d}, FP={val_metrics['seg_fp_025']:4d}, FN={val_metrics['seg_fn_025']:3d}, F1={val_metrics['seg_f1_025']:.4f}")
            print(f"  Segment (k=0.5):  TP={val_metrics['seg_tp']:3d}, FP={val_metrics['seg_fp']:4d}, FN={val_metrics['seg_fn']:3d}, F1={val_metrics['seg_f1']:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_metrics['val_loss'])
            
            # Save best model for this fold
            if val_metrics['val_f1'] > best_val_f1:
                best_val_f1 = val_metrics['val_f1']
                best_val_acc = val_metrics['val_acc']
                torch.save({
                    'epoch': epoch + 1,
                    'fold': fold_idx + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'val_f1': best_val_f1,
                    'val_loss': val_metrics['val_loss'],
                    'val_precision': val_metrics['val_precision'],
                    'val_recall': val_metrics['val_recall'],
                    # Segment-wise metrics for all thresholds
                    'seg_f1_01': val_metrics['seg_f1_01'],
                    'seg_f1_025': val_metrics['seg_f1_025'],
                    'seg_f1': val_metrics['seg_f1'],
                    'config': {
                        'num_stages': NUM_STAGES,
                        'num_layers': NUM_LAYERS,
                        'num_features': NUM_FEATURES,
                        'input_channels': INPUT_CHANNELS,
                        'use_bottleneck': USE_BOTTLENECK,
                        'window_size': WINDOW_SIZE
                    }
                }, RESULTS_DIR / "models" / f"fold{fold_idx+1}_best.pth")
                print(f"  [SAVED] Best model for Fold {fold_idx+1} (val_f1={best_val_f1:.4f}, seg_f1_05={val_metrics['seg_f1']:.4f})")
        
        # ==================== FOLD EVALUATION ====================
        
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1} Training Complete!")
        print(f"Best Val F1: {best_val_f1:.4f}")
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
        print(f"{'='*60}\n")
        
        # Load best model for this fold and evaluate
        print(f"\nLoading best model for Fold {fold_idx + 1}...")
        checkpoint = torch.load(RESULTS_DIR / "models" / f"fold{fold_idx+1}_best.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Comprehensive evaluation with overlap-merge reconstruction
        fold_results = evaluate_full_metrics(
            model, X_val, y_val,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            device=DEVICE,
            iou_thresholds=[0.1, 0.25, 0.5]
        )
        
        # Add fold number and best metrics
        fold_results['fold'] = fold_idx + 1
        fold_results['best_val_f1'] = best_val_f1
        fold_results['best_val_acc'] = best_val_acc
        
        # Print fold summary
        print_evaluation_summary(fold_results)
        
        # Save predictions separately (can be large)
        predictions_data = {
            'fold': fold_idx + 1,
            'predictions': [pred.tolist() for pred in fold_results['predictions']],
            'targets': [tgt.tolist() for tgt in fold_results['targets']],
            'val_user_ids': fold_split_info['val_user_ids'],
            'val_sessions': fold_split_info['val_sessions']
        }
        with open(RESULTS_DIR / "predictions" / f"fold{fold_idx+1}_predictions.json", 'w') as f:
            json.dump(predictions_data, f, indent=2)
        
        # Remove predictions from fold_results before saving (to keep metrics file small)
        fold_results_to_save = fold_results.copy()
        fold_results_to_save.pop('predictions', None)
        fold_results_to_save.pop('targets', None)
        
        # Store fold results and split info
        all_fold_results.append(fold_results)
        all_fold_splits.append(fold_split_info)
        
        # Save fold-specific results
        import json
        with open(RESULTS_DIR / "metrics" / f"fold{fold_idx+1}_results.json", 'w') as f:
            json.dump(fold_results_to_save, f, indent=2)
        
        # Save fold training history
        fold_history = {
            'fold': fold_idx + 1,
            'train_loss': train_losses_history,
            'val_loss': val_losses_history,
            'train_acc': train_accs_history,
            'val_acc': val_accs_history,
            'val_f1': val_f1s_history,
            'best_val_f1': best_val_f1,
            'best_val_acc': best_val_acc
        }
        with open(RESULTS_DIR / "metrics" / f"fold{fold_idx+1}_history.json", 'w') as f:
            json.dump(fold_history, f, indent=2)
        
        # Save fold split information (userId/hand breakdown)
        with open(RESULTS_DIR / "splits" / f"fold{fold_idx+1}_split_info.json", 'w') as f:
            json.dump(fold_split_info, f, indent=2)
        
        print(f"\n[SAVED] Fold {fold_idx + 1} - model, results, history, split info, and predictions")
    
    # ==================== AGGREGATE K-FOLD RESULTS ====================
    
    print("\n\n")
    print("="*70)
    print("K-FOLD CROSS-VALIDATION COMPLETE!")
    print("="*70)
    
    print(f"\nAggregating results across {N_FOLDS} folds...")
    
    # Extract metrics from all folds
    frame_accuracies = [r['frame_wise']['frame_accuracy'] for r in all_fold_results]
    class1_precisions = [r['frame_wise']['class_1_precision'] for r in all_fold_results]
    class1_recalls = [r['frame_wise']['class_1_recall'] for r in all_fold_results]
    class1_f1s = [r['frame_wise']['class_1_f1'] for r in all_fold_results]
    
    # Segment-wise metrics for each threshold
    seg_metrics_by_threshold = {}
    for iou_thresh in [0.1, 0.25, 0.5]:
        key = f'k={iou_thresh}'
        seg_metrics_by_threshold[key] = {
            'f1s': [r['segment_wise'][key]['segmental_f1'] for r in all_fold_results],
            'precisions': [r['segment_wise'][key]['precision'] for r in all_fold_results],
            'recalls': [r['segment_wise'][key]['recall'] for r in all_fold_results]
        }
    
    # Print aggregated results
    print("\n" + "="*70)
    print("AGGREGATED RESULTS (Mean ± Std)")
    print("="*70)
    
    print("\n--- Frame-Wise Metrics ---")
    print(f"Accuracy:          {np.mean(frame_accuracies):.4f} ± {np.std(frame_accuracies):.4f}")
    print(f"\nClass 1 (Drinking):")
    print(f"  Precision:       {np.mean(class1_precisions):.4f} ± {np.std(class1_precisions):.4f}")
    print(f"  Recall:          {np.mean(class1_recalls):.4f} ± {np.std(class1_recalls):.4f}")
    print(f"  F1-score:        {np.mean(class1_f1s):.4f} ± {np.std(class1_f1s):.4f}")
    
    print(f"\n--- Segment-Wise Metrics (Segmental F1) ---")
    for iou_thresh in [0.1, 0.25, 0.5]:
        key = f'k={iou_thresh}'
        f1s = seg_metrics_by_threshold[key]['f1s']
        precs = seg_metrics_by_threshold[key]['precisions']
        recs = seg_metrics_by_threshold[key]['recalls']
        print(f"\nThreshold k={iou_thresh}:")
        print(f"  Precision:       {np.mean(precs):.4f} ± {np.std(precs):.4f}")
        print(f"  Recall:          {np.mean(recs):.4f} ± {np.std(recs):.4f}")
        print(f"  Segmental F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    print("\n--- Per-Fold Summary ---")
    for idx, r in enumerate(all_fold_results):
        print(f"\nFold {r['fold']}:")
        print(f"  Frame Acc: {r['frame_wise']['frame_accuracy']:.4f}")
        print(f"  Class 1 F1: {r['frame_wise']['class_1_f1']:.4f}")
        print(f"  Seg F1 (k=0.5): {r['segment_wise']['k=0.5']['segmental_f1']:.4f}")
    
    # Prepare aggregated results (without large prediction arrays)
    all_fold_results_clean = []
    for r in all_fold_results:
        r_clean = r.copy()
        r_clean.pop('predictions', None)
        r_clean.pop('targets', None)
        all_fold_results_clean.append(r_clean)
    
    # Save aggregated results
    aggregated_results = {
        'n_folds': N_FOLDS,
        'fold_splits': all_fold_splits,  # Include userId/hand for each fold
        'frame_wise': {
            'accuracy_mean': float(np.mean(frame_accuracies)),
            'accuracy_std': float(np.std(frame_accuracies)),
            'class_1_precision_mean': float(np.mean(class1_precisions)),
            'class_1_precision_std': float(np.std(class1_precisions)),
            'class_1_recall_mean': float(np.mean(class1_recalls)),
            'class_1_recall_std': float(np.std(class1_recalls)),
            'class_1_f1_mean': float(np.mean(class1_f1s)),
            'class_1_f1_std': float(np.std(class1_f1s))
        },
        'segment_wise': {},
        'per_fold_results': all_fold_results_clean
    }
    
    for iou_thresh in [0.1, 0.25, 0.5]:
        key = f'k={iou_thresh}'
        f1s = seg_metrics_by_threshold[key]['f1s']
        precs = seg_metrics_by_threshold[key]['precisions']
        recs = seg_metrics_by_threshold[key]['recalls']
        aggregated_results['segment_wise'][key] = {
            'precision_mean': float(np.mean(precs)),
            'precision_std': float(np.std(precs)),
            'recall_mean': float(np.mean(recs)),
            'recall_std': float(np.std(recs)),
            'segmental_f1_mean': float(np.mean(f1s)),
            'segmental_f1_std': float(np.std(f1s))
        }
    
    import json
    with open(RESULTS_DIR / "kfold_aggregated_results.json", 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    print(f"\n[SAVED] Aggregated results: {RESULTS_DIR / 'kfold_aggregated_results.json'}")
    
    # Print fold splits summary
    print("\n" + "="*70)
    print("FOLD SPLITS SUMMARY (User-Level)")
    print("="*70)
    for fold_info in all_fold_splits:
        print(f"\nFold {fold_info['fold']}:")
        print(f"  Training: {fold_info['num_train_users']} users, {len(fold_info['train_sessions'])} sessions")
        print(f"    Users: {', '.join(fold_info['train_user_ids'])}")
        print(f"  Validation: {fold_info['num_val_users']} users, {len(fold_info['val_sessions'])} sessions")
        print(f"    Users: {', '.join(fold_info['val_user_ids'])}")
        print(f"  ✓ No overlap: {fold_info['no_overlap_verified']}")
    
    print("\n" + "="*70)
    print("ALL K-FOLD EVALUATIONS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()