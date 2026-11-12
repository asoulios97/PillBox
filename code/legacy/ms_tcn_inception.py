"""
hybrid_mstcn_inception.py

Hybrid MS-TCN + Inception architecture for drinking gesture detection from IMU data.

Architecture combines:
- MS-TCN's multi-stage refinement for temporal boundary precision
- Inception's multi-scale feature extraction for robustness to variability

Based on:
[1] MS-TCN: "Multi-Stage Temporal Convolutional Networks for Action Segmentation"
[2] Inception: "Going Deeper with Convolutions" + temporal adaptations
[3] Target task: "Drinking Gesture Detection Using Wrist-Worn IMU Sensors..."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


# ============================================================================
#  1. INCEPTION-DILATED TEMPORAL BLOCK (IDTB)
# ============================================================================

class InceptionDilatedTemporalBlock(nn.Module):
    """
    Inception-style multi-scale temporal convolution with dilation.
    
    Replaces standard dilated conv in MS-TCN with parallel branches:
    - Branch 1: 1×1 conv (pointwise, captures local patterns)
    - Branch 2: 1×1 bottleneck → 3×3 dilated conv → 1×1 projection
    - Branch 3: 1×1 bottleneck → 5×5 dilated conv → 1×1 projection  
    - Branch 4: Residual skip connection
    
    All branches use same dilation rate for consistent receptive field growth.
    Features are concatenated and projected back to original channels.
    
    Args:
        in_channels: Number of input feature channels
        out_channels: Number of output feature channels
        kernel_sizes: List of temporal kernel sizes (e.g., [1, 3, 5])
        dilation: Dilation rate for all branches (MS-TCN's exponential schedule)
        dropout: Dropout probability for regularization
        bottleneck_ratio: Channel reduction ratio for bottleneck layers (default 0.25)
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
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.dilation = dilation
        
        # Bottleneck channels για υπολογιστική αποδοτικότητα
        bottleneck_ch = max(int(in_channels * bottleneck_ratio), 8)
        
        # Δημιουργούμε παράλληλα branches για κάθε kernel size
        self.branches = nn.ModuleList()
        
        for k in kernel_sizes:
            if k == 1:
                # Branch 1: Απλό 1×1 conv (no bottleneck needed)
                branch = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                # Branches 2,3: Bottleneck → Dilated Conv → Projection
                padding = ((k - 1) * dilation) // 2  # "same" padding
                branch = nn.Sequential(
                    # Bottleneck: μείωση channels
                    nn.Conv1d(in_channels, bottleneck_ch, kernel_size=1, bias=False),
                    nn.BatchNorm1d(bottleneck_ch),
                    nn.ReLU(inplace=True),
                    
                    # Dilated temporal convolution
                    nn.Conv1d(bottleneck_ch, bottleneck_ch, kernel_size=k, 
                             dilation=dilation, padding=padding, bias=False),
                    nn.BatchNorm1d(bottleneck_ch),
                    nn.ReLU(inplace=True),
                    
                    # Projection: επαναφορά σε out_channels
                    nn.Conv1d(bottleneck_ch, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            self.branches.append(branch)
        
        # Branch 4: Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual = nn.Identity()
        
        # Concatenation projection: όλα τα branches → out_channels
        total_branch_channels = out_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            nn.Conv1d(total_branch_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass με parallel branch execution και residual fusion.
        
        Args:
            x: [batch, in_channels, temporal_length]
        Returns:
            out: [batch, out_channels, temporal_length]
        """
        # Παράλληλη επεξεργασία όλων των branches
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate κατά το channel dimension
        concat = torch.cat(branch_outputs, dim=1)  # [B, out_ch*num_branches, T]
        
        # Fusion projection
        fused = self.fusion(concat)  # [B, out_ch, T]
        
        # Residual connection
        residual = self.residual(x)  # [B, out_ch, T]
        
        # Combine και activation
        out = self.relu(fused + residual)
        out = self.dropout(out)
        
        return out


# ============================================================================
#  2. MS-TCN ENCODER-DECODER με IDTB
# ============================================================================

class InceptionMSTCNStage(nn.Module):
    """
    Ένα stage του MS-TCN με Inception-Dilated Temporal Blocks.
    
    Αρχιτεκτονική:
    - Encoder: Στοίβα IDTB με αυξανόμενα dilations (1, 2, 4, 8, ...)
    - Decoder: Στοίβα IDTB με μειούμενα dilations (..., 8, 4, 2, 1)
    - Skip connections: Encoder → Decoder για multi-resolution features
    - Output: Frame-wise predictions [batch, num_classes, T]
    
    Args:
        num_layers: Πλήθος layers στον encoder (decoder είναι συμμετρικός)
        num_features: Channels σε κάθε IDTB layer
        num_classes: Αριθμός κλάσεων (2 για binary: 0=background, 1=drinking)
        kernel_sizes: Inception kernel sizes (default [1,3,5])
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_layers: int = 10,
        num_features: int = 64,
        num_classes: int = 2,
        kernel_sizes: List[int] = [1, 3, 5],
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Dilation schedule: εκθετική αύξηση [1, 2, 4, 8, 16, ...]
        self.dilations = [2 ** i for i in range(num_layers)]
        
        # -------------------- ENCODER --------------------
        self.encoder = nn.ModuleList()
        for i, dilation in enumerate(self.dilations):
            layer = InceptionDilatedTemporalBlock(
                in_channels=num_features,
                out_channels=num_features,
                kernel_sizes=kernel_sizes,
                dilation=dilation,
                dropout=dropout
            )
            self.encoder.append(layer)
        
        # -------------------- DECODER --------------------
        # Αντίστροφη σειρά dilations για συμμετρία
        self.decoder = nn.ModuleList()
        for i, dilation in enumerate(reversed(self.dilations)):
            # Decoder layers δέχονται encoder features μέσω skip connections
            # Άρα in_channels = 2 * num_features (concat από encoder + decoder path)
            layer = InceptionDilatedTemporalBlock(
                in_channels=2 * num_features if i > 0 else num_features,
                out_channels=num_features,
                kernel_sizes=kernel_sizes,
                dilation=dilation,
                dropout=dropout
            )
            self.decoder.append(layer)
        
        # -------------------- OUTPUT LAYER --------------------
        self.output_conv = nn.Conv1d(num_features, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass με encoder-decoder και skip connections.
        
        Args:
            x: [batch, num_features, temporal_length]
        Returns:
            out: [batch, num_classes, temporal_length]
        """
        # -------------------- ENCODER PASS --------------------
        encoder_features = []
        enc = x
        for layer in self.encoder:
            enc = layer(enc)
            encoder_features.append(enc)  # Αποθήκευση για skip connections
        
        # -------------------- DECODER PASS --------------------
        dec = encoder_features[-1]  # Ξεκινάμε από το τελευταίο encoder output
        
        # Διατρέχουμε decoder layers σε αντίστροφη σειρά
        for i, layer in enumerate(self.decoder):
            if i > 0:
                # Skip connection: concatenate με αντίστοιχο encoder feature
                skip_idx = len(encoder_features) - i - 1
                dec = torch.cat([dec, encoder_features[skip_idx]], dim=1)  # [B, 2*F, T]
            
            dec = layer(dec)
        
        # -------------------- OUTPUT --------------------
        out = self.output_conv(dec)  # [B, num_classes, T]
        
        return out


# ============================================================================
#  3. MULTI-STAGE MS-TCN (πλήρες μοντέλο)
# ============================================================================

class HybridMSTCN(nn.Module):
    """
    Hybrid Multi-Stage Temporal Convolutional Network with Inception blocks.
    
    Αρχιτεκτονική:
    1. Input projection: IMU features → num_features channels
    2. Stage 1: Initial coarse predictions από raw features
    3. Stages 2-N: Refinement stages που δέχονται previous predictions + features
    4. Output: Frame-wise class probabilities για κάθε stage
    
    Training:
    - Loss computed at each stage (multi-stage supervision)
    - Classification loss + Temporal smoothing loss
    - Στόχος: Κάθε stage βελτιώνει τις προβλέψεις του προηγούμενου
    
    Args:
        num_stages: Πλήθος refinement stages (συνήθως 4)
        num_layers: Layers ανά stage encoder
        num_features: Feature channels
        num_classes: Κλάσεις (2 για binary)
        input_channels: IMU input channels (από preprocessing)
    """
    
    def __init__(
        self,
        num_stages: int = 4,
        num_layers: int = 10,
        num_features: int = 64,
        num_classes: int = 2,
        input_channels: int = 11,  # 6 IMU + 2 mag + 3 jerk από prepromax.py
        kernel_sizes: List[int] = [1, 3, 5],
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_stages = num_stages
        self.num_classes = num_classes
        
        # -------------------- INPUT PROJECTION --------------------
        # Μετατροπή raw IMU features σε latent representation
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, num_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True)
        )
        
        # -------------------- STAGE 1 (initial prediction) --------------------
        self.stage1 = InceptionMSTCNStage(
            num_layers=num_layers,
            num_features=num_features,
            num_classes=num_classes,
            kernel_sizes=kernel_sizes,
            dropout=dropout
        )
        
        # -------------------- REFINEMENT STAGES (2 έως N) --------------------
        self.refinement_stages = nn.ModuleList()
        for _ in range(num_stages - 1):
            # Κάθε refinement stage δέχεται:
            # - Original features (num_features)
            # - Previous stage predictions (num_classes)
            # Άρα input = num_features + num_classes
            stage = InceptionMSTCNStage(
                num_layers=num_layers,
                num_features=num_features,
                num_classes=num_classes,
                kernel_sizes=kernel_sizes,
                dropout=dropout
            )
            
            # Projection για να συνδυάσουμε features + predictions
            stage_proj = nn.Sequential(
                nn.Conv1d(num_features + num_classes, num_features, 
                         kernel_size=1, bias=False),
                nn.BatchNorm1d(num_features),
                nn.ReLU(inplace=True)
            )
            
            self.refinement_stages.append(nn.ModuleDict({
                'proj': stage_proj,
                'stage': stage
            }))
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass μέσω όλων των stages.
        
        Args:
            x: [batch, input_channels, temporal_length] - Raw IMU features
        
        Returns:
            outputs: List of [batch, num_classes, T] για κάθε stage
                    outputs[0] = Stage 1 predictions
                    outputs[1] = Stage 2 refined predictions
                    ...
        """
        # -------------------- INPUT PROJECTION --------------------
        features = self.input_proj(x)  # [B, num_features, T]
        
        # -------------------- STAGE 1 --------------------
        stage1_out = self.stage1(features)  # [B, num_classes, T]
        outputs = [stage1_out]
        
        # -------------------- REFINEMENT STAGES --------------------
        prev_pred = stage1_out
        for stage_module in self.refinement_stages:
            # Συνδυασμός: original features + previous predictions
            combined = torch.cat([features, prev_pred], dim=1)  # [B, F+C, T]
            
            # Projection σε feature space
            stage_input = stage_module['proj'](combined)  # [B, num_features, T]
            
            # Pass μέσω stage
            stage_out = stage_module['stage'](stage_input)  # [B, num_classes, T]
            
            outputs.append(stage_out)
            prev_pred = stage_out  # Χρήση για το επόμενο stage
        
        return outputs  # List με length = num_stages


# ============================================================================
#  4. LOSS FUNCTIONS
# ============================================================================

class HybridMSTCNLoss(nn.Module):
    """
    Composite loss για MS-TCN training:
    
    L_total = Σ_{s=1}^{S} [λ_cls * L_cls(s) + λ_smooth * L_smooth(s)]
    
    Όπου:
    - L_cls: Cross-entropy loss για classification accuracy
    - L_smooth: Temporal smoothing loss για boundary coherence
    - S: Αριθμός stages
    - Weights: λ_cls, λ_smooth (hyperparameters)
    
    Temporal Smoothing Loss:
    Penalizes rapid transitions between frames:
    L_smooth = Σ_t ||p_t - p_{t+1}||^2
    
    Όπου p_t = softmax(logits_t) είναι τα probabilities στο frame t.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        class_weights: Optional[torch.Tensor] = None,
        smooth_weight: float = 0.15,
        ignore_index: int = -100
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.smooth_weight = smooth_weight
        self.ignore_index = ignore_index
        
        # Classification loss: Cross-Entropy με optional class weights για imbalance
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
    
    def temporal_smoothing_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Υπολογίζει L_smooth = Σ_t ||p_t - p_{t+1}||^2
        
        Args:
            predictions: [batch, num_classes, T] (logits)
        
        Returns:
            smooth_loss: Scalar tensor
        """
        # Softmax για να πάρουμε probabilities
        probs = F.softmax(predictions, dim=1)  # [B, C, T]
        
        # Temporal differences: p_t - p_{t+1}
        diff = probs[:, :, 1:] - probs[:, :, :-1]  # [B, C, T-1]
        
        # L2 norm και mean
        smooth_loss = torch.mean(diff ** 2)
        
        return smooth_loss
    
    def forward(
        self,
        stage_outputs: List[torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Υπολογίζει total loss για όλα τα stages.
        
        Args:
            stage_outputs: List of [batch, num_classes, T] predictions από κάθε stage
            targets: [batch, T] ground truth labels
        
        Returns:
            total_loss: Scalar tensor για backprop
            loss_dict: Dictionary με αναλυτικά losses για logging
        """
        total_loss = 0.0
        loss_dict = {}
        
        num_stages = len(stage_outputs)
        
        for stage_idx, pred in enumerate(stage_outputs):
            # -------------------- Classification Loss --------------------
            # pred: [B, C, T], targets: [B, T]
            # Cross-entropy expects [B, C, T] format
            cls_loss = self.ce_loss(pred, targets)
            
            # -------------------- Smoothing Loss --------------------
            smooth_loss = self.temporal_smoothing_loss(pred)
            
            # -------------------- Combined Stage Loss --------------------
            stage_loss = cls_loss + self.smooth_weight * smooth_loss
            total_loss += stage_loss
            
            # Logging
            loss_dict[f'stage{stage_idx+1}_cls'] = cls_loss.item()
            loss_dict[f'stage{stage_idx+1}_smooth'] = smooth_loss.item()
            loss_dict[f'stage{stage_idx+1}_total'] = stage_loss.item()
        
        # Μέσος όρος loss across stages (για σταθερότητα)
        total_loss = total_loss / num_stages
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# ============================================================================
#  5. DATA LOADING UTILITIES
# ============================================================================

def load_preprocessed_data(
    X_pkl_path: str,
    y_pkl_path: str,
    meta_pkl_path: str
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
    """
    Φορτώνει preprocessed data από τα .pkl αρχεία του prepromax.py.
    
    Args:
        X_pkl_path: Path to DX_I_X_16Hz_with_jerk.pkl
        y_pkl_path: Path to DX_I_Y_16Hz_with_jerk.pkl
        meta_pkl_path: Path to DX_I_meta_16Hz_with_jerk.pkl
    
    Returns:
        X_list: List of [T, C] arrays (samples × channels)
        y_list: List of [T] arrays (frame-wise labels)
        meta_list: List of metadata dicts
    """
    import pickle as pkl
    
    with open(X_pkl_path, 'rb') as f:
        X_list = pkl.load(f)
    
    with open(y_pkl_path, 'rb') as f:
        y_list = pkl.load(f)
    
    with open(meta_pkl_path, 'rb') as f:
        meta_list = pkl.load(f)
    
    print(f"Loaded {len(X_list)} sessions")
    print(f"Example session: X shape={X_list[0].shape}, y shape={y_list[0].shape}")
    print(f"Feature channels: {meta_list[0]['features']}")
    
    return X_list, y_list, meta_list


class DrinkingGestureDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset για drinking gesture detection.
    
    Κάθε session είναι ένα sequence μεταβλητού μήκους:
    - X: [T, C] IMU features
    - y: [T] frame-wise labels
    
    Επιστρέφει tensors σε format [C, T] για Conv1d.
    """
    
    def __init__(
        self,
        X_list: List[np.ndarray],
        y_list: List[np.ndarray],
        meta_list: List[Dict]
    ):
        self.X_list = X_list
        self.y_list = y_list
        self.meta_list = meta_list
        
        assert len(X_list) == len(y_list) == len(meta_list)
    
    def __len__(self) -> int:
        return len(self.X_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            X: [C, T] tensor (channels first για Conv1d)
            y: [T] tensor (frame-wise labels)
            meta: Dictionary με session info
        """
        X = self.X_list[idx]  # [T, C]
        y = self.y_list[idx]  # [T]
        meta = self.meta_list[idx]
        
        # Transpose για Conv1d format: [T, C] → [C, T]
        X = torch.from_numpy(X).float().T  # [C, T]
        y = torch.from_numpy(y).long()     # [T]
        
        return X, y, meta


def collate_variable_length(batch):
    """
    Custom collate function για variable-length sequences.
    
    Επειδή κάθε session έχει διαφορετικό μήκος, δεν μπορούμε να τα batch μαζί.
    Αυτή η συνάρτηση επιστρέφει batch_size=1 πάντα (process one session at a time).
    
    Για production: θα μπορούσαμε να κάνουμε:
    - Padding στο max length του batch
    - Bucketing (group similar lengths)
    - Chunking σε fixed-length windows
    
    Εδώ κρατάμε simple: batch_size=1 για σαφήνεια.
    """
    # Batch size θα είναι πάντα 1, αλλά κρατάμε list format για consistency
    return batch[0]


# ============================================================================
#  6. METRICS & EVALUATION
# ============================================================================

def compute_frame_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100
) -> float:
    """
    Frame-wise classification accuracy.
    
    Args:
        predictions: [T] predicted class indices
        targets: [T] ground truth labels
    
    Returns:
        accuracy: float in [0, 1]
    """
    mask = targets != ignore_index
    correct = (predictions[mask] == targets[mask]).float().sum()
    total = mask.float().sum()
    
    accuracy = (correct / total).item() if total > 0 else 0.0
    return accuracy


def compute_segmental_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Segmental F1, Precision, Recall για binary classification.
    
    Υπολογίζει metrics σε segment-level (όχι frame-level):
    - True Positive: ορθή ανίχνευση drinking segment
    - False Positive: λανθασμένη ανίχνευση
    - False Negative: missed drinking segment
    
    Args:
        predictions: [T] predicted labels (0 or 1)
        targets: [T] ground truth labels (0 or 1)
    
    Returns:
        metrics: Dict με F1, precision, recall
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    # Binary metrics: class 1 (drinking) είναι το θετικό
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=[1], average='binary', zero_division=0
    )
    
    metrics = {
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall)
    }
    
    return metrics


# ============================================================================
#  7. TRAINING LOOP (EXAMPLE)
# ============================================================================

def train_one_epoch(
    model: HybridMSTCN,
    dataloader: torch.utils.data.DataLoader,
    criterion: HybridMSTCNLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """
    Ένα training epoch.
    
    Returns:
        epoch_metrics: Dictionary με average losses
    """
    model.train()
    
    epoch_losses = []
    epoch_accs = []
    
    for batch_idx, (X, y, meta) in enumerate(dataloader):
        X = X.to(device)      # [1, C, T] (batch_size=1)
        y = y.to(device)      # [1, T]
        
        # Αφαιρούμε batch dimension αν είναι 1
        print("X shape:")
        print(X.shape)
        if X.shape[0] == 1:
            X = X.squeeze(0)  # [C, T]
            y = y.squeeze(0)  # [T]
        
        # Προσθέτουμε batch dimension για το model
        X = X.unsqueeze(0)    # [1, C, T]
        
        # Forward pass
        optimizer.zero_grad()
        stage_outputs = model(X)  # List of [1, num_classes, T]
        
        # Loss computation
        y_batch = y.unsqueeze(0)  # [1, T]
        loss, loss_dict = criterion(stage_outputs, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics (final stage predictions)
        final_pred = stage_outputs[-1].squeeze(0)  # [num_classes, T]
        pred_labels = final_pred.argmax(dim=0)     # [T]
        acc = compute_frame_accuracy(pred_labels, y)
        
        epoch_losses.append(loss.item())
        epoch_accs.append(acc)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}: "
                  f"Loss={loss.item():.4f}, Acc={acc:.4f}")
    
    metrics = {
        'train_loss': np.mean(epoch_losses),
        'train_acc': np.mean(epoch_accs)
    }
    
    return metrics


# ============================================================================
#  8. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Παράδειγμα χρήσης για training του hybrid model.
    """
    print("INITIATING ....")
    # -------------------- CONFIGURATION --------------------
    DATA_DIR = r"C:\Users\Apost\Desktop\Διπλωματική\others\open datasets\drinking gesture identifications.open\code"
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Model hyperparameters
    NUM_STAGES =  4
    NUM_LAYERS = 10
    NUM_FEATURES = 64
    NUM_CLASSES = 2
    INPUT_CHANNELS = 11  # από prepromax: 6 IMU + 2 mag + 1 jerk_mag + 2 extra αν add_jerk_axes=True
    
    # Training hyperparameters
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 15 # 50
    SMOOTH_WEIGHT = 0.15
    
    # -------------------- LOAD DATA --------------------
    print("\n=== Loading preprocessed data ===")
    X_list, y_list, meta_list = load_preprocessed_data(
        X_pkl_path=f"{DATA_DIR}/DX_I_X_16Hz_with_jerk.pkl",
        y_pkl_path=f"{DATA_DIR}/DX_I_Y_16Hz_with_jerk.pkl",
        meta_pkl_path=f"{DATA_DIR}/DX_I_meta_16Hz_with_jerk.pkl"
    )
    
    # Έλεγχος input channels από meta
    actual_input_ch = X_list[0].shape[1]
    print(f"Actual input channels: {actual_input_ch}")
    INPUT_CHANNELS = actual_input_ch
    
    # -------------------- CREATE DATASET --------------------
    dataset = DrinkingGestureDataset(X_list, y_list, meta_list)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_variable_length
    )
    
    # -------------------- INITIALIZE MODEL --------------------
    print("\n=== Initializing Hybrid MS-TCN + Inception ===")
    model = HybridMSTCN(
        num_stages=NUM_STAGES,
        num_layers=NUM_LAYERS,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        input_channels=INPUT_CHANNELS,
        kernel_sizes=[1, 3, 5],
        dropout=0.3
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # -------------------- LOSS & OPTIMIZER --------------------
    # Class weights για imbalance (optional - υπολογίζεται από data)
    all_labels = np.concatenate(y_list)
    class_counts = np.bincount(all_labels)
    class_weights = torch.tensor(
        [1.0 / c if c > 0 else 0.0 for c in class_counts],
        dtype=torch.float32,
        device=DEVICE
    )
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"Class weights: {class_weights}")
    
    criterion = HybridMSTCNLoss(
        num_classes=NUM_CLASSES,
        class_weights=class_weights,
        smooth_weight=SMOOTH_WEIGHT
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # -------------------- TRAINING LOOP --------------------
    print("\n=== Starting Training ===")
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        metrics = train_one_epoch(model, dataloader, criterion, optimizer, DEVICE)
        
        print(f"  Train Loss: {metrics['train_loss']:.4f}, "
              f"Train Acc: {metrics['train_acc']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(metrics['train_loss'])
        
        # Save best model
        if metrics['train_loss'] < best_loss:
            best_loss = metrics['train_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': {
                    'num_stages': NUM_STAGES,
                    'num_layers': NUM_LAYERS,
                    'num_features': NUM_FEATURES,
                    'num_classes': NUM_CLASSES,
                    'input_channels': INPUT_CHANNELS
                }
            }, f"{DATA_DIR}/hybrid_mstcn_best.pth")
            print(f"  [SAVED] Best model (loss={best_loss:.4f})")
    
    print("\n=== Training Complete ===")