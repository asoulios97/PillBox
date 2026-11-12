"""
mstcn_pipeline.py -- Drinking gesture detection on DX-I (wrist IMU, 16 Hz).

Paper-faithful MS-TCN baseline (Wang et al., EMBC 2022) plus Inception and
"hybrid legacy" variants, with a corrected evaluation pipeline.

This file supersedes legacy/new_bottle_window_max.py (kept untouched for the record).
Fixes relative to that script (numbers refer to the external review):
  P1  smoothing loss = MS-TCN truncated MSE on log-prob differences, masked
  P2  segmental F1 = one-to-one matching (official MS-TCN), counts pooled
  P3  evaluation on continuous per-session timelines (no window concatenation)
  P4  per-layer dilation span checked against training length
  P5  no test-fold model selection: fixed epochs ("last") or inner held-out subject
  P6  LOSO by *subject* (2-1/2-2 and 6-1/6-2 are the same person), left/right OR
      merge, pooled TP/FP/FN
  P7  saved config is the config that ran
  P8/P9 baseline = 2-stage MS-TCN; refinement stages see softmax(prev) only;
      ablation flags for Inception / decoder / bottleneck / refine input
  P10 class weighting configurable (default none, as in the paper)
  P11 labels cast to int64 and validated
  P12 stratified bins removed (LOSO or seeded GroupKFold by subject)
  P13 seeds, compressed predictions, ASCII logs, no dead code

Usage (from the code/ folder):
  .venv/Scripts/python.exe mstcn_pipeline.py --preset paper
  .venv/Scripts/python.exe mstcn_pipeline.py --preset paper_inception
  .venv/Scripts/python.exe mstcn_pipeline.py --preset legacy
  .venv/Scripts/python.exe mstcn_pipeline.py --preset paper --epochs 2 --max_folds 1   # smoke run
Any Config field can be overridden on the command line (--name value, booleans
as --flag / --no-flag, tuples as comma-separated lists).
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import math
import pickle as pkl
import platform
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = -100
HERE = Path(__file__).resolve().parent
EXPECTED_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz", "acc_mag", "gyro_mag", "jerk_mag"]

# Reference numbers, Wang et al. 2022, Table I (DX-I, MS-TCN), OR-merged, LOSO.
PAPER_DXI = {0.1: (401, 28, 9, 0.956), 0.25: (400, 29, 9, 0.955), 0.5: (391, 32, 15, 0.943)}


# ============================================================================
#  CONFIG
# ============================================================================

@dataclass
class Config:
    # ---- data ----
    data_dir: str = str(HERE)
    data_prefix: str = "DX_I"
    data_suffix: str = "16Hz_with_jerk"
    channels: int = 6                      # 6 = ax..gz (paper); 9 adds magnitudes + jerk
    # ---- protocol ----
    folds: str = "loso"                    # "loso" or an integer K (seeded GroupKFold by subject)
    max_folds: int = 0                     # >0: run only the first N folds (smoke tests)
    select: str = "last"                   # "last" (fixed epochs) | "inner" (held-out train subject)
    or_merge: bool = True                  # paper: OR left/right predictions per recording
    seg_rule: str = "paper"                # primary segmental-F1 rule: "paper" (Eq. 8) | "standard" (MS-TCN)
    iou_thresholds: Tuple[float, ...] = (0.1, 0.25, 0.5)
    # ---- model ----
    arch: str = "mstcn"                    # "mstcn" | "hybrid_legacy"
    layer_type: str = "dilated"            # mstcn: "dilated" (paper) | "inception"
    inception_stages: str = "all"          # mstcn+inception: "all" | "stage1"
    inception_kernels: Tuple[int, ...] = (3, 5, 9)
    inception_bottleneck: int = 32
    inception_maxpool: bool = True
    num_stages: int = 2
    num_layers: int = 9
    num_features: int = 128
    kernel_size: int = 3
    dropout: float = 0.3
    # ---- hybrid_legacy only ----
    legacy_kernels: Tuple[int, ...] = (1, 3, 5)
    use_decoder: bool = True
    use_bottleneck: bool = True
    bottleneck_interval: int = 3
    refine_input: str = "pred_only"        # "pred_only" (MS-TCN) | "features_and_pred" (old)
    # ---- loss ----
    smooth_weight: float = 0.15            # lambda
    tau: float = 4.0
    class_weight: str = "none"             # "none" | "inverse" | "sqrt_inverse"
    stage_reduce: str = "sum"              # "sum" (paper) | "mean"
    # ---- training ----
    train_mode: str = "full"               # "full" sessions | random "crops" | fixed "windows"
    infer_mode: str = "auto"               # "auto" | "full" | "windows"
    window: int = 128
    stride: int = 64
    crop_len: int = 4096
    batch_size: int = 1
    epochs: int = 50
    lr: float = 5e-4
    weight_decay: float = 0.0
    scheduler: str = "none"                # "none" | "plateau"
    grad_clip: float = 0.0
    eval_every: int = 5                    # inner-selection evaluation period (epochs)
    monitor_every: int = 1                 # held-out fold loss/F1 every N epochs (monitoring only; 0 = off)
    plots: bool = True                     # write plots/fold_XX.png + loss_all_folds.png
    vote_weight: str = "uniform"           # windows inference: "uniform" | "center"
    allow_large_span: bool = False
    # ---- misc ----
    preset: str = ""                       # filled in by build_config (for the experiment index)
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"
    save_models: bool = True
    results_dir: str = str(HERE / "results")
    tag: str = ""
    verbose: int = 1


PRESETS: Dict[str, Dict] = {
    # Wang et al. 2022: 2 stages x 9 dilated residual layers x 128 filters, k=3,
    # dropout 0.3, CE + 0.15 * T-MSE (tau 4), Adam 5e-4, whole recordings, LOSO, OR.
    "paper": {},
    # Minimal addition: every dilated layer becomes an Inception dilated layer.
    "paper_inception": {"layer_type": "inception"},
    # Same, but only the prediction-generation stage uses Inception layers.
    "paper_inception_s1": {"layer_type": "inception", "inception_stages": "stage1"},
    # Old architecture (new_bottle_window_max.py) with the fixed loss/metrics/protocol.
    # num_layers 6 instead of 10: with 5-tap kernels and 128-sample windows, dilations
    # >= 64 make every side tap fall outside the window (see check_dilation_span).
    "legacy": {
        "arch": "hybrid_legacy", "num_stages": 4, "num_layers": 6, "num_features": 64,
        "refine_input": "features_and_pred", "train_mode": "windows", "window": 128,
        "stride": 64, "batch_size": 128, "class_weight": "inverse",
        "weight_decay": 5e-4, "scheduler": "plateau", "stage_reduce": "mean",
    },
}

TUPLE_INT_FIELDS = {"inception_kernels", "legacy_kernels"}
TUPLE_FLOAT_FIELDS = {"iou_thresholds"}


def build_config(preset: str = "paper", /, **overrides) -> Config:
    """Preset values, then overrides (an override `preset=` only relabels, e.g. when reloading a saved config)."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    values = {"preset": preset, **PRESETS[preset]}
    values.update(overrides)
    names = {f.name for f in dataclasses.fields(Config)}
    bad = set(values) - names
    if bad:
        raise ValueError(f"unknown config fields: {sorted(bad)}")
    for k in TUPLE_INT_FIELDS & set(values):
        values[k] = tuple(int(v) for v in values[k])
    for k in TUPLE_FLOAT_FIELDS & set(values):
        values[k] = tuple(float(v) for v in values[k])
    cfg = Config(**values)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    choices = {
        "arch": {"mstcn", "hybrid_legacy"}, "layer_type": {"dilated", "inception"},
        "inception_stages": {"all", "stage1"}, "refine_input": {"pred_only", "features_and_pred"},
        "class_weight": {"none", "inverse", "sqrt_inverse"}, "stage_reduce": {"sum", "mean"},
        "train_mode": {"full", "crops", "windows"}, "infer_mode": {"auto", "full", "windows"},
        "select": {"last", "inner"}, "scheduler": {"none", "plateau"}, "seg_rule": {"paper", "standard"},
        "vote_weight": {"uniform", "center"},
    }
    for k, allowed in choices.items():
        if getattr(cfg, k) not in allowed:
            raise ValueError(f"{k}={getattr(cfg, k)!r} not in {sorted(allowed)}")
    if cfg.folds != "loso" and not str(cfg.folds).isdigit():
        raise ValueError("folds must be 'loso' or an integer")
    for k in (cfg.kernel_size, *cfg.inception_kernels, *cfg.legacy_kernels):
        if k % 2 == 0:
            raise ValueError(f"kernel sizes must be odd for same-length output (got {k})")
    if cfg.channels not in (6, 9):
        raise ValueError("channels must be 6 or 9")


def resolve_infer_mode(cfg: Config) -> str:
    if cfg.infer_mode != "auto":
        return cfg.infer_mode
    return "windows" if cfg.train_mode == "windows" else "full"


# ============================================================================
#  DATA
# ============================================================================

@dataclass
class Session:
    x: np.ndarray            # [T, C] float32
    y: np.ndarray            # [T] int64 in {0, 1}
    recording: str           # e.g. "2-1"  (subject 2, recording 1)
    subject: str             # e.g. "2"
    hand: str                # "Left" | "Right"

    @property
    def name(self) -> str:
        return f"{self.recording}/{self.hand}"


def load_sessions(cfg: Config) -> List[Session]:
    d = Path(cfg.data_dir)

    def _load(kind: str):
        with open(d / f"{cfg.data_prefix}_{kind}_{cfg.data_suffix}.pkl", "rb") as f:
            return pkl.load(f)

    X, Y, M = _load("X"), _load("Y"), _load("meta")
    if not (len(X) == len(Y) == len(M)):
        raise ValueError("X/Y/meta length mismatch")
    sessions = []
    for x, y, m in zip(X, Y, M):
        feats = list(m.get("features", EXPECTED_CHANNELS))
        if feats[:cfg.channels] != EXPECTED_CHANNELS[:cfg.channels]:
            raise ValueError(f"unexpected channel order {feats}")
        y = np.asarray(y).astype(np.int64)                       # P11
        if not set(np.unique(y).tolist()) <= {0, 1}:
            raise ValueError(f"labels outside {{0,1}} in {m.get('user_id')}")
        x = np.ascontiguousarray(np.asarray(x)[:, :cfg.channels], dtype=np.float32)
        if len(x) != len(y):
            raise ValueError("signal/label length mismatch")
        rec = str(m["user_id"])
        sessions.append(Session(x=x, y=y, recording=rec, subject=rec.split("-")[0],
                                hand=str(m.get("hand", "?"))))
    return sessions


def _subject_key(s: str):
    return (0, int(s)) if s.isdigit() else (1, s)


def make_folds(sessions: Sequence[Session], folds: str, seed: int) -> List[Dict]:
    """Subject-level folds. All recordings and both hands of a person share a fold."""
    subjects = sorted({s.subject for s in sessions}, key=_subject_key)
    if folds == "loso":
        groups = [[s] for s in subjects]
    else:
        k = int(folds)
        if not 2 <= k <= len(subjects):
            raise ValueError(f"folds={k} invalid for {len(subjects)} subjects")
        perm = list(np.random.RandomState(seed).permutation(subjects))
        groups = [sorted(perm[i::k], key=_subject_key) for i in range(k)]
    out = []
    for g in groups:
        g = set(g)
        test = [i for i, s in enumerate(sessions) if s.subject in g]
        train = [i for i, s in enumerate(sessions) if s.subject not in g]
        tr_subj = {sessions[i].subject for i in train}
        te_subj = {sessions[i].subject for i in test}
        if tr_subj & te_subj:
            raise RuntimeError(f"subject leakage: {tr_subj & te_subj}")
        out.append({"test_subjects": sorted(g, key=_subject_key), "train": train, "test": test})
    return out


def window_starts(T: int, window: int, stride: int) -> List[int]:
    """Start indices of fixed windows; the last one may run past T (zero/ignore padded)."""
    if T <= window:
        return [0]
    return [i * stride for i in range(1 + math.ceil((T - window) / stride))]


def create_sliding_windows(x: np.ndarray, y: np.ndarray, window: int, stride: int):
    starts = window_starts(len(y), window, stride)
    Xw = np.zeros((len(starts), window, x.shape[1]), dtype=np.float32)
    Yw = np.full((len(starts), window), IGNORE, dtype=np.int64)
    for i, s in enumerate(starts):
        seg = slice(s, min(s + window, len(y)))
        n = seg.stop - seg.start
        Xw[i, :n] = x[seg]
        Yw[i, :n] = y[seg]
    return Xw, Yw


def collate(items: Sequence[Tuple[np.ndarray, np.ndarray]]):
    """Pad a list of (x[T,C], y[T]) to [B,C,Tmax] / [B,Tmax]; padding labelled IGNORE."""
    T = max(len(y) for _, y in items)
    C = items[0][0].shape[1]
    X = np.zeros((len(items), C, T), dtype=np.float32)
    Y = np.full((len(items), T), IGNORE, dtype=np.int64)
    for i, (x, y) in enumerate(items):
        X[i, :, :len(y)] = x.T
        Y[i, :len(y)] = y
    return torch.from_numpy(X), torch.from_numpy(Y)


def epoch_items(cfg: Config, train: Sequence[Session], rng: np.random.RandomState,
                windows_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None):
    if cfg.train_mode == "full":
        return [(train[i].x, train[i].y) for i in rng.permutation(len(train))]
    if cfg.train_mode == "crops":
        lens = np.array([len(s.y) for s in train], dtype=np.float64)
        n = max(1, int(math.ceil(lens.sum() / cfg.crop_len)))
        picks = rng.choice(len(train), size=n, p=lens / lens.sum())
        items = []
        for i in picks:
            s = train[i]
            if len(s.y) <= cfg.crop_len:
                items.append((s.x, s.y))
            else:
                a = rng.randint(0, len(s.y) - cfg.crop_len + 1)
                items.append((s.x[a:a + cfg.crop_len], s.y[a:a + cfg.crop_len]))
        return items
    Xw, Yw = windows_cache
    return [(Xw[i], Yw[i]) for i in rng.permutation(len(Xw))]


# ============================================================================
#  MODELS
# ============================================================================

class DilatedResidualLayer(nn.Module):
    """MS-TCN layer: dilated conv -> ReLU -> 1x1 -> dropout -> + input."""

    def __init__(self, dilation: int, channels: int, kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.conv_1x1(F.relu(self.conv(x))))


class InceptionResidualLayer(nn.Module):
    """Inception dilated layer (InceptionTime-style) as a drop-in for DilatedResidualLayer.

    x -> 1x1 bottleneck -> parallel dilated convs (kernels, same dilation)
      +  maxpool(3) -> 1x1 branch on x
      -> concat -> ReLU -> 1x1 fuse to `channels` -> dropout -> + x
    """

    def __init__(self, dilation: int, channels: int, kernels: Sequence[int] = (3, 5, 9),
                 bottleneck: int = 32, maxpool: bool = True, dropout: float = 0.3):
        super().__init__()
        b = min(bottleneck, channels)
        self.bottleneck = nn.Conv1d(channels, b, 1)
        self.branches = nn.ModuleList([
            nn.Conv1d(b, b, k, padding=dilation * (k - 1) // 2, dilation=dilation) for k in kernels
        ])
        self.use_maxpool = maxpool
        n = len(kernels)
        if maxpool:
            self.pool_conv = nn.Conv1d(channels, b, 1)
            n += 1
        self.fuse = nn.Conv1d(n * b, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.bottleneck(x)
        outs = [br(z) for br in self.branches]
        if self.use_maxpool:
            outs.append(self.pool_conv(F.max_pool1d(x, 3, stride=1, padding=1)))
        return x + self.dropout(self.fuse(F.relu(torch.cat(outs, dim=1))))


class SingleStage(nn.Module):
    def __init__(self, in_dim: int, channels: int, num_classes: int, num_layers: int,
                 layer_factory: Callable[[int], nn.Module]):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dim, channels, 1)
        self.layers = nn.ModuleList([layer_factory(2 ** i) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(channels, num_classes, 1)

    def forward(self, x):
        h = self.conv_in(x)
        for layer in self.layers:
            h = layer(h)
        return self.conv_out(h)


class MSTCN(nn.Module):
    """Multi-stage TCN (Farha & Gall); refinement stages see softmax of the previous stage."""

    def __init__(self, cfg: Config, in_dim: int, num_classes: int = 2):
        super().__init__()

        def factory(inception: bool):
            if inception:
                return lambda d: InceptionResidualLayer(d, cfg.num_features, cfg.inception_kernels,
                                                        cfg.inception_bottleneck,
                                                        cfg.inception_maxpool, cfg.dropout)
            return lambda d: DilatedResidualLayer(d, cfg.num_features, cfg.kernel_size, cfg.dropout)

        inc = cfg.layer_type == "inception"
        inc_ref = inc and cfg.inception_stages == "all"
        self.stages = nn.ModuleList(
            [SingleStage(in_dim, cfg.num_features, num_classes, cfg.num_layers, factory(inc))]
            + [SingleStage(num_classes, cfg.num_features, num_classes, cfg.num_layers, factory(inc_ref))
               for _ in range(cfg.num_stages - 1)])

    def forward(self, x) -> List[torch.Tensor]:
        out = self.stages[0](x)
        outs = [out]
        for stage in self.stages[1:]:
            out = stage(F.softmax(out, dim=1))
            outs.append(out)
        return outs


# ---- hybrid legacy (architecture of new_bottle_window_max.py, with flags) ----

class LegacyBottleneck(nn.Module):
    """1x1 C -> max(ratio*C, 8) -> C with BN, residual, ReLU; dropout inside and after
    (as in the original code, kept so the legacy architecture is unchanged)."""

    def __init__(self, channels: int, ratio: float = 0.15, dropout: float = 0.3):
        super().__init__()
        b = max(int(channels * ratio), 8)
        self.body = nn.Sequential(
            nn.Conv1d(channels, b, 1, bias=False), nn.BatchNorm1d(b), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(b, channels, 1, bias=False), nn.BatchNorm1d(channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(F.relu(self.body(x) + x))


class LegacyInceptionBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernels: Sequence[int], dilation: int,
                 dropout: float, bottleneck_ratio: float = 0.25):
        super().__init__()
        b = max(int(in_ch * bottleneck_ratio), 8)
        self.branches = nn.ModuleList()
        for k in kernels:
            if k == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv1d(in_ch, b, 1, bias=False), nn.BatchNorm1d(b), nn.ReLU(inplace=True),
                    nn.Conv1d(b, b, k, dilation=dilation, padding=dilation * (k - 1) // 2, bias=False),
                    nn.BatchNorm1d(b), nn.ReLU(inplace=True),
                    nn.Conv1d(b, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)))
        self.residual = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.fusion = nn.Sequential(nn.Conv1d(out_ch * len(kernels), out_ch, 1, bias=False),
                                    nn.BatchNorm1d(out_ch))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        fused = self.fusion(torch.cat([br(x) for br in self.branches], dim=1))
        return self.dropout(F.relu(fused + self.residual(x)))


class LegacyStage(nn.Module):
    def __init__(self, cfg: Config, num_classes: int):
        super().__init__()
        C, L = cfg.num_features, cfg.num_layers
        self.use_decoder = cfg.use_decoder
        dil = [2 ** i for i in range(L)]

        def bneck(i):
            if cfg.use_bottleneck and (i + 1) % cfg.bottleneck_interval == 0:
                return LegacyBottleneck(C, dropout=cfg.dropout)
            return nn.Identity()

        self.enc = nn.ModuleList([LegacyInceptionBlock(C, C, cfg.legacy_kernels, d, cfg.dropout) for d in dil])
        self.enc_b = nn.ModuleList([bneck(i) for i in range(L)])
        if self.use_decoder:
            self.dec = nn.ModuleList([
                LegacyInceptionBlock(2 * C if i > 0 else C, C, cfg.legacy_kernels, d, cfg.dropout)
                for i, d in enumerate(reversed(dil))])
            self.dec_b = nn.ModuleList([bneck(i) for i in range(L)])
        self.out = nn.Conv1d(C, num_classes, 1)

    def forward(self, x):
        feats, h = [], x
        for layer, b in zip(self.enc, self.enc_b):
            h = b(layer(h))
            feats.append(h)
        if self.use_decoder:
            for i, (layer, b) in enumerate(zip(self.dec, self.dec_b)):
                if i > 0:
                    h = torch.cat([h, feats[len(feats) - i - 1]], dim=1)
                h = b(layer(h))
        return self.out(h)


class HybridLegacy(nn.Module):
    def __init__(self, cfg: Config, in_dim: int, num_classes: int = 2):
        super().__init__()
        C = cfg.num_features
        self.refine_input = cfg.refine_input
        self.input_proj = nn.Sequential(nn.Conv1d(in_dim, C, 1, bias=False), nn.BatchNorm1d(C),
                                        nn.ReLU(inplace=True))
        self.stage1 = LegacyStage(cfg, num_classes)
        proj_in = C + num_classes if cfg.refine_input == "features_and_pred" else num_classes
        self.refine = nn.ModuleList([nn.ModuleDict({
            "proj": nn.Sequential(nn.Conv1d(proj_in, C, 1, bias=False), nn.BatchNorm1d(C), nn.ReLU(inplace=True)),
            "bneck": LegacyBottleneck(C, dropout=cfg.dropout) if cfg.use_bottleneck else nn.Identity(),
            "stage": LegacyStage(cfg, num_classes)}) for _ in range(cfg.num_stages - 1)])

    def forward(self, x) -> List[torch.Tensor]:
        feats = self.input_proj(x)
        out = self.stage1(feats)
        outs = [out]
        for m in self.refine:
            if self.refine_input == "features_and_pred":
                inp = torch.cat([feats, out], dim=1)      # original behaviour (raw logits)
            else:
                inp = F.softmax(out, dim=1)
            out = m["stage"](m["bneck"](m["proj"](inp)))
            outs.append(out)
        return outs


def build_model(cfg: Config, in_dim: int) -> nn.Module:
    return MSTCN(cfg, in_dim) if cfg.arch == "mstcn" else HybridLegacy(cfg, in_dim)


def _layer_kmax(cfg: Config, stage_idx: int) -> int:
    if cfg.arch == "hybrid_legacy":
        return max(cfg.legacy_kernels)
    if cfg.layer_type == "inception" and (stage_idx == 0 or cfg.inception_stages == "all"):
        return max(max(cfg.inception_kernels), 3)
    return cfg.kernel_size


def receptive_field(cfg: Config) -> Dict:
    """Theoretical receptive field (samples) per stage and for the whole stack."""
    per_stage = []
    for s in range(cfg.num_stages):
        k = _layer_kmax(cfg, s)
        rf = 1 + sum(max((k - 1) * 2 ** i, 2) for i in range(cfg.num_layers))
        if cfg.arch == "hybrid_legacy" and cfg.use_decoder:
            rf = 2 * rf - 1
        per_stage.append(rf)
    return {"per_stage": per_stage, "total": 1 + sum(r - 1 for r in per_stage),
            "max_layer_half_span": max((_layer_kmax(cfg, s) - 1) // 2 * 2 ** (cfg.num_layers - 1)
                                       for s in range(cfg.num_stages))}


def check_dilation_span(cfg: Config) -> Optional[str]:
    """P4: in windowed/cropped training, a layer whose side taps (half-span) reach >= the
    training length never sees real data on either side -> behaves like a 1x1 conv."""
    if cfg.train_mode == "full":
        return None
    L = cfg.window if cfg.train_mode == "windows" else cfg.crop_len
    half = receptive_field(cfg)["max_layer_half_span"]
    if half >= L:
        msg = (f"largest dilated layer reaches +/-{half} samples but training length is {L}: "
               f"reduce num_layers or increase window/crop_len")
        if not cfg.allow_large_span:
            raise ValueError(msg + " (or pass --allow_large_span)")
        return msg
    if half > L // 2:
        return f"largest dilated layer reaches +/-{half} samples, more than half the training length {L}"
    return None


# ============================================================================
#  LOSS
# ============================================================================

class MSTCNLoss(nn.Module):
    """sum/mean over stages of  CE + lambda * T-MSE,
    T-MSE = mean_{t,c} min(|log p(t,c) - log p(t-1,c)|, tau)^2   (t-1 term detached,
    as in the official MS-TCN code); padded frames (IGNORE) are excluded from both terms."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None, smooth_weight: float = 0.15,
                 tau: float = 4.0, stage_reduce: str = "sum"):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.tau = tau
        self.stage_reduce = stage_reduce
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE)

    def smoothing(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] < 2:
            return logits.sum() * 0.0
        log_p = F.log_softmax(logits, dim=1)
        delta = (log_p[:, :, 1:] - log_p[:, :, :-1].detach()).abs()
        loss = delta.clamp(max=self.tau) ** 2
        m = (mask[:, 1:] & mask[:, :-1]).unsqueeze(1).to(loss.dtype)
        return (loss * m).sum() / (m.sum() * loss.shape[1]).clamp(min=1.0)

    def forward(self, stage_outputs: List[torch.Tensor], targets: torch.Tensor):
        mask = targets != IGNORE
        total = 0.0
        parts = {"cls": 0.0, "smooth": 0.0}
        for out in stage_outputs:
            cls = self.ce(out, targets) if mask.any() else out.sum() * 0.0
            sm = self.smoothing(out, mask)
            total = total + cls + self.smooth_weight * sm
            parts["cls"] += cls.item()
            parts["smooth"] += sm.item()
        if self.stage_reduce == "mean":
            total = total / len(stage_outputs)
        return total, parts


def compute_class_weights(scheme: str, sessions: Sequence[Session]) -> Optional[torch.Tensor]:
    if scheme == "none":
        return None
    y = np.concatenate([s.y for s in sessions])
    n, n1 = len(y), int((y == 1).sum())
    n0 = n - n1
    w = np.array([n / (2 * max(n0, 1)), n / (2 * max(n1, 1))], dtype=np.float64)
    if scheme == "sqrt_inverse":
        w = np.sqrt(w)
    return torch.tensor(w / w[0], dtype=torch.float32)   # class 0 weight = 1


# ============================================================================
#  METRICS
# ============================================================================

def get_segments(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Half-open [start, end) runs of label 1."""
    b = (np.asarray(labels) == 1).astype(np.int8)
    if b.size == 0:
        return []
    d = np.diff(np.concatenate([[0], b, [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def segmental_counts(pred: np.ndarray, gt: np.ndarray, k: float) -> Tuple[int, int, int]:
    """One-to-one segment matching (official MS-TCN F1@k): each predicted segment is a TP
    if its best-IoU ground-truth segment has IoU >= k and is not already matched, else FP.
    Unmatched ground-truth segments are FN. Invariant: TP + FN == #GT segments."""
    P, G = get_segments(pred), get_segments(gt)
    hit = np.zeros(len(G), dtype=bool)
    tp = fp = 0
    for ps, pe in P:
        best, best_i = 0.0, -1
        for i, (gs, ge) in enumerate(G):
            inter = min(pe, ge) - max(ps, gs)
            if inter <= 0:
                continue
            iou = inter / (max(pe, ge) - min(ps, gs))
            if iou > best:
                best, best_i = iou, i
        if best_i >= 0 and best >= k and not hit[best_i]:
            tp += 1
            hit[best_i] = True
        else:
            fp += 1
    return tp, fp, int(len(G) - hit.sum())


def segmental_counts_paper(pred: np.ndarray, gt: np.ndarray, k: float) -> Tuple[int, int, int]:
    """Decision rules described in Wang et al. 2022 (Sec. II-E, Eq. 8):
      - prediction overlapping no GT -> FP;  GT overlapped by no prediction -> FN
      - several predictions inside one GT -> best one is judged, the rest are FP
      - one prediction spanning several GTs -> judged against its best GT, the other GTs are FN
      - judged pair with IoU >= k -> TP; IoU < k -> ONE error: FP if the GT is shorter than the
        prediction (over-segmentation), else FN.
    Unlike segmental_counts, a weak match costs one error instead of two, so TP + FN can be
    smaller than #GT (paper Table I, k=0.5: 391 + 15 = 406 < 410)."""
    P, G = get_segments(pred), get_segments(gt)
    assigned: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    tp = fp = fn = 0
    for j, (ps, pe) in enumerate(P):
        best, best_i = 0.0, -1
        for i, (gs, ge) in enumerate(G):
            inter = min(pe, ge) - max(ps, gs)
            if inter > 0:
                iou = inter / (max(pe, ge) - min(ps, gs))
                if iou > best:
                    best, best_i = iou, i
        if best_i < 0:
            fp += 1
        else:
            assigned[best_i].append((best, j))
    for i, (gs, ge) in enumerate(G):
        cands = assigned.get(i)
        if not cands:
            fn += 1
            continue
        iou, j = max(cands)
        fp += len(cands) - 1
        if iou >= k:
            tp += 1
        elif (ge - gs) < (P[j][1] - P[j][0]):
            fp += 1
        else:
            fn += 1
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d > 0 else 0.0


def or_merge(sessions: Sequence[Session], preds: Sequence[np.ndarray]):
    """Per recording, OR left/right predictions and labels (paper). Hands share t=0 at 64 Hz
    but end a few seconds apart, so both are truncated to the shorter length."""
    groups: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for s, p in zip(sessions, preds):
        groups[s.recording].append((np.asarray(p), s.y))
    merged = []
    for rec in sorted(groups):
        arrs = groups[rec]
        L = min(len(y) for _, y in arrs)
        p_or = np.zeros(L, dtype=np.int64)
        y_or = np.zeros(L, dtype=np.int64)
        for p, y in arrs:
            p_or |= (p[:L] == 1)
            y_or |= (y[:L] == 1)
        merged.append((rec, p_or, y_or))
    return merged


def evaluate_predictions(sessions: Sequence[Session], probs: Sequence[np.ndarray],
                         ks: Sequence[float]) -> Dict:
    """Counts only (no ratios) so they can be pooled across folds."""
    preds = [p.argmax(axis=1).astype(np.int64) for p in probs]
    res = {"per_hand": {}, "or_merged": {}, "frame": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
           "n_gt_segments_per_hand": int(sum(len(get_segments(s.y)) for s in sessions))}
    for s, p in zip(sessions, preds):
        y = s.y
        res["frame"]["tp"] += int(((p == 1) & (y == 1)).sum())
        res["frame"]["fp"] += int(((p == 1) & (y == 0)).sum())
        res["frame"]["fn"] += int(((p == 0) & (y == 1)).sum())
        res["frame"]["tn"] += int(((p == 0) & (y == 0)).sum())
    merged = or_merge(sessions, preds)
    res["n_gt_segments_or"] = int(sum(len(get_segments(y)) for _, _, y in merged))
    res["per_hand_paperrule"], res["or_merged_paperrule"] = {}, {}
    for k in ks:
        key = f"k={k}"
        for suffix, fn_counts in (("", segmental_counts), ("_paperrule", segmental_counts_paper)):
            c = np.zeros(3, dtype=np.int64)
            for s, p in zip(sessions, preds):
                c += fn_counts(p, s.y, k)
            res["per_hand" + suffix][key] = dict(zip(("tp", "fp", "fn"), map(int, c)))
            c = np.zeros(3, dtype=np.int64)
            for _, p, y in merged:
                c += fn_counts(p, y, k)
            res["or_merged" + suffix][key] = dict(zip(("tp", "fp", "fn"), map(int, c)))
    return res


def frame_scores(c: Dict) -> Dict:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / max(tp + fp + fn + tn, 1), "precision": prec, "recall": rec,
            "f1": f1_from_counts(tp, fp, fn)}


# ============================================================================
#  INFERENCE
# ============================================================================

def make_prob_fn(model: nn.Module, device: torch.device):
    """numpy [B,C,T] -> numpy probabilities [B,T,2] from the last stage."""

    @torch.no_grad()
    def fn(xb: np.ndarray) -> np.ndarray:
        model.eval()
        out = model(torch.from_numpy(xb).to(device))[-1]
        return F.softmax(out, dim=1).permute(0, 2, 1).float().cpu().numpy()

    return fn


def predict_full(prob_fn, x: np.ndarray) -> np.ndarray:
    return prob_fn(np.ascontiguousarray(x.T[None]))[0]


def predict_windows(prob_fn, x: np.ndarray, window: int, stride: int,
                    vote_weight: str = "uniform", batch: int = 256) -> np.ndarray:
    """Soft voting: weighted average of per-window probabilities over overlaps (P3)."""
    T, C = x.shape
    starts = window_starts(T, window, stride)
    if vote_weight == "center":
        w = 1.0 - np.abs(np.arange(window) - (window - 1) / 2) / (window / 2)
        w = np.maximum(w, 0.05)
    else:
        w = np.ones(window)
    acc = np.zeros((T, 2), dtype=np.float64)
    wsum = np.zeros(T, dtype=np.float64)
    for b in range(0, len(starts), batch):
        ss = starts[b:b + batch]
        xb = np.zeros((len(ss), C, window), dtype=np.float32)
        for i, s in enumerate(ss):
            seg = x[s:s + window]
            xb[i, :, :len(seg)] = seg.T
        pb = prob_fn(xb)
        for i, s in enumerate(ss):
            n = min(window, T - s)
            acc[s:s + n] += pb[i, :n] * w[:n, None]
            wsum[s:s + n] += w[:n]
    assert (wsum > 0).all(), "uncovered frames in window reconstruction"
    return (acc / wsum[:, None]).astype(np.float32)


def predict_sessions(cfg: Config, prob_fn, sessions: Sequence[Session]) -> List[np.ndarray]:
    mode = resolve_infer_mode(cfg)
    if mode == "full":
        return [predict_full(prob_fn, s.x) for s in sessions]
    return [predict_windows(prob_fn, s.x, cfg.window, cfg.stride, cfg.vote_weight) for s in sessions]


# ============================================================================
#  TRAINING
# ============================================================================

def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def log(cfg: Config, msg: str, level: int = 1) -> None:
    if cfg.verbose >= level:
        print(msg, flush=True)


def primary_view(cfg: Config, rule: Optional[str] = None) -> str:
    """Name of the result view used for monitoring, selection, fold summaries and the index."""
    rule = rule or cfg.seg_rule
    return ("or_merged" if cfg.or_merge else "per_hand") + ("_paperrule" if rule == "paper" else "")


def secondary_view(cfg: Config) -> str:
    return primary_view(cfg, "standard" if cfg.seg_rule == "paper" else "paper")


def selection_score(ev: Dict, ks: Sequence[float], key: str) -> float:
    return float(np.mean([f1_from_counts(**ev[key][f"k={k}"]) for k in ks]))


@torch.no_grad()
def monitor_eval(cfg: Config, model: nn.Module, criterion: nn.Module, sessions: Sequence[Session],
                 device: torch.device) -> Tuple[Dict, Dict, List[np.ndarray]]:
    """Eval-mode loss (frame-weighted over sessions) + segment/frame counts on `sessions`."""
    model.eval()
    sums = {"loss": 0.0, "cls": 0.0, "smooth": 0.0}
    n_frames = 0
    probs = []
    full = resolve_infer_mode(cfg) == "full"
    for s in sessions:
        if full:
            X, Y = collate([(s.x, s.y)])
            outs = model(X.to(device))
            loss, parts = criterion(outs, Y.to(device))
            w = len(s.y)
            sums["loss"] += loss.item() * w
            sums["cls"] += parts["cls"] * w
            sums["smooth"] += parts["smooth"] * w
            n_frames += w
            probs.append(F.softmax(outs[-1], dim=1)[0].T.float().cpu().numpy())
        else:
            Xw, Yw = create_sliding_windows(s.x, s.y, cfg.window, cfg.stride)
            for b in range(0, len(Xw), 256):
                X, Y = collate(list(zip(Xw[b:b + 256], Yw[b:b + 256])))
                loss, parts = criterion(model(X.to(device)), Y.to(device))
                w = int((Y != IGNORE).sum())
                sums["loss"] += loss.item() * w
                sums["cls"] += parts["cls"] * w
                sums["smooth"] += parts["smooth"] * w
                n_frames += w
            probs.append(predict_windows(make_prob_fn(model, device), s.x, cfg.window, cfg.stride,
                                         cfg.vote_weight))
    losses = {k: v / max(n_frames, 1) for k, v in sums.items()}
    return losses, evaluate_predictions(sessions, probs, cfg.iou_thresholds), probs


HISTORY_FIELDS_BASE = ["epoch", "train_loss", "train_cls", "train_smooth", "heldout_loss", "heldout_cls",
                       "heldout_smooth", "heldout_frameF1", "inner_loss", "inner_seg_f1", "lr", "seconds"]


def history_fields(cfg: Config) -> List[str]:
    f1 = [f"heldout_segF1_k{k}" for k in cfg.iou_thresholds]
    return HISTORY_FIELDS_BASE[:7] + f1 + HISTORY_FIELDS_BASE[7:]


def train_fold(cfg: Config, train: List[Session], inner: List[Session], device: torch.device,
               fold_seed: int, heldout: Sequence[Session] = (),
               history_csv: Optional[Path] = None) -> Tuple[nn.Module, List[Dict], Dict]:
    """Train one fold. `heldout` (the test fold) is only *monitored* (loss/F1 per epoch for the
    curves); it never influences training or model selection."""
    model = build_model(cfg, train[0].x.shape[1]).to(device)
    cw = compute_class_weights(cfg.class_weight, train)
    criterion = MSTCNLoss(cw.to(device) if cw is not None else None, cfg.smooth_weight, cfg.tau,
                          cfg.stage_reduce)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
                 if cfg.scheduler == "plateau" else None)
    rng = np.random.RandomState(fold_seed)
    windows_cache = None
    if cfg.train_mode == "windows":
        ws = [create_sliding_windows(s.x, s.y, cfg.window, cfg.stride) for s in train]
        windows_cache = (np.concatenate([w[0] for w in ws]), np.concatenate([w[1] for w in ws]))

    history, best = [], {"score": -1.0, "epoch": 0, "state": None}
    csv_file = writer = None
    if history_csv is not None:
        csv_file = open(history_csv, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=history_fields(cfg), restval="", extrasaction="ignore")
        writer.writeheader()
        csv_file.flush()
    view = primary_view(cfg)
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        items = epoch_items(cfg, train, rng, windows_cache)
        tot = cls = sm = 0.0
        nb = 0
        for b in range(0, len(items), cfg.batch_size):
            X, Y = collate(items[b:b + cfg.batch_size])
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            outs = model(X)
            loss, parts = criterion(outs, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            tot += loss.item()
            cls += parts["cls"]
            sm += parts["smooth"]
            nb += 1
        rec = {"epoch": epoch, "train_loss": tot / nb, "train_cls": cls / nb, "train_smooth": sm / nb,
               "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 2)}
        last = epoch == cfg.epochs
        if heldout and cfg.monitor_every > 0 and (epoch % cfg.monitor_every == 0 or last):
            hl, hev, _ = monitor_eval(cfg, model, criterion, heldout, device)
            rec.update({"heldout_loss": hl["loss"], "heldout_cls": hl["cls"], "heldout_smooth": hl["smooth"],
                        "heldout_frameF1": frame_scores(hev["frame"])["f1"]})
            for k in cfg.iou_thresholds:
                rec[f"heldout_segF1_k{k}"] = f1_from_counts(**hev[view][f"k={k}"])
        if inner and (epoch % cfg.eval_every == 0 or last):
            il, iev, _ = monitor_eval(cfg, model, criterion, inner, device)
            rec["inner_loss"] = il["loss"]
            rec["inner_seg_f1"] = selection_score(iev, cfg.iou_thresholds, view)
            if rec["inner_seg_f1"] > best["score"]:
                best = {"score": rec["inner_seg_f1"], "epoch": epoch,
                        "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
        if scheduler is not None:
            scheduler.step(rec.get("inner_loss", rec["train_loss"]))
        history.append(rec)
        if writer is not None:
            writer.writerow(rec)
            csv_file.flush()
        extra = ""
        if "heldout_loss" in rec:
            extra += f"  | heldout loss={rec['heldout_loss']:.4f} segF1=" + "/".join(
                f"{rec[f'heldout_segF1_k{k}']:.2f}" for k in cfg.iou_thresholds)
        if "inner_seg_f1" in rec:
            extra += f"  | inner segF1={rec['inner_seg_f1']:.3f}"
        log(cfg, f"    epoch {epoch:3d}/{cfg.epochs}  loss={rec['train_loss']:.4f} "
                 f"(cls={rec['train_cls']:.4f} smooth={rec['train_smooth']:.4f})  "
                 f"lr={rec['lr']:.1e}  {rec['seconds']:.1f}s{extra}")
    if csv_file is not None:
        csv_file.close()
    info = {"selected_epoch": cfg.epochs}
    if inner and best["state"] is not None:
        model.load_state_dict(best["state"])
        info = {"selected_epoch": best["epoch"], "inner_seg_f1": best["score"]}
    return model, history, info


def pick_inner_subject(train: List[Session], seed: int) -> str:
    """One training subject with drinking gestures, chosen reproducibly."""
    cands = sorted({s.subject for s in train if s.y.any()}, key=_subject_key)
    return cands[np.random.RandomState(seed).randint(len(cands))]


# ============================================================================
#  EXPERIMENT
# ============================================================================

def environment_info(device: torch.device) -> Dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
            "command": " ".join(sys.argv)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def aggregate(fold_results: List[Dict], ks: Sequence[float]) -> Dict:
    out = {"segment": {}, "frame": {}}
    for view in ("or_merged", "or_merged_paperrule", "per_hand", "per_hand_paperrule"):
        out["segment"][view] = {}
        for k in ks:
            key = f"k={k}"
            c = {m: int(sum(fr["eval"][view][key][m] for fr in fold_results)) for m in ("tp", "fp", "fn")}
            per_fold = [f1_from_counts(**fr["eval"][view][key]) for fr in fold_results
                        if sum(fr["eval"][view][key].values()) > 0]
            out["segment"][view][key] = {**c, "f1_pooled": f1_from_counts(**c),
                                         "precision": c["tp"] / max(c["tp"] + c["fp"], 1),
                                         "recall": c["tp"] / max(c["tp"] + c["fn"], 1),
                                         "f1_fold_mean": float(np.mean(per_fold)) if per_fold else 0.0,
                                         "f1_fold_std": float(np.std(per_fold)) if per_fold else 0.0}
    fc = {m: int(sum(fr["eval"]["frame"][m] for fr in fold_results)) for m in ("tp", "fp", "fn", "tn")}
    out["frame"] = {**fc, **frame_scores(fc)}
    out["n_gt_segments_or"] = int(sum(fr["eval"]["n_gt_segments_or"] for fr in fold_results))
    out["n_gt_segments_per_hand"] = int(sum(fr["eval"]["n_gt_segments_per_hand"] for fr in fold_results))
    return out


def format_summary(cfg: Config, summary: Dict) -> str:
    lines = [f"Experiment: {summary['name']}", f"Preset/arch: {cfg.arch} layer={cfg.layer_type} "
             f"stages={cfg.num_stages} layers={cfg.num_layers} features={cfg.num_features}",
             f"Folds: {summary['n_folds']} ({cfg.folds}), select={cfg.select}, train_mode={cfg.train_mode}, "
             f"epochs={cfg.epochs}", f"GT segments: OR-merged={summary['n_gt_segments_or']} "
             f"per-hand={summary['n_gt_segments_per_hand']} (paper DX-I: 410)", ""]
    for view, title in (
            ("or_merged_paperrule", "Segmental F1, OR-merged, paper's decision rules (compare with paper)"),
            ("or_merged", "Segmental F1, OR-merged, standard MS-TCN one-to-one matching (stricter)"),
            ("per_hand", "Segmental F1, per hand, standard matching")):
        lines.append(title)
        show_ref = view.startswith("or_merged")
        lines.append(f"  {'k':<6}{'TP':>6}{'FP':>6}{'FN':>6}{'F1 pooled':>11}{'fold mean+-std':>18}"
                     + ("   paper TP/FP/FN F1" if show_ref else ""))
        for k in cfg.iou_thresholds:
            r = summary["segment"][view][f"k={k}"]
            ref = PAPER_DXI.get(k)
            refs = f"   {ref[0]}/{ref[1]}/{ref[2]} {ref[3]:.3f}" if (ref and show_ref) else ""
            lines.append(f"  {k:<6}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}{r['f1_pooled']:>11.4f}"
                         f"{r['f1_fold_mean']:>11.4f}+-{r['f1_fold_std']:.4f}{refs}")
        lines.append("")
    fr = summary["frame"]
    lines.append(f"Frame-wise (per hand): acc={fr['accuracy']:.4f} P={fr['precision']:.4f} "
                 f"R={fr['recall']:.4f} F1={fr['f1']:.4f}")
    return "\n".join(lines)


def append_fold_summary(path: Path, cfg: Config, fr: Dict, view: str) -> None:
    """One row per finished fold, written as soon as the fold ends (check progress mid-run)."""
    ev = fr["eval"]
    last = fr["history"][-1] if fr["history"] else {}
    row = {"fold": fr["fold"], "test_subjects": " ".join(fr["test_subjects"]), "seg_rule": cfg.seg_rule,
           "n_gt_segments": ev["n_gt_segments_or"] if view.startswith("or_merged") else ev["n_gt_segments_per_hand"],
           "selected_epoch": fr["selection"].get("selected_epoch"),
           "final_train_loss": round(last.get("train_loss", float("nan")), 5),
           "final_heldout_loss": round(last.get("heldout_loss", float("nan")), 5),
           "frame_f1": round(frame_scores(ev["frame"])["f1"], 4), "seconds": fr["seconds"]}
    for k in cfg.iou_thresholds:
        c = ev[view][f"k={k}"]
        row.update({f"tp@{k}": c["tp"], f"fp@{k}": c["fp"], f"fn@{k}": c["fn"],
                    f"segF1@{k}": round(f1_from_counts(**c), 4)})
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


INDEX_CONFIG_KEYS = ["preset", "arch", "layer_type", "inception_stages", "inception_kernels", "num_stages",
                     "num_layers", "num_features", "channels", "train_mode", "epochs", "batch_size", "lr",
                     "smooth_weight", "tau", "class_weight", "folds", "select", "seed"]


def append_experiment_index(index_csv: Path, cfg: Config, summary: Dict, derived: Dict, out: Path) -> None:
    """One row per finished run in results/experiments_index.csv (the experiment tracker)."""
    view, other = primary_view(cfg), secondary_view(cfg)
    row = {"experiment": summary["name"], "finished": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "tag": cfg.tag, **{k: (",".join(map(str, v)) if isinstance(v, tuple) else v)
                              for k, v in asdict(cfg).items() if k in INDEX_CONFIG_KEYS},
           "n_params": derived["n_parameters"], "rf_total": derived["receptive_field"]["total"],
           "n_folds_run": summary["n_folds"], "seg_rule": cfg.seg_rule, "view": view}
    for k in cfg.iou_thresholds:
        r = summary["segment"][view][f"k={k}"]
        row.update({f"segF1@{k}": round(r["f1_pooled"], 4), f"tp@{k}": r["tp"], f"fp@{k}": r["fp"],
                    f"fn@{k}": r["fn"]})
    for k in cfg.iou_thresholds:
        row[f"segF1_{'standard' if cfg.seg_rule == 'paper' else 'paperrule'}@{k}"] = round(
            summary["segment"][other][f"k={k}"]["f1_pooled"], 4)
    row.update({"frame_f1": round(summary["frame"]["f1"], 4), "minutes": round(summary["seconds"] / 60, 1),
                "folder": str(out)})
    new = not index_csv.exists()
    if not new:
        with open(index_csv, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != list(row):        # columns changed (e.g. other k values): start a new index file
            index_csv = index_csv.with_name(index_csv.stem + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
                                            + ".csv")
            new = True
    with open(index_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


class _Tee:
    """Duplicate stdout into the run's train.log (so background runs can be followed)."""

    def __init__(self, stream, path: Path):
        self.stream, self.file = stream, open(path, "a", encoding="utf-8", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def run_experiment(cfg: Config, sessions: Optional[List[Session]] = None,
                   out_dir: Optional[Path] = None) -> Dict:
    validate_config(cfg)
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{cfg.tag}" if cfg.tag else "")
    out = Path(out_dir) if out_dir is not None else Path(cfg.results_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    tee = _Tee(sys.stdout, out / "train.log")
    sys.stdout = tee
    try:
        return _run_experiment(cfg, sessions, out, name, out_dir is None)
    finally:
        sys.stdout = tee.stream
        tee.close()


def _run_experiment(cfg: Config, sessions: Optional[List[Session]], out: Path, name: str,
                    track: bool) -> Dict:
    set_seed(cfg.seed, cfg.deterministic)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto"
                          else cfg.device)
    span_warning = check_dilation_span(cfg)
    if sessions is None:
        sessions = load_sessions(cfg)
    folds = make_folds(sessions, cfg.folds, cfg.seed)
    if cfg.max_folds > 0:
        folds = folds[:cfg.max_folds]

    for sub in ("folds", "predictions", "models", "plots"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in build_model(cfg, sessions[0].x.shape[1]).parameters())
    rf = receptive_field(cfg)
    derived = {"receptive_field": rf, "n_parameters": n_params, "infer_mode": resolve_infer_mode(cfg),
               "span_warning": span_warning, "n_sessions": len(sessions),
               "n_subjects": len({s.subject for s in sessions}), "n_folds_run": len(folds)}
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "derived": derived, "env": environment_info(device)}, f, indent=2)

    log(cfg, f"== started {datetime.now():%Y-%m-%d %H:%M:%S}  command: {' '.join(sys.argv)}")
    log(cfg, f"== {name}  device={device}  params={n_params:,}  RF/stage={rf['per_stage']} "
             f"total={rf['total']}  folds={len(folds)}")
    if span_warning:
        log(cfg, f"   WARNING: {span_warning}")

    fold_results = []
    t_start = time.time()
    for fi, fold in enumerate(folds):
        fold_seed = cfg.seed + 1000 * (fi + 1)
        set_seed(fold_seed, cfg.deterministic)
        train = [sessions[i] for i in fold["train"]]
        test = [sessions[i] for i in fold["test"]]
        inner = []
        if cfg.select == "inner":
            inner_subj = pick_inner_subject(train, fold_seed)
            inner = [s for s in train if s.subject == inner_subj]
            train = [s for s in train if s.subject != inner_subj]
        log(cfg, f"-- fold {fi + 1}/{len(folds)}  test subjects={fold['test_subjects']}  "
                 f"train={len(train)} sess  test={len(test)} sess"
                 + (f"  inner={inner[0].subject}" if inner else ""))
        t0 = time.time()
        model, history, info = train_fold(cfg, train, inner, device, fold_seed, heldout=test,
                                          history_csv=out / "folds" / f"fold_{fi + 1:02d}_history.csv")
        probs = predict_sessions(cfg, make_prob_fn(model, device), test)
        ev = evaluate_predictions(test, probs, cfg.iou_thresholds)
        fr = {"fold": fi + 1, "test_subjects": fold["test_subjects"],
              "test_sessions": [s.name for s in test], "train_sessions": [s.name for s in train],
              "inner_sessions": [s.name for s in inner], "selection": info, "eval": ev,
              "history": history, "seconds": round(time.time() - t0, 1)}
        fold_results.append(fr)
        with open(out / "folds" / f"fold_{fi + 1:02d}.json", "w", encoding="utf-8") as f:
            json.dump(fr, f, indent=2)
        np.savez_compressed(out / "predictions" / f"fold_{fi + 1:02d}.npz",
                            **{f"{s.recording}_{s.hand}_prob1": p[:, 1].astype(np.float16)
                               for s, p in zip(test, probs)},
                            **{f"{s.recording}_{s.hand}_gt": s.y.astype(np.int8) for s in test})
        if cfg.save_models:
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "fold": fi + 1},
                       out / "models" / f"fold_{fi + 1:02d}.pt")
        key = primary_view(cfg)
        msg = "  ".join(f"k={k}: {ev[key][f'k={k}']['tp']}/{ev[key][f'k={k}']['fp']}/"
                        f"{ev[key][f'k={k}']['fn']}" for k in cfg.iou_thresholds)
        log(cfg, f"   fold {fi + 1} TP/FP/FN ({key}) {msg}   {fr['seconds']:.0f}s")
        append_fold_summary(out / "folds_summary.csv", cfg, fr, key)
        if cfg.plots:
            try:
                import plot_results
                plot_results.plot_fold(out / "folds" / f"fold_{fi + 1:02d}_history.csv",
                                       out / "plots" / f"fold_{fi + 1:02d}.png",
                                       title=f"Fold {fi + 1} - test subject(s) {', '.join(fold['test_subjects'])}"
                                             f"  (segmental F1: {cfg.seg_rule} rule)")
            except Exception as e:  # plotting must never kill a run
                log(cfg, f"   (plot failed: {e})")

    if cfg.plots:
        try:
            import plot_results
            plot_results.plot_all_folds(out)
        except Exception as e:
            log(cfg, f"   (plot failed: {e})")
    summary = {"name": name, "n_folds": len(folds), "seconds": round(time.time() - t_start, 1),
               **aggregate(fold_results, cfg.iou_thresholds)}
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    text = format_summary(cfg, summary)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    if track:                      # only real runs go into the shared index (not tests)
        append_experiment_index(Path(cfg.results_dir) / "experiments_index.csv", cfg, summary, derived, out)
    log(cfg, "\n" + text + f"\n\nResults: {out}")
    summary["out_dir"] = str(out)
    return summary


# ============================================================================
#  CLI
# ============================================================================

def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="paper", choices=sorted(PRESETS))
    for f in dataclasses.fields(Config):
        if f.name == "preset":
            continue
        if f.type in ("bool", bool):
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=None)
        elif f.name in TUPLE_INT_FIELDS | TUPLE_FLOAT_FIELDS:
            p.add_argument(f"--{f.name}", type=lambda s: tuple(x for x in s.split(",") if x), default=None)
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type if isinstance(f.type, str) else
                                                               f.type.__name__, str)
            p.add_argument(f"--{f.name}", type=typ, default=None)
    a = vars(p.parse_args(argv))
    preset = a.pop("preset")
    return build_config(preset, **{k: v for k, v in a.items() if v is not None})


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
