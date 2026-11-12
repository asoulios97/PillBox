"""
plot_results.py -- training curves for a mstcn_pipeline.py results folder.

Called automatically by mstcn_pipeline.py after every fold and at the end of a run.
Can also be run by hand (e.g. while a run is still going):
    .venv/Scripts/python.exe plot_results.py results/<experiment_folder>

Writes into <experiment>/plots/:
    fold_XX.png/.pdf   loss (train vs held-out) + held-out segmental F1 per epoch
    loss_all_folds.png/.pdf every fold's loss curves (thin) and the mean (bold)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Palette (validated: categorical slots 1-3 on light surface) and chart chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _save(fig, out_png: Path) -> None:
    """PNG for quick viewing + PDF (vector) for the thesis."""
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    fig.savefig(Path(out_png).with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)


def read_history(path: Path) -> Dict[str, List]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols: Dict[str, List] = {k: [] for k in (rows[0].keys() if rows else [])}
    for r in rows:
        for k, v in r.items():
            cols[k].append(float(v) if v not in ("", None) else None)
    return cols


def _xy(h: Dict[str, List], key: str):
    pts = [(e, v) for e, v in zip(h["epoch"], h.get(key, [])) if v is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_fold(history_csv: Path, out_png: Path, title: str = "") -> None:
    h = read_history(history_csv)
    if not h.get("epoch"):
        return
    f1_keys = [k for k in h if k.startswith("heldout_segF1_k")]
    fig, axes = plt.subplots(1, 2 if f1_keys else 1, figsize=(11 if f1_keys else 6, 3.8), squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0][0]
    for key, label, color in (("train_loss", "train (dropout on)", SERIES[0]),
                              ("heldout_loss", "held-out fold (eval mode)", SERIES[1])):
        x, y = _xy(h, key)
        if x:
            ax.plot(x, y, color=color, linewidth=2, label=label,
                    marker="o" if len(x) < 15 else None, markersize=4)
    _style(ax, "Loss", "CE + lambda * T-MSE")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    if f1_keys:
        ax = axes[0][1]
        for i, key in enumerate(f1_keys):
            x, y = _xy(h, key)
            if not x:
                continue
            k = key.replace("heldout_segF1_k", "")
            ax.plot(x, y, color=SERIES[i % len(SERIES)], linewidth=2, label=f"F1@{k}",
                    marker="o" if len(x) < 15 else None, markersize=4)
        ax.set_ylim(0, 1.02)
        _style(ax, "Held-out segmental F1 (monitoring only)", "segmental F1")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")

    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, out_png)


def plot_all_folds(exp_dir: Path) -> None:
    files = sorted((exp_dir / "folds").glob("fold_*_history.csv"))
    if not files:
        return
    hs = [read_history(f) for f in files]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(SURFACE)
    for key, label, color in (("train_loss", "train", SERIES[0]), ("heldout_loss", "held-out", SERIES[1])):
        curves = [_xy(h, key) for h in hs]
        curves = [c for c in curves if c[0]]
        if not curves:
            continue
        for x, y in curves:
            ax.plot(x, y, color=color, linewidth=0.8, alpha=0.25)
        # mean over folds at epochs present in every fold
        common = sorted(set.intersection(*[set(x) for x, _ in curves]))
        if common:
            means = [sum(dict(zip(x, y))[e] for x, y in curves) / len(curves) for e in common]
            ax.plot(common, means, color=color, linewidth=2.2, label=f"{label} (mean of {len(curves)} folds)")
    _style(ax, f"Loss per fold - {exp_dir.name}", "CE + lambda * T-MSE")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    (exp_dir / "plots").mkdir(exist_ok=True)
    _save(fig, exp_dir / "plots" / "loss_all_folds.png")


def plot_experiment(exp_dir: Path) -> None:
    exp_dir = Path(exp_dir)
    for f in sorted((exp_dir / "folds").glob("fold_*_history.csv")):
        n = f.name.split("_")[1]
        plot_fold(f, exp_dir / "plots" / f"fold_{n}.png", title=f"Fold {int(n)}")
    plot_all_folds(exp_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: plot_results.py <experiment_folder>")
    plot_experiment(Path(sys.argv[1]))
    print(f"plots written to {Path(sys.argv[1]) / 'plots'}")
