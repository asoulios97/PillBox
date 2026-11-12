"""
Sanity check script to verify that mirroring and downsampling work correctly.
Visualizes the transformations step by step.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import functions from prepromax
from prepromax import (
    apply_hand_mirroring,
    apply_antialias_lowpass,
    downsample_df,
    RAW_FS,
    TARGET_FS
)

# Define paths
base_path = Path(
    r"C:\Users\Apost\Desktop\Διπλωματική\others\open datasets\drinking gesture identifications.open\dataverse_files\DX-I\DX-I\Raw data and Annotation"
)

print("="*70)
print("SANITY CHECK: Mirroring and Downsampling Transformations")
print("="*70)

# Function to find continuous regions where anno == 1
def find_anno_regions(df):
    """Find start and end times of regions where anno == 1"""
    anno = df['anno'].values
    t_seconds = df['t_seconds'].values
    
    regions = []
    in_region = False
    start_time = None
    
    for i, (a, t) in enumerate(zip(anno, t_seconds)):
        if a == 1 and not in_region:
            # Start of annotation region
            start_time = t
            in_region = True
        elif a == 0 and in_region:
            # End of annotation region
            regions.append((start_time, t_seconds[i-1]))
            in_region = False
    
    # Handle case where annotation extends to the end
    if in_region:
        regions.append((start_time, t_seconds[-1]))
    
    return regions

# Load data for user 1-3
print("\n1. Loading raw data for User 1-3...")
df_hand_left = pd.read_csv(base_path / "Left hand" / "1-3.csv")
df_hand_right = pd.read_csv(base_path / "Right hand" / "1-3.csv")

print(f"   Left hand shape: {df_hand_left.shape}")
print(f"   Right hand shape: {df_hand_right.shape}")

# Convert time to seconds
df_hand_left['t_seconds'] = pd.to_timedelta(df_hand_left['t']).dt.total_seconds()
df_hand_right['t_seconds'] = pd.to_timedelta(df_hand_right['t']).dt.total_seconds()

# ========== STEP 1: Apply Mirroring to Left Hand ==========
print("\n2. Applying hand mirroring transformation to Left hand...")
df_hand_left_mirrored = apply_hand_mirroring(df_hand_left.copy())

print(f"   ✓ Left hand mirrored (ax, gy, gz multiplied by -1)")
print(f"\n   Verification:")
print(f"   - ax_mirrored[0] = {df_hand_left_mirrored['ax'].iloc[0]:.4f} (was {df_hand_left['ax'].iloc[0]:.4f})")
print(f"   - ay_mirrored[0] = {df_hand_left_mirrored['ay'].iloc[0]:.4f} (was {df_hand_left['ay'].iloc[0]:.4f}) [unchanged]")
print(f"   - gy_mirrored[0] = {df_hand_left_mirrored['gy'].iloc[0]:.4f} (was {df_hand_left['gy'].iloc[0]:.4f})")

# ========== STEP 2: Apply Anti-aliasing ==========
print("\n3. Applying anti-aliasing filter...")
df_left_filtered = apply_antialias_lowpass(df_hand_left_mirrored.copy(), fs=RAW_FS, cutoff=7.5, order=4)
df_right_filtered = apply_antialias_lowpass(df_hand_right.copy(), fs=RAW_FS, cutoff=7.5, order=4)
print(f"   ✓ Anti-aliasing applied (Butterworth low-pass, cutoff=7.5Hz)")

# ========== STEP 3: Apply Downsampling ==========
print("\n4. Applying downsampling (64Hz → 16Hz)...")
df_left_downsampled = downsample_df(df_left_filtered, label_reduce="any")
df_right_downsampled = downsample_df(df_right_filtered, label_reduce="any")

print(f"   Left hand:  {len(df_hand_left):,} → {len(df_left_downsampled):,} samples (factor: {len(df_hand_left)/len(df_left_downsampled):.2f})")
print(f"   Right hand: {len(df_hand_right):,} → {len(df_right_downsampled):,} samples (factor: {len(df_hand_right)/len(df_right_downsampled):.2f})")

# Convert downsampled time to seconds
df_left_downsampled['t_seconds'] = pd.to_timedelta(df_left_downsampled['t']).dt.total_seconds()
df_right_downsampled['t_seconds'] = pd.to_timedelta(df_right_downsampled['t']).dt.total_seconds()

# Find annotation regions for downsampled data
left_anno_regions = find_anno_regions(df_left_downsampled)
right_anno_regions = find_anno_regions(df_right_downsampled)

print(f"   Left hand: {len(left_anno_regions)} annotation regions (drinking gestures)")
print(f"   Right hand: {len(right_anno_regions)} annotation regions (drinking gestures)")

# ========== VISUALIZATION ==========
print("\n5. Creating visualizations...")

# Figure 1: Before and After Mirroring (Left Hand)
fig1, axes = plt.subplots(3, 1, figsize=(16, 10))
fig1.suptitle('MIRRORING TRANSFORMATION - Left Hand (User 1-3)', fontsize=14, fontweight='bold')

# Show first 2000 samples for clarity
n_show = 2000

# ax - should be flipped
axes[0].plot(df_hand_left['t_seconds'].iloc[:n_show], 
             df_hand_left['ax'].iloc[:n_show], 
             linewidth=0.5, color='blue', alpha=0.7, label='Original Left ax')
axes[0].plot(df_hand_left_mirrored['t_seconds'].iloc[:n_show], 
             df_hand_left_mirrored['ax'].iloc[:n_show], 
             linewidth=0.5, color='red', alpha=0.7, label='Mirrored Left ax (×-1)')
axes[0].set_title('ax (MIRRORED: multiplied by -1)', fontweight='bold')
axes[0].set_ylabel('ax (m/s²)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ay - should be unchanged
axes[1].plot(df_hand_left['t_seconds'].iloc[:n_show], 
             df_hand_left['ay'].iloc[:n_show], 
             linewidth=0.5, color='blue', alpha=0.7, label='Original Left ay')
axes[1].plot(df_hand_left_mirrored['t_seconds'].iloc[:n_show], 
             df_hand_left_mirrored['ay'].iloc[:n_show], 
             linewidth=0.5, color='red', alpha=0.7, linestyle='--', label='Mirrored Left ay (unchanged)')
axes[1].set_title('ay (UNCHANGED)', fontweight='bold')
axes[1].set_ylabel('ay (m/s²)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# gy - should be flipped
axes[2].plot(df_hand_left['t_seconds'].iloc[:n_show], 
             df_hand_left['gy'].iloc[:n_show], 
             linewidth=0.5, color='blue', alpha=0.7, label='Original Left gy')
axes[2].plot(df_hand_left_mirrored['t_seconds'].iloc[:n_show], 
             df_hand_left_mirrored['gy'].iloc[:n_show], 
             linewidth=0.5, color='red', alpha=0.7, label='Mirrored Left gy (×-1)')
axes[2].set_title('gy (MIRRORED: multiplied by -1)', fontweight='bold')
axes[2].set_ylabel('gy (°/s)')
axes[2].set_xlabel('Time (seconds)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# Figure 2: Before and After Downsampling
fig2, axes = plt.subplots(2, 1, figsize=(16, 10))
fig2.suptitle('DOWNSAMPLING TRANSFORMATION - 64Hz → 16Hz (User 1-3)', fontsize=14, fontweight='bold')

# Left Hand
axes[0].plot(df_hand_left_mirrored['t_seconds'].iloc[:n_show], 
             df_hand_left_mirrored['ax'].iloc[:n_show], 
             linewidth=0.5, color='blue', alpha=0.5, label=f'Left Hand @64Hz ({len(df_hand_left):,} samples)')
axes[0].plot(df_left_downsampled['t_seconds'].iloc[:n_show//4], 
             df_left_downsampled['ax'].iloc[:n_show//4], 
             linewidth=1.5, color='darkblue', marker='o', markersize=2, 
             label=f'Left Hand @16Hz ({len(df_left_downsampled):,} samples)', alpha=0.8)
axes[0].set_title('Left Hand ax - Downsampling Effect', fontweight='bold')
axes[0].set_ylabel('ax (m/s²)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right Hand
axes[1].plot(df_hand_right['t_seconds'].iloc[:n_show], 
             df_hand_right['ax'].iloc[:n_show], 
             linewidth=0.5, color='red', alpha=0.5, label=f'Right Hand @64Hz ({len(df_hand_right):,} samples)')
axes[1].plot(df_right_downsampled['t_seconds'].iloc[:n_show//4], 
             df_right_downsampled['ax'].iloc[:n_show//4], 
             linewidth=1.5, color='darkred', marker='o', markersize=2,
             label=f'Right Hand @16Hz ({len(df_right_downsampled):,} samples)', alpha=0.8)
axes[1].set_title('Right Hand ax - Downsampling Effect', fontweight='bold')
axes[1].set_ylabel('ax (m/s²)')
axes[1].set_xlabel('Time (seconds)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# Figure 3: Final Comparison - Both Hands After All Transformations WITH ANNOTATIONS
fig3, axes = plt.subplots(2, 1, figsize=(16, 10))
fig3.suptitle('FINAL RESULT - Both Hands After Mirroring & Downsampling with Annotations (User 1-3)', 
              fontsize=14, fontweight='bold')

# Left hand with annotation regions
axes[0].plot(df_left_downsampled['t_seconds'], 
             df_left_downsampled['ax'], 
             linewidth=0.5, color='navy', alpha=0.7, label='ax signal')

# Add annotation regions (drinking gestures)
for start, end in left_anno_regions:
    axes[0].axvspan(start, end, alpha=0.3, color='yellow', label='Drinking gesture (anno=1)')

# Remove duplicate labels in legend
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axes[0].legend(by_label.values(), by_label.keys(), loc='upper right')

axes[0].set_title('Left Hand (Mirrored) - Acceleration X @16Hz with Annotations', 
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('ax (m/s²)')
axes[0].grid(True, alpha=0.3)

# Right hand with annotation regions
axes[1].plot(df_right_downsampled['t_seconds'], 
             df_right_downsampled['ax'], 
             linewidth=0.5, color='darkred', alpha=0.7, label='ax signal')

# Add annotation regions (drinking gestures)
for start, end in right_anno_regions:
    axes[1].axvspan(start, end, alpha=0.3, color='yellow', label='Drinking gesture (anno=1)')

# Remove duplicate labels in legend
handles, labels = axes[1].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axes[1].legend(by_label.values(), by_label.keys(), loc='upper right')

axes[1].set_title('Right Hand - Acceleration X @16Hz with Annotations', 
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('ax (m/s²)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# Figure 4: Both on same plot with annotations
fig4, ax = plt.subplots(figsize=(16, 7))

# Plot signals
ax.plot(df_left_downsampled['t_seconds'], 
        df_left_downsampled['ax'], 
        linewidth=0.5, color='blue', alpha=0.6, label='Left Hand (Mirrored)')
ax.plot(df_right_downsampled['t_seconds'], 
        df_right_downsampled['ax'], 
        linewidth=0.5, color='red', alpha=0.6, label='Right Hand')

# Add annotation regions for left hand (cyan)
for start, end in left_anno_regions:
    ax.axvspan(start, end, alpha=0.2, color='cyan', label='Left drinking gesture')

# Add annotation regions for right hand (orange)
for start, end in right_anno_regions:
    ax.axvspan(start, end, alpha=0.2, color='orange', label='Right drinking gesture')

# Remove duplicate labels
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right')

ax.set_title('Both Hands - Acceleration X @16Hz with Drinking Gesture Annotations', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('ax (m/s²)')
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

# ========== STATISTICS ==========
print("\n" + "="*70)
print("STATISTICS")
print("="*70)

print("\nOriginal Data (64Hz):")
print(f"  Left hand  - ax range: [{df_hand_left['ax'].min():.2f}, {df_hand_left['ax'].max():.2f}]")
print(f"  Right hand - ax range: [{df_hand_right['ax'].min():.2f}, {df_hand_right['ax'].max():.2f}]")
print(f"  Left hand  - Duration: {df_hand_left['t_seconds'].max():.2f} seconds")
print(f"  Right hand - Duration: {df_hand_right['t_seconds'].max():.2f} seconds")

print("\nAfter Mirroring (64Hz):")
print(f"  Left hand  - ax range: [{df_hand_left_mirrored['ax'].min():.2f}, {df_hand_left_mirrored['ax'].max():.2f}] (flipped)")
print(f"  Left hand  - ay range: [{df_hand_left_mirrored['ay'].min():.2f}, {df_hand_left_mirrored['ay'].max():.2f}] (unchanged)")

print("\nAfter Downsampling (16Hz):")
print(f"  Left hand  - ax range: [{df_left_downsampled['ax'].min():.2f}, {df_left_downsampled['ax'].max():.2f}]")
print(f"  Right hand - ax range: [{df_right_downsampled['ax'].min():.2f}, {df_right_downsampled['ax'].max():.2f}]")
print(f"  Left hand  - Duration: {df_left_downsampled['t_seconds'].max():.2f} seconds")
print(f"  Right hand - Duration: {df_right_downsampled['t_seconds'].max():.2f} seconds")
print(f"  Left hand  - Samples: {len(df_left_downsampled):,}")
print(f"  Right hand - Samples: {len(df_right_downsampled):,}")

print("\nAnnotations:")
print(f"  Left hand  - anno=1: {df_hand_left['anno'].sum():,} / {len(df_hand_left):,} ({df_hand_left['anno'].mean()*100:.2f}%)")
print(f"  Right hand - anno=1: {df_hand_right['anno'].sum():,} / {len(df_hand_right):,} ({df_hand_right['anno'].mean()*100:.2f}%)")
print(f"  Left hand (downsampled)  - anno=1: {df_left_downsampled['anno'].sum():,} / {len(df_left_downsampled):,} ({df_left_downsampled['anno'].mean()*100:.2f}%)")
print(f"  Right hand (downsampled) - anno=1: {df_right_downsampled['anno'].sum():,} / {len(df_right_downsampled):,} ({df_right_downsampled['anno'].mean()*100:.2f}%)")

print("\n" + "="*70)
print("SANITY CHECK COMPLETE!")
print("="*70)
print("\nVerification Points:")
print("  ✓ Mirroring: Check that ax values are flipped for left hand")
print("  ✓ Mirroring: Check that ay values are UNCHANGED for left hand")
print("  ✓ Mirroring: Check that gy, gz values are flipped for left hand")
print("  ✓ Downsampling: Check that sample count reduced by ~4x (64Hz → 16Hz)")
print("  ✓ Downsampling: Check that signal shape is preserved")
print("  ✓ Labels: Check that annotation labels are preserved appropriately")
print("="*70)

