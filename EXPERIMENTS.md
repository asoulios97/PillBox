# Experiments: MS-TCN vs Inception MS-TCN on DX-I

The thesis experiment log. Each row changes **one thing** relative to its parent row, so its effect can be attributed. All runs share the same setup, as shown below. Results are filled in after each run from `code/results/experiments_index.csv`; the run folder holds the full details.

## Common protocol (all rows)
| Item | Setting |
|---|---|
| Data | DX-I, 13 subjects, 15 recordings × 2 hands; 405 drinking gestures after left/right OR merge |
| Preprocessing | Left hand mirrored (ax, gy, gz × −1); 4th-order zero-phase Butterworth low-pass at 7.5 Hz; decimation 64 → 16 Hz; per-session z-score; 6 channels unless the row says otherwise |
| Evaluation | Leave-one-subject-out (13 folds); left/right predictions OR-merged per recording; TP/FP/FN pooled over folds |
| Metric | Segmental F1 = 2TP / (2TP + FP + FN) at IoU k = 0.1 / 0.25 / 0.5, using the **paper's matching rule** (Wang et al. 2022, Eq. 8). The stricter standard MS-TCN rule is reported as a secondary column. |
| Training | Adam, lr 5e-4, 50 epochs, whole sessions, batch 1; CE + 0.15 × T-MSE (tau 4), summed over stages; no class weights; reported model = last epoch; seed 42 |
| Hardware | NVIDIA RTX 4060 Ti 16 GB |

## Reference (paper, DX-I, MS-TCN, LOSO)
| k | TP | FP | FN | Segmental F1 |
|---|---|---|---|---|
| 0.1 | 401 | 28 | 9 | 0.956 |
| 0.25 | 400 | 29 | 9 | 0.955 |
| 0.5 | 391 | 32 | 15 | 0.943 |

## Planned rows
| Row | Parent | Change | Command (from `code/`) | Question | Est. time | Status |
|---|---|---|---|---|---|---|
| **A** | — | Paper baseline: MS-TCN, 2 stages × 9 dilated layers, 128 filters | `mstcn_pipeline.py --preset paper --tag A_paper` | Do we reproduce the paper? | ~40 min | planned |
| **B** | A | Every layer → Inception dilated layer (bottleneck 32, kernels 3/5/9, max-pool branch) | `mstcn_pipeline.py --preset paper_inception --tag B_inception` | Do Inception layers help? | ~50 min | planned |
| C | A | Inception layers in stage 1 only | `mstcn_pipeline.py --preset paper_inception_s1 --tag C_inception_s1` | Where do Inception layers matter? | ~45 min | planned |
| D | B | Longer kernels 3/9/17 | `... --preset paper_inception --inception_kernels 3,9,17 --tag D_kernels` | Do longer kernels help (InceptionTime)? | ~60 min | planned |
| E | B | No max-pool branch | `... --preset paper_inception --no-inception_maxpool --tag E_nomaxpool` | Does the max-pool branch contribute? | ~50 min | planned |
| F | best of A–E | 3 stages | `... <best> --num_stages 3 --tag F_3stages` | Does an extra refinement stage help? | TBD | planned |
| G | best | 9 channels (+ acc/gyro magnitude, jerk) | `... <best> --channels 9 --tag G_9ch` | Do the extra features help? | TBD | planned |
| H | best | sqrt-inverse class weights | `... <best> --class_weight sqrt_inverse --tag H_weights` | Fewer misses vs more false alarms? | TBD | planned |

Fallbacks if A falls clearly short of the paper (one at a time, each its own row A1, A2, …): `--epochs 100`, `--train_mode crops --crop_len 4096 --batch_size 8`, `--select inner`.

## Results (paper matching rule, OR-merged, pooled over 13 folds)
| Row | Params | TP/FP/FN @0.1 | F1 @0.1 | TP/FP/FN @0.25 | F1 @0.25 | TP/FP/FN @0.5 | F1 @0.5 | F1 @0.5 standard rule | Fold mean ± std F1 @0.5 | Run folder |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 1.19 M | | | | | | | | | |
| B | 0.76 M | | | | | | | | | |

## Notes per run
<!-- After each run: date, anything unusual (outlier subjects, curve behaviour), and the figure paths
     (plots/loss_all_folds.pdf, plots/fold_XX.pdf) to use in the thesis. -->
