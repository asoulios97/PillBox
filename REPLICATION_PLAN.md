# Replication & enhancement plan: MS-TCN drinking gesture detection (DX-I)

Audience: an engineering agent working on this repo. Read this file and `HAND_OVER.md` fully before you change anything.
Last updated: 2026-11-12.

## 1. Goal
1. **Replicate** Wang et al., EMBC 2022 (the PDF in the repo root) on DX-I: a 2-stage MS-TCN, scored with segmental F1 at k = 0.1, 0.25 and 0.5. The paper reports **0.956 / 0.955 / 0.943**, from TP/FP/FN of 401/28/9, 400/29/9 and 391/32/15.
2. **Minor enhancement (the thesis contribution):** replace the plain dilated layers of that MS-TCN with **Inception dilated layers** (ideas from TCN-Inception and InceptionTime; papers in `tcn inception/`). Keep everything else identical.
3. Produce an **ablation table** under one protocol, changing one thing per row, plus **figures of the training process** for the thesis report.

## 2. Paper setup (the baseline to match)
| Item | Value |
|---|---|
| Input | 6 channels (ax ay az gx gy gz) at 16 Hz. Left hand mirrored by negating ax, gy, gz. |
| Model | 2 stages × 9 dilated residual layers (dilation 1..256), 128 filters, kernel 3, dropout 0.3, non-causal. Receptive field 1023 samples per stage. |
| Refinement input | Softmax of the previous stage only |
| Loss | Per stage: CE + 0.15 × T-MSE (tau = 4), summed over stages. No class weights. |
| Optimizer | Adam, lr 5e-4. Epochs and batch size are not stated; the MS-TCN default is 50 epochs, batch 1. |
| Protocol | Leave-one-subject-out over the 13 DX-I subjects. Left/right predictions are OR-merged per recording. TP/FP/FN are pooled over folds. |
| Matching rule (Sec. II-E, Eq. 8) | A weak match (IoU < k) counts as **one** error: FP if the ground truth is shorter than the prediction, otherwise FN. That's why TP+FN < 410 in their Table I. |
| Not stated by the paper | Filtering, normalization, downsampling method, epochs, batch size, model selection |

## 3. Data facts (verified)
- **Data files:** `code/DX_I_{X,Y,meta}_16Hz_with_jerk.pkl` hold 30 sessions (15 recordings × 2 hands).
  - They are produced by `code/prepromax.py` in this order: mirror left hand → 4th-order zero-phase Butterworth low-pass at 7.5 Hz → keep every 4th sample (64 → 16 Hz) → labels reduced with "any" → magnitude and jerk channels → per-session z-score.
  - They are git-ignored. Regenerate with `cd code && .venv/Scripts/python.exe prepromax.py`, which needs the raw `dataverse_files/DX-I` and has hardcoded absolute paths.
- **Channels:** `ax ay az gx gy gz acc_mag gyro_mag jerk_mag`. The paper uses only the first 6 (`--channels 6`, the default).
- **Mirroring:** the left hand is already mirrored in the pkls. It was checked to match the dataset's own Hand-Mirrored CSVs exactly.
- **IDs:** `meta['user_id']` is a **recording** id (`2-1`, `2-2`); the **subject** is the prefix (`2`). Folds group by subject, giving 13 LOSO folds.
- **Hand alignment:** both hands of a recording start at t=0 but end a few seconds apart. They are truncated to the shorter length for the OR merge.
- **Labels:** {0,1}, 3.75% positive overall. Six hand-sessions have no positives (the non-drinking hand).
- **Ground-truth count:** **405** OR-merged gestures (448 per hand) vs the paper's 410. Consistent.
- **Preprocessing differences from the paper** (document them in the thesis; don't change them): the anti-alias filter and the per-session z-score.

## 4. Current state of the code (all in `code/`)
| File | Role |
|---|---|
| `prepromax.py` | Preprocessing, raw CSVs → pkls |
| `sanity_check_transformations.py` | Visual check of mirroring and downsampling on recording 1-3. Interactive plots, no assertions. |
| `mstcn_pipeline.py` | **Training and evaluation** (fixed pipeline). Presets `paper`, `paper_inception`, `paper_inception_s1`, `legacy`. Every `Config` field is a CLI flag (`--name value`, `--flag/--no-flag`, tuples comma-separated). |
| `plot_results.py` | Training-curve figures. Called automatically; can also be run by hand on any results folder. |
| `test_mstcn_pipeline.py` | 31 tests, all passing |
| `legacy/` | Superseded scripts and old results. **Do not edit.** Their numbers are not trustworthy; see `legacy/README.md`. |

### Outputs per run: `code/results/<timestamp>_<tag>/`
| Path | Contents |
|---|---|
| `config.json` | The exact config that ran, plus receptive field, parameter count and environment |
| `folds/fold_XX_history.csv` | **Per epoch, written while training:** train loss (total/cls/smooth), held-out loss, held-out segmental F1 at each k, frame F1, lr, seconds |
| `folds/fold_XX.json` | Fold result: subjects, sessions, TP/FP/FN for all views, history |
| `folds_summary.csv` | One row per finished fold, appended as soon as the fold ends |
| `plots/fold_XX.{png,pdf}`, `plots/loss_all_folds.{png,pdf}` | Thesis-ready figures (PDF is vector) |
| `predictions/fold_XX.npz` | Per-session P(drinking) as float16, plus labels (git-ignored) |
| `models/fold_XX.pt` | Checkpoints (git-ignored) |
| `summary.json`, `summary.txt` | Pooled results in three tables: OR-merged with the paper rule (**compare with the paper**), OR-merged with the standard one-to-one MS-TCN rule (stricter), and per hand. Frame-wise metrics too. |
| `code/results/experiments_index.csv` | **Experiment tracker**: one row per finished run, with key settings, both F1 variants, TP/FP/FN, runtime and folder |

**Primary metric = the paper's matching rule** (`--seg_rule paper`, the default). It is used for:
- the per-epoch held-out F1 in the history CSV and plots;
- `folds_summary.csv`;
- `--select inner`;
- the main `segF1@k` columns of the index.

The standard one-to-one rule is kept as the secondary view (`segF1_standard@k` in the index; second table in `summary.txt`). `--seg_rule standard` swaps the two.

The held-out monitoring is used only for curves. It never influences training or model selection. With `--select last` (the default), the reported model is the last epoch.

## 5. Work plan
### Phase 0: correctness (DONE, 2026-11-12)
- 31/31 tests pass. Re-run `.venv/Scripts/python.exe test_mstcn_pipeline.py` after **any** change.
- Smoke runs on real data on the GPU completed for all presets.

  | Preset | Params | Time per epoch | Estimated full LOSO run (13 folds × 50 epochs) |
  |---|---|---|---|
  | `paper` | 1.19 M | about 3 s | **about 40 min** |
  | `paper_inception` | 0.76 M | about 4.1 s | **about 50 min** |
  | `legacy` | 1.69 M | about 207 s | about 37 h, impractical (fewer folds/epochs, or `--no-deterministic`) |

### Phase 1: replicate the paper (NEXT)
1. `.venv/Scripts/python.exe mstcn_pipeline.py --preset paper --tag paper_baseline`
   - Target: paper-rule OR-merged segmental F1 ≥ about 0.93 at k=0.5.
   - If it falls short, try these one at a time and record each in the index: more epochs (100–200); `--train_mode crops --crop_len 4096 --batch_size 8`; `--select inner`; no per-session z-score (needs new pkls).

### Phase 2: minor enhancements (same protocol, one change per row)
The rows below are tracked, with status and results, in **`EXPERIMENTS.md`**. Update it after every run.

| Row | Command flags | What it tests |
|---|---|---|
| A | `--preset paper` | replication |
| B | `--preset paper_inception` | Inception layers in all stages |
| C | `--preset paper_inception_s1` | Inception in stage 1 only |
| D | B + `--inception_kernels 3,9,17` | longer kernels (InceptionTime finding) |
| E | B + `--no-inception_maxpool` | max-pool branch contribution |
| F | best of A–E + `--num_stages 3` | an extra refinement stage |
| G | best + `--channels 9` | magnitude + jerk channels |
| H | best + `--class_weight sqrt_inverse` | class weighting vs FP count |
| L | `--preset legacy` (reduced) | old hybrid architecture under the fixed protocol (reference only) |

For each row, report:
- pooled OR-merged TP/FP/FN and F1 at the three k values, under **both** matching rules;
- the per-fold mean ± std;
- the parameter count and receptive field.

FP is the paper's main error mode (mostly eating confused with drinking), so comment on FP explicitly.

### Phase 3 (optional, ask first)
- Statistical check: run each row with 3 seeds (`--seed`), then a paired per-fold comparison.
- Thesis figures across runs, built from `experiments_index.csv`: an ablation bar or dot chart, and per-fold F1 comparisons.
- DX-II (free-living, 45 h): needs preprocessing with `prepromax.py` pointed at DX-II. Three subjects are shared with DX-I (DX-I 1-3, 3-1, 7-1 = DX-II 4-1, 5-1, 6-1).

## 6. Rules
- **Ask the user before any long run** (anything beyond a smoke test). Give the command and an estimated duration. Once approved, run it in the background so the chat stays free.
- Do not modify anything in `code/legacy/`, the pkls, or `dataverse_files/`.
- Never report numbers unless all tests pass.
- Keep every run trackable. Don't delete result folders the user didn't ask to delete; smoke-test folders you created yourself are fine to remove.
- Heavy files (data, pkls, checkpoints, predictions) are git-ignored. Don't force-add them.
