# Hand-over: drinking gesture detection thesis (DX-I, MS-TCN + Inception)

Audience: whoever picks this repo up next (a new agent session or a collaborator). This file covers **where things stand and why**. The work plan and the protocol details are in `REPLICATION_PLAN.md`.
State as of **2026-11-12**.

## 1. TL;DR
- **Thesis:** reproduce the MS-TCN of Wang et al. (EMBC 2022) on DX-I, then add Inception-style dilated layers as a minor, measurable improvement. The report needs a comparison table and training-process figures.
- **The legacy code produced untrustworthy numbers** (segmental F1 about 0.84/0.83/0.77). It is now archived in `code/legacy/`.
- **A new, tested pipeline replaces it:** `code/mstcn_pipeline.py` (31/31 tests pass; smoke runs on the GPU completed).
- **No full experiment has been run with it yet.** The next step is the paper baseline (`--preset paper`, about 40 min), then the Inception variant (about 50 min).

## 2. Quick start
```
cd code
.venv/Scripts/python.exe test_mstcn_pipeline.py                         # must print 31/31 passed
.venv/Scripts/python.exe mstcn_pipeline.py --preset paper --epochs 2 --max_folds 1 --tag smoke   # ~15 s check
.venv/Scripts/python.exe mstcn_pipeline.py --preset paper --tag paper_baseline                  # full LOSO, ~40 min
.venv/Scripts/python.exe mstcn_pipeline.py --preset paper_inception --tag inception_all         # full LOSO, ~80 min
.venv/Scripts/python.exe plot_results.py results/<experiment_folder>                            # (re)draw figures
```
- **Results:** each run writes `code/results/<timestamp>_<tag>/`. Every run is also listed in `code/results/experiments_index.csv`.
- **Watching a run:** while it runs, open `folds/fold_XX_history.csv`, `folds_summary.csv` and `plots/`. They update during the run.
- **Environment:** `code/.venv` (Python 3.11.9, torch 2.5.1+cu121, numpy 2.3, scipy, pandas, sklearn, matplotlib), on an NVIDIA RTX 4060 Ti 16 GB. The venv is tied to this machine. To rebuild elsewhere, install torch (CUDA), numpy, scipy, pandas, scikit-learn, matplotlib and tqdm.

## 3. Repository map
```
REPLICATION_PLAN.md   protocol, paper setup, work plan, ablation rows, rules
EXPERIMENTS.md        thesis experiment log: comparison rows, commands, status, results (fill after each run)
HAND_OVER.md          this file
Drinking_Gesture_...pdf          main paper (Wang et al. 2022)
tcn inception/*.pdf              background: InceptionTime, TCN-Inception, IncepTCN, GoogLeNet
drinking-gesture-dataset-main/   dataset authors' original scripts (reference only)
dataverse_files/                 raw DX-I / DX-II data (git-ignored except ReadMe + MANIFEST)
code/
  prepromax.py                   preprocessing  raw CSV -> DX_I_*_16Hz_with_jerk.pkl
  sanity_check_transformations.py  visual check of mirroring/downsampling (recording 1-3)
  mstcn_pipeline.py              model, loss, LOSO, training, evaluation, tracking
  plot_results.py                training-curve figures (PNG + PDF)
  test_mstcn_pipeline.py         31 tests
  results/                       new runs + experiments_index.csv
  legacy/                        old scripts, old results, old plots (README inside; do not edit)
```

## 4. Decisions made, and why
| Decision | Reason |
|---|---|
| Wrote a new pipeline instead of patching `new_bottle_window_max.py` | The old file was untracked, so editing it in place would have lost the record of earlier runs. Several of its problems were also structural (windowed timeline, split, metric). |
| **Split by subject, not recording** (13 LOSO folds) | `2-1`/`2-2` and `6-1`/`6-2` are the same person. The old split treated them as different people, so the same person could appear in both train and test. |
| Train on **whole sessions**, batch 1, fixed 50 epochs, report the last epoch (`--select last`) | Matches MS-TCN and the paper. The old 8 s windows were shorter than a gesture's context and wasted the dilations. Reporting the last epoch avoids picking the best epoch on the test fold. `--select inner` exists as an alternative. |
| Smoothing loss rewritten (T-MSE on log-prob differences, tau 4, lambda 0.15, padding masked) | The old version penalized confident predictions instead of jitter, and used weight 0.75. |
| Segmental F1 computed under **two rules**; **paper rule is primary** (`seg_rule="paper"`) | **Paper rule** (a weak match = 1 error) is used everywhere for comparison with their Table I: curves, fold summaries, index, inner selection. **Standard MS-TCN one-to-one rule** (a weak match = FP + FN) is reported as the stricter secondary view the external review asked for. The paper's own numbers show it used its rule (391 + 15 ≠ 410). |
| Left/right OR merge per recording, counts pooled over folds | Paper protocol. A two-hand drink counts once. |
| Preprocessing kept as is | Mirroring verified exactly. 405 gestures vs the paper's 410. The filter and z-score are defensible choices that the paper doesn't specify; document them as differences. |
| Inception layer kept MS-TCN-style (no BatchNorm): 1×1 bottleneck (32), parallel dilated kernels {3,5,9}, max-pool branch, 1×1 fuse, residual | It's a drop-in replacement for the dilated layer, so a difference in results can be attributed to the layer alone. |
| Held-out fold monitored every epoch, for curves only | Training-process figures for the thesis, without letting the test fold influence model selection. |
| `legacy` preset uses 6 layers, not 10 | With 5-tap kernels and 128-sample windows, layers with dilation ≥ 64 never see real data on either side. |

## 5. Known issues and open questions
- **Paper details not stated:** epochs, batch size, normalization, filtering and model selection. We use MS-TCN defaults (50 epochs, batch 1, last epoch). If the baseline misses about 0.93 at k=0.5, the fallbacks to try are listed in `REPLICATION_PLAN.md` Phase 1.
- **The `legacy` preset is too slow** for full LOSO (about 37 h). Run it only reduced, if at all.
- **`prepromax.py` and `sanity_check_transformations.py` have hardcoded absolute paths**, so they only run on this machine. They're easy to make relative, but haven't been yet.
- **`sanity_check_transformations.py` blocks** on `plt.show()` and asserts nothing.
- **Downsampling filter:** its suppression just above 8 Hz is moderate (−8.5 dB at 8 Hz). That's fine for arm motion below about 5 Hz. An optional check against `scipy.signal.decimate` was proposed but not run.
- **DX-II is unused.** It's optional Phase 3 work.
- **Old numbers:** the 4 legacy PNGs in `code/legacy/plots/` have no generating script.

## 6. Git and sharing
- **Repo state:** branch `main`, **no commits yet**. Everything relevant is staged (about 11 MB). Heavy files are git-ignored: data, zips, pkls, checkpoints, prediction dumps and legacy prediction JSONs.
- **Planned remote:** `https://github.com/asoulios97/PillBox.git`. Not added and not pushed yet. Ask the user before pushing. The repo contains third-party paper PDFs, which the user may not want in a public repo.
- **Commit messages** must end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## 7. Working agreements with the user
- **Long runs need explicit approval.** Give the command and an estimated time, then run in the background so the chat stays free.
- **Ping the user** when a full LOSO run is ready to start, and when it finishes.
- **Keep everything tracked** (history CSV, fold summaries, PNG + PDF figures, experiment index). The thesis needs figures of the training process.
- **Another Claude session may work on the same repo in parallel.** Check `git status` and coordinate before editing `mstcn_pipeline.py` or the tests.

## 8. Next steps
1. Run `--preset paper --tag paper_baseline` (approved by the user before starting).
2. Inspect `summary.txt` (paper-rule table vs the paper's 0.956/0.955/0.943), `plots/loss_all_folds.pdf` and `folds_summary.csv`. Look for outlier subjects.
3. If the result is close to the paper, run `--preset paper_inception`, then the remaining ablation rows (`REPLICATION_PLAN.md` Phase 2).
4. Build comparison figures and tables from `experiments_index.csv`.
