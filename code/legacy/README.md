# Legacy (superseded, kept for the record)

Nothing in this folder is needed to run the current pipeline, and nothing here should be edited. The current entry points are `code/prepromax.py` (preprocessing) and `code/mstcn_pipeline.py` (training and evaluation).

| File | What it was |
|---|---|
| `new_bottle_window_max.py` | Last main training script: Hybrid MS-TCN, 128-sample windows, 5-fold CV. Its results are **not trustworthy** (see `REPLICATION_PLAN.md`, section 4). |
| `new_bottle_window.py` | Earlier version with a single 80/20 split |
| `ms_tcn_inception.py` | First version, batch size 1 |
| `view_results.py` | Results viewer for the `results/experiment_*` format of the scripts above |
| `QUICK_START.md` | Docs for the legacy training and viewer workflow |
| `training_history.json` | History of a `new_bottle_window.py` run |
| `prepro_tolias.py` | Older preprocessing on the Hand-Mirrored CSVs. It concatenated the two hands into one sequence, so it was replaced by `prepromax.py`. **Running it overwrites the pkls.** |
| `prepro_v0.py` | Earliest preprocessing draft, right hand only (recovered from git, formerly `code/prepro`) |
| `test_gpu.py`, `gpu_test.py` | CUDA sanity checks |
| `results/experiment_*` | Outputs of `new_bottle_window_max.py`, 2026-01-17/18. The model and prediction dumps are git-ignored. |
| `plots/*.png` | Ad-hoc analysis plots; no generating script exists in the repo |

Hardcoded absolute paths in these scripts point at the old locations, so they won't run from here without editing.
