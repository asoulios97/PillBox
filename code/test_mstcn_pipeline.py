"""
Tests for mstcn_pipeline.py.  Run from code/:
    .venv/Scripts/python.exe test_mstcn_pipeline.py        (plain runner)
    .venv/Scripts/python.exe -m pytest test_mstcn_pipeline.py   (if pytest is installed)
"""
import json
import sys
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mstcn_pipeline as mp  # noqa: E402


def labels_from(segs, T):
    y = np.zeros(T, dtype=np.int64)
    for s, e in segs:
        y[s:e] = 1
    return y


def random_labels(rng, T, n_max=6, min_len=3, max_len=40):
    y = np.zeros(T, dtype=np.int64)
    for _ in range(rng.randint(0, n_max + 1)):
        s = rng.randint(0, T - min_len)
        y[s:s + rng.randint(min_len, max_len)] = 1
    return y


# ---------------------------------------------------------------- metrics (P2)

def test_segments_helper():
    assert mp.get_segments(np.array([0, 1, 1, 0, 1])) == [(1, 3), (4, 5)]
    assert mp.get_segments(np.array([1, 1])) == [(0, 2)]
    assert mp.get_segments(np.zeros(5)) == []


def test_perfect_match():
    y = labels_from([(10, 20)], 100)
    assert mp.segmental_counts(y, y, 0.5) == (1, 0, 0)


def test_no_overlap():
    assert mp.segmental_counts(labels_from([(50, 60)], 100), labels_from([(10, 20)], 100), 0.1) == (0, 1, 1)


def test_oversegmentation_inside_one_gt():
    gt = labels_from([(10, 30)], 100)
    pr = labels_from([(10, 18), (20, 30)], 100)
    assert mp.segmental_counts(pr, gt, 0.25) == (1, 1, 0)


def test_one_pred_spans_two_gts():
    gt = labels_from([(10, 20), (30, 40)], 100)
    pr = labels_from([(10, 40)], 100)
    assert mp.segmental_counts(pr, gt, 0.25) == (1, 0, 1)


def test_below_threshold_counts_fp_and_fn_once():
    gt = labels_from([(10, 20)], 100)
    pr = labels_from([(18, 40)], 100)          # IoU 2/30
    assert mp.segmental_counts(pr, gt, 0.5) == (0, 1, 1)


def test_invariants_random():
    rng = np.random.RandomState(0)
    for _ in range(2000):
        T = rng.randint(20, 300)
        gt, pr = random_labels(rng, T), random_labels(rng, T)
        for k in (0.1, 0.25, 0.5):
            tp, fp, fn = mp.segmental_counts(pr, gt, k)
            assert tp + fn == len(mp.get_segments(gt))
            assert tp + fp == len(mp.get_segments(pr))


def test_paper_rule_cases():
    gt = labels_from([(10, 20)], 100)
    assert mp.segmental_counts_paper(gt, gt, 0.5) == (1, 0, 0)
    assert mp.segmental_counts_paper(labels_from([(50, 60)], 100), gt, 0.1) == (0, 1, 1)
    # weak match, prediction longer than GT -> one FP only (standard rule: FP + FN)
    assert mp.segmental_counts_paper(labels_from([(18, 40)], 100), gt, 0.5) == (0, 1, 0)
    # weak match, prediction shorter than GT -> one FN only
    assert mp.segmental_counts_paper(labels_from([(18, 21)], 100), gt, 0.5) == (0, 0, 1)
    # over-segmentation inside one GT, and one prediction spanning two GTs
    assert mp.segmental_counts_paper(labels_from([(10, 18), (20, 30)], 100),
                                     labels_from([(10, 30)], 100), 0.25) == (1, 1, 0)
    assert mp.segmental_counts_paper(labels_from([(10, 40)], 100),
                                     labels_from([(10, 20), (30, 40)], 100), 0.25) == (1, 0, 1)


def test_paper_rule_never_stricter_than_standard():
    rng = np.random.RandomState(3)
    for _ in range(2000):
        T = rng.randint(20, 300)
        gt, pr = random_labels(rng, T), random_labels(rng, T)
        for k in (0.1, 0.25, 0.5):
            tp_p, fp_p, fn_p = mp.segmental_counts_paper(pr, gt, k)
            tp_s, fp_s, fn_s = mp.segmental_counts(pr, gt, k)
            assert tp_p + fn_p <= len(mp.get_segments(gt))
            assert mp.f1_from_counts(tp_p, fp_p, fn_p) >= mp.f1_from_counts(tp_s, fp_s, fn_s) - 1e-12


def test_f1_from_counts():
    assert mp.f1_from_counts(0, 0, 0) == 0.0
    assert abs(mp.f1_from_counts(391, 32, 15) - 0.943) < 1e-3     # paper Table I, k=0.5


# ------------------------------------------------- oracle through pipeline (P3)

def oracle_prob_fn(xb):
    """Fake model: channel 0 carries the label -> confident logits."""
    z = 10.0 * (2.0 * xb[:, 0, :] - 1.0)
    p1 = 1.0 / (1.0 + np.exp(-2 * z))
    return np.stack([1 - p1, p1], axis=-1).astype(np.float32)


def make_session(y, rec="1-1", hand="Right", C=6, seed=0):
    x = np.random.RandomState(seed).randn(len(y), C).astype(np.float32)
    x[:, 0] = y
    return mp.Session(x=x, y=y, recording=rec, subject=rec.split("-")[0], hand=hand)


def test_oracle_windows_and_full():
    rng = np.random.RandomState(1)
    for T in (50, 128, 129, 192, 1000, 1337):          # shorter, exact fit, off-by-one, long
        y = random_labels(rng, T, n_max=8)
        s = make_session(y)
        for weight in ("uniform", "center"):
            p = mp.predict_windows(oracle_prob_fn, s.x, 128, 64, weight)
            assert p.shape == (T, 2)
            assert np.array_equal(p.argmax(1), y), (T, weight)
        assert np.array_equal(mp.predict_full(oracle_prob_fn, s.x).argmax(1), y)
        ev = mp.evaluate_predictions([s], [mp.predict_windows(oracle_prob_fn, s.x, 128, 64)], (0.1, 0.25, 0.5))
        for k in ("k=0.1", "k=0.25", "k=0.5"):
            c = ev["per_hand"][k]
            assert c["fp"] == 0 and c["fn"] == 0 and c["tp"] == len(mp.get_segments(y))


def test_all_zero_prediction():
    y = labels_from([(10, 20), (40, 60)], 200)
    s = make_session(y)
    probs = np.tile([[1.0, 0.0]], (200, 1)).astype(np.float32)
    ev = mp.evaluate_predictions([s], [probs], (0.5,))
    assert ev["per_hand"]["k=0.5"] == {"tp": 0, "fp": 0, "fn": 2}


def test_window_starts_cover_everything():
    for T in range(1, 600, 7):
        starts = mp.window_starts(T, 128, 64)
        covered = np.zeros(T, bool)
        for s in starts:
            covered[s:s + 128] = True
        assert covered.all() and starts[-1] < T


# ------------------------------------------------------------------ OR merge (P6)

def test_or_merge_two_hand_drink_counts_once():
    L = labels_from([(10, 20)], 100)
    R = labels_from([(12, 22), (60, 70)], 97)            # different length (truncated to 97)
    sL, sR = make_session(L, hand="Left"), make_session(R, hand="Right")
    merged = mp.or_merge([sL, sR], [L, R])
    assert len(merged) == 1 and len(merged[0][2]) == 97
    assert mp.get_segments(merged[0][2]) == [(10, 22), (60, 70)]
    ev = mp.evaluate_predictions([sL, sR], [np.eye(2)[L], np.eye(2)[R]], (0.5,))
    assert ev["n_gt_segments_or"] == 2 and ev["n_gt_segments_per_hand"] == 3
    assert ev["or_merged"]["k=0.5"] == {"tp": 2, "fp": 0, "fn": 0}


# ---------------------------------------------------------------------- loss (P1)

def _loss():
    return mp.MSTCNLoss(smooth_weight=0.15, tau=4.0)


def test_smoothing_constant_logits_zero():
    logits = torch.randn(2, 2, 1).repeat(1, 1, 50)
    assert _loss().smoothing(logits, torch.ones(2, 50, dtype=torch.bool)).item() < 1e-10


def test_smoothing_confident_constant_near_zero():
    logits = torch.tensor([5.0, -5.0]).view(1, 2, 1).repeat(1, 1, 50)
    assert _loss().smoothing(logits, torch.ones(1, 50, dtype=torch.bool)).item() < 1e-10


def test_smoothing_single_flip_saturates():
    logits = torch.tensor([20.0, -20.0]).view(1, 2, 1).repeat(1, 1, 10)
    logits[:, :, 5:] = torch.tensor([-20.0, 20.0]).view(1, 2, 1)
    loss = _loss().smoothing(logits, torch.ones(1, 10, dtype=torch.bool)).item()
    # one transition out of 9, both classes saturate at tau^2 = 16
    assert abs(loss - 16.0 / 9.0) < 1e-4, loss


def test_smoothing_ignores_padding():
    logits = torch.zeros(1, 2, 10)
    logits[:, :, 6:] = torch.tensor([30.0, -30.0]).view(1, 2, 1)     # jump only inside padding
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[:, :6] = True
    assert _loss().smoothing(logits, mask).item() < 1e-10
    targets = torch.full((1, 10), mp.IGNORE, dtype=torch.long)
    targets[:, :6] = 0
    total, _ = _loss()([logits], targets)
    assert torch.isfinite(total)


def test_loss_reduction_sum_vs_mean():
    logits = [torch.randn(1, 2, 20), torch.randn(1, 2, 20)]
    t = torch.randint(0, 2, (1, 20))
    s, _ = mp.MSTCNLoss(stage_reduce="sum")(logits, t)
    m, _ = mp.MSTCNLoss(stage_reduce="mean")(logits, t)
    assert abs(s.item() - 2 * m.item()) < 1e-5


# --------------------------------------------------- models, RF, span check (P4/P8)

def test_paper_receptive_field():
    rf = mp.receptive_field(mp.build_config("paper"))
    assert rf["per_stage"] == [1023, 1023] and rf["total"] == 2045


def test_output_lengths_all_presets():
    for preset in ("paper", "paper_inception", "paper_inception_s1", "legacy"):
        cfg = mp.build_config(preset, num_layers=4, num_features=16)
        model = mp.build_model(cfg, 6).eval()
        for T in (37, 128, 255):
            outs = model(torch.randn(2, 6, T))
            assert len(outs) == cfg.num_stages
            assert all(o.shape == (2, 2, T) for o in outs), preset


def test_paper_param_count_reasonable():
    n = sum(p.numel() for p in mp.build_model(mp.build_config("paper"), 6).parameters())
    assert 0.5e6 < n < 1.5e6, n


def test_span_check():
    try:
        mp.check_dilation_span(mp.build_config("legacy", num_layers=10))
        raise AssertionError("expected ValueError for dilation 512 in 128-sample windows")
    except ValueError:
        pass
    assert mp.check_dilation_span(mp.build_config("legacy")) is None
    assert mp.check_dilation_span(mp.build_config("paper")) is None          # full sessions


def test_refine_input_pred_only_shapes():
    cfg = mp.build_config("legacy", num_layers=3, num_features=16, refine_input="pred_only")
    outs = mp.build_model(cfg, 6).eval()(torch.randn(1, 6, 64))
    assert len(outs) == 4


# ---------------------------------------------------------------- config (P7)

def test_config_cli_roundtrip():
    cfg = mp.parse_args(["--preset", "paper_inception", "--epochs", "7", "--inception_kernels", "3,9,17",
                         "--no-inception_maxpool", "--iou_thresholds", "0.1,0.5", "--folds", "5"])
    assert cfg.layer_type == "inception" and cfg.epochs == 7
    assert cfg.inception_kernels == (3, 9, 17) and cfg.inception_maxpool is False
    assert cfg.iou_thresholds == (0.1, 0.5) and cfg.folds == "5"
    again = mp.build_config("paper", **json.loads(json.dumps(asdict(cfg))))
    assert asdict(again) == asdict(cfg)


def test_bad_config_rejected():
    for bad in ({"train_mode": "x"}, {"kernel_size": 4}, {"nonexistent": 1}):
        try:
            mp.build_config("paper", **bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


# ----------------------------------------------------------------- folds (P6/P12)

def test_folds_group_by_subject():
    recs = ["1-3", "2-1", "2-2", "3-1", "6-1", "6-2"]
    sessions = [make_session(np.zeros(10, np.int64), rec=r, hand=h) for r in recs for h in ("Left", "Right")]
    folds = mp.make_folds(sessions, "loso", 0)
    assert len(folds) == 4
    for f in folds:
        test_recs = {sessions[i].recording for i in f["test"]}
        if "2" in f["test_subjects"]:
            assert test_recs == {"2-1", "2-2"} and len(f["test"]) == 4
    k = mp.make_folds(sessions, "2", 0)
    assert len(k) == 2 and sorted(sum((f["test"] for f in k), [])) == list(range(len(sessions)))
    assert mp.make_folds(sessions, "2", 0)[0]["test"] == k[0]["test"]           # seeded


def test_class_weights():
    s = [make_session(labels_from([(0, 10)], 100))]
    assert mp.compute_class_weights("none", s) is None
    w = mp.compute_class_weights("inverse", s)
    assert abs(w[1].item() / w[0].item() - 9.0) < 1e-5
    w = mp.compute_class_weights("sqrt_inverse", s)
    assert abs(w[1].item() / w[0].item() - 3.0) < 1e-5


# --------------------------------------------------------------- end-to-end smoke

def synthetic_sessions(n_subjects=3, T=600):
    rng = np.random.RandomState(0)
    out = []
    for subj in range(1, n_subjects + 1):
        for hand in ("Left", "Right"):
            y = random_labels(rng, T, n_max=5, min_len=10, max_len=40)
            x = rng.randn(T, 6).astype(np.float32) * 0.3
            x[:, 0] += 2 * y
            out.append(mp.Session(x=x, y=y, recording=f"{subj}-1", subject=str(subj), hand=hand))
    return out


def _smoke(preset, **kw):
    cfg = mp.build_config(preset, epochs=2, num_layers=3, num_features=8, device="cpu", verbose=0,
                          save_models=False, eval_every=1, **kw)
    with tempfile.TemporaryDirectory() as d:
        summary = mp.run_experiment(cfg, sessions=synthetic_sessions(), out_dir=Path(d))
        for f in ("config.json", "summary.json", "summary.txt", "folds/fold_01.json", "predictions/fold_01.npz",
                  "folds/fold_01_history.csv", "folds_summary.csv", "train.log", "plots/fold_01.png", "plots/loss_all_folds.png"):
            assert (Path(d) / f).exists(), f
        import csv as _csv
        with open(Path(d) / "folds" / "fold_01_history.csv", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        assert len(rows) == cfg.epochs and all(r["heldout_loss"] != "" for r in rows)
        with open(Path(d) / "folds_summary.csv", encoding="utf-8") as fh:
            assert len(list(_csv.DictReader(fh))) == summary["n_folds"]
        # final evaluation == last-epoch monitoring when select=last
        if cfg.select == "last":
            last = json.loads((Path(d) / "folds" / "fold_01.json").read_text(encoding="utf-8"))
            view = mp.primary_view(cfg)
            k0 = cfg.iou_thresholds[0]
            assert abs(last["history"][-1][f"heldout_segF1_k{k0}"]
                       - mp.f1_from_counts(**last["eval"][view][f"k={k0}"])) < 1e-9
        saved = json.loads((Path(d) / "config.json").read_text(encoding="utf-8"))
        assert saved["config"] == json.loads(json.dumps(asdict(cfg)))              # config fidelity
    return summary


def test_end_to_end_modes():
    _smoke("paper")
    _smoke("paper_inception", train_mode="crops", crop_len=256, batch_size=4, select="inner")
    _smoke("legacy", window=64, stride=32, batch_size=16, num_stages=2)


def test_reproducible_with_seed():
    a, b = _smoke("paper"), _smoke("paper")
    assert a["segment"] == b["segment"] and a["frame"] == b["frame"]


# ------------------------------------------------------------- real data check

def test_real_data_if_present():
    cfg = mp.build_config("paper")
    if not (Path(cfg.data_dir) / "DX_I_X_16Hz_with_jerk.pkl").exists():
        print("    (skipped: pkl files not present)")
        return
    sessions = mp.load_sessions(cfg)
    assert len(sessions) == 30 and all(s.x.shape[1] == 6 for s in sessions)
    assert len({s.subject for s in sessions}) == 13
    folds = mp.make_folds(sessions, "loso", 42)
    assert len(folds) == 13
    gt = [s.y for s in sessions]
    n_or = sum(len(mp.get_segments(y)) for _, _, y in mp.or_merge(sessions, gt))
    n_hand = sum(len(mp.get_segments(y)) for y in gt)
    print(f"    DX-I GT segments: OR-merged={n_or}  per-hand={n_hand}  (paper: 410)")


# ----------------------------------------------------------------------- runner

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
