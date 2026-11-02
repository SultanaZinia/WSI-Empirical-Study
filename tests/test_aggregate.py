import csv
import json
import math

import numpy as np
import pytest

from lesionbench.eval.aggregate import lesion_hits, mean_sd, pooled_lsr, retention_means, run_dirs, slide_auc

STRATA = {"itc_max": 0.2, "micro_max": 2.0}


def _run(root, method, fold, K, preds, rets, kept, done=True):
    d = root / method / f"fold{fold}" / f"K{K}"
    d.mkdir(parents=True)
    for name, rows in (("predictions.csv", preds), ("retention.csv", rets)):
        with open(d / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    np.savez(d / "retained.npz", **{f"retained/{k}": np.array(v) for k, v in kept.items()})
    if done:
        (d / "done.json").write_text(json.dumps({}))


def _pred(sid, split, label, prob):
    return {"slide_id": sid, "split": split, "label": label, "prob": prob}


def _ret(sid, split, stratum, cov, lsr, tep):
    return {"slide_id": sid, "split": split, "stratum": stratum, "cov": cov, "lsr": lsr, "tep": tep,
            "cov_strict": cov, "tep_strict": tep}


@pytest.fixture
def runs(tmp_path):
    preds0 = [_pred("tumor_001", "val", 1, 0.9), _pred("normal_001", "val", 0, 0.2),
              _pred("test_001", "test", 1, 0.4), _pred("test_002", "test", 0, 0.6)]
    preds1 = [_pred("tumor_002", "val", 1, 0.3), _pred("normal_002", "val", 0, 0.1),
              _pred("test_001", "test", 1, 0.8), _pred("test_002", "test", 0, 0.7)]
    rets = [_ret("test_001", "test", "micro", 1.0, 0.5, 0.25),
            _ret("test_002", "test", "negative", "nan", "nan", 0.0)]
    _run(tmp_path, "lts", 0, 32, preds0, rets, {"tumor_001": [0], "normal_001": [0], "test_001": [0, 5],
                                                 "test_002": [1]})
    _run(tmp_path, "lts", 1, 32, preds1, rets, {"tumor_002": [3], "normal_002": [0], "test_001": [9],
                                                 "test_002": [1]})
    _run(tmp_path, "lts", 2, 32, preds1, rets, {}, done=False)  # unfinished: ignored
    return run_dirs(tmp_path)


def test_run_dirs_only_finished(runs):
    assert [(m, K, f) for m, K, f, _ in runs] == [("lts", "32", 0), ("lts", "32", 1)]


def test_slide_auc_per_fold_and_pooled_oof(runs):
    a = {(r["fold"], r["split"]): r["auc"] for r in slide_auc(runs)}
    assert a[(0, "test")] == 0.0 and a[(1, "test")] == 1.0
    assert a[(0, "val")] == 1.0 and a[(1, "val")] == 1.0
    # pooled val: tumour 0.9, 0.3 vs normal 0.2, 0.1 -> all pairs ordered
    assert a[("all", "val_oof")] == 1.0


def test_lesion_hits_primary_and_strict(runs):
    no_lesion = (np.array([], int), np.array([], int), np.array([]), np.array([]))
    overlaps = {
        # test_001: lesion 0 (micro, 500 um) on rows 5 (0.30) and 9 (0.06); lesion 1 (ITC, 100 um) on row 7
        "test_001": (np.array([5, 9, 7]), np.array([0, 0, 1]), np.array([0.30, 0.06, 0.5]),
                     np.array([500.0, 100.0])),
        "tumor_001": (np.array([0]), np.array([0]), np.array([0.02]), np.array([3000.0])),
        "tumor_002": (np.array([3]), np.array([0]), np.array([0.5]), np.array([1500.0])),
        "test_002": no_lesion, "normal_001": no_lesion, "normal_002": no_lesion,
    }
    h = lesion_hits(runs, overlaps, 0.05, 0.10, ["tumor_002"], STRATA)
    r = {(x["fold"], x["slide_id"], x["lesion_id"]): x for x in h}
    assert r[(0, "test_001", 0)]["hit"] == 1 and r[(0, "test_001", 0)]["hit_strict"] == 1  # row 5, 0.30
    assert r[(1, "test_001", 0)]["hit"] == 1 and r[(1, "test_001", 0)]["hit_strict"] == 0  # row 9, 0.06
    assert r[(0, "test_001", 1)]["hit"] == 0 and r[(0, "test_001", 1)]["size_class"] == "itc"
    assert r[(0, "tumor_001", 0)]["hittable"] == 0 and r[(0, "tumor_001", 0)]["size_class"] == "macro"
    assert r[(1, "tumor_002", 0)]["exhaustive"] == 0 and r[(1, "tumor_002", 0)]["hit"] == 1
    micro = pooled_lsr(h, split="test", size="micro", exhaustive_only=True)
    assert micro[("lts", "32")] == [1.0, 1.0]
    assert pooled_lsr(h, split="val", size="micro", exhaustive_only=True) == {}  # tumor_002 excluded
    assert pooled_lsr(h, split="val", size="micro", exhaustive_only=False)[("lts", "32")] == [1.0]


def test_retention_means_skip_nan(runs):
    m = {(r["fold"], r["stratum"]): r for r in retention_means(runs)}
    assert m[(0, "micro")]["lsr"] == 0.5 and m[(0, "micro")]["n_slides"] == 1
    assert math.isnan(m[(0, "negative")]["cov"]) and m[(0, "negative")]["tep"] == 0.0


def test_mean_sd_format():
    assert mean_sd([0.5, 0.7]) == "0.600 ± 0.141"
    assert mean_sd([0.5, float("nan")]) == "0.500" and mean_sd([]) == "–"
