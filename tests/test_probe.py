import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lesionbench.metrics.probe import (
    Probe,
    detection_by_band,
    lesion_separation,
    max_overlap,
    probe_training_set,
    separation,
)


def test_max_overlap_takes_largest_lesion_per_patch():
    out = max_overlap(5, rows=[1, 1, 3], fracs=[0.2, 0.6, 0.05])
    np.testing.assert_allclose(out, [0, 0.6, 0, 0.05, 0])


def test_probe_training_set_uses_clear_tumour_and_normal_slides_only():
    feats = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    ov = (np.array([0, 1, 2]), None, np.array([0.9, 0.3, 0.5]))
    tumour = SimpleNamespace(features=feats, label=1, overlap=ov)
    normal = SimpleNamespace(features=feats, label=0, overlap=(np.array([], int), None, np.array([])))
    X, y = probe_training_set([tumour, normal], np.random.default_rng(0))
    assert y.tolist() == [1, 1] + [0] * 6  # tumour rows 0 and 2 (>= 0.5); row 1 (0.3) is not used at all
    np.testing.assert_array_equal(X[:2], feats[[0, 2]].numpy())
    X, y = probe_training_set([normal], np.random.default_rng(0), cap=4)
    assert len(y) == 4


def test_detection_by_band_hand_computed():
    normal = np.arange(100.0)  # 95% quantile = 94.05
    tau, rows = detection_by_band([10.0, 99.0, 99.0, 50.0, 95.0], [0.03, 0.03, 0.2, 0.2, 0.7], normal)
    assert tau == pytest.approx(94.05)
    got = {(r["band_lo"], r["band_hi"]): (r["n_patches"], r["detected"]) for r in rows}
    assert got[(0.0, 0.05)] == (2, 0.5) and got[(0.10, 0.25)] == (2, 0.5) and got[(0.5, 1.0)] == (1, 1.0)
    assert got[(0.05, 0.10)][0] == 0 and math.isnan(got[(0.05, 0.10)][1])


def test_separation_hand_computed():
    auc, e = separation([2.0, 4.0], [0.0, 1.0, 2.0])
    # AUC: pairs pos > neg: 2 vs (0,1,2) -> 2 + 0.5 tie, 4 vs all -> 3 => 5.5 / 6
    assert auc == pytest.approx(5.5 / 6)
    # means 3 and 1; population vars 1 and 2/3 -> ESI = 4 / (0.5 * 5/3) = 4.8
    assert e == pytest.approx(4.8)
    auc1, e1 = separation([5.0], [0.0, 1.0])
    assert auc1 == 1.0 and math.isnan(e1)
    assert all(math.isnan(v) for v in separation([], [1.0]))


def test_lesion_separation_thresholds_and_clean_reference():
    scores = np.array([5.0, 4.0, 0.0, 1.0, 9.0, 0.5])
    # lesion 0 covers rows 0 (0.6) and 1 (0.05); lesion 1 covers row 4 (0.3); rows 2, 3, 5 touch no lesion
    out = lesion_separation(scores, [0, 1, 4], [0, 0, 1], [0.6, 0.05, 0.3], 2, [0.0, 0.1])
    r = {(o["lesion_id"], o["threshold"]): o for o in out}
    assert r[(0, 0.0)]["n_patches"] == 2 and r[(0, 0.1)]["n_patches"] == 1
    assert r[(0, 0.0)]["probe_auc"] == 1.0 and r[(1, 0.1)]["probe_auc"] == 1.0
    assert math.isnan(r[(0, 0.1)]["probe_esi"])  # one patch: ESI undefined
    # lesion 0 at t=0: pos (5, 4), neg (0, 1, 0.5): means 4.5 and 0.5, vars 0.25 and 1/6
    assert r[(0, 0.0)]["probe_esi"] == pytest.approx(16 / (0.5 * (0.25 + 1 / 6)))


def test_probe_separates_shifted_class():
    g = np.random.default_rng(0)
    X = np.r_[g.normal(size=(200, 5)), g.normal(size=(200, 5)) + [3, 0, 0, 0, 0]]
    y = np.r_[np.zeros(200), np.ones(200)]
    p = Probe(X, y, seed=0)
    s = p.score(X)
    assert separation(s[y == 1], s[y == 0])[0] > 0.95
