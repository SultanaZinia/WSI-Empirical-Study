import numpy as np
import pytest

from lesionbench.eval.lesion_analysis import (
    attribute,
    boot_share_diff,
    d50,
    nrs,
    retention_auc,
    survival_by_stage,
)


def test_d50_interpolates_on_log_diameter():
    # already monotone (0, 0, 1, 1 at 100..800 um): isotonic fit = data, crossing between 200 and 400
    d = np.array([100.0, 200, 400, 800])
    assert d50(d, np.array([0.0, 0, 1, 1])) == pytest.approx(np.sqrt(200 * 400))


def test_d50_pools_violations_and_censors():
    d = np.array([100.0, 200, 400, 800])
    # 1, 0 at the two smallest pool to 0.5 each -> the fit reaches 0.5 at the smallest diameter
    assert d50(d, np.array([1.0, 0, 1, 1])) == pytest.approx(100.0)
    assert np.isnan(d50(d, np.array([0.0, 0, 0, 0.4])))  # never reaches 0.5
    assert d50(d, np.array([0.6, 0.6, 1, 1])) == pytest.approx(100.0)  # already above at the smallest lesion


def test_d50_weight_equals_duplication_and_zero_drops():
    d = np.array([100.0, 200, 400, 800])
    h = np.array([0.0, 1, 0, 1])
    dup = d50(np.r_[d, 200.0], np.r_[h, 1.0])
    assert d50(d, h, weight=np.array([1, 2, 1, 1])) == pytest.approx(dup)
    assert d50(d, h, weight=np.array([1, 0, 1, 1])) == pytest.approx(d50(d[[0, 2, 3]], h[[0, 2, 3]]))


def test_nrs_and_retention_auc():
    assert nrs(0.5, 0.1, 0.9) == pytest.approx(0.5)
    assert np.isnan(nrs(0.5, 0.9, 0.9))
    # LSR 0.2, 0.4, 0.6, 0.8 at K 32..256: trapezoids on log2 K (unit steps) / 3 = (0.3 + 0.5 + 0.7) / 3
    assert retention_auc([32, 64, 128, 256], [0.2, 0.4, 0.6, 0.8]) == pytest.approx(0.5)


def test_attribute_precedence():
    cat = attribute(probe_auc=np.array([0.95, 0.80, 0.80, 0.95, np.nan]),
                    hittable=np.array([1, 1, 1, 1, 0]), hit=np.array([0, 0, 1, 1, 0]), cut=0.90)
    assert cat.tolist() == ["selector", "encoder", "retained", "retained", "unhittable"]


def test_boot_share_diff_is_fold_mean_and_clusters_by_slide():
    # fold 0: slide a 2 selector, slide b 1 encoder; fold 1: slide a 1 selector 1 encoder, slide b 1 encoder
    sel = np.array([[2, 0], [1, 0]])
    enc = np.array([[0, 1], [1, 1]])
    point = boot_share_diff(sel, enc, np.ones((1, 2)))[0]
    assert point == pytest.approx(((2 - 1) / 3 + (1 - 2) / 3) / 2)
    only_a = boot_share_diff(sel, enc, np.array([[2, 0]]))[0]  # slide a drawn twice
    assert only_a == pytest.approx((1.0 + 0.0) / 2)


def test_survival_by_stage():
    # layer of death per lesion: -1 = survived all 5 pruners
    out = survival_by_stage(np.array([0, 2, -1, -1]), n_stages=5)
    assert out.tolist() == pytest.approx([0.75, 0.75, 0.5, 0.5, 0.5])
