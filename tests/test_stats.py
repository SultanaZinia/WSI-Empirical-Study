import numpy as np
import pytest

from lesionbench.stats.inference import (
    boot_auc,
    boot_p,
    holm,
    paired_wilcoxon,
    percentile_ci,
    pooled_rate,
    resample_weights,
    slide_mean,
)


def test_resample_weights_counts_and_seed():
    W = resample_weights(7, 50, seed=3)
    assert W.shape == (50, 7)
    assert (W.sum(axis=1) == 7).all()
    assert np.array_equal(W, resample_weights(7, 50, seed=3))
    assert not np.array_equal(W, resample_weights(7, 50, seed=4))


def test_weighted_auc_equals_auc_on_duplicated_slides():
    y = np.array([0, 0, 1, 1])
    p = np.array([[0.1, 0.4, 0.35, 0.8]])
    assert boot_auc(y, p, np.ones((1, 4)))[0] == pytest.approx(0.75)
    # slide 0 drawn twice, slide 1..3 once: negatives .1 .1 .4, positives .35 .8 -> 5 of 6 pairs ordered
    assert boot_auc(y, p, np.array([[2, 1, 1, 1]]))[0] == pytest.approx(5 / 6)


def test_auc_is_mean_over_folds_and_nan_without_both_classes():
    y = np.array([0, 0, 1, 1])
    probs = np.array([[0.1, 0.4, 0.35, 0.8], [0.1, 0.2, 0.3, 0.4]])  # fold AUCs 0.75 and 1.0
    out = boot_auc(y, probs, np.array([[1, 1, 1, 1], [4, 0, 0, 0]]))
    assert out[0] == pytest.approx(0.875)
    assert np.isnan(out[1])


def test_slide_mean_skips_undefined_slides():
    v = np.array([1.0, np.nan, 0.0, 0.5])
    W = np.array([[1, 1, 1, 1], [0, 3, 1, 0], [0, 4, 0, 0]])
    out = slide_mean(v, W)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(0.0)
    assert np.isnan(out[2])


def test_pooled_rate_keeps_lesions_of_a_slide_together():
    hits = np.array([[1, 0], [2, 1]])  # (folds, slides): lesions hit
    totals = np.array([2, 1])  # lesions per slide
    assert pooled_rate(hits, totals, np.ones((1, 2)))[0] == pytest.approx((1 / 3 + 3 / 3) / 2)
    assert pooled_rate(hits, totals, np.array([[0, 2]]))[0] == pytest.approx((0 + 1) / 2)


def test_boot_p_two_sided_with_plus_one():
    assert boot_p(np.ones(999)) == pytest.approx(2 / 1000)
    assert boot_p(np.array([-1.0, 1.0] * 500)) == 1.0


def test_percentile_ci_ignores_nan():
    lo, hi = percentile_ci(np.r_[np.arange(101.0), np.nan])
    assert (lo, hi) == pytest.approx((2.5, 97.5))


def test_holm_step_down_monotone_and_nan_passthrough():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = holm([0.02, np.nan, 0.5])
    assert out[0] == pytest.approx(0.04) and np.isnan(out[1]) and out[2] == pytest.approx(0.5)
    assert holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])


def test_paired_wilcoxon():
    a = np.array([1.0, 2, 3, 4, 5, np.nan])
    assert paired_wilcoxon(a, np.zeros(6)) == pytest.approx(2 / 32)  # exact, n = 5 all positive
    assert paired_wilcoxon(np.ones(4), np.ones(4)) == 1.0
