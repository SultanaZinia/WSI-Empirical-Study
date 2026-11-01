import math

import numpy as np
import pytest

from lesionbench.metrics.encoder import esi, svm_patch_auc
from lesionbench.metrics.retention import cov_at_k, lsr_at_k, retained_from_assignment, sdi, tep_at_k

# 6 patches: lesion 0 covers rows 0-1, lesion 1 covers row 4, lesion 2 has no positive patch.
Y = np.array([1, 1, 0, 0, 1, 0])
IDS = np.array([0, 0, -1, -1, 1, -1])
N_LES = 3


@pytest.mark.parametrize("retained,cov,lsr,tep", [
    ([0, 2, 3], 1 / 3, 1 / 3, 1 / 3),
    ([0, 1, 4], 1.0, 2 / 3, 1.0),  # lesion 2 is missed even when every positive patch is kept
    ([1, 0], 2 / 3, 1 / 3, 1.0),  # two patches of the same lesion count once
    (range(6), 1.0, 2 / 3, 0.5),  # K > N: everything retained, purity over the 6 retained rows
])
def test_retention_hand_computed(retained, cov, lsr, tep):
    retained = list(retained)
    assert cov_at_k(retained, Y) == pytest.approx(cov)
    assert lsr_at_k(retained, IDS, N_LES) == pytest.approx(lsr)
    assert tep_at_k(retained, Y) == pytest.approx(tep)


def test_retention_zero_tumor_slide_is_undefined():
    y0, ids0 = np.zeros(4, int), np.full(4, -1)
    assert math.isnan(cov_at_k([0, 1], y0))
    assert math.isnan(lsr_at_k([0, 1], ids0, 0))
    assert tep_at_k([0, 1], y0) == 0.0


def test_retention_empty_selection():
    assert cov_at_k([], Y) == 0.0 and lsr_at_k([], IDS, N_LES) == 0.0
    assert math.isnan(tep_at_k([], Y))


W = np.array([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0],   # slot 0: all mass on tumor rows 0, 1
              [0.0, 0.0, 0.6, 0.4, 0.0, 0.0]])  # slot 1: all mass on normal rows 2, 3


def test_sdi_per_slot():
    np.testing.assert_allclose(sdi(W, Y), [0.0 / (1e-6 + 1.0), 1.0 / 1e-6])


def test_retained_from_assignment_rules():
    np.testing.assert_array_equal(retained_from_assignment(W), [0, 2])  # argmax tie in slot 0 -> row 0
    np.testing.assert_array_equal(retained_from_assignment(W, "threshold", 0.45), [0, 1, 2])
    with pytest.raises(ValueError):
        retained_from_assignment(W, "threshold")
    with pytest.raises(ValueError):
        retained_from_assignment(W, "nope")
    empty_slot = np.vstack([W, np.zeros(6)])  # a slot no patch was routed to must not "retain" row 0
    np.testing.assert_array_equal(retained_from_assignment(empty_slot), [0, 2])
    np.testing.assert_array_equal(retained_from_assignment(np.zeros((2, 6))), [])


def test_esi_hand_computed():
    # tumor [[2,0],[4,0]]: mean (3,0), per-dim var (1,0) -> 0.5
    # normal [[0,0],[0,2]]: mean (0,1), per-dim var (0,1) -> 0.5
    # ||(3,-1)||^2 = 10; 10 / (0.5 * (0.5 + 0.5)) = 20
    X = np.array([[2, 0], [4, 0], [0, 0], [0, 2]])
    assert esi(X, [1, 1, 0, 0]) == pytest.approx(20.0)
    assert math.isnan(esi(X, [0, 0, 0, 0])) and math.isnan(esi(X, [1, 1, 1, 1]))


def test_svm_patch_auc_separable_random_and_degenerate():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(40, int), np.zeros(200, int)]
    sep = rng.normal(size=(240, 16)) + 6.0 * y[:, None]
    assert svm_patch_auc(sep, y, seed=0) == 1.0
    noise = rng.normal(size=(240, 16))
    assert 0.25 < svm_patch_auc(noise, y, seed=0) < 0.75
    assert svm_patch_auc(noise, y, seed=1) == svm_patch_auc(noise, y, seed=1)  # deterministic per seed
    one_pos = np.r_[1, np.zeros(9, int)]
    assert math.isnan(svm_patch_auc(rng.normal(size=(10, 4)), one_pos, seed=0))
