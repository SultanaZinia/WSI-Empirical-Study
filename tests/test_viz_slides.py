import numpy as np
import pytest

from lesionbench.viz.slides import choose_slides, slot_tumour_fraction


def test_choose_slides_by_rule():
    info = [
        {"slide_id": "test_001", "stratum": "micro", "n_patches": 100, "n_micro": 1},
        {"slide_id": "test_002", "stratum": "micro", "n_patches": 300, "n_micro": 4},
        {"slide_id": "test_003", "stratum": "micro", "n_patches": 500, "n_micro": 4},
        {"slide_id": "test_004", "stratum": "macro", "n_patches": 50, "n_micro": 0},
        {"slide_id": "test_005", "stratum": "macro", "n_patches": 60, "n_micro": 2},
        {"slide_id": "test_006", "stratum": "negative", "n_patches": 10, "n_micro": 0},
    ]
    out = choose_slides(info)
    assert out["micro_median"] == "test_002"  # median 300
    assert out["micro_most_lesions"] == "test_002"  # tie on 4 lesions -> lower slide id
    assert out["macro_median"] == "test_004"  # median 55: both 5 away -> lower slide id
    assert out["negative_median"] == "test_006"


def test_slot_tumour_fraction_hand_computed():
    # slot 0: patch 0 (w .5, tumour) + patch 1 (w .5) -> 0.5; slot 1: patch 1 (w .25) + patch 2 (w 1) -> 0
    k = np.array([0, 0, 1, 1])
    i = np.array([0, 1, 1, 2])
    w = np.array([0.5, 0.5, 0.25, 1.0])
    out = slot_tumour_fraction(k, i, w, labels=np.array([1, 0, 0, 0]), n=4)
    assert out[:3] == pytest.approx([0.5, 0.5, 0.0])  # patch 1 contributes most to slot 0
    assert np.isnan(out[3])
