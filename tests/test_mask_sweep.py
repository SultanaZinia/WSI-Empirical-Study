import numpy as np

from lesionbench.preprocess.mask_qc import score_mask


def test_score_mask_counts_dropped_tumour_and_lost_lesions():
    cells = np.array([[0, 0], [10, 0], [20, 0], [30, 0]])
    frac = np.array([0.9, 0.05, 0.3, 0.0])  # cell 3 is empty under the candidate; cell 2 is new to it
    old = np.array([[0, 0], [10, 0], [30, 0]])  # current rows 0, 1, 2 -> cells 0, 1, 3
    # lesion 0: rows 0 (kept) and 1; lesion 1: row 1 only (row 2 is below the hit threshold); lesion 2: row 2
    rows = np.array([0, 1, 1, 2, 2])
    les = np.array([0, 0, 1, 2, 1])
    ov = np.array([0.5, 0.2, 0.06, 0.3, 0.01])
    diam = np.array([3000.0, 150.0, 900.0])
    out, lost = score_mask(old, cells, frac, rows, les, ov, diam, 0.05, cuts=(0.0, 0.10))
    r0, r1 = out
    assert (r0["n_new"], r0["n_dropped"], r0["n_added"]) == (3, 1, 1)
    assert (r0["tumour_patches"], r0["tumour_dropped"]) == (3, 1)
    assert (r0["lesions_hittable"], r0["lesions_lost"]) == (3, 1)
    assert (r1["tumour_dropped"], r1["lesions_lost"]) == (2, 2)  # row 1 (frac 0.05) now dropped
    assert [(d["cut"], d["lesion_id"]) for d in lost] == [(0.0, 2), (0.10, 1), (0.10, 2)]
    assert lost[1]["diameter_um"] == 150.0


def test_patch_fraction_looks_up_each_patch_cell():
    from lesionbench.preprocess.grid import patch_fraction

    cells = np.array([[0, 0], [10, 0], [0, 10], [10, 10]])
    frac = np.array([0.9, 0.05, 0.3, 0.0])
    coords = np.array([[10, 10], [0, 0], [10, 0]])
    assert patch_fraction(coords, cells, frac).tolist() == [0.0, 0.9, 0.05]
