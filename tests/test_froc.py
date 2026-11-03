import numpy as np
import pytest

from lesionbench.metrics.froc import (
    detections_from_kept,
    evaluation_mask,
    froc_curve,
    froc_score,
    slide_probabilities,
    to_level,
)


def test_evaluation_mask_merges_within_75um_and_flags_itcs():
    # 10 µm per pixel: dilation keeps distance < 3.75 px; ITC if major axis < 27.5 px
    m = np.ones((100, 100), np.uint8)  # 1 = normal tissue, 2 = tumour
    m[10:13, 10:13] = 2  # small lesion ...
    m[10:13, 19:22] = 2  # ... 6 px (60 µm) away: merges with it
    m[50:53, 10:13] = 2  # 9 px (90 µm) from the next: stays apart
    m[50:53, 22:25] = 2
    m[80:85, 20:70] = 2  # 50 px long: not an ITC
    labels, itc = evaluation_mask(m, level_spacing_um=10.0)
    assert labels.max() == 4
    assert labels[11, 11] == labels[11, 20] != 0
    assert labels[51, 11] != labels[51, 23]
    big = labels[82, 40]
    assert big not in itc and {labels[11, 11], labels[51, 11], labels[51, 23]} == itc
    assert evaluation_mask(m, 10.0, include_itcs=True)[1] == set()


def test_evaluation_mask_fills_holes():
    m = np.ones((60, 60), np.uint8)
    m[10:50, 10:50] = 2
    m[20:40, 20:40] = 1  # a 20 px hole, wider than the dilation closes
    labels, _ = evaluation_mask(m, 10.0)
    assert labels[30, 30] == labels[15, 15] == 1


def test_to_level_rounds_level0_xy_to_row_col():
    assert to_level([(0.9, 100, 50)], 32.0) == [(0.9, 2, 3)]


def test_slide_probabilities_one_hit_per_region_itcs_ignored():
    mask = np.zeros((10, 10), int)
    mask[0:3, 0:3] = 1
    mask[6:9, 6:9] = 2
    dets = [(0.3, 1, 1), (0.7, 2, 2), (0.9, 7, 7), (0.5, 5, 0)]
    fp, tp, n = slide_probabilities(dets, mask, itc_labels={2})
    assert fp == [0.5] and tp == [0.7, 0.0] and n == 1


def test_slide_probabilities_normal_slide_all_false_positives():
    assert slide_probabilities([(0.4, 0, 0), (0.6, 3, 3)], None, set()) == ([0.4, 0.6], [], 0)


def test_froc_curve_and_score_by_hand():
    tumour = ([0.5], [0.7, 0.4], 2)  # one FP, two regions hit at 0.7 and 0.4
    normal = ([0.6, 0.2], [], 0)
    fps, sens = froc_curve([tumour, normal])
    assert fps == [1.5, 1.0, 1.0, 0.5, 0.0, 0.0]  # thresholds 0.2, 0.4, 0.5, 0.6, 0.7, then none
    assert sens == [1.0, 1.0, 0.5, 0.5, 0.5, 0.0]
    # step lookup: 1/4, 1/2 and 1 FP read 0.5; 2, 4 and 8 FP run past the end and read 1.0
    assert froc_score(fps, sens) == pytest.approx(0.75)


def test_froc_curve_ignores_missed_regions_as_thresholds():
    fps, sens = froc_curve([([], [0.8, 0.0], 2)])
    assert fps == [0.0, 0.0] and sens == [0.5, 0.0]


def test_detections_from_kept_groups_touching_patches():
    ps = 10
    coords = np.array([[0, 0], [10, 10], [50, 0], [20, 0]])
    probs = np.array([0.2, 0.8, 0.5, 0.99])  # row 3 touches row 1 but is not kept
    dets = detections_from_kept([0, 1, 2], coords, ps, probs)
    assert sorted(dets) == [(0.5, 55, 5), (0.8, 15, 15)]
    assert detections_from_kept([], coords, ps, probs) == []
