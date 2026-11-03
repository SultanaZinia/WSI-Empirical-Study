"""CAMELYON16 FROC, ported from the official evaluation code shipped with the data (evaluation/evaluate.py).

The logic is kept line for line: evaluation mask at level 5 (tumour value 2, dilated by half of 75 µm,
holes filled, 8-connected labels); ITC = region whose fitted-ellipse major axis (skimage
``axis_major_length``, formerly ``major_axis_length``) is < 200 + 75 µm; one hit per region at its highest
probability; every detection off tumour (and every detection on a normal slide) a false positive;
thresholds = every distinct probability except 0.0; score = mean sensitivity at 1/4, 1/2, 1, 2, 4 and
8 FP per slide by a step lookup. Only mask reading differs (the caller passes the level array and its
spacing; the official code reads them with ASAP). Probabilities must be > 0: a missed region is 0.0.

``detections_from_kept`` is ours (docs/deviations.md): local maxima of the kept patches.
"""

import bisect

import numpy as np
import scipy.ndimage
import skimage.measure

TUMOUR_VALUE = 2
DILATION_UM = 75.0
EVAL_LEVEL = 5
TARGET_FPS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def evaluation_mask(mask: np.ndarray, level_spacing_um: float, include_itcs: bool = False
                    ) -> tuple[np.ndarray, set]:
    """(label image, ITC labels) from a mask read at the evaluation level (``compute_evaluation_mask``)."""
    distance = scipy.ndimage.distance_transform_edt(np.not_equal(mask, TUMOUR_VALUE))
    binary = np.less(distance, DILATION_UM / (level_spacing_um * 2.0))
    labels = skimage.measure.label(scipy.ndimage.binary_fill_holes(binary), connectivity=2)
    if include_itcs:
        return labels, set()
    threshold = (200.0 + DILATION_UM) / level_spacing_um
    props = skimage.measure.regionprops(labels)
    return labels, {i + 1 for i in range(len(props)) if props[i].axis_major_length < threshold}


def to_level(detections, downsample: float) -> list[tuple[float, int, int]]:
    """Level-0 (p, x, y) -> (p, row, col) at the evaluation level (``load_detections``)."""
    return [(p, round(y / downsample), round(x / downsample)) for p, x, y in detections]


def slide_probabilities(detections, eval_mask, itc_labels) -> tuple[list, list, int]:
    """(FP probabilities, per-region TP probabilities (0.0 = missed), number of non-ITC regions) for one
    slide (``compute_probabilities``). ``eval_mask`` is None for a normal slide; ``detections`` are
    (p, row, col)."""
    if eval_mask is None:
        return [d[0] for d in detections], [], 0
    max_label = int(eval_mask.max())
    fp, tp = [], [0.0] * (max_label + 1)
    for p, r, c in detections:
        hit = eval_mask[r, c]
        if hit == 0:
            fp.append(p)
        elif hit not in itc_labels and tp[hit] < p:
            tp[hit] = p
    return fp, tp[1:], max_label - len(itc_labels)


def froc_curve(slides: list[tuple[list, list, int]]) -> tuple[list, list]:
    """(average FPs per slide, sensitivity) over every threshold (``compute_froc``)."""
    fps = [p for s in slides for p in s[0]]
    tps = [p for s in slides for p in s[1]]
    thresholds = sorted(set(fps + tps) - {0.0})
    fps, tps = np.asarray(fps, dtype=np.float64), np.asarray(tps, dtype=np.float64)
    total_fp = [int(np.greater_equal(fps, t).sum()) for t in thresholds] + [0]
    total_tp = [int(np.greater_equal(tps, t).sum()) for t in thresholds] + [0]
    n_tumours = sum(s[2] for s in slides)
    return [n / len(slides) for n in total_fp], [n / n_tumours for n in total_tp]


def froc_score(average_fps, sensitivities) -> float:
    """Mean sensitivity at 1/4 .. 8 FP per slide (``compute_score``)."""
    fps_r, sens_r = list(reversed(average_fps)), list(reversed(sensitivities))
    n = len(sens_r)
    return sum(sens_r[min(bisect.bisect_left(fps_r, t), n - 1)] for t in TARGET_FPS) / len(TARGET_FPS)


def detections_from_kept(kept, coords, patch_size_l0: int, probs) -> list[tuple[float, int, int]]:
    """Local maxima of the kept patches: a kept patch is a detection (probability, level-0 centre x, y) if its
    probability beats every kept 8-neighbour on the patch grid; ties go to the lower row. Isolated kept
    patches are their own detections; keeping every patch gives the heatmap's peaks. ``kept`` indexes
    ``coords`` (N, 2) and ``probs`` (N,)."""
    kept = np.asarray(kept, dtype=np.int64)
    if len(kept) == 0:
        return []
    coords = np.asarray(coords)
    cx, cy = ((coords[kept] - coords.min(axis=0)) // patch_size_l0).T
    p = np.asarray(probs, dtype=np.float64)[kept]
    prob = np.full((cy.max() + 3, cx.max() + 3), -np.inf)  # one cell of padding on every side
    row = np.full(prob.shape, np.iinfo(np.int64).max)
    prob[cy + 1, cx + 1], row[cy + 1, cx + 1] = p, kept
    peak = np.ones(len(kept), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                q, r = prob[cy + 1 + dy, cx + 1 + dx], row[cy + 1 + dy, cx + 1 + dx]
                peak &= (p > q) | ((p == q) & (kept < r))
    half = patch_size_l0 // 2
    return [(float(p[j]), int(coords[kept[j], 0]) + half, int(coords[kept[j], 1]) + half)
            for j in np.flatnonzero(peak)]
