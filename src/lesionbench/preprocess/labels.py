"""Patch labels and lesion ids from lesion polygons."""

import logging

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

log = logging.getLogger(__name__)


def label_patches(
    coords: np.ndarray,
    patch_size_l0: int,
    lesion_polys: list[BaseGeometry],
    overlap_threshold: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Label each patch from its overlap with the lesions.

    A patch is positive when ``Area(patch ∩ lesion) / Area(patch) >= overlap_threshold``. Its ``lesion_id``
    is the lesion with the largest overlap (ties go to the lower id); negatives get -1.

    Returns (patch_labels uint8 (N,), lesion_ids int32 (N,), ids of lesions left with no positive patch).
    Those lesions are logged, not dropped: they stay in lesion-level denominators.
    """
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 2)
    n = len(coords)
    lesion_ids = np.full(n, -1, dtype=np.int32)
    best = np.zeros(n)
    ps = patch_size_l0
    boxes = shapely.box(coords[:, 0], coords[:, 1], coords[:, 0] + ps, coords[:, 1] + ps)
    tree = shapely.STRtree(boxes)

    for m, lesion in enumerate(lesion_polys):
        idx = tree.query(lesion, predicate="intersects")
        if len(idx) == 0:
            continue
        frac = shapely.area(shapely.intersection(boxes[idx], lesion)) / (ps * ps)
        win = (frac >= overlap_threshold) & (frac > best[idx])
        best[idx[win]] = frac[win]
        lesion_ids[idx[win]] = m

    unlabeled = sorted(set(range(len(lesion_polys))) - set(lesion_ids[lesion_ids >= 0].tolist()))
    for m in unlabeled:
        log.warning("lesion %d (area %.0f px^2) has no positive patch", m, lesion_polys[m].area)
    return (lesion_ids >= 0).astype(np.uint8), lesion_ids, unlabeled
