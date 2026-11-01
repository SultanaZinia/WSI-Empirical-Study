"""Patch labels and lesion ids from lesion polygons."""

import logging

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

log = logging.getLogger(__name__)


def lesion_overlaps(
    coords: np.ndarray, patch_size_l0: int, lesion_polys: list[BaseGeometry]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (patch, lesion) pair with overlap > 0, ordered by lesion then patch row.

    Returns (patch rows int64, lesion ids int32, overlap float32), where overlap is
    ``Area(patch ∩ lesion) / Area(patch)``. Evaluation-only: lets LSR be computed at any overlap threshold.
    """
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 2)
    ps = patch_size_l0
    boxes = shapely.box(coords[:, 0], coords[:, 1], coords[:, 0] + ps, coords[:, 1] + ps)
    tree = shapely.STRtree(boxes)
    rows, ids, fracs = [], [], []
    for m, lesion in enumerate(lesion_polys):
        idx = np.sort(tree.query(lesion, predicate="intersects"))
        frac = shapely.area(shapely.intersection(boxes[idx], lesion)) / (ps * ps)
        keep = frac > 0
        rows.append(idx[keep])
        ids.append(np.full(keep.sum(), m))
        fracs.append(frac[keep])
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32)
    return (np.concatenate(rows).astype(np.int64), np.concatenate(ids).astype(np.int32),
            np.concatenate(fracs).astype(np.float32))


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
    n = len(np.asarray(coords).reshape(-1, 2))
    lesion_ids = np.full(n, -1, dtype=np.int32)
    best = np.zeros(n)
    rows, ids, fracs = lesion_overlaps(coords, patch_size_l0, lesion_polys)
    for r, m, frac in zip(rows, ids, fracs.astype(float), strict=True):
        if frac >= overlap_threshold and frac > best[r]:
            best[r] = frac
            lesion_ids[r] = m

    unlabeled = sorted(set(range(len(lesion_polys))) - set(lesion_ids[lesion_ids >= 0].tolist()))
    for m in unlabeled:
        log.warning("lesion %d (area %.0f px^2) has no positive patch", m, lesion_polys[m].area)
    return (lesion_ids >= 0).astype(np.uint8), lesion_ids, unlabeled
