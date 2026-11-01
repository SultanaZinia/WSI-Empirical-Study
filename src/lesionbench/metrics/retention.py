"""Evidence-retention metrics for one slide at one budget K.

``retained`` is an index array into the slide's cached patch rows. Metrics that are undefined on a slide
(no tumor patches, no lesions) return NaN, so aggregation can exclude them explicitly.
"""

import numpy as np


def cov_at_k(retained, patch_labels) -> float:
    """Patch-level tumor recall: retained positives / all positives. NaN on slides without positives."""
    y = np.asarray(patch_labels)
    n_pos = y.sum()
    return float(y[np.asarray(retained, dtype=int)].sum() / n_pos) if n_pos else float("nan")


def lsr_at_k(retained, lesion_ids, n_lesions: int) -> float:
    """Fraction of lesions with at least one retained patch. NaN on slides without lesions.

    The denominator is ``n_lesions`` (all lesions on the slide), so lesions that have no positive patch
    count as missed by every method, Oracle included.
    """
    if n_lesions == 0:
        return float("nan")
    hit = np.unique(np.asarray(lesion_ids)[np.asarray(retained, dtype=int)])
    return float((hit >= 0).sum() / n_lesions)


def lsr_curve(retained, overlap_rows, overlap_lesions, overlap_fracs, n_lesions: int, thresholds) -> dict:
    """LSR@K at several hit thresholds, from the stored (patch, lesion, overlap) pairs.

    Lesion m is hit at threshold t if a retained patch overlaps it by ``>= t``; t = 0 means any overlap (> 0).
    Same denominator as ``lsr_at_k``: all lesions on the slide. NaN on slides without lesions.
    """
    if n_lesions == 0:
        return {t: float("nan") for t in thresholds}
    kept = np.isin(np.asarray(overlap_rows), np.asarray(retained, dtype=int))
    les, frac = np.asarray(overlap_lesions)[kept], np.asarray(overlap_fracs, dtype=float)[kept]
    return {t: float(len(np.unique(les[frac >= t] if t > 0 else les)) / n_lesions) for t in thresholds}


def tep_at_k(retained, patch_labels) -> float:
    """Purity of the retained set: retained positives / number retained (= K unless K > N)."""
    retained = np.asarray(retained, dtype=int)
    if len(retained) == 0:
        return float("nan")
    return float(np.asarray(patch_labels)[retained].sum() / len(retained))


def sdi(W, patch_labels, eps: float = 1e-6) -> np.ndarray:
    """Slot dilution index per slot: normal mass / (eps + tumor mass) of each row of ``W`` (K, N)."""
    W = np.asarray(W, dtype=float)
    y = np.asarray(patch_labels).astype(bool)
    return W[:, ~y].sum(axis=1) / (eps + W[:, y].sum(axis=1))


def retained_from_assignment(W, rule: str = "argmax", threshold: float | None = None) -> np.ndarray:
    """Hard retention for merging methods: sorted unique patch rows a slot ``W`` (K, N) keeps.

    ``argmax``: each non-empty slot keeps its largest contributor (an all-zero slot keeps nothing).
    ``threshold``: every patch with ``W[k, i] > threshold`` in any slot.
    """
    W = np.asarray(W, dtype=float)
    if rule == "argmax":
        return np.unique(W[W.any(axis=1)].argmax(axis=1))
    if rule == "threshold":
        if threshold is None:
            raise ValueError("rule='threshold' needs a threshold")
        return np.unique(np.nonzero(W > threshold)[1])
    raise ValueError(f"unknown rule {rule!r}")
