"""Lesion-level analyses (docs/metrics.md Q1 and Q3): survival vs size (D50), normalised gain to the ceiling,
encoder- vs selector-stage loss, and Layer-of-Death for the in-encoder pruner.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


def d50(diam_um: np.ndarray, hit: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Diameter where the increasing isotonic fit of hit vs log d crosses 0.5, interpolated on log d.

    NaN if the fit never reaches 0.5; the smallest diameter if it is already >= 0.5 there (left-censored).
    Hits may be fractional (Random's exact hit probability). ``weight``: bootstrap draw counts (0 = left out).
    """
    w = np.ones(len(hit)) if weight is None else np.asarray(weight, dtype=float)
    keep = w > 0
    x = np.log(diam_um[keep])
    order = np.argsort(x, kind="stable")
    x = x[order]
    fit = IsotonicRegression(increasing=True).fit_transform(x, hit[keep][order], sample_weight=w[keep][order])
    above = np.nonzero(fit >= 0.5)[0]
    if len(above) == 0:
        return float("nan")
    j = above[0]
    if j == 0:
        return float(np.exp(x[0]))
    x0, x1, y0, y1 = x[j - 1], x[j], fit[j - 1], fit[j]
    return float(np.exp(x0 + (0.5 - y0) / (y1 - y0) * (x1 - x0)))


def nrs(lsr: float, random: float, oracle: float) -> float:
    """Normalised retention score: 0 = Random, 1 = the oracle ceiling (from aggregate rates)."""
    return (lsr - random) / (oracle - random) if oracle > random else float("nan")


def retention_auc(budgets, lsr) -> float:
    """Area under LSR vs log2 K, divided by the log2 range of the budget grid."""
    x = np.log2(np.asarray(budgets, dtype=float))
    y = np.asarray(lsr, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)) / (x[-1] - x[0]))


def attribute(probe_auc: np.ndarray, hittable: np.ndarray, hit: np.ndarray, cut: float) -> np.ndarray:
    """Per lesion: unhittable (no eligible patch), retained (hit), encoder (probe AUC < cut), selector loss,
    or unknown (lost but no probe AUC, e.g. no lesion-free patches on the slide)."""
    out = np.where(np.isnan(probe_auc), "unknown",
                   np.where(probe_auc < cut, "encoder", "selector")).astype(object)
    out[hit.astype(bool)] = "retained"
    out[~hittable.astype(bool)] = "unhittable"
    return out


def boot_share_diff(sel: np.ndarray, enc: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(B,) fold mean of (selector - encoder) / (selector + encoder) losses; sel, enc (F, S) counts per slide,
    W (B, S) slide resample counts (lesions of a slide stay together)."""
    s, e = W @ sel.T, W @ enc.T  # (B, F)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean((s - e) / (s + e), axis=1)


def survival_by_stage(layer_of_death: np.ndarray, n_stages: int) -> np.ndarray:
    """Fraction of lesions still holding a patch after each pruner stage (-1 = survived every stage).
    No lesion is censored (each is followed to the end), so this is the Kaplan-Meier curve."""
    lod = np.asarray(layer_of_death)
    return np.array([np.mean((lod == -1) | (lod > b)) for b in range(n_stages)])
