"""Slide-level inference: bootstrap over test slides, paired tests against a reference method, Holm.

The resampling unit is the slide (one slide per patient on CAMELYON16). A replicate is a (n,) vector of draw
counts. Each replicate scores every fold model on the same resample and averages over folds, so the point
estimate (all counts 1) is the fold mean of Table 1 and the CI reflects test-set sampling; training
variability is the fold SD, reported separately. Lesion-level rates resample whole slides, so lesions of one
slide stay together. Methods compared on the same replicates are paired.
"""

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score


def resample_weights(n: int, B: int, seed: int) -> np.ndarray:
    """(B, n) counts: how often each of n slides is drawn in each of B replicates."""
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    W = np.zeros((B, n), dtype=np.int64)
    np.add.at(W, (np.repeat(np.arange(B), n), idx.ravel()), 1)
    return W


def boot_auc(y: np.ndarray, probs: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(B,) fold-mean slide AUC; probs (F, n), W (B, n). NaN when a replicate lacks a class."""
    out = np.full(len(W), np.nan)
    for b, w in enumerate(W):
        if (w * y).sum() > 0 and (w * (1 - y)).sum() > 0:
            out[b] = np.mean([roc_auc_score(y, p, sample_weight=w) for p in probs])
    return out


def slide_mean(values: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(B,) mean of per-slide values (already averaged over folds); NaN slides (undefined) are skipped."""
    ok = np.isfinite(values)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W[:, ok] @ values[ok]) / W[:, ok].sum(axis=1)


def pooled_rate(hits: np.ndarray, totals: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(B,) fold mean of lesion-pooled rate sum(hits) / sum(lesions); hits (F, S), totals (S,), W (B, S)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((W @ hits.T) / (W @ totals)[:, None]).mean(axis=1)


def percentile_ci(boot: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = 100 * (1 - level) / 2
    lo, hi = np.nanpercentile(boot, [a, 100 - a])
    return float(lo), float(hi)


def boot_p(delta: np.ndarray) -> float:
    """Two-sided paired-bootstrap p for H0: difference = 0, with the +1 correction (never exactly 0)."""
    d = delta[np.isfinite(delta)]
    tail = min((d <= 0).sum(), (d >= 0).sum())
    return float(min(1.0, 2 * (tail + 1) / (len(d) + 1)))


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p on slides where both values are defined; 1.0 if all pairs tie."""
    ok = np.isfinite(a) & np.isfinite(b)
    if not (a[ok] != b[ok]).any():
        return 1.0
    return float(wilcoxon(a[ok], b[ok]).pvalue)


def holm(pvals) -> list[float]:
    """Holm step-down adjusted p values (same order as the input); NaN entries are left out of the family."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    idx = np.where(np.isfinite(p))[0]
    order = idx[np.argsort(p[idx], kind="stable")]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        out[i] = running
    return out.tolist()
