"""Linear probe on frozen encoder features: an evaluation instrument (docs/metrics.md, Q3), never a method.

Per CV fold, a logistic-regression probe is trained on the fold's training slides: clear tumour patches
(max lesion overlap >= ``pos_min``) against patches of normal slides. Scored on held-out slides, it gives
(1) detectability: the share of tumour patches flagged, by overlap band, at the cut-off that flags only 5% of
held-out normal-slide patches; (2) per lesion: probe AUC of the lesion's patches against the same slide's
lesion-free patches, and probe ESI along the probe direction.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BANDS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.01)]  # [lo, hi) of max overlap


def max_overlap(n_patches: int, rows, fracs) -> np.ndarray:
    """(N,) largest lesion overlap of each patch; 0 for patches touching no lesion."""
    out = np.zeros(n_patches)
    np.maximum.at(out, np.asarray(rows, dtype=int), np.asarray(fracs, dtype=float))
    return out


def probe_training_set(slides, rng, pos_min: float = 0.5, cap: int = 2000):
    """(X, y): up to ``cap`` patches per tumour slide with overlap >= pos_min (y = 1) and per normal slide
    (y = 0). Patches of tumour slides below pos_min are not used (their label depends on the threshold)."""
    X, y = [], []
    for s in slides:
        feats = s.features.numpy()
        if s.label:
            idx = np.flatnonzero(max_overlap(len(feats), s.overlap[0], s.overlap[2]) >= pos_min)
        else:
            idx = np.arange(len(feats))
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        X.append(feats[idx])
        y.append(np.full(len(idx), int(s.label)))
    return np.concatenate(X), np.concatenate(y)


class Probe:
    """Standardised logistic regression; ``score(X)`` is the decision value x.w + b."""

    def __init__(self, X, y, seed: int):
        self.scaler = StandardScaler().fit(X)
        self.clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        self.clf.fit(self.scaler.transform(X), y)

    def score(self, X) -> np.ndarray:
        return self.clf.decision_function(self.scaler.transform(np.asarray(X)))


def detection_by_band(tumour_scores, tumour_overlap, normal_scores, specificity: float = 0.95,
                      bands=BANDS) -> tuple[float, list[dict]]:
    """Cut-off = the ``specificity`` quantile of normal-slide scores; per band of max overlap (patches with
    overlap > 0 only), the share of patches scoring above it."""
    tau = float(np.quantile(np.asarray(normal_scores), specificity))
    s, ov = np.asarray(tumour_scores), np.asarray(tumour_overlap)
    rows = []
    for lo, hi in bands:
        m = (ov > 0) & (ov >= lo) & (ov < hi)
        rows.append({"band_lo": lo, "band_hi": min(hi, 1.0), "n_patches": int(m.sum()),
                     "detected": float((s[m] > tau).mean()) if m.any() else float("nan")})
    return tau, rows


def separation(pos, neg) -> tuple[float, float]:
    """(AUC, ESI) of one-dimensional scores: ESI = (mean_pos - mean_neg)^2 / (0.5 (var_pos + var_neg)),
    population variances. ESI is NaN with fewer than 2 positives; both NaN without positives or negatives."""
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
    if len(pos) < 2:
        return auc, float("nan")
    denom = 0.5 * (pos.var() + neg.var())
    return auc, float((pos.mean() - neg.mean()) ** 2 / denom) if denom > 0 else float("inf")


def lesion_separation(scores, rows, lesions, fracs, n_lesions: int, thresholds) -> list[dict]:
    """Per lesion and threshold t: n_patches (H_l = patches overlapping the lesion by >= t; > 0 when t = 0),
    probe AUC and ESI of H_l against the slide's patches that touch no lesion."""
    scores = np.asarray(scores)
    rows, lesions, fracs = np.asarray(rows, dtype=int), np.asarray(lesions), np.asarray(fracs, dtype=float)
    clean = np.ones(len(scores), dtype=bool)
    clean[rows] = False
    out = []
    for m in range(n_lesions):
        for t in thresholds:
            sel = (lesions == m) & ((fracs >= t) if t > 0 else (fracs > 0))
            auc, e = separation(scores[np.unique(rows[sel])], scores[clean])
            out.append({"lesion_id": m, "threshold": t, "n_patches": int(len(np.unique(rows[sel]))),
                        "probe_auc": auc, "probe_esi": e})
    return out
