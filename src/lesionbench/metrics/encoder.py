"""Encoder separability on one slide: ESI and a linear-SVM patch AUC on raw embeddings."""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.svm import LinearSVC


def esi(features, patch_labels) -> float:
    """||mu_tumor - mu_normal||^2 / (0.5 * (var_tumor + var_normal)).

    var is the mean per-dimension population variance (ddof=0). NaN unless both classes are present.
    """
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(patch_labels).astype(bool)
    if y.all() or not y.any():
        return float("nan")
    t, n = X[y], X[~y]
    gap = np.sum((t.mean(axis=0) - n.mean(axis=0)) ** 2)
    return float(gap / (0.5 * (t.var(axis=0).mean() + n.var(axis=0).mean())))


def svm_patch_auc(features, patch_labels, seed: int, n_splits: int = 5) -> float:
    """Out-of-fold ROC-AUC of a linear SVM separating tumor from normal patches within one slide.

    Uses min(n_splits, n_tumor, n_normal) stratified folds; NaN if a class has fewer than 2 patches.
    """
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(patch_labels).astype(int)
    folds = min(n_splits, int(y.sum()), int((1 - y).sum()))
    if folds < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    clf = LinearSVC(class_weight="balanced", random_state=seed)
    scores = cross_val_predict(clf, X, y, cv=cv, method="decision_function")
    return float(roc_auc_score(y, scores))
