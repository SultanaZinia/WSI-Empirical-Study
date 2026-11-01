"""Seeded, label-stratified K-fold cross-validation over the training slides."""

import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_folds(slide_ids, slide_labels, n_folds: int, seed: int) -> list[tuple[list[str], list[str]]]:
    """[(train_ids, val_ids)] per fold; every slide is in exactly one val fold."""
    ids = np.asarray(slide_ids)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(ids[tr].tolist(), ids[va].tolist()) for tr, va in skf.split(ids, np.asarray(slide_labels))]
