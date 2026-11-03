"""Slide-level results at a fixed operating point, by slide stage (and centre), for test and external slides.

The threshold of each fold model is set on its own CAMELYON16 validation fold (out of fold): the
``spec``-quantile of the validation normal slides' P(tumour), so the evaluated slides never influence it.
A slide is called positive if P(tumour) > threshold.
"""

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

STAGES = ("negative", "itc", "micro", "macro")


def spec_threshold(normal_probs, spec: float = 0.95) -> float:
    """Smallest observed value t with at least ``spec`` of ``normal_probs`` <= t."""
    v = np.sort(np.asarray(normal_probs, dtype=float))
    return float(v[int(np.ceil(spec * len(v))) - 1])


def slide_metrics(rows, threshold: float, stage_of: dict, group_of: dict | None = None) -> list[dict]:
    """Per group ("all", plus each value of ``group_of``): n, AUC (negative vs any tumour), specificity on
    negative slides and sensitivity per tumour stage at ``threshold``. ``rows``: dicts with slide_id and
    prob; ``stage_of``: slide -> stage in STAGES."""
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        if group_of is not None:
            groups[str(group_of[r["slide_id"]])].append(r)
    out = []
    for g, rs in groups.items():
        p = np.array([float(r["prob"]) for r in rs])
        st = np.array([stage_of[r["slide_id"]] for r in rs])
        y = st != "negative"
        row = {"group": g, "n": len(rs), "threshold": threshold,
               "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
               "specificity": float(np.mean(p[~y] <= threshold)) if (~y).any() else float("nan")}
        for s in STAGES[1:]:
            m = st == s
            row[f"n_{s}"] = int(m.sum())
            row[f"sens_{s}"] = float(np.mean(p[m] > threshold)) if m.any() else float("nan")
        out.append(row)
    return out
