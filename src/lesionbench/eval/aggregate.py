"""Collect runner outputs (runs/<backbone>/<method>/fold<f>/K<K>/) into tidy tables.

- slide_auc: per method x K x fold x split, slide ROC-AUC; plus "val_oof" = the 5 validation folds pooled
  (every training slide once, out of fold).
- lesion_hits: per method x K x fold x lesion, whether a kept patch overlaps the lesion by >= the primary and
  the strict threshold, with lesion size and the slide's exhaustive flag. Lesion-pooled LSR comes from here.
- retention_means: per method x K x fold x split x stratum, slide means of Cov / LSR / TEP (primary).
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from ..preprocess.annotations import size_class


def run_dirs(root):
    """(method, K, fold, path) for every finished run (done.json present)."""
    out = []
    for done in sorted(Path(root).glob("*/fold*/K*/done.json")):
        d = done.parent
        out.append((d.parent.parent.name, d.name[1:], int(d.parent.name[4:]), d))
    return out


def read_csv(path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _auc(rows) -> float:
    y = [int(r["label"]) for r in rows]
    return float(roc_auc_score(y, [float(r["prob"]) for r in rows])) if 0 < sum(y) < len(y) else float("nan")


def slide_auc(runs) -> list[dict]:
    out, oof = [], defaultdict(list)
    for method, K, fold, d in runs:
        preds = read_csv(d / "predictions.csv")
        for split in ("val", "test"):
            rows = [r for r in preds if r["split"] == split]
            out.append({"method": method, "K": K, "fold": fold, "split": split, "n": len(rows),
                        "auc": _auc(rows)})
        oof[(method, K)] += [r for r in preds if r["split"] == "val"]
    for (method, K), rows in oof.items():
        out.append({"method": method, "K": K, "fold": "all", "split": "val_oof", "n": len(rows),
                    "auc": _auc(rows)})
    return out


def lesion_hits(runs, overlaps: dict, primary: float, strict: float, non_exhaustive, strata) -> list[dict]:
    """``overlaps[slide_id]`` = (rows, lesion ids, fracs, diameters in um) from the patch file."""
    non_exh, out = set(non_exhaustive), []
    for method, K, fold, d in runs:
        kept = np.load(d / "retained.npz")
        split_of = {r["slide_id"]: r["split"] for r in read_csv(d / "predictions.csv")}
        for sid, split in split_of.items():
            rows, les, fracs, diam = overlaps[sid]
            if len(diam) == 0:
                continue
            m = np.isin(rows, kept[f"retained/{sid}"])
            for lid, dm in enumerate(diam):
                own = les == lid
                out.append({"method": method, "K": K, "fold": fold, "split": split, "slide_id": sid,
                            "lesion_id": lid, "diameter_um": float(dm),
                            "size_class": size_class(dm / 1000, strata["itc_max"], strata["micro_max"]),
                            "exhaustive": int(sid not in non_exh),
                            "hittable": int((own & (fracs >= primary)).any()),
                            "hit": int((own & m & (fracs >= primary)).any()),
                            "hit_strict": int((own & m & (fracs >= strict)).any())})
    return out


def retention_means(runs) -> list[dict]:
    out = []
    for method, K, fold, d in runs:
        groups = defaultdict(list)
        for r in read_csv(d / "retention.csv"):
            groups[(r["split"], r["stratum"])].append(r)
        for (split, stratum), rs in sorted(groups.items()):
            row = {"method": method, "K": K, "fold": fold, "split": split, "stratum": stratum,
                   "n_slides": len(rs)}
            for col in ("cov", "lsr", "tep", "cov_strict", "tep_strict"):
                v = np.array([float(r[col]) for r in rs])
                row[col] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            out.append(row)
    return out


def pooled_lsr(hits, *, split: str, size: str, exhaustive_only: bool, key: str = "hit") -> dict:
    """(method, K) -> list over folds of lesion-pooled LSR (hits / lesions) for one size class."""
    acc = defaultdict(lambda: defaultdict(list))
    for h in hits:
        if h["split"] == split and h["size_class"] == size and (h["exhaustive"] or not exhaustive_only):
            acc[(h["method"], h["K"])][h["fold"]].append(h[key])
    return {k: [float(np.mean(v)) for _, v in sorted(f.items())] for k, f in acc.items()}


def mean_sd(values) -> str:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return "–"
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"
