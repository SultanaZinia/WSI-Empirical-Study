"""Train one method on one CV fold at every budget K, then evaluate its out-of-fold (val) and test slides.

Writes OUT/<method>/fold<f>/K<K>/ with predictions.csv (slide P(tumour) from the kept patches), retention.csv
(Cov / LSR at every overlap threshold / TEP per slide), retained.npz (kept rows per slide; layer_of_drop for
in-encoder pruners; nonzeros of W for merging), history.json and, last, done.json (run provenance). A K
whose done.json exists is skipped, so a rerun resumes. ``abmil`` also writes the full-bag reference
``abmil_full/fold<f>/Kfull``.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np

from ..metrics.retention import cov_at_k, lsr_at_k, lsr_curve, tep_at_k
from ..splits import make_folds
from .data import Slide
from .methods import fit

PRED_FIELDS = ["slide_id", "split", "label", "prob"]
RET_FIELDS = ["slide_id", "split", "label", "stratum", "n_patches", "n_lesions", "n_kept",
              "cov", "lsr", "tep"]


def fold_split(slides: list[Slide], n_folds: int, seed: int, fold: int) -> tuple[list[Slide], list[Slide]]:
    train = [s for s in slides if s.split == "train"]
    tr_ids, va_ids = make_folds([s.slide_id for s in train], [s.label for s in train], n_folds, seed)[fold]
    by_id = {s.slide_id: s for s in train}
    return [by_id[i] for i in tr_ids], [by_id[i] for i in va_ids]


def _write_csv(path, fields, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def evaluate_run(fitted, slides: list[Slide], thresholds) -> tuple[list, list, list, dict]:
    """(prediction rows, retention rows, LSR column names, arrays for retained.npz) over ``slides``."""
    lsr_cols = {t: f"lsr_t{round(100 * t):02d}" for t in thresholds}
    preds, rets, arrays = [], [], {}
    for s in slides:
        r = fitted.select(s)
        idx = r.retained_idx.numpy()
        split = "test" if s.split == "test" else "val"
        preds.append({"slide_id": s.slide_id, "split": split, "label": s.label, "prob": fitted.prob(s, r)})
        curve = lsr_curve(idx, *s.overlap, s.n_lesions, thresholds)
        rets.append({"slide_id": s.slide_id, "split": split, "label": s.label, "stratum": s.stratum,
                     "n_patches": len(s.features), "n_lesions": s.n_lesions, "n_kept": len(idx),
                     "cov": cov_at_k(idx, s.patch_labels), "lsr": lsr_at_k(idx, s.lesion_ids, s.n_lesions),
                     "tep": tep_at_k(idx, s.patch_labels), **{lsr_cols[t]: v for t, v in curve.items()}})
        arrays[f"retained/{s.slide_id}"] = idx.astype(np.int64)
        if r.layer_of_drop is not None:
            arrays[f"layer_of_drop/{s.slide_id}"] = r.layer_of_drop.numpy().astype(np.int16)
        if r.W is not None:
            k, i = np.nonzero(r.W.numpy())
            arrays[f"W/{s.slide_id}"] = np.stack([k, i]).astype(np.int32)
            arrays[f"W_val/{s.slide_id}"] = r.W.numpy()[k, i].astype(np.float32)
    return preds, rets, list(lsr_cols.values()), arrays


def run_method_fold(method: str, fold: int, slides: list[Slide], cfg: dict, out_root, device,
                    provenance: dict, budgets=None, log=print) -> None:
    tcfg, seed = cfg["training"], int(cfg["seed"]) + fold
    train, val = fold_split(slides, tcfg["n_folds"], int(cfg["seed"]), fold)
    test = [s for s in slides if s.split == "test"]
    cache: dict = {}
    runs = [(method, K) for K in (budgets or cfg["budgets"])]
    if method == "abmil":
        runs.append(("abmil_full", "full"))
    for name, K in runs:
        out = Path(out_root) / name / f"fold{fold}" / f"K{K}"
        if (out / "done.json").exists():
            log(f"{name} fold {fold} K {K}: done, skipping")
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        fitted = fit(name, 0 if K == "full" else K, train, val, tcfg, seed, device, cache)
        t_fit = time.time() - t0
        preds, rets, lsr_fields, arrays = evaluate_run(fitted, val + test, cfg["lsr_overlap_thresholds"])
        _write_csv(out / "predictions.csv", PRED_FIELDS, preds)
        _write_csv(out / "retention.csv", RET_FIELDS + lsr_fields, rets)
        np.savez_compressed(out / "retained.npz", **arrays)
        (out / "history.json").write_text(json.dumps(fitted.history))
        (out / "done.json").write_text(json.dumps({
            **provenance, "method": name, "K": K, "fold": fold, "seed": seed, "n_train": len(train),
            "n_val": len(val), "n_test": len(test), "fit_seconds": round(t_fit, 1),
            "total_seconds": round(time.time() - t0, 1)}, indent=1))
        log(f"{name} fold {fold} K {K}: fit {t_fit:.0f} s, total {time.time() - t0:.0f} s")
