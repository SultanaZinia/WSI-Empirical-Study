"""Train one method on one CV fold at every budget K, then evaluate its out-of-fold (val) and test slides.

Writes OUT/<method>/fold<f>/K<K>/ with predictions.csv (slide P(tumour) from the kept patches), retention.csv
(Cov / LSR at every overlap threshold / TEP per slide), retained.npz (kept rows per slide; per-patch selection
scores (fp16) when the selector has them; layer_of_drop and per-stage pruner scores for in-encoder pruners;
nonzeros of W for merging), model.pt (state_dict of the fitted model), history.json and, last, done.json
(run provenance). ``pmil`` also writes the per-fold max-MIL head ``pmil/fold<f>/maxmil_head.pt``. A K
whose done.json exists is skipped, so a rerun resumes. ``abmil`` also writes the full-bag reference
``abmil_full/fold<f>/Kfull``.

``run_external`` applies a finished fold's saved models, without training, to slides of another dataset.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..metrics.probe import max_overlap
from ..metrics.retention import cov_at_k, lsr_curve, tep_at_k
from ..splits import make_folds
from .data import Slide
from .methods import fit

PRED_FIELDS = ["slide_id", "split", "label", "prob"]
RET_FIELDS = ["slide_id", "split", "label", "stratum", "n_patches", "n_lesions", "n_kept",
              "cov", "lsr", "tep", "cov_strict", "tep_strict"]


def fold_split(slides: list[Slide], n_folds: int, seed: int, fold: int,
               exclude=()) -> tuple[list[Slide], list[Slide]]:
    """(fold training slides, fold validation slides). ``exclude`` slides are in neither."""
    train = [s for s in slides if s.split == "train" and s.slide_id not in set(exclude)]
    tr_ids, va_ids = make_folds([s.slide_id for s in train], [s.label for s in train], n_folds, seed)[fold]
    by_id = {s.slide_id: s for s in train}
    return [by_id[i] for i in tr_ids], [by_id[i] for i in va_ids]


def _write_csv(path, fields, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def evaluate_run(fitted, slides: list[Slide], thresholds, primary: float, strict: float
                 ) -> tuple[list, list, list, dict]:
    """(prediction rows, retention rows, LSR column names, arrays for retained.npz) over ``slides``.

    cov / lsr / tep use the primary overlap threshold (``s.patch_labels`` must be labelled at it); the
    *_strict columns and lsr_tXX give the other thresholds."""
    lsr_cols = {t: f"lsr_t{round(100 * t):02d}" for t in thresholds}
    preds, rets, arrays = [], [], {}
    for s in slides:
        r = fitted.select(s)
        idx = r.retained_idx.numpy()
        split = s.split if s.split in ("test", "external") else "val"
        preds.append({"slide_id": s.slide_id, "split": split, "label": s.label, "prob": fitted.prob(s, r)})
        curve = lsr_curve(idx, *s.overlap, s.n_lesions, sorted({*thresholds, primary}))
        y_strict = max_overlap(len(s.features), s.overlap[0], s.overlap[2]) >= strict
        rets.append({"slide_id": s.slide_id, "split": split, "label": s.label, "stratum": s.stratum,
                     "n_patches": len(s.features), "n_lesions": s.n_lesions, "n_kept": len(idx),
                     "cov": cov_at_k(idx, s.patch_labels), "lsr": curve[primary],
                     "tep": tep_at_k(idx, s.patch_labels), "cov_strict": cov_at_k(idx, y_strict),
                     "tep_strict": tep_at_k(idx, y_strict), **{lsr_cols[t]: curve[t] for t in thresholds}})
        arrays[f"retained/{s.slide_id}"] = idx.astype(np.int64)
        if r.scores is not None:
            arrays[f"scores/{s.slide_id}"] = r.scores.numpy().astype(np.float16)
        if r.stage_scores is not None:
            arrays[f"stage_scores/{s.slide_id}"] = r.stage_scores.numpy().astype(np.float16)
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
    train, val = fold_split(slides, tcfg["n_folds"], int(cfg["seed"]), fold,
                            cfg.get("exclude_from_training", []))
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
        preds, rets, lsr_fields, arrays = evaluate_run(fitted, val + test, cfg["lsr_overlap_thresholds"],
                                                       cfg["primary_overlap_threshold"],
                                                       cfg["strict_overlap_threshold"])
        _write_csv(out / "predictions.csv", PRED_FIELDS, preds)
        _write_csv(out / "retention.csv", RET_FIELDS + lsr_fields, rets)
        np.savez_compressed(out / "retained.npz", **arrays)
        torch.save(fitted.model.state_dict(), out / "model.pt")
        (out / "history.json").write_text(json.dumps(fitted.history))
        (out / "done.json").write_text(json.dumps({
            **provenance, "method": name, "K": K, "fold": fold, "seed": seed, "n_train": len(train),
            "n_val": len(val), "n_test": len(test), "fit_seconds": round(t_fit, 1),
            "total_seconds": round(time.time() - t0, 1)}, indent=1))
        log(f"{name} fold {fold} K {K}: fit {t_fit:.0f} s, total {time.time() - t0:.0f} s")
    if "pmil" in cache:  # per-fold max-MIL instance head, before PMIL's per-K fine-tuning (a FROC scorer)
        torch.save(cache["pmil"][0].state_dict(), Path(out_root) / "pmil" / f"fold{fold}" / "maxmil_head.pt")


def run_external(method: str, fold: int, slides: list[Slide], external: list[Slide], cfg: dict, source_root,
                 out_root, device, provenance: dict, budgets=None, log=print) -> None:
    """External validation: rebuild each (method, K) of ``fold`` from its saved model.pt under
    ``source_root`` (no training; normal key banks are rebuilt from the same fold training slides of
    ``slides``) and evaluate ``external`` slides. Writes predictions.csv, retention.csv, retained.npz and
    done.json under ``out_root``, as ``run_method_fold`` does; finished K are skipped."""
    tcfg, seed = cfg["training"], int(cfg["seed"]) + fold
    train, _ = fold_split(slides, tcfg["n_folds"], int(cfg["seed"]), fold,
                          cfg.get("exclude_from_training", []))
    cache: dict = {}
    runs = [(method, K) for K in (budgets or cfg["budgets"])]
    if method == "abmil":
        runs.append(("abmil_full", "full"))
    for name, K in runs:
        src = Path(source_root) / name / f"fold{fold}" / f"K{K}"
        out = Path(out_root) / name / f"fold{fold}" / f"K{K}"
        if (out / "done.json").exists():
            log(f"{name} fold {fold} K {K}: done, skipping")
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        state = torch.load(src / "model.pt", map_location=device)
        fitted = fit(name, 0 if K == "full" else K, train, [], tcfg, seed, device, cache, state=state)
        preds, rets, lsr_fields, arrays = evaluate_run(fitted, external, cfg["lsr_overlap_thresholds"],
                                                       cfg["primary_overlap_threshold"],
                                                       cfg["strict_overlap_threshold"])
        _write_csv(out / "predictions.csv", PRED_FIELDS, preds)
        _write_csv(out / "retention.csv", RET_FIELDS + lsr_fields, rets)
        np.savez_compressed(out / "retained.npz", **arrays)
        (out / "done.json").write_text(json.dumps({
            **provenance, "method": name, "K": K, "fold": fold, "seed": seed, "source": str(src),
            "n_external": len(external), "total_seconds": round(time.time() - t0, 1)}, indent=1))
        log(f"{name} fold {fold} K {K}: {len(external)} external slides, {time.time() - t0:.0f} s")
