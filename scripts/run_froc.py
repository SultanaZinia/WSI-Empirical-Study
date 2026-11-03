"""CAMELYON16 FROC of every method's kept patches, under the shared tumour scorers (docs/deviations.md).

    python scripts/run_froc.py --config configs/server.yaml --backbone uni --scorer-auc
    python scripts/run_froc.py --config configs/server.yaml --backbone uni

--scorer-auc: patch-level AUC of each scorer on the out-of-fold validation slides (tumour patches at the
primary overlap threshold vs every patch of the normal slides); the higher one is the primary scorer. Writes
OUTPUT_ROOT/froc/<backbone>/scorer_auc.csv.

Default: for each scorer and fold, every test patch is scored; each run's kept patches
(OUTPUT_ROOT/runs/<backbone>/<method>/fold<f>/K<K>/retained.npz; ``abmil_full`` Kfull = the full bag) become
detections (local maxima) and are evaluated with the ported official FROC, without ITCs (challenge rule) and
with them (the official --itc option). Writes OUTPUT_ROOT/froc/<backbone>/<scorer>/{runs.csv, summary.csv,
curves.npz} and run.json.
"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from lesionbench.config import load_config, write_run_json
from lesionbench.eval.data import load_slide, slide_ids
from lesionbench.eval.froc import SCORERS, load_scorer, read_mask
from lesionbench.eval.runner import fold_split
from lesionbench.metrics.froc import (
    EVAL_LEVEL,
    TARGET_FPS,
    detections_from_kept,
    evaluation_mask,
    froc_curve,
    froc_score,
    slide_probabilities,
    to_level,
)

log = logging.getLogger("run_froc")


def _sd(v):
    return np.std(v, ddof=1) if len(v) > 1 else float("nan")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def scorer_auc(cfg, backbone, run_root, out_dir):
    cache, t = cfg["paths"]["cache_root"], cfg["primary_overlap_threshold"]
    train = [SimpleNamespace(slide_id=s, split="train", label=int(s.startswith("tumor_")))
             for s in slide_ids(cache, backbone) if not s.startswith("test_")]
    rows = []
    for fold in range(cfg["training"]["n_folds"]):
        _, val = fold_split(train, cfg["training"]["n_folds"], int(cfg["seed"]), fold,
                            cfg.get("exclude_from_training", []))
        slides = [load_slide(cache, backbone, s.slide_id, label_threshold=t) for s in val]
        in_dim = slides[0].features.shape[1]
        for name in SCORERS:
            score = load_scorer(name, run_root, fold, in_dim, cfg["training"]["dropout"])
            pos = np.concatenate([score(s.features)[s.patch_labels == 1] for s in slides if s.label == 1])
            neg = np.concatenate([score(s.features) for s in slides if s.label == 0])
            auc = roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])
            rows.append({"scorer": name, "fold": fold, "patch_auc": round(float(auc), 4),
                         "n_tumour_patches": len(pos), "n_normal_patches": len(neg)})
            log.info("fold %d %s: patch AUC %.4f", fold, name, auc)
    write_csv(out_dir / "scorer_auc.csv", rows)
    for name in SCORERS:
        v = [r["patch_auc"] for r in rows if r["scorer"] == name]
        print(f"{name}: patch AUC {np.mean(v):.4f} ± {np.std(v, ddof=1):.4f} over folds")


def test_set(cfg, backbone):
    """{slide: (features, coords, patch size)} and {tumour slide: {itc flag: (eval mask, ITC labels, ds)}}."""
    data, cache = Path(cfg["paths"]["data_root"]), Path(cfg["paths"]["cache_root"])
    tumour = set()
    with open(data / "evaluation" / "reference.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["image"].startswith("test_") and r["type"] == "tumor":
                tumour.add(r["image"][:-4])
    slides, masks = {}, {}
    for sid in [s for s in slide_ids(cache, backbone) if s.startswith("test_")]:
        with h5py.File(cache / backbone / f"{sid}.h5") as f:
            slides[sid] = (torch.from_numpy(f["features"][:]), f["coords"][:],
                           int(f.attrs["patch_size_level0"]))
            if (int(f.attrs["n_lesions"]) > 0) != (sid in tumour):
                log.warning("%s: reference.csv type and annotations disagree", sid)
        if sid in tumour:
            mask, spacing, ds = read_mask(data / "masks" / f"{sid}_mask.tif", EVAL_LEVEL)
            masks[sid] = {itc: (*evaluation_mask(mask, spacing, include_itcs=itc), ds)
                          for itc in (False, True)}
    log.info("%d test slides, %d tumour", len(slides), len(masks))
    return slides, masks


def froc(cfg, backbone, run_root, out_root):
    slides, masks = test_set(cfg, backbone)
    run_dirs = sorted(run_root.glob("*/fold*/K*/retained.npz"))
    for name in SCORERS:
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        rows, curves = [], {}
        for fold in range(cfg["training"]["n_folds"]):
            score = load_scorer(name, run_root, fold, next(iter(slides.values()))[0].shape[1],
                                cfg["training"]["dropout"])
            probs = {sid: score(x) for sid, (x, _, _) in slides.items()}
            for rf in [r for r in run_dirs if r.parts[-3] == f"fold{fold}"]:
                method, K = rf.parts[-4], rf.parts[-2][1:]
                kept = np.load(rf)
                dets = {sid: detections_from_kept(kept[f"retained/{sid}"], c, ps, probs[sid])
                        for sid, (_, c, ps) in slides.items()}
                for itc in (False, True):
                    per_slide = []
                    for sid, d in dets.items():
                        if sid in masks:
                            em, itc_labels, ds = masks[sid][itc]
                            per_slide.append(slide_probabilities(to_level(d, ds), em, itc_labels))
                        else:
                            per_slide.append(slide_probabilities(d, None, set()))
                    fps, sens = froc_curve(per_slide)
                    fr, sr = list(reversed(fps)), list(reversed(sens))
                    at = {f"sens_fp{t:g}": sr[min(int(np.searchsorted(fr, t)), len(sr) - 1)]
                          for t in TARGET_FPS}  # the step lookup of froc_score
                    rows.append({"method": method, "K": K, "fold": fold, "itc": int(itc),
                                 "froc": round(froc_score(fps, sens), 4),
                                 **{k: round(v, 4) for k, v in at.items()},
                                 "detections_per_slide": round(sum(map(len, dets.values())) / len(dets), 2)})
                    curves[f"{method}/K{K}/fold{fold}/itc{int(itc)}"] = np.array([fps, sens])
                log.info("%s fold %d %s K%s: FROC %.3f (no ITC)", name, fold, method, K, rows[-2]["froc"])
        write_csv(out_dir / "runs.csv", rows)
        np.savez_compressed(out_dir / "curves.npz", **curves)
        groups = defaultdict(list)
        for r in rows:
            groups[(r["method"], r["K"], r["itc"])].append(r)
        cols = ["froc", *[f"sens_fp{t:g}" for t in TARGET_FPS], "detections_per_slide"]
        summary = [{"method": m, "K": K, "itc": itc, "n_folds": len(g),
                    **{f"{c}_{s}": round(float(f(np.array([r[c] for r in g]))), 4)
                       for c in cols for s, f in (("mean", np.mean), ("sd", _sd))}}
                   for (m, K, itc), g in sorted(groups.items())]
        write_csv(out_dir / "summary.csv", summary)
    write_run_json(out_root, cfg, backbone=backbone, scorers=list(SCORERS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="uni")
    ap.add_argument("--runs", default="runs", help="run folder under OUTPUT_ROOT")
    ap.add_argument("--scorer-auc", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    out = Path(cfg["paths"]["output_root"])
    run_root = out / args.runs / args.backbone
    out_root = out / "froc" / args.backbone
    out_root.mkdir(parents=True, exist_ok=True)
    if args.scorer_auc:
        scorer_auc(cfg, args.backbone, run_root, out_root)
    else:
        froc(cfg, args.backbone, run_root, out_root)


if __name__ == "__main__":
    main()
