"""Encoder audit (docs/metrics.md Q3) and detectability (hit-threshold rule): a linear probe per CV fold.

    python scripts/encoder_audit.py --config configs/server.yaml --backbone uni

Per fold, a probe is trained on the fold's training slides (lesionbench.metrics.probe) and scores the fold's
validation slides and the test slides. Writes OUTPUT_ROOT/encoder_audit/<backbone>/:
- detectability.csv: per fold and overlap band, the share of tumour patches flagged at 95% specificity on
  held-out normal-slide patches. Validation slides only (subset "all" and "exhaustive").
- lesions.csv: per lesion x threshold x fold: n_patches, probe AUC / ESI against the slide's lesion-free
  patches.
- slides_spec.csv: CLAUDE.md's per-slide ESI + linear-SVM patch AUC on micro-stratum slides (secondary).
- summary.md, run.json. Patch labels and overlaps are used here as evaluation ground truth only.
"""

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from lesionbench.config import ROOT, load_config
from lesionbench.eval.data import load_slide, slide_ids
from lesionbench.eval.runner import fold_split
from lesionbench.metrics.encoder import esi, svm_patch_auc
from lesionbench.metrics.probe import (
    Probe,
    detection_by_band,
    lesion_separation,
    max_overlap,
    probe_training_set,
)
from lesionbench.preprocess.annotations import size_class

VISIBLE_AUC = 0.90  # metrics.md: encoder-stage loss if probe AUC < 0.90 (0.85 / 0.95 from lesions.csv)


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="uni")
    ap.add_argument("--skip-spec", action="store_true", help="skip the per-slide ESI + SVM (slow)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg["paths"]["output_root"]) / "encoder_audit" / args.backbone
    out.mkdir(parents=True, exist_ok=True)
    git = ["git", "-C", str(ROOT), "rev-parse", "HEAD"]
    sha = subprocess.run(git, capture_output=True, text=True).stdout.strip()
    (out / "run.json").write_text(json.dumps({"git_sha": sha, "backbone": args.backbone, "config": cfg},
                                             indent=1, default=str))

    cache, seed, n_folds = cfg["paths"]["cache_root"], int(cfg["seed"]), cfg["training"]["n_folds"]
    thresholds, st = cfg["lsr_overlap_thresholds"], cfg["strata_mm"]
    non_exh, exclude = set(cfg.get("non_exhaustive_slides", [])), cfg.get("exclude_from_training", [])
    t_primary = cfg["primary_overlap_threshold"]  # patch labels (secondary ESI/SVM check only)
    slides = [load_slide(cache, args.backbone, sid, label_threshold=t_primary)
              for sid in slide_ids(cache, args.backbone)]
    test = [s for s in slides if s.split == "test"]
    print(f"{len(slides)} slides, git {sha}", flush=True)

    det_rows, les_rows = [], []
    for fold in range(n_folds):
        train, val = fold_split(slides, n_folds, seed, fold, exclude)
        X, y = probe_training_set(train, np.random.default_rng(seed + fold))
        probe = Probe(X, y, seed + fold)
        scores = {s.slide_id: probe.score(s.features.numpy()) for s in val + test}
        normal = np.concatenate([scores[s.slide_id] for s in val if s.label == 0])
        for subset in ("all", "exhaustive"):
            tum = [s for s in val if s.label == 1 and (subset == "all" or s.slide_id not in non_exh)]
            ov = np.concatenate([max_overlap(len(s.features), s.overlap[0], s.overlap[2]) for s in tum])
            tau, bands = detection_by_band(np.concatenate([scores[s.slide_id] for s in tum]), ov, normal)
            det_rows += [{"fold": fold, "subset": subset, "cutoff": tau, "n_probe_train": len(y), **b}
                         for b in bands]
        for s in val + test:
            if s.n_lesions == 0:
                continue
            for r in lesion_separation(scores[s.slide_id], *s.overlap, s.n_lesions, thresholds):
                d = float(s.lesion_diameter_um[r["lesion_id"]])
                les_rows.append({"slide_id": s.slide_id, "split": "test" if s.split == "test" else "val",
                                 "fold": fold, "exhaustive": int(s.slide_id not in non_exh),
                                 "diameter_um": d,
                                 "size_class": size_class(d / 1000, st["itc_max"], st["micro_max"]),
                                 **r, "encoder_visible": int(r["probe_auc"] >= VISIBLE_AUC)})
        print(f"fold {fold}: probe on {len(y)} patches; {len(val)} val + {len(test)} test slides", flush=True)
    write_csv(out / "detectability.csv", det_rows)
    write_csv(out / "lesions.csv", les_rows)

    if not args.skip_spec:
        spec = []
        for s in slides:
            if s.stratum == "micro" and s.slide_id not in exclude:
                X, y = s.features.numpy(), s.patch_labels
                spec.append({"slide_id": s.slide_id, "split": s.split,
                             "exhaustive": int(s.slide_id not in non_exh),
                             "n_tumour_patches": int(y.sum()), "esi": esi(X, y),
                             "svm_auc": svm_patch_auc(X, y, seed)})
        write_csv(out / "slides_spec.csv", spec)

    # summary: detection rate by band (mean over folds), and encoder-visible micro lesions on val slides
    lines = ["# Encoder audit", "", f"git {sha}, backbone {args.backbone}", "",
             "## Detection rate at 95% specificity (validation slides, mean over folds)", "",
             "| overlap band | all slides | exhaustive only |", "|---|---|---|"]
    by = defaultdict(list)
    for r in det_rows:
        by[(r["band_lo"], r["band_hi"], r["subset"])].append(r["detected"])
    for lo, hi in sorted({(k[0], k[1]) for k in by}):
        lines.append(f"| {lo:.2f}-{hi:.2f} | {np.nanmean(by[(lo, hi, 'all')]):.3f} | "
                     f"{np.nanmean(by[(lo, hi, 'exhaustive')]):.3f} |")
    lines += ["", "## Validation lesions (exhaustive slides): n with >= 1 patch / encoder-visible", "",
              "| size class | " + " | ".join(f"t = {t}" for t in thresholds) + " |",
              "|---|" + "---|" * len(thresholds)]
    for cls in ("itc", "micro", "macro"):
        cells = []
        for t in thresholds:
            rs = [r for r in les_rows if r["split"] == "val" and r["exhaustive"] and r["size_class"] == cls
                  and r["threshold"] == t and r["n_patches"] > 0]
            cells.append(f"{len(rs)} / {sum(r['encoder_visible'] for r in rs)}")
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(out / "summary.md")


if __name__ == "__main__":
    main()
