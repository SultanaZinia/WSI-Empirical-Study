"""Tumour slot share (TSS) of the merging arm (TC-SSA), per lesion (docs/metrics.md Q1).

    python scripts/tss_analysis.py --config configs/server.yaml --backbone uni

Reads each TC-SSA run's stored W (runs/<backbone>/tc_ssa/fold<f>/K<K>/retained.npz: W/<sid> = slot and row
of each nonzero, W_val/<sid> = its weight) and the patch overlaps (CACHE_ROOT/patches). Writes
OUTPUT_ROOT/analysis/<backbone>/:

- tss.csv: per K x fold x lesion: TSS at the primary threshold, whether the lesion is hit under the argmax
  retention rule, size class, exhaustive flag. Unhittable lesions (no patch >= threshold) are left out.
- tss_summary.csv: test slides, exhaustive annotations, per size class x K, mean ± SD over folds of
  median TSS, the share of lesions owning a slot (TSS >= 0.5) and the share kept but diluted (hit by the
  argmax rule, TSS < 0.5).
"""

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from lesionbench.config import load_config, write_run_json
from lesionbench.eval.aggregate import read_csv, run_dirs, write_csv
from lesionbench.metrics.retention import tss
from lesionbench.preprocess.annotations import size_class

OWN = 0.5  # a lesion "owns" a slot at TSS >= 0.5 (the rule of the 30-10 report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="uni")
    ap.add_argument("--method", default="tc_ssa")
    args = ap.parse_args()
    cfg = load_config(args.config)
    t, strata = cfg["primary_overlap_threshold"], cfg["strata_mm"]
    non_exh = set(cfg.get("non_exhaustive_slides", []))
    out = Path(cfg["paths"]["output_root"]) / "analysis" / args.backbone
    out.mkdir(parents=True, exist_ok=True)
    patches = Path(cfg["paths"]["cache_root"]) / "patches"

    rows = []
    runs = [r for r in run_dirs(Path(cfg["paths"]["output_root"]) / "runs" / args.backbone)
            if r[0] == args.method]
    overlaps = {}
    for _, K, fold, d in runs:
        z = np.load(d / "retained.npz")
        for p in read_csv(d / "predictions.csv"):
            sid = p["slide_id"]
            if sid not in overlaps:
                with h5py.File(patches / f"{sid}.h5") as f:
                    o = f["overlap"]
                    overlaps[sid] = (o["rows"][:], o["lesion_ids"][:], o["fracs"][:],
                                     o["lesion_diameter_um"][:])
            ov_rows, ov_les, ov_f, diam = overlaps[sid]
            if len(diam) == 0:
                continue
            slot, row = z[f"W/{sid}"]
            share = tss(slot, row, z[f"W_val/{sid}"], ov_rows, ov_les, ov_f, len(diam), t)
            kept = np.isin(ov_rows, z[f"retained/{sid}"]) & (ov_f >= t)
            for lid, dm in enumerate(diam):
                if np.isnan(share[lid]):
                    continue
                rows.append({"method": args.method, "K": K, "fold": fold, "split": p["split"],
                             "slide_id": sid, "lesion_id": lid, "diameter_um": float(dm),
                             "size_class": size_class(dm / 1000, strata["itc_max"], strata["micro_max"]),
                             "exhaustive": int(sid not in non_exh), "tss": float(share[lid]),
                             "hit_argmax": int((kept & (ov_les == lid)).any())})
    write_csv(out / "tss.csv", rows)

    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["split"] == "test" and r["exhaustive"]:
            acc[(r["size_class"], r["K"])][r["fold"]].append(r)
    summary = []
    for (size, K), folds in sorted(acc.items(), key=lambda x: (x[0][0], int(x[0][1]))):
        stats = {"median_tss": [], "own_slot": [], "kept_diluted": []}
        for rs in folds.values():
            v = np.array([r["tss"] for r in rs])
            hit = np.array([r["hit_argmax"] for r in rs], dtype=bool)
            stats["median_tss"].append(np.median(v))
            stats["own_slot"].append(np.mean(v >= OWN))
            stats["kept_diluted"].append(np.mean(hit & (v < OWN)))
        row = {"size_class": size, "K": K, "n_folds": len(folds),
               "n_lesions_per_fold": round(np.mean([len(rs) for rs in folds.values()]), 1)}
        for k, v in stats.items():
            row[f"{k}_mean"] = round(float(np.mean(v)), 4)
            row[f"{k}_sd"] = round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else float("nan")
        summary.append(row)
    write_csv(out / "tss_summary.csv", summary)
    write_run_json(out, cfg, "run_tss.json", backbone=args.backbone, method=args.method, own_threshold=OWN)
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
