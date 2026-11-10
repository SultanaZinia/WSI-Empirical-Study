"""Aggregate finished runs into tables and a draft Table 1.

    python scripts/aggregate_runs.py --config configs/server.yaml --backbone uni

Reads OUTPUT_ROOT/runs/<backbone>/ (only K folders with done.json) and CACHE_ROOT/patches/*.h5 (lesion
overlaps and diameters, evaluation only). Writes OUTPUT_ROOT/analysis/<backbone>/: slide_auc.csv,
lesion_hits.csv, retention_means.csv, table1.md. Validation (out-of-fold) lesion results use exhaustively
annotated slides only (primary analysis); test slides are all exhaustive.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from lesionbench.config import load_config, write_run_json
from lesionbench.eval.aggregate import (
    lesion_hits,
    mean_sd,
    pooled_lsr,
    retention_means,
    run_dirs,
    slide_auc,
    write_csv,
)

ORDER = ["random", "abmil", "siimil", "nrk_abmil", "pmil", "tc_ssa", "lts", "slicechat", "dtfd",
         "oracle", "abmil_full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="uni")
    args = ap.parse_args()

    cfg = load_config(args.config)
    runs = run_dirs(Path(cfg["paths"]["output_root"]) / "runs" / args.backbone)
    out = Path(cfg["paths"]["output_root"]) / "analysis" / args.backbone
    out.mkdir(parents=True, exist_ok=True)
    write_run_json(out, cfg, "run_aggregate.json", backbone=args.backbone)
    overlaps = {}
    for p in (Path(cfg["paths"]["cache_root"]) / "patches").glob("*.h5"):
        with h5py.File(p) as f:
            o = f["overlap"]
            overlaps[p.stem] = (o["rows"][:], o["lesion_ids"][:], o["fracs"][:], o["lesion_diameter_um"][:])

    aucs = slide_auc(runs)
    hits = lesion_hits(runs, overlaps, cfg["primary_overlap_threshold"], cfg["strict_overlap_threshold"],
                       cfg.get("non_exhaustive_slides", []), cfg["strata_mm"])
    write_csv(out / "slide_auc.csv", aucs)
    write_csv(out / "lesion_hits.csv", hits)
    write_csv(out / "retention_means.csv", retention_means(runs))

    def auc_cell(method, K, split):
        return mean_sd([r["auc"] for r in aucs
                        if r["method"] == method and r["K"] == K and r["split"] == split])

    lsr = {(split, size): pooled_lsr(hits, split=split, size=size, exhaustive_only=True)
           for split in ("test", "val") for size in ("itc", "micro", "macro")}
    def order(mk):
        return ORDER.index(mk[0]) if mk[0] in ORDER else 99, int(mk[1]) if mk[1].isdigit() else 1e9

    keys = sorted({(m, K) for m, K, _, _ in runs}, key=order)
    folds = {(m, K): sum(1 for r in runs if r[:2] == (m, K)) for m, K in keys}
    lines = ["# Table 1 (draft)", "",
             f"Primary overlap threshold {cfg['primary_overlap_threshold']}; mean ± SD over folds. LSR = "
             "lesion-pooled survival (kept patch overlapping the lesion >= threshold). Val = out-of-fold "
             "training slides, exhaustive annotations only.", "",
             "| method | K | folds | AUC test | AUC val (OOF) | LSR ITC test | LSR micro test | "
             "LSR macro test | LSR micro val |", "|---|---|---|---|---|---|---|---|---|"]
    for m, K in keys:
        oof = [r["auc"] for r in aucs if r["method"] == m and r["K"] == K and r["split"] == "val_oof"]
        cells = [lsr[(s, z)].get((m, K), []) for s, z in
                 (("test", "itc"), ("test", "micro"), ("test", "macro"), ("val", "micro"))]
        lines.append(f"| {m} | {K} | {folds[(m, K)]} | {auc_cell(m, K, 'test')} | "
                     f"{np.mean(oof):.3f} | " + " | ".join(mean_sd(c) for c in cells) + " |")
    (out / "table1.md").write_text("\n".join(lines) + "\n")
    print(out / "table1.md")


if __name__ == "__main__":
    main()
