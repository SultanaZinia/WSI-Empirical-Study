"""Slide- and lesion-level results of the external validation (CAMELYON17), with CAMELYON16 test alongside.

    python scripts/external_analysis.py --config configs/c17.local.yaml --source-config configs/server.yaml \
        --backbone uni

Operating point: each fold model's threshold = 95% specificity on its own CAMELYON16 validation fold's normal
slides (source runs/<backbone>/<method>/fold<f>/K<K>/predictions.csv). Reads the external runs
(OUTPUT_ROOT/external/<backbone>/), DATA_ROOT/stages.csv (stage, centre) and the external patch overlaps.
Writes OUTPUT_ROOT/analysis/<backbone>/ (target config):

- slide_metrics.csv: per dataset (c16_test, c17) x method x K x fold x group (all, centre 0-4 for c17): AUC,
  specificity, sensitivity per stage. CAMELYON16 test stages come from its annotations (retention.csv).
- slide_summary.csv: mean and SD over folds.
- lesion_hits.csv: CAMELYON17 annotated slides, per method x K x fold x lesion (as aggregate_runs.py).
- table.md: K = 32 overview.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from lesionbench.config import load_config, write_run_json
from lesionbench.eval.aggregate import lesion_hits, mean_sd, pooled_lsr, read_csv, run_dirs, write_csv
from lesionbench.eval.external import STAGES, slide_metrics, spec_threshold

SPEC = 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-config", required=True)
    ap.add_argument("--backbone", default="uni")
    args = ap.parse_args()
    cfg, src = load_config(args.config), load_config(args.source_config)
    out = Path(cfg["paths"]["output_root"]) / "analysis" / args.backbone
    out.mkdir(parents=True, exist_ok=True)

    with open(Path(cfg["paths"]["data_root"]) / "stages.csv", newline="") as f:
        meta = {r["patient"][:-4]: r for r in csv.DictReader(f) if r["patient"].endswith(".tif")}
    stage17 = {s: m["stage"] for s, m in meta.items()}
    centre17 = {s: m["center"] for s, m in meta.items()}

    src_root = Path(src["paths"]["output_root"]) / "runs" / args.backbone
    src_runs = {(m, K, f): d for m, K, f, d in run_dirs(src_root)}
    ext_runs = run_dirs(Path(cfg["paths"]["output_root"]) / "external" / args.backbone)
    rows = []
    for (m, K, fold), d in sorted(src_runs.items()):
        preds = read_csv(d / "predictions.csv")
        normals = [float(r["prob"]) for r in preds if r["split"] == "val" and r["label"] == "0"]
        t = spec_threshold(normals, SPEC)
        test = [r for r in preds if r["split"] == "test"]
        stage16 = {r["slide_id"]: r["stratum"] for r in read_csv(d / "retention.csv") if r["split"] == "test"}
        rows += [{"dataset": "c16_test", "method": m, "K": K, "fold": fold, **x}
                 for x in slide_metrics(test, t, stage16)]
        ext = [e for e in ext_runs if e[:3] == (m, K, fold)]
        if ext:
            rows += [{"dataset": "c17", "method": m, "K": K, "fold": fold, **x}
                     for x in slide_metrics(read_csv(ext[0][3] / "predictions.csv"), t, stage17, centre17)]
    write_csv(out / "slide_metrics.csv", rows)

    cols = ["auc", "specificity", *[f"sens_{s}" for s in STAGES[1:]]]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["method"], r["K"], r["group"])].append(r)
    summary = []
    for (ds, m, K, g), rs in sorted(groups.items()):
        row = {"dataset": ds, "method": m, "K": K, "group": g, "n_folds": len(rs), "n": rs[0]["n"],
               **{f"n_{s}": rs[0][f"n_{s}"] for s in STAGES[1:]}}
        for c in cols:
            v = np.array([r[c] for r in rs], dtype=float)
            row[f"{c}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            row[f"{c}_sd"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan")
        summary.append(row)
    write_csv(out / "slide_summary.csv", summary)

    overlaps = {}
    for p in (Path(cfg["paths"]["cache_root"]) / "patches").glob("*.h5"):
        with h5py.File(p) as f:
            o = f["overlap"]
            overlaps[p.stem] = (o["rows"][:], o["lesion_ids"][:], o["fracs"][:], o["lesion_diameter_um"][:])
    hits = lesion_hits(ext_runs, overlaps, src["primary_overlap_threshold"], src["strict_overlap_threshold"],
                       [], src["strata_mm"])
    write_csv(out / "lesion_hits.csv", hits)

    lsr = {z: pooled_lsr(hits, split="external", size=z, exhaustive_only=False)
           for z in ("itc", "micro", "macro")}
    lines = ["# External validation, K = 32 (and full bag)", "",
             f"Operating point: {SPEC:.0%} specificity on each fold's CAMELYON16 validation normals. "
             "Mean ± SD over 5 fold models. C17 stages from stages.csv (all 500 training slides); LSR on "
             "the 50 annotated slides, lesion-pooled, primary threshold (ITC descriptive: not "
             "exhaustively outlined).", "",
             "| method | C16 AUC | C17 AUC | C17 spec | C17 sens ITC | C17 sens micro | C17 sens macro | "
             "C16 sens micro | C17 LSR micro | C17 LSR macro |", "|---|---|---|---|---|---|---|---|---|---|"]

    def cell(ds, m, K, c):
        key = (ds, m, K, "all")
        return mean_sd([r[c] for r in rows if (r["dataset"], r["method"], r["K"], r["group"]) == key])

    for m, K in sorted({(r["method"], r["K"]) for r in rows if r["K"] in ("32", "full")}):
        lines.append(f"| {m} | {cell('c16_test', m, K, 'auc')} | {cell('c17', m, K, 'auc')} | "
                     f"{cell('c17', m, K, 'specificity')} | {cell('c17', m, K, 'sens_itc')} | "
                     f"{cell('c17', m, K, 'sens_micro')} | {cell('c17', m, K, 'sens_macro')} | "
                     f"{cell('c16_test', m, K, 'sens_micro')} | {mean_sd(lsr['micro'].get((m, K), []))} | "
                     f"{mean_sd(lsr['macro'].get((m, K), []))} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    write_run_json(out, cfg, "run_external_analysis.json", backbone=args.backbone, spec=SPEC,
                   source=str(src_root))
    print(out / "table.md")


if __name__ == "__main__":
    main()
