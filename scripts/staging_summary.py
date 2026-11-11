"""Summary of the pN staging runs (scripts/pn_staging.py): one row per method x K, mean and SD over folds.

    python scripts/staging_summary.py --config configs/c17.local.yaml --backbone uni

Columns: kappa (quadratic weighted, patients); pn_acc (exact pN); pn1mi_acc (pN1mi patients staged pN1mi);
slide_acc (4-class); micro_missed / macro_missed (micro or macro slides called negative or itc).
Writes OUTPUT_ROOT/staging/<backbone>/summary.csv.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from lesionbench.config import load_config


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_metrics(d: Path) -> dict:
    pts, sls = read(d / "patients.csv"), read(d / "slides.csv")
    pn1mi = [p for p in pts if p["true"] == "pN1mi"]

    def missed(stage):
        s = [r for r in sls if r["true"] == stage]
        return float(np.mean([r["pred"] in ("negative", "itc") for r in s])) if s else float("nan")

    return {"kappa": json.loads((d / "done.json").read_text())["kappa"],
            "pn_acc": float(np.mean([p["true"] == p["pred"] for p in pts])),
            "pn1mi_acc": float(np.mean([p["pred"] == "pN1mi" for p in pn1mi])) if pn1mi else float("nan"),
            "slide_acc": float(np.mean([r["true"] == r["pred"] for r in sls])),
            "micro_missed": missed("micro"), "macro_missed": missed("macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", required=True)
    args = ap.parse_args()
    root = Path(load_config(args.config)["paths"]["output_root"]) / "staging" / args.backbone
    acc = defaultdict(list)
    for done in sorted(root.glob("*/fold*/K*/done.json")):
        d = done.parent
        acc[(d.parent.parent.name, d.name[1:])].append(run_metrics(d))
    rows = []
    for (method, K), ms in sorted(acc.items()):
        row = {"method": method, "K": K, "n_source_folds": len(ms)}
        for k in ms[0]:
            v = np.array([m[k] for m in ms], dtype=float)
            row[f"{k}_mean"] = float(np.nanmean(v))
            row[f"{k}_sd"] = float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan")
        rows.append(row)
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} (method, K) -> {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
