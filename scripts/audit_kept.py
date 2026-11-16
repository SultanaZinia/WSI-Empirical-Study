"""Count the kept patches per slide in every saved run: does each run keep exactly K unique patches?

    python scripts/audit_kept.py --config configs/server.yaml --out DIR

Reads OUTPUT_ROOT/runs/<backbone>/<method>/fold<f>/K<K>/retained.npz (the full bag, Kfull, is skipped).
Writes DIR/kept_audit.csv, one row per backbone x method x fold x K x split (test / other), and prints every
row where a slide has fewer or more than K unique patches. Merging methods (TC-SSA) are expected to fall below
K: their spatial output is the unique largest contributors of the non-empty slots. Reads only.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from lesionbench.config import load_config


def audit(path: Path, K: int) -> list[dict]:
    z = np.load(path)
    per = {"test": [], "other": []}
    for key in z.files:
        if key.startswith("retained/"):
            sid, idx = key.split("/", 1)[1], z[key]
            per["test" if sid.startswith("test_") else "other"].append((len(idx), len(np.unique(idx))))
    rows = []
    for split, v in per.items():
        if v:
            n, u = np.array(v).T
            rows.append({"split": split, "n_slides": len(v), "min_unique": int(u.min()),
                         "max_unique": int(u.max()), "mean_unique": round(float(u.mean()), 2),
                         "slides_unique_ne_K": int((u != K).sum()),
                         "slides_with_duplicates": int((u < n).sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(load_config(args.config)["paths"]["output_root"]) / "runs"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(root.glob("*/*/fold*/K*/retained.npz")):
        backbone, method, fold, K = f.parts[-5], f.parts[-4], f.parts[-3][4:], f.parts[-2][1:]
        if K == "full":
            continue
        rows += [{"backbone": backbone, "method": method, "fold": int(fold), "K": int(K), **r}
                 for r in audit(f, int(K))]
    with open(out / "kept_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad = [r for r in rows if r["slides_unique_ne_K"] or r["slides_with_duplicates"]]
    print(f"{len(rows)} rows from {root}; {len(bad)} not exactly K unique:")
    for r in bad:
        print(r)


if __name__ == "__main__":
    main()
