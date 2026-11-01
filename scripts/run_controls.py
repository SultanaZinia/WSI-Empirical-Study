"""Random and Oracle controls at every budget K, from the patch files alone (no features needed).

    python scripts/run_controls.py --config configs/server.yaml

Reads CACHE_ROOT/patches/*.h5 (needs /overlap: run scripts/add_overlap.py first on older files); writes
OUTPUT_ROOT/controls/metrics.csv (one row per slide x method x K, plus LSR at each lsr_overlap_thresholds)
and run.json (config, git SHA, seed). Per-slide seed = crc32(slide_id) XOR config seed.
"""

import argparse
import csv
import json
import subprocess
import zlib
from pathlib import Path

import h5py
import torch

from lesionbench.config import ROOT, load_config
from lesionbench.metrics.retention import cov_at_k, lsr_at_k, lsr_curve, tep_at_k
from lesionbench.selectors.controls import OracleSelector, RandomSelector

FIELDS = ["slide_id", "split", "stratum", "n_patches", "n_pos", "n_lesions", "method", "K", "seed",
          "cov", "lsr", "tep"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["output_root"]) / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    git = ["git", "-C", str(ROOT), "rev-parse", "HEAD"]
    sha = subprocess.run(git, capture_output=True, text=True).stdout.strip()
    (out_dir / "run.json").write_text(json.dumps({"git_sha": sha, "config": cfg}, indent=1, default=str))

    methods = {"random": RandomSelector(), "oracle": OracleSelector()}
    thresholds = cfg["lsr_overlap_thresholds"]
    lsr_cols = {t: f"lsr_t{round(100 * t):02d}" for t in thresholds}
    with open(out_dir / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + list(lsr_cols.values()))
        w.writeheader()
        for p in sorted((Path(cfg["paths"]["cache_root"]) / "patches").glob("*.h5")):
            with h5py.File(p) as f:
                labels, ids, a = f["patch_labels"][:], f["lesion_ids"][:], dict(f.attrs)
                ov = [f["overlap"][k][:] for k in ("rows", "lesion_ids", "fracs")]
            sid = str(a["slide_id"])
            seed = zlib.crc32(sid.encode()) ^ int(cfg["seed"])
            feats, y = torch.empty(len(labels), 0), torch.from_numpy(labels)
            for name, sel in methods.items():
                for K in cfg["budgets"]:
                    idx = sel.select(feats, K, patch_labels=y, seed=seed).retained_idx.numpy()
                    curve = lsr_curve(idx, *ov, int(a["n_lesions"]), thresholds)
                    w.writerow({
                        "slide_id": sid, "split": "test" if sid.startswith("test_") else "train",
                        "stratum": a["stratum"], "n_patches": len(labels), "n_pos": int(labels.sum()),
                        "n_lesions": int(a["n_lesions"]), "method": name, "K": K, "seed": seed,
                        "cov": cov_at_k(idx, labels), "lsr": lsr_at_k(idx, ids, int(a["n_lesions"])),
                        "tep": tep_at_k(idx, labels),
                        **{lsr_cols[t]: v for t, v in curve.items()},
                    })
    print(out_dir / "metrics.csv")


if __name__ == "__main__":
    main()
