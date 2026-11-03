"""Lesion-level controls: Random (exact), Random-Hilbert, Oracle-Cov, Oracle-LSR, Oracle-LSR-micro, at every K
and threshold.

    python scripts/lesion_controls.py --config configs/server.yaml

Reads CACHE_ROOT/patches/*.h5 (needs /overlap with lesion diameters: run scripts/add_overlap.py first).
Writes OUTPUT_ROOT/lesion_controls/lesions.csv, one row per lesion x threshold x K x arm with ``hit`` = 0/1
(Random: the exact hit probability; Random-Hilbert: the mean over HILBERT_REPS seeded draws), and summary.md
with lesion-pooled LSR by lesion size class.
Oracle arms use the per-slide seed crc32(slide_id) XOR config seed, as in run_controls.py.
"""

import argparse
import csv
import json
import subprocess
import zlib
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from lesionbench.config import ROOT, load_config
from lesionbench.metrics.probe import max_overlap
from lesionbench.preprocess.annotations import size_class
from lesionbench.selectors.controls import (
    OracleLSRSelector,
    OracleSelector,
    RandomHilbertSelector,
    random_hit_prob,
)
from lesionbench.selectors.slicechat import grid_from_coords

ARMS = ("random", "random_hilbert", "oracle_cov", "oracle_lsr", "oracle_lsr_micro")
HILBERT_REPS = 20  # random window draws per slide and K (seeds: slide seed + rep)
PRIORITY = {"micro": 0, "itc": 1, "macro": 2}  # Oracle-LSR-micro: micro lesions first
FIELDS = [
    "slide_id",
    "split",
    "stratum",
    "exhaustive",
    "lesion_id",
    "diameter_um",
    "size_class",
    "threshold",
    "n_eligible",
    "K",
    "arm",
    "hit",
]


def eligible(fracs, t):
    return fracs >= t if t > 0 else fracs > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["output_root"]) / "lesion_controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    (out_dir / "run.json").write_text(json.dumps({"git_sha": sha, "config": cfg}, indent=1, default=str))
    thresholds, budgets = cfg["lsr_overlap_thresholds"], cfg["budgets"]
    st = cfg["strata_mm"]
    non_exh = set(cfg.get("non_exhaustive_slides", []))

    rows_out = []
    for p in sorted((Path(cfg["paths"]["cache_root"]) / "patches").glob("*.h5")):
        with h5py.File(p) as f:
            a = dict(f.attrs)
            if int(a["n_lesions"]) == 0:
                continue
            grid = grid_from_coords(f["coords"][:], int(a["patch_size_level0"]))
            ov_rows, ov_les, ov_frac = (f["overlap"][k][:] for k in ("rows", "lesion_ids", "fracs"))
            # Oracle-Cov patch labels at the primary threshold, as in the runner (not the cached 0.10 labels)
            t_primary = cfg["primary_overlap_threshold"]
            labels = (max_overlap(len(grid), ov_rows, ov_frac) >= t_primary).astype(np.uint8)
            diam = f["overlap/lesion_diameter_um"][:]
        sid, n, m = str(a["slide_id"]), len(labels), int(a["n_lesions"])
        seed = zlib.crc32(sid.encode()) ^ int(cfg["seed"])
        cls = [size_class(d / 1000.0, st["itc_max"], st["micro_max"]) for d in diam]
        feats = torch.empty(n, 0)
        hilbert_kept = {
            K: [
                RandomHilbertSelector().select(feats, K, grid=grid, seed=seed + r).retained_idx.numpy()
                for r in range(HILBERT_REPS)
            ]
            for K in budgets
        }
        base = {
            "slide_id": sid,
            "split": "test" if sid.startswith("test_") else "train",
            "stratum": a["stratum"],
            "exhaustive": int(sid not in non_exh),
        }
        for t in thresholds:
            ok = eligible(ov_frac, t)
            n_elig = np.bincount(ov_les[ok], minlength=m)
            ctx = {
                "overlap_rows": ov_rows,
                "overlap_lesions": ov_les,
                "overlap_fracs": ov_frac,
                "threshold": t,
                "seed": seed,
            }
            for K in budgets:
                picks = {
                    "oracle_cov": OracleSelector().select(
                        feats, K, patch_labels=torch.from_numpy(labels), seed=seed
                    ),
                    "oracle_lsr": OracleLSRSelector().select(feats, K, **ctx),
                    "oracle_lsr_micro": OracleLSRSelector().select(
                        feats, K, lesion_priority=[PRIORITY[c] for c in cls], **ctx
                    ),
                }
                hits = {}
                for arm, res in picks.items():
                    kept = np.isin(ov_rows, res.retained_idx.numpy()) & ok
                    hits[arm] = np.isin(np.arange(m), ov_les[kept]).astype(float)
                hits["random"] = np.array([random_hit_prob(n, K, int(e)) for e in n_elig])
                hits["random_hilbert"] = np.mean(
                    [np.isin(np.arange(m), ov_les[np.isin(ov_rows, kept) & ok]) for kept in hilbert_kept[K]],
                    axis=0,
                )
                for lid in range(m):
                    for arm in ARMS:
                        rows_out.append(
                            {
                                **base,
                                "lesion_id": lid,
                                "diameter_um": round(float(diam[lid]), 1),
                                "size_class": cls[lid],
                                "threshold": t,
                                "n_eligible": int(n_elig[lid]),
                                "K": K,
                                "arm": arm,
                                "hit": hits[arm][lid],
                            }
                        )

    with open(out_dir / "lesions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_out)
    (out_dir / "summary.md").write_text(summary(rows_out, thresholds, sha))
    print(out_dir / "summary.md")


def summary(rows, thresholds, sha) -> str:
    """Lesion-pooled LSR by split x size class x arm x K (one column per threshold), with lesion counts."""
    hit, cnt, unhit = defaultdict(float), defaultdict(int), defaultdict(int)
    for r in rows:
        key = (r["split"], r["size_class"], r["arm"], r["K"], r["threshold"])
        hit[key] += r["hit"]
        cnt[key] += 1
        unhit[key] += r["n_eligible"] == 0
    out = [
        f"# Lesion-level controls (git {sha[:7]})",
        "",
        "Lesion-pooled LSR@K by lesion size class (AJCC). Columns: hit threshold (patch overlap with the",
        "lesion; 0 = any overlap). Random is the exact expectation. `unhittable` = lesions with no eligible",
        "patch at that threshold (no arm can hit them; they stay in the denominator). All slides, including",
        "non-exhaustive ones.",
        "",
    ]
    keys = sorted(
        {k[:4] for k in cnt},
        key=lambda k: (k[0] != "train", ["itc", "micro", "macro"].index(k[1]), ARMS.index(k[2]), k[3]),
    )
    head = " | ".join(f"{'any' if t == 0 else f'≥{t:.0%}'}" for t in thresholds)
    un_head = ", ".join(f"{t:g}" for t in thresholds)
    out += [
        f"| split | class | arm | K | lesions | {head} | unhittable ({un_head}) |",
        "|" + "---|" * (6 + len(thresholds)),
    ]
    for k in keys:
        vals = " | ".join(f"{hit[(*k, t)] / cnt[(*k, t)]:.3f}" for t in thresholds)
        un = ", ".join(str(unhit[(*k, t)]) for t in thresholds)
        out.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {cnt[(*k, thresholds[0])]} | {vals} | {un} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
