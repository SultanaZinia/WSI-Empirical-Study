"""pN staging on CAMELYON17 from one method's kept patches (selection fixed from the CAMELYON16 fold models).

    python scripts/pn_staging.py --config configs/c17.local.yaml --source-config configs/server.yaml \
        --backbone uni --method lts

Kept rows: OUTPUT_ROOT/external/<backbone>/<method>/fold<f>/K<K>/retained.npz (written by run_external.py);
``abmil_full`` = every patch (no compression; source fold 0 only, as its bag does not depend on the fold).
For each source fold f and K: a 4-class gated-ABMIL head (negative / itc / micro / macro; shared CLAM recipe
from the source config) on the kept features, 5-fold cross-validation over CAMELYON17 patients stratified by
pN (staging.patient_folds), then pN per patient by the challenge rules. Labels: DATA_ROOT/stages.csv.
Writes OUTPUT_ROOT/<out-name>/<backbone>/<method>/fold<f>/K<K>/{slides.csv, patients.csv, done.json}
(out-name default "staging").
Finished (f, K) are skipped.
"""

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from lesionbench.config import ROOT, load_config
from lesionbench.eval.staging import cross_validate, patient_folds, patient_rows, quadratic_kappa


def read_stages(path):
    slide_stage, patient_stage = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["patient"].endswith(".tif"):
                slide_stage[r["patient"][:-4]] = r["stage"]
            elif r["patient"].endswith(".zip"):
                patient_stage[r["patient"][:-4]] = r["stage"]
    return slide_stage, patient_stage


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="CAMELYON17 config")
    ap.add_argument("--source-config", required=True,
                    help="CAMELYON16 config (training recipe, budgets, seed)")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--budgets", type=int, nargs="*", help="default: source config budgets")
    ap.add_argument("--folds", type=int, nargs="*", help="source folds (default: all)")
    ap.add_argument("--out-name", default="staging",
                    help="output folder under OUTPUT_ROOT (default: staging)")
    args = ap.parse_args()

    cfg, src = load_config(args.config), load_config(args.source_config)
    tcfg, seed = src["training"], int(src["seed"])
    out_root = Path(cfg["paths"]["output_root"])
    ext = out_root / "external" / args.backbone / args.method
    out = out_root / args.out_name / args.backbone / args.method
    full = args.method == "abmil_full"
    runs = [(f, "full") for f in [0]] if full else \
        [(f, K) for f in (args.folds or range(tcfg["n_folds"])) for K in (args.budgets or src["budgets"])]
    runs = [(f, K) for f, K in runs if not (out / f"fold{f}" / f"K{K}" / "done.json").exists()]
    if not runs:
        print("all done")
        return

    slide_stage, patient_stage = read_stages(Path(cfg["paths"]["data_root"]) / "stages.csv")
    patient_of = {s: s.split("_node_")[0] for s in slide_stage}
    folds = patient_folds(patient_stage, 5, seed)
    retained = {(f, K): np.load(ext / f"fold{f}" / f"K{K}" / "retained.npz") for f, K in runs if not full}

    # read each slide once; keep only the rows some run needs
    cache = Path(cfg["paths"]["cache_root"]) / args.backbone
    bags = {r: {} for r in runs}
    for s in sorted(slide_stage):
        with h5py.File(cache / f"{s}.h5") as h:
            x = torch.from_numpy(h["features"][:])
        for r in runs:
            bags[r][s] = x if full else x[torch.from_numpy(retained[r][f"retained/{s}"])]

    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for f, K in runs:
        t0 = time.time()
        d = out / f"fold{f}" / f"K{K}"
        d.mkdir(parents=True, exist_ok=True)
        rows = cross_validate(bags[(f, K)], slide_stage, patient_of, folds, tcfg, seed, device=device,
                              dropout=tcfg["dropout"])
        prs = patient_rows(rows, patient_stage)
        write_csv(d / "slides.csv", rows)
        write_csv(d / "patients.csv", prs)
        kappa = quadratic_kappa([r["true"] for r in prs], [r["pred"] for r in prs])
        (d / "done.json").write_text(json.dumps({
            "git_sha": sha.strip(), "backbone": args.backbone, "method": args.method, "K": K,
            "source_fold": f,
            "seed": seed, "device": device, "n_slides": len(rows), "n_patients": len(prs), "kappa": kappa,
            "best_epochs": sorted({r["best_epoch"] for r in rows}),
            "total_seconds": round(time.time() - t0, 1)}, indent=1))
        print(f"{args.method} fold {f} K {K}: kappa {kappa:.3f}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
