"""MIDL: does mean pooling dilute PMIL's micro-metastasis signal at larger budgets? (causal check, inference only)

    python midl/scripts/pmil_dilution.py --config configs/server.yaml --backbone uni \
        --out $SCRATCH/lesionbench/outputs_midl/pmil_dilution

PMIL codes its s kept patches against prototypes and AVERAGES the codes (selectors/pmil.py, forward). Each saved PMIL
model (runs/<backbone>/pmil/fold<f>/K<K>/model.pt, trained with s = K) is evaluated, without retraining, with
s = K (as trained; must reproduce the stored test predictions) and with s = 32 (same instance ranking, mean over its
top 32 patches only). Each (model, s) gets its own threshold: 95% specificity on that fold's validation normal
slides (out of fold), as in the main analysis. Reports per fold and mean: test AUC, specificity, sensitivity on
micro and macro slides, mean P(tumour) on micro slides. If micro sensitivity at s = 32 recovers towards the K = 32
model's, dilution by the mean is the cause.
Writes per_fold.csv and summary.md. Reads only; writes only to --out.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from lesionbench.config import ROOT, load_config
from lesionbench.eval.aggregate import read_csv
from lesionbench.eval.data import load_slide, slide_ids
from lesionbench.eval.external import spec_threshold
from lesionbench.eval.methods import fit
from lesionbench.splits import make_folds

SPEC = 0.95


def fold_val_ids(cache, backbone, cfg, fold):
    ex = set(cfg.get("exclude_from_training", []))
    ids = [i for i in slide_ids(cache, backbone) if not i.startswith("test_") and i not in ex]
    return make_folds(ids, [int(i.startswith("tumor_")) for i in ids], cfg["training"]["n_folds"],
                      int(cfg["seed"]))[fold][1]


@torch.no_grad()
def probs(model, slides, s, device):
    model.s = s
    return np.array([torch.softmax(model(x.features.to(device))[1], 0)[1].item() for x in slides])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="uni")
    ap.add_argument("--budgets", type=int, nargs="*", default=[32, 64, 128, 256])
    ap.add_argument("--small-s", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache, t = cfg["paths"]["cache_root"], cfg["primary_overlap_threshold"]
    runs = Path(cfg["paths"]["output_root"]) / "runs" / args.backbone / "pmil"
    test = [load_slide(cache, args.backbone, i, label_threshold=t)
            for i in slide_ids(cache, args.backbone) if i.startswith("test_")]
    y = np.array([s.label for s in test])
    strat = np.array([s.stratum for s in test])
    rows, worst = [], 0.0
    for fold in range(cfg["training"]["n_folds"]):
        val = [load_slide(cache, args.backbone, i, label_threshold=t) for i in fold_val_ids(cache, args.backbone, cfg, fold)]
        val_norm = [s for s in val if s.label == 0]
        tcfg, seed = cfg["training"], int(cfg["seed"]) + fold
        for K in args.budgets:
            d = runs / f"fold{fold}" / f"K{K}"
            stored = {r["slide_id"]: float(r["prob"]) for r in read_csv(d / "predictions.csv") if r["split"] == "test"}
            state = torch.load(d / "model.pt", map_location=device)
            model = fit("pmil", K, test[:1], [], tcfg, seed, device, {}, state=state).model
            for s in sorted({K, args.small_s}):
                if s > K:
                    continue
                p = probs(model, test, s, device)
                if s == K:
                    worst = max(worst, float(np.max(np.abs(p - np.array([stored[x.slide_id] for x in test])))))
                thr = spec_threshold(probs(model, val_norm, s, device), SPEC)
                call = p > thr
                rows.append({"fold": fold, "K_model": K, "s": s, "as_trained": int(s == K), "threshold": thr,
                             "auc": float(roc_auc_score(y, p)), "specificity": float(np.mean(~call[y == 0])),
                             "sens_micro": float(np.mean(call[strat == "micro"])),
                             "sens_macro": float(np.mean(call[strat == "macro"])),
                             "mean_p_micro": float(np.mean(p[strat == "micro"]))})
            print(f"fold {fold} K {K}: done (max |as-trained - stored| so far {worst:.2e})", flush=True)
    if worst > 1e-3:
        raise SystemExit(f"as-trained predictions do not reproduce the stored runs (max diff {worst:.3g}); not written")
    with open(out / "per_fold.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = ["auc", "specificity", "sens_micro", "sens_macro", "mean_p_micro"]
    lines = ["# PMIL dilution check (CAMELYON16 test; mean ± SD over 5 folds)", "",
             f"Same saved PMIL model per fold and K; s = patches averaged in the bag code. Thresholds: 95% "
             f"specificity on each fold's validation normals, per (model, s). Max |as-trained - stored| = {worst:.1e}.",
             "", "| K (model) | s | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for K in args.budgets:
        for s in sorted({K, args.small_s}):
            rs = [r for r in rows if r["K_model"] == K and r["s"] == s]
            if rs:
                lines.append(f"| {K} | {s}{' (as trained)' if s == K else ''} | " + " | ".join(
                    f"{np.mean([r[c] for r in rs]):.3f} ± {np.std([r[c] for r in rs], ddof=1):.3f}" for c in cols) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "run.json").write_text(json.dumps({"git_sha": sha, "backbone": args.backbone, "device": device,
                                              "small_s": args.small_s, "max_as_trained_diff": worst}, indent=1))
    print(out / "summary.md")


if __name__ == "__main__":
    main()
