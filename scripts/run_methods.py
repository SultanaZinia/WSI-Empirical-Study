"""Train and evaluate one method on one CV fold at every budget K (see lesionbench.eval.runner).

    python scripts/run_methods.py --config configs/server.yaml --backbone uni --method lts --fold 0

Reads CACHE_ROOT/<backbone>/*.h5 (features) and CACHE_ROOT/patches/*.h5 (overlaps, evaluation only); writes
OUTPUT_ROOT/runs/<backbone>/<method>/fold<f>/K<K>/. Finished K are skipped. ``--epoch-cap`` shortens every
training loop for smoke runs; never use it for reported results.
"""

import argparse
import logging
import subprocess
from pathlib import Path

import torch

from lesionbench.config import ROOT, load_config
from lesionbench.eval.data import load_slide, slide_ids
from lesionbench.eval.methods import METHODS
from lesionbench.eval.runner import run_method_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--budgets", type=int, nargs="*", help="default: config budgets")
    ap.add_argument("--epoch-cap", type=int, help="smoke runs only")
    ap.add_argument("--out-name", default="runs", help="output folder under OUTPUT_ROOT (e.g. smoke)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epoch_cap:
        cfg["training"]["epoch_cap"] = args.epoch_cap
    out_root = Path(cfg["paths"]["output_root"]) / args.out_name / args.backbone
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    log_file = out_root / "logs" / f"{args.method}_fold{args.fold}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)])
    log = logging.getLogger("run_methods").info

    git = ["git", "-C", str(ROOT), "rev-parse", "HEAD"]
    sha = subprocess.run(git, capture_output=True, text=True).stdout.strip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = cfg["paths"]["cache_root"]
    t = cfg["primary_overlap_threshold"]
    slides = [load_slide(cache, args.backbone, s, label_threshold=t) for s in slide_ids(cache, args.backbone)]
    log(f"{args.method} fold {args.fold}: {len(slides)} slides, {args.backbone}, {device}, git {sha}")
    provenance = {"git_sha": sha, "backbone": args.backbone, "device": device, "epoch_cap": args.epoch_cap,
                  "training": cfg["training"], "budgets": args.budgets or cfg["budgets"]}
    run_method_fold(args.method, args.fold, slides, cfg, out_root, device, provenance, args.budgets, log)


if __name__ == "__main__":
    main()
