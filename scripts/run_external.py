"""External validation: apply one method's saved CAMELYON16 fold models, without training, to another dataset.

    python scripts/run_external.py --config configs/c17.local.yaml --source-config configs/server.yaml \
        --backbone uni --method lts --fold 0

Source (--source-config): the finished runs OUTPUT_ROOT/runs/<backbone>/<method>/fold<f>/K<K>/model.pt and the
CAMELYON16 training slides (fold split; normal key banks are rebuilt from the fold's training normals).
Target (--config): every slide in CACHE_ROOT/<backbone>/. Slide label and stage come from DATA_ROOT/stages.csv
(CAMELYON17: negative / itc / micro / macro; label = stage != negative); lesion annotations exist for some
slides only. Writes OUTPUT_ROOT/external/<backbone>/<method>/fold<f>/K<K>/. Finished K are skipped.
"""

import argparse
import csv
import logging
import subprocess
from dataclasses import replace
from pathlib import Path

import torch

from lesionbench.config import ROOT, load_config
from lesionbench.eval.data import load_slide, slide_ids
from lesionbench.eval.methods import METHODS
from lesionbench.eval.runner import run_external


def stages(path) -> dict[str, str]:
    with open(path, newline="") as f:
        return {r["patient"][:-4]: r["stage"] for r in csv.DictReader(f) if r["patient"].endswith(".tif")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="target dataset config")
    ap.add_argument("--source-config", required=True, help="config of the dataset the models were trained on")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--budgets", type=int, nargs="*", help="default: config budgets")
    args = ap.parse_args()

    cfg, src = load_config(args.config), load_config(args.source_config)
    out_root = Path(cfg["paths"]["output_root"]) / "external" / args.backbone
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    log_file = out_root / "logs" / f"{args.method}_fold{args.fold}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)])
    log = logging.getLogger("run_external").info

    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = src["primary_overlap_threshold"]
    src_cache = src["paths"]["cache_root"]
    train = [load_slide(src_cache, args.backbone, s, label_threshold=t)
             for s in slide_ids(src_cache, args.backbone) if not s.startswith("test_")]
    stage = stages(Path(cfg["paths"]["data_root"]) / "stages.csv")
    cache = cfg["paths"]["cache_root"]
    external = [replace(s, split="external", label=int(stage[s.slide_id] != "negative"),
                        stratum=stage[s.slide_id])
                for s in (load_slide(cache, args.backbone, i, label_threshold=t)
                          for i in slide_ids(cache, args.backbone))]
    log(f"{args.method} fold {args.fold}: {len(train)} source training slides, {len(external)} external, "
        f"{args.backbone}, {device}, git {sha.strip()}")
    provenance = {"git_sha": sha.strip(), "backbone": args.backbone, "device": device,
                  "source_runs": str(Path(src["paths"]["output_root"]) / "runs" / args.backbone)}
    # Evaluation rules (thresholds, budgets, training recipe for the seeds) come from the source config.
    run_external(args.method, args.fold, train, external, src, Path(src["paths"]["output_root"]) / "runs" /
                 args.backbone, out_root, device, provenance, args.budgets, log)


if __name__ == "__main__":
    main()
