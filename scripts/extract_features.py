"""Embed every cached patch with a frozen encoder and write CACHE_ROOT/<backbone>/<slide>.h5.

    python scripts/extract_features.py --config configs/server.yaml --backbone uni [--task-id I --n-tasks N]

Reads the row order from CACHE_ROOT/patches/<slide>.h5 and copies coords/labels/lesion ids alongside the
features. Task I of N handles every N-th slide; slides whose output exists are skipped. Logs per slide to
OUTPUT_ROOT/logs/features/<backbone>/. ``--slides`` restricts the run, e.g. for a smoke test with
``--backbone dummy``.
"""

import argparse
import logging
import os
import subprocess
import time
from pathlib import Path

from lesionbench.config import ROOT, load_config

log = logging.getLogger("extract_features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", required=True, choices=["uni", "conch", "dummy"])
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--slides", nargs="*", help="only these slide ids")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.environ.setdefault("HF_HOME", cfg["paths"]["hf_home"])  # must precede any Hugging Face import

    import h5py
    import torch

    from lesionbench.features.encoders import load_encoder
    from lesionbench.features.extract import PatchDataset, extract_features, write_feature_h5

    cache = Path(cfg["paths"]["cache_root"])
    data = Path(cfg["paths"]["data_root"])
    log_dir = Path(cfg["paths"]["output_root"]) / "logs" / "features" / args.backbone
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    git = ["git", "-C", str(ROOT), "rev-parse", "HEAD"]
    sha = subprocess.run(git, capture_output=True, text=True).stdout.strip()

    patch_files = sorted((cache / "patches").glob("*.h5"))
    if args.slides:
        patch_files = [p for p in patch_files if p.stem in set(args.slides)]
    patch_files = patch_files[args.task_id :: args.n_tasks]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed, transform, dim = load_encoder(args.backbone, device)
    log.info("task %d/%d: %d slides, backbone %s (%d-d) on %s, git %s",
             args.task_id, args.n_tasks, len(patch_files), args.backbone, dim, device, sha)

    failed = []
    for pf in patch_files:
        out = cache / args.backbone / pf.name
        if out.exists():
            log.info("%s: exists, skipping", pf.stem)
            continue
        handler = logging.FileHandler(log_dir / f"{pf.stem}.log", mode="w")
        logging.getLogger().addHandler(handler)
        try:
            with h5py.File(pf) as f:
                coords, a = f["coords"][:], dict(f.attrs)
            ds = PatchDataset(data / "images" / f"{pf.stem}.tif", coords, int(a["patch_size_level0"]),
                              int(a["patch_size"]), transform)
            t0 = time.time()
            feats = extract_features(ds, embed, args.batch_size, args.num_workers, device)
            write_feature_h5(out, feats, pf, args.backbone, feature_git_sha=sha)
            dt = time.time() - t0
            log.info("%s: %d patches -> %s in %.0f s (%.0f patches/s)", pf.stem, len(coords),
                     tuple(feats.shape), dt, len(coords) / max(dt, 1e-9))
        except Exception:
            log.exception("%s: failed", pf.stem)
            failed.append(pf.stem)
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
    if failed:
        log.error("failed slides: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
