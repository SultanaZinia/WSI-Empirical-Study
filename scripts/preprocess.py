"""Preprocess CAMELYON16 slides into per-slide patch files (coords, patch labels, lesion ids).

    python scripts/preprocess.py --config configs/server.yaml [--task-id I --n-tasks N]

Reads DATA_ROOT/images/*.tif and DATA_ROOT/annotations/*.xml, writes CACHE_ROOT/patches/<slide>.h5.
Task I of N handles every N-th slide. Slides whose output exists are skipped. Per-slide logs go to
OUTPUT_ROOT/logs/preprocess/. Exits non-zero if any slide failed.
"""

import argparse
import logging
import subprocess
from pathlib import Path

import openslide

from lesionbench.config import ROOT, load_config
from lesionbench.preprocess.pipeline import preprocess_slide, write_patch_h5
from lesionbench.preprocess.wsi import read_thumbnail

log = logging.getLogger("preprocess")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = Path(cfg["paths"]["data_root"])
    out_dir = Path(cfg["paths"]["cache_root"]) / "patches"
    log_dir = Path(cfg["paths"]["output_root"]) / "logs" / "preprocess"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    git = ["git", "-C", str(ROOT), "rev-parse", "HEAD"]
    sha = subprocess.run(git, capture_output=True, text=True).stdout.strip()
    thumb_mpp = 10.0 / cfg["preprocess"]["thumbnail_magnification"]

    slides = sorted((data / "images").glob("*.tif"))[args.task_id :: args.n_tasks]
    log.info("task %d/%d: %d slides, git %s", args.task_id, args.n_tasks, len(slides), sha or "unknown")
    failed = []
    for tif in slides:
        sid = tif.stem
        out = out_dir / f"{sid}.h5"
        if out.exists():
            log.info("%s: exists, skipping", sid)
            continue
        handler = logging.FileHandler(log_dir / f"{sid}.log", mode="w")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logging.getLogger().addHandler(handler)
        try:
            slide = openslide.OpenSlide(str(tif))
            thumb, ds, mpp0 = read_thumbnail(slide, thumb_mpp)
            xml = data / "annotations" / f"{sid}.xml"
            res = preprocess_slide(sid, thumb, ds, slide.dimensions, mpp0, xml if xml.exists() else None, cfg)
            write_patch_h5(out, res, git_sha=sha)
            a = res["attrs"]
            log.info("%s: %d patches, %d positive, %d lesions (%d without a positive patch), stratum %s",
                     sid, len(res["coords"]), int(res["patch_labels"].sum()), a["n_lesions"],
                     len(a["unlabeled_lesions"]), a["stratum"])
        except Exception:
            log.exception("%s: failed", sid)
            failed.append(sid)
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
    if failed:
        log.error("failed slides: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
