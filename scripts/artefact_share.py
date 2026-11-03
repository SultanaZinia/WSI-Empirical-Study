"""Share of each method's kept budget that lands on low-tissue cells or thin-line (glass-edge) artefacts.

    python scripts/artefact_share.py --config configs/server.yaml [--task-id I --n-tasks N]
    python scripts/artefact_share.py --config configs/server.yaml --summarise

Per slide: OUTPUT_ROOT/artefact_share/flags/<slide>.npz with, per patch row of the current grid (grid v1),
``tissue_frac`` (tissue fraction on the grid-v1 mask) and ``line`` (cell removed by the length-gated line
rule). --summarise reads every OUTPUT_ROOT/runs/<backbone>/<method>/fold<f>/K<K>/retained.npz and writes
per_slide.csv (share of kept patches flagged, and the whole-bag share) and summary.csv (mean over slides,
then mean and SD over folds, per method x K x split). Pen marks pass the mask as tissue and are not flagged.
"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from lesionbench.config import load_config, write_run_json

log = logging.getLogger("artefact_share")

LINE_RULE = {"min_width_um": 30, "line_length_um": 500}  # mask_sweep "lines_500um"
LOW_TISSUE = 0.10


def flags(cfg, task_id, n_tasks):
    import h5py
    import openslide

    from lesionbench.preprocess.grid import cell_tissue_fraction, patch_fraction
    from lesionbench.preprocess.tissue import tissue_mask
    from lesionbench.preprocess.wsi import read_thumbnail

    data = Path(cfg["paths"]["data_root"])
    patches = Path(cfg["paths"]["cache_root"]) / "patches"
    out_dir = Path(cfg["paths"]["output_root"]) / "artefact_share" / "flags"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_mpp = 10.0 / cfg["preprocess"]["thumbnail_magnification"]
    for pf in sorted(patches.glob("*.h5"))[task_id::n_tasks]:
        sid = pf.stem
        if (out_dir / f"{sid}.npz").exists():
            continue
        with h5py.File(pf) as f:
            coords, ps0 = f["coords"][:], int(f.attrs["patch_size_level0"])
        slide = openslide.OpenSlide(str(data / "images" / f"{sid}.tif"))
        thumb, ds, mpp0 = read_thumbnail(slide, thumb_mpp)
        frac = {}
        for name, kw in (("v1", {}), ("line", LINE_RULE)):
            cells, f_cells = cell_tissue_fraction(tissue_mask(thumb, um_per_px=ds * mpp0, **kw), ds,
                                                  slide.dimensions, ps0)
            frac[name] = patch_fraction(coords, cells, f_cells)
        np.savez_compressed(out_dir / f"{sid}.npz", tissue_frac=frac["v1"].astype(np.float32),
                            line=frac["line"] == 0)
        log.info("%s done", sid)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarise(cfg, backbone):
    out = Path(cfg["paths"]["output_root"])
    flag_dir = out / "artefact_share" / "flags"
    fl = {}
    rows = []
    for rf in sorted((out / "runs" / backbone).glob("*/fold*/K*/retained.npz")):
        method, fold, K = rf.parts[-4], int(rf.parts[-3][4:]), rf.parts[-2][1:]
        z = np.load(rf)
        for key in z.files:
            if not key.startswith("retained/"):
                continue
            sid = key.split("/", 1)[1]
            if sid not in fl:
                f = np.load(flag_dir / f"{sid}.npz")
                fl[sid] = (f["tissue_frac"] < LOW_TISSUE, f["line"])
            low, line = fl[sid]
            idx = z[key]
            rows.append({"method": method, "fold": fold, "K": K, "slide_id": sid,
                         "split": "test" if sid.startswith("test_") else "val", "n_kept": len(idx),
                         "low_share": float(low[idx].mean()) if len(idx) else 0.0,
                         "line_share": float(line[idx].mean()) if len(idx) else 0.0,
                         "bag_low_share": float(low.mean()), "bag_line_share": float(line.mean())})
    write_csv(out / "artefact_share" / "per_slide.csv", rows)

    cols = ["low_share", "line_share", "bag_low_share", "bag_line_share"]
    per_fold = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for c in cols:
            per_fold[(r["method"], r["K"], r["split"])][(r["fold"], c)].append(r[c])
    summary = []
    for (method, K, split), d in sorted(per_fold.items()):
        s = {"method": method, "K": K, "split": split}
        for c in cols:
            fold_means = [np.mean(v) for (_, cc), v in sorted(d.items()) if cc == c]
            s[f"{c}_mean"] = round(float(np.mean(fold_means)), 4)
            s[f"{c}_sd"] = round(float(np.std(fold_means, ddof=1)), 4) if len(fold_means) > 1 else 0.0
        s["n_folds"] = len(fold_means)
        summary.append(s)
    write_csv(out / "artefact_share" / "summary.csv", summary)
    write_run_json(out / "artefact_share", cfg, backbone=backbone, low_tissue=LOW_TISSUE, line_rule=LINE_RULE)
    print(",".join(summary[0]))
    for s in summary:
        print(",".join(str(v) for v in s.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="uni")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.summarise:
        summarise(cfg, args.backbone)
    else:
        flags(cfg, args.task_id, args.n_tasks)


if __name__ == "__main__":
    main()
