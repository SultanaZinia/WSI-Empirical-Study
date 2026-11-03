"""Tissue-mask candidates scored against the annotations, before any grid is rebuilt.

    python scripts/mask_sweep.py --config configs/server.yaml [--task-id I --n-tasks N]
    python scripts/mask_sweep.py --config configs/server.yaml --summarise

Every candidate selects cells of the same lattice as the current grid, so its effect on tumour is exact from
the existing patch files (preprocess.mask_qc.score_mask). Per slide: OUTPUT_ROOT/mask_sweep/<slide>.csv (one
row per candidate x tissue cut) and <slide>_lesions.csv (lesions lost). --summarise writes summary.csv,
lesions_lost.csv and run.json over all slides.
"""

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

from lesionbench.config import load_config, write_run_json

log = logging.getLogger("mask_sweep")

CANDIDATES = {  # tissue_mask keyword arguments, physical units
    "current": {},
    "median": {"median_um": 50},
    "median_width": {"median_um": 50, "min_width_um": 30},
    "median_width_area": {"median_um": 50, "min_width_um": 30, "min_area_mm2": 0.01},
    "width_area": {"min_width_um": 30, "min_area_mm2": 0.01},
    "lines_1mm": {"min_width_um": 30, "line_length_um": 1000},
    "lines_500um": {"min_width_um": 30, "line_length_um": 500},
}
CUTS = (0.0, 0.10, 0.25, 0.50)
LESION_COLS = ["slide_id", "candidate", "cut", "lesion_id", "diameter_um"]


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def run(cfg, task_id, n_tasks):
    import h5py
    import openslide

    from lesionbench.preprocess.grid import cell_tissue_fraction
    from lesionbench.preprocess.mask_qc import score_mask
    from lesionbench.preprocess.tissue import tissue_mask
    from lesionbench.preprocess.wsi import read_thumbnail

    data = Path(cfg["paths"]["data_root"])
    patches = Path(cfg["paths"]["cache_root"]) / "patches"
    out_dir = Path(cfg["paths"]["output_root"]) / "mask_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_mpp = 10.0 / cfg["preprocess"]["thumbnail_magnification"]
    t_primary = cfg["primary_overlap_threshold"]
    for pf in sorted(patches.glob("*.h5"))[task_id::n_tasks]:
        sid = pf.stem
        if (out_dir / f"{sid}.csv").exists():
            continue
        with h5py.File(pf) as f:
            coords, a, g = f["coords"][:], dict(f.attrs), f["overlap"]
            rows, les, ov, diam = g["rows"][:], g["lesion_ids"][:], g["fracs"][:], g["lesion_diameter_um"][:]
        slide = openslide.OpenSlide(str(data / "images" / f"{sid}.tif"))
        thumb, ds, mpp0 = read_thumbnail(slide, thumb_mpp)
        res, lost = [], []
        for name, kw in CANDIDATES.items():
            mask = tissue_mask(thumb, um_per_px=ds * mpp0, **kw)
            cells, frac = cell_tissue_fraction(mask, ds, slide.dimensions, int(a["patch_size_level0"]))
            r, lo = score_mask(coords, cells, frac, rows, les, ov, diam, t_primary, CUTS)
            res += [{"slide_id": sid, "stratum": a["stratum"], "candidate": name, **x} for x in r]
            lost += [{"slide_id": sid, "candidate": name, **x} for x in lo]
        write_csv(out_dir / f"{sid}_lesions.csv", lost, LESION_COLS)
        write_csv(out_dir / f"{sid}.csv", res, list(res[0]))  # written last: marks the slide done
        log.info("%s done", sid)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarise(cfg):
    out_dir = Path(cfg["paths"]["output_root"]) / "mask_sweep"
    slide_files = [p for p in sorted(out_dir.glob("*.csv"))
                   if not p.stem.endswith("_lesions") and p.stem not in ("summary", "lesions_lost")]
    rows = [r for p in slide_files for r in read_csv(p)]
    lost = [r for p in sorted(out_dir.glob("*_lesions.csv")) for r in read_csv(p)]
    excl = set(cfg["non_exhaustive_slides"])
    num = ["n_old", "n_new", "n_added", "tumour_patches", "tumour_dropped", "lesions_hittable",
           "lesions_lost"]
    agg = defaultdict(lambda: defaultdict(float))
    for r in rows:
        k = (r["candidate"], float(r["cut"]))
        agg[k]["slides"] += 1
        for c in num:
            agg[k][c] += float(r[c])
    for r in lost:
        k = (r["candidate"], float(r["cut"]))
        small = float(r["diameter_um"]) <= cfg["strata_mm"]["micro_max"] * 1000
        agg[k]["lost_le_2mm"] += small
        agg[k]["lost_le_2mm_exhaustive"] += small and r["slide_id"] not in excl
    out = []
    for (cand, cut), a in agg.items():
        out.append({"candidate": cand, "cut": cut, **{c: int(a[c]) for c in ["slides", *num]},
                    "patches_kept_share": round(a["n_new"] / a["n_old"], 4),
                    "tumour_dropped_share": round(a["tumour_dropped"] / max(a["tumour_patches"], 1), 5),
                    "lost_le_2mm": int(a["lost_le_2mm"]),
                    "lost_le_2mm_exhaustive": int(a["lost_le_2mm_exhaustive"])})
    write_csv(out_dir / "summary.csv", out, list(out[0]))
    write_csv(out_dir / "lesions_lost.csv", lost, LESION_COLS)
    write_run_json(out_dir, cfg, candidates=CANDIDATES, cuts=list(CUTS))
    print(",".join(out[0]))
    for o in out:
        print(",".join(str(v) for v in o.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.summarise:
        summarise(cfg)
    else:
        run(cfg, args.task_id, args.n_tasks)


if __name__ == "__main__":
    main()
