"""Quality-control overlays for preprocessing, one PNG per slide.

    python scripts/qc_preprocess.py --config configs/server.yaml normal_042 tumor_106 ...

Left: thumbnail with the tissue-mask outline. Right: kept patches (green), positive patches (red) and
lesion outlines (blue). Reads CACHE_ROOT/patches/<slide>.h5; writes OUTPUT_ROOT/qc/preprocess/<slide>.png.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide

from lesionbench.config import load_config
from lesionbench.preprocess.annotations import lesions, read_asap_xml
from lesionbench.preprocess.tissue import tissue_mask
from lesionbench.preprocess.wsi import read_thumbnail


def patch_overlay(shape, coords, labels, ps0, ds):
    """RGBA image at thumbnail resolution: kept patches green, positive patches red."""
    over = np.zeros((*shape, 4))
    for (x, y), lab in zip(coords, labels, strict=True):
        x0, y0 = int(x / ds), int(y / ds)
        x1, y1 = max(int((x + ps0) / ds), x0 + 1), max(int((y + ps0) / ds), y0 + 1)
        over[y0:y1, x0:x1] = (0.85, 0.1, 0.1, 0.55) if lab else (0.1, 0.7, 0.2, 0.25)
    return over


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("slides", nargs="+")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = Path(cfg["paths"]["data_root"])
    out_dir = Path(cfg["paths"]["output_root"]) / "qc" / "preprocess"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_mpp = 10.0 / cfg["preprocess"]["thumbnail_magnification"]

    for sid in args.slides:
        slide = openslide.OpenSlide(str(data / "images" / f"{sid}.tif"))
        thumb, ds, mpp0 = read_thumbnail(slide, thumb_mpp)
        with h5py.File(Path(cfg["paths"]["cache_root"]) / "patches" / f"{sid}.h5") as f:
            coords, labels, a = f["coords"][:], f["patch_labels"][:], dict(f.attrs)
        xml = data / "annotations" / f"{sid}.xml"
        les = []
        if xml.exists():
            les = lesions(*read_asap_xml(xml), merge_px=cfg["preprocess"]["lesion_merge_um"] / mpp0)

        fig, axes = plt.subplots(1, 2, figsize=(16, 16 * thumb.shape[0] / thumb.shape[1] / 2 + 1))
        axes[0].imshow(thumb)
        axes[0].contour(tissue_mask(thumb), levels=[0.5], colors="k", linewidths=0.6)
        axes[1].imshow(thumb)
        axes[1].imshow(patch_overlay(thumb.shape[:2], coords, labels, a["patch_size_level0"], ds))
        for lesion in les:
            for poly in getattr(lesion, "geoms", [lesion]):
                xy = np.asarray(poly.exterior.coords) / ds
                axes[1].plot(xy[:, 0], xy[:, 1], color="#1f4fd1", linewidth=0.8)
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(f"{sid}  stratum={a['stratum']}  patches={len(coords)}  positive={int(labels.sum())}  "
                     f"lesions={a['n_lesions']} (no positive patch: {len(a['unlabeled_lesions'])})")
        fig.tight_layout()
        fig.savefig(out_dir / f"{sid}.png", dpi=110)
        plt.close(fig)
        print(out_dir / f"{sid}.png")


if __name__ == "__main__":
    main()
