"""Qualitative slide figures from finished runs (fold 0): where each method's K kept patches fall.

    python scripts/render_slides.py --config configs/server.yaml --backbone uni --K 32

Slides are chosen by a fixed rule (lesionbench.viz.slides.choose_slides). Writes PNGs and slides.json to
OUTPUT_ROOT/figures/slides/<backbone>/K<K>/:

- <slide>_kept.png: thumbnail, lesion outlines, each method's kept patches (on tumour / not on tumour).
- <slide>_lesion<id>_zoom.png: micro slides only, up to 4 micro lesions (diameter quantiles), every method.
- <slide>_mechanisms.png: SLICEChat-style patches coloured by the pruner that removed them, and TC-SSA patches
  coloured by the tumour share of the slot they joined.

"On tumour" = max overlap with any lesion >= the primary threshold. Needs OpenSlide and the WSIs (Fir only).
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import openslide  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from lesionbench.config import load_config, write_run_json  # noqa: E402
from lesionbench.metrics.probe import max_overlap  # noqa: E402
from lesionbench.preprocess.annotations import lesions, read_asap_xml, size_class  # noqa: E402
from lesionbench.viz.slides import choose_slides, slot_tumour_fraction  # noqa: E402

METHODS = [
    ("random", "Random"),
    ("abmil", "Top-K ABMIL"),
    ("siimil", "SiiMIL"),
    ("nrk_abmil", "NRK-ABMIL"),
    ("pmil", "PMIL"),
    ("tc_ssa", "TC-SSA"),
    ("lts", "LTS"),
    ("slicechat", "SLICEChat-style"),
    ("oracle", "Oracle-Cov"),
]
KEPT_NORMAL, KEPT_TUMOUR, INK = "#2a78d6", "#eb6834", "#0b0b0b"
SLOT_CMAP = ListedColormap(
    ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
STAGES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#184f95"]  # pruner 1..5, light -> dark


def outlines(ax, geoms, scale, lw=1.0):
    for g in geoms:
        for p in getattr(g, "geoms", [g]):
            x, y = p.exterior.xy
            ax.plot(np.array(x) / scale, np.array(y) / scale, color=INK, lw=lw)


def rings(ax, geoms, scale, radius=28):
    """Fixed-size ring (thumbnail px) around each lesion, so micro lesions are findable at slide scale."""
    for g in geoms:
        c = g.centroid
        ax.add_patch(Circle((c.x / scale, c.y / scale), radius, fill=False, ec=INK, lw=1.0))


def squares(ax, coords, size, scale, colors, alpha=1.0):
    rects = [Rectangle((x / scale, y / scale), size / scale, size / scale) for x, y in coords]
    ax.add_collection(PatchCollection(rects, facecolor=colors, edgecolor="none", alpha=alpha))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backbone", default="uni")
    ap.add_argument("--K", type=int, default=32)
    args = ap.parse_args()

    cfg = load_config(args.config)
    t, st = cfg["primary_overlap_threshold"], cfg["strata_mm"]
    data, cache, out_root = (Path(cfg["paths"][k]) for k in ("data_root", "cache_root", "output_root"))
    runs = out_root / "runs" / args.backbone
    out = out_root / "figures" / "slides" / args.backbone / f"K{args.K}"
    out.mkdir(parents=True, exist_ok=True)

    info = []
    for p in sorted((cache / "patches").glob("test_*.h5")):
        with h5py.File(p) as f:
            d = f["overlap/lesion_diameter_um"][:]
            info.append(
                {
                    "slide_id": p.stem,
                    "stratum": str(f.attrs["stratum"]),
                    "n_patches": len(f["coords"]),
                    "n_micro": sum(
                        size_class(x / 1000, st["itc_max"], st["micro_max"]) == "micro" for x in d
                    ),
                }
            )
    chosen = choose_slides(info)
    write_run_json(out, cfg, backbone=args.backbone, K=args.K, fold=0, slides=chosen)
    (out / "slides.json").write_text(json.dumps(chosen, indent=1))
    kept = {m: np.load(runs / m / "fold0" / f"K{args.K}" / "retained.npz") for m, _ in METHODS}

    for sid, role in {v: k for k, v in chosen.items()}.items():  # a slide chosen twice is drawn once
        with h5py.File(cache / "patches" / f"{sid}.h5") as f:
            coords, a = f["coords"][:], dict(f.attrs)
            rows, les, fr = (f["overlap"][k][:] for k in ("rows", "lesion_ids", "fracs"))
            diam = f["overlap/lesion_diameter_um"][:]
        size, mpp0 = int(a["patch_size_level0"]), float(a["mpp0"])
        tumour = max_overlap(len(coords), rows, fr) >= t
        xml = data / "annotations" / f"{sid}.xml"
        geoms = (
            lesions(*read_asap_xml(xml), merge_px=cfg["preprocess"]["lesion_merge_um"] / mpp0)
            if xml.exists()
            else []
        )
        wsi = openslide.OpenSlide(str(data / "images" / f"{sid}.tif"))
        thumb = wsi.get_thumbnail((1600, 1600))
        scale = wsi.dimensions[0] / thumb.size[0]
        title = f"{sid} ({a['stratum']}, {len(coords):,} patches, K = {args.K}, fold 0)"
        k_, i_ = kept["tc_ssa"][f"W/{sid}"]
        frac = slot_tumour_fraction(k_, i_, kept["tc_ssa"][f"W_val/{sid}"], tumour.astype(float), len(coords))

        # 1. kept-patch map per method
        fig, axes = plt.subplots(3, 3, figsize=(15, 15 * thumb.size[1] / thumb.size[0] + 1.2))
        for ax, (m, name) in zip(axes.flat, METHODS, strict=True):
            idx = kept[m][f"retained/{sid}"]
            ax.imshow(thumb)
            outlines(ax, geoms, scale, lw=0.8)
            rings(ax, geoms, scale)
            # draw kept squares 3x enlarged so K = 32 patches stay visible on a thumbnail
            c = coords[idx] - size
            squares(
                ax, c, 3 * size, scale, [KEPT_TUMOUR if tumour[i] else KEPT_NORMAL for i in idx], alpha=0.85
            )
            hit = int(tumour[idx].sum())
            ax.set_title(f"{name}: {hit}/{len(idx)} kept on tumour", fontsize=11)
            ax.axis("off")
        fig.suptitle(
            title + "\norange = kept patch on tumour, blue = kept patch elsewhere (drawn 3x size); "
            "black = lesion outline, ringed so small lesions are findable",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"{sid}_kept.png", dpi=110)
        plt.close(fig)

        # 2. micro-lesion zooms
        if a["stratum"] == "micro":
            micro = [
                i
                for i, d in enumerate(diam)
                if size_class(d / 1000, st["itc_max"], st["micro_max"]) == "micro"
            ]
            micro = sorted(micro, key=lambda i: diam[i])
            pick = (
                sorted({micro[int(round(q * (len(micro) - 1)))] for q in (0, 1 / 3, 2 / 3, 1)})
                if micro
                else []
            )
            level = wsi.get_best_level_for_downsample(1.0 / mpp0)  # about 1 um / px
            ds = wsi.level_downsamples[level]
            for lid in pick:
                x0, y0, x1, y1 = geoms[lid].bounds
                side = max(2000 / mpp0, 1.6 * max(x1 - x0, y1 - y0))  # >= 2 mm window
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                ox, oy = int(cx - side / 2), int(cy - side / 2)
                img = wsi.read_region((ox, oy), level, (int(side / ds), int(side / ds))).convert("RGB")
                own = rows[(les == lid) & (fr >= t)]
                fig, axes = plt.subplots(2, 5, figsize=(25, 12.8))
                for ax, (m, name) in zip(axes.flat, METHODS, strict=False):  # last axis: TC-SSA slots
                    idx = kept[m][f"retained/{sid}"]
                    ax.imshow(img, extent=(ox, ox + side, oy + side, oy))
                    inwin = idx[
                        (coords[idx, 0] > ox - size)
                        & (coords[idx, 0] < ox + side)
                        & (coords[idx, 1] > oy - size)
                        & (coords[idx, 1] < oy + side)
                    ]
                    for i in own:
                        ax.add_patch(Rectangle(coords[i], size, size, fill=False, ec=INK, lw=0.6, ls=":"))
                    squares(
                        ax,
                        coords[inwin],
                        size,
                        1.0,
                        [KEPT_TUMOUR if tumour[i] else KEPT_NORMAL for i in inwin],
                        alpha=0.55,
                    )
                    outlines(ax, [geoms[lid]], 1.0, lw=1.4)
                    got = np.isin(own, idx).any()
                    ax.set_title(f"{name}: lesion {'KEPT' if got else 'missed'}", fontsize=11)
                    ax.set_xlim(ox, ox + side)
                    ax.set_ylim(oy + side, oy)
                    ax.axis("off")
                ax = axes.flat[-1]  # TC-SSA dilution at lesion scale
                ax.imshow(img, extent=(ox, ox + side, oy + side, oy))
                win = np.flatnonzero(
                    (coords[:, 0] > ox - size)
                    & (coords[:, 0] < ox + side)
                    & (coords[:, 1] > oy - size)
                    & (coords[:, 1] < oy + side)
                    & np.isfinite(frac)
                )
                squares(ax, coords[win], size, 1.0, SLOT_CMAP(frac[win]), alpha=0.7)
                outlines(ax, [geoms[lid]], 1.0, lw=1.4)
                ax.set_xlim(ox, ox + side)
                ax.set_ylim(oy + side, oy)
                ax.set_title(
                    "TC-SSA: tumour share of each patch's slot\n(light = all normal, dark = all tumour)",
                    fontsize=11,
                )
                ax.axis("off")
                fig.suptitle(
                    f"{sid}, lesion {lid}: {diam[lid] / 1000:.2f} mm micro-metastasis, K = {args.K}, "
                    f"fold 0\nblack = lesion outline, dotted = patches that can hit it (≥ {t:.0%} "
                    "overlap), orange / blue = kept on tumour / elsewhere",
                    fontsize=12,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.93))
                fig.savefig(out / f"{sid}_lesion{lid}_zoom.png", dpi=100)
                plt.close(fig)

        # 3. SLICEChat-style drop map and TC-SSA slot map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8 * thumb.size[1] / thumb.size[0] + 1.4))
        drop = kept["slicechat"][f"layer_of_drop/{sid}"]
        ax1.imshow(thumb, alpha=0.35)
        for b in range(5):
            sel = np.flatnonzero(drop == b)
            squares(ax1, coords[sel], size, scale, [STAGES[b]] * len(sel))
        sel = np.flatnonzero(drop == -1)
        squares(ax1, coords[sel] - size, 3 * size, scale, [KEPT_TUMOUR] * len(sel))
        outlines(ax1, geoms, scale, lw=0.8)
        rings(ax1, geoms, scale)
        ax1.set_title(
            "SLICEChat-style: pruner that removed each patch\n(light = 1st … dark = 5th; "
            "orange = kept, drawn 3x)",
            fontsize=11,
        )
        ax1.axis("off")
        ax2.imshow(thumb, alpha=0.35)
        ok = np.flatnonzero(np.isfinite(frac))
        squares(ax2, coords[ok], size, scale, SLOT_CMAP(frac[ok]))
        outlines(ax2, geoms, scale, lw=0.8)
        rings(ax2, geoms, scale)
        ax2.set_title(
            "TC-SSA: tumour share of the slot each patch joined\n(light = all-normal slot, dark = "
            "all-tumour slot)",
            fontsize=11,
        )
        ax2.axis("off")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"{sid}_mechanisms.png", dpi=110)
        plt.close(fig)
        print(role, sid, flush=True)
    print(out)


if __name__ == "__main__":
    main()
