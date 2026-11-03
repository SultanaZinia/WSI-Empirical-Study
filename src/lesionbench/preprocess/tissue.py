"""Tissue mask from a low-magnification RGB thumbnail (~1.25x)."""

import numpy as np
import scipy.ndimage as ndi


def otsu_threshold(values: np.ndarray) -> int:
    """Otsu threshold for uint8 values; the upper class is ``values > t``.

    A constant input has no split, so it returns that value (nothing lies above it).
    """
    values = np.asarray(values, dtype=np.uint8).ravel()
    if values.min() == values.max():
        return int(values[0])
    p = np.bincount(values, minlength=256) / values.size
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    return int(np.argmax(np.nan_to_num(between)))


def tissue_mask(
    rgb: np.ndarray,
    um_per_px: float = 1.0,
    median_um: float = 0.0,
    min_area_mm2: float = 0.0,
    min_width_um: float = 0.0,
    line_length_um: float = 0.0,
) -> np.ndarray:
    """Boolean tissue mask: two-level Otsu on HSV saturation and value.

    Tissue is saturated (S above its Otsu threshold) and not bright (V at or below its Otsu threshold).
    Glass and adipose are bright and unsaturated, so both tests drop them. Optional cleaning, in physical
    units (``um_per_px`` is the thumbnail resolution); all off by default, which gives the raw Otsu mask:

    - ``median_um``: median-filter S and V over a window this wide before thresholding (CLAM's ``mthresh``),
      which removes single-pixel speckle.
    - ``min_area_mm2``: drop 8-connected components smaller than this.
    - ``min_width_um``: drop components with no part at least this wide, i.e. that vanish under a
      morphological opening with a square this wide (coverslip edges, scratches and other thin lines).
      Porous tissue is kept whole: one wide part is enough.
    - ``line_length_um``: apply the width rule only to components at least this long (longest bounding-box
      side), so that thin lines go but short specks, which can hold isolated tumour cells, stay.
    """
    rgb = np.asarray(rgb, dtype=np.int32)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    v = mx.astype(np.uint8)
    s = np.where(mx > 0, (mx - mn) * 255 // np.maximum(mx, 1), 0).astype(np.uint8)
    k = int(round(median_um / um_per_px)) | 1  # odd window; 1 = no filtering
    if k > 1:
        s, v = ndi.median_filter(s, size=k), ndi.median_filter(v, size=k)
    mask = (s > otsu_threshold(s)) & (v <= otsu_threshold(v))
    if min_area_mm2 <= 0 and min_width_um <= 0:
        return mask
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    area = np.bincount(lab.ravel(), minlength=n + 1).astype(float)
    keep = area * um_per_px**2 / 1e6 >= min_area_mm2
    w = int(round(min_width_um / um_per_px))
    if w > 1:
        wide = np.zeros(n + 1, dtype=bool)
        wide[np.unique(lab[ndi.binary_opening(mask, structure=np.ones((w, w)))])] = True
        long = np.ones(n + 1, dtype=bool)
        if line_length_um > 0:
            ext = [max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) for sl in ndi.find_objects(lab)]
            long[1:] = np.asarray(ext) * um_per_px >= line_length_um
        keep &= wide | ~long
    keep[0] = False
    return keep[lab]
