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
) -> np.ndarray:
    """Boolean tissue mask: two-level Otsu on HSV saturation and value.

    Tissue is saturated (S above its Otsu threshold) and not bright (V at or below its Otsu threshold).
    Glass and adipose are bright and unsaturated, so both tests drop them. Optional cleaning, in physical
    units (``um_per_px`` is the thumbnail resolution); all off by default, which gives the raw Otsu mask:

    - ``median_um``: median-filter S and V over a window this wide before thresholding (CLAM's ``mthresh``),
      which removes single-pixel speckle.
    - ``min_area_mm2``: drop 8-connected components smaller than this.
    - ``min_width_um``: drop components whose mean width, 2 * area / boundary length, is below this
      (coverslip edges, scratches and other thin lines).
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
    edge = mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    boundary = np.bincount(lab[edge], minlength=n + 1).astype(float)
    width_um = np.divide(2 * area, boundary, out=np.zeros_like(area), where=boundary > 0) * um_per_px
    keep = (area * um_per_px**2 / 1e6 >= min_area_mm2) & (width_um >= min_width_um)
    keep[0] = False
    return keep[lab]
