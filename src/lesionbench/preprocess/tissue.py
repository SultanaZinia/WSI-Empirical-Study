"""Tissue mask from a low-magnification RGB thumbnail (~1.25x)."""

import numpy as np


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


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean tissue mask: two-level Otsu on HSV saturation and value.

    Tissue is saturated (S above its Otsu threshold) and not bright (V at or below its Otsu threshold).
    Glass and adipose are bright and unsaturated, so both tests drop them.
    """
    rgb = np.asarray(rgb, dtype=np.int32)
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    v = mx.astype(np.uint8)
    s = np.where(mx > 0, (mx - mn) * 255 // np.maximum(mx, 1), 0).astype(np.uint8)
    return (s > otsu_threshold(s)) & (v <= otsu_threshold(v))
