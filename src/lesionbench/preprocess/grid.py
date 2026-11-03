"""Non-overlapping patch grid at a target resolution, in level-0 pixel coordinates."""

import numpy as np


def patch_size_level0(patch_size: int, target_mpp: float, mpp0: float) -> int:
    """Side length in level-0 pixels of a ``patch_size`` patch at ``target_mpp`` (e.g. 256 @ 0.5 µm/px)."""
    return int(round(patch_size * target_mpp / mpp0))


def cell_tissue_fraction(
    mask: np.ndarray, mask_downsample: float, level0_size: tuple[int, int], patch_size_l0: int
) -> tuple[np.ndarray, np.ndarray]:
    """Every whole grid cell, row-major (y, then x): ((M, 2) level-0 top-left coords, (M,) tissue fraction
    measured on the mask)."""
    width, height = level0_size
    ps = patch_size_l0
    xs = np.arange(0, width - ps + 1, ps)
    ys = np.arange(0, height - ps + 1, ps)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()

    h, w = mask.shape
    integral = np.zeros((h + 1, w + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.float64), axis=0), axis=1)
    x0 = np.clip(np.floor(gx / mask_downsample).astype(int), 0, w)
    y0 = np.clip(np.floor(gy / mask_downsample).astype(int), 0, h)
    x1 = np.clip(np.ceil((gx + ps) / mask_downsample).astype(int), 0, w)
    y1 = np.clip(np.ceil((gy + ps) / mask_downsample).astype(int), 0, h)
    area = (x1 - x0) * (y1 - y0)
    tissue = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    frac = np.divide(tissue, area, out=np.zeros_like(tissue), where=area > 0)
    return np.stack([gx, gy], axis=1).astype(np.int64), frac


def patch_fraction(coords: np.ndarray, cells: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """(N,) tissue fraction of each patch in ``coords`` (N, 2), looked up in ``cell_tissue_fraction``'s
    output for the same lattice."""
    pos = {(int(x), int(y)): i for i, (x, y) in enumerate(cells)}
    return frac[np.array([pos[(int(x), int(y))] for x, y in coords], dtype=np.int64)]


def patch_grid(
    mask: np.ndarray,
    mask_downsample: float,
    level0_size: tuple[int, int],
    patch_size_l0: int,
    min_tissue_fraction: float = 0.0,
) -> np.ndarray:
    """Top-left (x, y) level-0 coords of grid cells that contain tissue, row-major (y, then x).

    ``mask`` is the thumbnail tissue mask, ``mask_downsample`` the level-0 pixels per mask pixel and
    ``level0_size`` the slide's (width, height). Only whole patches inside the slide are emitted.
    A cell is kept if its tissue fraction (measured on the mask) is > 0 and >= ``min_tissue_fraction``.
    """
    coords, frac = cell_tissue_fraction(mask, mask_downsample, level0_size, patch_size_l0)
    return coords[(frac > 0) & (frac >= min_tissue_fraction)]
