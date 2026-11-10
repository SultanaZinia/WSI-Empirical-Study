"""Slides for training and evaluation: features + patch grid (model inputs) kept apart from evaluation-only
annotations (patch labels, lesion ids, lesion overlaps), which never enter a bag."""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor

from ..metrics.probe import max_overlap
from ..selectors.slicechat import grid_from_coords


@dataclass
class Slide:
    slide_id: str
    split: str  # "train" (official training set) or "test"
    label: int  # official slide label: training from the file name (tumor_/normal_), test from annotations
    features: Tensor  # (N, D) float32
    grid: Tensor  # (N, 2) long, patch grid from level-0 coords
    stratum: str
    n_lesions: int
    patch_labels: np.ndarray = field(repr=False)  # evaluation only (at the primary threshold if given)
    lesion_ids: np.ndarray = field(repr=False)  # evaluation only
    overlap: tuple = field(repr=False)  # (rows, lesion ids, fracs), evaluation only
    lesion_diameter_um: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)  # evaluation only
    groups: Tensor | None = field(default=None, repr=False)  # HistoSelect: CONCH zero-shot tissue group
    question: Tensor | None = field(default=None, repr=False)  # HistoSelect: fixed question embedding


def slide_ids(cache_root, backbone: str) -> list[str]:
    return sorted(p.stem for p in (Path(cache_root) / backbone).glob("*.h5"))


@lru_cache(maxsize=4)
def _question(path: str) -> Tensor:
    return torch.from_numpy(np.load(path))


def load_slide(cache_root, backbone: str, slide_id: str, label_threshold: float | None = None) -> Slide:
    """``label_threshold``: patch tumour label = max lesion overlap >= t, from the stored overlaps (the
    primary threshold); None keeps the cached labels (overlap >= 0.10)."""
    root = Path(cache_root)
    with h5py.File(root / backbone / f"{slide_id}.h5") as f:
        feats, coords, a = f["features"][:], f["coords"][:], dict(f.attrs)
        labels, ids = f["patch_labels"][:], f["lesion_ids"][:]
    with h5py.File(root / "patches" / f"{slide_id}.h5") as f:
        overlap = tuple(f["overlap"][k][:] for k in ("rows", "lesion_ids", "fracs"))
        diam = f["overlap"]["lesion_diameter_um"][:] if "lesion_diameter_um" in f["overlap"] else np.zeros(0)
    if label_threshold is not None:
        labels = (max_overlap(len(feats), overlap[0], overlap[2]) >= label_threshold).astype(np.uint8)
    split = "test" if slide_id.startswith("test_") else "train"
    label = int(slide_id.startswith("tumor_")) if split == "train" else int(int(a["n_lesions"]) > 0)
    grid = grid_from_coords(coords, int(a["patch_size_level0"]))
    hs = root / "histoselect"  # written by scripts/histoselect_groups.py; absent -> None (other methods)
    groups = torch.from_numpy(np.load(hs / f"{slide_id}.npy").astype(np.int64)) \
        if (hs / f"{slide_id}.npy").exists() else None
    question = _question(str(hs / "question.npy")) if (hs / "question.npy").exists() else None
    if groups is not None and len(groups) != len(feats):
        raise ValueError(f"{slide_id}: {len(groups)} tissue groups for {len(feats)} patches")
    return Slide(slide_id, split, label, torch.from_numpy(feats), grid, str(a["stratum"]),
                 int(a["n_lesions"]), labels, ids, overlap, diam, groups, question)
