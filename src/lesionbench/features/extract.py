"""Read 0.5 µm/px patches at the cached coordinates, embed them, and write the per-backbone feature cache."""

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def read_patch(slide, x: int, y: int, patch_size_l0: int, out_size: int) -> Image.Image:
    """The level-0 square (x, y, patch_size_l0) as an ``out_size`` RGB image, read from the coarsest pyramid
    level that still has at least ``out_size`` pixels across it, then resized."""
    level = slide.get_best_level_for_downsample(patch_size_l0 / out_size)
    size = round(patch_size_l0 / slide.level_downsamples[level])
    img = slide.read_region((int(x), int(y)), level, (size, size)).convert("RGB")
    return img if size == out_size else img.resize((out_size, out_size), Image.BILINEAR)


class PatchDataset(Dataset):
    """Patches of one slide in cached row order. The slide is opened lazily, once per DataLoader worker."""

    def __init__(self, slide_path, coords, patch_size_l0: int, out_size: int, transform, opener=None):
        self.slide_path, self.coords = str(slide_path), np.asarray(coords)
        self.ps0, self.out_size, self.transform = patch_size_l0, out_size, transform
        self.opener, self._slide = opener, None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        if self._slide is None:
            if self.opener is None:
                import openslide

                self.opener = openslide.OpenSlide
            self._slide = self.opener(self.slide_path)
        x, y = self.coords[i]
        return self.transform(read_patch(self._slide, x, y, self.ps0, self.out_size))


@torch.inference_mode()
def extract_features(dataset: Dataset, embed, batch_size: int = 256, num_workers: int = 0,
                     device: str | torch.device = "cpu") -> np.ndarray:
    """(N, D) float32 features in dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=str(device).startswith("cuda"))
    out = [embed(batch.to(device)).float().cpu() for batch in loader]
    return torch.cat(out).numpy() if out else np.zeros((0, 0), dtype=np.float32)


def write_feature_h5(path, features: np.ndarray, patch_h5, backbone: str, **extra_attrs) -> None:
    """Cache schema: /features plus /coords, /patch_labels, /lesion_ids copied from the patch file in the same
    row order. Written atomically (temp file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(patch_h5) as src, h5py.File(tmp, "w") as f:
        if len(features) != len(src["coords"]):
            raise ValueError(f"{len(features)} feature rows for {len(src['coords'])} patches")
        f.create_dataset("features", data=features.astype(np.float32))
        for key in ("coords", "patch_labels", "lesion_ids"):
            f.create_dataset(key, data=src[key][:])
        f.attrs.update(dict(src.attrs))
        f.attrs.update({"backbone": backbone, "feature_dim": features.shape[1], **extra_attrs})
    os.replace(tmp, path)
