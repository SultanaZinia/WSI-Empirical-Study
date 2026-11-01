"""Per-slide preprocessing: thumbnail -> tissue grid -> lesions -> patch labels -> patch HDF5."""

import json
import os
from pathlib import Path

import h5py
import numpy as np

from .annotations import lesions, read_asap_xml, stratum
from .grid import patch_grid, patch_size_level0
from .labels import label_patches
from .tissue import tissue_mask


def preprocess_slide(slide_id, thumb_rgb, mask_downsample, level0_size, mpp0, xml_path, cfg) -> dict:
    """Patch coords, labels and lesion ids for one slide. ``xml_path`` is None for slides without tumor."""
    p = cfg["preprocess"]
    ps0 = patch_size_level0(p["patch_size"], p["target_mpp"], mpp0)
    coords = patch_grid(tissue_mask(thumb_rgb), mask_downsample, level0_size, ps0, p["min_tissue_fraction"])
    les = []
    if xml_path is not None:
        les = lesions(*read_asap_xml(xml_path), merge_px=p["lesion_merge_um"] / mpp0)
    labels, lesion_ids, unlabeled = label_patches(coords, ps0, les, p["label_overlap_threshold"])
    return {
        "coords": coords,
        "patch_labels": labels,
        "lesion_ids": lesion_ids,
        "attrs": {
            "slide_id": slide_id,
            "mpp0": mpp0,
            "target_mpp": p["target_mpp"],
            "magnification": 10.0 / p["target_mpp"],  # 1 µm/px = 10x convention
            "patch_size": p["patch_size"],
            "patch_size_level0": ps0,
            "mask_downsample": mask_downsample,
            "n_lesions": len(les),
            "unlabeled_lesions": np.asarray(unlabeled, dtype=np.int32),
            "stratum": stratum(les, mpp0, cfg["strata_mm"]["itc_max"], cfg["strata_mm"]["micro_max"]),
            "preprocess_config": json.dumps(p, sort_keys=True),
        },
    }


def write_patch_h5(path, result: dict, **extra_attrs) -> None:
    """Write atomically (temp file, then rename) so an interrupted job never leaves a file that looks done."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("coords", data=result["coords"].astype(np.int64))
        f.create_dataset("patch_labels", data=result["patch_labels"].astype(np.uint8))
        f.create_dataset("lesion_ids", data=result["lesion_ids"].astype(np.int32))
        f.attrs.update({**result["attrs"], **extra_attrs})
    os.replace(tmp, path)
