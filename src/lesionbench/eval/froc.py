"""Shared tumour scorers for FROC and the official lesion masks (docs/deviations.md, FROC row).

Two weakly supervised per-fold scorers give every patch a tumour probability:
- ``abmil``: the full-bag ABMIL classifier on single patches. Its classifier is linear, so the slide logit is
  exactly the attention-weighted mean of these patch logits.
- ``maxmil``: PMIL's max-MIL instance head, saved before PMIL's per-K fine-tuning.
"""

from pathlib import Path

import numpy as np
import tifffile
import torch

from ..selectors.abmil import GatedABMIL

SCORERS = ("abmil", "maxmil")


def load_scorer(name: str, run_root, fold: int, in_dim: int, dropout: float = 0.25):
    """features (N, D) Tensor -> (N,) float64 tumour probabilities, from the fold's saved weights."""
    fold_dir = Path(run_root)
    if name == "abmil":
        model = GatedABMIL(in_dim, dropout=dropout)
        model.load_state_dict(torch.load(fold_dir / "abmil_full" / f"fold{fold}" / "Kfull" / "model.pt",
                                         map_location="cpu"))
        model.eval()

        def logits(x):
            return model.classifier(model.fc(x))
    elif name == "maxmil":
        head = torch.nn.Linear(in_dim, 2)
        head.load_state_dict(torch.load(fold_dir / "pmil" / f"fold{fold}" / "maxmil_head.pt",
                                        map_location="cpu"))
        head.eval()
        logits = head
    else:
        raise ValueError(f"unknown scorer {name!r}")

    @torch.no_grad()
    def score(features: torch.Tensor) -> np.ndarray:
        return torch.softmax(logits(features), dim=1)[:, 1].double().numpy()

    return score


def read_mask(path, level: int) -> tuple[np.ndarray, float, float]:
    """(mask at ``level``, its pixel spacing in µm, its downsample from level 0) of a CAMELYON16 mask TIFF
    (0 background, 1 normal, 2 tumour). The spacing comes from the TIFF's own resolution tags (cm units),
    as the official code takes it from the mask image."""
    with tifffile.TiffFile(path) as t:
        levels = t.series[0].levels
        mask = levels[level].asarray()
        page = t.pages[0]
        num, den = page.tags["XResolution"].value
        if page.tags["ResolutionUnit"].value != 3:  # 3 = centimetre
            raise ValueError(f"{path}: unexpected resolution unit")
        downsample = levels[0].shape[1] / levels[level].shape[1]
    return mask, 1e4 * den / num * downsample, downsample
