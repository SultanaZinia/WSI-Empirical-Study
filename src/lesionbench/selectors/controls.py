"""Control selectors: Random (floor) and Oracle (ceiling)."""

import torch
from torch import Tensor

from .base import SelectionResult


class RandomSelector:
    """K patches sampled uniformly without replacement from the slide's tissue patches."""

    def select(self, features: Tensor, K: int, *, seed: int, **ctx) -> SelectionResult:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(features.shape[0], generator=g)
        return SelectionResult(retained_idx=perm[:K])


class OracleSelector:
    """All tumor patches first (up to K, in random order), then random normal patches to fill K.

    The only selector allowed to see ``patch_labels``.
    """

    def select(self, features: Tensor, K: int, *, patch_labels: Tensor, seed: int, **ctx) -> SelectionResult:
        g = torch.Generator().manual_seed(seed)
        y = torch.as_tensor(patch_labels).bool()
        pos = torch.nonzero(y).flatten()
        neg = torch.nonzero(~y).flatten()
        pos = pos[torch.randperm(len(pos), generator=g)]
        neg = neg[torch.randperm(len(neg), generator=g)]
        return SelectionResult(retained_idx=torch.cat([pos, neg])[:K])
