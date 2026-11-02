"""Normal-referenced novelty selectors: SiiMIL (Su et al. 2023) and NRK-ABMIL (Sajjad et al. 2023).

Both keep the patches least similar to a bank of "key" patches from normal training slides. They differ only
in how the bank is built and how many nearest keys are averaged:

- SiiMIL (github.com/JoeSu666/SiiMIL, keyset_lrn.py + sii.py): CUR leverage keys, t = 100 per slide; k = 150.
- NRK-ABMIL (github.com/cialab/NRKMIL, notebooks): random greedy de-duplication at cosine 0.95; k = 5.

Keys must come only from normal slides in the training split, never from the slides being evaluated.
"""

import torch
from torch import Tensor

from .base import SelectionResult, topk_indices


def cur_keys(features: Tensor, t: int = 100) -> Tensor:
    """SiiMIL key set of one normal slide: up to ``t`` patches with the highest CUR leverage score.

    Leverage of patch j = squared norm of row j of the top-``rank`` left singular vectors (equal to the column
    scores of the official code's SVD of ``features.T``), capped at 1. Ties keep the lower row first.
    """
    X = features.double()
    rank = int(torch.linalg.matrix_rank(X))
    U, _, _ = torch.linalg.svd(X, full_matrices=False)
    leverage = (U[:, :rank] ** 2).sum(dim=1).clamp(max=1.0)
    keep = torch.argsort(-leverage, stable=True)[: min(rank, t)]
    return features[keep]


def dfi_keys(features: Tensor, threshold: float = 0.95, seed: int = 0) -> Tensor:
    """NRK key set of one normal slide: repeatedly take a random remaining patch as a key and discard every
    remaining patch with cosine similarity > ``threshold`` to it.

    Runs on ``features.device``. A mask of remaining patches avoids copying them every step; the random
    draws come from a CPU generator over the remaining patches in row order, so keys do not depend on the
    device.
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.nn.functional.normalize(features.double(), dim=1)
    alive = torch.ones(len(features), dtype=torch.bool, device=features.device)
    keys, n = [], len(features)
    while n:
        r = int(torch.randint(n, (1,), generator=g))
        pick = int(torch.nonzero(alive).flatten()[r])
        keys.append(pick)
        alive &= (X @ X[pick]) <= threshold
        alive[pick] = False
        n = int(alive.sum())
    return features[torch.tensor(keys, device=features.device)]


def mean_topk_similarity(features: Tensor, keys: Tensor, k: int, chunk: int | None = None,
                         max_entries: int = 2**31) -> Tensor:
    """Per patch: mean of its ``k`` highest cosine similarities to the keys (all keys if k > #keys).

    Patches are scored ``chunk`` at a time; by default the chunk keeps each (chunk x #keys) similarity table
    under ``max_entries`` (NRK banks on UNI hold ~10^6 keys). The result does not depend on the chunk.
    """
    Kn = torch.nn.functional.normalize(keys.float(), dim=1)
    k = min(k, len(keys))
    chunk = chunk or max(1, min(4096, max_entries // max(len(keys), 1)))
    out = []
    for i in range(0, len(features), chunk):
        Q = torch.nn.functional.normalize(features[i : i + chunk].float(), dim=1)
        out.append((Q @ Kn.T).topk(k, dim=1).values.mean(dim=1))
    return torch.cat(out) if out else torch.empty(0)


class NormalBankSelector:
    """Keep the K patches least similar to the normal key bank (lowest mean top-k cosine similarity)."""

    def __init__(self, keys: Tensor, k: int):
        self.keys, self.k = keys, k

    def select(self, features: Tensor, K: int, **ctx) -> SelectionResult:
        novelty = -mean_topk_similarity(features, self.keys, self.k)
        return SelectionResult(retained_idx=topk_indices(novelty, K), scores=novelty)
