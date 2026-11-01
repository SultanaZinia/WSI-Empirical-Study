"""Control selectors: Random (floor), Oracle (Cov@K ceiling) and Oracle-LSR (LSR@K ceiling)."""

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


class OracleLSRSelector:
    """Lesion round-robin: each lesion's best patch, then each lesion's second, ..., then random fill.

    A patch is eligible for lesion m when its overlap with m is ``>= threshold`` (``> 0`` at threshold 0).
    Lesions are visited in random order, or by ``lesion_priority`` (lower first, random within a priority).
    In the first pass a lesion already hit by an earlier pick is skipped, so with K picks the oracle hits at
    least min(K, hittable lesions). Like ``OracleSelector``, it reads evaluation-only data.
    """

    def select(self, features: Tensor, K: int, *, overlap_rows, overlap_lesions, overlap_fracs,
               threshold: float, seed: int, lesion_priority=None, **ctx) -> SelectionResult:
        g = torch.Generator().manual_seed(seed)
        n = features.shape[0]
        rows, les = torch.as_tensor(overlap_rows).long(), torch.as_tensor(overlap_lesions).long()
        frac = torch.as_tensor(overlap_fracs).double()
        ok = frac >= threshold if threshold > 0 else frac > 0
        rows, les, frac = rows[ok], les[ok], frac[ok]
        lesion_list = torch.unique(les)
        lesion_list = lesion_list[torch.randperm(len(lesion_list), generator=g)]
        if lesion_priority is not None:
            pri = torch.as_tensor(lesion_priority)[lesion_list]
            lesion_list = lesion_list[torch.argsort(pri, stable=True)]
        queues = []  # per lesion: its eligible rows, highest overlap first (ties: lower row)
        for m in lesion_list.tolist():
            r, f = rows[les == m], frac[les == m]
            order = torch.argsort(r, stable=True)
            r, f = r[order], f[order]
            queues.append((m, r[torch.argsort(-f, stable=True)].tolist()))
        lesions_of = {}
        for r, m in zip(rows.tolist(), les.tolist(), strict=True):
            lesions_of.setdefault(r, set()).add(m)

        picked, taken, hit = [], set(), set()
        depth = 0
        while len(picked) < K and any(depth < len(q) for _, q in queues):
            for m, q in queues:
                if len(picked) >= K:
                    break
                if depth >= len(q) or (depth == 0 and m in hit):
                    continue
                r = q[depth]
                if r not in taken:
                    picked.append(r)
                    taken.add(r)
                    hit |= lesions_of[r]
            depth += 1
        rest = [i for i in torch.randperm(n, generator=g).tolist() if i not in taken]
        return SelectionResult(retained_idx=torch.tensor(picked + rest, dtype=torch.long)[:K])


def random_hit_prob(n_patches: int, K: int, n_eligible: int) -> float:
    """Exact P(a uniform K-subset of n_patches rows contains at least one of n_eligible rows).

    1 - C(N - n, K) / C(N, K), computed as a product to stay exact in floating point for large N.
    """
    if n_eligible <= 0:
        return 0.0
    if K >= n_patches or n_eligible > n_patches - K:
        return 1.0
    miss = 1.0
    for j in range(K):
        miss *= (n_patches - n_eligible - j) / (n_patches - j)
    return 1.0 - miss
