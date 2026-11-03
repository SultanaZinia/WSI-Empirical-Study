"""Shared selector interface. Every method returns indices into the slide's cached patch rows."""

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


@dataclass
class SelectionResult:
    retained_idx: Tensor  # (min(K, N),) long, rows of the cached slide
    scores: Tensor | None = None  # (N,) per-patch selection score
    W: Tensor | None = None  # (K, N) soft assignment, merging methods only
    layer_of_drop: Tensor | None = None  # (N,) layer at which each token was pruned, in-encoder pruners only
    stage_scores: Tensor | None = None  # (L, N) pruner scores per stage, NaN once dropped; in-encoder pruners


class Selector(Protocol):
    def select(self, features: Tensor, K: int, **ctx) -> SelectionResult: ...


def topk_indices(scores: Tensor, K: int) -> Tensor:
    """Indices of the K highest scores; ties go to the lower row index. Returns all N rows if K > N."""
    order = torch.argsort(-scores, stable=True)
    return order[:K]
