"""SLICEChat-style progressive in-encoder pruner (Bozkurt et al., arXiv 2609.24894; github.com/ali-kerem/
SLICEChat-encoder, Apache-2.0), scaled down and trained from scratch with slide labels (docs/deviations.md).

Follows models/mamba/vision_mamba.py and token_compressors/cropr.py: tokens run in raster order through
blocks of bidirectional Mamba layers and one Transformer layer; after every block but the last a pruner
scores tokens with one learned query and removes the contiguous Hilbert-curve window with the lowest total
score; the pruner's own attention-pooled output is trained by an auxiliary loss on detached tokens.
Per-stage counts end at exactly K.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .abmil import GatedABMIL
from .base import SelectionResult, budget_k
from .mamba import BiMamba

HILBERT_BITS = 10  # as in SLICEChat-encoder sorters/hilbert.py: grid coordinates must be < 1024


def hilbert_index(grid: Tensor, bits: int = HILBERT_BITS) -> Tensor:
    """2D Hilbert index of integer grid coords (N, 2) as (x, y).

    Port of SLICEChat's _hilbert_index_2d_batched.
    """
    if grid.numel() and int(grid.max()) >= 1 << bits:
        raise ValueError(f"grid coordinates must be < {1 << bits}")
    x, y = grid[:, 0].long().clone(), grid[:, 1].long().clone()
    n, d, s = 1 << bits, torch.zeros_like(x), (1 << bits) >> 1
    while s > 0:
        rx, ry = ((x & s) != 0).long(), ((y & s) != 0).long()
        d = d + s * s * ((3 * rx) ^ ry)
        flip = (ry == 0) & (rx == 1)
        x, y = torch.where(flip, n - 1 - x, x), torch.where(flip, n - 1 - y, y)
        swap = ry == 0
        x, y = torch.where(swap, y, x), torch.where(swap, x, y)
        s >>= 1
    return d


def raster_order(grid: Tensor) -> Tensor:
    """Permutation sorting tokens top-to-bottom, then left-to-right (sort by y, then x)."""
    return torch.argsort(grid[:, 1] * (int(grid[:, 0].max()) + 1) + grid[:, 0], stable=True)


def grid_from_coords(coords, step: int) -> Tensor:
    """Level-0 top-left coords (N, 2) -> integer patch grid (N, 2), origin at the minimum."""
    coords = torch.as_tensor(coords, dtype=torch.long)
    return (coords - coords.min(dim=0).values) // step


def keep_schedule(N: int, K: int, n_prune: int) -> list[int]:
    """Tokens left after each pruner: N_{l+1} = round(k N_l), k = (K/N)^(1/n_prune), the last forced to K."""
    if K >= N:
        return [N] * n_prune
    k, n, out = (K / N) ** (1 / n_prune), N, []
    for _ in range(n_prune - 1):
        n = max(round(n * k), K)
        out.append(n)
    return out + [K]


def remove_window(scores: Tensor, grid: Tensor, n_keep: int) -> Tensor:
    """Indices kept (in input order) after removing the contiguous Hilbert window of N - n_keep tokens with
    the lowest total score (ties: the earliest window)."""
    S = len(scores) - n_keep
    if S <= 0:
        return torch.arange(len(scores), device=scores.device)
    perm = torch.argsort(hilbert_index(grid), stable=True)
    cs = torch.cat([scores.new_zeros(1), torch.cumsum(scores[perm].float(), dim=0)])
    j = int(torch.argmin(cs[S:] - cs[:-S]))
    return torch.sort(torch.cat([perm[:j], perm[j + S:]])).values


class RegionPruner(nn.Module):
    """Cropr, SLICEChat's base config: 1 query, 1 head, key and value projections, MLP on pooled output."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.query = nn.Parameter(torch.empty(dim))
        nn.init.trunc_normal_(self.query, std=dim**-0.5)
        self.k, self.v = nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False)
        self.mlp_norm = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.scale = dim**-0.5

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x (N, dim) -> (pre-softmax scores (N,), attention-pooled output (dim,))."""
        scores = self.k(x) @ self.query * self.scale
        out = torch.softmax(scores.float(), dim=0).to(x.dtype) @ self.v(x)
        return scores, out + self.mlp(self.mlp_norm(out))


class SLICEChatMIL(nn.Module):
    """(features (N, D), grid (N, 2)) -> (slide logits (2,), layer_of_drop (N,)). Trained for one budget K.

    layer_of_drop[i] = index of the pruner (0 .. n_blocks-2) that removed patch i, or -1 if it reached the
    end. In training mode ``aux_logits`` holds each pruner's auxiliary logits for the last forward pass.
    """

    def __init__(self, in_dim: int, K: int | str, dim: int = 256, n_blocks: int = 6, block: str = "MT",
                 n_heads: int = 8, d_state: int = 16, expand: int = 2, dropout: float = 0.25):
        super().__init__()
        self.K, self.block = K, block
        self.proj = nn.Linear(in_dim, dim)
        self.mamba_norms, self.mambas, self.transformers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for op in block * n_blocks:
            if op == "M":
                self.mamba_norms.append(nn.RMSNorm(dim, eps=1e-5))
                self.mambas.append(BiMamba(dim, d_state=d_state, expand=expand))
            else:
                self.transformers.append(nn.TransformerEncoderLayer(dim, n_heads, 4 * dim, 0.0, "gelu",
                                                                    layer_norm_eps=1e-6, batch_first=True,
                                                                    norm_first=True))
        self.pruners = nn.ModuleList([RegionPruner(dim) for _ in range(n_blocks - 1)])
        self.norm_f = nn.RMSNorm(dim, eps=1e-5)
        self.aux_head = nn.Linear(dim, 2)  # shared by all pruners, like SLICEChat's shared CLIP space
        self.head = GatedABMIL(dim, dropout=dropout)
        self.aux_logits: list[Tensor] = []

    def encode(self, features: Tensor, grid: Tensor, use_kernel=None):
        """Returns (final tokens (K', dim), their original rows (K',), layer_of_drop (N,), stage_scores
        (n_pruners, N): pruner b's pre-softmax score per patch, NaN for patches removed before pruner b)."""
        order = raster_order(grid)
        rows, grid = order, grid[order]
        h = self.proj(features[order])
        drop = torch.full((len(features),), -1, dtype=torch.long, device=features.device)
        stage_scores = torch.full((len(self.pruners), len(features)), float("nan"), device=features.device)
        schedule = keep_schedule(len(features), budget_k(self.K, len(features)), len(self.pruners))
        self.aux_logits = []
        m = t = 0
        for b in range(len(self.pruners) + 1):
            for op in self.block:
                if op == "M":
                    h = h + self.mambas[m](self.mamba_norms[m](h)[None], use_kernel)[0]
                    m += 1
                else:
                    h = self.transformers[t](h[None])[0]
                    t += 1
            if b == len(self.pruners):
                break
            scores, pooled = self.pruners[b](h.detach() if self.training else h)
            stage_scores[b, rows] = scores.detach().float()
            if self.training:
                self.aux_logits.append(self.aux_head(pooled))
            keep = remove_window(scores.detach(), grid, schedule[b])
            dropped = torch.ones(len(rows), dtype=torch.bool, device=rows.device)
            dropped[keep] = False
            drop[rows[dropped]] = b
            h, rows, grid = h[keep], rows[keep], grid[keep]
            order = raster_order(grid)  # retained tokens continue in raster order
            h, rows, grid = h[order], rows[order], grid[order]
        return self.norm_f(h), rows, drop, stage_scores

    def forward(self, inp: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        features, grid = inp
        h, _, drop, _ = self.encode(features, grid)
        return self.head(h)[0], drop


def aux_loss(model: SLICEChatMIL, y: int) -> Tensor:
    """Sum of the pruners' auxiliary cross-entropies for the last training forward (weight 1, as in code)."""
    device = model.aux_head.weight.device
    target = torch.tensor([y], device=device)
    return sum((F.cross_entropy(lg[None], target) for lg in model.aux_logits), torch.zeros((), device=device))


class SLICEChatSelector:
    """Retained = the K patches that reach the end of the encoder; also returns each patch's layer of drop."""

    def __init__(self, model: SLICEChatMIL):
        self.model = model

    @torch.no_grad()
    def select(self, features: Tensor, K: int, *, grid: Tensor, **ctx) -> SelectionResult:
        if K != self.model.K:
            raise ValueError(f"model was trained for K = {self.model.K}, asked for K = {K}")
        self.model.eval()
        _, rows, drop, stage_scores = self.model.encode(features, grid)
        return SelectionResult(retained_idx=rows, layer_of_drop=drop, stage_scores=stage_scores)
