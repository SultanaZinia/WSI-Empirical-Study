"""LTS, Learnable Token Sparsification (Chen et al., arXiv 2606.08641). No public code: built from the paper.

Training keeps all N tokens: a Transformer scorer gives each patch a score, a soft top-K maps the z-scored
scores to keep weights alpha (sum K), tokens are noised as sqrt(alpha) x + sqrt(1 - alpha) eps, and a
per-token denoiser (one Transformer block with a diagonal attention mask) feeds the slide head. Inference
keeps the hard top-K and drops the denoiser. The paper's LLM text loss is replaced by our gated-ABMIL slide
head (docs/deviations.md).
"""

import math

import torch
from torch import Tensor, nn

from .abmil import GatedABMIL
from .base import SelectionResult, topk_indices


def soft_topk(scores: Tensor, K: int, tau: float, iters: int = 50) -> Tensor:
    """alpha_i = sigmoid((s_i - lam) / tau), with lam found by bisection so that sum(alpha) = K.

    lam is treated as a constant for the gradient. Returns all ones when K >= N.
    """
    if K >= len(scores):
        return torch.ones_like(scores)
    with torch.no_grad():
        lo, hi = scores.min() - 50 * tau, scores.max() + 50 * tau  # sum(alpha) ~ N at lo, ~ 0 at hi
        for _ in range(iters):
            lam = (lo + hi) / 2
            if torch.sigmoid((scores - lam) / tau).sum() > K:
                lo = lam
            else:
                hi = lam
    return torch.sigmoid((scores - (lo + hi) / 2) / tau)


def cosine_tau(epoch: int, tau_start: float = 1.0, tau_end: float = 0.1, anneal_epochs: int = 50) -> float:
    """Cosine anneal from tau_start to tau_end over ``anneal_epochs``, then hold at tau_end."""
    t = min(epoch / anneal_epochs, 1.0)
    return tau_end + 0.5 * (tau_start - tau_end) * (1 + math.cos(math.pi * t))


class LTS(nn.Module):
    """features (N, D) -> (slide logits (n_classes,), patch scores (N,)). Trained for one budget K."""

    def __init__(self, in_dim: int, K: int, dim: int = 512, n_layers: int = 2, n_heads: int = 8,
                 dropout: float = 0.25):
        super().__init__()
        self.K, self.tau = K, 1.0
        self.proj = nn.Linear(in_dim, dim)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, n_heads, 2 * dim, dropout, batch_first=True, norm_first=True),
            n_layers, enable_nested_tensor=False)
        self.score = nn.Linear(dim, 1)
        # Diagonal attention mask = each token attends only to itself: run tokens as length-1 sequences.
        self.denoiser = nn.TransformerEncoderLayer(in_dim, 1, dim, dropout, batch_first=True, norm_first=True)
        self.head = GatedABMIL(in_dim, dropout=dropout)

    def scores(self, x: Tensor) -> Tensor:
        s = self.score(self.blocks(self.proj(x)[None]))[0, :, 0]
        return (s - s.mean()) / (s.std(unbiased=False) + 1e-6)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        s = self.scores(x)
        if not self.training:
            return self.head(x[topk_indices(s, self.K)])[0], s
        # Clamp away from 0 and 1: a saturated alpha makes the sqrt gradient inf * 0 = NaN at low tau.
        alpha = soft_topk(s, self.K, self.tau).clamp(1e-6, 1 - 1e-6)[:, None]
        noisy = alpha.sqrt() * x + (1 - alpha).sqrt() * torch.randn_like(x)
        return self.head(self.denoiser(noisy[:, None])[:, 0])[0], s


class LTSSelector:
    """Keep the K patches with the highest LTS score (the paper's inference rule)."""

    def __init__(self, model: LTS):
        self.model = model

    @torch.no_grad()
    def select(self, features: Tensor, K: int, **ctx) -> SelectionResult:
        if K != self.model.K:
            raise ValueError(f"model was trained for K = {self.model.K}, asked for K = {K}")
        self.model.eval()
        s = self.model.scores(features)
        return SelectionResult(retained_idx=topk_indices(s, K), scores=s)
