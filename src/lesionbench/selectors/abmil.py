"""Gated-attention ABMIL (Ilse et al., ICML 2018), CLAM "small" sizes, and its top-K attention selector."""

import torch
from torch import Tensor, nn

from .base import SelectionResult, topk_indices


class GatedABMIL(nn.Module):
    """features (N, L) -> (slide logits (n_classes,), raw attention scores (N,))."""

    def __init__(self, in_dim: int, hidden: int = 512, attn_dim: int = 256, dropout: float = 0.25,
                 n_classes: int = 2):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attn_a = nn.Sequential(nn.Linear(hidden, attn_dim), nn.Tanh(), nn.Dropout(dropout))
        self.attn_b = nn.Sequential(nn.Linear(hidden, attn_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.attn_c = nn.Linear(attn_dim, 1)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.fc(x)
        scores = self.attn_c(self.attn_a(h) * self.attn_b(h)).squeeze(-1)
        pooled = torch.softmax(scores, dim=0) @ h
        return self.classifier(pooled), scores


class ABMILTopK:
    """Top-K ABMIL: keep the K patches with the highest attention score of a trained model."""

    def __init__(self, model: GatedABMIL):
        self.model = model

    @torch.no_grad()
    def select(self, features: Tensor, K: int, **ctx) -> SelectionResult:
        self.model.eval()
        _, scores = self.model(features)
        return SelectionResult(retained_idx=topk_indices(scores, K), scores=scores)
