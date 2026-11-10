"""DTFD-MIL (Zhang et al., CVPR 2022; github.com/hrzhang1123/DTFD-MIL @ 10964f4), MaxMinS, on frozen features.

As in ``Main_DTFD_MIL.py`` / ``Model/``:
1. A slide's patches are split at random into M pseudo-bags; each inherits the slide label.
2. Tier 1: dimension reduction (no bias, ReLU), gated attention within each pseudo-bag, linear classifier on
   the attention-pooled feature. Instance probability = softmax of the classifier weights applied to each
   attention-weighted patch feature (``get_cam_1d``; paper Eq. 8-9).
3. MaxMinS distillation: per pseudo-bag keep the highest- and the lowest-probability patches.
4. Tier 2: gated-attention classifier on the kept patches' reduced features -> slide prediction.

Here the number kept per pseudo-bag is K / (2M) on each side (the code's ``total_instance`` knob), so exactly
K patches reach Tier 2. Changes are listed in docs/deviations.md (2026-11-10).
"""

import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn

from .base import SelectionResult


class GatedAttention(nn.Module):
    """Attention_Gated (L=512, D=128, K=1): unnormalised score per row, (N, L) -> (N,)."""

    def __init__(self, L: int = 512, D: int = 128):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.w = nn.Linear(D, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.w(self.V(x) * self.U(x)).squeeze(-1)


class DTFD(nn.Module):
    def __init__(self, in_dim: int, n_groups: int = 4, m_dim: int = 512):
        super().__init__()
        self.n_groups = n_groups
        self.dim_reduction = nn.Sequential(nn.Linear(in_dim, m_dim, bias=False), nn.ReLU())
        self.attention = GatedAttention(m_dim)
        self.classifier = nn.Linear(m_dim, 2)
        self.t2_attention = GatedAttention(m_dim)
        self.t2_classifier = nn.Linear(m_dim, 2)

    def per_side(self, K: int) -> int:
        if K % (2 * self.n_groups):
            raise ValueError(f"DTFD MaxMinS needs K divisible by 2 * M = {2 * self.n_groups}, got {K}")
        return K // (2 * self.n_groups)

    def split(self, n: int, generator: torch.Generator) -> list[Tensor]:
        """Random pseudo-bags of near-equal size (np.array_split of a shuffled index list)."""
        perm = torch.randperm(n, generator=generator)
        return [g for g in torch.tensor_split(perm, self.n_groups) if len(g)]

    def tier1(self, mid: Tensor, groups: list[Tensor]):
        """Pseudo-bag logits (M, 2) and each pseudo-bag's instance probabilities."""
        a = self.attention(mid)
        logits, probs = [], []
        for idx in groups:
            weighted = mid[idx] * torch.softmax(a[idx], dim=0)[:, None]
            logits.append(self.classifier(weighted.sum(dim=0)))
            cam = weighted @ self.classifier.weight.T  # get_cam_1d: classifier weights, no bias
            probs.append(torch.softmax(cam, dim=1)[:, 1])
        return torch.stack(logits), probs

    @staticmethod
    def distill(groups: list[Tensor], probs: list[Tensor], k: int) -> Tensor:
        """MaxMinS: the k highest- and k lowest-probability rows of each pseudo-bag (all if it has <= 2k)."""
        kept = []
        for idx, p in zip(groups, probs, strict=True):
            if len(idx) <= 2 * k:
                kept.append(idx)
                continue
            order = torch.argsort(-p.detach(), stable=True)
            kept.append(idx[torch.cat([order[:k], order[-k:]])])
        return torch.cat(kept)

    def tier2(self, mid_kept: Tensor) -> Tensor:
        w = torch.softmax(self.t2_attention(mid_kept), dim=0)
        return self.t2_classifier((mid_kept * w[:, None]).sum(dim=0))

    def forward(self, x: Tensor, K: int, generator: torch.Generator):
        """-> (pseudo-bag logits (M, 2), slide logits (2,), kept rows (<= K,), instance probs (N,))."""
        k = self.per_side(K)
        mid = self.dim_reduction(x)
        groups = [g.to(x.device) for g in self.split(len(x), generator)]
        g_logits, probs = self.tier1(mid, groups)
        kept = self.distill(groups, probs, k)
        bag = self.tier2(mid[kept].detach())  # Tier-2 loss does not update Tier 1 (as in the official loop)
        scores = torch.empty(len(x), device=x.device)
        scores[torch.cat(groups)] = torch.cat(probs).detach()
        return g_logits, bag, kept, scores

    def predict_kept(self, x_kept: Tensor) -> Tensor:
        """Slide logits from the kept patches only."""
        return self.tier2(self.dim_reduction(x_kept))


def _val_auc(model, val_bags, K, device) -> float:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y, split_seed in val_bags:
            _, bag, _, _ = model(x.to(device), K, torch.Generator().manual_seed(split_seed))
            ys.append(y)
            ps.append(torch.softmax(bag, dim=-1)[1].item())
    return float(roc_auc_score(ys, ps)) if len(set(ys)) == 2 else 0.5


def train_dtfd(build_model, train_bags, val_bags, K: int, seed: int, epochs: int = 200, device="cpu"):
    """Official recipe: Tier 1 and Tier 2 each with Adam lr 1e-4, wd 1e-4, x0.2 at epoch 100, gradient
    clipping 5; keep the state with the best Tier-2 validation AUC after epoch int(0.8 * epochs) (the last
    state if none). ``val_bags``: (features, label, split_seed)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = build_model().to(device)
    tier1 = [model.dim_reduction, model.attention, model.classifier]
    tier2 = [model.t2_attention, model.t2_classifier]
    opt0 = torch.optim.Adam([p for m in tier1 for p in m.parameters()], lr=1e-4, weight_decay=1e-4)
    opt1 = torch.optim.Adam([p for m in tier2 for p in m.parameters()], lr=1e-4, weight_decay=1e-4)
    sched = [torch.optim.lr_scheduler.MultiStepLR(o, milestones=[100], gamma=0.2) for o in (opt0, opt1)]
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, history = -np.inf, None, []
    for epoch in range(epochs):
        model.train()
        for i in torch.randperm(len(train_bags), generator=g).tolist():
            x, y = train_bags[i]
            g_logits, bag, _, _ = model(x.to(device), K, g)
            loss0 = loss_fn(g_logits, torch.full((len(g_logits),), y, device=g_logits.device))
            opt0.zero_grad()
            loss0.backward()
            for m in tier1:
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt0.step()
            loss1 = loss_fn(bag[None], torch.tensor([y], device=bag.device))
            opt1.zero_grad()
            loss1.backward()
            nn.utils.clip_grad_norm_([p for m in tier2 for p in m.parameters()], 5.0)
            opt1.step()
        for s in sched:
            s.step()
        auc = _val_auc(model, val_bags, K, device)
        history.append({"epoch": epoch, "val_auc": auc})
        if epoch > int(0.8 * epochs) and auc > best:
            best, best_state = auc, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), history


class DTFDSelector:
    """MaxMinS kept rows of one seeded pseudo-bag split; scores = Tier-1 instance probabilities."""

    def __init__(self, model: DTFD):
        self.model = model

    @torch.no_grad()
    def select(self, features: Tensor, K: int, seed: int = 0, **ctx) -> SelectionResult:
        _, _, kept, scores = self.model(features, K, torch.Generator().manual_seed(seed))
        return SelectionResult(retained_idx=kept, scores=scores)
