"""TC-SSA (Chen, Young, Xu, arXiv 2603.01143; github.com/OzzyChen97/TC-SSA, MIT) with K semantic slots.

Follows src/models/moe_compressor.py and wsi_classifier.py: a linear gate routes each patch to its top-2
slots, each slot is the routing-weighted mean of its patches refined by its own residual expert MLP, and the
slide is classified from the mean slot. Trained with slide labels as in tools/train.py (its CLI defaults).
"""

import copy
import math

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn

from ..metrics.retention import retained_from_assignment
from .base import SelectionResult


class Expert(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, drop: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, dim),
                                 nn.LayerNorm(dim))

    def forward(self, x):
        return x + self.net(x)


def aux_loss(logits: Tensor, probs: Tensor, topk_idx: Tensor, n_slots: int) -> Tensor:
    """Switch load-balancing + 0.5 * normalised-entropy + 1e-4 * z-loss, over (N, K) routing."""
    p = probs.mean(dim=0)
    f = torch.zeros_like(probs).scatter(-1, topk_idx, 1.0).mean(dim=0)
    switch = n_slots * torch.sum(p * f)
    entropy = -torch.sum(p * torch.log(p + 1e-8))
    entropy_loss = 1.0 - entropy / math.log(n_slots)
    z = torch.logsumexp(logits, dim=-1).mean() ** 2 * 1e-4
    return switch + 0.5 * entropy_loss + z


class TCSSA(nn.Module):
    """features (N, D) -> (slide logits (2,), aux loss, W (K, N) slot-assignment weights)."""

    def __init__(self, in_dim: int, n_slots: int, top_k: int = 2, n_classes: int = 2):
        super().__init__()
        self.n_slots, self.top_k = n_slots, min(top_k, n_slots)
        self.gate = nn.Linear(in_dim, n_slots)
        self.experts = nn.ModuleList([Expert(in_dim) for _ in range(n_slots)])
        self.classifier = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 512), nn.GELU(),
                                        nn.Dropout(0.3), nn.Linear(512, n_classes))

    def forward(self, x: Tensor):
        logits = self.gate(x)
        if self.training:
            logits = logits + 0.1 * torch.randn_like(logits)  # exploration noise, as in the official code
        probs = F.softmax(logits, dim=-1)
        topk_idx = probs.topk(self.top_k, dim=-1).indices
        loss = aux_loss(logits, probs, topk_idx, self.n_slots) if self.training else x.new_zeros(())
        routed = probs * torch.zeros_like(probs).scatter(-1, topk_idx, 1.0)  # (N, K), top-k entries only
        W = (routed / (routed.sum(dim=0, keepdim=True) + 1e-9)).T  # (K, N), each used slot sums to 1
        slots = torch.stack([expert(w @ x) for expert, w in zip(self.experts, W, strict=True)])
        return self.classifier(slots.mean(dim=0)), loss, W


def train_tcssa(build_model, train_bags, val_bags, seed: int, max_epochs: int = 50, aux_weight: float = 0.01,
                accum: int = 8, patience: int = 15, device="cpu"):
    """tools/train.py defaults: AdamW 1e-4, wd 1e-5, gradient accumulation over 8 slides, LR x0.5 on a val-AUC
    plateau (patience 5, min 1e-6), early stopping on val AUC; returns the best-val-AUC model."""
    torch.manual_seed(seed)
    model = build_model().to(device)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    ce = nn.CrossEntropyLoss()
    best_auc, best_state, bad, history = -1.0, copy.deepcopy(model.state_dict()), 0, []
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        order = torch.randperm(len(train_bags), generator=g).tolist()
        for step, i in enumerate(order, 1):
            x, y = train_bags[i]
            logits, aux, _ = model(x.to(device))
            ((ce(logits[None], torch.tensor([y], device=device)) + aux_weight * aux) / accum).backward()
            if step % accum == 0 or step == len(order):
                opt.step()
                opt.zero_grad()
        model.eval()
        with torch.no_grad():
            probs = [torch.softmax(model(x.to(device))[0], dim=0)[1].item() for x, _ in val_bags]
        auc = roc_auc_score([y for _, y in val_bags], probs)
        sched.step(auc)
        history.append({"epoch": epoch, "val_auc": auc})
        if auc > best_auc:
            best_auc, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


class TCSSASelector:
    """Merging method: returns the slot assignment W and, for hard-retention metrics, the patches kept under
    ``rule`` (default: each non-empty slot's largest contributor)."""

    def __init__(self, model: TCSSA, rule: str = "argmax", threshold: float | None = None):
        self.model, self.rule, self.threshold = model, rule, threshold

    @torch.no_grad()
    def select(self, features: Tensor, K: int, **ctx) -> SelectionResult:
        if K != self.model.n_slots:
            raise ValueError(f"model has {self.model.n_slots} slots, asked for K = {K}")
        self.model.eval()
        _, _, W = self.model(features)
        kept = retained_from_assignment(W.cpu().numpy(), self.rule, self.threshold)
        return SelectionResult(retained_idx=torch.as_tensor(kept, dtype=torch.long), W=W)
