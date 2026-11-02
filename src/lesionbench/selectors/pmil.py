"""Prototypical MIL (Yu et al., Med Image Anal 2023; github.com/zihou98/PMIL) on frozen patch features.

Three stages, as in the official code, with a linear instance head on frozen features in place of the
fine-tuned ResNet-34:
1. ``maxmil_pretrain``: max-MIL (Campanella et al. 2019) — each epoch, train on every slide's top-1 patch
   with the slide label (max-mil.py).
2. ``affinity_prototypes``: Affinity Propagation within each training slide (on its top-n patches), then
   across slides on the per-slide exemplars (cluster.py).
3. ``PMIL`` / ``train_pmil``: keep the top-s patches by instance tumor probability (s = K here), code them by
   normalised Euclidean distance to the prototypes, classify the mean code; loss = bag + max-instance
   (model_pmil.py, pmil.py).
"""

import copy
import logging
import warnings

import numpy as np
import torch
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances
from torch import Tensor, nn

from .base import SelectionResult, topk_indices

log = logging.getLogger(__name__)


def _recipe_optimizer(params, epochs_milestones=(10, 20)):
    opt = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-4)
    return opt, torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(epochs_milestones), gamma=0.1)


def maxmil_pretrain(bags, in_dim: int, seed: int, epochs: int = 30, batch_size: int = 128,
                    device="cpu") -> nn.Linear:
    """Instance head (D -> 2) trained by max-MIL on (features, slide label) bags."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    head = nn.Linear(in_dim, 2).to(device)
    opt, sched = _recipe_optimizer(head.parameters())
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        with torch.no_grad():
            top = [x[torch.softmax(head(x.to(device)), dim=1)[:, 1].argmax().cpu()] for x, _ in bags]
        X, y = torch.stack(top).to(device), torch.tensor([lab for _, lab in bags], device=device)
        for idx in torch.randperm(len(X), generator=g).to(device).split(batch_size):
            loss = loss_fn(head(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return head


def _ap_exemplars(feats: np.ndarray, lamb: float, preference: float, damping: float, seed: int) -> np.ndarray:
    """Indices of Affinity Propagation exemplars on exp(-lamb * d / max d) similarities (cluster.py).

    sklearn returns every point (or none) as exemplars when AP does not converge. We allow 2000 iterations,
    retry once with damping 0.9 (damping changes the iteration, not the fixed point), and raise otherwise.
    """
    if len(feats) == 1:
        return np.array([0])
    d = pairwise_distances(feats, metric="euclidean")
    sim = np.exp(-lamb * d / d.max()) if d.max() > 0 else np.ones_like(d)
    for damp in (damping, 0.9):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            ap = AffinityPropagation(preference=preference, damping=damp, max_iter=2000,
                                     affinity="precomputed", random_state=seed).fit(sim)
        if not any(issubclass(w.category, ConvergenceWarning) for w in caught):
            if damp != damping:
                log.warning("affinity propagation converged only with damping %.1f (n=%d)", damp, len(feats))
            return np.asarray(ap.cluster_centers_indices_)
    raise RuntimeError(f"affinity propagation did not converge on {len(feats)} points")


def affinity_prototypes(slide_feats: list[np.ndarray], lamb: float = 0.25, preference: float = 0.0,
                        damping: float = 0.5, seed: int = 24) -> np.ndarray:
    """Prototypes (P, D): per-slide AP exemplars, then AP across all of them."""
    local = np.concatenate([f[_ap_exemplars(f, lamb, preference, damping, seed)] for f in slide_feats])
    return local[_ap_exemplars(local, lamb, preference, damping, seed)].astype(np.float32)


class PMIL(nn.Module):
    """features (N, D) -> (instance logits (N, 2), bag logits (2,), code (P,), kept rows (min(s, N),))."""

    def __init__(self, head: nn.Linear, prototypes: Tensor, s: int, metric_dim: int = 128):
        super().__init__()
        if len(prototypes) < 2:
            # each distance row is divided by its max, so a single prototype makes every code exactly 1
            raise ValueError(f"PMIL needs at least 2 prototypes, got {len(prototypes)}")
        self.head = head
        self.register_buffer("prototypes", prototypes)
        self.s = s
        self.transformer = nn.Linear(prototypes.shape[1], metric_dim)
        self.classifier = nn.Sequential(nn.Linear(len(prototypes), 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x: Tensor):
        ins = self.head(x)
        kept = topk_indices(torch.softmax(ins, dim=1)[:, 1].detach(), self.s)
        dist = torch.cdist(self.transformer(x[kept]), self.transformer(self.prototypes))
        code = (dist / dist.max(dim=1, keepdim=True).values.clamp_min(1e-12)).mean(dim=0)
        return ins, self.classifier(code), code, kept


def _pmil_loss(model, x, y, loss_fn):
    ins, bag, _, _ = model(x)
    target = torch.tensor([y], device=bag.device)
    max_row = torch.softmax(ins, dim=1)[:, 1].argmax()
    return loss_fn(bag[None], target) + loss_fn(ins[max_row][None], target), bag


def train_pmil(build_model, train_bags, val_bags, seed: int, epochs: int = 30, device="cpu"):
    """Official recipe: Adam 1e-4, wd 1e-4, x0.1 at epochs 10 and 20; keep the best val-loss state."""
    torch.manual_seed(seed)
    model = build_model().to(device)
    g = torch.Generator().manual_seed(seed)
    opt, sched = _recipe_optimizer(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, history = float("inf"), copy.deepcopy(model.state_dict()), []
    for epoch in range(epochs):
        model.train()
        for i in torch.randperm(len(train_bags), generator=g).tolist():
            x, y = train_bags[i]
            loss, _ = _pmil_loss(model, x.to(device), y, loss_fn)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = sum(_pmil_loss(model, x.to(device), y, loss_fn)[0].item()
                      for x, y in val_bags) / max(len(val_bags), 1)
        history.append({"epoch": epoch, "val_loss": val})
        if val < best:
            best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history


class PMILSelector:
    """Top-K patches by the trained instance head's tumor probability (PMIL's selection with s = K)."""

    def __init__(self, head: nn.Linear):
        self.head = head

    @torch.no_grad()
    def select(self, features: Tensor, K: int, **ctx) -> SelectionResult:
        prob = torch.softmax(self.head(features), dim=1)[:, 1]
        return SelectionResult(retained_idx=topk_indices(prob, K), scores=prob)
