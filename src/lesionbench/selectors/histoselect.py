"""HistoSelect (Huang et al., CVPR 2026; github.com/winston52/HistoSelect @ e4bb3c1) on frozen patch features.

Follows ``LLaVAModel_Selector.forward`` (``xtuner/model/llava_selector.py``) with the selector parts
unchanged: projector -> per-group prototypes (mean projected feature of each CONCH tissue group) ->
``SimpleGroupSelector`` (group keep probability g_i from [prototype, projected question]) ->
``PatchSelector`` (patch probability p_j from [patch, projected question]) -> top-K_i patches of each
group -> kept features weighted by the straight-through value g_i * p_j. Training adds the two
information-bottleneck terms: BCE of the group / patch logits toward sigmoid(cos(question, prototype or
patch) * exp(4.0315) + bias) priors, weighted by beta (0 -> 0.2) and gamma (0 -> 0.1), annealed linearly over
5,000 training steps.

Changes (docs/deviations.md, 2026-11-10): one fixed question; exactly K kept, split over groups by
``allocate``; no LongNet before selection; the LLM is replaced by a gated-ABMIL head.
Model input: (features (N, D), groups (N,) long in [0, n_groups), question embedding (Q,)).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .abmil import GatedABMIL
from .base import SelectionResult

CONCH_LOGIT_SCALE = 4.0315  # fixed logit_scale for the CONCH text encoder, as in the official code


def allocate(K: int, weights: Tensor, counts: Tensor) -> Tensor:
    """Patches kept per group: exactly min(K, sum(counts)), at most counts_i each.

    Non-empty groups first get one patch each (the paper's ceil rule keeps >= 1 per group); when K is smaller
    than the number of non-empty groups, the K groups with the largest weight get one. The rest is split in
    proportion to ``weights`` (g_i * n_i) by largest remainder, repeated until K is used or groups are full.
    Ties go to the lower group index.
    """
    counts = counts.long()
    K = min(K, int(counts.sum()))
    alloc = torch.zeros_like(counts)
    active = (counts > 0).nonzero().flatten()
    w = weights.detach().double().clamp_min(0)
    if K < len(active):
        order = active[torch.argsort(-w[active], stable=True)]
        alloc[order[:K]] = 1
        return alloc
    alloc[active] = 1
    left = K - len(active)
    while left > 0:
        room = counts - alloc
        open_ = room > 0
        ww = torch.where(open_, w, torch.zeros_like(w))
        if ww.sum() <= 0:  # all remaining weight is zero: share by remaining room instead
            ww = room.double() * open_
        share = left * ww / ww.sum()
        add = torch.minimum(share.floor().long(), room)
        if add.sum() == 0:  # every share < 1: one each by largest remainder
            frac = torch.where(open_, share - share.floor(), torch.full_like(share, -1.0))
            order = torch.argsort(-frac, stable=True)[:left]
            add = torch.zeros_like(alloc)
            add[order[open_[order]]] = 1
        alloc += add
        left = K - int(alloc.sum())
    return alloc


class HistoSelect(nn.Module):
    """(features, groups, question) -> (slide logits (2,), per-patch selection probability g_i * p_j (N,))."""

    def __init__(self, in_dim: int, q_dim: int, K: int, n_groups: int = 13, hidden: int = 512,
                 embed_dim: int = 1024, dropout: float = 0.25, group_prior_bias: float = -1.25,
                 patch_prior_bias: float = 0.0, beta_final: float = 0.2, gamma_final: float = 0.1,
                 anneal_steps: int = 5000):
        super().__init__()
        self.K, self.n_groups = K, n_groups
        self.projector = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.question_prior_proj = nn.Linear(q_dim, hidden)
        self.group_selector = nn.Sequential(nn.Linear(2 * hidden, embed_dim), nn.LayerNorm(embed_dim),
                                            nn.GELU(), nn.Linear(embed_dim, 1))
        self.patch_question_proj = nn.Linear(hidden, hidden)  # PatchSelector.question_proj
        self.patch_selector = nn.Sequential(nn.Linear(2 * hidden, embed_dim), nn.LayerNorm(embed_dim),
                                            nn.GELU(), nn.Linear(embed_dim, 1))
        self.head = GatedABMIL(hidden, dropout=dropout)
        self.group_prior_bias, self.patch_prior_bias = group_prior_bias, patch_prior_bias
        self.beta_final, self.gamma_final, self.anneal_steps = beta_final, gamma_final, anneal_steps
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self.ib_loss: Tensor | None = None  # weighted IB terms of the last training forward

    def weights(self) -> tuple[float, float]:
        t = min(self.step.item() / self.anneal_steps, 1.0) if self.anneal_steps > 0 else 1.0
        return self.beta_final * t, self.gamma_final * t

    def select(self, x: Tensor, groups: Tensor, question: Tensor):
        """-> (projected features (N, h), kept rows (<= K,), g_i * p_j per patch (N,), IB terms (2,))."""
        h = self.projector(x)
        q = self.question_prior_proj(question.to(h.dtype))
        counts = torch.bincount(groups, minlength=self.n_groups)
        proto = torch.zeros(self.n_groups, h.shape[1], device=h.device).index_add_(0, groups, h)
        proto = proto / counts.clamp_min(1)[:, None]
        g_logit = self.group_selector(torch.cat([proto, q.expand(self.n_groups, -1)], dim=1)).squeeze(-1)
        g = torch.sigmoid(g_logit)
        qp = self.patch_question_proj(q)
        p_logit = self.patch_selector(torch.cat([h, qp.expand(len(h), -1)], dim=1)).squeeze(-1)
        p = torch.sigmoid(p_logit)
        alloc = allocate(self.K, g * counts, counts)
        kept = []
        for i in (counts > 0).nonzero().flatten().tolist():
            rows = (groups == i).nonzero().flatten()
            if alloc[i] > 0:
                kept.append(rows[torch.topk(p[rows].detach(), int(alloc[i])).indices])
        kept = torch.cat(kept)
        active = counts > 0
        qn = F.normalize(q, dim=-1)
        scale = math.exp(CONCH_LOGIT_SCALE)
        g_prior = torch.sigmoid(F.normalize(proto, dim=-1) @ qn * scale + self.group_prior_bias).detach()
        g_ib = (F.binary_cross_entropy_with_logits(g_logit, g_prior, reduction="none") * active).sum() \
            / active.sum()
        p_prior = torch.sigmoid(F.normalize(h, dim=-1) @ qn * scale + self.patch_prior_bias).detach()
        p_bce = F.binary_cross_entropy_with_logits(p_logit, p_prior, reduction="none")
        p_ib = torch.stack([p_bce[groups == i].mean() for i in active.nonzero().flatten().tolist()]).mean()
        return h, kept, (g[groups] * p), torch.stack([g_ib, p_ib])

    def forward(self, inputs):
        x, groups, question = inputs
        h, kept, prob, ib = self.select(x, groups, question)
        ste = 1.0 + prob[kept] - prob[kept].detach()  # hard selection forward, soft gradient (STE)
        logits = self.head(h[kept] * ste[:, None])[0]
        if self.training:
            beta, gamma = self.weights()
            self.ib_loss = beta * ib[0] + gamma * ib[1]
            self.step += 1
        return logits, prob.detach()


def ib_loss(model: HistoSelect, y: int) -> Tensor:
    """The weighted information-bottleneck terms of the last training forward (for train_mil's extra_loss)."""
    return model.ib_loss


class HistoSelectSelector:
    """Kept rows = the top-K_i patches of each tissue group; scores = g_i * p_j."""

    def __init__(self, model: HistoSelect):
        self.model = model

    @torch.no_grad()
    def select(self, features: Tensor, K: int, groups: Tensor = None, question: Tensor = None,
               **ctx) -> SelectionResult:
        if K != self.model.K:
            raise ValueError(f"model was trained for K = {self.model.K}, asked for K = {K}")
        self.model.eval()
        _, kept, prob, _ = self.model.select(features, groups, question)
        return SelectionResult(retained_idx=kept, scores=prob)
