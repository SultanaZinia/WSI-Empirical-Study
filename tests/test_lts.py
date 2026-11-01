import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from lesionbench.selectors.base import topk_indices
from lesionbench.selectors.lts import LTS, LTSSelector, cosine_tau, soft_topk
from lesionbench.train import evaluate, train_mil


def test_soft_topk_sums_to_k_and_orders_like_scores():
    s = torch.tensor([2.0, -1.0, 0.5, 3.0, 0.0, -2.0])
    a = soft_topk(s, 2, tau=0.5)
    assert a.sum().item() == pytest.approx(2.0, abs=1e-4)
    assert ((a >= 0) & (a <= 1)).all()
    assert torch.equal(torch.argsort(a, descending=True), torch.argsort(s, descending=True))


def test_soft_topk_hand_computed_symmetric_case():
    # Scores symmetric around 0 with K = N/2 -> lam = 0, so alpha = sigmoid(s / tau).
    s = torch.tensor([-1.0, 1.0])
    torch.testing.assert_close(soft_topk(s, 1, tau=1.0), torch.sigmoid(s), atol=1e-5, rtol=0)


def test_soft_topk_low_tau_approaches_hard_topk_and_k_ge_n_keeps_all():
    s = torch.tensor([0.3, 1.2, -0.4, 0.9, 0.0])
    hard = torch.zeros(5).index_fill(0, topk_indices(s, 2), 1.0)
    torch.testing.assert_close(soft_topk(s, 2, tau=1e-3), hard, atol=1e-3, rtol=0)
    torch.testing.assert_close(soft_topk(s, 5, tau=0.5), torch.ones(5))
    torch.testing.assert_close(soft_topk(s, 9, tau=0.5), torch.ones(5))


def test_soft_topk_passes_gradient_to_scores():
    s = torch.tensor([0.5, -0.2, 1.0, 0.1], requires_grad=True)
    (soft_topk(s, 2, tau=0.5) * torch.tensor([1.0, 0.0, 0.0, 0.0])).sum().backward()
    assert s.grad[0] > 0


def test_cosine_tau_schedule():
    assert cosine_tau(0) == pytest.approx(1.0)
    assert cosine_tau(25) == pytest.approx(0.55)
    assert cosine_tau(50) == pytest.approx(0.1)
    assert cosine_tau(120) == pytest.approx(0.1)


def test_forward_eval_uses_only_top_k_and_train_is_noisy():
    torch.manual_seed(0)
    m = LTS(in_dim=16, K=3, dim=32, n_heads=4)
    x = torch.randn(20, 16)
    m.eval()
    logits, s = m(x)
    assert logits.shape == (2,) and s.shape == (20,)
    assert s.mean().abs() < 1e-5 and s.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-4)
    torch.testing.assert_close(logits, m.head(x[topk_indices(s, 3)])[0])
    torch.testing.assert_close(m(x)[0], logits)  # eval is deterministic
    m.train()
    assert not torch.equal(m(x)[0], m(x)[0])


def test_selector_k_must_match_and_k_gt_n():
    torch.manual_seed(0)
    m = LTS(in_dim=8, K=4, dim=16, n_heads=2)
    x = torch.randn(10, 8)
    res = LTSSelector(m).select(x, 4)
    assert torch.equal(res.retained_idx, topk_indices(res.scores, 4))
    with pytest.raises(ValueError, match="K = 4"):
        LTSSelector(m).select(x, 8)
    assert len(LTSSelector(m).select(torch.randn(3, 8), 4).retained_idx) == 3


def test_train_step_gradients_finite_when_alpha_saturates():
    torch.manual_seed(0)
    m = LTS(in_dim=8, K=2, dim=16, n_heads=2).train()
    m.tau = 1e-3  # alpha is exactly 0 or 1 in float32
    m(torch.randn(10, 8))[0].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def _bags(n, seed):
    """Half the bags are positive: 3 of their 30 instances are shifted by +3."""
    g = np.random.default_rng(seed)
    bags, tumor_rows = [], []
    for i in range(n):
        x, y = g.normal(size=(30, 16)), i % 2
        rows = g.choice(30, 3, replace=False) if y else np.array([], int)
        x[rows] += 3.0
        bags.append((torch.tensor(x, dtype=torch.float32), y))
        tumor_rows.append(set(rows.tolist()))
    return bags, tumor_rows


def test_train_lts_learns_and_selects_tumor_and_is_reproducible():
    train, _ = _bags(40, seed=0)
    val, val_tumor = _bags(20, seed=1)
    tcfg = {"lr": 1e-3, "weight_decay": 1e-5, "max_epochs": 20, "patience": 20, "stop_epoch": 20}

    def build():
        return LTS(in_dim=16, K=3, dim=32, n_heads=4)

    def anneal(model, epoch):
        model.tau = cosine_tau(epoch, anneal_epochs=10)

    model, hist = train_mil(build, train, val, tcfg, seed=0, on_epoch=anneal)
    _, probs = evaluate(model, val)
    assert roc_auc_score([y for _, y in val], probs) > 0.9
    sel = LTSSelector(model)
    precision = [len(set(sel.select(x, 3).retained_idx.tolist()) & t) / 3
                 for (x, y), t in zip(val, val_tumor, strict=True) if y]
    assert np.mean(precision) > 0.8
    assert hist == train_mil(build, train, val, tcfg, seed=0, on_epoch=anneal)[1]
