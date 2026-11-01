import math

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from lesionbench.selectors.tcssa import TCSSA, TCSSASelector, aux_loss, train_tcssa


def test_aux_loss_uniform_routing_hand_computed():
    # Uniform probabilities over K slots, each patch routed to 2 slots:
    # switch = K * sum_i (1/K) f_i = sum_i f_i = 2; entropy term 0; z = (log K)^2 * 1e-4
    K, N = 4, 6
    logits = torch.zeros(N, K)
    probs = torch.softmax(logits, dim=-1)
    idx = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [1, 2]])
    assert aux_loss(logits, probs, idx, K).item() == pytest.approx(2 + math.log(K) ** 2 * 1e-4, abs=1e-6)


def test_forward_shapes_and_assignment_structure():
    torch.manual_seed(0)
    m = TCSSA(in_dim=8, n_slots=4).eval()
    x = torch.randn(10, 8)
    logits, aux, W = m(x)
    assert logits.shape == (2,) and aux.item() == 0.0 and W.shape == (4, 10)
    assert ((W > 0).sum(dim=0) == 2).all()  # every patch feeds exactly its top-2 slots
    used = W.sum(dim=1) > 0
    torch.testing.assert_close(W.sum(dim=1)[used], torch.ones(int(used.sum())))
    torch.testing.assert_close(m(x)[2], W)  # eval mode: no routing noise


def test_train_mode_adds_routing_noise_and_aux_loss():
    torch.manual_seed(0)
    m = TCSSA(in_dim=8, n_slots=4).train()
    x = torch.randn(10, 8)
    _, aux1, W1 = m(x)
    _, _, W2 = m(x)
    assert aux1.item() > 0 and not torch.equal(W1, W2)


def test_selector_k_must_match_slots_and_kept_rows_are_valid():
    torch.manual_seed(0)
    m = TCSSA(in_dim=8, n_slots=4)
    x = torch.randn(10, 8)
    res = TCSSASelector(m).select(x, 4)
    kept = res.retained_idx.tolist()
    assert 1 <= len(kept) <= 4 and all(0 <= i < 10 for i in kept) and res.W.shape == (4, 10)
    with pytest.raises(ValueError, match="4 slots"):
        TCSSASelector(m).select(x, 8)


def _bags(n, seed):
    g = np.random.default_rng(seed)
    bags = []
    for i in range(n):
        x, y = g.normal(size=(30, 16)), i % 2
        if y:
            x[g.choice(30, 3, replace=False)] += 3.0
        bags.append((torch.tensor(x, dtype=torch.float32), y))
    return bags


def test_train_tcssa_learns_and_is_reproducible():
    train, val = _bags(48, seed=0), _bags(20, seed=1)

    def build():
        return TCSSA(in_dim=16, n_slots=4)

    model, hist = train_tcssa(build, train, val, seed=0, max_epochs=15)
    model.eval()
    with torch.no_grad():
        probs = [torch.softmax(model(x)[0], dim=0)[1].item() for x, _ in val]
    assert roc_auc_score([y for _, y in val], probs) > 0.9
    assert hist == train_tcssa(build, train, val, seed=0, max_epochs=15)[1]
