import math

import torch

from lesionbench.selectors.normal_bank import NormalBankSelector, cur_keys, dfi_keys, mean_topk_similarity

E1, E2, E3 = torch.eye(3)


def test_cur_keys_leverage_order():
    # Rows e1, e2, e1, e1 (rank 2). Leverage = diagonal of the projection onto the row space:
    # each e1 row 1/3, the e2 row 1. Keep the top min(rank, t) rows; ties go to the lower row.
    X = torch.stack([E1, E2, E1, E1])
    assert torch.equal(cur_keys(X, t=100), torch.stack([E2, E1]))  # rows 1, 0
    assert torch.equal(cur_keys(X, t=1), E2[None])


def test_dfi_keys_drops_near_duplicates():
    a = torch.tensor([1.0, 0.0, 0.0])
    a2 = torch.tensor([1.0, 0.1, 0.0])  # cos(a, a2) = 0.995
    X = torch.stack([a, a2, E2, E3])
    for seed in range(5):
        keys = dfi_keys(X, threshold=0.95, seed=seed)
        assert len(keys) == 3  # one of {a, a2}, plus e2 and e3
        assert any(torch.equal(k, E2) for k in keys) and any(torch.equal(k, E3) for k in keys)
    assert len(dfi_keys(X, threshold=0.999, seed=0)) == 4
    assert torch.equal(dfi_keys(X, seed=3), dfi_keys(X, seed=3))


def test_mean_topk_similarity_hand_computed():
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # cos to (1, 0): 1, 0, 0.7071
    q = torch.tensor([[2.0, 0.0]])
    r = 1 / math.sqrt(2)
    assert mean_topk_similarity(q, keys, k=1).item() == torch.tensor(1.0).item()
    assert abs(mean_topk_similarity(q, keys, k=2).item() - (1 + r) / 2) < 1e-6
    assert abs(mean_topk_similarity(q, keys, k=99).item() - (1 + r) / 3) < 1e-6  # k > #keys -> all keys


def test_mean_topk_similarity_chunking_is_exact():
    g = torch.Generator().manual_seed(0)
    X, keys = torch.randn(103, 16, generator=g), torch.randn(40, 16, generator=g)
    torch.testing.assert_close(mean_topk_similarity(X, keys, 5, chunk=7), mean_topk_similarity(X, keys, 5))


def test_normal_bank_selector_keeps_least_normal_first():
    keys = torch.stack([E1, E1 + 0.05 * E2])  # "normal" tissue points along e1
    feats = torch.stack([E1, E3, E1 * 2, E2 + E3, E2])  # rows 1, 3, 4 are orthogonal to e1-ish keys
    res = NormalBankSelector(keys, k=1).select(feats, 2)
    # novelty: e3 and e2+e3 -> sim 0 (ties, lower row first); e2 -> sim ~0.05; e1 rows -> ~1
    assert res.retained_idx.tolist() == [1, 3]
    assert NormalBankSelector(keys, k=1).select(feats, 3).retained_idx.tolist() == [1, 3, 4]
    assert sorted(NormalBankSelector(keys, k=1).select(feats, 10).retained_idx.tolist()) == [0, 1, 2, 3, 4]
