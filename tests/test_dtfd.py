import numpy as np
import pytest
import torch

from lesionbench.selectors.dtfd import DTFD, DTFDSelector, train_dtfd


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_split_is_a_near_equal_partition():
    m = DTFD(8, n_groups=4)
    groups = m.split(30, _gen())
    assert sorted(len(g) for g in groups) == [7, 7, 8, 8]
    assert sorted(torch.cat(groups).tolist()) == list(range(30))
    assert [g.tolist() for g in groups] == [g.tolist() for g in m.split(30, _gen())]  # seeded


def test_per_side_requires_k_divisible_by_2m():
    m = DTFD(8, n_groups=4)
    assert m.per_side(32) == 4 and m.per_side(256) == 32
    with pytest.raises(ValueError, match="divisible"):
        m.per_side(12)


def test_distill_keeps_k_highest_and_k_lowest_per_pseudo_bag():
    groups = [torch.tensor([0, 1, 2, 3, 4]), torch.tensor([5, 6, 7, 8, 9])]
    probs = [torch.tensor([.5, .9, .1, .7, .3]), torch.tensor([.2, .2, .8, .6, .4])]
    kept = DTFD.distill(groups, probs, 1)
    assert kept.tolist() == [1, 2, 7, 6]  # tied minimum: later row
    small = DTFD.distill([torch.tensor([0, 1, 2])], [torch.tensor([.1, .2, .3])], 2)
    assert small.tolist() == [0, 1, 2]  # a pseudo-bag with <= 2k rows keeps every row once


def test_instance_probability_is_the_cam_of_the_weighted_feature():
    torch.manual_seed(0)
    m = DTFD(8, n_groups=1).eval()
    x = torch.randn(10, 8)
    mid = m.dim_reduction(x)
    groups = [torch.arange(10)]
    _, probs = m.tier1(mid, groups)
    a = torch.softmax(m.attention(mid), dim=0)
    logits = (mid * a[:, None]) @ m.classifier.weight.T  # get_cam_1d, no bias
    torch.testing.assert_close(probs[0], torch.softmax(logits, dim=1)[:, 1])


def test_forward_keeps_exactly_k_unique_rows_and_predict_kept_matches():
    torch.manual_seed(0)
    m = DTFD(16, n_groups=4).eval()
    x = torch.randn(200, 16)
    with torch.no_grad():
        _, bag, kept, scores = m(x, 32, _gen(3))
        assert len(kept) == 32 and len(set(kept.tolist())) == 32
        assert scores.shape == (200,) and ((scores >= 0) & (scores <= 1)).all()
        torch.testing.assert_close(m.predict_kept(x[kept]), bag)
    r = DTFDSelector(m).select(x, 32, seed=3)
    assert r.retained_idx.tolist() == kept.tolist()


def test_tier2_loss_does_not_reach_tier1():
    torch.manual_seed(0)
    m = DTFD(16, n_groups=2)
    _, bag, _, _ = m(torch.randn(40, 16), 4, _gen())
    bag[1].backward()
    assert all(p.grad is None for p in m.dim_reduction.parameters())
    assert all(p.grad is None for p in m.attention.parameters())
    assert m.t2_classifier.weight.grad is not None


def _bags(n, seed):
    """Half the bags positive: 4 of 40 instances shifted by +3 in every dimension."""
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x, y = g.normal(size=(40, 16)), i % 2
        if y:
            x[g.choice(40, 4, replace=False)] += 3.0
        out.append((torch.tensor(x, dtype=torch.float32), y))
    return out


def test_train_dtfd_learns_and_is_reproducible():
    train = _bags(40, 0)
    val = [(x, y, i) for i, (x, y) in enumerate(_bags(12, 1))]

    def run():
        return train_dtfd(lambda: DTFD(16, n_groups=2), train, val, K=4, seed=0, epochs=6)

    m1, h1 = run()
    m2, h2 = run()
    assert len(h1) == 6 and h1 == h2
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b)
    assert h1[-1]["val_auc"] > 0.8
