import numpy as np
import pytest
import torch

from lesionbench.selectors.histoselect import HistoSelect, HistoSelectSelector, allocate, ib_loss
from lesionbench.train import train_mil


def test_allocate_one_per_group_then_proportional():
    a = allocate(10, torch.tensor([6.0, 2.0, 0.0, 2.0]), torch.tensor([20, 20, 0, 20]))
    assert a.tolist() == [6, 2, 0, 2]  # 1 each, then 7 left split 6:2:2 -> 4.2, 1.4, 1.4 -> remainders
    assert int(a.sum()) == 10


def test_allocate_caps_at_group_size_and_refills():
    a = allocate(8, torch.tensor([100.0, 1.0]), torch.tensor([3, 10]))
    assert a.tolist() == [3, 5] and int(a.sum()) == 8


def test_allocate_fewer_slots_than_groups_and_k_gt_n():
    assert allocate(2, torch.tensor([1.0, 5.0, 3.0]), torch.tensor([4, 4, 4])).tolist() == [0, 1, 1]
    assert allocate(50, torch.tensor([1.0, 1.0]), torch.tensor([3, 4])).tolist() == [3, 4]


def test_allocate_zero_weights_shares_by_room():
    a = allocate(6, torch.zeros(2), torch.tensor([2, 10]))
    assert int(a.sum()) == 6 and (a <= torch.tensor([2, 10])).all()


def _bag(seed, tumour):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(40, 8, generator=g)
    groups = (torch.arange(40) >= 20).long()
    if tumour:
        x[:3] += 3.0
    return (x, groups, torch.ones(5)), int(tumour)


def test_forward_keeps_exactly_k_per_group_split_and_ste_gradients():
    torch.manual_seed(0)
    m = HistoSelect(8, 5, K=6, n_groups=3)
    (x, groups, q), _ = _bag(0, True)
    h, kept, prob, ib = m.select(x, groups, q)
    assert len(kept) == 6 and len(set(kept.tolist())) == 6
    assert (groups[kept] == 0).sum() >= 1 and (groups[kept] == 1).sum() >= 1  # >= 1 per non-empty group
    assert prob.shape == (40,) and ib.shape == (2,)
    m.train()
    logits, _ = m((x, groups, q))
    (logits[1] + ib_loss(m, 1)).backward()
    assert m.patch_selector[0].weight.grad is not None and m.group_selector[0].weight.grad is not None
    assert m.step.item() == 1


def test_ib_weights_anneal_linearly():
    m = HistoSelect(8, 5, K=4, anneal_steps=10)
    assert m.weights() == (0.0, 0.0)
    m.step.fill_(5)
    assert m.weights() == pytest.approx((0.1, 0.05))
    m.step.fill_(50)
    assert m.weights() == pytest.approx((0.2, 0.1))


def test_selector_matches_eval_forward_and_checks_k():
    torch.manual_seed(0)
    m = HistoSelect(8, 5, K=4).eval()
    (x, groups, q), _ = _bag(1, True)
    r = HistoSelectSelector(m).select(x, 4, groups=groups, question=q)
    _, kept, _, _ = m.select(x, groups, q)
    assert r.retained_idx.tolist() == kept.tolist()
    with pytest.raises(ValueError, match="trained for K"):
        HistoSelectSelector(m).select(x, 8, groups=groups, question=q)


def test_train_learns_and_is_reproducible():
    train = [_bag(i, i % 2) for i in range(30)]
    val = [_bag(100 + i, i % 2) for i in range(10)]
    tc = {"lr": 1e-3, "weight_decay": 1e-5, "max_epochs": 8, "patience": 8, "stop_epoch": 8}

    def run():
        return train_mil(lambda: HistoSelect(8, 5, K=4, n_groups=2), train, val, tc, 0, extra_loss=ib_loss)

    m1, h1 = run()
    m2, h2 = run()
    assert h1 == h2
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b)
    m1.eval()
    with torch.no_grad():
        p = [torch.softmax(m1(x)[0], 0)[1].item() for x, _ in val]
    y = [lab for _, lab in val]
    p, y = np.array(p), np.array(y)
    assert p[y == 1].mean() > p[y == 0].mean()
