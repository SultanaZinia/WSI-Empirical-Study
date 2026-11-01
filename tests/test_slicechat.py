import math

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from lesionbench.selectors import mamba
from lesionbench.selectors.mamba import BiMamba, selective_scan, selective_scan_ref
from lesionbench.selectors.slicechat import (
    SLICEChatMIL,
    SLICEChatSelector,
    aux_loss,
    grid_from_coords,
    hilbert_index,
    keep_schedule,
    raster_order,
    remove_window,
)
from lesionbench.train import evaluate, train_mil


def test_hilbert_order_one_hand_computed_and_adjacency():
    g = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]])
    assert hilbert_index(g, bits=1).tolist() == [0, 1, 2, 3]
    xs, ys = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    grid = torch.stack([xs.flatten(), ys.flatten()], dim=1)
    idx = hilbert_index(grid)
    assert len(set(idx.tolist())) == 64
    path = grid[torch.argsort(idx)]
    assert ((path[1:] - path[:-1]).abs().sum(dim=1) == 1).all()  # consecutive cells are neighbours
    with pytest.raises(ValueError, match="1024"):
        hilbert_index(torch.tensor([[1024, 0]]))


def test_raster_order_and_grid_from_coords():
    grid = torch.tensor([[2, 1], [0, 1], [5, 0], [1, 0]])
    assert raster_order(grid).tolist() == [3, 2, 1, 0]
    coords = np.array([[1000, 512], [1512, 512], [1000, 1024]])
    assert grid_from_coords(coords, 512).tolist() == [[0, 0], [1, 0], [0, 1]]


def test_keep_schedule_matches_paper_figure_and_ends_at_k():
    # SLICEChat Fig. 3: 4,785 tokens, keep rate 0.2 over five pruners -> 3468, 2514, 1822, 1321, 957
    assert keep_schedule(4785, 957, 5) == [3468, 2514, 1822, 1321, 957]
    s = keep_schedule(20000, 32, 5)
    assert s[-1] == 32 and all(a >= b >= 32 for a, b in zip(s, s[1:], strict=False))
    assert keep_schedule(10, 32, 5) == [10] * 5
    assert keep_schedule(33, 32, 5) == [33, 33, 33, 33, 32]  # round(33 * 0.994) = 33 until the forced last


def test_remove_window_drops_lowest_contiguous_hilbert_window():
    torch.manual_seed(0)
    grid = torch.tensor([[x, y] for x in range(3) for y in range(2)])
    perm = torch.argsort(hilbert_index(grid), stable=True)
    s_hilbert = torch.tensor([5.0, 1.0, 1.0, 5.0, 0.0, 6.0])  # windows of 2: 6, 2, 6, 5, 6 -> start 1
    scores = torch.empty(6)
    scores[perm] = s_hilbert
    kept = remove_window(scores, grid, 4)
    assert set(kept.tolist()) == set(range(6)) - {perm[1].item(), perm[2].item()}
    assert kept.tolist() == sorted(kept.tolist())
    assert torch.equal(remove_window(scores, grid, 6), torch.arange(6))


def test_selective_scan_ref_hand_computed_scalar():
    # delta = softplus(0) = ln 2, A = -1 -> decay 0.5; B = C = 1, D = 0, u = [1, 2]
    # h1 = ln2, h2 = 0.5 ln2 + 2 ln2 = 2.5 ln2; output gated by silu(z)
    u = torch.tensor([[[1.0, 2.0]]])
    z = torch.full((1, 1, 2), 10.0)
    ones = torch.ones(1, 1, 2)
    y = selective_scan_ref(u, torch.zeros(1, 1, 2), -torch.ones(1, 1), ones, ones, torch.zeros(1), z,
                           torch.zeros(1))
    silu10 = 10 / (1 + math.exp(-10))
    torch.testing.assert_close(y[0, 0], torch.tensor([math.log(2), 2.5 * math.log(2)]) * silu10)


def test_bimamba_shapes_and_sees_both_directions():
    torch.manual_seed(0)
    m = BiMamba(8)
    x = torch.randn(1, 6, 8)
    y = m(x)
    assert y.shape == (1, 6, 8)
    x2 = x.clone()
    x2[0, -1] += 1.0
    assert not torch.allclose(m(x2)[0, 0], y[0, 0])  # the backward scan carries the last token to the first
    x3 = x.clone()
    x3[0, 0] += 1.0
    assert not torch.allclose(m(x3)[0, -1], y[0, -1])


def _slide(n_side=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.tensor([[x, y] for y in range(n_side) for x in range(n_side)])
    return torch.randn(n_side * n_side, 8, generator=g), grid[torch.randperm(n_side * n_side, generator=g)]


def test_encode_retains_k_and_records_layer_of_drop():
    torch.manual_seed(0)
    m = SLICEChatMIL(in_dim=8, K=4, dim=16, n_blocks=4, n_heads=2).eval()
    x, grid = _slide()
    h, rows, drop = m.encode(x, grid)
    assert h.shape == (4, 16) and len(set(rows.tolist())) == 4
    assert (drop[rows] == -1).all() and (drop == -1).sum() == 4
    sched = keep_schedule(36, 4, 3)
    removed = [36 - sched[0]] + [a - b for a, b in zip(sched, sched[1:], strict=False)]
    assert [(drop == b).sum().item() for b in range(3)] == removed
    assert raster_order(grid[rows]).tolist() == list(range(4))  # survivors stay in raster order
    torch.testing.assert_close(m((x, grid))[0], m((x, grid))[0])


def test_aux_loss_trains_pruners_not_encoder():
    torch.manual_seed(0)
    m = SLICEChatMIL(in_dim=8, K=4, dim=16, n_blocks=3, n_heads=2).train()
    x, grid = _slide()
    m((x, grid))
    assert len(m.aux_logits) == 2
    aux_loss(m, 1).backward()
    assert m.proj.weight.grad is None and m.pruners[0].query.grad is not None
    assert m.aux_head.weight.grad is not None


def test_selector_k_must_match():
    torch.manual_seed(0)
    m = SLICEChatMIL(in_dim=8, K=4, dim=16, n_blocks=3, n_heads=2)
    x, grid = _slide()
    res = SLICEChatSelector(m).select(x, 4, grid=grid)
    assert len(res.retained_idx) == 4 and res.layer_of_drop.shape == (36,)
    with pytest.raises(ValueError, match="K = 4"):
        SLICEChatSelector(m).select(x, 8, grid=grid)
    small_x, small_grid = x[:3], grid[:3]
    assert len(SLICEChatSelector(m).select(small_x, 4, grid=small_grid).retained_idx) == 3


def _bags(n, seed):
    """Positive bags carry a 2x2 tumour cluster (+3 shift) at a random spot of a 6x6 grid."""
    g = np.random.default_rng(seed)
    grid = torch.tensor([[x, y] for y in range(6) for x in range(6)])
    bags, tumor = [], []
    for i in range(n):
        x, y = g.normal(size=(36, 8)), i % 2
        rows = set()
        if y:
            cx, cy = g.integers(0, 5, size=2)
            rows = {int((cy + dy) * 6 + cx + dx) for dx in (0, 1) for dy in (0, 1)}
            x[list(rows)] += 3.0
        bags.append(((torch.tensor(x, dtype=torch.float32), grid), y))
        tumor.append(rows)
    return bags, tumor


def test_train_slicechat_learns_and_is_reproducible():
    train, _ = _bags(40, seed=0)
    val, val_tumor = _bags(20, seed=1)
    tcfg = {"lr": 1e-3, "weight_decay": 1e-5, "max_epochs": 12, "patience": 12, "stop_epoch": 12}

    def build():
        return SLICEChatMIL(in_dim=8, K=4, dim=16, n_blocks=3, n_heads=2)

    model, hist = train_mil(build, train, val, tcfg, seed=0, extra_loss=aux_loss)
    _, probs = evaluate(model, val)
    assert roc_auc_score([y for _, y in val], probs) > 0.9
    sel = SLICEChatSelector(model)
    kept = [len(set(sel.select(x, 4, grid=g).retained_idx.tolist()) & t) / 4
            for ((x, g), y), t in zip(val, val_tumor, strict=True) if y]
    assert np.mean(kept) > 4 / 36  # better than random (expected 4/36 of kept rows are tumour)
    assert hist == train_mil(build, train, val, tcfg, seed=0, extra_loss=aux_loss)[1]


@pytest.mark.server
@pytest.mark.skipif(mamba.selective_scan_fn is None or not torch.cuda.is_available(),
                    reason="needs mamba_ssm + GPU")
def test_cuda_kernel_matches_reference():
    torch.manual_seed(0)
    b, d, n, L = 1, 32, 16, 300
    args = [torch.randn(b, d, L), torch.randn(b, d, L), -torch.rand(d, n) - 0.5, torch.randn(b, n, L),
            torch.randn(b, n, L), torch.randn(d), torch.randn(b, d, L), torch.randn(d)]
    ref = selective_scan(*args, use_kernel=False)
    cuda = selective_scan(*[a.cuda() for a in args], use_kernel=True).cpu()
    torch.testing.assert_close(cuda, ref, atol=1e-3, rtol=1e-3)
