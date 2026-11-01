import logging

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from lesionbench.selectors.pmil import (
    PMIL,
    PMILSelector,
    _ap_exemplars,
    affinity_prototypes,
    maxmil_pretrain,
    train_pmil,
)


def _blobs(centres, n, spread, seed):
    g = np.random.default_rng(seed)
    return np.concatenate([g.normal(c, spread, size=(n, 8)) for c in centres])


def test_ap_exemplars_one_per_blob_and_damping_fallback(caplog):
    loose = _blobs((0, 5, 10), 50, 1.0, seed=0)  # converges at the official damping 0.5
    assert sorted((_ap_exemplars(loose, 0.25, 0.0, 0.5, 24) // 50).tolist()) == [0, 1, 2]
    tight = _blobs((0, 5, 10), 50, 0.1, seed=0)  # oscillates at 0.5; must fall back, not return all 150
    with caplog.at_level(logging.WARNING):
        ex = _ap_exemplars(tight, 0.25, 0.0, 0.5, 24)
    assert sorted((ex // 50).tolist()) == [0, 1, 2]
    assert "damping 0.9" in caplog.text
    assert _ap_exemplars(np.zeros((1, 8)), 0.25, 0.0, 0.5, 24).tolist() == [0]


def test_affinity_prototypes_are_input_rows_and_deterministic():
    slides = [_blobs(cs, 40, 0.5, seed=i) for i, cs in enumerate([(0, 5), (5, 10), (0, 10)])]
    p = affinity_prototypes(slides)
    rows = np.concatenate(slides).astype(np.float32)
    assert p.dtype == np.float32 and p.shape[1] == 8 and 1 <= len(p) <= 6
    assert all((rows == r).all(axis=1).any() for r in p)
    np.testing.assert_array_equal(p, affinity_prototypes(slides))


def _bags(n, seed):
    """Half the bags are positive: 3 of 30 instances shifted by +3."""
    g = np.random.default_rng(seed)
    bags, tumor = [], []
    for i in range(n):
        x, y = g.normal(size=(30, 16)), i % 2
        rows = g.choice(30, 3, replace=False) if y else np.array([], int)
        x[rows] += 3.0
        bags.append((torch.tensor(x, dtype=torch.float32), y))
        tumor.append(set(rows.tolist()))
    return bags, tumor


def _precision_at_3(selector, bags, tumor):
    return np.mean([len(set(selector.select(x, 3).retained_idx.tolist()) & t) / 3
                    for (x, y), t in zip(bags, tumor, strict=True) if y])


def test_maxmil_head_ranks_tumor_instances_first():
    train, _ = _bags(60, seed=0)
    val, val_tumor = _bags(20, seed=1)
    head = maxmil_pretrain(train, in_dim=16, seed=0, epochs=30)
    assert _precision_at_3(PMILSelector(head), val, val_tumor) > 0.8


def test_pmil_forward_shapes_and_code_range():
    head = torch.nn.Linear(16, 2)
    model = PMIL(head, torch.randn(4, 16), s=5)
    ins, bag, code, kept = model(torch.randn(30, 16))
    assert ins.shape == (30, 2) and bag.shape == (2,) and code.shape == (4,) and kept.shape == (5,)
    assert (code >= 0).all() and (code <= 1).all()
    assert model(torch.randn(3, 16))[3].shape == (3,)  # s > N keeps every patch


def test_pmil_rejects_single_prototype():
    with pytest.raises(ValueError, match="at least 2 prototypes"):
        PMIL(torch.nn.Linear(16, 2), torch.randn(1, 16), s=3)


def test_train_pmil_end_to_end_and_reproducible():
    train, _ = _bags(60, seed=0)
    val, val_tumor = _bags(20, seed=1)
    head = maxmil_pretrain(train, in_dim=16, seed=0, epochs=30)
    # Known prototypes (normal centre, tumor centre): this tests PMIL training on its own.
    # On this unimodal toy data AP finds a single prototype (see test_pmil_rejects_single_prototype).
    protos = torch.stack([torch.zeros(16), torch.full((16,), 3.0)])

    def build():  # fresh copy of the pre-trained head each time, so runs do not share state
        return PMIL(_copy(head), protos, s=3)

    model, hist = train_pmil(build, train, val, seed=0, epochs=15)
    model.eval()
    with torch.no_grad():
        probs = [torch.softmax(model(x)[1], dim=0)[1].item() for x, _ in val]
    assert roc_auc_score([y for _, y in val], probs) > 0.9
    assert _precision_at_3(PMILSelector(model.head), val, val_tumor) > 0.8
    assert hist == train_pmil(build, train, val, seed=0, epochs=15)[1]


def _copy(head):
    new = torch.nn.Linear(head.in_features, head.out_features)
    new.load_state_dict(head.state_dict())
    return new


def test_pmil_selector_ties_and_k_gt_n():
    head = torch.nn.Linear(4, 2)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)  # constant probability -> ties -> lowest rows
    assert PMILSelector(head).select(torch.randn(5, 4), 2).retained_idx.tolist() == [0, 1]
    assert sorted(PMILSelector(head).select(torch.randn(5, 4), 9).retained_idx.tolist()) == [0, 1, 2, 3, 4]
