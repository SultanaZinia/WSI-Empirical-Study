import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from lesionbench.selectors.abmil import ABMILTopK, GatedABMIL
from lesionbench.splits import make_folds
from lesionbench.train import EarlyStopping, evaluate, train_mil


def test_gated_abmil_shapes_and_single_instance_bag():
    m = GatedABMIL(in_dim=32)
    logits, scores = m(torch.randn(50, 32))
    assert logits.shape == (2,) and scores.shape == (50,)
    logits, scores = m(torch.randn(1, 32))
    assert logits.shape == (2,) and scores.shape == (1,)


def test_topk_selector_ties_and_k_gt_n():
    m = GatedABMIL(in_dim=8)
    torch.nn.init.zeros_(m.attn_c.weight)  # constant attention -> all ties -> lowest rows win
    res = ABMILTopK(m).select(torch.randn(6, 8), 3)
    assert res.retained_idx.tolist() == [0, 1, 2] and res.scores.shape == (6,)
    assert ABMILTopK(m).select(torch.randn(6, 8), 10).retained_idx.tolist() == [0, 1, 2, 3, 4, 5]


def test_early_stopping_patience_and_stop_epoch():
    es = EarlyStopping(patience=2, stop_epoch=3)
    # epoch: loss -> (improved, stop)
    steps = [(0, 1.0), (1, 0.9), (2, 0.95), (3, 0.95), (4, 0.96), (5, 0.8), (6, 0.81), (7, 0.82)]
    out = [es.step(e, v) for e, v in steps]
    assert out == [(True, False), (True, False), (False, False),
                   (False, False),  # 2 bad epochs but epoch 3 is not > stop_epoch
                   (False, True),   # 3 bad epochs at epoch 4 > 3
                   (True, False), (False, False), (False, True)]


def test_make_folds_stratified_disjoint_deterministic():
    ids = [f"s{i:02d}" for i in range(20)]
    labels = [1] * 8 + [0] * 12
    folds = make_folds(ids, labels, n_folds=5, seed=0)
    vals = [set(v) for _, v in folds]
    assert set().union(*vals) == set(ids) and sum(len(v) for v in vals) == 20
    for tr, va in folds:
        assert not set(tr) & set(va)
        assert sum(labels[ids.index(s)] for s in va) in (1, 2)  # 8 positives over 5 folds
    assert folds == make_folds(ids, labels, n_folds=5, seed=0)
    assert folds != make_folds(ids, labels, n_folds=5, seed=1)


def _bags(n, seed):
    """Half the bags are positive: 3 of their 30 instances are shifted by +3 (rows 0-2 after shuffling)."""
    g = np.random.default_rng(seed)
    bags, tumor_rows = [], []
    for i in range(n):
        x = g.normal(size=(30, 16))
        y = i % 2
        rows = g.choice(30, 3, replace=False) if y else np.array([], int)
        x[rows] += 3.0
        bags.append((torch.tensor(x, dtype=torch.float32), y))
        tumor_rows.append(set(rows.tolist()))
    return bags, tumor_rows


def test_train_mil_learns_slide_labels_and_attends_to_tumor():
    train, _ = _bags(40, seed=0)
    val, val_tumor = _bags(20, seed=1)
    tcfg = {"lr": 1e-3, "weight_decay": 1e-5, "max_epochs": 15, "patience": 5, "stop_epoch": 5}
    def build():
        return GatedABMIL(in_dim=16, hidden=32, attn_dim=16)

    model, hist = train_mil(build, train, val, tcfg, seed=0)
    _, probs = evaluate(model, val)
    assert roc_auc_score([y for _, y in val], probs) > 0.95
    sel = ABMILTopK(model)
    precision = [len(set(sel.select(x, 3).retained_idx.tolist()) & t) / 3
                 for (x, y), t in zip(val, val_tumor, strict=True) if y]
    assert np.mean(precision) > 0.8

    _, hist2 = train_mil(build, train, val, tcfg, seed=0)
    assert hist == hist2  # seeded -> reproducible, including the initial weights
    _, hist3 = train_mil(build, train, val, tcfg, seed=1)
    assert hist != hist3
