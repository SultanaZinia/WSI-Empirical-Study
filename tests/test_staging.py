from collections import Counter

import pytest
from sklearn.metrics import cohen_kappa_score

from lesionbench.eval.staging import PN_STAGES, patient_folds, pn_stage, quadratic_kappa


@pytest.mark.parametrize("nodes, stage", [
    (["negative"] * 5, "pN0"),
    (["itc", "negative", "negative", "negative", "negative"], "pN0(i+)"),
    (["micro", "itc", "negative", "negative", "negative"], "pN1mi"),
    (["micro", "micro", "micro", "micro", "negative"], "pN1mi"),  # no macro: pN1mi however many micro
    (["macro", "itc", "itc", "negative", "negative"], "pN1"),  # ITC is not a positive node
    (["macro", "micro", "micro", "negative", "negative"], "pN1"),
    (["macro", "micro", "micro", "micro", "negative"], "pN2"),
    (["macro"] * 5, "pN2"),
])
def test_pn_stage(nodes, stage):
    assert pn_stage(nodes) == stage


def test_quadratic_kappa_matches_sklearn():
    t = ["pN0", "pN1", "pN2", "pN1mi", "pN0(i+)", "pN0", "pN1", "pN2"]
    p = ["pN0", "pN1mi", "pN2", "pN0", "pN0", "pN0(i+)", "pN1", "pN1"]
    want = cohen_kappa_score(t, p, labels=PN_STAGES, weights="quadratic")
    assert quadratic_kappa(t, p) == pytest.approx(want)
    assert quadratic_kappa(t, t) == pytest.approx(1.0)


def test_patient_folds_stratified_and_balanced():
    stages = {f"p{i:03d}": PN_STAGES[i % 5] for i in range(23)}
    f = patient_folds(stages, 5, seed=0)
    assert set(f) == set(stages)
    sizes = Counter(f.values())
    assert max(sizes.values()) - min(sizes.values()) <= 1
    for s in PN_STAGES:  # each stage spread over folds (at most one more in any fold)
        c = Counter(f[p] for p, st in stages.items() if st == s)
        assert max(c.values()) - min(c.get(k, 0) for k in range(5)) <= 1
    assert f == patient_folds(stages, 5, seed=0)


def test_cross_validate_and_patient_rows():
    import torch

    from lesionbench.eval.staging import SLIDE_CLASSES, cross_validate, patient_rows

    g = torch.Generator().manual_seed(0)
    patient_stage, slide_stage, patient_of, bags = {}, {}, {}, {}
    for i in range(10):
        p = f"patient_{i:03d}"
        nodes = ["macro" if i % 2 else "negative"] + ["negative"] * 4
        patient_stage[p] = pn_stage(nodes)
        for j, st in enumerate(nodes):
            s = f"{p}_node_{j}"
            slide_stage[s], patient_of[s] = st, p
            bags[s] = torch.randn(4, 6, generator=g) + 3.0 * SLIDE_CLASSES.index(st)
    folds = patient_folds(patient_stage, 5, seed=0)
    tcfg = {"lr": 1e-3, "weight_decay": 0.0, "patience": 2, "stop_epoch": 1, "max_epochs": 5}
    rows = cross_validate(bags, slide_stage, patient_of, folds, tcfg, seed=0)
    assert sorted(r["slide_id"] for r in rows) == sorted(bags)  # every slide predicted once, out of fold
    assert all(folds[r["patient"]] == r["cv_fold"] for r in rows)
    assert all(abs(sum(r[f"p_{k}"] for k in SLIDE_CLASSES) - 1) < 1e-5 for r in rows)
    pr = patient_rows(rows, patient_stage)
    assert len(pr) == 10 and all(r["n_slides"] == 5 for r in pr)
    assert rows == cross_validate(bags, slide_stage, patient_of, folds, tcfg, seed=0)  # deterministic
